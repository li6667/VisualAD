"""ATS-ViT: 论文级配图自动生成（300 DPI，同时导出 PNG 与矢量 PDF）。

产出（保存到 ../results_ats_vit/figures/）：
  fig1_qualitative.{png,pdf}   四联定性对比：Input | Ground Truth | VisualAD | ATS-ViT
  fig2_pca.{png,pdf}           特征分布 PCA（原始 ViT / +SAF / +ATS-SAF）
  fig3_attention.{png,pdf}     跨层 anchor 注意力（layer 6 / 12 / 18 / 24）

所有图都由真实模型前向得到，不使用论文里读图估计的数值。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import VisualAD_lib
from utils.feature_transform import create_feature_transform
from utils.transforms import get_transform
from utils.anomaly_detection import generate_anomaly_map_from_tokens

DATA = Path("/d/ls/Anomaly Detection/datasets")
OUT = Path("/d/ls/Anomaly Detection/results_ats_vit/figures")
OUT.mkdir(parents=True, exist_ok=True)
DPI = 300


# --------------------------------------------------------------------------- #
# 模型装配：支持原版 adapter 与 ATS adapter 两种 ckpt
# --------------------------------------------------------------------------- #
class Args:
    image_size = 518
    sigma = 4


class Pipeline:
    def __init__(self, ckpt_path: str, device="cuda:0", use_ats_override=None):
        self.device = torch.device(device)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        self.ckpt = ckpt
        self.backbone = ckpt.get("backbone", "ViT-L/14@336px")
        self.features_list = ckpt.get("features_list", [6, 12, 18, 24])
        args = Args()
        args.image_size = ckpt.get("image_size", 518)
        self.args = args
        self.uses_ats = bool(ckpt.get("uses_ats", False)) if use_ats_override is None else use_ats_override
        self.ats_config = ckpt.get("ats_config", {}) or {}

        self.model, _ = VisualAD_lib.load(self.backbone, device=self.device)
        self.model.eval().to(self.device)
        self.model.visual.anomaly_token.data = ckpt["anomaly_token"].to(self.device)
        self.model.visual.normal_token.data = ckpt["normal_token"].to(self.device)
        self.model.visual.ln_post.load_state_dict(
            {"weight": ckpt["ln_post_weight"], "bias": ckpt["ln_post_bias"]}
        )
        dim = self.model.visual.embed_dim

        self.layer_transforms = nn.ModuleDict()
        for name, sd in ckpt.get("layer_transforms", {}).items():
            hidden = sd["mlp.0.weight"].shape[0]
            if self.uses_ats:
                from utils.ats_module import create_ats_transform
                m = create_ats_transform(
                    input_dim=dim, hidden_dim=hidden, output_dim=dim, dropout=0.0,
                    reduction=int(self.ats_config.get("reduction", 16)),
                    alpha_init=float(self.ats_config.get("alpha_init", 0.05)),
                    use_spectral=bool(self.ats_config.get("use_spectral", True)),
                ).to(self.device)
                m.load_state_dict(sd, strict=False)
            else:
                m = create_feature_transform(
                    transform_type="mlp", input_dim=dim, hidden_dim=hidden,
                    output_dim=dim, dropout=0.0).to(self.device)
                m.load_state_dict(sd)
            m.eval()
            self.layer_transforms[name] = m

        self.cross_attn = None
        if "cross_attn" in ckpt:
            from utils.spatial_cross_attention import build_layer_adaptive_cross_attention
            cfg = ckpt.get("cross_attn_config", {}) or {}
            self.cross_attn = build_layer_adaptive_cross_attention(
                layers=self.features_list, embed_dim=dim,
                num_anchors=cfg.get("num_anchors", 4),
                dropout=cfg.get("dropout", 0.1),
                res_scale_init=cfg.get("res_scale_init", 0.01),
            ).to(self.device)
            self.cross_attn.load_state_dict(ckpt["cross_attn"])
            self.cross_attn.eval()

        self.preprocess, _ = get_transform(args)

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def forward(self, pil_img: Image.Image, want="map"):
        """返回 anomaly map（518x518 numpy）、图像级分数，以及可选的中间量。"""
        x = self.preprocess(pil_img.convert("RGB")).unsqueeze(0).to(self.device)
        vo = self.model.encode_image(x, self.features_list)
        a, n = vo["anomaly_features"], vo["normal_features"]
        pts, ps = vo["patch_tokens"], vo["patch_start_idx"]

        if self.cross_attn is not None:
            adapted = self.cross_attn(a, n, [p[:, ps:, :] for p in pts], self.features_list)
            a_list = [d["anomaly"] for d in adapted]
            n_list = [d["normal"] for d in adapted]
        else:
            a_list = [a] * len(pts)
            n_list = [n] * len(pts)

        maps, feats = [], []
        for i, pf in enumerate(pts):
            key = f"layer_{self.features_list[i]}"
            if key in self.layer_transforms:
                pf = (self.layer_transforms[key](pf) if self.uses_ats
                      else self.layer_transforms[key](pf.view(-1, pf.shape[-1])).view(pf.shape))
            feats.append((key, pf))
            am = generate_anomaly_map_from_tokens(
                F.normalize(a_list[i], dim=1), F.normalize(n_list[i], dim=1),
                pf[:, ps:, :], self.args.image_size)
            maps.append(am)

        final = torch.stack(maps).sum(0).cpu()
        fmap = gaussian_filter(final[0].numpy(), sigma=self.args.sigma)
        fmap = torch.from_numpy(fmap).unsqueeze(0)
        flat = fmap.reshape(1, -1)
        k = max(1, int(np.ceil(flat.shape[-1] * 0.01)))
        score = float(torch.topk(flat, k, dim=-1).values.mean())

        if want == "all":
            return fmap[0].numpy(), score, feats, ps
        return fmap[0].numpy(), score

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def attention_maps(self, pil_img: Image.Image):
        """返回 {layer: (4, 37, 37) numpy} 的 anchor 注意力。"""
        x = self.preprocess(pil_img.convert("RGB")).unsqueeze(0).to(self.device)
        vo = self.model.encode_image(x, self.features_list)
        a, n, pts, ps = vo["anomaly_features"], vo["normal_features"], vo["patch_tokens"], vo["patch_start_idx"]
        if self.cross_attn is None:
            return {}
        self.cross_attn(a, n, [p[:, ps:, :] for p in pts], self.features_list)
        out = {}
        for key, mod in self.cross_attn.layer_attentions.items():
            att = getattr(mod, "last_attn", None)
            if att is None:
                continue
            att = att[0].float().cpu().numpy()  # [num_anchors, N]
            side = int(round(att.shape[-1] ** 0.5))
            if side * side != att.shape[-1]:
                continue
            out[key] = att.reshape(att.shape[0], side, side)
        return out


# --------------------------------------------------------------------------- #
# 数据采样
# --------------------------------------------------------------------------- #
def pick_samples(n_per_set=2):
    """从金属相关数据集挑代表性样本（图, 掩码）。"""
    picks = []

    def add(img_p, mask_p, tag):
        if img_p.exists() and mask_p.exists():
            picks.append((img_p, mask_p, tag))

    # KSDD2 金属微裂纹
    ks = DATA / "ksdd2/Metal"
    for i, f in enumerate(sorted((ks / "test/anomaly").glob("*.png"))[:n_per_set]):
        add(f, ks / "ground_truth/anomaly" / f.name, "KSDD2")
    # MVTec 金属子类
    for cls in ["metal_nut", "grid", "screw"]:
        d = DATA / f"mvtec/{cls}"
        for f in sorted((d / "test").glob("*/*.png")):
            if f.parent.name == "good":
                continue
            rel = f.relative_to(d / "test")
            m = d / "ground_truth" / rel.parent.name / (rel.stem + "_mask.png")
            if m.exists():
                add(f, m, f"MVTec/{cls}")
                break
    # DAGM 各向异性纹理
    dg = DATA / "dagm/pic1/Class1/Test"
    for f in sorted(dg.glob("*.PNG"))[:60]:
        m = dg / "Label" / f"{f.stem}_label.PNG"
        if m.exists():
            add(f, m, "DAGM/Class1")
            break
    return picks


def load_pair(img_p, mask_p, size=518):
    img = Image.open(img_p).convert("RGB")
    m = Image.open(mask_p).convert("L").resize((size, size), Image.NEAREST)
    return img, (np.array(m) > 127)


# --------------------------------------------------------------------------- #
# Fig 1: 定性四联
# --------------------------------------------------------------------------- #
def fig_qualitative(pipe_base, pipe_ats, samples, out_png):
    n = len(samples)
    fig, axes = plt.subplots(n, 4, figsize=(12.2, 3.05 * n))
    if n == 1:
        axes = axes[None, :]
    for r, (img_p, mask_p, tag) in enumerate(samples):
        img, gt = load_pair(img_p, mask_p)
        m_b, s_b = pipe_base.forward(img)
        m_a, s_a = pipe_ats.forward(img)
        axes[r, 0].imshow(img); axes[r, 0].set_title(f"Input ({tag})", fontsize=9)
        axes[r, 1].imshow(gt, cmap="gray"); axes[r, 1].set_title("Ground Truth", fontsize=9)
        axes[r, 2].imshow(img); axes[r, 2].imshow(m_b, cmap="jet", alpha=0.5)
        axes[r, 2].set_title(f"VisualAD  score={s_b:.2f}", fontsize=9)
        axes[r, 3].imshow(img); axes[r, 3].imshow(m_a, cmap="jet", alpha=0.5)
        axes[r, 3].set_title(f"ATS-ViT (Ours)  score={s_a:.2f}", fontsize=9)
        for c in range(4):
            axes[r, c].axis("off")
    fig.tight_layout()
    fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Fig 2: PCA
# --------------------------------------------------------------------------- #
@torch.no_grad()
def collect_features(pipe, samples, layer_idx=-1, max_patches=4000):
    """收集最后一层 patch 特征；按 GT 掩码（下采样到 37x37）打 normal/anomaly 标签。"""
    feats, labels = [], []
    for img_p, mask_p, _ in samples:
        img, gt = load_pair(img_p, mask_p)
        _, _, feats_layers, ps = pipe.forward(img, want="all")
        key, pf = feats_layers[layer_idx]
        pf = pf[0, ps:, :].float().cpu().numpy()
        side = int(round(pf.shape[0] ** 0.5))
        if side * side != pf.shape[0]:
            continue
        g = torch.from_numpy(gt.astype(np.float32))[None, None]
        g = F.interpolate(g, size=(side, side), mode="area")[0, 0].numpy()
        lab = (g.reshape(-1) > 0.30).astype(np.int64)
        feats.append(pf)
        labels.append(lab)
    if not feats:
        return None, None
    X = np.concatenate(feats, 0)
    y = np.concatenate(labels, 0)
    if len(X) > max_patches:
        idx = np.random.default_rng(0).choice(len(X), max_patches, replace=False)
        X, y = X[idx], y[idx]
    return X, y


def pca2d(X):
    X = X - X.mean(0, keepdims=True)
    # 少量主成分用 SVD 实现，避免依赖 sklearn 版本差异
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    Z = X @ Vt[:2].T
    var = (S ** 2) / (S ** 2).sum()
    return Z, var


def fig_pca(panels, out_png):
    fig, axes = plt.subplots(1, len(panels), figsize=(4.3 * len(panels), 4.0))
    for ax, (title, Z, y, var) in zip(np.atleast_1d(axes), panels):
        m = y == 0
        ax.scatter(Z[m, 0], Z[m, 1], s=2, c="#2E86C1", alpha=0.35, label="normal patch", rasterized=True)
        ax.scatter(Z[~m, 0], Z[~m, 1], s=2, c="#C0392B", alpha=0.55, label="anomalous patch", rasterized=True)
        ax.set_title(f"{title}\nPC1 var = {var[0]*100:.1f}%", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        ax.legend(fontsize=7, markerscale=3, loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Fig 3: 跨层 attention
# --------------------------------------------------------------------------- #
def fig_attention(pipe, img_p, out_png):
    img = Image.open(img_p).convert("RGB")
    att = pipe.attention_maps(img)
    if not att:
        print("  [warn] 没有拿到注意力，跳过 fig3")
        return
    keys = sorted(att.keys(), key=lambda k: int(k.split("_")[-1]))
    fig, axes = plt.subplots(1, len(keys) + 1, figsize=(3.0 * (len(keys) + 1), 3.0))
    axes[0].imshow(img); axes[0].set_title("Input", fontsize=9); axes[0].axis("off")
    for ax, k in zip(axes[1:], keys):
        a = att[k].mean(0)  # 平均各 anchor
        a = (a - a.min()) / (a.max() - a.min() + 1e-8)
        ax.imshow(img.resize((a.shape[1], a.shape[0])))
        ax.imshow(a, cmap="inferno", alpha=0.55)
        ax.set_title(f"{k.replace('layer_','layer ')} anchors", fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_ckpt", default=str(REPO / "weight/train_on_visa/CLIP.pth"))
    ap.add_argument("--ats_ckpt", default=str(Path("/d/ls/Anomaly Detection/results_ats_vit/ckpt/finetune_ats.pth")))
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    print("[1/3] 装配 baseline 与 ATS 两条管线 ...")
    pipe_base = Pipeline(args.baseline_ckpt, args.device)
    if not os.path.exists(args.ats_ckpt):
        print(f"  [warn] ATS ckpt 不存在：{args.ats_ckpt}，fig1 将退化为 baseline vs baseline")
        pipe_ats = pipe_base
    else:
        pipe_ats = Pipeline(args.ats_ckpt, args.device)
    print(f"  baseline uses_ats={pipe_base.uses_ats} | ats uses_ats={getattr(pipe_ats,'uses_ats',None)}")

    samples = pick_samples()
    print(f"[samples] 选中 {len(samples)} 组: {[s[2] for s in samples]}")

    print("[2/3] Fig.1 定性对比 ...")
    fig_qualitative(pipe_base, pipe_ats, samples, OUT / "fig1_qualitative.png")

    print("[3/3] Fig.2 PCA + Fig.3 attention ...")
    panels = []
    # (1) 原始冻结 ViT 特征：用 baseline 管线但把 SAF 置为恒等（等价"无变换"）
    combos = [("Original ViT", pipe_base, "identity"),
              ("+ SAF (VisualAD)", pipe_base, "saf"),
              ("+ ATS-SAF (Ours)", pipe_ats, "saf")]
    for title, pipe, mode in combos:
        orig = dict(pipe.layer_transforms.items())
        if mode == "identity":
            pipe.layer_transforms = nn.ModuleDict({k: nn.Identity() for k in orig})
        try:
            X, y = collect_features(pipe, samples)
        finally:
            pipe.layer_transforms = nn.ModuleDict(orig)
        if X is None:
            continue
        Z, var = pca2d(X)
        panels.append((title, Z, y, var))
        print(f"  {title}: {X.shape[0]} patches | PC1 var = {var[0]*100:.1f}%")
    fig_pca(panels, OUT / "fig2_pca.png")

    # Fig.3 用 carpet（论文原文用的类别）与一张金属样本各画一次
    carpet = sorted((DATA / "mvtec/carpet/test").glob("*/*.png"))
    carpet = [p for p in carpet if p.parent.name != "good"] or carpet
    if carpet:
        fig_attention(pipe_ats, carpet[0], OUT / "fig3_attention_carpet.png")
    if samples:
        fig_attention(pipe_ats, samples[0][0], OUT / "fig3_attention_metal.png")

    print("完成，输出目录:", OUT)


if __name__ == "__main__":
    main()
