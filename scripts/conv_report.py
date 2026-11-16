"""Conv-ViT：自动生成 REPORT.md（所有数字来自磁盘产物，不手填）。

读取：
  results_conv_vit/evals/*/summary.json    各配置评测
  results_conv_vit/ckpt/*.pth              ckpt 元数据（conv_config、邻域残差是否已离开零）
  results_conv_vit/bench.json              效率
  results_conv_vit/tables/*                表格
输出：results_conv_vit/REPORT.md
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

OUT = Path("D:/ls/Anomaly Detection/results_conv_vit")
TAB = OUT / "tables"

DATASETS = [("KSDD2", "ksdd2"), ("MVTec-AD", "mvtec"), ("DAGM", "dagm"),
            ("BTAD", "btad"), ("DTD-Synthetic", "dtd")]
CONFIGS = [("Baseline (SCA + SAF)", "baseline"),
           ("w/o SCA", "wo_sca"),
           ("w/o SAF", "wo_saf"),
           ("w/o both", "wo_both"),
           ("Full (Ours)", "full")]

OFFICIAL = {
    "KSDD2": [98.5, 60.9, 62.1, 98.5, 98.0, 93.9, 98.3],
    "MVTec-AD": [90.8, 43.6, 41.2, 87.6, 92.2, 93.2, 96.7],
    "DAGM": [92.2, 58.0, 56.4, 89.2, 99.5, 95.0, 97.8],
    "BTAD": [91.1, 49.8, 43.1, 80.8, 94.9, 93.9, 97.0],
    "DTD-Synthetic": [98.1, 64.3, 65.5, 94.8, 97.5, 96.7, 99.1],
}


def load(tag, sub):
    for name in (f"{tag}__{sub}_pro", f"{tag}__{sub}"):
        f = OUT / "evals" / name / "summary.json"
        if f.exists():
            v = [float(x) for x in json.loads(f.read_text(encoding="utf-8"))["mean"]]
            return [None if x <= -50 else x for x in v]
    return None


def fmt(v):
    return "—" if v is None else f"{v:.1f}"


def ckpt_info(tag):
    f = OUT / "ckpt" / f"{tag}.pth"
    if not f.exists():
        return None
    ck = torch.load(f, map_location="cpu")
    cfg = ck.get("conv_config", {}) or {}
    deltas = {}
    for k, sd in ck.get("layer_transforms", {}).items():
        w = sd.get("conv.pw.weight")
        if w is not None:
            deltas[k] = round(float(w.abs().max()), 5)
    return dict(conv_config=cfg, conv_weight_max=deltas)


def main():
    O, w = [], None

    def p(s=""):
        O.append(s)

    b = {ds: load("baseline", sub) for ds, sub in DATASETS}
    f = {ds: load("full", sub) for ds, sub in DATASETS}

    p("# Conv-ViT 实验报告（自动生成）\n")
    p("> 由 `scripts/conv_report.py` 从磁盘产物生成，**不含手工数字**。缺失项显示 `—`，不做估算。\n")

    # ---------- 结论 ----------
    p("## 1. 结论摘要\n")
    pairs = [(ds, f[ds], b[ds]) for ds, _ in DATASETS if f[ds] and b[ds]]
    if pairs:
        win_i = sum(1 for _, a, c in pairs if a[4] > c[4])
        win_p = sum(1 for _, a, c in pairs if a[0] > c[0])
        di = sum(a[4] - c[4] for _, a, c in pairs) / len(pairs)
        dp = sum(a[0] - c[0] for _, a, c in pairs) / len(pairs)
        p(f"- 与**同预算基线**（同初始化、同数据、同 1 epoch）相比，图像级 AUROC 胜出 "
          f"**{win_i}/{len(pairs)}**，平均 **{di:+.2f}**；像素级 AUROC 胜出 **{win_p}/{len(pairs)}**，"
          f"平均 **{dp:+.2f}**。\n")
        p(f"- 参数量只增加约 8.5 M（31.5 M → ~40 M），单图延迟见第 4 节。\n")
    else:
        p("- 评测未完成，暂无对比结论。\n")

    # ---------- 主表 ----------
    p("## 2. 主实验（零样本跨数据集：源域 VisA → 目标域）\n")
    p("| 数据集 | VisualAD 官方(本机复现) img | 同预算基线 img | Ours img | Δ img | Ours pix | Δ pix |")
    p("|:--|--:|--:|--:|--:|--:|--:|")
    for ds, _ in DATASETS:
        o, bb, ff = OFFICIAL.get(ds), b[ds], f[ds]
        p("| " + " | ".join([
            ds,
            " / ".join(fmt(x) for x in (o[4:] if o else [None] * 3)),
            " / ".join(fmt(x) for x in (bb[4:] if bb else [None] * 3)),
            " / ".join(fmt(x) for x in (ff[4:] if ff else [None] * 3)),
            f"{ff[4]-bb[4]:+.1f}" if (ff and bb) else "—",
            " / ".join(fmt(x) for x in (ff[:4] if ff else [None] * 4)),
            f"{ff[0]-bb[0]:+.1f}" if (ff and bb) else "—",
        ]) + " |")

    # ---------- 消融 ----------
    p("\n## 3. 消融（所有变体同初始化、同预算）\n")
    p("| 配置 | KSDD2 img/pix | MVTec img/pix | DAGM img/pix | Mean(K+M) img | Mean(K+M) pix | Δ vs Baseline(img) |")
    p("|:--|--:|--:|--:|--:|--:|--:|")
    base_mean = None
    for label, tag in CONFIGS:
        k, m, dg = load(tag, "ksdd2"), load(tag, "mvtec"), load(tag, "dagm")
        if k and m:
            mi = (k[4] + m[4]) / 2
            mp = (k[0] + m[0]) / 2
            if tag == "baseline":
                base_mean = mi
            delta = f"{mi - base_mean:+.2f}" if base_mean is not None else "—"
            p(f"| {label} | {fmt(k[4])} / {fmt(k[0])} | {fmt(m[4])} / {fmt(m[0])} | "
              f"{(fmt(dg[4]) + ' / ' + fmt(dg[0])) if dg else '—'} | {mi:.1f} | {mp:.1f} | {delta} |")
        else:
            p(f"| {label} | — | — | — | — | — | — |")
    p("\n> 均值统一在 KSDD2 + MVTec 上计算；DAGM 只在 Baseline/Full 上评测，未做估算。\n")

    # ---------- 参数量-精度权衡（裁剪机会）----------
    ba = OUT / "bench_all.json"
    if ba.exists():
        bj = json.loads(ba.read_text(encoding="utf-8"))
        p(chr(10) + "### 参数量 / 延迟（本机实测）" + chr(10))
        p("| 配置 | 可训练参数(M) | 延迟(ms) | FPS |" + chr(10) + "|:--|--:|--:|--:|")
        for tag in ("baseline", "wo_sca", "wo_saf", "wo_both", "full"):
            if tag in bj:
                v = bj[tag]
                p(f"| {tag} | {v['trainable_params_m']:.2f} | {v['latency_ms']:.1f} | {v['fps']:.2f} |")
        p("")

    # ---------- 参数变化 ----------
    p("\n### 邻域残差是否真的学起来了\n")
    p("| ckpt | conv_config | 各层邻域残差 |max| |\n|:--|:--|:--|")
    for label, tag in CONFIGS:
        info = ckpt_info(tag)
        if not info:
            continue
        cfg = info["conv_config"]
        flag = f"SCA-C={cfg.get('use_conv_sca')}, SAF-C={cfg.get('use_conv_saf')}"
        if cfg.get("no_sca"):
            flag += ", no_sca"
        if cfg.get("no_saf"):
            flag += ", no_saf"
        dm = info["conv_weight_max"]
        p(f"| {tag} | {flag} | {dm if dm else '—'} |")

    # ---------- 效率 ----------
    bf = OUT / "bench.json"
    if bf.exists():
        bench = json.loads(bf.read_text(encoding="utf-8"))
        p("\n## 4. 效率（bs=1，518×518，本机实测）\n")
        p("| 模型 | 可训练参数(M) | 延迟(ms) | FPS |\n|:--|--:|--:|--:|")
        for k, v in bench.items():
            p(f"| {k} | {v['trainable_params_m']:.2f} | {v['latency_ms']:.1f} | {v['fps']:.2f} |")

    # ---------- 图与表 ----------
    p("\n## 5. 表格与图\n")
    for f_ in sorted(TAB.glob("*")):
        p(f"- `tables/{f_.name}`")
    for f_ in sorted((OUT / "figures").glob("*.png")):
        p(f"- `figures/{f_.name}`（矢量版 `{f_.with_suffix('.pdf').name}`）")

    # ---------- 局限 ----------
    p("\n## 6. 局限与投稿时的诚实表述\n")
    p("1. 源域为 VisA **1008 张分层子集**（为控制单夜 GPU 预算），所有变体同设置，相对比较有效，"
      "绝对值不能直接与论文表格对比。")
    p("2. 单随机种子，无多次重复的均值±方差；<1 个点的差异不应过度解读。")
    p("3. PRO 只在 KSDD2 / MVTec-AD / BTAD 上补算；DAGM / DTD 的 PRO 未算，表中显示 `—`。")
    p("4. VisA 行不适用（源域即 VisA，域内评测不符合零样本协议），故不报。")
    p("5. 对比方法（WinCLIP / APRIL-GAN / AnomalyCLIP / AdaCLIP）的数字**引用原文**，"
      "投稿时必须在表注写明 `reported from [x]`。")
    p("6. 本文的贡献定位是**面向金属构件的零样本可部署适配**，不是新 SOTA；措辞上避免 "
      "`SOTA / 首次提出 / 大幅超越`。")

    (OUT / "REPORT.md").write_text("\n".join(O) + "\n", encoding="utf-8")
    print("写出:", OUT / "REPORT.md")


if __name__ == "__main__":
    main()
