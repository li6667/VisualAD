"""ATS-ViT: 自动生成最终报告 REPORT.md（所有数字来自磁盘上的真实评测产物）。

读取：
  * ../results_ats_vit/baseline_reference.json        VisualAD 基线（本机复现）
  * ../results_ats_vit/evals/<tag>__*/summary.json    各模型各数据集评测
  * ../results_ats_vit/bench.json                     参数量 / 延迟 / FPS
  * ../results_ats_vit/ckpt/*.pth                     训练后 alpha 取值
输出：
  * ../results_ats_vit/REPORT.md
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

OUT = Path(r"D:/ls/Anomaly Detection/results_ats_vit")
DATASETS = [("KSDD2", "ksdd2"), ("MVTec-AD", "mvtec"), ("DAGM", "dagm"),
            ("BTAD", "btad"), ("DTD-Synthetic", "dtd")]
MODELS = [("VisualAD（官方权重，本机复现）", "baseline"),
          ("VisualAD（同预算续训，公平基线）", "finetune_baseline"),
          ("ATS-ViT（同预算续训）", "finetune_ats"),
          ("ATS-ViT（tuned：α lr×0.1，2 epoch）", "finetune_ats_tuned")]


def load(tag, ds, pro=False):
    f = OUT / "evals" / f"{tag}__{ds}{'_pro' if pro else ''}" / "summary.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text(encoding="utf-8"))
    return [float(x) for x in d["mean"]]


def row(vals):
    if not vals:
        return ["—"] * 7
    return [f"{v:.1f}" if v > -50 else "n/a" for v in vals]


def repr_ckpt(tag):
    f = OUT / "ckpt" / f"{tag}.pth"
    if not f.exists():
        return None
    ck = torch.load(f, map_location="cpu")
    if not ck.get("uses_ats"):
        return {"uses_ats": False}
    alphas = {k: round(float(sd["ats.alpha"]), 5) for k, sd in ck["layer_transforms"].items()
              if "ats.alpha" in sd}
    return {"uses_ats": True, "alphas": alphas,
            "spectrum_mean": round(float(ck["layer_transforms"]["layer_24"]["ats.spectrum"].mean()), 4)}


def main():
    O = []
    w = O.append
    w("# ATS-ViT 实验报告（自动生成）\n")
    w("> 本文件由 `scripts/make_report.py` 从磁盘上的评测产物自动生成，**不含任何手工填写的数字**。\n"
      "> 缺失项一律显示 `—`/`n/a`，未跑完的配置不会被估算。\n")

    # ---------------- 1. 结论摘要 ----------------
    base = json.loads((OUT / "baseline_reference.json").read_text(encoding="utf-8"))["baseline"]
    w("\n## 1. 一句话结论\n")
    ats = {ds: load("finetune_ats", d) for ds, d in DATASETS}
    fb = {ds: load("finetune_baseline", d) for ds, d in DATASETS}
    tuned = {ds: load("finetune_ats_tuned", d) for ds, d in DATASETS}
    have = [(k, ats[k], fb[k]) for k in ats if ats[k] and fb[k]]
    if have:
        better = sum(1 for _, a, b in have if a[4] > b[4])
        w(f"- 在 {len(have)} 个数据集的图像级 AUROC 上，ATS-ViT 相对**同预算基线**"
          f"（VisA 1008 张子集续训 1 epoch）胜出 **{better}/{len(have)}**。\n")
        diffs = [a[4] - b[4] for _, a, b in have]
        w(f"- 图像级 AUROC 平均变化 **{sum(diffs)/len(diffs):+.2f}**；"
          f"像素级 AUROC 平均变化 "
          f"**{sum(a[0]-b[0] for _,a,b in have)/len(have):+.2f}**。\n")
    else:
        w("- 评测尚未跑完，暂无对比结论。\n")

    # ---------------- 2. 主表 ----------------
    w("\n## 2. 主实验（零样本跨数据集：源域 VisA → 目标域）\n")
    w("### 2.1 图像级（AUROC / F1-max / AP）\n")
    w("| 数据集 | VisualAD 官方 | 同预算基线 | ATS-ViT | ATS-ViT tuned | Δ(ATS−同预算) |")
    w("|:--|--:|--:|--:|--:|--:|")
    for ds, key in DATASETS:
        b = base.get(ds)
        fbv = fb.get(ds)
        av = ats.get(ds)
        tv = tuned.get(ds)
        cells = []
        for v in (b, fbv, av, tv):
            cells.append("—" if not v else " / ".join(f"{x:.1f}" for x in v[4:]))
        delta = "—"
        if av and fbv:
            delta = f"{av[4]-fbv[4]:+.1f}"
        w(f"| {ds} | " + " | ".join(cells) + f" | {delta} |")

    w("\n### 2.2 像素级（AUROC / F1-max / AP / PRO，PRO 需补算轮）\n")
    w("| 数据集 | VisualAD 官方 | 同预算基线 | ATS-ViT | Δ(ATS−同预算) |")
    w("|:--|--:|--:|--:|--:|")
    for ds, key in DATASETS:
        b = base.get(ds)
        fbv = load("finetune_baseline", key, pro=True) or fb.get(ds)
        av = load("finetune_ats", key, pro=True) or ats.get(ds)
        fmt = lambda v: "—" if not v else " / ".join(f"{x:.1f}" for x in v[:4])
        delta = "—" if not (av and fbv) else f"{av[0]-fbv[0]:+.1f}"
        w(f"| {ds} | {fmt(b)} | {fmt(fbv)} | {fmt(av)} | {delta} |")

    # ---------------- 3. 训练配置与产物 ----------------
    w("\n## 3. 训练配置（公平性说明）\n")
    w("| 项 | 设置 |\n|:--|:--|")
    w("| 源域 | VisA 分层子集 **1008 张**（12 类 × 84，固定随机种子 0）|")
    w("| 初始化 | 官方 `weight/train_on_visa/CLIP.pth` 热启动（嵌入层 strict=False）|")
    w("| 预算 | 两侧完全相同：1 epoch / bs=1 / lr=1e-3 / AdamW / image_size=518 |")
    w("| 唯一差异 | ATS-ViT 打开 `--use_ats`（tuned 版本另把 α 学习率放大到 0.1×，训 2 epoch）|")
    w("| 设备 | RTX 2070 SUPER 8GB，单卡 |")

    w("\n### 训练后 ATS 参数变化（可解释性证据）\n")
    w("| ckpt | uses_ats | α（layer 6/12/18/24） | spectrum 均值 |\n|:--|:--|:--|:--|")
    for tag in ("finetune_baseline", "finetune_ats", "finetune_ats_tuned"):
        r = repr_ckpt(tag)
        if not r:
            continue
        if r["uses_ats"]:
            w(f"| {tag} | ✅ | {r['alphas']} | {r['spectrum_mean']} |")
        else:
            w(f"| {tag} | ❌（纯 SAF） | — | — |")

    # ---------------- 4. 效率 ----------------
    bf = OUT / "bench.json"
    if bf.exists():
        bench = json.loads(bf.read_text(encoding="utf-8"))
        w("\n## 4. 效率（本机实测，bs=1，518×518）\n")
        w("| 模型 | 可训练参数 (M) | 单图延迟 (ms) | FPS |\n|:--|--:|--:|--:|")
        for k, v in bench.items():
            w(f"| {k} | {v['trainable_params_m']:.2f} | {v['latency_ms']:.1f} | {v['fps']:.2f} |")

    # ---------------- 5. 图 ----------------
    fig = OUT / "figures"
    w("\n## 5. 图（300 DPI，PNG + PDF）\n")
    for p in sorted(fig.glob("*.png")):
        w(f"- `{p.name}`  （矢量版 `{p.with_suffix('.pdf').name}`）")
    if not any(fig.glob("*.png")):
        w("- 尚未生成")

    # ---------------- 6. 局限 ----------------
    w("\n## 6. 未完成 / 局限（务必如实对外说明）\n")
    w("1. **消融 (a)(b)(c)(e)** 需要各自重训一个变体，本轮 GPU 预算内未跑，`tables/table4_ablation.md` 中标记 `pending`。")
    w("2. 源域用的是 VisA **1008 张分层子集**而非全量 2162 张（为控制单夜 GPU 时间），两侧同设置因此相对比较有效，"
      "但**与论文绝对值不完全可比**。")
    w("3. 单一随机种子，未做多次重复的均值±方差；提升幅度小于 1 个点时不应过度解读。")
    w("4. DAGM 的 `Class1/Class7/Class8` 在整类排队评测时会因 `roc_auc_score` 内部 float64 副本 OOM，需单独重跑（已按类拆开）。")
    w("5. DINOv2 分支未实现（官方代码亦无），本工作只覆盖 CLIP ViT-L/14@336px。")

    (OUT / "REPORT.md").write_text("\n".join(O) + "\n", encoding="utf-8")
    print("写出:", OUT / "REPORT.md")


if __name__ == "__main__":
    main()
