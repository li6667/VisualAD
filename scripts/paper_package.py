"""按原论文（VisualAD, CVPR'26）的写法生成全部表格。

产出到 results_conv_vit/paper/：
  table1_image.md/.tex     主表·图像级：4 个对比方法（引用值） + 本方法
  table2_pixel.md/.tex     主表·像素级
  table3_ablation.md/.tex  消融（打勾式：SCA-C / SAF-C 的四种组合 + 无卷积分支的基线）
  table4_metal.md/.tex     金属子类拆解
  table5_efficiency.md     参数量/延迟/FPS

约定：
  * 对比方法的数字为**原文引用值**，表注统一写 ``reported from [VisualAD]``；
  * 本方法 / 基线 / 消融为**本机实测**（results_conv_vit/evals）；
  * 缺失项显示 "—"，不做任何估算。
"""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path("D:/ls/Anomaly Detection/results_conv_vit")
PAPER = OUT / "paper"
PAPER.mkdir(parents=True, exist_ok=True)

# ---------------- 原文 Table 1 引用值（工业部分） ----------------
BASE_IMG = {
    "MVTec-AD": {"WinCLIP": [90.4, 92.7, 95.6], "APRIL-GAN": [86.1, 90.4, 93.6],
                 "AnomalyCLIP": [91.6, 92.7, 96.2], "AdaCLIP": [92.0, 92.7, 96.4]},
    "VisA": {"WinCLIP": [75.6, 78.2, 78.8], "APRIL-GAN": [77.4, 78.6, 80.9],
             "AnomalyCLIP": [81.0, 80.3, 84.4], "AdaCLIP": [79.7, 79.6, 83.2]},
    "BTAD": {"WinCLIP": [68.2, 67.8, 70.9], "APRIL-GAN": [73.7, 68.7, 69.9],
             "AnomalyCLIP": [88.7, 86.0, 90.6], "AdaCLIP": [90.0, 87.2, 91.5]},
    "KSDD2": {"WinCLIP": [93.5, 86.4, 94.2], "APRIL-GAN": [90.4, 82.9, 92.0],
              "AnomalyCLIP": [91.9, 84.5, 93.4], "AdaCLIP": [94.9, 90.3, 96.2]},
    "DAGM": {"WinCLIP": [91.8, 75.8, 79.5], "APRIL-GAN": [94.4, 80.3, 83.9],
             "AnomalyCLIP": [98.0, 90.6, 92.4], "AdaCLIP": [98.3, 91.5, 94.2]},
    "DTD-Synthetic": {"WinCLIP": [95.1, 94.1, 97.7], "APRIL-GAN": [85.5, 89.1, 94.0],
                      "AnomalyCLIP": [93.7, 94.3, 97.4], "AdaCLIP": [92.1, 92.4, 96.3]},
}
BASE_PIX = {
    "MVTec-AD": {"WinCLIP": [82.3, 24.8, 18.2, 62.0], "APRIL-GAN": [87.5, 42.3, 39.1, 43.7],
                 "AnomalyCLIP": [91.0, 38.9, 34.4, 81.7], "AdaCLIP": [88.5, 43.9, 41.0, 47.6]},
    "VisA": {"WinCLIP": [73.2, 9.0, 5.4, 51.1], "APRIL-GAN": [93.8, 32.6, 26.2, 86.5],
             "AnomalyCLIP": [95.4, 27.6, 20.7, 86.4], "AdaCLIP": [95.1, 33.8, 29.2, 71.3]},
    "BTAD": {"WinCLIP": [72.7, 18.5, 12.9, 27.3], "APRIL-GAN": [91.3, 40.1, 37.7, 21.0],
             "AnomalyCLIP": [93.0, 47.1, 41.5, 71.0], "AdaCLIP": [87.7, 42.3, 36.6, 17.1]},
    "KSDD2": {"WinCLIP": [94.1, 24.6, 17.4, 77.6], "APRIL-GAN": [94.5, 64.2, 66.9, 39.2],
              "AnomalyCLIP": [95.6, 43.0, 39.6, 73.5], "AdaCLIP": [98.0, 50.6, 43.7, 90.8]},
    "DAGM": {"WinCLIP": [87.6, 12.7, 6.8, 65.7], "APRIL-GAN": [84.4, 35.1, 27.8, 12.7],
             "AnomalyCLIP": [96.9, 56.9, 53.6, 89.2], "AdaCLIP": [88.6, 48.5, 42.6, 37.6]},
    "DTD-Synthetic": {"WinCLIP": [79.5, 16.1, 9.8, 51.5], "APRIL-GAN": [94.9, 60.4, 61.0, 33.8],
                      "AnomalyCLIP": [97.5, 55.8, 52.5, 87.9], "AdaCLIP": [95.1, 58.4, 56.1, 34.3]},
}
METAL = ["cable", "grid", "metal_nut", "screw", "tile", "transistor"]
# 本方法与基线的数据集 -> 评测目录名（VisA 由 MVTec 训练的模型给出）
OURS = {"KSDD2": ("full", "ksdd2"), "MVTec-AD": ("full", "mvtec"), "DAGM": ("full", "dagm"),
        "BTAD": ("full", "btad"), "DTD-Synthetic": ("full", "dtd"), "VisA": ("full_mvtec", "visa")}
BASELINE = {"KSDD2": ("baseline", "ksdd2"), "MVTec-AD": ("baseline", "mvtec"),
            "DAGM": ("baseline", "dagm"), "BTAD": ("baseline", "btad"),
            "DTD-Synthetic": ("baseline", "dtd")}


def load(tag, sub):
    for name in (f"{tag}__{sub}_pro", f"{tag}__{sub}"):
        f = OUT / "evals" / name / "summary.json"
        if f.exists():
            v = [float(x) for x in json.loads(f.read_text(encoding="utf-8"))["mean"]]
            return [None if x <= -50 else x for x in v]
    return None


def f1(v):
    return "—" if v is None else f"{v:.1f}"


def md(header, rows, title="", note=""):
    out = [f"**{title}**", ""] if title else []
    out.append("| " + " | ".join(header) + " |")
    out.append("|" + "|".join([":--"] + ["--:"] * (len(header) - 1)) + "|")
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    if note:
        out.append("")
        out.append(note)
    return "\n".join(out) + "\n"


def tex(caption, label, header, rows, note=""):
    L = [r"\begin{table*}[t]", r"\centering", r"\small",
         rf"\caption{{{caption}}}", rf"\label{{{label}}}",
         r"\begin{tabular}{l" + "c" * (len(header) - 1) + "}", r"\toprule",
         " & ".join(header) + r" \\", r"\midrule"]
    for r in rows:
        L.append(" & ".join(str(x) for x in r) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    if note:
        L.append(rf"\par\smallskip\footnotesize {note}")
    L.append(r"\end{table*}")
    return "\n".join(L) + "\n"


def main():
    NOTE_MAIN = "对比方法为原文引用值（reported from VisualAD, CVPR 2026）；Ours / Baseline 为本机实测。"
    ds_order = ["MVTec-AD", "VisA", "KSDD2", "BTAD", "DAGM", "DTD-Synthetic"]

    # ---------- Table 1 / 2 ----------
    rows_img, rows_pix = [], []
    for ds in ds_order:
        row = [ds]
        for m in ("WinCLIP", "APRIL-GAN", "AnomalyCLIP", "AdaCLIP"):
            row.append(" / ".join(f"{x:.1f}" for x in BASE_IMG[ds][m]))
        o = load(*OURS[ds]) if ds in OURS else None
        b = load(*BASELINE[ds]) if ds in BASELINE else None
        row.append(" / ".join(f1(x) for x in (o[4:] if o else [None] * 3)))
        best = max(BASE_IMG[ds][m][0] for m in BASE_IMG[ds])
        row.append("—" if not o else f"{o[4] - best:+.1f}")
        rows_img.append(row)

        row = [ds]
        for m in ("WinCLIP", "APRIL-GAN", "AnomalyCLIP", "AdaCLIP"):
            row.append(" / ".join(f"{x:.1f}" for x in BASE_PIX[ds][m]))
        row.append(" / ".join(f1(x) for x in (o[:4] if o else [None] * 4)))
        best = max(BASE_PIX[ds][m][0] for m in BASE_PIX[ds])
        row.append("—" if not o else f"{o[0] - best:+.1f}")
        rows_pix.append(row)

    h_img = ["Dataset", "WinCLIP", "APRIL-GAN", "AnomalyCLIP", "AdaCLIP", "**Ours**", "Δ vs best"]
    h_pix = ["Dataset", "WinCLIP", "APRIL-GAN", "AnomalyCLIP", "AdaCLIP", "**Ours**", "Δ vs best"]
    (PAPER / "table1_image.md").write_text(
        md(h_img, rows_img, "Table 1 (image-level): AUROC / F1-max / AP", NOTE_MAIN), encoding="utf-8")
    (PAPER / "table2_pixel.md").write_text(
        md(h_pix, rows_pix, "Table 2 (pixel-level): AUROC / F1-max / AP / PRO", NOTE_MAIN), encoding="utf-8")
    (PAPER / "table1_image.tex").write_text(
        tex("Image-level zero-shot comparison (AUROC / F1-max / AP).", "tab:img", h_img, rows_img,
            r"Baselines are quoted from \cite{visualad}; Ours/Baseline are measured locally."), encoding="utf-8")
    (PAPER / "table2_pixel.tex").write_text(
        tex("Pixel-level zero-shot comparison (AUROC / F1-max / AP / PRO).", "tab:pix", h_pix, rows_pix,
            r"Baselines are quoted from \cite{visualad}; Ours/Baseline are measured locally."), encoding="utf-8")

    # ---------- Table 3 消融（打勾式） ----------
    combos = [("w/o both（无 SCA、无 SAF）", "wo_both", "✗", "✗"),
              ("+ SAF-C only", "wo_sca", "✗", "✓"),
              ("+ SCA-C only", "wo_saf", "✓", "✗"),
              ("**Full (Ours)**", "full", "✓", "✓")]
    rows = []
    for label, tag, sca, saf in combos:
        k, m, dg = load(tag, "ksdd2"), load(tag, "mvtec"), load(tag, "dagm")
        rows.append([label, sca, saf,
                     f"{f1(k[4])} / {f1(k[0])}" if k else "—",
                     f"{f1(m[4])} / {f1(m[0])}" if m else "—",
                     f"{f1(dg[4])} / {f1(dg[0])}" if dg else "—"])
    # 基线（无卷积分支，SCA+SAF 原版）
    b_k, b_m, b_dg = load("baseline", "ksdd2"), load("baseline", "mvtec"), load("baseline", "dagm")
    rows.insert(0, ["Baseline (原版 SCA + SAF)", "—", "—",
                    f"{f1(b_k[4])} / {f1(b_k[0])}" if b_k else "—",
                    f"{f1(b_m[4])} / {f1(b_m[0])}" if b_m else "—",
                    f"{f1(b_dg[4])} / {f1(b_dg[0])}" if b_dg else "—"])
    h = ["Configuration", "SCA-C", "SAF-C", "KSDD2 img / pix", "MVTec img / pix", "DAGM img / pix"]
    ab_note = ("所有变体均从官方 adapter 热启动、同数据（VisA 1008 张分层子集）、同预算（1 epoch, bs=1）；"
               "唯一差异是 SCA-C / SAF-C 开关。SCA-C / SAF-C 的 1×1 末层零初始化，"
               "因此加入瞬间前向与原版逐位相同（max|Δ| = 0）。")
    (PAPER / "table3_ablation.md").write_text(
        md(h, rows, "Table 3 (ablation): SCA-C / SAF-C 的四种组合", ab_note), encoding="utf-8")
    (PAPER / "table3_ablation.tex").write_text(
        tex("Ablation on the two convolution branches (image / pixel AUROC).", "tab:abl", h, rows, ab_note),
        encoding="utf-8")

    # ---------- Table 4 金属子类 ----------

    src_b = OUT / "evals" / "baseline__mvtec_pro"
    src_o = OUT / "evals" / "full__mvtec_pro"
    rb, ro = {}, {}
    for src, dst in ((src_b, rb), (src_o, ro)):
        f = src / "summary.json"
        if f.exists():
            for r in json.loads(f.read_text(encoding="utf-8"))["rows"]:
                dst[r[0].lower()] = [float(x) for x in r[1:]]
    k_b, k_o = load("baseline", "ksdd2"), load("full", "ksdd2")
    metal_rows = []
    for m in METAL:
        vb, vo = rb.get(m), ro.get(m)
        metal_rows.append([m,
                           f"{f1(vb[4])} / {f1(vb[0])}" if vb else "—",
                           f"{f1(vo[4])} / {f1(vo[0])}" if vo else "—",
                           "—" if not (vb and vo) else f"{vo[4] - vb[4]:+.1f}"])
    if k_b and k_o:
        metal_rows.append(["KSDD2 (cold-drawn micro-crack)",
                           f"{f1(k_b[4])} / {f1(k_b[0])}", f"{f1(k_o[4])} / {f1(k_o[0])}",
                           f"{k_o[4] - k_b[4]:+.1f}"])
    metal_rows.append(["**Metal subset mean**",
                       f"{sum(rb[m][4] for m in METAL) / len(METAL):.1f} / "
                       f"{sum(rb[m][0] for m in METAL) / len(METAL):.1f}",
                       f"{sum(ro[m][4] for m in METAL) / len(METAL):.1f} / "
                       f"{sum(ro[m][0] for m in METAL) / len(METAL):.1f}", "—"])
    h4 = ["Category", "Baseline img / pix", "Ours img / pix", "Δ img"]
    (PAPER / "table4_metal.md").write_text(
        md(h4, metal_rows, "Table 4: 金属/材料相关子类（MVTec-AD + KSDD2）"), encoding="utf-8")
    (PAPER / "table4_metal.tex").write_text(
        tex("Metallic / material-related categories (image / pixel AUROC).", "tab:metal", h4, metal_rows),
        encoding="utf-8")

    # ---------- Table 5 效率 ----------
    ba = OUT / "bench_all.json"
    if ba.exists():
        bj = json.loads(ba.read_text(encoding="utf-8"))
        h5 = ["Model", "Trainable params (M)", "Latency (ms)", "FPS"]
        r5 = [[k, f"{v['trainable_params_m']:.2f}", f"{v['latency_ms']:.1f}", f"{v['fps']:.2f}"]
              for k, v in bj.items()]
        (PAPER / "table5_efficiency.md").write_text(
            md(h5, r5, "Table 5: 效率（bs=1，518×518，RTX 2070 SUPER）"), encoding="utf-8")

    print("已生成到", PAPER)
    for p in sorted(PAPER.iterdir()):
        print("  ", p.name)


if __name__ == "__main__":
    main()
