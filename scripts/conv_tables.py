"""Conv-ViT：把评测结果汇总成论文表格（Markdown + LaTeX）。

数据来源（全部真实评测产物，不手填）：
  * D:/ls/Anomaly Detection/results/            VisualAD 基线的本机复现（官方权重）
  * D:/ls/Anomaly Detection/results_conv_vit/   本次 Conv-ViT 各配置的评测

产出到 results_conv_vit/tables/：
  table1_main.md/.tex        主表：官方 / 同预算基线 / Ours，图像级 + 像素级
  table2_ablation.md/.tex    消融：w/o SCA、w/o SAF、w/o both、Full
  table3_metal.md            金属子类逐类对比
  table4_efficiency.md       参数量 / 延迟 / FPS
"""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path("D:/ls/Anomaly Detection/results_conv_vit")
TAB = OUT / "tables"
TAB.mkdir(parents=True, exist_ok=True)

# 数据集 -> (评测子目录名, 是否有 PRO 补算)
DATASETS = [
    ("KSDD2", "ksdd2", True),
    ("MVTec-AD", "mvtec", True),
    ("DAGM", "dagm", False),
    ("BTAD", "btad", True),
    ("DTD-Synthetic", "dtd", False),
]
METAL = ["metal_nut", "grid", "screw", "cable", "transistor", "tile"]

# 官方权重复现结果（本机实测，见 VisualAD/docs/REPRODUCTION.md）
OFFICIAL = {
    "KSDD2": [98.5, 60.9, 62.1, 98.5, 98.0, 93.9, 98.3],
    "MVTec-AD": [90.8, 43.6, 41.2, 87.6, 92.2, 93.2, 96.7],
    "DAGM": [92.2, 58.0, 56.4, 89.2, 99.5, 95.0, 97.8],
    "BTAD": [91.1, 49.8, 43.1, 80.8, 94.9, 93.9, 97.0],
    "DTD-Synthetic": [98.1, 64.3, 65.5, 94.8, 97.5, 96.7, 99.1],
}


def summary(tag: str, sub: str):
    f = OUT / "evals" / f"{tag}__{sub}" / "summary.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def get(tag: str, sub: str):
    """返回 7 项 [pix_auroc, pix_f1, pix_ap, pix_pro, img_auroc, img_f1, img_ap]；PRO 缺失记 None。"""
    s = summary(tag, sub)
    if s:
        v = [float(x) for x in s["mean"]]
        return [None if x <= -50 else x for x in v]
    pro = summary(tag, f"{sub}_pro")
    if pro:
        v = [float(x) for x in pro["mean"]]
        return [None if x <= -50 else x for x in v]
    return None


def fmt(v, nd=1):
    return "—" if v is None else f"{v:.{nd}f}"


def d(a, b):
    return "—" if (a is None or b is None) else f"{a - b:+.1f}"


def md(header, rows, title=None):
    out = []
    if title:
        out += [f"# {title}", ""]
    out.append("| " + " | ".join(header) + " |")
    out.append("|" + "|".join([":--"] + ["--:"] * (len(header) - 1)) + "|")
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out) + "\n"


def tex(caption, label, header, rows):
    L = [r"\begin{table}[t]", r"\centering", r"\small",
         rf"\caption{{{caption}}}", rf"\label{{{label}}}",
         r"\begin{tabular}{l" + "c" * (len(header) - 1) + "}", r"\toprule",
         " & ".join(header) + r" \\", r"\midrule"]
    for r in rows:
        L.append(" & ".join(str(x) for x in r) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L) + "\n"


PROVENANCE = """表格生成说明（投稿时请保留或在文中说明）

1. 所有「Ours / 同预算基线 / 消融」数字来自本机实测：results_conv_vit/evals/*/summary.json
2. 「VisualAD 官方」列为使用官方发布权重在本机复现的结果（docs/REPRODUCTION.md），
   不直接引用论文数字，避免协议不一致。
3. 消融的所有配置均从官方 adapter 热启动、同数据（VisA 1008 张分层子集）、同预算（1 epoch, bs=1, lr=1e-3），
   唯一差异是模块开关，保证可比。
4. PRO 只在 KSDD2 / MVTec-AD / BTAD 上补算；未算的显示 “—”，不做估算。
"""


def main():
    (TAB / "PROVENANCE.txt").write_text(PROVENANCE, encoding="utf-8")

    # ---------------- Table 1 主表 ----------------
    img_rows, pix_rows = [], []
    for name, sub, _ in DATASETS:
        o = OFFICIAL.get(name)
        b = get("baseline", sub)
        f = get("full", sub)
        img_rows.append([
            name,
            " / ".join(fmt(x) for x in (o[4:] if o else [None] * 3)),
            " / ".join(fmt(x) for x in (b[4:] if b else [None] * 3)),
            " / ".join(fmt(x) for x in (f[4:] if f else [None] * 3)),
            ", ".join(d(f[i], b[i]) for i in (4, 5, 6)) if (f and b) else "—",
        ])
        pix_rows.append([
            name,
            " / ".join(fmt(x) for x in (o[:4] if o else [None] * 4)),
            " / ".join(fmt(x) for x in (b[:4] if b else [None] * 4)),
            " / ".join(fmt(x) for x in (f[:4] if f else [None] * 4)),
            ", ".join(d(f[i], b[i]) for i in (0, 1, 2, 3)) if (f and b) else "—",
        ])

    h1 = ["Dataset", "VisualAD (official, reproduced)", "Baseline (same budget)",
          "Ours (SCA-C + SAF-C)", "Δ(Ours−Baseline)"]
    (TAB / "table1_main.md").write_text(
        md(h1, img_rows, "Table 1a. Image-level (AUROC / F1-max / AP)")
        + "\n" + md(h1, pix_rows, "Table 1b. Pixel-level (AUROC / F1-max / AP / PRO)"),
        encoding="utf-8")
    (TAB / "table1_main.tex").write_text(
        tex("Image-level zero-shot comparison (AUROC / F1-max / AP).", "tab:main_img", h1, img_rows)
        + "\n" + tex("Pixel-level zero-shot comparison (AUROC / F1-max / AP / PRO).", "tab:main_pix", h1, pix_rows),
        encoding="utf-8")

    # ---------------- Table 2 消融 ----------------
    # 注意：DAGM 只对 Baseline / Full 跑过；均值统一只在 KSDD2+MVTec 上算，
    # 避免"少一个数据集的行"均值被抬高而误导。
    configs = [("Baseline (SCA + SAF)", "baseline"),
               ("w/o SCA", "wo_sca"),
               ("w/o SAF", "wo_saf"),
               ("w/o both", "wo_both"),
               ("Full (SCA-C + SAF-C)", "full")]
    abl_rows = []
    for label, tag in configs:
        k = get(tag, "ksdd2")
        m = get(tag, "mvtec")
        dg = get(tag, "dagm")
        km_img = (k[4] + m[4]) / 2 if (k and m) else None
        km_pix = (k[0] + m[0]) / 2 if (k and m) else None
        abl_rows.append([
            label,
            f"{fmt(k[4])} / {fmt(k[0])}" if k else "—",
            f"{fmt(m[4])} / {fmt(m[0])}" if m else "—",
            f"{fmt(dg[4])} / {fmt(dg[0])}" if dg else "—",
            fmt(km_img), fmt(km_pix),
        ])

    h2 = ["Configuration", "KSDD2 img / pix", "MVTec img / pix", "DAGM img / pix",
          "Mean(K+M) img", "Mean(K+M) pix"]
    (TAB / "table2_ablation.md").write_text(
        md(h2, abl_rows, "Table 2. Ablation (image / pixel AUROC)")
        + (chr(10) + "> 所有行均从官方 adapter 热启动、同数据同预算（VisA 1008 张 × 1 epoch），唯一差异是模块开关。"
           + chr(10) + "> DAGM 只在 Baseline 与 Full 上评测（其余行为节省预算未跑，显示 “—”，未做任何估算）；"
           + chr(10) + "> 均值列统一在 KSDD2 + MVTec 上计算，保证各行可比。" + chr(10)),
        encoding="utf-8")
    (TAB / "table2_ablation.tex").write_text(
        tex("Module ablation (image / pixel AUROC); all variants share the same initialisation and budget.",
            "tab:ablation", h2, abl_rows), encoding="utf-8")

    # ---------------- Table 3 金属子类 ----------------
    b_met = summary("baseline", "mvtec_pro") or summary("baseline", "mvtec")
    f_met = summary("full", "mvtec_pro") or summary("full", "mvtec")
    k_b, k_f = get("baseline", "ksdd2"), get("full", "ksdd2")

    def rows_of(s):
        return {r[0].lower(): [float(x) for x in r[1:]] for r in (s or {}).get("rows", [])}

    rb, rf = rows_of(b_met), rows_of(f_met)
    metal_rows = []
    for m in METAL:
        vb, vf = rb.get(m), rf.get(m)
        metal_rows.append([m,
                           fmt(vb[4]) if vb else "—", fmt(vf[4]) if vf else "—",
                           fmt(vb[0]) if vb else "—", fmt(vf[0]) if vf else "—",
                           d(vf[4], vb[4]) if (vb and vf) else "—"])
    if k_b and k_f:
        metal_rows.append(["KSDD2 (cold-drawn micro-crack)",
                           fmt(k_b[4]), fmt(k_f[4]), fmt(k_b[0]), fmt(k_f[0]), d(k_f[4], k_b[4])])
    h3 = ["Category", "img AUROC (base)", "img AUROC (Ours)", "pix AUROC (base)", "pix AUROC (Ours)", "Δ img"]
    (TAB / "table3_metal.md").write_text(
        md(h3, metal_rows, "Table 3. Metallic / material-related categories"), encoding="utf-8")

    # ---------------- Table 4 效率 ----------------
    bench_f = OUT / "bench.json"
    if bench_f.exists():
        bench = json.loads(bench_f.read_text(encoding="utf-8"))
        h4 = ["Model", "Trainable params (M)", "Latency (ms/img)", "FPS"]
        r4 = [[k, f"{v['trainable_params_m']:.2f}", f"{v['latency_ms']:.1f}", f"{v['fps']:.2f}"]
              for k, v in bench.items()]
        (TAB / "table4_efficiency.md").write_text(
            md(h4, r4, "Table 4. Efficiency (bs=1, 518×518)"), encoding="utf-8")

    print("已生成：")
    for p in sorted(TAB.iterdir()):
        print("  ", p.name)


if __name__ == "__main__":
    main()
