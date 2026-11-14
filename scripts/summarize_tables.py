"""ATS-ViT: 汇总实验结果为 Markdown / LaTeX 表格。

数据来源（全部为本机真实评测，不手工填写）：
  * ../results_ats_vit/baseline_reference.json —— VisualAD 基线的逐类/整表复现结果
  * ../results_ats_vit/evals/<ckpt>__<dataset>/summary.json —— eval_per_class.py 产出

输出：
  * tables/table1_image_level.md/.tex  图像级
  * tables/table2_pixel_level.md/.tex  像素级
  * tables/table3_metallic.md/.tex     金属/材料子类拆解 + 参数量/FPS
  * tables/table4_ablation.md          消融（未跑的配置显式标注 pending）
"""
from __future__ import annotations

import json
import os
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "results_ats_vit"
TAB = OUT / "tables"
TAB.mkdir(parents=True, exist_ok=True)

DATASETS = ["KSDD2", "MVTec-AD", "DAGM", "BTAD", "VisA", "DTD-Synthetic"]
DS_DIR = {
    "KSDD2": "ksdd2", "MVTec-AD": "mvtec", "DAGM": "dagm",
    "BTAD": "btad", "VisA": "visa", "DTD-Synthetic": "dtd",
}
MODELS = {
    "same_budget_baseline": "finetune_baseline",
    "ATS-ViT": "finetune_ats",
    "ATS-ViT-tuned": "ats_a0.05_lr0.1",
    "ATS-ViT-a0.20": "ats_a0.20_lr0.1",
}
SKIP_SENTINEL = -50.0  # SKIP_PRO 时 PRO 记 -100
METALS = ["cable", "grid", "metal_nut", "screw", "tile", "transistor"]


def load_summary(tag: str, ds_key: str, pro: bool = False):
    d = OUT / "evals" / f"{tag}__{ds_key}{'_pro' if pro else ''}"
    f = d / "summary.json"
    if not f.exists():
        return None
    return json.loads(f.read_text(encoding="utf-8"))


def get(model: str, ds: str, pro: bool = False):
    """返回 7 项指标 [pix_auroc, pix_f1, pix_ap, pix_pro, img_auroc, img_f1, img_ap]。

    pro=True 时优先取带 PRO 的补算结果；若该轮是 SKIP_PRO（PRO=-100）则视为缺失。
    """
    tag = MODELS[model]
    for want_pro in ((True, False) if pro else (False,)):
        s = load_summary(tag, DS_DIR[ds], pro=want_pro)
        if not s or "mean" not in s:
            continue
        vals = [float(x) for x in s["mean"]]
        if pro and vals[3] <= SKIP_SENTINEL:
            continue
        return vals
    return None


def baseline(ds: str):
    ref = json.loads((OUT / "baseline_reference.json").read_text(encoding="utf-8"))
    return ref["baseline"].get(ds)


def fmt(v, dash="-"):
    if v is None:
        return dash
    if isinstance(v, float) and v <= SKIP_SENTINEL:
        return "n/a"
    return f"{v:.1f}"


def delta(o, b):
    if o is None or b is None:
        return "n/a"
    if (isinstance(o, float) and o <= SKIP_SENTINEL) or (isinstance(b, float) and b <= SKIP_SENTINEL):
        return "n/a"
    return f"{o - b:+.1f}"


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join([":--"] + ["--:"] * (len(header) - 1)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out) + "\n"


def tex_table(caption, label, header, rows):
    colspec = "l" + "c" * (len(header) - 1)
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             rf"\begin{{tabular}}{{{colspec}}}", r"\toprule",
             " & ".join(header) + r" \\", r"\midrule"]
    for r in rows:
        lines.append(" & ".join(str(x).replace("%", r"\%") for x in r) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def main():
    (TAB / "PROVENANCE.txt").write_text(
        "所有数字来自本机真实评测：\n"
        "  baseline  -> ../results/ 下的逐类/整表结果（官方权重，已复现校验）\n"
        "  ATS-ViT   -> ../results_ats_vit/evals/<ckpt>__<dataset>/summary.json\n"
        "标 'pending' 的格子表示该配置尚未跑完，未做任何数值填充。\n",
        encoding="utf-8")

    # ---------------- Table 1 / 2: image & pixel ----------------
    def cols(ds, sl, pro):
        out = [baseline(ds)[sl] if baseline(ds) else None]
        out.append((get("same_budget_baseline", ds, pro=pro) or [None] * 7)[sl])
        out.append((get("ATS-ViT", ds, pro=pro) or [None] * 7)[sl])
        out.append((get("ATS-ViT-tuned", ds, pro=pro) or [None] * 7)[sl])
        return out

    img_rows, pix_rows = [], []
    for ds in DATASETS:
        oi = cols(ds, slice(4, 7), False)
        op = cols(ds, slice(0, 4), True)
        img_rows.append([ds] + [", ".join(fmt(x) for x in c) if isinstance(c, list) else fmt(c) for c in oi]
                        + [", ".join(delta(a, b) for a, b in zip(oi[2], oi[1]))])
        pix_rows.append([ds] + [", ".join(fmt(x) for x in c) if isinstance(c, list) else fmt(c) for c in op]
                        + [", ".join(delta(a, b) for a, b in zip(op[2], op[1]))])

    hdr_img = ["Dataset", "VisualAD 官方", "同预算基线", "ATS-ViT", "ATS-ViT (α-lr×10)", "Δ(ATS−同预算)"]
    hdr_pix = ["Dataset", "VisualAD 官方", "同预算基线", "ATS-ViT", "ATS-ViT (α-lr×10)", "Δ(ATS−同预算)"]
    title_img = "# 图像级（AUROC / F1-max / AP）" + chr(10) + chr(10)
    title_pix = "# 像素级（AUROC / F1-max / AP / PRO）" + chr(10) + chr(10)
    (TAB / "table1_image_level.md").write_text(title_img + md_table(hdr_img, img_rows), encoding="utf-8")
    (TAB / "table2_pixel_level.md").write_text(title_pix + md_table(hdr_pix, pix_rows), encoding="utf-8")
    (TAB / "table1_image_level.tex").write_text(
        tex_table("Image-level zero-shot comparison (AUROC/F1-max/AP).", "tab:image", hdr_img, img_rows),
        encoding="utf-8")
    (TAB / "table2_pixel_level.tex").write_text(
        tex_table("Pixel-level zero-shot comparison (AUROC/F1-max/AP/PRO).", "tab:pixel", hdr_pix, pix_rows),
        encoding="utf-8")

    # ---------------- Table 3: metallic subsets ----------------
    ref = json.loads((OUT / "baseline_reference.json").read_text(encoding="utf-8"))
    metal_rows = []
    for name, bvals in ref["metallic_subset_baseline"].items():
        s = load_summary("finetune_ats", DS_DIR["KSDD2"] if name == "KSDD2" else "mvtec", pro=True)
        ovals = None
        if s:
            for row in s.get("rows", []):
                if row[0].lower() == name.lower():
                    ovals = [float(x) for x in row[1:]]
                    break
        metal_rows.append([
            name,
            fmt(bvals[4]), fmt(bvals[5]), fmt(bvals[6]),
            fmt(ovals[4]) if ovals else "pending",
            fmt(ovals[5]) if ovals else "pending",
            fmt(ovals[6]) if ovals else "pending",
            fmt(bvals[0]), fmt(ovals[0]) if ovals else "pending",
        ])
    hdr_metal = ["Category", "img AUROC (base)", "img F1", "img AP",
                 "img AUROC (Ours)", "img F1", "img AP", "pix AUROC (base)", "pix AUROC (Ours)"]
    (TAB / "table3_metallic.md").write_text(md_table(hdr_metal, metal_rows), encoding="utf-8")

    # 参数量 / FPS
    bench_f = OUT / "bench.json"
    if bench_f.exists():
        bench = json.loads(bench_f.read_text(encoding="utf-8"))
        extra = ["\n### 效率（本机实测）\n",
                 md_table(["Model", "Trainable Params (M)", "Inference FPS (bs=1, 518px)", "Latency (ms/img)"],
                          [[k, f"{v['trainable_params_m']:.2f}", f"{v['fps']:.1f}", f"{v['latency_ms']:.1f}"]
                           for k, v in bench.items()])]
        (TAB / "table3_metallic.md").open("a", encoding="utf-8").write("\n".join(extra))

    # ---------------- Table 4: ablation ----------------
    abl = [
        ["(a) Baseline (Dual Tokens only)", "pending"],
        ["(b) + SCA", "pending"],
        ["(c) + SAF (Original MLP)", "pending"],
        ["(d) + SCA + SAF (= VisualAD)", ", ".join(fmt(x) for x in (baseline("VisA") or []))],
        ["(e) + SCA + ATS-SAF w/o spectral (channel gate only)", "pending"],
        ["(f) + SCA + ATS-SAF (Full, Ours)", "pending"],
    ]
    (TAB / "table4_ablation.md").write_text(
        "# 消融实验（Ablation Study）\n\n"
        "> (a)-(c)(e) 需要重新训练对应变体（每个变体 VisA 1 epoch ≈ 1.4h / 8GB bs=1），"
        "本轮 GPU 预算内未跑，标记 pending；**未做任何数值填充**。\n"
        "> (d) 为本机复现的 VisualAD 官方权重结果，(f) 为本次训练的 ATS-ViT 结果。\n\n"
        + md_table(["Configuration", "VisA pix(AUROC/F1/AP/PRO) img(AUROC/F1/AP)"], abl),
        encoding="utf-8")

    print("已生成:")
    for f in sorted(TAB.iterdir()):
        print("  ", f.name)


if __name__ == "__main__":
    main()
