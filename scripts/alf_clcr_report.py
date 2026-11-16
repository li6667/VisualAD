"""ALF / CLCR 结果汇总（对比同一 1008 张子集、同预算的 baseline）。

读取：
  results_conv_vit/evals/baseline__{ksdd2,mvtec}/summary.json   ← 同配方基线（无 ALF/CLCR）
  results_alf_clcr/evals/{alf,clcr,both}__*/summary.json
  results_alf_clcr/ckpt/*.pth（读 ALF 学到的权重）
输出：results_alf_clcr/REPORT_ALF_CLCR.md 与 tables/*.md
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

BASE = Path("D:/ls/Anomaly Detection/results_conv_vit")
OUT = Path("D:/ls/Anomaly Detection/results_alf_clcr")
TAB = OUT / "tables"

DATASETS = [("KSDD2", "ksdd2"), ("MVTec-AD", "mvtec"), ("DAGM", "dagm")]
CONFIGS = [("Baseline（等权相加，无 ALF/CLCR）", "baseline"),
           ("+ ALF（自适应层融合权重）", "alf"),
           ("+ CLCR（跨层一致性正则）", "clcr"),
           ("+ ALF + CLCR（Ours）", "both")]


def load(tag, sub):
    for root in (OUT / "evals", BASE / "evals"):
        for name in (f"{tag}__{sub}_pro", f"{tag}__{sub}"):
            f = root / name / "summary.json"
            if f.exists():
                v = [float(x) for x in json.loads(f.read_text(encoding="utf-8"))["mean"]]
                return [None if x <= -50 else x for x in v]
    return None


def fmt(v):
    return "—" if v is None else f"{v:.1f}"


def md(header, rows, title=None):
    out = [f"### {title}", ""] if title else []
    out.append("| " + " | ".join(header) + " |")
    out.append("|" + "|".join([":--"] + ["--:"] * (len(header) - 1)) + "|")
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out) + "\n"


def alf_weights(tag):
    f = OUT / "ckpt" / f"{tag}.pth"
    if not f.exists():
        return None
    cfg = torch.load(f, map_location="cpu").get("conv_config", {}) or {}
    lg = cfg.get("alf_logits")
    if lg is None:
        return None
    t = torch.as_tensor(lg, dtype=torch.float32)
    w = len(t) * torch.softmax(t, dim=0)
    return [round(float(x), 3) for x in w]


def main():
    TAB.mkdir(parents=True, exist_ok=True)
    O, p = [], None

    def w(s=""):
        O.append(s)

    w("# ALF / CLCR 实验报告（自动生成）\n")
    w("> 基线为**同一 1008 张 VisA 子集、同 1 epoch、同热启动**的 `baseline.pth`，"
      "唯一差异是 ALF / CLCR 开关。数字全部来自磁盘评测产物。\n")

    # ---------- 主对比 ----------
    rows_img, rows_pix = [], []
    for label, tag in CONFIGS:
        cells_i, cells_p = [], []
        for ds, sub in DATASETS:
            v = load(tag, sub)
            if v is None:
                cells_i += ["—", "—"]
                cells_p += ["—", "—"]
            else:
                cells_i += [fmt(v[4]), fmt(v[0])]
                cells_p += [f"{fmt(v[0])} / {fmt(v[1])} / {fmt(v[2])}", f"{fmt(v[4])} / {fmt(v[5])} / {fmt(v[6])}"]
        rows_img.append([label] + cells_i)
        rows_pix.append([label] + cells_p)

    h_img = ["配置", "KSDD2 img", "KSDD2 pix", "MVTec img", "MVTec pix", "DAGM img", "DAGM pix"]
    h_pix = ["配置", "KSDD2 (pix AUROC/F1/AP)", "KSDD2 (img AUROC/F1/AP)",
             "MVTec (pix AUROC/F1/AP)", "MVTec (img AUROC/F1/AP)"]
    w(md(h_img, rows_img, "1. AUROC 对比（img / pix）"))
    w(md(h_pix, rows_pix, "2. 三指标对比"))

    # ---------- ALF 权重 ----------
    w("\n### 3. ALF 学到的层权重（初值均为 1.0）\n")
    rows = []
    for label, tag in CONFIGS:
        wt = alf_weights(tag)
        if wt:
            rows.append([label, *wt, f"{sum(wt):.3f}"])
        else:
            rows.append([label, "—", "—", "—", "—", "—"])
    w(md(["配置", "layer 6", "layer 12", "layer 18", "layer 24", "sum"], rows))

    # ---------- 结论 ----------
    w("\n### 4. 自动结论\n")
    b = {ds: load("baseline", sub) for ds, sub in DATASETS}
    for label, tag in CONFIGS[1:]:
        cur = {ds: load(tag, sub) for ds, sub in DATASETS}
        pairs = [(ds, cur[ds], b[ds]) for ds, _ in DATASETS if cur[ds] and b[ds]]
        if not pairs:
            w(f"- {label}: 评测未完成")
            continue
        di = sum(c[4] - bb[4] for _, c, bb in pairs) / len(pairs)
        dp = sum(c[0] - bb[0] for _, c, bb in pairs) / len(pairs)
        wins = sum(1 for _, c, bb in pairs if c[4] > bb[4])
        w(f"- {label}: 图像 AUROC 平均变化 **{di:+.2f}**（胜 {wins}/{len(pairs)}），"
          f"像素 AUROC 平均变化 **{dp:+.2f}**")

    (OUT / "REPORT_ALF_CLCR.md").write_text("\n".join(O) + "\n", encoding="utf-8")
    (TAB / "alf_clcr_table.md").write_text("\n".join(O), encoding="utf-8")
    print("写出:", OUT / "REPORT_ALF_CLCR.md")


if __name__ == "__main__":
    main()
