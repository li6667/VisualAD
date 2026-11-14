#!/bin/bash
# ============================================================================
# ATS-ViT 全量实验流水线
#   Phase 3: warm-start 微调（baseline vs +ATS，同一数据同一预算，公平对比）
#   Phase 4: 6 数据集零样本评测（论文协议）+ 金属子类拆解 + 消融
#   Phase 5: 论文级图表（另见 scripts/generate_paper_figures.py）
#
# 设计说明（重要，写进论文/汇报里要如实说明）：
#   1. 官方发布权重 weight/train_on_visa/CLIP.pth 已经是「VisA 1 epoch」的产物，
#      因此本脚本对 baseline 与 ATS 采用**完全相同**的续训预算：
#      VisA 的 6 个类 × 1 epoch（约 1080 张），避免"ATS 多训了一轮"的不公平比较。
#   2. 评测走论文零样本跨数据集协议：源域 VisA 训完 → 测 MVTec/KSDD2/DAGM/BTAD/DTD。
#   3. 快速筛选轮默认 SKIP_PRO=1（PRO 是最慢的一步），随后对 KSDD2/BTAD/MVTec 补算 PRO。
#
# 用法: bash scripts/run_ats_vit_all.sh [--no-train]
# ============================================================================
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/../.venv/Scripts/python.exe"
DATA="/d/ls/Anomaly Detection/datasets"
OUT="$ROOT/../results_ats_vit"
LOG="$OUT/experiment.log"
EPOCHS=1
BS=1
DEV=cuda:0
SRC_CLASSES=(candle capsules cashew chewinggum fryum macaroni1)
USE_SUBSET=${USE_SUBSET:-0}   # 0=完整 VisA(12类/2162张，与论文配方一致)

mkdir -p "$OUT/ckpt" "$OUT/figures" "$OUT/tables" "$OUT/evals"
export PYTHONIOENCODING=utf-8

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

subset_meta() {  # 生成 VisA 子集 meta（源域训练用）
  "$PY" - "$DATA/visa" "${SRC_CLASSES[@]}" <<'PYEOF'
import json, sys
root, classes = sys.argv[1], sys.argv[2:]
m = json.load(open(f"{root}/meta.json", encoding="utf-8"))
sub = {"train": {}, "test": {c: m["test"][c] for c in classes if c in m["test"]}}
for c in classes:
    if c in m.get("train", {}):
        sub["train"][c] = m["train"][c]
json.dump(sub, open(f"{root}/meta.subset.json", "w", encoding="utf-8"), indent=2)
print("subset meta:", len(sub["test"]), "类,", sum(len(v) for v in sub["test"].values()), "张")
PYEOF
}

train_one() {  # $1=tag  $2=额外参数
  local tag="$1"; shift
  local ck="$OUT/ckpt/${tag}.pth"
  [ -f "$ck" ] && { log "跳过训练（已存在）$ck"; return 0; }
  log "TRAIN [$tag] 源域=VisA/6类 epoch=$EPOCHS bs=$BS extra=$*"
  "$PY" "$ROOT/train.py" \
      --train_data_path "$DATA/visa" --train_dataset visa \
      --save_path "$OUT/ckpt" --epoch $EPOCHS --batch_size $BS \
      --image_size 518 --device $DEV \
      --pretrained_path "$ROOT/weight/train_on_visa/CLIP.pth" "$@" >> "$LOG" 2>&1
  # train.py 固定输出 epoch_1.pth / final_model.pth，重命名成本次 tag
  mv -f "$OUT/ckpt/epoch_${EPOCHS}.pth" "$ck"
  log "TRAIN [$tag] 完成 -> $ck"
}

evaluate() {  # $1=ckpt tag  $2=结果子目录名  $3=data_path  $4=dataset_name  $5=SKIP_PRO(1/0)
  local tag="$1" ds="$2" path="$3" dname="$4" skip="${5:-1}"
  local res="$OUT/evals/${tag}__${ds}"
  if [ -f "$res/summary.json" ]; then log "跳过评测（已存在）$res"; return 0; fi
  mkdir -p "$res"
  log "EVAL [$tag] on $ds (SKIP_PRO=$skip)"
  if [ "$skip" = "1" ]; then export SKIP_PRO=1; else unset SKIP_PRO 2>/dev/null || true; fi
  "$PY" -u "$DATA/eval_per_class.py" "$path" "$dname"       "$OUT/ckpt/${tag}.pth" "$res" >> "$LOG" 2>&1
  unset SKIP_PRO 2>/dev/null || true
  [ -f "$res/summary.json" ] && log "  -> $(grep -o '"mean": \[[^]]*\]' "$res/summary.json" | head -1)"                              || log "  !! $ds 评测未产出 summary.json"
}

# ---------------------------------------------------------------- Phase 3
if [ "${1:-}" != "--no-train" ]; then
  cp -f "$DATA/visa/meta.json" "$OUT/meta.visa.full.json"
  if [ "$USE_SUBSET" = "1" ]; then
    subset_meta | tee -a "$LOG"
    cp -f "$DATA/visa/meta.subset.json" "$DATA/visa/meta.json"
    log "源域 meta 已切换为 VisA 6 类子集"
  else
    log "源域 = 完整 VisA（12 类 / 2162 张），与论文 1 epoch 配方一致"
  fi
  train_one "finetune_baseline"
  train_one "finetune_ats" --use_ats
  cp -f "$OUT/meta.visa.full.json" "$DATA/visa/meta.json"
  log "源域 meta 已还原为完整 VisA"
fi

# ---------------------------------------------------------------- Phase 4
for tag in finetune_baseline finetune_ats; do
  evaluate "$tag" ksdd2 "$DATA/ksdd2" SDD 1
  evaluate "$tag" btad  "$DATA/btad/BTech_Dataset_transformed" btad 1
  evaluate "$tag" dtd   "$DATA/dtd" DTD 1
  evaluate "$tag" dagm  "$DATA/dagm/pic1" DAGM_KaggleUpload 1
  evaluate "$tag" mvtec "$DATA/mvtec" mvtec 1
done

# PRO 补算（KSDD2 / BTAD / MVTec 便宜，DAGM 太慢放在最后按需）
for tag in finetune_baseline finetune_ats; do
  for spec in "ksdd2|$DATA/ksdd2|SDD" "btad|$DATA/btad/BTech_Dataset_transformed|btad" "mvtec|$DATA/mvtec|mvtec"; do
    IFS='|' read -r ds path dname <<< "$spec"
    rm -rf "$OUT/evals/${tag}__${ds}_pro"
    evaluate "$tag" "${ds}_pro" "$path" "$dname" 0
  done
done

log "全部评测完成，开始汇总表格"
"$PY" "$ROOT/scripts/summarize_tables.py" >> "$LOG" 2>&1 && log "表格已生成: $OUT/tables/"
log "PIPELINE DONE"
