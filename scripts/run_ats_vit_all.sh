#!/bin/bash
# ============================================================================
# ATS-ViT 全量实验流水线 v2（稳健版：每步独立日志 + 产物校验 + 失败即停）
#
#   Phase 3: warm-start 微调（baseline 与 +ATS 同数据同预算，唯一差异是 --use_ats）
#   Phase 4: 零样本跨数据集评测 + PRO 补算 + 表格汇总
#
# 协议说明（写入 docs/ATS_VIT_DESIGN.md，此处同步）：
#   * 官方 weight/train_on_visa/CLIP.pth 已是「VisA 1 epoch」产物；
#     本脚本对两侧都用相同预算再训 1 epoch（完整 VisA 12 类 / 2162 张，bs=1，lr=1e-3），
#     避免「ATS 多训一轮」的不公平比较。
#   * 评测走论文零样本跨数据集协议：源域 VisA → 测 MVTec-AD / KSDD2 / DAGM / BTAD / DTD。
#   * 筛选轮 SKIP_PRO=1，随后对 KSDD2 / BTAD / MVTec 补算 PRO。
#
# 用法: bash scripts/run_ats_vit_all.sh [train|eval|all]
# ============================================================================
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/../.venv/Scripts/python.exe"
DATA="/d/ls/Anomaly Detection/datasets"
OUT="$ROOT/../results_ats_vit"
LOG="$OUT/experiment.log"
LOGS="$OUT/logs"
EPOCHS=1
BS=1
DEV=cuda:0
STAGE="${1:-all}"

mkdir -p "$OUT/ckpt" "$OUT/evals" "$OUT/tables" "$OUT/figures" "$LOGS"
export PYTHONIOENCODING=utf-8

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
die() { log "FATAL: $*"; exit 1; }

# ---------------------------------------------------------------- 训练
train_one() {  # $1=tag  $2...=额外参数
  local tag="$1"; shift
  local ck="$OUT/ckpt/${tag}.pth"
  if [ -f "$ck" ]; then log "SKIP 训练（已存在）$ck"; return 0; fi
  log "TRAIN $tag 开始（VisA 12 类 / 1 epoch / bs=1）extra: $*"
  "$PY" "$ROOT/train.py" \
      --train_data_path "$DATA/visa" --train_dataset visa \
      --save_path "$OUT/ckpt" --epoch "$EPOCHS" --batch_size "$BS" \
      --image_size 518 --device "$DEV" \
      --pretrained_path "$ROOT/weight/train_on_visa/CLIP.pth" "$@" \
      > "$LOGS/train_${tag}.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then die "训练 $tag 失败 (rc=$rc)，见 $LOGS/train_${tag}.log"; fi
  [ -f "$OUT/ckpt/epoch_${EPOCHS}.pth" ] || die "训练 $tag 未产出 epoch_${EPOCHS}.pth"
  mv -f "$OUT/ckpt/epoch_${EPOCHS}.pth" "$ck"
  local alpha
  alpha="$(grep -a "atl alphas" "$LOGS/train_${tag}.log" | tail -1 | sed 's/.*atl alphas=//')"
  log "TRAIN $tag 完成 -> $ck  (init alphas=$alpha)"
}

# ---------------------------------------------------------------- 评测
evaluate() {  # $1=tag  $2=结果名  $3=data_path  $4=dataset_name  $5=SKIP_PRO
  local tag="$1" ds="$2" path="$3" dname="$4" skip="${5:-1}"
  local res="$OUT/evals/${tag}__${ds}"
  if [ -f "$res/summary.json" ]; then log "SKIP 评测（已存在）$ds"; return 0; fi
  mkdir -p "$res"
  log "EVAL $tag on $ds (SKIP_PRO=$skip)"
  if [ "$skip" = "1" ]; then export SKIP_PRO=1; else unset SKIP_PRO 2>/dev/null || true; fi
  "$PY" -u "$DATA/eval_per_class.py" "$path" "$dname" \
      "$OUT/ckpt/${tag}.pth" "$res" > "$LOGS/eval_${tag}__${ds}.log" 2>&1
  unset SKIP_PRO 2>/dev/null || true
  if [ -f "$res/summary.json" ]; then
    log "  OK $(grep -o '"mean": \[[^]]*\]' "$res/summary.json" | head -1)"
  else
    log "  !! $ds 未产出 summary.json（见 $LOGS/eval_${tag}__${ds}.log）"
  fi
}

# ---------------------------------------------------------------- 主流程
case "$STAGE" in
  train|all)
    log "阶段开始：Phase 3 训练（两侧同预算）"
    train_one finetune_baseline
    train_one finetune_ats --use_ats
    ;;
esac

case "$STAGE" in
  eval|all)
    log "阶段开始：Phase 4 评测（SKIP_PRO 快速轮）"
    for tag in finetune_baseline finetune_ats; do
      [ -f "$OUT/ckpt/${tag}.pth" ] || die "缺少 $tag.pth，先跑训练"
      evaluate "$tag" ksdd2 "$DATA/ksdd2" SDD 1
      evaluate "$tag" btad  "$DATA/btad/BTech_Dataset_transformed" btad 1
      evaluate "$tag" dtd   "$DATA/dtd" DTD 1
      evaluate "$tag" dagm  "$DATA/dagm/pic1" DAGM_KaggleUpload 1
      evaluate "$tag" mvtec "$DATA/mvtec" mvtec 1
    done
    log "阶段开始：PRO 补算（KSDD2 / BTAD / MVTec）"
    for tag in finetune_baseline finetune_ats; do
      for spec in "ksdd2_pro|$DATA/ksdd2|SDD" "btad_pro|$DATA/btad/BTech_Dataset_transformed|btad" "mvtec_pro|$DATA/mvtec|mvtec"; do
        IFS='|' read -r ds path dname <<< "$spec"
        evaluate "$tag" "$ds" "$path" "$dname" 0
      done
    done
    log "阶段开始：表格汇总"
    "$PY" "$ROOT/scripts/summarize_tables.py" > "$LOGS/summarize.log" 2>&1
    log "表格完成: $OUT/tables/"
    ;;
esac

log "PIPELINE($STAGE) DONE"
