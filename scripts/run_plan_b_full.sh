#!/usr/bin/env bash
# =============================================================================
# Plan B Full — one-shot GRPO-inspired Group-Relative Refinement Policy
#
# Runs:
#   1) formal policy training (temporal crossfit oracle rewards)
#   2) BasicTS VALID + TEST evaluation (MAE/RMSE/MAPE + cost/stages)
#
# SAFETY:
#   * Requires --confirm-full-run
#   * Does NOT retrain supernet / crossfit teachers
#   * Does NOT build a TEST route oracle
#
# Example:
#   bash scripts/run_plan_b_full.sh --confirm-full-run --gpu 0
#   bash scripts/run_plan_b_full.sh --confirm-full-run --gpu 0 --skip-train
#   bash scripts/run_plan_b_full.sh --confirm-full-run --gpu 0 --eval-only
# =============================================================================

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONFIRM_FULL_RUN=0
GPU=0
NUM_EPOCHS=30
BATCH_SIZE=32
ETA_MODE=discrete
SKIP_TRAIN=0
EVAL_ONLY=0
SKIP_EVAL=0

CROSSFIT_ORACLE="results/pems04_temporal_crossfit_refinement_oracle.json"
VALID_ORACLE="results/pems04_budget_f2f_oracle_valid_rawscale.json"
STABLE_CKPT="checkpoints/PEMS04/H12/budget_f2f/supernet_eta0p50_dynamic_fair_rawscale_loss_v2_60f53aa1c6/seed1/b5678fda5e8d94ed028c6c8bb073461d/BudgetConditionedAdaptiveF2FNet_best_val_MAE.pt"
STABLE_CFG="checkpoints/PEMS04/H12/budget_f2f/supernet_eta0p50_dynamic_fair_rawscale_loss_v2_60f53aa1c6/seed1/b5678fda5e8d94ed028c6c8bb073461d/H12_supernet_eta0p50_dynamic_fair_rawscale_loss_v2_60f53aa1c6_seed1.py"
POLICY_OUT="checkpoints/PEMS04/H12/budget_f2f/group_relative_policy.pt"
EVAL_OUT="results/planB_policy_eval.json"
LOG_DIR="results/plan_b_full_logs"
STAMP="$(date +%Y%m%d_%H%M%S)"
MASTER_LOG="${LOG_DIR}/plan_b_full_${STAMP}.log"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_plan_b_full.sh --confirm-full-run [options]

Required:
  --confirm-full-run          Allow formal Plan B training

Options:
  --gpu ID                    CUDA device id (default: 0)
  --num-epochs N              Policy epochs (default: 30)
  --batch-size N              Batch size (default: 32)
  --eta-mode MODE             discrete|continuous (default: discrete)
  --skip-train                Skip training; require existing policy ckpt
  --eval-only                 Alias of --skip-train (train skipped, eval runs)
  --skip-eval                 Train only; do not run VALID/TEST eval
  --crossfit-oracle PATH
  --valid-oracle PATH
  --supernet-checkpoint PATH
  --cfg PATH
  --policy-out PATH
  --eval-out PATH
  -h, --help
EOF
}

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$MASTER_LOG"; }
die() { log "ERROR: $*"; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --confirm-full-run) CONFIRM_FULL_RUN=1; shift ;;
    --gpu) GPU="$2"; shift 2 ;;
    --num-epochs) NUM_EPOCHS="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --eta-mode) ETA_MODE="$2"; shift 2 ;;
    --skip-train|--eval-only) SKIP_TRAIN=1; shift ;;
    --skip-eval) SKIP_EVAL=1; shift ;;
    --crossfit-oracle) CROSSFIT_ORACLE="$2"; shift 2 ;;
    --valid-oracle) VALID_ORACLE="$2"; shift 2 ;;
    --supernet-checkpoint) STABLE_CKPT="$2"; shift 2 ;;
    --cfg) STABLE_CFG="$2"; shift 2 ;;
    --policy-out) POLICY_OUT="$2"; shift 2 ;;
    --eval-out) EVAL_OUT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown arg: $1" ;;
  esac
done

mkdir -p "$LOG_DIR" "$(dirname "$POLICY_OUT")" "$(dirname "$EVAL_OUT")"
: >"$MASTER_LOG"

[[ "$CONFIRM_FULL_RUN" -eq 1 ]] || die "refusing: pass --confirm-full-run"

for f in "$CROSSFIT_ORACLE" "$VALID_ORACLE" "$STABLE_CKPT" "$STABLE_CFG"; do
  [[ -f "$f" ]] || die "missing required file: $f"
done

DEVICE="cuda:${GPU}"
log "Plan B full start"
log "device=$DEVICE epochs=$NUM_EPOCHS batch=$BATCH_SIZE eta_mode=$ETA_MODE"
log "policy_out=$POLICY_OUT"
log "eval_out=$EVAL_OUT"
log "master_log=$MASTER_LOG"

if [[ "$SKIP_TRAIN" -eq 0 ]]; then
  log "=== STEP 1/2: train group-relative policy ==="
  python scripts/train_group_relative_refinement_policy.py \
    --confirm-full-run \
    --device "$DEVICE" \
    --batch-size "$BATCH_SIZE" \
    --num-epochs "$NUM_EPOCHS" \
    --eta-mode "$ETA_MODE" \
    --crossfit-oracle "$CROSSFIT_ORACLE" \
    --valid-oracle "$VALID_ORACLE" \
    --supernet-checkpoint "$STABLE_CKPT" \
    --cfg "$STABLE_CFG" \
    --out "$POLICY_OUT" \
    2>&1 | tee -a "$MASTER_LOG"
else
  log "=== STEP 1/2: skip train ==="
fi

[[ -f "$POLICY_OUT" ]] || die "policy checkpoint missing after train: $POLICY_OUT"

if [[ "$SKIP_EVAL" -eq 0 ]]; then
  log "=== STEP 2/2: BasicTS VALID+TEST eval ==="
  python scripts/eval_group_relative_refinement_policy.py \
    --policy-checkpoint "$POLICY_OUT" \
    --supernet-checkpoint "$STABLE_CKPT" \
    --cfg "$STABLE_CFG" \
    --valid-oracle "$VALID_ORACLE" \
    --split both \
    --device "$DEVICE" \
    --batch-size "$BATCH_SIZE" \
    --etas 0.0 0.25 0.5 0.75 1.0 \
    --out "$EVAL_OUT" \
    2>&1 | tee -a "$MASTER_LOG"
else
  log "=== STEP 2/2: skip eval ==="
fi

log "DONE"
log "policy: $POLICY_OUT"
log "eval:   $EVAL_OUT"
log "log:    $MASTER_LOG"
