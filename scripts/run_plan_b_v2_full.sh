#!/usr/bin/env bash
# =============================================================================
# Plan B-v2 Full — Exact Full-Information Sequential Forecast Refinement Policy
#
# NOT original GRPO/GSPO.
#
# Steps:
#   1) build/reuse dual-view OOF state cache
#   2) train exact full-information policy
#   3) VALID evaluation / checkpoint selection (nontrivial regimes)
#   4) TEST BasicTS final evaluation (NO test oracle)
#
# SAFETY:
#   * Requires --confirm-full-run
#   * Does NOT overwrite Plan B-v1 group_relative_policy.pt
#   * Does NOT retrain supernet / crossfit teachers / Plan A
#
# Example:
#   bash scripts/run_plan_b_v2_full.sh --confirm-full-run --gpu 0
# =============================================================================

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONFIRM_FULL_RUN=0
GPU=0
NUM_EPOCHS=30
BATCH_SIZE=32
SKIP_CACHE=0
SKIP_TRAIN=0
SKIP_EVAL=0

CROSSFIT_ORACLE="results/pems04_temporal_crossfit_refinement_oracle.json"
VALID_ORACLE="results/pems04_budget_f2f_oracle_valid_rawscale.json"
STABLE_CKPT="checkpoints/PEMS04/H12/budget_f2f/supernet_eta0p50_dynamic_fair_rawscale_loss_v2_60f53aa1c6/seed1/b5678fda5e8d94ed028c6c8bb073461d/BudgetConditionedAdaptiveF2FNet_best_val_MAE.pt"
CACHE_DIR="results/planB_v2_oof_state_cache"
POLICY_OUT="checkpoints/PEMS04/H12/budget_f2f/plan_b_v2_exact_policy.pt"
EVAL_OUT="results/planB_v2_policy_eval.json"
LOG_DIR="results/plan_b_v2_full_logs"
STAMP="$(date +%Y%m%d_%H%M%S)"
MASTER_LOG="${LOG_DIR}/plan_b_v2_full_${STAMP}.log"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_plan_b_v2_full.sh --confirm-full-run [options]

Required:
  --confirm-full-run          Allow formal Plan B-v2 training

Options:
  --gpu ID
  --num-epochs N
  --batch-size N
  --skip-cache
  --skip-train
  --skip-eval
  --cache-dir PATH
  --policy-out PATH
  --eval-out PATH
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
    --skip-cache) SKIP_CACHE=1; shift ;;
    --skip-train) SKIP_TRAIN=1; shift ;;
    --skip-eval) SKIP_EVAL=1; shift ;;
    --cache-dir) CACHE_DIR="$2"; shift 2 ;;
    --policy-out) POLICY_OUT="$2"; shift 2 ;;
    --eval-out) EVAL_OUT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown arg: $1" ;;
  esac
done

mkdir -p "$LOG_DIR" "$(dirname "$POLICY_OUT")" "$(dirname "$EVAL_OUT")"
: >"$MASTER_LOG"

[[ "$CONFIRM_FULL_RUN" -eq 1 ]] || die "refusing: pass --confirm-full-run"

# Never overwrite V1
if [[ "$POLICY_OUT" == *"group_relative_policy.pt" ]]; then
  die "refusing to overwrite Plan B-v1 path: $POLICY_OUT"
fi

DEVICE="cuda:${GPU}"
log "Plan B-v2 full start (Exact Full-Information — NOT GRPO/GSPO)"
log "device=$DEVICE epochs=$NUM_EPOCHS batch=$BATCH_SIZE"
log "policy_out=$POLICY_OUT cache=$CACHE_DIR"

if [[ "$SKIP_CACHE" -eq 0 ]]; then
  log "=== STEP 1/4: dual-view OOF state cache ==="
  python scripts/train_plan_b_v2.py \
    --confirm-full-run \
    --build-cache-only \
    --device "$DEVICE" \
    --crossfit-oracle "$CROSSFIT_ORACLE" \
    --supernet-checkpoint "$STABLE_CKPT" \
    --cache-dir "$CACHE_DIR" \
    2>&1 | tee -a "$MASTER_LOG"
fi

if [[ "$SKIP_TRAIN" -eq 0 ]]; then
  log "=== STEP 2/4: train exact full-information policy ==="
  python scripts/train_plan_b_v2.py \
    --confirm-full-run \
    --device "$DEVICE" \
    --batch-size "$BATCH_SIZE" \
    --num-epochs "$NUM_EPOCHS" \
    --crossfit-oracle "$CROSSFIT_ORACLE" \
    --valid-oracle "$VALID_ORACLE" \
    --supernet-checkpoint "$STABLE_CKPT" \
    --cache-dir "$CACHE_DIR" \
    --out "$POLICY_OUT" \
    2>&1 | tee -a "$MASTER_LOG"
fi

[[ -f "$POLICY_OUT" ]] || die "policy checkpoint missing: $POLICY_OUT"

if [[ "$SKIP_EVAL" -eq 0 ]]; then
  log "=== STEP 3-4/4: VALID (+ TEST BasicTS path) ==="
  python scripts/eval_plan_b_v2.py \
    --policy-checkpoint "$POLICY_OUT" \
    --supernet-checkpoint "$STABLE_CKPT" \
    --valid-oracle "$VALID_ORACLE" \
    --split both \
    --device "$DEVICE" \
    --batch-size "$BATCH_SIZE" \
    --out "$EVAL_OUT" \
    2>&1 | tee -a "$MASTER_LOG"
fi

log "DONE"
log "policy: $POLICY_OUT"
log "eval:   $EVAL_OUT"
log "log:    $MASTER_LOG"
