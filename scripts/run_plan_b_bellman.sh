#!/usr/bin/env bash
# Formal Budgeted Bellman Plan B runner (long). Requires --confirm-full-run.
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIRM=0
GPU=0
SEED=1
SEEDS=()
MAX_Q1=300
MAX_Q0=300
MAX_JOINT=100
ENABLE_JOINT=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --confirm-full-run) CONFIRM=1; shift ;;
    --gpu) GPU="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --seeds) shift; while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do SEEDS+=("$1"); shift; done ;;
    --max-epochs-q1) MAX_Q1="$2"; shift 2 ;;
    --max-epochs-q0) MAX_Q0="$2"; shift 2 ;;
    --max-epochs-joint) MAX_JOINT="$2"; shift 2 ;;
    --no-joint) ENABLE_JOINT=0; shift ;;
    *) echo "Unknown arg: $1"; exit 2 ;;
  esac
done

if [[ $CONFIRM -ne 1 ]]; then
  echo "Refusing: pass --confirm-full-run to launch formal Bellman training."
  exit 2
fi

if [[ ${#SEEDS[@]} -eq 0 ]]; then
  SEEDS=("$SEED")
fi

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONUNBUFFERED=1
LOGDIR=results/plan_b_bellman_full_logs
mkdir -p "$LOGDIR" results checkpoints/PEMS04/H12/budget_f2f/plan_b_bellman
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="$LOGDIR/plan_b_bellman_full_${STAMP}.log"

exec > >(tee -a "$LOG") 2>&1
echo "=== Plan B Bellman formal run ==="
echo "log=$LOG seeds=${SEEDS[*]} gpu=$GPU"

for S in "${SEEDS[@]}"; do
  OUT="checkpoints/PEMS04/H12/budget_f2f/plan_b_bellman/seed${S}"
  mkdir -p "$OUT"
  CACHE=results/planB_bellman_oof_cache

  echo "STEP 1: OOF cache"
  if [[ ! -f "$CACHE/manifest.json" ]]; then
    python - <<PY
from basicts.archs.arch_zoo.ChainForecasting_arch.bellman_refinement_dataset import build_bellman_oof_cache
import torch
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
meta=build_bellman_oof_cache(out_dir="$CACHE", device=device)
print(meta)
PY
  else
    echo "reusing $CACHE"
  fi

  echo "STEP 2: train Q1 to convergence"
  python scripts/train_bellman_refinement.py \
    --phase q1 --seed "$S" --device cuda:0 \
    --cache-dir "$CACHE" --out-dir "$OUT" \
    --max-epochs-q1 "$MAX_Q1"

  echo "STEP 3: train Q0 to convergence"
  python scripts/train_bellman_refinement.py \
    --phase q0 --seed "$S" --device cuda:0 \
    --cache-dir "$CACHE" --out-dir "$OUT" \
    --max-epochs-q0 "$MAX_Q0"

  if [[ $ENABLE_JOINT -eq 1 ]]; then
    echo "STEP 4: optional joint (enable flag in train script)"
    python scripts/train_bellman_refinement.py \
      --phase joint --enable-joint --seed "$S" --device cuda:0 \
      --cache-dir "$CACHE" --out-dir "$OUT" \
      --max-epochs-joint "$MAX_JOINT" || true
  fi

  echo "STEP 5-8: VALID+TEST eval (TEST once, no TEST oracle)"
  python scripts/eval_bellman_refinement.py \
    --router "$OUT/router_best.pt" \
    --device cuda:0 \
    --split both \
    --out-valid results/planB_bellman_valid_eval.json \
    --out-test results/planB_bellman_test_eval.json

  python scripts/audit_bellman_refinement.py --cache-dir "$CACHE"
done

echo "DONE formal Bellman. log=$LOG"
