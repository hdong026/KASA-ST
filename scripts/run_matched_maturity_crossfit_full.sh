#!/usr/bin/env bash
# Matched-maturity crossfit FULL runner (real pipeline — not a stub).
#
# Trains 5 matched-maturity teachers + OOF oracle/Z3 + diagnostic audits.
# Does NOT train any controller / Q / Plan A / Plan B / Bellman / PPO / GRPO.
#
# Gates:
#   - engineering smoke should PASS before Fold1 acceptance
#   - full 5-fold requires Fold1 acceptance PASS unless --override-acceptance
set -euo pipefail
cd "$(dirname "$0")/.."

MODE=""
CONFIRM=0
OVERRIDE=0
GPU=0
SEED=1
ACCEPTANCE_FOLD=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke) MODE="smoke"; shift ;;
    --acceptance-fold) ACCEPTANCE_FOLD="$2"; MODE="acceptance"; shift 2 ;;
    --confirm-full-run) CONFIRM=1; MODE="full"; shift ;;
    --override-acceptance) OVERRIDE=1; shift ;;
    --gpu) GPU="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --help|-h)
      cat <<'EOF'
Usage:
  # Engineering smoke (1–5 min)
  bash scripts/run_matched_maturity_crossfit_full.sh --smoke --gpu 0 --seed 1

  # Fold1 acceptance (full 100-epoch Fold1 + OOF; ~1h)
  bash scripts/run_matched_maturity_crossfit_full.sh --acceptance-fold 1 --gpu 0 --seed 1

  # Full 5-fold (requires Fold1 acceptance PASS)
  bash scripts/run_matched_maturity_crossfit_full.sh --confirm-full-run --gpu 0 --seed 1

  # Explicit override (loud warning)
  bash scripts/run_matched_maturity_crossfit_full.sh --confirm-full-run --override-acceptance --gpu 0 --seed 1
EOF
      exit 0
      ;;
    *) echo "Unknown: $1"; exit 2 ;;
  esac
done

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONUNBUFFERED=1
LOGDIR=results/matched_maturity_crossfit_full_logs
mkdir -p "$LOGDIR" results checkpoints/PEMS04/H12/budget_f2f/matched_maturity_crossfit_v2
STAMP=$(date +%Y%m%d_%H%M%S)

if [[ -z "$MODE" ]]; then
  echo "Refusing: specify --smoke, --acceptance-fold 1, or --confirm-full-run"
  echo "Example full run (only after Fold1 acceptance PASS):"
  echo "  bash scripts/run_matched_maturity_crossfit_full.sh --confirm-full-run --gpu 0 --seed 1"
  exit 2
fi

if [[ "$MODE" == "full" && "$CONFIRM" -ne 1 ]]; then
  echo "Refusing: pass --confirm-full-run to launch full matched-maturity teacher training."
  exit 2
fi

if [[ "$MODE" == "full" ]]; then
  ACC=results/matched_maturity_crossfit_fold1_acceptance.json
  if [[ ! -f "$ACC" ]] || ! python -c "import json; from pathlib import Path; d=json.loads(Path('$ACC').read_text()); raise SystemExit(0 if d.get('MATCHED_FOLD1_ACCEPTANCE')=='PASS' else 1)"; then
    if [[ "$OVERRIDE" -ne 1 ]]; then
      echo "Refusing full launch: Fold1 acceptance PASS metadata missing/failed."
      echo "Run acceptance first:"
      echo "  bash scripts/run_matched_maturity_crossfit_full.sh --acceptance-fold 1 --gpu $GPU --seed $SEED"
      exit 2
    fi
    echo "======================================================================"
    echo "WARNING: --override-acceptance — launching WITHOUT Fold1 acceptance PASS"
    echo "======================================================================"
  fi
fi

LOG="$LOGDIR/matched_maturity_${MODE}_${STAMP}.log"
exec > >(tee -a "$LOG") 2>&1

echo "=== Matched-maturity crossfit (${MODE}) ==="
echo "gpu=$GPU seed=$SEED log=$LOG"
echo "NO controller/Q/adaptive-router training in this runner."
echo "Future protocol: adaptive selection only within matched OOF TRAIN;"
echo "  official VALID = forecasting/reporting only; TEST = final frozen eval only."

PY=scripts/run_matched_maturity_crossfit_pipeline.py
case "$MODE" in
  smoke)
    python "$PY" --smoke --gpu "$GPU" --seed "$SEED"
    ;;
  acceptance)
    python "$PY" --acceptance-fold "$ACCEPTANCE_FOLD" --gpu "$GPU" --seed "$SEED"
    ;;
  full)
    EXTRA=()
    if [[ "$OVERRIDE" -eq 1 ]]; then EXTRA+=(--override-acceptance); fi
    python "$PY" --confirm-full-run --gpu "$GPU" --seed "$SEED" "${EXTRA[@]}"
    ;;
esac

echo "done mode=$MODE log=$LOG"
