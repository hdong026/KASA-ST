#!/usr/bin/env bash
# =============================================================================
# ForecastTrajectory — ONE master runner
#
# The user does NOT need to call train/cache/policy/eval scripts individually.
#
# Engineering acceptance (mandatory before full science):
#   bash scripts/run_forecast_trajectory_learning.sh --acceptance-1epoch --gpu 0 --seed 1
#
# Full scientific pipeline (resumable):
#   bash scripts/run_forecast_trajectory_learning.sh --full --gpu 0 --seed 1
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODE=""
GPU=0
SEED=1

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_forecast_trajectory_learning.sh --acceptance-1epoch --gpu 0 --seed 1
  bash scripts/run_forecast_trajectory_learning.sh --full --gpu 0 --seed 1

This single command orchestrates:
  PHASE 0  preflight / unit tests
  PHASE 1  acceptance-1epoch if the PASS marker is missing
  PHASE 2  formal transition training (auto-extend to <=250)
  PHASE 3  restore best transition checkpoint
  PHASE 4  full CUDA latency profiling
  PHASE 5  TRAIN trajectory + prefix cache
  PHASE 6  VALID diagnostic cache
  PHASE 7  exact trajectory-oracle analysis (TRAIN; no eta)
  PHASE 8  policy internal train/valid split
  PHASE 9  exact policy training
  PHASE 10 restore best policy checkpoint
  PHASE 11 official VALID evaluation
  PHASE 12 final TEST evaluation (frozen; no TEST oracle)
  PHASE 13 final report
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --acceptance-1epoch) MODE="acceptance"; shift ;;
    --full) MODE="full"; shift ;;
    --gpu) GPU="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1"; usage; exit 2 ;;
  esac
done

if [[ -z "$MODE" ]]; then
  echo "Refusing: specify --acceptance-1epoch or --full"
  usage
  exit 2
fi

# Prefer an environment that already has torch; otherwise activate conda env basicts.
if ! python -c "import torch" >/dev/null 2>&1; then
  if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
    conda activate basicts
  elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "${HOME}/anaconda3/etc/profile.d/conda.sh"
    conda activate basicts
  fi
fi

export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
mkdir -p results checkpoints/PEMS04/H12/forecast_trajectory
STAMP="$(date +%Y%m%d_%H%M%S)"
LOGDIR=results/forecast_trajectory_logs
mkdir -p "$LOGDIR"
LOG="${LOGDIR}/forecast_trajectory_${MODE}_${STAMP}.log"
exec > >(tee -a "$LOG") 2>&1

echo "=== ForecastTrajectory (${MODE}) ==="
echo "gpu=${GPU} seed=${SEED} log=${LOG}"
echo "python=$(command -v python)"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

PY=scripts/run_forecast_trajectory_pipeline.py
case "$MODE" in
  acceptance)
    python "$PY" --acceptance-1epoch --gpu 0 --seed "$SEED"
    ;;
  full)
    python "$PY" --full --gpu 0 --seed "$SEED"
    ;;
esac

echo "done mode=${MODE} log=${LOG}"
