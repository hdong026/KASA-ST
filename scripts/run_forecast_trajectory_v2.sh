#!/usr/bin/env bash
# =============================================================================
# ForecastTrajectory V2 — ONE master runner
#
# V1 is scientifically invalid and must NOT be resumed.
#
# Engineering acceptance:
#   bash scripts/run_forecast_trajectory_v2.sh --acceptance-1epoch --gpu 0 --seed 1
#
# Full scientific pipeline (resumable, new run dir, never reuses V1 markers):
#   bash scripts/run_forecast_trajectory_v2.sh --full --gpu 0 --seed 1
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
  bash scripts/run_forecast_trajectory_v2.sh --acceptance-1epoch --gpu 0 --seed 1
  bash scripts/run_forecast_trajectory_v2.sh --full --gpu 0 --seed 1
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
mkdir -p results checkpoints/PEMS04/H12/forecast_trajectory_v2
STAMP="$(date +%Y%m%d_%H%M%S)"
LOGDIR=results/forecast_trajectory_v2_logs
mkdir -p "$LOGDIR"
LOG="${LOGDIR}/forecast_trajectory_v2_${MODE}_${STAMP}.log"
exec > >(tee -a "$LOG") 2>&1

echo "=== ForecastTrajectoryV2 (${MODE}) ==="
echo "gpu=${GPU} seed=${SEED} log=${LOG}"
echo "python=$(command -v python)"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
echo "V1 artifacts preserved (not deleted). V1 training is not resumed."

PY=scripts/run_forecast_trajectory_v2_pipeline.py
case "$MODE" in
  acceptance)
    python "$PY" --acceptance-1epoch --gpu 0 --seed "$SEED"
    ;;
  full)
    python "$PY" --full --gpu 0 --seed "$SEED"
    ;;
esac

echo "done mode=${MODE} log=${LOG}"
