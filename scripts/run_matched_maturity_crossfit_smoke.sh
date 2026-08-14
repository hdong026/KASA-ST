#!/usr/bin/env bash
# Engineering smoke wrapper — delegates to the real full runner --smoke path.
set -euo pipefail
cd "$(dirname "$0")/.."
GPU="${1:-1}"
SEED="${2:-1}"
exec bash scripts/run_matched_maturity_crossfit_full.sh --smoke --gpu "$GPU" --seed "$SEED"
