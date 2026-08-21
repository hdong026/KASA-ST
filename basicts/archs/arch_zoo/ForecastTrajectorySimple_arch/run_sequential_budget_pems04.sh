#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
source /home/dhz/miniconda3/etc/profile.d/conda.sh
conda activate basicts
cd "${REPO_ROOT}"

GPU="${GPU:-0}"
exec python -m basicts.archs.arch_zoo.ForecastTrajectorySimple_arch.run_online_sequential_rl \
  --device "cuda:${GPU}" "$@"
