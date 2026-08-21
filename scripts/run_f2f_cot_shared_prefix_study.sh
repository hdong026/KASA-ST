#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
exec /home/dhz/miniconda3/envs/basicts/bin/python scripts/f2f_cot_shared_prefix_study.py "$@"
