#!/usr/bin/env bash
# Budgeted Bellman Plan B — mandatory smoke (hard wall 300s).
set -euo pipefail
cd "$(dirname "$0")/.."
GPU=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 2 ;;
  esac
done

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${GPU}"

mkdir -p results checkpoints/PEMS04/H12/budget_f2f/plan_b_bellman_smoke

# hard timeout 300s for main smoke
timeout 300s python scripts/run_bellman_smoke_pipeline.py --gpu 0
EC=$?
if [[ $EC -eq 124 ]]; then
  echo "BELLMAN_SMOKE_FAIL: exceeded 300s hard timeout"
  python - <<'PY'
import json
from pathlib import Path
p=Path("results/planB_bellman_smoke.json")
obj={"verdict":"BELLMAN_SMOKE_FAIL","reason":"hard_timeout_300s"}
if p.is_file():
    try:
        obj=json.loads(p.read_text()); obj["verdict"]="BELLMAN_SMOKE_FAIL"; obj["reason"]="hard_timeout_300s"
    except Exception:
        pass
p.write_text(json.dumps(obj, indent=2))
print(json.dumps(obj, indent=2)[:500])
PY
  exit 1
fi
if [[ $EC -ne 0 ]]; then
  exit $EC
fi

# Mandatory: exercise the SAME non-smoke train loop used by formal runner
# (previous gap: smoke-only path hid cuda/cpu bugs in eval_q1_decision_regret).
echo "=== formal-path smoke (1 epoch Q1+Q0, non-smoke loop) ==="
timeout 300s python scripts/train_bellman_refinement.py \
  --phase all \
  --formal-path-smoke \
  --cache-dir results/planB_bellman_oof_cache_smoke \
  --out-dir checkpoints/PEMS04/H12/budget_f2f/plan_b_bellman_formal_path_smoke \
  --device cuda:0 \
  --batch-size 8 \
  --seed 1
EC2=$?
if [[ $EC2 -ne 0 ]]; then
  echo "BELLMAN_SMOKE_FAIL: formal-path smoke failed"
  exit $EC2
fi
echo "FORMAL_PATH_SMOKE_PASS"
exit 0
