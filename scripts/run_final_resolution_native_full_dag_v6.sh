#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="/home/dhz/miniconda3/envs/basicts/bin/python"
GPU_ID="${GPU_ID:-0}"
RESULT_ROOT="$ROOT_DIR/results/f2f_cot_resolution_native_full_dag_rich_grpo"
CHECKPOINT_ROOT="$ROOT_DIR/checkpoints/PEMS04/H12/f2f_cot_resolution_native_full_dag_rich_grpo"
SEED1_CHECKPOINT="$CHECKPOINT_ROOT/formal_v5_full_anchor_seed1/router_best.pt"
TRAIN_CACHE="$ROOT_DIR/results/f2f_cot_resolution_native_route_complete_oracle/continuation_c60_seed1/cache/train_8route_cache.npz"
VALID_CACHE="$ROOT_DIR/results/f2f_cot_resolution_native_route_complete_oracle/continuation_c60_seed1/cache/valid_8route_cache.npz"
SEEDS=(1 2 3 4 5)

[[ "$GPU_ID" =~ ^[0-9]+$ ]] || { echo "GPU_ID must be a non-negative integer" >&2; exit 2; }
for required in "$PYTHON_BIN" "$SEED1_CHECKPOINT" "$TRAIN_CACHE" "$VALID_CACHE"; do
  [[ -e "$required" ]] || { echo "Missing required path: $required" >&2; exit 2; }
done

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
tag="final_formal_v6_${stamp}_pid$$"
run_dir="$RESULT_ROOT/$tag"
while [[ -e "$run_dir" ]]; do
  tag="final_formal_v6_${stamp}_pid$$-$RANDOM"
  run_dir="$RESULT_ROOT/$tag"
done
mkdir -p "$run_dir"

exec > >(tee -a "$run_dir/launcher.log") 2>&1
echo "[final-v6] run_dir=$run_dir"
echo "[final-v6] gpu=$GPU_ID seeds=${SEEDS[*]}"
printf 'seed_protocol=%s\n' "${SEEDS[*]}" > "$run_dir/manifest.txt"
printf 'selection_protocol=VALID-only; TEST evaluated only after each checkpoint is frozen\n' >> "$run_dir/manifest.txt"

echo "[final-v6] structural checks"
CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" scripts/test_f2f_cot_resolution_native_v1_route_complete.py
CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" scripts/test_resolution_native_route_complete_grpo.py

valid_reports=()
test_reports=()
for seed in "${SEEDS[@]}"; do
  if [[ "$seed" == "1" ]]; then
    checkpoint="$SEED1_CHECKPOINT"
    echo "[final-v6] reusing protected finalized seed-1 checkpoint: $checkpoint"
  else
    echo "[final-v6] training unchanged finalized router seed=$seed"
    CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" scripts/train_resolution_native_route_complete_rich_grpo.py \
      --gpu 0 \
      --seed "$seed" \
      --tag "$tag" \
      --epochs 7 \
      --patience 3 \
      --selection-budget 0.75
    checkpoint="$CHECKPOINT_ROOT/${tag}_seed${seed}/router_best.pt"
  fi
  [[ -f "$checkpoint" ]] || { echo "Missing seed-$seed router checkpoint: $checkpoint" >&2; exit 3; }
  sha256sum "$checkpoint" | tee "$run_dir/router_seed${seed}.sha256"

  seed_dir="$run_dir/seed${seed}"
  mkdir -p "$seed_dir"
  echo "[final-v6] VALID evaluation seed=$seed"
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" scripts/evaluate_resolution_native_route_complete_rich_grpo.py \
    --checkpoint "$checkpoint" \
    --gpu 0 \
    --seed "$seed" \
    --split valid \
    --output "$seed_dir/valid_report.json" \
    --bootstrap 500 \
    --full-budget-fallback train-best
  echo "[final-v6] TEST evaluation seed=$seed (post-freeze only)"
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" scripts/evaluate_resolution_native_route_complete_rich_grpo.py \
    --checkpoint "$checkpoint" \
    --gpu 0 \
    --seed "$seed" \
    --split test \
    --output "$seed_dir/test_report.json" \
    --bootstrap 500 \
    --full-budget-fallback train-best
  valid_reports+=("$seed_dir/valid_report.json")
  test_reports+=("$seed_dir/test_report.json")
done

echo "[final-v6] mean/std aggregation"
"$PYTHON_BIN" scripts/aggregate_resolution_native_route_complete_rich_grpo.py \
  --reports "${valid_reports[@]}" "${test_reports[@]}" \
  --output "$run_dir/mean_std_aggregate.json"

echo "[final-v6] TRAIN/VALID all-route oracle decomposition"
"$PYTHON_BIN" scripts/analyze_resolution_native_full_budget_oracle.py \
  --train-cache "$TRAIN_CACHE" \
  --valid-cache "$VALID_CACHE" \
  --output "$run_dir/full_budget_oracle_decomposition.json"

echo "[final-v6] complete: $run_dir"
