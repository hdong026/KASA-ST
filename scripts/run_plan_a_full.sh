#!/usr/bin/env bash
# =============================================================================
# Plan A Full — one-shot Temporal Cross-Fitted Refinement Supervision pipeline
#
# Runs end-to-end (user-operated formal experiment):
#   A5  prepare rolling-origin temporal crossfit manifest
#   A6  per-fold teacher supernet training (stable sandwich F2F)
#   A7  per-fold unseen oracle (raw physical scale)
#   A8  merge crossfit oracles
#   A9  train crossfit refinement gain controller
#   A10 validation evaluation
#   A11 optional test evaluation (final only)
#
# SAFETY:
#   * Requires --confirm-full-run (refuses otherwise)
#   * Always restores datasets/PEMS04/index_in12_out12.pkl on EXIT/ERR
#   * Does NOT use test for training or model selection
#
# Example:
#   bash scripts/run_plan_a_full.sh --confirm-full-run --gpu 1
#   bash scripts/run_plan_a_full.sh --confirm-full-run --gpu 1 --skip-completed
#   bash scripts/run_plan_a_full.sh --confirm-full-run --gpu 1 --from-step merge
# =============================================================================

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# -------------------- defaults --------------------
CONFIRM_FULL_RUN=0
GPU=1
SEED=1
NUM_BLOCKS=5
DATASET=PEMS04
HORIZON=12
SKIP_COMPLETED=0
RUN_TEST=0
FROM_STEP=prepare   # prepare|folds|merge|controller|eval_valid|eval_test
TO_STEP=eval_valid  # same vocabulary; inclusive
MAX_ROUTE_MAE=40.0  # oracle health gate (stable F2F ~18)

INDEX="datasets/PEMS04/index_in12_out12.pkl"
INDEX_BAK="datasets/PEMS04/index_in12_out12.pkl.bak_planA_full"
MANIFEST="results/temporal_crossfit_manifest.json"
MERGED_ORACLE="results/pems04_temporal_crossfit_refinement_oracle.json"
VALID_ORACLE="results/pems04_budget_f2f_oracle_valid_rawscale.json"
STABLE_CKPT="checkpoints/PEMS04/H12/budget_f2f/supernet_eta0p50_dynamic_fair_rawscale_loss_v2_60f53aa1c6/seed1/b5678fda5e8d94ed028c6c8bb073461d/BudgetConditionedAdaptiveF2FNet_best_val_MAE.pt"
CTRL_OUT_DIR="checkpoints/PEMS04/H12/budget_f2f/crossfit_refinement_controller"
LOG_DIR="results/plan_a_full_logs"
STAMP="$(date +%Y%m%d_%H%M%S)"
MASTER_LOG="${LOG_DIR}/plan_a_full_${STAMP}.log"

# -------------------- helpers --------------------
usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_plan_a_full.sh --confirm-full-run [options]

Required:
  --confirm-full-run          Allow formal supernet / controller training

Options:
  --gpu ID                    GPU id (default: 1)
  --seed N                    Seed (default: 1)
  --num-blocks N              Temporal blocks (default: 5 => 4 folds)
  --skip-completed            Skip fold if oracle JSON already healthy
  --run-test                  Also run final TEST eval after VALID
  --from-step STEP            prepare|folds|merge|controller|eval_valid|eval_test
  --to-step STEP              inclusive end step (default: eval_valid)
  --stable-ckpt PATH          Frozen forecasting supernet for controller
  --valid-oracle PATH         Official VALID route oracle
  --manifest PATH             Crossfit manifest output/input
  --merged-oracle PATH        Merged crossfit oracle path
  --ctrl-out-dir PATH         Controller checkpoint directory
  -h, --help                  Show help
EOF
}

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$MASTER_LOG"; }
die() { log "ERROR: $*"; exit 1; }

step_rank() {
  case "$1" in
    prepare) echo 1 ;;
    folds) echo 2 ;;
    merge) echo 3 ;;
    controller) echo 4 ;;
    eval_valid) echo 5 ;;
    eval_test) echo 6 ;;
    *) die "unknown step: $1" ;;
  esac
}

should_run() {
  local s="$1"
  local r fr tr
  r="$(step_rank "$s")"
  fr="$(step_rank "$FROM_STEP")"
  tr="$(step_rank "$TO_STEP")"
  [[ "$r" -ge "$fr" && "$r" -le "$tr" ]]
}

restore_index() {
  if [[ -f "$INDEX_BAK" ]]; then
    log "Restoring official index from $INDEX_BAK"
    mv -f "$INDEX_BAK" "$INDEX"
  fi
}

# Always restore index on EXIT/ERR/INT if a swap is in progress.
trap 'restore_index' EXIT INT TERM

find_fold_ckpt() {
  local fold="$1"
  local pattern="checkpoints/PEMS04/H12/budget_f2f/supernet_eta0p50_dynamic_fair_temporal_cf_fold${fold}_teacher_*/seed1/*/BudgetConditionedAdaptiveF2FNet_best_val_MAE.pt"
  # shellcheck disable=SC2086
  local hits
  hits="$(ls -1 $pattern 2>/dev/null | sort || true)"
  [[ -n "$hits" ]] || return 1
  echo "$hits" | tail -1
}

find_fold_cfg() {
  local ckpt="$1"
  local d
  d="$(dirname "$ckpt")"
  local cfg
  cfg="$(ls -1 "$d"/H12_supernet_eta0p50_dynamic_fair_temporal_cf_fold*_seed*.py 2>/dev/null | head -1 || true)"
  if [[ -z "$cfg" ]]; then
    # fallback: generated temp configs
    cfg="$(ls -1 generated/temp_configs_budget_f2f_pems04/H12_supernet_eta0p50_dynamic_fair_temporal_cf_fold*_seed*.py 2>/dev/null | sort | tail -1 || true)"
  fi
  [[ -n "$cfg" ]] || return 1
  echo "$cfg"
}

oracle_healthy() {
  local path="$1"
  python - "$path" "$MAX_ROUTE_MAE" <<'PY'
import json, sys
import numpy as np
path, thr = sys.argv[1], float(sys.argv[2])
d = json.loads(open(path).read())
recs = d.get("records") or []
if len(recs) < 10:
    print(f"FAIL n={len(recs)}"); sys.exit(1)
L = np.array([[e["final_mae"] for e in r["route_final_losses"]] for r in recs], dtype=float)
means = L.mean(0)
print("n=", len(recs), "mean_MAE=", np.round(means, 4).tolist())
if not np.isfinite(means).all():
    print("FAIL non-finite"); sys.exit(1)
if (means > thr).any() or (means < 1.0).any():
    print(f"FAIL health gate thr={thr}"); sys.exit(1)
print("OK")
PY
}

# -------------------- parse args --------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --confirm-full-run) CONFIRM_FULL_RUN=1; shift ;;
    --gpu) GPU="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --num-blocks) NUM_BLOCKS="$2"; shift 2 ;;
    --skip-completed) SKIP_COMPLETED=1; shift ;;
    --run-test) RUN_TEST=1; TO_STEP=eval_test; shift ;;
    --from-step) FROM_STEP="$2"; shift 2 ;;
    --to-step) TO_STEP="$2"; shift 2 ;;
    --stable-ckpt) STABLE_CKPT="$2"; shift 2 ;;
    --valid-oracle) VALID_ORACLE="$2"; shift 2 ;;
    --manifest) MANIFEST="$2"; shift 2 ;;
    --merged-oracle) MERGED_ORACLE="$2"; shift 2 ;;
    --ctrl-out-dir) CTRL_OUT_DIR="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown arg: $1 (see --help)" ;;
  esac
done

mkdir -p "$LOG_DIR" results "$CTRL_OUT_DIR"

if [[ "$CONFIRM_FULL_RUN" -ne 1 ]]; then
  cat <<EOF
Full Plan A training is disabled.
Pass --confirm-full-run manually, e.g.:

  bash scripts/run_plan_a_full.sh --confirm-full-run --gpu ${GPU}

EOF
  exit 1
fi

[[ -f "$INDEX" ]] || die "missing $INDEX"
[[ -f "$VALID_ORACLE" ]] || die "missing valid oracle: $VALID_ORACLE"
[[ -f "$STABLE_CKPT" ]] || die "missing stable supernet ckpt: $STABLE_CKPT"

# Refuse to start if a previous crashed run left a bak file without restoring
if [[ -f "$INDEX_BAK" ]]; then
  die "Found leftover $INDEX_BAK — restore/move it manually before re-running (official index may be dirty)."
fi

# Verify current index looks like full TRAIN
python - "$INDEX" <<'PY'
from basicts.utils import load_pkl
import sys
idx = load_pkl(sys.argv[1])
n = len(idx["train"])
print("official_train_len", n)
if n < 10000:
    raise SystemExit(f"refusing: train split looks filtered (n={n}); restore full index first")
PY

log "========== Plan A Full START =========="
log "ROOT=$ROOT gpu=$GPU seed=$SEED blocks=$NUM_BLOCKS"
log "from=$FROM_STEP to=$TO_STEP skip_completed=$SKIP_COMPLETED run_test=$RUN_TEST"
log "master_log=$MASTER_LOG"

# -------------------- A5 prepare --------------------
if should_run prepare; then
  log "[A5] prepare temporal crossfit manifest"
  python scripts/prepare_temporal_crossfit.py \
    --dataset "$DATASET" \
    --horizon "$HORIZON" \
    --num-blocks "$NUM_BLOCKS" \
    --out "$MANIFEST" \
    2>&1 | tee -a "$MASTER_LOG"
  [[ -f "$MANIFEST" ]] || die "manifest not written"
else
  [[ -f "$MANIFEST" ]] || die "--from-step skips prepare but $MANIFEST missing"
  log "[A5] skip prepare (using existing manifest)"
fi

FOLDS="$(python - "$MANIFEST" <<'PY'
import json,sys
m=json.load(open(sys.argv[1]))
print(" ".join(str(f["fold"]) for f in m["folds"]))
PY
)"
log "folds: $FOLDS"

# -------------------- A6+A7 per fold --------------------
if should_run folds; then
  for FOLD in $FOLDS; do
    ORACLE_OUT="results/pems04_cf_fold${FOLD}_oracle.json"
    VIEW="results/temporal_cf_fold${FOLD}_oracle_view.json"
    TAG="temporal_cf_fold${FOLD}_teacher"
    FOLD_LOG="${LOG_DIR}/fold${FOLD}_${STAMP}.log"

    if [[ "$SKIP_COMPLETED" -eq 1 && -f "$ORACLE_OUT" ]]; then
      if oracle_healthy "$ORACLE_OUT" >>"$MASTER_LOG" 2>&1; then
        log "[fold $FOLD] skip — healthy oracle exists: $ORACLE_OUT"
        continue
      else
        log "[fold $FOLD] existing oracle failed health check — will rebuild"
      fi
    fi

    log "[A6] fold=$FOLD write teacher index + train supernet"
    python scripts/write_temporal_subset_index.py \
      --manifest "$MANIFEST" \
      --which fold_teacher --fold "$FOLD" \
      --out "datasets/PEMS04/index_in12_out12_cf_fold${FOLD}_teacher.pkl" \
      2>&1 | tee -a "$MASTER_LOG"

    # Swap index under lock (trap restores on failure)
    cp -f "$INDEX" "$INDEX_BAK"
    cp -f "datasets/PEMS04/index_in12_out12_cf_fold${FOLD}_teacher.pkl" "$INDEX"
    python - "$INDEX" <<'PY'
from basicts.utils import load_pkl
import sys
print("active_train_len", len(load_pkl(sys.argv[1])["train"]))
PY

    log "[A6] training teacher fold=$FOLD tag=$TAG (this is the long step)"
    python scripts/run_budget_conditioned_f2f.py \
      --dataset "$DATASET" \
      --horizons "$HORIZON" \
      --seeds "$SEED" \
      --gpus "$GPU" \
      --training-phase supernet \
      --loss-mode dynamic_fair \
      --route-sampling sandwich \
      --run-tag "$TAG" \
      2>&1 | tee -a "$FOLD_LOG" | tee -a "$MASTER_LOG"

    # Restore index IMMEDIATELY after training
    restore_index
    python - "$INDEX" <<'PY'
from basicts.utils import load_pkl
import sys
n=len(load_pkl(sys.argv[1])["train"])
print("restored_train_len", n)
assert n >= 10000, n
PY

    CKPT="$(find_fold_ckpt "$FOLD" || true)"
    [[ -n "$CKPT" ]] || die "fold $FOLD: cannot find best_val_MAE.pt"
    CFG="$(find_fold_cfg "$CKPT" || true)"
    [[ -n "$CFG" ]] || die "fold $FOLD: cannot find generated cfg next to ckpt=$CKPT"
    log "[A6] fold=$FOLD ckpt=$CKPT"
    log "[A6] fold=$FOLD cfg=$CFG"

    log "[A7] fold=$FOLD build oracle view + raw-scale oracle"
    python - "$MANIFEST" "$FOLD" "$VIEW" <<'PY'
import json, sys
from pathlib import Path
manifest_path, fold_s, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
fold_i = int(fold_s)
m = json.loads(Path(manifest_path).read_text())
fold = next(f for f in m["folds"] if int(f["fold"]) == fold_i)
out = {
    "dataset": m.get("dataset", "PEMS04"),
    "horizon": int(m.get("horizon", 12)),
    "manifest_hash": f"cf_fold{fold_i}",
    "oracle_holdout_samples": list(fold["oracle_indices"]),
    "supernet_train_samples": list(fold["teacher_train_indices"]),
    "purged_samples": list(fold["purge_indices"]),
    "teacher_fold": fold_i,
    "split_type": "temporal_crossfit_oracle_block",
}
Path(out_path).write_text(json.dumps(out, indent=2) + "\n")
print("oracle_view n=", len(out["oracle_holdout_samples"]))
PY

    python scripts/build_temporal_holdout_route_oracle.py \
      --manifest "$VIEW" \
      --cfg "$CFG" \
      --checkpoint "$CKPT" \
      --device "cuda:${GPU}" \
      --out "$ORACLE_OUT" \
      2>&1 | tee -a "$FOLD_LOG" | tee -a "$MASTER_LOG"

    # Annotate metadata with fold id for merge
    python - "$ORACLE_OUT" "$FOLD" "$CKPT" <<'PY'
import json, hashlib, sys
from pathlib import Path
path, fold, ckpt = sys.argv[1], int(sys.argv[2]), sys.argv[3]
d = json.loads(Path(path).read_text())
md = d.setdefault("metadata", {})
md["teacher_fold"] = fold
md["fold"] = fold
md["teacher_checkpoint"] = ckpt
md["teacher_hash"] = hashlib.sha1(Path(ckpt).read_bytes()).hexdigest()[:16]
md["split_type"] = "temporal_crossfit_oracle_block"
Path(path).write_text(json.dumps(d, indent=2) + "\n")
print("annotated fold", fold)
PY

    oracle_healthy "$ORACLE_OUT" 2>&1 | tee -a "$MASTER_LOG" \
      || die "fold $FOLD oracle failed health check — aborting before merge"
    log "[fold $FOLD] DONE oracle=$ORACLE_OUT"
  done
else
  log "[A6/A7] skip folds"
fi

# Ensure index is clean before later stages
restore_index

# -------------------- A8 merge --------------------
if should_run merge; then
  log "[A8] merge temporal crossfit oracles"
  ORACLE_ARGS=()
  for FOLD in $FOLDS; do
    f="results/pems04_cf_fold${FOLD}_oracle.json"
    [[ -f "$f" ]] || die "missing $f — run folds first"
    oracle_healthy "$f" 2>&1 | tee -a "$MASTER_LOG" || die "unhealthy $f"
    ORACLE_ARGS+=("$f")
  done
  python scripts/merge_temporal_crossfit_oracles.py \
    --fold-oracles "${ORACLE_ARGS[@]}" \
    --manifest "$MANIFEST" \
    --out "$MERGED_ORACLE" \
    2>&1 | tee -a "$MASTER_LOG"
  [[ -f "$MERGED_ORACLE" ]] || die "merged oracle missing"
else
  [[ -f "$MERGED_ORACLE" ]] || die "merged oracle missing for later steps"
  log "[A8] skip merge"
fi

# -------------------- A9 controller --------------------
if should_run controller; then
  log "[A9] train crossfit refinement controller (frozen backbone)"
  python scripts/train_crossfit_refinement_controller.py \
    --crossfit-oracle "$MERGED_ORACLE" \
    --valid-oracle "$VALID_ORACLE" \
    --supernet-checkpoint "$STABLE_CKPT" \
    --confirm-full-run \
    --device "cuda:${GPU}" \
    --out-dir "$CTRL_OUT_DIR" \
    2>&1 | tee -a "$MASTER_LOG"
else
  log "[A9] skip controller"
fi

CTRL_CKPT="${CTRL_OUT_DIR}/refinement_controller_best_val_regret.pt"

# -------------------- A10 valid eval --------------------
if should_run eval_valid; then
  [[ -f "$CTRL_CKPT" ]] || die "missing controller ckpt: $CTRL_CKPT"
  log "[A10] VALID evaluation"
  python scripts/eval_forecast_refinement_controller.py \
    --controller-checkpoint "$CTRL_CKPT" \
    --supernet-checkpoint "$STABLE_CKPT" \
    --split valid \
    --valid-oracle "$VALID_ORACLE" \
    --device "cuda:${GPU}" \
    --out "results/pems04_crossfit_controller_eval_valid.json" \
    2>&1 | tee -a "$MASTER_LOG"
else
  log "[A10] skip valid eval"
fi

# -------------------- A11 test eval (optional) --------------------
if should_run eval_test; then
  if [[ "$RUN_TEST" -ne 1 && "$TO_STEP" != "eval_test" ]]; then
    log "[A11] skip test (pass --run-test to enable)"
  else
    [[ -f "$CTRL_CKPT" ]] || die "missing controller ckpt: $CTRL_CKPT"
    log "[A11] TEST evaluation (final only; do not tune on this)"
    python scripts/eval_forecast_refinement_controller.py \
      --controller-checkpoint "$CTRL_CKPT" \
      --supernet-checkpoint "$STABLE_CKPT" \
      --split test \
      --device "cuda:${GPU}" \
      --out "results/pems04_crossfit_controller_eval_test.json" \
      2>&1 | tee -a "$MASTER_LOG"
  fi
fi

log "========== Plan A Full FINISHED =========="
log "manifest:      $MANIFEST"
log "merged oracle: $MERGED_ORACLE"
log "controller:    $CTRL_CKPT"
log "valid eval:    results/pems04_crossfit_controller_eval_valid.json"
log "master log:    $MASTER_LOG"
log "Official index restored: train should be full PEMS04 TRAIN."
