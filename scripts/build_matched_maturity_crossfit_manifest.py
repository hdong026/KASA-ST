#!/usr/bin/env python3
"""Build matched-maturity crossfit v2 manifest (blocked LOO-block, ~80% train each).

This is OFFLINE deployment-matched supervision — NOT causal online validation.
Does NOT train teachers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from basicts.archs.arch_zoo.ChainForecasting_arch.temporal_crossfit_refinement import (
    compute_min_purge_samples,
    load_split_index,
    sample_raw_span,
    spans_overlap,
)
from basicts.utils import load_pkl


def sha1_file(path: Path, n: int | None = 16) -> str:
    h = hashlib.sha1(path.read_bytes()).hexdigest()
    return h if n is None else h[:n]


def build_matched_maturity_manifest(
    *,
    index_path: str = "datasets/PEMS04/index_in12_out12.pkl",
    data_path: str = "datasets/PEMS04/data_in12_out12.pkl",
    K: int = 5,
    seed: int = 1,
    stable_cfg: str | None = None,
) -> dict:
    index = load_split_index(index_path, "train")
    n = len(index)
    purge = compute_min_purge_samples(index)
    # span check
    s0 = sample_raw_span(index[0])
    span_len = s0[1] - s0[0]
    P = index[0][1] - index[0][0]
    H = index[0][2] - index[0][1]

    # K contiguous blocks over TRAIN indices
    edges = [int(round(i * n / K)) for i in range(K + 1)]
    folds = []
    for k in range(K):
        hold_lo, hold_hi = edges[k], edges[k + 1]
        holdout = list(range(hold_lo, hold_hi))
        # teacher candidates = all other blocks
        train_cand = [i for i in range(n) if not (hold_lo <= i < hold_hi)]
        # purge: remove any train sample whose raw span overlaps any holdout span
        hold_spans = [sample_raw_span(index[i]) for i in holdout]
        # efficient: holdout raw range union
        h0 = min(s[0] for s in hold_spans)
        h1 = max(s[1] for s in hold_spans)
        hold_union = (h0, h1)

        kept = []
        purged = []
        for i in train_cand:
            sp = sample_raw_span(index[i])
            if spans_overlap(sp, hold_union):
                purged.append(i)
            else:
                kept.append(i)

        # verify no residual overlap
        overlap = 0
        for i in kept:
            sp = sample_raw_span(index[i])
            for hs in hold_spans:
                if spans_overlap(sp, hs):
                    overlap += 1
                    break

        folds.append(
            {
                "fold": k + 1,
                "heldout_sample_indices": holdout,
                "heldout_raw_time_range": [h0, h1],
                "n_holdout": len(holdout),
                "n_train_before_purge": len(train_cand),
                "n_purged": len(purged),
                "n_train_after_purge": len(kept),
                "teacher_train_indices": kept,
                "purge_indices": purged,
                "raw_window_overlap_after_purge": overlap,
                "initialization_seed": int(seed),
            }
        )

    if stable_cfg is None:
        stable_cfg = (
            "checkpoints/PEMS04/H12/budget_f2f/"
            "supernet_eta0p50_dynamic_fair_rawscale_loss_v2_60f53aa1c6/seed1/"
            "b5678fda5e8d94ed028c6c8bb073461d/"
            "H12_supernet_eta0p50_dynamic_fair_rawscale_loss_v2_60f53aa1c6_seed1.py"
        )
    cfg_hash = sha1_file(Path(stable_cfg), 16) if Path(stable_cfg).is_file() else None

    # expected updates: same as stable (100 epochs, batch 32) — approx
    train_counts = [f["n_train_after_purge"] for f in folds]
    protocol = {
        "architecture": "BudgetConditionedAdaptiveF2FNet",
        "match_stable_protocol": True,
        "stable_cfg": stable_cfg,
        "stable_cfg_hash16": cfg_hash,
        "NUM_EPOCHS": 100,
        "OPTIM": {"lr": 0.002, "weight_decay": 0.0001},
        "LR_SCHEDULER": {"milestones": [1, 35, 60, 80, 95], "gamma": 0.5},
        "BATCH_SIZE": 32,
        "SEED": seed,
        "training_phase": "supernet",
        "route_sampling": "sandwich",
        "loss_mode": "dynamic_fair",
        "inference_intensity": 0.5,
        "note": "Each teacher uses SAME protocol as stable supernet except heldout/purged samples.",
    }

    manifest = {
        "scheme": "matched_maturity_crossfit_v2_blocked_loo",
        "distinction": (
            "BLOCKED CROSS-FITTING for offline deployment-matched supervision. "
            "NOT a causal online forecasting validation protocol."
        ),
        "dataset": "PEMS04",
        "horizon": H,
        "input_len_P": P,
        "span_len_P_plus_H": span_len,
        "purge_samples_derived": purge,
        "purge_formula": "ceil(span_len / start_step) from index; not hard-coded 23",
        "K": K,
        "n_train_total": n,
        "index_path": index_path,
        "data_path": data_path,
        "folds": folds,
        "teacher_train_count_range_after_purge": [min(train_counts), max(train_counts)],
        "any_raw_window_overlap": any(f["raw_window_overlap_after_purge"] > 0 for f in folds),
        "training_protocol": protocol,
        "teacher_checkpoints": None,
        "note": "No teacher checkpoints created by this manifest builder.",
    }
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/matched_maturity_crossfit_manifest.json")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--K", type=int, default=5)
    args = ap.parse_args()
    man = build_matched_maturity_manifest(K=args.K, seed=args.seed)
    Path(args.out).write_text(json.dumps(man, indent=2))
    print(
        json.dumps(
            {
                "out": args.out,
                "K": man["K"],
                "purge": man["purge_samples_derived"],
                "train_count_range": man["teacher_train_count_range_after_purge"],
                "overlap": man["any_raw_window_overlap"],
                "per_fold_train": [f["n_train_after_purge"] for f in man["folds"]],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
