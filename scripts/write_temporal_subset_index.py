#!/usr/bin/env python3
"""Write a filtered index_*.pkl whose TRAIN split is holdout supernet samples only.

Used so ``run_budget_conditioned_f2f.py`` can train the stable architecture on the
temporal holdout-supernet subset without changing forecasting code.
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basicts.utils import dump_pkl, load_pkl
from basicts.archs.arch_zoo.ChainForecasting_arch.temporal_crossfit_refinement import (
    load_manifest,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument(
        "--index-file",
        default="datasets/PEMS04/index_in12_out12.pkl",
    )
    p.add_argument(
        "--out",
        default="datasets/PEMS04/index_in12_out12_temporal_holdout_supernet.pkl",
    )
    p.add_argument(
        "--which",
        choices=["supernet_train", "oracle_holdout", "fold_teacher", "fold_oracle"],
        default="supernet_train",
    )
    p.add_argument("--fold", type=int, default=None)
    args = p.parse_args()

    manifest = load_manifest(args.manifest)
    raw = load_pkl(args.index_file)
    out_index = copy.deepcopy(raw)
    train = list(raw["train"])

    if args.which == "supernet_train":
        idxs = list(manifest["supernet_train_samples"])
    elif args.which == "oracle_holdout":
        idxs = list(manifest["oracle_holdout_samples"])
    elif args.which == "fold_teacher":
        if args.fold is None:
            raise RuntimeError("--fold required for fold_teacher")
        fold = next(f for f in manifest["folds"] if int(f["fold"]) == int(args.fold))
        idxs = list(fold["teacher_train_indices"])
    elif args.which == "fold_oracle":
        if args.fold is None:
            raise RuntimeError("--fold required for fold_oracle")
        fold = next(f for f in manifest["folds"] if int(f["fold"]) == int(args.fold))
        idxs = list(fold["oracle_indices"])
    else:
        raise ValueError(args.which)

    out_index["train"] = [train[i] for i in idxs]
    # Keep valid/test untouched for BasicTS compatibility, but training only sees subset.
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    dump_pkl(out_index, str(out))
    print(
        f"wrote {out} train={len(out_index['train'])} "
        f"(from {len(train)} via {args.which})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
