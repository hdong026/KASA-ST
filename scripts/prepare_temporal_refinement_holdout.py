#!/usr/bin/env python3
"""Prepare temporal purged 80/20 holdout manifest (Plan A pilot)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basicts.archs.arch_zoo.ChainForecasting_arch.temporal_crossfit_refinement import (
    build_temporal_holdout_manifest,
    load_split_index,
    save_manifest,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="PEMS04")
    p.add_argument("--horizon", type=int, default=12)
    p.add_argument("--train-fraction", type=float, default=0.8)
    p.add_argument("--split-seed", type=int, default=1, help="metadata only; no shuffle")
    p.add_argument("--purge-mode", default="auto")
    p.add_argument(
        "--index-file",
        default=None,
        help="defaults to datasets/{dataset}/index_in12_out{H}.pkl",
    )
    p.add_argument("--out", default="results/temporal_holdout_split_manifest.json")
    args = p.parse_args()

    index_file = args.index_file or f"datasets/{args.dataset}/index_in12_out{args.horizon}.pkl"
    index = load_split_index(index_file, "train")
    manifest = build_temporal_holdout_manifest(
        index,
        train_fraction=args.train_fraction,
        dataset=args.dataset,
        horizon=args.horizon,
        purge_mode=args.purge_mode,
    )
    manifest["split_seed_metadata_only"] = int(args.split_seed)
    manifest["index_file"] = index_file
    save_manifest(manifest, args.out)
    print(
        f"[holdout] n={manifest['original_train_samples']} "
        f"supernet={len(manifest['supernet_train_samples'])} "
        f"purged={len(manifest['purged_samples'])} "
        f"oracle={len(manifest['oracle_holdout_samples'])} "
        f"purge={manifest['purge_samples']} "
        f"overlap={manifest['overlap_audit']}"
    )
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
