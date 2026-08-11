#!/usr/bin/env python3
"""Prepare rolling-origin temporal cross-fit manifest (Plan A full; not executed by agent)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basicts.archs.arch_zoo.ChainForecasting_arch.temporal_crossfit_refinement import (
    build_rolling_crossfit_manifest,
    load_split_index,
    save_manifest,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="PEMS04")
    p.add_argument("--horizon", type=int, default=12)
    p.add_argument("--num-blocks", type=int, default=5)
    p.add_argument("--index-file", default=None)
    p.add_argument("--out", default="results/temporal_crossfit_manifest.json")
    args = p.parse_args()
    index_file = args.index_file or f"datasets/{args.dataset}/index_in12_out{args.horizon}.pkl"
    index = load_split_index(index_file, "train")
    manifest = build_rolling_crossfit_manifest(
        index, num_blocks=args.num_blocks, dataset=args.dataset, horizon=args.horizon
    )
    manifest["index_file"] = index_file
    save_manifest(manifest, args.out)
    print(
        f"[crossfit] blocks={args.num_blocks} folds={len(manifest['folds'])} "
        f"purge={manifest['purge_samples']}"
    )
    for f in manifest["folds"]:
        print(
            f"  fold{f['fold']}: teacher={f['n_teacher']} purged={f['n_purged']} "
            f"oracle={f['n_oracle']}"
        )
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
