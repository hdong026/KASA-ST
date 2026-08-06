#!/usr/bin/env python3
"""CPU synthetic tests for one-shot resolution hierarchy helpers."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basicts.archs.arch_zoo.ChainForecasting_arch.one_shot_resolution_hierarchy import (
    SpatialResolutionTree,
    TemporalResolutionTree,
    build_leaf_cover_matrix,
    frontier_to_leaf_assignment,
    gather_lift_full,
    root_frontier_mask,
    full_leaf_frontier_mask,
    scatter_pool_full,
    frontier_active_counts,
)


def _adj(n=7):
    a = np.eye(n)
    for i in range(n - 1):
        a[i, i + 1] = a[i + 1, i] = 1.0
    return a


def test_trees():
    for h in (1, 2, 6, 7):
        t = TemporalResolutionTree(h)
        assert t.num_leaves == h
    s = SpatialResolutionTree(_adj(7), cache_dir=None, dataset_name="t")
    assert s.num_leaves == 7
    print("[ok] trees from H / adjacency only")


def test_pool_lift():
    h, n, c = 6, 7, 1
    ttree = TemporalResolutionTree(h)
    stree = SpatialResolutionTree(_adj(n), cache_dir=None, dataset_name="t2")
    cover_t = build_leaf_cover_matrix(ttree, h, "temporal")
    cover_s = build_leaf_cover_matrix(stree, n, "spatial")
    # Full leaf
    tf = full_leaf_frontier_mask(1, ttree, torch.device("cpu"))
    sf = full_leaf_frontier_mask(1, stree, torch.device("cpu"))
    _, slot_t = frontier_to_leaf_assignment(tf, cover_t)
    _, slot_s = frontier_to_leaf_assignment(sf, cover_s)
    full = torch.randn(1, h, n, c)
    coarse = scatter_pool_full(full, slot_t, slot_s, h, n)
    recon = gather_lift_full(coarse, slot_t, slot_s, h, n)
    assert torch.allclose(recon, full, atol=1e-5)
    # Root
    tr = root_frontier_mask(1, ttree, torch.device("cpu"))
    sr = root_frontier_mask(1, stree, torch.device("cpu"))
    assert int(frontier_active_counts(tr)) == 1
    assert int(frontier_active_counts(sr)) == 1
    _, slot_t = frontier_to_leaf_assignment(tr, cover_t)
    _, slot_s = frontier_to_leaf_assignment(sr, cover_s)
    const = torch.ones(1, h, n, c) * 2.5
    coarse = scatter_pool_full(const, slot_t, slot_s, 1, 1)
    recon = gather_lift_full(coarse, slot_t, slot_s, h, n)
    assert torch.allclose(recon, const, atol=1e-5)
    print("[ok] scatter pool/lift identity + constant")


def main():
    test_trees()
    test_pool_lift()
    print("[ok] hierarchy tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
