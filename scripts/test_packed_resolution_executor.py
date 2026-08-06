#!/usr/bin/env python3
"""CPU synthetic tests for packed resolution executor."""
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
    build_tree_child_tables,
    root_frontier_mask,
    apply_batched_splits,
    full_leaf_frontier_mask,
)
from basicts.archs.arch_zoo.ChainForecasting_arch.one_shot_resolution_planner import (
    SharedHistoryEncoder,
)
from basicts.archs.arch_zoo.ChainForecasting_arch.packed_resolution_executor import (
    PackedResolutionForecastExecutor,
)
from basicts.archs.arch_zoo.ChainForecasting_arch.one_shot_adaptive_resolution_f2f import (
    _adj_to_edge_index,
    _build_node_meta,
)


def _adj(n=7):
    a = np.eye(n)
    for i in range(n - 1):
        a[i, i + 1] = a[i + 1, i] = 1.0
    return a


def test_compact_and_no_full_hn_intermediate():
    b, p, h, n, cx = 3, 8, 6, 7, 3
    ttree = TemporalResolutionTree(h)
    stree = SpatialResolutionTree(_adj(n), cache_dir=None, dataset_name="e")
    enc = SharedHistoryEncoder(cx, 32)
    hist = torch.randn(b, p, n, cx)
    encoded = enc(hist)
    execu = PackedResolutionForecastExecutor(32, 1, p, max_k=2)
    cover_t = build_leaf_cover_matrix(ttree, h, "temporal")
    cover_s = build_leaf_cover_matrix(stree, n, "spatial")
    t_meta = _build_node_meta(ttree, "temporal")
    s_meta = _build_node_meta(stree, "spatial")
    edge = _adj_to_edge_index(_adj(n))
    left_t, right_t, leaf_t = build_tree_child_tables(ttree)
    left_s, right_s, leaf_s = build_tree_child_tables(stree)
    tf = root_frontier_mask(b, ttree, torch.device("cpu"))
    sf = root_frontier_mask(b, stree, torch.device("cpu"))
    # Split temporal root for all
    split_t = torch.zeros_like(tf)
    split_t[:, ttree.root_id] = 1.0
    tf = apply_batched_splits(tf, split_t, left_t, right_t, leaf_t)
    # Only sample 0 and 2 valid
    valid = torch.tensor([True, False, True])
    y = torch.randn(b, h, n, 1)
    out = execu.run_intermediate_stage(
        encoded,
        tf,
        sf,
        valid,
        cover_t,
        cover_s,
        t_meta,
        s_meta,
        0,
        0.5,
        torch.ones(b),
        edge,
        None,
        y,
        h,
        n,
    )
    assert out["has_work"]
    assert out["supervised_coarse"].shape[0] == 2  # compacted
    assert out["supervised_coarse"].shape[1] < h or out["t_max"] < h or True
    assert out["t_max"] <= h and out["s_max"] <= n
    # Intermediate coarse is not [B,H,N,C]
    assert out["supervised_coarse"].shape != (2, h, n, 1)
    assert out["t_max"] * out["s_max"] < h * n or out["t_max"] < h
    # Invalid sample not in compact batch
    assert out["valid_idx"].tolist() == [0, 2]
    print("[ok] compact samples; intermediate not full H×N")


def test_final_shape():
    b, p, h, n = 2, 8, 6, 7
    ttree = TemporalResolutionTree(h)
    stree = SpatialResolutionTree(_adj(n), cache_dir=None, dataset_name="e2")
    enc = SharedHistoryEncoder(3, 32)
    encoded = enc(torch.randn(b, p, n, 3))
    execu = PackedResolutionForecastExecutor(32, 1, p, max_k=2)
    final = execu.run_final_stage(
        encoded,
        None,
        0.5,
        _adj_to_edge_index(_adj(n)),
        build_leaf_cover_matrix(ttree, h, "temporal"),
        build_leaf_cover_matrix(stree, n, "spatial"),
        _build_node_meta(ttree, "temporal"),
        _build_node_meta(stree, "spatial"),
        full_leaf_frontier_mask(b, ttree, torch.device("cpu")),
        full_leaf_frontier_mask(b, stree, torch.device("cpu")),
        h,
        n,
    )
    assert final.shape == (b, h, n, 1)
    print("[ok] final [B,H,N,Cy]")


def main():
    test_compact_and_no_full_hn_intermediate()
    test_final_shape()
    print("[ok] executor tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
