#!/usr/bin/env python3
"""CPU synthetic tests for automatic temporal/spatial resolution trees."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basicts.archs.arch_zoo.ChainForecasting_arch.adaptive_resolution_hierarchy import (
    SpatialResolutionTree,
    TemporalResolutionTree,
    build_frontier_projections,
    build_membership_matrix,
    lift_resolution_to_full,
    pool_full_to_resolution,
    validate_projection_row_sums,
)
from basicts.archs.arch_zoo.ChainForecasting_arch.adaptive_resolution_state import (
    AdaptiveResolutionState,
)


def _synth_adj(n: int) -> np.ndarray:
    adj = np.eye(n, dtype=np.float64)
    for i in range(n - 1):
        adj[i, i + 1] = adj[i + 1, i] = 1.0
    return adj


def test_temporal_tree_from_h() -> None:
    for h in (1, 2, 3, 6, 7, 12):
        tree = TemporalResolutionTree(h)
        assert tree.num_leaves == h
        assert tree.nodes[tree.root_id].start == 0
        assert tree.nodes[tree.root_id].end == h
        assert tree.nodes[tree.root_id].leaf_count == h
        # Odd splits differ by at most 1
        for n in tree.nodes:
            if n.is_leaf:
                continue
            l = tree.nodes[n.left_child_id].leaf_count
            r = tree.nodes[n.right_child_id].leaf_count
            assert abs(l - r) <= 1
    print("[ok] temporal tree auto-built from H")


def test_spatial_tree_from_adj() -> None:
    n = 7
    tree = SpatialResolutionTree(
        _synth_adj(n),
        clustering_seed=0,
        cache_dir=None,
        dataset_name="synth_test",
    )
    assert tree.num_leaves == n
    covered = []
    for node in tree.nodes:
        if node.is_leaf:
            covered.extend(node.original_node_indices)
    assert sorted(covered) == list(range(n))
    # No overlap among leaves
    assert len(covered) == len(set(covered))
    summary = tree.summary()
    assert summary["coverage_ok"] and summary["overlap_ok"]
    print("[ok] spatial tree auto-built from adjacency N=7", summary)


def test_root_counts_and_splits() -> None:
    h, n = 6, 7
    ttree = TemporalResolutionTree(h)
    stree = SpatialResolutionTree(_synth_adj(n), clustering_seed=0, cache_dir=None)
    state = AdaptiveResolutionState.root_init(
        2, ttree, stree, total_budget=1.0, device=torch.device("cpu")
    )
    state.validate_frontier()
    assert torch.all(state.count_active_temporal_units() == 1)
    assert torch.all(state.count_active_spatial_units() == 1)

    # Force temporal root split
    t_split = torch.zeros_like(state.temporal_frontier_mask)
    t_split[:, ttree.root_id] = 1.0
    changed = state.split_temporal_nodes(t_split)
    assert bool(changed.all())
    assert torch.all(state.count_active_temporal_units() == 2)
    assert torch.all(state.count_active_spatial_units() == 1)
    state.validate_frontier()

    # Force spatial root split
    s_split = torch.zeros_like(state.spatial_frontier_mask)
    s_split[:, stree.root_id] = 1.0
    changed_s = state.split_spatial_nodes(s_split)
    assert bool(changed_s.all())
    assert torch.all(state.count_active_spatial_units() == 2)
    assert torch.all(state.count_active_temporal_units() == 2)
    state.validate_frontier()

    # Joint further split both children of temporal? split one temporal + one spatial
    t_ids = state.active_temporal_ids(0)
    s_ids = state.active_spatial_ids(0)
    t_mask = torch.zeros_like(state.temporal_frontier_mask)
    s_mask = torch.zeros_like(state.spatial_frontier_mask)
    # split first non-leaf if any
    for nid in t_ids:
        if not ttree.nodes[nid].is_leaf:
            t_mask[:, nid] = 1.0
            break
    for nid in s_ids:
        if not stree.nodes[nid].is_leaf:
            s_mask[:, nid] = 1.0
            break
    before_t = state.count_active_temporal_units().clone()
    before_s = state.count_active_spatial_units().clone()
    out = state.apply_joint_split(t_mask, s_mask)
    if t_mask.any():
        assert torch.all(state.count_active_temporal_units() > before_t)
    if s_mask.any():
        assert torch.all(state.count_active_spatial_units() > before_s)
    assert bool(out["any"].any())
    state.validate_frontier()

    # Halt freeze
    state.halted[:] = True
    t2 = torch.zeros_like(state.temporal_frontier_mask)
    t2[:, state.active_temporal_ids(0)[0]] = 1.0
    frozen_t = state.count_active_temporal_units().clone()
    state.split_temporal_nodes(t2)
    assert torch.equal(state.count_active_temporal_units(), frozen_t)
    print("[ok] root counts, temporal/spatial/joint splits, halt freeze")


def test_pool_lift_identity_and_constant() -> None:
    h, n, c = 6, 7, 1
    ttree = TemporalResolutionTree(h)
    stree = SpatialResolutionTree(_synth_adj(n), clustering_seed=0, cache_dir=None)
    t_mem = build_membership_matrix(ttree.nodes, h, "temporal")
    s_mem = build_membership_matrix(stree.nodes, n, "spatial")

    # Full leaf frontier
    state = AdaptiveResolutionState.root_init(
        1, ttree, stree, 1.0, torch.device("cpu")
    )
    # Expand to all leaves
    state.temporal_frontier_mask.zero_()
    state.spatial_frontier_mask.zero_()
    for node in ttree.nodes:
        if node.is_leaf:
            state.temporal_frontier_mask[:, node.node_id] = 1.0
    for node in stree.nodes:
        if node.is_leaf:
            state.spatial_frontier_mask[:, node.node_id] = 1.0
    p_t, l_t, tm = build_frontier_projections(t_mem, state.temporal_frontier_mask, h)
    p_s, l_s, sm = build_frontier_projections(s_mem, state.spatial_frontier_mask, n)
    assert validate_projection_row_sums(p_t, tm)
    assert validate_projection_row_sums(p_s, sm)

    full = torch.randn(1, h, n, c)
    coarse = pool_full_to_resolution(full, p_t, p_s)
    recon = lift_resolution_to_full(coarse, l_t, l_s)
    assert torch.allclose(recon, full, atol=1e-5)

    # Constant signal root pool/lift
    state2 = AdaptiveResolutionState.root_init(
        1, ttree, stree, 1.0, torch.device("cpu")
    )
    p_t, l_t, tm = build_frontier_projections(t_mem, state2.temporal_frontier_mask, h)
    p_s, l_s, sm = build_frontier_projections(s_mem, state2.spatial_frontier_mask, n)
    assert int(tm.sum()) == 1 and int(sm.sum()) == 1
    const = torch.ones(1, h, n, c) * 3.14
    coarse = pool_full_to_resolution(const, p_t, p_s)
    recon = lift_resolution_to_full(coarse, l_t, l_s)
    assert torch.allclose(recon, const, atol=1e-5)
    print("[ok] pool/lift identity on leaves; constant preserved; root active=1")


def main() -> int:
    test_temporal_tree_from_h()
    test_spatial_tree_from_adj()
    test_root_counts_and_splits()
    test_pool_lift_identity_and_constant()
    print("[ok] all hierarchy/state tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
