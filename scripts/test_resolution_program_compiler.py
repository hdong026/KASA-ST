#!/usr/bin/env python3
"""CPU synthetic tests for resolution program compiler."""
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
    build_tree_child_tables,
    frontier_active_counts,
)
from basicts.archs.arch_zoo.ChainForecasting_arch.resolution_program_compiler import (
    ResolutionProgramCompiler,
)


def _adj(n=7):
    a = np.eye(n)
    for i in range(n - 1):
        a[i, i + 1] = a[i + 1, i] = 1.0
    return a


def _compiler():
    t = TemporalResolutionTree(6)
    s = SpatialResolutionTree(_adj(7), cache_dir=None, dataset_name="c")
    lt, rt, leaf_t = build_tree_child_tables(t)
    ls, rs, leaf_s = build_tree_child_tables(s)
    return ResolutionProgramCompiler(t, s, lt, rt, leaf_t, ls, rs, leaf_s, k_steps=2), t, s


def test_programs_0_1_2():
    comp, ttree, stree = _compiler()
    b, kt, ks = 3, len(ttree.nodes), len(stree.nodes)
    # Strong stop → 0 intermediates
    planner = {
        "temporal_split_logits": torch.full((b, 2, kt), -5.0),
        "spatial_split_logits": torch.full((b, 2, ks), -5.0),
        "continue_logits": torch.full((b, 2), -5.0),
        "budget_allocation": torch.ones(b, 2, 2) / 2,
        "expected_cost": torch.zeros(b, 2),
    }
    budget = torch.ones(b)
    prog = comp.compile(planner, budget, 0.5, deterministic=True)
    assert torch.all(prog["intermediate_stage_count"] == 0)
    assert prog["final_temporal_frontier"].sum(-1).tolist() == [6, 6, 6]
    assert prog["final_spatial_frontier"].sum(-1).tolist() == [7, 7, 7]

    # Continue once with temporal splits only
    planner["continue_logits"] = torch.tensor([[5.0, -5.0], [5.0, -5.0], [5.0, -5.0]])
    planner["temporal_split_logits"] = torch.full((b, 2, kt), 5.0)
    planner["spatial_split_logits"] = torch.full((b, 2, ks), -5.0)
    prog = comp.compile(planner, budget, 0.5, deterministic=True)
    assert torch.all(prog["stage_valid"][:, 0])
    assert torch.all(~prog["stage_valid"][:, 1])
    assert torch.all(frontier_active_counts(prog["temporal_frontiers"][:, 0]) > 1)
    assert torch.all(frontier_active_counts(prog["spatial_frontiers"][:, 0]) == 1)

    # Continue twice with joint
    planner["continue_logits"] = torch.full((b, 2), 5.0)
    planner["spatial_split_logits"] = torch.full((b, 2, ks), 5.0)
    prog = comp.compile(planner, budget * 3, 0.9, deterministic=True)
    assert torch.all(prog["intermediate_stage_count"] == 2)
    assert torch.all(frontier_active_counts(prog["temporal_frontiers"][:, 1]) > 1)
    assert torch.all(frontier_active_counts(prog["spatial_frontiers"][:, 1]) > 1)
    print("[ok] compiler 0/1/2 stages; T-only and joint")


def test_nested_legality_counts():
    comp, ttree, stree = _compiler()
    b = 2
    kt, ks = len(ttree.nodes), len(stree.nodes)
    planner = {
        "temporal_split_logits": torch.full((b, 2, kt), 3.0),
        "spatial_split_logits": torch.full((b, 2, ks), -3.0),
        "continue_logits": torch.tensor([[4.0, -4.0], [-4.0, -4.0]]),
        "budget_allocation": torch.ones(b, 2, 2) / 2,
        "expected_cost": torch.zeros(b, 2),
    }
    prog = comp.compile(planner, torch.ones(b), 0.5, deterministic=True)
    # Different samples different stage counts
    assert prog["intermediate_stage_count"][0] == 1
    assert prog["intermediate_stage_count"][1] == 0
    print("[ok] sample-wise different programs")


def main():
    test_programs_0_1_2()
    test_nested_legality_counts()
    print("[ok] compiler tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
