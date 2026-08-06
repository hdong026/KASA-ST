#!/usr/bin/env python3
"""CPU architecture inspection for OneShotAdaptiveResolutionF2FNet.

Allowed: synthetic tensors, single CPU forward, shape/call-count reports.
Forbidden: PEMS/KnowAir, CUDA, training, optimizer, checkpoint.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basicts.archs.arch_zoo.ChainForecasting_arch.one_shot_adaptive_resolution_f2f import (
    OneShotAdaptiveResolutionF2FNet,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu"])
    args = parser.parse_args()
    assert args.device == "cpu"

    n, p, h, b = 7, 8, 6, 3
    adj = np.eye(n)
    for i in range(n - 1):
        adj[i, i + 1] = adj[i + 1, i] = 1.0
    model = OneShotAdaptiveResolutionF2FNet(
        node_size=n,
        input_len=p,
        output_len=h,
        input_dim=3,
        output_dim=1,
        adj_mx=adj,
        thinking_intensity=0.5,
        planner_hidden_dim=32,
        executor_hidden_dim=32,
        dataset_name="inspect_synth",
        hierarchy_cache_dir=None,
    ).to("cpu").eval()

    x = torch.randn(b, p, n, 3)
    y = torch.randn(b, h, n, 1)
    with torch.no_grad():
        out = model(history_data=x, future_data=y, train=False, return_all=True, return_intermediates=True)

    print("=== OneShotAdaptiveResolution inspection (CPU synthetic) ===")
    print("pred_shape:", tuple(out["pred"].shape))
    print("history_encoder_call_count:", out["history_encoder_call_count"])
    print("planner_call_count:", out["planner_call_count"])
    print("intermediate_stage_count:", out["intermediate_stage_count"].tolist())
    print("optional_budget:", out["target_budget"].tolist())
    print("expected_optional_cost:", out["expected_optional_cost"].tolist())
    print("MAX_OPTIONAL_INTERMEDIATE_STEPS:", out["max_optional_intermediate_steps"])
    if out.get("intermediates"):
        for i, st in enumerate(out["intermediates"]):
            print(
                f"stage{i}: valid={st['stage_valid'].tolist()} "
                f"active_t={st['active_temporal_count'].tolist()} "
                f"active_s={st['active_spatial_count'].tolist()} "
                f"packed_tokens={st['packed_token_count'].tolist()} "
                f"has_work={st['has_work']}"
            )
    print("dense_nxn_flag:", model.executor.last_dense_nxn_created)
    print("intermediate_used_full_hn_flag:", model.executor.last_intermediate_used_full_hn)
    print("teacher_in_state_dict:", any("teacher" in k for k in model.state_dict()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
