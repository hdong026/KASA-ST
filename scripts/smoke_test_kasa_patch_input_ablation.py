#!/usr/bin/env python3
"""Smoke test for KASA patch data input ablation."""
from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from basicts.archs import KASA_v2


def base_model_args(**overrides):
    args = {
        "node_size": 307,
        "input_len": 12,
        "output_len": 12,
        "input_dim": 4,
        "patch_len": 3,
        "stride": 4,
        "td_size": 288,
        "dw_size": 7,
        "d_td": 32,
        "d_dw": 32,
        "d_d": 32,
        "d_spa": 32,
        "if_time_in_day": True,
        "if_day_in_week": True,
        "if_spatial": True,
        "num_layer": 2,
        "spatial_scheme": "C",
        "adj_mx_path": None,
        "use_gcn": False,
        "use_dynamic_spatial": False,
        "use_adaptive_adj": True,
        "adp_hidden_dim": 32,
        "adp_topk": 20,
        "adp_tau": 0.5,
        "use_hybrid_graph": False,
        "hybrid_alpha": 0.2,
        "post_spatial_mode": "adaptive_only",
        "use_pre_temporal_spatial_enhancement": False,
        "keep_output_prior_residual": False,
        "use_input_prior_enhancement": False,
        "use_patch_branch": True,
        "use_downsample_branch": True,
        "use_linear_residual_branch": True,
        "use_extra_prior_input": False,
        "main_input_dim": 3,
        "use_graph_spectral_calibration": False,
        "patch_data_input_mode": "all",
    }
    args.update(overrides)
    return args


def run_forward(name: str, overrides: dict, channels: int) -> None:
    model = KASA_v2(**base_model_args(**overrides))
    history = torch.randn(2, 12, 307, channels)
    future = torch.randn(2, 12, 307, channels)
    out = model(history, future, batch_seen=1, epoch=0, train=False)
    assert out.shape == (2, 12, 307, 1), f"{name}: expected (2,12,307,1), got {out.shape}"

    patch_mode = overrides.get("patch_data_input_mode", "all")
    if patch_mode == "flow_only":
        assert model.patch_encoder.data_input_dim == 1
        assert model.patch_encoder.patch_data_input_mode == "flow_only"
    else:
        expected_dim = overrides.get("main_input_dim", 3) if overrides.get("use_extra_prior_input") else 3
        assert model.patch_encoder.data_input_dim == expected_dim
    print(f"[ok] {name}: output shape {tuple(out.shape)}, patch_data_input_mode={patch_mode}")


def main() -> int:
    run_forward("all_mode", {"patch_data_input_mode": "all"}, channels=4)
    run_forward("flow_only_mode", {"patch_data_input_mode": "flow_only"}, channels=4)
    run_forward(
        "flow_only_extra_prior",
        {
            "input_dim": 5,
            "use_extra_prior_input": True,
            "main_input_dim": 5,
            "patch_data_input_mode": "flow_only",
        },
        channels=5,
    )
    print("All smoke tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
