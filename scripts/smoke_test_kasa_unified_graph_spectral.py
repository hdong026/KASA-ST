#!/usr/bin/env python3
"""Smoke test for KASA unified graph spectral prior / calibration."""
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
        "graph_spectral_calibration_mode": "none",
        "graph_spectral_k": 32,
        "graph_spectral_gamma_init": 0.0,
    }
    args.update(overrides)
    return args


CASES = [
    ("base", {"input_dim": 4}),
    ("offline_low", {
        "input_dim": 4,
        "use_extra_prior_input": True,
        "main_input_dim": 4,
    }),
    ("offline_low_high", {
        "input_dim": 5,
        "use_extra_prior_input": True,
        "main_input_dim": 5,
    }),
    ("online_calib", {
        "input_dim": 4,
        "use_graph_spectral_calibration": True,
        "graph_spectral_calibration_mode": "residual_safe_low_high",
    }),
    ("offline_low_online_calib", {
        "input_dim": 4,
        "use_extra_prior_input": True,
        "main_input_dim": 4,
        "use_graph_spectral_calibration": True,
        "graph_spectral_calibration_mode": "residual_safe_low_high",
    }),
]


def run_case(name: str, overrides: dict) -> None:
    channels = overrides.get("input_dim", 4)
    model = KASA_v2(**base_model_args(**overrides))
    history = torch.randn(2, 12, 307, channels)
    future = torch.randn(2, 12, 307, channels)
    out = model(history, future, batch_seen=1, epoch=0, train=False)
    assert out.shape == (2, 12, 307, 1), f"{name}: expected (2,12,307,1), got {out.shape}"
    print(f"[ok] {name}: output shape {tuple(out.shape)}")


def main() -> int:
    for name, overrides in CASES:
        run_case(name, overrides)
    print("All smoke tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
