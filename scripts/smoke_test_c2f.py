#!/usr/bin/env python3
"""Smoke test for C2F coarse-to-fine model."""
from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from basicts.archs import C2F


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
        "patch_embedding_mode": "serial_concat",
        "patch_data_input_mode": "all",
        "c2f_mode": "none",
        "coarse_len": 3,
        "use_coarse_loss": False,
        "use_linear_residual_in_c2f": True,
        "patch_residual_condition": "none",
        "use_direct_patch_in_c2f": True,
    }
    args.update(overrides)
    return args


def run_forward(name: str, overrides: dict, channels: int, coarse_len: int | None = None) -> None:
    model = C2F(**base_model_args(**overrides))
    history = torch.randn(2, 12, 307, channels)
    future = torch.randn(2, 12, 307, channels)
    out = model(history, future, batch_seen=1, epoch=0, train=False)
    assert out.shape == (2, 12, 307, 1), f"{name}: expected (2,12,307,1), got {out.shape}"

    mode = overrides.get("c2f_mode", "none")
    if mode == "coarse_residual":
        expected_coarse = coarse_len if coarse_len is not None else overrides.get("coarse_len", 3)
        coarse_pred = model.get_latest_coarse_pred()
        assert coarse_pred is not None, f"{name}: latest_coarse_pred is None"
        assert coarse_pred.shape == (2, expected_coarse, 307, 1), (
            f"{name}: expected coarse (2,{expected_coarse},307,1), got {coarse_pred.shape}"
        )
        print(
            f"[ok] {name}: output {tuple(out.shape)}, "
            f"coarse_pred {tuple(coarse_pred.shape)}, c2f_mode={mode}"
        )
    else:
        assert model.get_latest_coarse_pred() is None
        print(f"[ok] {name}: output {tuple(out.shape)}, c2f_mode={mode}")


def test_build_coarse_target() -> None:
    future = torch.randn(2, 12, 307, 4)
    coarse3 = C2F.build_coarse_target(future, coarse_len=3)
    assert coarse3.shape == (2, 3, 307, 1), f"coarse3 shape {coarse3.shape}"
    coarse6 = C2F.build_coarse_target(future, coarse_len=6)
    assert coarse6.shape == (2, 6, 307, 1), f"coarse6 shape {coarse6.shape}"
    print(f"[ok] build_coarse_target: coarse3 {tuple(coarse3.shape)}, coarse6 {tuple(coarse6.shape)}")


def main() -> int:
    run_forward("none_mode", {"c2f_mode": "none"}, channels=4)
    run_forward(
        "coarse_residual_fc3",
        {"c2f_mode": "coarse_residual", "coarse_len": 3},
        channels=4,
        coarse_len=3,
    )
    run_forward(
        "coarse_residual_fc6",
        {"c2f_mode": "coarse_residual", "coarse_len": 6},
        channels=4,
        coarse_len=6,
    )
    run_forward(
        "coarse_residual_fc3_no_linear_residual",
        {
            "c2f_mode": "coarse_residual",
            "coarse_len": 3,
            "use_linear_residual_in_c2f": False,
        },
        channels=4,
        coarse_len=3,
    )
    run_forward(
        "coarse_residual_extra_prior",
        {
            "input_dim": 5,
            "use_extra_prior_input": True,
            "main_input_dim": 5,
            "c2f_mode": "coarse_residual",
            "coarse_len": 3,
        },
        channels=5,
        coarse_len=3,
    )
    test_build_coarse_target()
    print("All smoke tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
