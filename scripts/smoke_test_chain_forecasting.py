#!/usr/bin/env python3
"""Smoke test for ChainForecasting model."""
from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from basicts.archs import ChainForecasting


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
        "spatial_placement": "final",
        "use_pre_temporal_spatial_enhancement": False,
        "use_patch_branch": True,
        "use_downsample_branch": True,
        "use_linear_residual_branch": True,
        "patch_embedding_mode": "serial_concat",
        "patch_data_input_mode": "all",
        "chain_lengths": [3, 6, 12],
        "chain_loss_weights": [0.2, 0.3, 1.0],
        "use_prev_condition": True,
    }
    args.update(overrides)
    return args


def test_forward(name: str, overrides: dict | None = None) -> None:
    overrides = overrides or {}
    model = ChainForecasting(**base_model_args(**overrides))
    history = torch.randn(2, 12, 307, 4)

    out_final = model(history, return_all=False)
    assert out_final.shape == (2, 12, 307, 1), f"{name}: final shape {out_final.shape}"

    out_all = model(history, return_all=True)
    assert isinstance(out_all, dict)
    assert out_all["pred"].shape == (2, 12, 307, 1)
    preds = out_all["chain_preds"]
    chain_lengths = overrides.get("chain_lengths", [3, 6, 12])
    for pred, length in zip(preds, chain_lengths):
        assert pred.shape == (2, length, 307, 1), (
            f"{name}: chain pred shape {pred.shape}, expected length {length}"
        )
    print(f"[ok] {name}: final {tuple(out_final.shape)}, chain {[tuple(p.shape) for p in preds]}")


def test_pool_target() -> None:
    future = torch.randn(2, 12, 307, 1)
    t3 = ChainForecasting.pool_target(future, 3)
    t6 = ChainForecasting.pool_target(future, 6)
    assert t3.shape == (2, 3, 307, 1), f"pool 3 shape {t3.shape}"
    assert t6.shape == (2, 6, 307, 1), f"pool 6 shape {t6.shape}"
    print(f"[ok] pool_target: 12->3 {tuple(t3.shape)}, 12->6 {tuple(t6.shape)}")


def main() -> int:
    test_forward("chain_3_6_12")
    test_forward("chain_6_12", {"chain_lengths": [6, 12], "chain_loss_weights": [0.3, 1.0]})
    test_forward("chain_3_12", {"chain_lengths": [3, 12], "chain_loss_weights": [0.2, 1.0]})
    test_forward(
        "no_prev_condition",
        {"use_prev_condition": False},
    )
    test_forward("spatial_final", {"spatial_placement": "final"})
    test_forward("spatial_each_level", {"spatial_placement": "each_level"})
    test_forward("spatial_none", {"spatial_placement": "none"})
    test_pool_target()
    print("All smoke tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
