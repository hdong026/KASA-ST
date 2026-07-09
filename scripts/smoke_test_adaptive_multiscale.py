#!/usr/bin/env python3
"""Smoke test for adaptive_multiscale_only spatial refine."""
from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from basicts.archs import ChainForecasting
from basicts.archs.arch_zoo.ChainForecasting_arch.gcn import ABCDSpatialModule


def base_args(**overrides):
    args = {
        "node_size": 64,
        "input_len": 12,
        "output_len": 12,
        "input_dim": 4,
        "main_input_dim": 3,
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
        "adp_topk": 32,
        "adp_tau": 0.5,
        "use_hybrid_graph": False,
        "hybrid_alpha": 0.10,
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


def test_adaptive_only_unchanged() -> None:
    torch.manual_seed(0)
    model_a = ChainForecasting(**base_args(post_spatial_mode="adaptive_only", adp_topk=32, hybrid_alpha=0.10))
    model_a.eval()
    history = torch.randn(2, 12, 64, 4)
    with torch.no_grad():
        out_a1 = model_a(history, return_all=False)
        out_a2 = model_a(history, return_all=False)
    assert torch.allclose(out_a1, out_a2), "adaptive_only should be deterministic in eval mode"
    assert out_a1.shape == (2, 12, 64, 1)
    print("[ok] adaptive_only forward unchanged")


def test_adaptive_multiscale_forward() -> None:
    model = ChainForecasting(
        **base_args(
            post_spatial_mode="adaptive_multiscale_only",
            adaptive_ms_topks=[8, 16, 32],
            adaptive_ms_alpha=0.10,
            adaptive_ms_init="favor_largest",
        )
    )
    history = torch.randn(2, 12, 64, 4)
    out = model(history, return_all=True)
    assert out["pred"].shape == (2, 12, 64, 1)
    weights = out["adaptive_ms_weights"]
    assert weights is not None
    assert (weights > 0).all()
    assert torch.isclose(weights.sum(), torch.tensor(1.0), atol=1e-5)
    assert out["adaptive_ms_topks"] == [8, 16, 32]
    assert abs(float(out["adaptive_ms_alpha"]) - 0.10) < 1e-6
    print(f"[ok] adaptive_multiscale_only forward, weights={weights.tolist()}")


def test_adaptive_ms_logits_grad() -> None:
    module = ABCDSpatialModule(
        node_size=32,
        input_len=12,
        d_spa=32,
        if_spatial=True,
        use_adaptive_adj=True,
        adp_hidden_dim=16,
        adp_topk=16,
        post_spatial_mode="adaptive_multiscale_only",
        adaptive_ms_topks=[8, 16],
        adaptive_ms_alpha=0.10,
        adaptive_ms_init="uniform",
    )
    output = torch.randn(2, 12, 32, 1, requires_grad=True)
    history_flow = torch.randn(2, 12, 32)
    refined = module.refine_prediction(output, history_flow)
    loss = refined.sum()
    loss.backward()
    assert module.adaptive_ms_logits.grad is not None
    assert torch.isfinite(module.adaptive_ms_logits.grad).all()

    opt = torch.optim.Adam(module.parameters(), lr=0.05)
    before = module.adaptive_ms_logits.detach().clone()
    opt.zero_grad()
    refined = module.refine_prediction(output, history_flow)
    refined.sum().backward()
    opt.step()
    after = module.adaptive_ms_logits.detach()
    assert not torch.allclose(before, after), "adaptive_ms_logits should update after optimizer.step"
    print("[ok] adaptive_ms_logits has gradient and updates")


def test_no_static_dynamic_hybrid() -> None:
    module = ABCDSpatialModule(
        node_size=32,
        input_len=12,
        d_spa=32,
        if_spatial=True,
        use_adaptive_adj=True,
        use_dynamic_spatial=True,
        use_hybrid_graph=True,
        use_gcn=True,
        adj_mx_path=None,
        post_spatial_mode="adaptive_multiscale_only",
        adaptive_ms_topks=[8, 16, 32],
    )
    assert module.adj_mx is None
    assert module.use_dynamic_spatial is False
    assert module.use_hybrid_graph is False
    assert module.use_gcn is False
    print("[ok] adaptive_multiscale_only disables static/dynamic/hybrid graph paths")


def test_single_topk_near_adaptive_only() -> None:
    torch.manual_seed(42)
    shared = dict(
        node_size=48,
        input_len=12,
        d_spa=32,
        if_spatial=True,
        use_adaptive_adj=True,
        adp_hidden_dim=32,
        adp_topk=32,
        adp_tau=0.5,
        hybrid_alpha=0.10,
    )
    mod_only = ABCDSpatialModule(**shared, post_spatial_mode="adaptive_only")
    mod_ms = ABCDSpatialModule(
        **shared,
        post_spatial_mode="adaptive_multiscale_only",
        adaptive_ms_topks=[32],
        adaptive_ms_alpha=0.10,
        adaptive_ms_init="favor_largest",
    )
    with torch.no_grad():
        mod_ms.adaptive_src.copy_(mod_only.adaptive_src)
        mod_ms.adaptive_dst.copy_(mod_only.adaptive_dst)

    output = torch.randn(2, 12, 48, 1)
    history = torch.randn(2, 12, 48)
    out_only = mod_only.refine_prediction(output, history)
    out_ms = mod_ms.refine_prediction(output, history)
    max_diff = (out_only - out_ms).abs().max().item()
    assert max_diff < 1e-4, f"single-scale ms should match adaptive_only, max_diff={max_diff}"
    print(f"[ok] adaptive_ms_topks=[32] approx adaptive_only (max_diff={max_diff:.2e})")


def test_graph_resolution_adaptive_ms() -> None:
    model = ChainForecasting(
        **base_args(
            node_size=64,
            spatial_placement="temporal_first_graph_resolution",
            post_spatial_mode="adaptive_multiscale_only",
            graph_resolution_ratios=[0.25, 0.50, 1.00],
            graph_resolution_alphas=[0.03, 0.06, 0.10],
            graph_resolution_topks=[8, 16, 32],
            graph_resolution_betas=[1.0, 1.0, 1.0],
            graph_resolution_rhos=[0.25, 0.50, 1.00],
            adaptive_ms_topks=[8, 16, 32],
            adaptive_ms_alpha=0.10,
            dataset_name="PEMS04",
            clustering_seed=0,
        )
    )
    history = torch.randn(2, 12, 64, 4)
    out = model(history, return_all=True)
    assert out["pred"].shape == (2, 12, 64, 1)
    assert out.get("adaptive_ms_weights") is not None
    print("[ok] temporal_first_graph_resolution + adaptive_multiscale_only")


def main() -> int:
    test_adaptive_only_unchanged()
    test_adaptive_multiscale_forward()
    test_adaptive_ms_logits_grad()
    test_no_static_dynamic_hybrid()
    test_single_topk_near_adaptive_only()
    test_graph_resolution_adaptive_ms()
    print("\nAll adaptive multiscale smoke tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
