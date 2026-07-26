#!/usr/bin/env python3
"""Checks for chain_interleaved_progressive_spatial_light_router."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basicts.archs import ChainForecasting
from scripts.run_chain_forecasting_horizon_pems04 import generate_temp_config, load_cfg, variant_spec

ADJ = str(ROOT / "datasets" / "PEMS04" / "adj_mx.pkl")


def interleaved_args(**overrides):
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
        "adj_mx_path": ADJ,
        "use_gcn": False,
        "use_dynamic_spatial": False,
        "use_adaptive_adj": True,
        "adp_hidden_dim": 32,
        "adp_topk": 20,
        "adp_tau": 0.5,
        "use_hybrid_graph": False,
        "hybrid_alpha": 0.2,
        "post_spatial_mode": "adaptive_only",
        "spatial_placement": "interleaved_progressive",
        "progressive_spatial_ratios": [0.25, 0.5, 1.0],
        "progressive_spatial_topks": [8, 16, 32],
        "progressive_spatial_alphas": [0.03, 0.06, 0.10],
        "use_patch_branch": True,
        "use_downsample_branch": True,
        "use_linear_residual_branch": True,
        "patch_embedding_mode": "serial_concat",
        "patch_data_input_mode": "all",
        "chain_lengths": [3, 6, 12],
        "chain_loss_weights": [0.2, 0.3, 1.0],
        "use_prev_condition": True,
        "use_spectral_stage_router": False,
        "use_light_spectral_router": False,
    }
    args.update(overrides)
    return args


def _copy_shared_weights(src: ChainForecasting, dst: ChainForecasting) -> None:
    src_sd = src.state_dict()
    dst_sd = dst.state_dict()
    shared = {
        k: v
        for k, v in src_sd.items()
        if k in dst_sd and "light_spectral_router" not in k and "spectral_branch_router" not in k
    }
    missing = dst.load_state_dict(shared, strict=False)
    assert all(
        ("light_spectral_router" in k or "spectral_branch_router" in k)
        for k in missing.missing_keys
    ), missing.missing_keys


def test_original_variant() -> None:
    spec = variant_spec("chain_interleaved_progressive_spatial", 12)
    assert spec["spatial_placement"] == "interleaved_progressive"
    assert "use_light_spectral_router" not in spec
    cfg = load_cfg(generate_temp_config(12, "chain_interleaved_progressive_spatial", 1))
    assert cfg.MODEL.PARAM.get("use_light_spectral_router") in (None, False)
    model = ChainForecasting(**dict(cfg.MODEL.PARAM))
    out = model(torch.randn(2, 12, 307, 4), return_all=True)
    assert out["pred"].shape == (2, 12, 307, 1)
    print("[ok] original interleaved variant unchanged")


def test_config_isolation() -> None:
    cfg = load_cfg(generate_temp_config(12, "chain_interleaved_progressive_spatial_light_router", 1))
    assert cfg.MODEL.NAME == "ChainForecasting_LightSpectralRouter"
    assert "light_router_seed1" in str(cfg.TRAIN.CKPT_SAVE_DIR)
    p = cfg.MODEL.PARAM
    assert p["use_light_spectral_router"] is True
    assert p["router_hidden_dim"] == 8
    assert p["router_max_deviation"] == 0.05
    assert p["router_shared_across_stages"] is True
    assert p["chain_loss_weights"] == [0.2, 0.3, 1.0]
    assert p["progressive_spatial_topks"] == [8, 16, 32]
    print("[ok] light_router config / ckpt / model name isolated")


def test_zero_init_matches_baseline() -> None:
    torch.manual_seed(0)
    baseline = ChainForecasting(**interleaved_args())
    light = ChainForecasting(
        **interleaved_args(
            use_light_spectral_router=True,
            router_hidden_dim=8,
            router_max_deviation=0.05,
            router_shared_across_stages=True,
        )
    )
    _copy_shared_weights(baseline, light)
    assert light.light_spectral_router is not None
    assert torch.count_nonzero(light.light_spectral_router.fc2.weight) == 0
    assert torch.count_nonzero(light.light_spectral_router.fc2.bias) == 0

    history = torch.randn(2, 12, 307, 4)
    baseline.eval()
    light.eval()
    with torch.no_grad():
        y_b = baseline(history, return_all=True)
        y_l = light(history, return_all=True)
        # Also check coefficients at zero-init
        for h in (3, 6, 12):
            coef = light.light_spectral_router(history[..., 0], h / 12.0)
            assert torch.allclose(coef, torch.ones_like(coef), atol=1e-6)
    assert y_l["pred"].shape == (2, 12, 307, 1)
    assert torch.allclose(y_b["pred"], y_l["pred"], atol=1e-5, rtol=1e-5)
    for i in range(3):
        assert torch.allclose(y_b["temporal_preds"][i], y_l["temporal_preds"][i], atol=1e-5, rtol=1e-5)
        assert torch.allclose(y_b["chain_preds"][i], y_l["chain_preds"][i], atol=1e-5, rtol=1e-5)
    print("[ok] zero-init light router matches baseline T3/T6/T12/final")


def test_shared_router_and_coef_constraints() -> None:
    torch.manual_seed(1)
    model = ChainForecasting(
        **interleaved_args(
            use_light_spectral_router=True,
            router_hidden_dim=8,
            router_max_deviation=0.05,
        )
    )
    # Shared single module
    assert model.light_spectral_router is not None
    router_ids = {id(model.light_spectral_router)}
    assert len(router_ids) == 1

    # Nudge last layer so coefficients leave the all-ones point
    with torch.no_grad():
        model.light_spectral_router.fc2.weight.add_(0.5)
        model.light_spectral_router.fc2.bias.add_(torch.tensor([0.2, -0.1, -0.1]))

    history = torch.randn(4, 12, 307, 4)
    for h in (3, 6, 12):
        coef = model.light_spectral_router(history[..., 0], h / 12.0)
        assert coef.shape == (4, 3)
        assert torch.allclose(coef.sum(dim=-1), torch.full((4,), 3.0), atol=1e-5)
        assert float(coef.min()) >= 0.95 - 1e-5
        assert float(coef.max()) <= 1.05 + 1e-5
    print("[ok] shared router; coef sum=3 and range in [0.95, 1.05]")


def test_finite_grads_all_stages() -> None:
    torch.manual_seed(2)
    model = ChainForecasting(
        **interleaved_args(
            use_light_spectral_router=True,
            router_hidden_dim=8,
            router_max_deviation=0.05,
        )
    )
    with torch.no_grad():
        model.light_spectral_router.fc2.weight.add_(1e-2)
    history = torch.randn(2, 12, 307, 4)
    out = model(history, return_all=True)
    for stage_idx, stage_pred in enumerate(out["temporal_preds"]):
        model.zero_grad(set_to_none=True)
        stage_pred.sum().backward(retain_graph=True)
        g = model.light_spectral_router.fc2.weight.grad
        assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0, stage_idx
    model.zero_grad(set_to_none=True)
    sum(p.sum() for p in out["temporal_preds"]).backward()
    for name, p in model.light_spectral_router.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, name
    print("[ok] light router finite grads through T3/T6/T12")


def test_train_val_batch() -> None:
    torch.manual_seed(3)
    model = ChainForecasting(
        **interleaved_args(
            use_light_spectral_router=True,
            router_hidden_dim=8,
            router_max_deviation=0.05,
        )
    )
    history = torch.randn(4, 12, 307, 4)
    future = torch.randn(4, 12, 307, 4)
    model.train()
    out_tr = model(history_data=history, future_data=future, train=True, return_all=True)
    assert out_tr["pred"].shape == (4, 12, 307, 1)
    model.eval()
    with torch.no_grad():
        out_va = model(history_data=history, future_data=future, train=False, return_all=True)
    assert out_va["pred"].shape == (4, 12, 307, 1)
    print("[ok] train/val batch shapes [B,12,N,1]")


def main() -> int:
    test_original_variant()
    test_config_isolation()
    test_zero_init_matches_baseline()
    test_shared_router_and_coef_constraints()
    test_finite_grads_all_stages()
    test_train_val_batch()
    print("\nAll light spectral router checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
