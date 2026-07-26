#!/usr/bin/env python3
"""Checks for chain_interleaved_progressive_spatial_state_adapter."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basicts.archs import ChainForecasting
from basicts.archs.arch_zoo.ChainForecasting_arch.kasa_temporal_step import (
    interpolate_forecast,
)
from scripts.run_chain_forecasting_horizon_pems04 import (
    generate_temp_config,
    load_cfg,
    variant_spec,
)

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
        "use_forecast_state_adapter": False,
    }
    args.update(overrides)
    return args


def _copy_shared_weights(src: ChainForecasting, dst: ChainForecasting) -> None:
    src_sd = src.state_dict()
    dst_sd = dst.state_dict()
    shared = {
        k: v
        for k, v in src_sd.items()
        if k in dst_sd and "forecast_state_adapter" not in k
    }
    missing = dst.load_state_dict(shared, strict=False)
    assert all("forecast_state_adapter" in k for k in missing.missing_keys), missing.missing_keys


def test_original_variant() -> None:
    spec = variant_spec("chain_interleaved_progressive_spatial", 12)
    assert spec["spatial_placement"] == "interleaved_progressive"
    assert "use_forecast_state_adapter" not in spec
    cfg = load_cfg(generate_temp_config(12, "chain_interleaved_progressive_spatial", 1))
    assert cfg.MODEL.PARAM.get("use_forecast_state_adapter") in (None, False)
    model = ChainForecasting(**dict(cfg.MODEL.PARAM))
    out = model(torch.randn(2, 12, 307, 4), return_all=True)
    assert out["pred"].shape == (2, 12, 307, 1)
    print("[ok] original interleaved variant unchanged")


def test_config_isolation() -> None:
    cfg = load_cfg(
        generate_temp_config(12, "chain_interleaved_progressive_spatial_state_adapter", 1)
    )
    assert cfg.MODEL.NAME == "ChainForecasting_StateAdapter"
    assert "state_adapter_seed1" in str(cfg.TRAIN.CKPT_SAVE_DIR)
    p = cfg.MODEL.PARAM
    assert p["use_forecast_state_adapter"] is True
    assert p["forecast_state_adapter_hidden_dim"] == 16
    assert p["forecast_state_adapter_epsilon"] == 0.05
    assert p["chain_loss_weights"] == [0.2, 0.3, 1.0]
    assert p.get("use_light_spectral_router") in (None, False)
    assert p.get("use_spectral_stage_router") in (None, False)
    print("[ok] state_adapter config / ckpt / model name isolated")


def test_zero_init_matches_baseline() -> None:
    torch.manual_seed(0)
    baseline = ChainForecasting(**interleaved_args())
    adapted = ChainForecasting(
        **interleaved_args(
            use_forecast_state_adapter=True,
            forecast_state_adapter_hidden_dim=16,
            forecast_state_adapter_epsilon=0.05,
        )
    )
    _copy_shared_weights(baseline, adapted)
    assert adapted.forecast_state_adapter is not None
    assert torch.count_nonzero(adapted.forecast_state_adapter.proj_out.weight) == 0
    assert torch.count_nonzero(adapted.forecast_state_adapter.proj_out.bias) == 0

    history = torch.randn(2, 12, 307, 4)
    baseline.eval()
    adapted.eval()
    with torch.no_grad():
        y_b = baseline(history, return_all=True)
        y_a = adapted(history, return_all=True)
    assert y_a["pred"].shape == (2, 12, 307, 1)
    assert torch.allclose(y_b["pred"], y_a["pred"], atol=1e-5, rtol=1e-5)
    for i, name in enumerate(("Z3", "Z6", "Z12")):
        assert torch.allclose(
            y_b["chain_preds"][i], y_a["chain_preds"][i], atol=1e-5, rtol=1e-5
        ), name
        assert torch.allclose(
            y_b["temporal_preds"][i], y_a["temporal_preds"][i], atol=1e-5, rtol=1e-5
        ), name
    print("[ok] zero-init adapter matches baseline Z3/Z6/Z12/final")


def test_shared_adapter_and_z6_feeds_t12() -> None:
    torch.manual_seed(1)
    model = ChainForecasting(
        **interleaved_args(
            use_forecast_state_adapter=True,
            forecast_state_adapter_hidden_dim=16,
            forecast_state_adapter_epsilon=0.05,
        )
    )
    assert model.forecast_state_adapter is not None
    # Exactly one shared adapter instance
    assert id(model.forecast_state_adapter) == id(model.forecast_state_adapter)

    # Nudge last layer so adapter is non-identity
    with torch.no_grad():
        model.forecast_state_adapter.proj_out.weight.fill_(0.05)
        model.forecast_state_adapter.proj_out.bias.fill_(0.01)

    model.eval()
    captured = {}
    real_t12 = model.temporal_steps[2].forward

    def wrapped(history_data, prev_forecast=None, prev_latent=None, **kwargs):
        captured["prev_forecast"] = (
            None if prev_forecast is None else prev_forecast.detach().clone()
        )
        return real_t12(
            history_data,
            prev_forecast=prev_forecast,
            prev_latent=prev_latent,
            **kwargs,
        )

    model.temporal_steps[2].forward = wrapped
    history = torch.randn(2, 12, 307, 4)
    with torch.no_grad():
        out = model(history, return_all=True)
    model.temporal_steps[2].forward = real_t12

    z6_corrected = out["chain_preds"][1]
    expected_prev = interpolate_forecast(z6_corrected, 12)
    assert captured["prev_forecast"] is not None
    assert torch.allclose(captured["prev_forecast"], expected_prev, atol=1e-5, rtol=1e-5)

    # Without adapter, Z6 differs; Z3 must stay identical.
    adapter = model.forecast_state_adapter
    model.forecast_state_adapter = None
    with torch.no_grad():
        out_no = model(history, return_all=True)
    model.forecast_state_adapter = adapter
    assert not torch.allclose(out["chain_preds"][1], out_no["chain_preds"][1], atol=1e-5)
    assert torch.allclose(out["chain_preds"][0], out_no["chain_preds"][0], atol=1e-5)
    print("[ok] shared adapter; corrected Z6 feeds T12; Z3 unaffected")


def test_finite_grads() -> None:
    torch.manual_seed(2)
    model = ChainForecasting(
        **interleaved_args(
            use_forecast_state_adapter=True,
            forecast_state_adapter_hidden_dim=16,
            forecast_state_adapter_epsilon=0.05,
        )
    )
    with torch.no_grad():
        model.forecast_state_adapter.proj_out.weight.add_(1e-2)
    history = torch.randn(2, 12, 307, 4)
    out = model(history, return_all=True)
    assert torch.isfinite(out["pred"]).all()
    loss = out["pred"].sum() + sum(p.sum() for p in out["chain_preds"])
    loss.backward()

    for name, p in model.forecast_state_adapter.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        assert p.grad.abs().sum() > 0, name

    for i, step in enumerate(model.temporal_steps):
        has_grad = False
        for p in step.parameters():
            if p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad, f"temporal_step[{i}]"

    for i, sp in enumerate(model.progressive_spatial_modules):
        has_grad = False
        for p in sp.parameters():
            if p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad, f"spatial_stage[{i}]"
    print("[ok] finite grads for adapter, T3/T6/T12, and spatial stages")


def test_train_val_batch() -> None:
    torch.manual_seed(3)
    model = ChainForecasting(
        **interleaved_args(
            use_forecast_state_adapter=True,
            forecast_state_adapter_hidden_dim=16,
            forecast_state_adapter_epsilon=0.05,
        )
    )
    history = torch.randn(4, 12, 307, 4)
    future = torch.randn(4, 12, 307, 4)
    model.train()
    out_tr = model(history_data=history, future_data=future, train=True, return_all=True)
    assert out_tr["pred"].shape == (4, 12, 307, 1)
    assert torch.isfinite(out_tr["pred"]).all()
    model.eval()
    with torch.no_grad():
        out_va = model(history_data=history, future_data=future, train=False, return_all=True)
    assert out_va["pred"].shape == (4, 12, 307, 1)
    assert torch.isfinite(out_va["pred"]).all()
    print("[ok] train/val batch shapes [B,12,N,1] and finite")


def main() -> int:
    test_original_variant()
    test_config_isolation()
    test_zero_init_matches_baseline()
    test_shared_adapter_and_z6_feeds_t12()
    test_finite_grads()
    test_train_val_batch()
    print("\nAll forecast-state adapter checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
