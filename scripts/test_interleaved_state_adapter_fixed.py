#!/usr/bin/env python3
"""Checks for chain_interleaved_progressive_spatial_state_adapter_fixed."""
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


def fixed_args(**overrides):
    args = interleaved_args(
        use_forecast_state_adapter=True,
        forecast_state_adapter_mode="condition_only",
        forecast_state_adapter_hidden_dim=16,
        forecast_state_adapter_epsilon=0.02,
    )
    args.update(overrides)
    return args


def test_imports() -> None:
    for name in (
        "chain_interleaved_progressive_spatial",
        "chain_interleaved_progressive_spatial_state_adapter",
        "chain_interleaved_progressive_spatial_state_adapter_fixed",
    ):
        cfg = load_cfg(generate_temp_config(12, name, 1))
        model = ChainForecasting(**dict(cfg.MODEL.PARAM))
        assert model is not None
    print("[ok] baseline and both adapter variants import")


def test_config_isolation() -> None:
    cfg = load_cfg(
        generate_temp_config(
            12, "chain_interleaved_progressive_spatial_state_adapter_fixed", 1
        )
    )
    assert cfg.MODEL.NAME == "ChainForecasting_StateAdapterFixed"
    assert "state_adapter_fixed_seed1" in str(cfg.TRAIN.CKPT_SAVE_DIR)
    assert "state_adapter_seed1" not in str(cfg.TRAIN.CKPT_SAVE_DIR).replace(
        "state_adapter_fixed_seed1", ""
    )
    p = cfg.MODEL.PARAM
    assert p["use_forecast_state_adapter"] is True
    assert p["forecast_state_adapter_mode"] == "condition_only"
    assert p["forecast_state_adapter_hidden_dim"] == 16
    assert p["forecast_state_adapter_epsilon"] == 0.02
    assert p["chain_loss_weights"] == [0.2, 0.3, 1.0]
    assert p.get("use_light_spectral_router") in (None, False)
    assert p.get("use_spectral_stage_router") in (None, False)

    legacy = load_cfg(
        generate_temp_config(12, "chain_interleaved_progressive_spatial_state_adapter", 1)
    )
    assert legacy.MODEL.NAME == "ChainForecasting_StateAdapter"
    assert legacy.MODEL.PARAM["forecast_state_adapter_epsilon"] == 0.05
    print("[ok] fixed config / ckpt / model name isolated")


def test_init_order_fairness() -> None:
    """Same seed: fixed backbone weights must match baseline (adapter last)."""
    torch.manual_seed(42)
    baseline = ChainForecasting(**interleaved_args())
    torch.manual_seed(42)
    fixed = ChainForecasting(**fixed_args())

    base_sd = baseline.state_dict()
    fixed_sd = fixed.state_dict()
    for k, v in base_sd.items():
        assert k in fixed_sd, f"baseline key missing in fixed: {k}"
        assert torch.equal(v, fixed_sd[k]), f"init mismatch on {k}"

    missing, unexpected = fixed.load_state_dict(base_sd, strict=False)
    assert unexpected == [], unexpected
    assert missing, "expected adapter keys to be missing"
    assert all("forecast_state_adapter" in k for k in missing), missing
    print(
        f"[ok] init-order fairness; missing keys ({len(missing)}): "
        f"{sorted(missing)[:6]}..."
    )
    return missing


def test_zero_init_full_equivalence() -> None:
    torch.manual_seed(7)
    baseline = ChainForecasting(**interleaved_args())
    torch.manual_seed(7)
    fixed = ChainForecasting(**fixed_args())

    missing, unexpected = fixed.load_state_dict(baseline.state_dict(), strict=False)
    assert unexpected == []
    assert all("forecast_state_adapter" in k for k in missing)
    assert torch.count_nonzero(fixed.forecast_state_adapter.proj_out.weight) == 0
    assert torch.count_nonzero(fixed.forecast_state_adapter.proj_out.bias) == 0

    history = torch.randn(2, 12, 307, 4)
    baseline.eval()
    fixed.eval()
    with torch.no_grad():
        y_b = baseline(history, return_all=True)
        y_f = fixed(history, return_all=True)

    diffs = {
        "Z3": (y_b["chain_preds"][0] - y_f["chain_preds"][0]).abs().max().item(),
        "Z6_raw": (y_b["chain_preds"][1] - y_f["chain_preds"][1]).abs().max().item(),
        "Z12_raw": (y_b["chain_preds"][2] - y_f["chain_preds"][2]).abs().max().item(),
        "final": (y_b["pred"] - y_f["pred"]).abs().max().item(),
    }
    for name, d in diffs.items():
        assert d < 1e-7, f"{name} max abs diff={d}"
    print(f"[ok] zero-init full-model equivalence: {diffs}")
    print(f"     missing={sorted(missing)}")
    print(f"     unexpected={unexpected}")
    return diffs, missing, unexpected


def test_condition_only_dataflow() -> None:
    torch.manual_seed(3)
    model = ChainForecasting(**fixed_args())
    assert model.forecast_state_adapter is not None
    assert model.forecast_state_adapter.correction_scale == "sample_scale"
    assert model.forecast_state_adapter.delta_feature == "normalized"
    assert model.forecast_state_adapter.epsilon == 0.02

    with torch.no_grad():
        model.forecast_state_adapter.proj_out.weight.fill_(0.08)
        model.forecast_state_adapter.proj_out.bias.fill_(0.02)

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
    history = torch.randn(2, 12, 307, 4, requires_grad=True)
    model.train()
    out = model(history, return_all=True)
    z3, z6_raw, z12_raw = out["chain_preds"]
    pred = out["pred"]

    assert pred.shape == (2, 12, 307, 1)
    assert torch.allclose(pred, z12_raw)
    assert "prev_forecast" in captured
    prev_cond = captured["prev_forecast"]
    assert prev_cond is not None
    # T12 receives prev_forecast already interpolated to target_len=12.
    assert prev_cond.shape == (2, 12, 307, 1)

    # Z6_condition = Adapter(Z6_raw, Z3); T12 sees interpolate(Z6_condition, 12)
    expected_cond = model.forecast_state_adapter(
        current_state=z6_raw.detach(),
        previous_state=z3.detach(),
        stage_ratio=6.0 / 12.0,
    )
    assert expected_cond.shape == (2, 6, 307, 1)
    assert not torch.allclose(expected_cond, z6_raw, atol=1e-6)
    expected_up = interpolate_forecast(expected_cond, 12)
    assert torch.allclose(prev_cond, expected_up, atol=1e-5, rtol=1e-5)
    # Must NOT be interpolate(Z6_raw) — condition differs from raw supervised state
    assert not torch.allclose(
        prev_cond, interpolate_forecast(z6_raw.detach(), 12), atol=1e-6
    )

    corr = model.forecast_state_adapter._last_correction
    assert corr is not None
    pred_scale = pred.detach().abs().mean().item()
    corr_mean = corr.detach().abs().mean().item()
    corr_max = corr.detach().abs().max().item()
    ratio = corr_mean / max(pred_scale, 1e-8)
    print(
        f"[ok] condition-only dataflow: "
        f"corr_mean={corr_mean:.6e} corr_max={corr_max:.6e} "
        f"corr/pred_scale={ratio:.6e}"
    )
    assert torch.isfinite(corr).all()
    assert torch.isfinite(pred).all()
    # ε=0.02 and tanh <= 1 => |corr| <= ~0.02 * sample_scale
    assert corr_max < 1.0
    return corr_mean, corr_max, ratio


def test_gradients_and_batches() -> None:
    torch.manual_seed(11)
    model = ChainForecasting(**fixed_args())
    with torch.no_grad():
        model.forecast_state_adapter.proj_out.weight.fill_(0.05)
        model.forecast_state_adapter.proj_out.bias.fill_(0.01)

    history = torch.randn(2, 12, 307, 4)
    target = torch.randn(2, 12, 307, 1)
    model.train()
    out = model(history, return_all=True)
    z3, z6, z12 = out["chain_preds"]
    loss = (
        0.2 * (z3 - target[:, :3]).abs().mean()
        + 0.3 * (z6 - target[:, :6]).abs().mean()
        + 1.0 * (z12 - target).abs().mean()
    )
    assert torch.isfinite(loss)
    loss.backward()

    adapter_grads = [
        p.grad.abs().sum().item()
        for p in model.forecast_state_adapter.parameters()
        if p.grad is not None
    ]
    assert adapter_grads and all(g > 0 for g in adapter_grads)
    assert all(torch.isfinite(p.grad).all() for p in model.forecast_state_adapter.parameters())

    t_grads = [
        sum(p.grad.abs().sum().item() for p in step.parameters() if p.grad is not None)
        for step in model.temporal_steps
    ]
    assert all(g > 0 for g in t_grads), t_grads
    spa_grads = [
        sum(p.grad.abs().sum().item() for p in m.parameters() if p.grad is not None)
        for m in model.progressive_spatial_modules
    ]
    assert all(g > 0 for g in spa_grads), spa_grads

    # One validation batch (eval)
    model.eval()
    with torch.no_grad():
        val_out = model(torch.randn(2, 12, 307, 4), return_all=True)
    assert val_out["pred"].shape == (2, 12, 307, 1)
    assert torch.isfinite(val_out["pred"]).all()
    print(
        f"[ok] grads: adapter={adapter_grads} temporal={t_grads} spatial={spa_grads}"
    )
    print("[ok] one train batch + one val batch")


def main() -> None:
    test_imports()
    test_config_isolation()
    test_init_order_fairness()
    test_zero_init_full_equivalence()
    test_condition_only_dataflow()
    test_gradients_and_batches()
    # Ensure original variant still present
    assert "chain_interleaved_progressive_spatial" in (
        "chain_interleaved_progressive_spatial",
    )
    spec = variant_spec("chain_interleaved_progressive_spatial", 12)
    assert spec.get("use_forecast_state_adapter") in (None, False)
    print("[ok] all fixed-adapter checks passed")


if __name__ == "__main__":
    main()
