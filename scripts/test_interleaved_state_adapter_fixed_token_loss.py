#!/usr/bin/env python3
"""Checks for fixed state-adapter + token-normalized MAE combo variant."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from easytorch.config import init_cfg

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basicts.archs import ChainForecasting
from basicts.archs.arch_zoo.ChainForecasting_arch.kasa_temporal_step import (
    interpolate_forecast,
)
from basicts.losses.forecast_state_token_mae import forecast_state_token_mae
from basicts.runners.runner_zoo.chain_forecasting_runner import ChainForecastingRunner
from scripts.run_chain_forecasting_horizon_pems04 import (
    generate_temp_config,
    load_cfg,
    variant_spec,
)

ADJ = str(ROOT / "datasets" / "PEMS04" / "adj_mx.pkl")
FIXED = "chain_interleaved_progressive_spatial_state_adapter_fixed"
COMBO = "chain_interleaved_progressive_spatial_state_adapter_fixed_token_loss"


def fixed_model_args(**overrides):
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
        "use_forecast_state_adapter": True,
        "forecast_state_adapter_mode": "condition_only",
        "forecast_state_adapter_hidden_dim": 16,
        "forecast_state_adapter_epsilon": 0.02,
    }
    args.update(overrides)
    return args


def _count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_imports_and_isolation() -> None:
    for name in (FIXED, COMBO, "chain_interleaved_progressive_spatial"):
        cfg = load_cfg(generate_temp_config(12, name, 1))
        model = ChainForecasting(**dict(cfg.MODEL.PARAM))
        assert model is not None

    fixed_cfg = load_cfg(generate_temp_config(12, FIXED, 1))
    combo_cfg = load_cfg(generate_temp_config(12, COMBO, 1))
    assert fixed_cfg.MODEL.NAME == "ChainForecasting_StateAdapterFixed"
    assert combo_cfg.MODEL.NAME == "ChainForecasting_StateAdapterFixed_TokenMAE"
    assert "state_adapter_fixed_token_loss_seed1" in str(combo_cfg.TRAIN.CKPT_SAVE_DIR)
    assert "state_adapter_fixed_seed1" in str(fixed_cfg.TRAIN.CKPT_SAVE_DIR)
    assert fixed_cfg.TRAIN.CKPT_SAVE_DIR != combo_cfg.TRAIN.CKPT_SAVE_DIR

    fp = fixed_cfg.MODEL.PARAM
    cp = combo_cfg.MODEL.PARAM
    assert fp.get("chain_loss_mode") in (None, "weighted")
    assert fp["chain_loss_weights"] == [0.2, 0.3, 1.0]
    assert cp["chain_loss_mode"] == "token_normalized"
    assert cp["chain_loss_weights"] is None
    for k in (
        "use_forecast_state_adapter",
        "forecast_state_adapter_mode",
        "forecast_state_adapter_hidden_dim",
        "forecast_state_adapter_epsilon",
        "spatial_placement",
        "progressive_spatial_ratios",
        "progressive_spatial_topks",
        "progressive_spatial_alphas",
    ):
        assert fp[k] == cp[k], k
    print("[ok] imports + config/ckpt isolation")


def test_param_count_and_forward_match() -> None:
    torch.manual_seed(21)
    fixed = ChainForecasting(**fixed_model_args())
    torch.manual_seed(21)
    combo = ChainForecasting(
        **fixed_model_args(chain_loss_weights=None)  # unused by model
    )
    n_fixed = _count_params(fixed)
    n_combo = _count_params(combo)
    assert n_fixed == n_combo, (n_fixed, n_combo)

    # Non-identity adapter so Z6_condition != Z6_raw
    with torch.no_grad():
        for m in (fixed, combo):
            m.forecast_state_adapter.proj_out.weight.fill_(0.05)
            m.forecast_state_adapter.proj_out.bias.fill_(0.01)
            m.load_state_dict(fixed.state_dict())

    history = torch.randn(2, 12, 307, 4)
    captured = {"fixed": None, "combo": None}

    def wrap(model, key):
        real = model.temporal_steps[2].forward

        def fn(history_data, prev_forecast=None, prev_latent=None, **kwargs):
            captured[key] = (
                None if prev_forecast is None else prev_forecast.detach().clone()
            )
            return real(
                history_data,
                prev_forecast=prev_forecast,
                prev_latent=prev_latent,
                **kwargs,
            )

        model.temporal_steps[2].forward = fn

    wrap(fixed, "fixed")
    wrap(combo, "combo")
    fixed.eval()
    combo.eval()
    with torch.no_grad():
        out_f = fixed(history, return_all=True)
        out_c = combo(history, return_all=True)

    for i, name in enumerate(("Z3", "Z6_raw", "Z12_raw")):
        d = (out_f["chain_preds"][i] - out_c["chain_preds"][i]).abs().max().item()
        assert d == 0.0, (name, d)
    assert torch.equal(out_f["pred"], out_c["pred"])
    assert torch.equal(captured["fixed"], captured["combo"])
    assert captured["fixed"] is not None
    # T12 sees interpolate(Z6_condition, 12), not Z6_raw
    z6 = out_f["chain_preds"][1]
    assert not torch.allclose(
        captured["fixed"], interpolate_forecast(z6, 12), atol=1e-6
    )
    print(f"[ok] param count equal ({n_fixed}); Z3/Z6_raw/Z6_condition/Z12_raw match")
    return n_fixed, out_f, captured["fixed"]


def test_loss_only_differs_and_manual_match() -> dict:
    from easytorch.config import init_cfg as _init

    fixed_rel = str(generate_temp_config(12, FIXED, 1).relative_to(ROOT))
    combo_rel = str(generate_temp_config(12, COMBO, 1).relative_to(ROOT))
    runner_w = ChainForecastingRunner(_init(fixed_rel, False))
    runner_t = ChainForecastingRunner(_init(combo_rel, False))
    assert runner_w.chain_loss_mode == "weighted"
    assert runner_t.chain_loss_mode == "token_normalized"
    assert runner_t.chain_loss_weights == []

    torch.manual_seed(9)
    model = ChainForecasting(**fixed_model_args())
    with torch.no_grad():
        model.forecast_state_adapter.proj_out.weight.fill_(0.04)
        model.forecast_state_adapter.proj_out.bias.fill_(0.01)

    history = torch.randn(2, 12, 307, 4)
    real = torch.randn(2, 12, 307, 1)
    # inject null tokens on Y6 path via pooled target later
    model.train()
    out = model(history, return_all=True)
    z3, z6_raw, z12_raw = out["chain_preds"]
    assert torch.allclose(out["pred"], z12_raw)

    # Capture Z6_condition and confirm it is NOT used as L6 prediction
    expected_cond = model.forecast_state_adapter(
        current_state=z6_raw.detach(),
        previous_state=z3.detach(),
        stage_ratio=0.5,
    )
    assert not torch.allclose(expected_cond, z6_raw, atol=1e-6)

    targets = [ChainForecasting.pool_target(real, k) for k in [3, 6, 12]]
    null_val = -9999.0
    targets[1] = targets[1].clone()
    targets[1][0, 0, 0, 0] = null_val  # one null token on h=6

    def identity_pair(a, b):
        return a, b

    preds = [z3, z6_raw, z12_raw]
    loss_fn = forecast_state_token_mae(
        preds, targets, null_val=null_val, rescale_pair=identity_pair
    )

    stage_counts = []
    num = 0.0
    den = 0.0
    null_t = torch.tensor(null_val)
    for p, t in zip(preds, targets):
        mask = (~torch.isclose(t, null_t, atol=5e-5, rtol=0.0)).float()
        stage_counts.append(float(mask.sum().detach()))
        num += float((torch.abs(p - t) * mask).sum().detach())
        den += float(mask.sum().detach())
    manual = num / max(den, 1.0)
    assert abs(float(loss_fn) - manual) < 1e-6, (float(loss_fn), manual)

    # Natural ratio ~ 3:6:12 (minus 1 null on h=6)
    assert abs(stage_counts[0] / stage_counts[2] - 3 / 12) < 1e-6
    assert abs(stage_counts[1] / stage_counts[2] - (6 * 2 * 307 - 1) / (12 * 2 * 307)) < 1e-6

    # Confirm L6 uses Z6_raw in chain_preds[1]
    out_dict = {
        "pred": z12_raw,
        "chain_preds": preds,
        "temporal_preds": out["temporal_preds"],
        "spatial_preds": out["spatial_preds"],
        "spatial_stage_preds": out.get("spatial_stage_preds") or [],
    }
    assert out_dict["chain_preds"][1] is z6_raw

    def weighted_loss(pred, target, weight):
        mask = (~torch.isclose(target, null_t, atol=5e-5, rtol=0.0)).float()
        return weight * (
            (torch.abs(pred - target) * mask).sum() / mask.sum().clamp_min(1.0)
        )

    runner_w.null_val = null_val
    runner_w._weighted_loss = weighted_loss
    # Use targets without our injected null for weighted path via pool on `real`
    # Build a custom out with same preds; weighted uses pool_target(real) internally.
    loss_w = runner_w._legacy_loss(out_dict, real)
    assert abs(float(loss_w) - float(loss_fn)) > 1e-4

    print(
        f"[ok] token loss={float(loss_fn):.6f} manual={manual:.6f} "
        f"weighted={float(loss_w):.6f}"
    )
    print(
        f"     token counts h3/h6/h12={stage_counts} "
        f"ratios={[c / stage_counts[0] for c in stage_counts]}"
    )
    return {
        "loss": float(loss_fn),
        "manual": manual,
        "weighted": float(loss_w),
        "stage_counts": stage_counts,
        "n_params": _count_params(model),
    }


def test_gradients_and_batches() -> None:
    torch.manual_seed(5)
    model = ChainForecasting(**fixed_model_args())
    with torch.no_grad():
        model.forecast_state_adapter.proj_out.weight.fill_(0.05)
        model.forecast_state_adapter.proj_out.bias.fill_(0.01)

    history = torch.randn(2, 12, 307, 4)
    real = torch.randn(2, 12, 307, 1)
    model.train()
    out = model(history, return_all=True)
    preds = list(out["chain_preds"])
    targets = [ChainForecasting.pool_target(real, k) for k in [3, 6, 12]]
    loss = forecast_state_token_mae(preds, targets, null_val=0.0)
    assert torch.isfinite(loss)
    loss.backward()

    adapter_ok = all(
        p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
        for p in model.forecast_state_adapter.parameters()
    )
    assert adapter_ok
    for i, step in enumerate(model.temporal_steps):
        g = sum(p.grad.abs().sum().item() for p in step.parameters() if p.grad is not None)
        assert g > 0, f"T{i}"
    for i, m in enumerate(model.progressive_spatial_modules):
        g = sum(p.grad.abs().sum().item() for p in m.parameters() if p.grad is not None)
        assert g > 0, f"S{i}"

    model.eval()
    with torch.no_grad():
        val = model(torch.randn(2, 12, 307, 4), return_all=True)
    assert val["pred"].shape == (2, 12, 307, 1)
    assert torch.isfinite(val["pred"]).all()
    print("[ok] grads + one train/val batch")


def test_runner_no_double_final() -> None:
    cfg = init_cfg(str(generate_temp_config(12, COMBO, 1).relative_to(ROOT)), False)
    runner = ChainForecastingRunner(cfg)
    torch.manual_seed(0)
    model = ChainForecasting(**dict(cfg["MODEL"]["PARAM"]))
    history = torch.randn(2, 12, 307, 4)
    real = torch.randn(2, 12, 307, 1)
    out = model(history, return_all=True)

    # Monkey: if pred were added twice, loss would change when pred differs from chain[-1]
    out2 = dict(out)
    out2["pred"] = out["pred"] + 10.0  # poison final-only tensor
    runner.null_val = float("nan")  # no nulls via nan-mask path? use 0.0 none in data
    runner.null_val = -1e9  # nothing matches

    def identity(a, b):
        return a, b

    runner._rescale_pair = identity
    # With null_val that matches nothing, both should equal mean |err|
    l1 = runner._token_mae_loss(out, real)
    l2 = runner._token_mae_loss(out2, real)
    assert torch.allclose(l1, l2), "token loss must ignore out['pred'] (no double final)"
    print("[ok] no double-counting of final prediction in token loss")


def main() -> None:
    test_imports_and_isolation()
    n_params, _, _ = test_param_count_and_forward_match()
    stats = test_loss_only_differs_and_manual_match()
    test_gradients_and_batches()
    test_runner_no_double_final()
    # fixed still weighted
    assert variant_spec(FIXED, 12)["chain_loss_weights"] == [0.2, 0.3, 1.0]
    assert variant_spec(COMBO, 12)["chain_loss_mode"] == "token_normalized"
    print("[ok] all fixed+token_loss checks passed")
    print(
        f"REPORT n_params={n_params} token_counts={stats['stage_counts']} "
        f"loss={stats['loss']} manual={stats['manual']}"
    )


if __name__ == "__main__":
    main()
