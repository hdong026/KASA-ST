#!/usr/bin/env python3
"""Dry-run checks for interleaved Spectral Router and Token MAE variants."""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basicts.archs import ChainForecasting
from basicts.losses.forecast_state_token_mae import forecast_state_token_mae
from basicts.runners.runner_zoo.chain_forecasting_runner import ChainForecastingRunner
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
        "use_spectral_stage_router": False,
    }
    args.update(overrides)
    return args


def _copy_shared_weights(src: ChainForecasting, dst: ChainForecasting) -> None:
    src_sd = src.state_dict()
    dst_sd = dst.state_dict()
    shared = {k: v for k, v in src_sd.items() if k in dst_sd and "spectral_branch_router" not in k}
    missing = dst.load_state_dict(shared, strict=False)
    assert all("spectral_branch_router" in k for k in missing.missing_keys), missing.missing_keys


def test_original_variant_import() -> None:
    spec = variant_spec("chain_interleaved_progressive_spatial", 12)
    assert spec["spatial_placement"] == "interleaved_progressive"
    assert spec["use_prev_condition"] is True
    assert spec["chain_loss_weights"] == [0.2, 0.3, 1.0]
    assert "use_spectral_stage_router" not in spec or not spec.get("use_spectral_stage_router")
    cfg_path = generate_temp_config(12, "chain_interleaved_progressive_spatial", 1)
    cfg = load_cfg(cfg_path)
    assert cfg.RUNNER is ChainForecastingRunner
    assert cfg.MODEL.ARCH is ChainForecasting
    assert cfg.TRAIN.LOSS.__name__ == "masked_mae"
    model = cfg.MODEL.ARCH(**cfg.MODEL.PARAM)
    history = torch.randn(2, 12, 307, 4)
    out = model(history, return_all=True)
    assert out["pred"].shape == (2, 12, 307, 1)
    print("[ok] original interleaved variant import/forward")


def test_router_zero_init_matches_baseline() -> None:
    torch.manual_seed(0)
    baseline = ChainForecasting(**interleaved_args())
    router = ChainForecasting(**interleaved_args(use_spectral_stage_router=True))
    _copy_shared_weights(baseline, router)
    assert router.spectral_branch_router is not None
    assert torch.count_nonzero(router.spectral_branch_router.fc2.weight) == 0
    assert torch.count_nonzero(router.spectral_branch_router.fc2.bias) == 0

    history = torch.randn(2, 12, 307, 4)
    baseline.eval()
    router.eval()
    with torch.no_grad():
        y_b = baseline(history, return_all=True)
        y_r = router(history, return_all=True)
    assert y_r["pred"].shape == (2, 12, 307, 1)
    assert torch.allclose(y_b["pred"], y_r["pred"], atol=1e-5, rtol=1e-5), (
        float((y_b["pred"] - y_r["pred"]).abs().max())
    )
    for i in range(3):
        assert torch.allclose(y_b["chain_preds"][i], y_r["chain_preds"][i], atol=1e-5, rtol=1e-5)
    print("[ok] router zero-init matches baseline outputs")


def test_router_finite_grads_all_stages() -> None:
    torch.manual_seed(1)
    # At exact zero-init, fc2 zeros make fc1 unused (logits always 0); still require
    # finite grads on the last layer, then check all params after a tiny warmup.
    model = ChainForecasting(**interleaved_args(use_spectral_stage_router=True))
    history = torch.randn(2, 12, 307, 4)
    out = model(history, return_all=True)
    loss = sum(p.sum() for p in out["temporal_preds"])
    loss.backward()
    for name, p in model.spectral_branch_router.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name
    assert model.spectral_branch_router.fc2.weight.grad.abs().sum() > 0

    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        model.spectral_branch_router.fc2.weight.add_(1e-3)
    out2 = model(history, return_all=True)
    # Stage-wise connectivity: each temporal stage contributes to router grads.
    for stage_idx, stage_pred in enumerate(out2["temporal_preds"]):
        model.zero_grad(set_to_none=True)
        stage_pred.sum().backward(retain_graph=True)
        g = model.spectral_branch_router.fc2.weight.grad
        assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0, stage_idx
    model.zero_grad(set_to_none=True)
    sum(p.sum() for p in out2["temporal_preds"]).backward()
    for name, p in model.spectral_branch_router.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, name
    print("[ok] router has finite grads through T3/T6/T12")


def test_token_loss_ignores_weights_and_matches_manual() -> None:
    torch.manual_seed(2)
    preds = [
        torch.randn(2, 3, 307, 1),
        torch.randn(2, 6, 307, 1),
        torch.randn(2, 12, 307, 1),
    ]
    targets = [p + 0.1 for p in preds]
    # Inject nulls
    targets[1][0, 0, 0, 0] = 0.0
    preds[1][0, 0, 0, 0] = 5.0

    def identity_pair(a, b):
        return a, b

    loss = forecast_state_token_mae(preds, targets, null_val=0.0, rescale_pair=identity_pair)

    num = 0.0
    den = 0.0
    for p, t in zip(preds, targets):
        mask = (~torch.isclose(t, torch.tensor(0.0), atol=5e-5, rtol=0.0)).float()
        num += float((torch.abs(p - t) * mask).sum())
        den += float(mask.sum())
    manual = num / max(den, 1.0)
    assert abs(float(loss) - manual) < 1e-6, (float(loss), manual)

    # Runner path should not require weights.
    from easytorch.config import init_cfg

    cfg_path = generate_temp_config(12, "chain_interleaved_progressive_spatial_token_loss", 1)
    rel = str(cfg_path.relative_to(ROOT))
    cfg = init_cfg(rel, False)
    assert cfg["MODEL"]["PARAM"].get("chain_loss_mode") == "token_mae"
    assert cfg["MODEL"]["PARAM"].get("chain_loss_weights") is None
    assert cfg["MODEL"]["NAME"] == "ChainForecasting_TokenMAE"
    runner = ChainForecastingRunner(cfg)
    assert runner.chain_loss_mode == "token_mae"
    # Token loss must not read artificial weights for aggregation.
    assert runner.chain_loss_weights == []
    print("[ok] token loss manual match + ignores chain_loss_weights")


def test_prediction_shapes_and_one_batch() -> None:
    torch.manual_seed(3)
    for variant, overrides in [
        ("router", {"use_spectral_stage_router": True}),
        ("token_model", {"use_spectral_stage_router": False}),
    ]:
        model = ChainForecasting(**interleaved_args(**overrides))
        history = torch.randn(4, 12, 307, 4)
        future = torch.randn(4, 12, 307, 4)
        model.train()
        out = model(history_data=history, future_data=future, train=True, return_all=True)
        assert out["pred"].shape == (4, 12, 307, 1), variant
        model.eval()
        with torch.no_grad():
            out_v = model(history_data=history, future_data=future, train=False, return_all=True)
        assert out_v["pred"].shape == (4, 12, 307, 1), variant
        print(f"[ok] {variant} train/val batch shapes {(4, 12, 307, 1)}")


def test_generated_configs_isolation() -> None:
    base = generate_temp_config(12, "chain_interleaved_progressive_spatial", 1)
    router = generate_temp_config(12, "chain_interleaved_progressive_spatial_router", 1)
    token = generate_temp_config(12, "chain_interleaved_progressive_spatial_token_loss", 1)
    b, r, t = load_cfg(base), load_cfg(router), load_cfg(token)
    assert "chain_interleaved_progressive_spatial_seed1" in str(b.TRAIN.CKPT_SAVE_DIR)
    assert "chain_interleaved_progressive_spatial_router_seed1" in str(r.TRAIN.CKPT_SAVE_DIR)
    assert "chain_interleaved_progressive_spatial_token_loss_seed1" in str(t.TRAIN.CKPT_SAVE_DIR)
    assert r.MODEL.NAME == "ChainForecasting_SpectralRouter"
    assert t.MODEL.NAME == "ChainForecasting_TokenMAE"
    assert r.MODEL.PARAM["use_spectral_stage_router"] is True
    assert t.MODEL.PARAM.get("use_spectral_stage_router") is False
    assert t.MODEL.PARAM["chain_loss_mode"] == "token_mae"
    assert r.MODEL.PARAM["chain_loss_weights"] == [0.2, 0.3, 1.0]
    print("[ok] independent model/ckpt names for A/B; baseline path unchanged")


def test_runner_token_and_weighted_one_batch() -> None:
    """One train-style loss call for both modes without launching full training."""
    from easytorch.config import init_cfg

    torch.manual_seed(4)
    history = torch.randn(2, 12, 307, 4)
    future = torch.randn(2, 12, 307, 4)
    real = future[..., :1]

    cfg_w = init_cfg(
        str(generate_temp_config(12, "chain_interleaved_progressive_spatial", 1).relative_to(ROOT)),
        False,
    )
    runner_w = ChainForecastingRunner(cfg_w)
    runner_w.model = ChainForecasting(**dict(cfg_w["MODEL"]["PARAM"]))
    runner_w._rescale_pair = lambda p, t: (p, t)
    runner_w.null_val = 0.0
    out = runner_w.model(history_data=history, future_data=future, train=True, return_all=True)
    loss_w = runner_w._legacy_loss(out, real)
    assert torch.isfinite(loss_w)

    cfg_t = init_cfg(
        str(
            generate_temp_config(12, "chain_interleaved_progressive_spatial_token_loss", 1).relative_to(
                ROOT
            )
        ),
        False,
    )
    runner_t = ChainForecastingRunner(cfg_t)
    runner_t.model = ChainForecasting(**dict(cfg_t["MODEL"]["PARAM"]))
    runner_t._rescale_pair = lambda p, t: (p, t)
    runner_t.null_val = 0.0
    out_t = runner_t.model(history_data=history, future_data=future, train=True, return_all=True)
    loss_t = runner_t._token_mae_loss(out_t, real)
    assert torch.isfinite(loss_t)
    # Token loss must not depend on artificial weights even if present on object.
    runner_t.chain_loss_weights = [100.0, 100.0, 100.0]
    loss_t2 = runner_t._token_mae_loss(out_t, real)
    assert torch.allclose(loss_t, loss_t2)
    print("[ok] weighted/token loss one-batch finite; token ignores weights")


def main() -> int:
    test_original_variant_import()
    test_generated_configs_isolation()
    test_router_zero_init_matches_baseline()
    test_router_finite_grads_all_stages()
    test_token_loss_ignores_weights_and_matches_manual()
    test_prediction_shapes_and_one_batch()
    test_runner_token_and_weighted_one_batch()
    print("\nAll interleaved router / token-loss checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
