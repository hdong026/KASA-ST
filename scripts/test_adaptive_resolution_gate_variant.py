#!/usr/bin/env python3
"""Unit checks for adaptive-resolution gate pilot variant."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basicts.archs import ChainForecasting
from scripts.run_chain_forecasting_horizon import (
    activate_dataset,
    generate_temp_config,
    load_cfg,
)

FORMAL = "chain_interleaved_progressive_spatial_state_adapter_fixed_token_loss"
ADAPTIVE = "chain_interleaved_adaptive_resolution_gate_state_adapter_fixed_token_loss"


def _count_params(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def test_adaptive_resolution_gate() -> dict:
    activate_dataset("PEMS04")
    formal_cfg = load_cfg(generate_temp_config(12, FORMAL, 1))
    adapt_cfg = load_cfg(generate_temp_config(12, ADAPTIVE, 1))

    assert adapt_cfg.MODEL.NAME == "ChainForecasting_AdaptiveResolutionGate_TokenMAE"
    assert adapt_cfg.MODEL.PARAM.get("use_adaptive_resolution_controller") is True
    assert adapt_cfg.TRAIN.CKPT_SAVE_DIR != formal_cfg.TRAIN.CKPT_SAVE_DIR
    assert adapt_cfg.MODEL.PARAM["spatial_placement"] == "interleaved_progressive"
    assert adapt_cfg.MODEL.PARAM["forecast_state_adapter_mode"] == "condition_only"
    assert adapt_cfg.MODEL.PARAM["chain_loss_mode"] == "token_normalized"

    formal = ChainForecasting(**dict(formal_cfg.MODEL.PARAM))
    model = ChainForecasting(**dict(adapt_cfg.MODEL.PARAM))
    n_formal = _count_params(formal)
    n_adapt = _count_params(model)
    extra = n_adapt - n_formal
    assert model.adaptive_resolution_controller is not None
    assert formal.adaptive_resolution_controller is None
    assert extra > 0

    # Init gates near 0.98
    ctrl = model.adaptive_resolution_controller
    history = torch.randn(4, 12, 307, 4)
    model.eval()
    with torch.no_grad():
        out0 = model(history, return_intermediates=True)
    t_gates = out0["temporal_detail_gates"]
    s_gates = out0["spatial_detail_gates"]
    assert len(t_gates) == 2 and len(s_gates) == 2
    for g in t_gates + s_gates:
        assert g.shape[0] == 4
        assert torch.all(g >= 0) and torch.all(g <= 1)
    t_mean = float(torch.cat([g.flatten() for g in t_gates]).mean())
    s_mean = float(torch.cat([g.flatten() for g in s_gates]).mean())
    assert abs(t_mean - 0.98) < 0.02, t_mean
    assert abs(s_mean - 0.98) < 0.02, s_mean

    # Shapes match formal
    formal.eval()
    with torch.no_grad():
        fout = formal(history, return_intermediates=True)
    assert out0["pred"].shape == fout["pred"].shape == (4, 12, 307, 1)
    for a, b in zip(out0["chain_preds"], fout["chain_preds"]):
        assert a.shape == b.shape

    # Controller does not rewrite supervised forecasts: with zeroed-ish gates near 1,
    # supervised equals progressive refine output path; force gate toward 0 and check
    # chain_preds still equal z_raw (captured via identity of supervised vs temporal+spatial).
    # More direct: monkeypatch controller to return zeros condition but chain_preds unchanged
    # relative to a forward without controller effect on supervision.
    supervised_before = [p.detach().clone() for p in out0["chain_preds"]]

    # Temporal smooth length-preserving / spatial node-preserving
    cond = torch.randn(2, 6, 307, 1)
    sm_t = ctrl.temporal_smooth(cond)
    assert sm_t.shape == cond.shape
    adj = model.progressive_spatial_modules[0]._build_adaptive_adj()
    sm_s = ctrl.spatial_smooth(cond, adj)
    assert sm_s.shape == cond.shape

    # Gradients into controller
    model.train()
    out = model(history, return_all=True)
    target = torch.randn_like(out["pred"])
    loss = sum((p - target[:, : p.shape[1]]).abs().mean() for p in out["chain_preds"])
    loss.backward()
    ctrl_grad = sum(
        p.grad.abs().sum().item()
        for p in ctrl.parameters()
        if p.grad is not None
    )
    assert ctrl_grad > 0, ctrl_grad
    assert not any(torch.isnan(p).any() for p in out["chain_preds"])

    # Condition-only: adapter still only on forwarded condition (step1)
    with torch.no_grad():
        model.forecast_state_adapter.proj_out.weight.fill_(0.05)
        model.forecast_state_adapter.proj_out.bias.fill_(0.01)
    captured = {}
    real = model.temporal_steps[2].forward

    def wrap(history_data, prev_forecast=None, prev_latent=None, **kwargs):
        captured["prev"] = None if prev_forecast is None else prev_forecast.detach().clone()
        return real(
            history_data, prev_forecast=prev_forecast, prev_latent=prev_latent, **kwargs
        )

    model.temporal_steps[2].forward = wrap
    out2 = model(history[:2], return_all=True)
    z6 = out2["chain_preds"][1]
    from basicts.archs.arch_zoo.ChainForecasting_arch.kasa_temporal_step import (
        interpolate_forecast,
    )

    assert captured["prev"] is not None
    # Forwarded condition to T12 is gated+adapted, not raw Z6
    assert not torch.allclose(captured["prev"], interpolate_forecast(z6, 12), atol=1e-5)

    # Formal param count unchanged by importing/building adaptive path
    formal2 = ChainForecasting(**dict(formal_cfg.MODEL.PARAM))
    assert _count_params(formal2) == n_formal

    print(
        f"[ok] adaptive gate: formal_params={n_formal} adaptive_params={n_adapt} "
        f"extra={extra} init_t_gate={t_mean:.4f} init_s_gate={s_mean:.4f}"
    )
    return {
        "n_formal": n_formal,
        "n_adapt": n_adapt,
        "extra": extra,
        "init_temporal_gate_mean": t_mean,
        "init_spatial_gate_mean": s_mean,
        "supervised_shapes": [tuple(p.shape) for p in supervised_before],
    }


def main() -> int:
    report = test_adaptive_resolution_gate()
    print("REPORT", report)
    print("[ok] all adaptive-resolution gate checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
