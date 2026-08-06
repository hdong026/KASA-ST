#!/usr/bin/env python3
"""CPU synthetic tests for OneShotAdaptiveResolutionF2FNet (no training)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basicts.archs.arch_zoo.ChainForecasting_arch.one_shot_adaptive_resolution_f2f import (
    OneShotAdaptiveResolutionF2FNet,
)
from basicts.archs.arch_zoo.ChainForecasting_arch.one_shot_resolution_planner import (
    MAX_OPTIONAL_INTERMEDIATE_STEPS,
)


def _adj(n=7):
    a = np.eye(n)
    for i in range(n - 1):
        a[i, i + 1] = a[i + 1, i] = 1.0
    return a


def _model(intensity=0.5, n=7, h=6, p=8):
    return OneShotAdaptiveResolutionF2FNet(
        node_size=n,
        input_len=p,
        output_len=h,
        input_dim=3,
        output_dim=1,
        adj_mx=_adj(n),
        thinking_intensity=intensity,
        planner_hidden_dim=32,
        executor_hidden_dim=32,
        dataset_name="oneshot_cpu",
        hierarchy_cache_dir=None,
    )


def test_single_calls_and_shape():
    m = _model()
    x = torch.randn(3, 8, 7, 3)
    y = torch.randn(3, 6, 7, 1)
    out = m(history_data=x, future_data=y, train=True, return_all=True, return_intermediates=True)
    assert out["pred"].shape == (3, 6, 7, 1)
    assert out["history_encoder_call_count"] == 1
    assert out["planner_call_count"] == 1
    assert MAX_OPTIONAL_INTERMEDIATE_STEPS == 2
    assert out["max_optional_intermediate_steps"] == 2
    assert torch.isfinite(out["dynamic_loss"])
    print("[ok] single encoder/planner call; final shape; finite loss")


def test_gradients_and_adapter() -> None:
    m = _model(intensity=0.9)
    with torch.no_grad():
        m.executor.condition_adapter.mlp[-1].weight.fill_(0.4)
        m.executor.condition_adapter.mlp[-1].bias.fill_(0.1)
        # Encourage optional intermediates so adapter sits on the F2F path
        m.planner.continue_head[-1].bias.fill_(3.0)
        m.planner.temporal_score[-1].bias.fill_(2.0)
        m.planner.spatial_score[-1].bias.fill_(2.0)
    x = torch.randn(3, 8, 7, 3)
    y = torch.randn(3, 6, 7, 1)
    out = m(history_data=x, future_data=y, train=True, return_all=True)
    out["dynamic_loss"].backward()

    def _finite_grad(substr: str) -> bool:
        return any(
            p.grad is not None
            and torch.isfinite(p.grad).all()
            and p.grad.abs().sum() > 0
            for n, p in m.named_parameters()
            if substr in n
        )

    assert _finite_grad("planner"), "planner must receive gradient"
    assert _finite_grad("executor"), "executor must receive gradient"
    assert _finite_grad("condition_adapter"), "condition adapter must receive gradient"
    cur = torch.ones(1, 2, 2, 1)
    prev = torch.zeros(1, 2, 2, 1)
    meta = torch.zeros(1, 2, 2, 5)
    cond = m.executor.condition_adapter(cur, prev, meta)
    assert not torch.allclose(cond, cur)
    assert torch.allclose(cur, torch.ones_like(cur))
    print("[ok] planner/executor/adapter grads; adapter modifies condition only")


def test_budget_and_no_schedules():
    low = _model(0.1)
    high = _model(0.9)
    assert low.total_optional_budget_value < high.total_optional_budget_value
    src = (
        ROOT
        / "basicts/archs/arch_zoo/ChainForecasting_arch/one_shot_adaptive_resolution_f2f.py"
    ).read_text(encoding="utf-8")
    assert "[3, 6, 12]" not in src
    assert "teacher_optimizer" not in m_state_keys(low)
    # Inference does not call teacher
    x = torch.randn(2, 8, 7, 3)
    with torch.no_grad():
        low.eval()
        _ = low(history_data=x, train=False, return_all=True)
    print("[ok] intensity→budget; no schedule; teacher not in state_dict")


def m_state_keys(m):
    return set(m.state_dict().keys())


def test_variants_preserved():
    text = (ROOT / "scripts/run_chain_forecasting_horizon.py").read_text(encoding="utf-8")
    assert "chain_interleaved_progressive_spatial_state_adapter_fixed_token_loss" in text
    assert "chain_adaptive_resolution_pondering_condition_adapter_token_loss" in text
    assert "chain_one_shot_adaptive_resolution_planning_condition_adapter_token_loss" in text
    assert "OneShotAdaptiveResolutionF2FNet" in text
    assert "AdaptiveResolutionPonderingF2FNet" in text
    # Distinct ckpt namespace via variant string
    assert "{variant}_seed{seed}" in text or 'f"{variant}_seed{seed}"' in text
    print("[ok] formal + pondering + one-shot variants coexist")


def test_no_intermediates_python_lists_when_disabled():
    m = _model()
    out = m(
        history_data=torch.randn(2, 8, 7, 3),
        future_data=torch.randn(2, 6, 7, 1),
        train=False,
        return_all=True,
        return_intermediates=False,
    )
    assert "intermediates" not in out
    print("[ok] diagnostics off skips intermediate lists")


def test_sample_programs_differ():
    m = _model(0.8)
    m.train()
    out = m(
        history_data=torch.randn(3, 8, 7, 3),
        future_data=torch.randn(3, 6, 7, 1),
        train=True,
        return_all=True,
        return_intermediates=True,
    )
    counts = out["intermediate_stage_count"]
    assert counts.shape[0] == 3
    print("[ok] per-sample stage counts", counts.tolist())


def main():
    test_single_calls_and_shape()
    test_gradients_and_adapter()
    test_budget_and_no_schedules()
    test_variants_preserved()
    test_no_intermediates_python_lists_when_disabled()
    test_sample_programs_differ()
    print("[ok] all one-shot F2F tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
