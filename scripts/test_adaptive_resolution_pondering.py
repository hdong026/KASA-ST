#!/usr/bin/env python3
"""CPU synthetic tests for AdaptiveResolutionPonderingF2FNet (no training)."""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basicts.archs.arch_zoo.ChainForecasting_arch.adaptive_resolution_hierarchy import (
    TemporalResolutionTree,
)
from basicts.archs.arch_zoo.ChainForecasting_arch.adaptive_resolution_loss import (
    dynamic_resolution_total_loss,
    matched_token_mae,
)
from basicts.archs.arch_zoo.ChainForecasting_arch.adaptive_resolution_pondering import (
    AdaptiveResolutionPonderingF2FNet,
)
from basicts.archs.arch_zoo.ChainForecasting_arch.adaptive_resolution_pondering_controller import (
    hard_gumbel_sigmoid,
)


def _synth_adj(n: int) -> np.ndarray:
    adj = np.eye(n, dtype=np.float64)
    for i in range(n - 1):
        adj[i, i + 1] = adj[i + 1, i] = 1.0
    return adj


def _build_model(h=6, n=7, p=12, cx=3, cy=1, intensity=0.5):
    return AdaptiveResolutionPonderingF2FNet(
        node_size=n,
        input_len=p,
        output_len=h,
        input_dim=cx,
        output_dim=cy,
        adj_mx=_synth_adj(n),
        thinking_intensity=intensity,
        controller_hidden_dim=32,
        forecast_cell_hidden_dim=32,
        dataset_name="synth_cpu_test",
        hierarchy_cache_dir=None,
    )


def test_shapes_and_matched_target() -> None:
    b, p, h, n, cx, cy = 2, 12, 6, 7, 3, 1
    model = _build_model(h, n, p, cx, cy)
    x = torch.randn(b, p, n, cx)
    y = torch.randn(b, h, n, cy)
    out = model(history_data=x, future_data=y, train=True, return_all=True)
    pred = out["pred"]
    assert pred.shape == (b, h, n, cy)
    for i, cf in enumerate(out["matched_preds"]):
        mt = out["matched_targets"][i]
        assert cf.shape == mt.shape
        assert cf.shape[0] == b
    print("[ok] final [B,H,N,Cy]; matched target shapes align")


def test_adapter_only_modifies_condition() -> None:
    model = _build_model()
    # Force nonzero adapter so condition can move while supervised stays fixed
    with torch.no_grad():
        model.condition_adapter.mlp[-1].weight.fill_(0.5)
        model.condition_adapter.mlp[-1].bias.fill_(0.1)
    x = torch.randn(2, 12, 7, 3)
    y = torch.randn(2, 6, 7, 1)
    out = model(
        history_data=x,
        future_data=y,
        train=True,
        return_all=True,
        return_intermediates=True,
    )
    for step in out["intermediates"]:
        assert torch.allclose(
            step["coarse_supervised"], step["coarse_after_adapter_check"]
        )
    # After step 0, condition is adapter output used for next alignment;
    # within a step, condition tensor may equal supervised at zero-init, so we
    # unit-test the adapter module directly as well.
    cur = torch.ones(2, 2, 2, 1)
    prev = torch.zeros(2, 2, 2, 1)
    meta = torch.zeros(2, 2, 2, 5)
    cond = model.condition_adapter(cur, prev, meta)
    assert not torch.allclose(cond, cur)
    assert torch.allclose(cur, torch.ones_like(cur))  # input unchanged
    print("[ok] condition adapter modifies condition only; supervised unchanged")


def test_hard_split_and_gradients() -> None:
    logits = torch.randn(4, 5, requires_grad=True)
    hard = hard_gumbel_sigmoid(logits, tau=1.0, hard=True)
    assert set(hard.detach().unique().tolist()).issubset({0.0, 1.0})
    (hard.sum()).backward()
    assert logits.grad is not None and logits.grad.abs().sum() > 0

    model = _build_model()
    model.train()
    x = torch.randn(2, 12, 7, 3)
    y = torch.randn(2, 6, 7, 1)
    out = model(history_data=x, future_data=y, train=True, return_all=True)
    loss_pack = dynamic_resolution_total_loss(
        matched_preds=out["matched_preds"],
        matched_targets=out["matched_targets"],
        matched_masks=out["matched_masks"],
        full_candidates=out["full_candidates"],
        halt_weights=out["halt_weights"],
        full_target=y,
        expected_cost=out["expected_cost"],
        budget=out["target_budget"],
        dual=out["dual"],
    )
    loss = loss_pack["loss"]
    loss.backward()

    halt_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for n, p in model.named_parameters()
        if "halt_mlp" in n and p.requires_grad
    )
    split_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for n, p in model.named_parameters()
        if ("temporal_mlp" in n or "spatial_mlp" in n) and p.requires_grad
    )
    def _finite_grad(name_substr: str) -> bool:
        ok = False
        for n, p in model.named_parameters():
            if name_substr not in n or not p.requires_grad or p.grad is None:
                continue
            g = p.grad
            if torch.isfinite(g).all() and g.abs().sum() > 0:
                ok = True
        return ok

    cell_grad = _finite_grad("forecast_cell")
    assert cell_grad, "forecast cell must receive finite nonzero gradient"
    assert halt_grad and _finite_grad("halt_mlp"), "halt head must receive gradient"
    assert split_grad and (
        _finite_grad("temporal_mlp") or _finite_grad("spatial_mlp")
    ), "split heads must receive gradient"
    print("[ok] hard 0/1 splits; controller/cell gradients nonzero")


def test_sample_wise_frontiers_and_halt() -> None:
    model = _build_model(intensity=0.9)
    model.eval()
    x = torch.randn(2, 12, 7, 3)
    y = torch.randn(2, 6, 7, 1)
    out = model(
        history_data=x,
        future_data=y,
        train=False,
        return_all=True,
        return_intermediates=True,
    )
    inter = out["intermediates"]
    assert len(inter) >= 1
    halt_steps = out["halt_step"]
    assert halt_steps is not None and halt_steps.shape[0] == 2
    # Frontières evolve for at least one sample unless immediate full refine
    t0 = inter[0]["active_temporal_count"]
    s0 = inter[0]["active_spatial_count"]
    assert torch.all(t0 >= 1) and torch.all(s0 >= 1)
    print(
        "[ok] sample-wise intermediates; halt_step=",
        halt_steps.tolist(),
        "t0=",
        t0.tolist(),
        "s0=",
        s0.tolist(),
    )


def test_thinking_intensity_budget_not_schedule() -> None:
    low = _build_model(intensity=0.1)
    high = _build_model(intensity=0.9)
    assert low.total_budget < high.total_budget
    src = (
        ROOT
        / "basicts/archs/arch_zoo/ChainForecasting_arch/adaptive_resolution_pondering.py"
    ).read_text(encoding="utf-8")
    assert "[3, 6, 12]" not in src and "[3,6,12]" not in src
    # Forbidden schedules must not be consumed as runtime resolution controls
    assert not hasattr(low, "graph_resolution_capacities")
    assert low.spatial_tree.num_leaves == 7
    print(
        "[ok] thinking_intensity maps to budget only;",
        float(low.total_budget),
        "->",
        float(high.total_budget),
    )


def test_padding_ignored_in_loss() -> None:
    b, t, s, c = 2, 4, 4, 1
    pred = torch.ones(b, t, s, c)
    target = torch.zeros(b, t, s, c)
    mask = torch.zeros(b, t, s)
    mask[:, :2, :2] = 1.0
    loss = matched_token_mae([pred], [target], [mask])
    assert torch.allclose(loss, torch.tensor(1.0), atol=1e-5)
    print("[ok] padding ignored in matched token loss")


def test_no_formal_f2f_overwrite() -> None:
    run_path = ROOT / "scripts/run_chain_forecasting_horizon.py"
    text = run_path.read_text(encoding="utf-8")
    assert "chain_adaptive_resolution_pondering_condition_adapter_token_loss" in text
    assert "AdaptiveResolutionPonderingF2FNet" in text
    assert "chain_interleaved_progressive_spatial_state_adapter_fixed_token_loss" in text
    # Formal control still uses ChainForecasting path (is_chain True)
    tree = ast.parse(text)
    found_formal = False
    found_new = False
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "VARIANT_SPECS":
                    # Walk dict keys in source text instead
                    pass
    found_formal = (
        '"chain_interleaved_progressive_spatial_state_adapter_fixed_token_loss"'
        in text
    )
    found_new = (
        '"chain_adaptive_resolution_pondering_condition_adapter_token_loss"' in text
    )
    assert found_formal and found_new
    # Distinct checkpoint namespace via variant string in ckpt path template
    assert "{variant}_seed{seed}" in text or 'f"{variant}_seed{seed}"' in text
    print("[ok] formal F2FNet variant preserved; new variant registered separately")


def test_max_ponder_from_tree_depth() -> None:
    h = 6
    model = _build_model(h=h)
    tdepth = TemporalResolutionTree(h).depth
    assert model.max_ponder_steps == tdepth + model.spatial_tree.depth + 2
    print("[ok] max_ponder_steps = t_depth + s_depth + 2 =", model.max_ponder_steps)


def test_different_sample_frontiers() -> None:
    model = _build_model(intensity=0.7)
    model.train()
    x = torch.randn(4, 12, 7, 3)
    y = torch.randn(4, 6, 7, 1)
    out = model(history_data=x, future_data=y, train=True, return_all=True)
    # Collect frontier id tuples across samples at last recorded step
    last = out["intermediates"][-1]
    t_fronts = [tuple(ids) for ids in last["temporal_frontier_ids"]]
    s_fronts = [tuple(ids) for ids in last["spatial_frontier_ids"]]
    # Not required that they differ every seed, but with B=4 stochastic training
    # splits they often do; assert at least valid frontiers.
    assert all(len(f) >= 1 for f in t_fronts)
    assert all(len(f) >= 1 for f in s_fronts)
    print(
        "[ok] per-sample frontiers recorded; unique_t=",
        len(set(t_fronts)),
        "unique_s=",
        len(set(s_fronts)),
        "halt_steps=",
        out["halt_step"].tolist(),
    )


def main() -> int:
    test_shapes_and_matched_target()
    test_adapter_only_modifies_condition()
    test_hard_split_and_gradients()
    test_sample_wise_frontiers_and_halt()
    test_thinking_intensity_budget_not_schedule()
    test_padding_ignored_in_loss()
    test_no_formal_f2f_overwrite()
    test_max_ponder_from_tree_depth()
    test_different_sample_frontiers()
    print("[ok] all pondering tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
