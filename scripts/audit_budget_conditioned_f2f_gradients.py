#!/usr/bin/env python3
"""Single synthetic forward/backward gradient audit (no optimizer.step)."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basicts.archs import BudgetConditionedAdaptiveF2FNet
from basicts.archs.arch_zoo.ChainForecasting_arch.budget_conditioned_f2f_loss import (
    baseline_compatible_token_mae,
    dynamic_fair_total_loss,
)
from scripts.budget_f2f_synth_kwargs import synthetic_budget_f2f_kwargs


def _grad_report(model, tag: str) -> None:
    rows = []
    for n, p in model.named_parameters():
        if p.grad is None:
            gsum = 0.0
            finite = True
        else:
            gsum = float(p.grad.abs().sum())
            finite = bool(torch.isfinite(p.grad).all())
        if gsum > 0 and finite:
            rows.append((n, gsum))
    print(f"=== {tag}: nonzero finite grads ({len(rows)}) ===")
    for n, g in rows[:30]:
        print(f"  {g:.6f}  {n}")
    if len(rows) > 30:
        print(f"  ... +{len(rows)-30} more")


def main() -> int:
    h, n, p, b = 12, 7, 12, 2
    x = torch.randn(b, p, n, 4)
    y = torch.randn(b, h, n, 1)

    # Forced full route: executed KASA stages should get grads
    m = BudgetConditionedAdaptiveF2FNet(
        **synthetic_budget_f2f_kwargs(
            forced_route=[3, 6, 12],
            route_selection_mode="forced",
            loss_mode="baseline_compatible",
        )
    )
    m.train()
    out = m(history_data=x, future_data=y, train=True, return_all=True)
    loss = baseline_compatible_token_mae(
        out["chain_preds"], out["chain_resolutions"], y
    )
    loss.backward()
    _grad_report(m, "forced_[3,6,12]_baseline")
    # Planner unused → may have zero grad
    plan_g = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for n, p in m.named_parameters()
        if n.startswith("planner.")
    )
    print("planner_grad_in_forced_mode:", plan_g)

    # Planner imitation: CE on route logits
    m2 = BudgetConditionedAdaptiveF2FNet(
        **synthetic_budget_f2f_kwargs(
            route_selection_mode="batch",
            inference_intensity=1.0,
            training_phase="planner",
            loss_mode="dynamic_fair",
        )
    )
    m2.train()
    oracle = torch.tensor([3, 3], dtype=torch.long)  # full route id for default pool
    out2 = m2(history_data=x, future_data=y, train=True, return_all=True, oracle_route_id=oracle)
    parts = dynamic_fair_total_loss(
        final_pred=out2["pred"],
        full_target=y,
        chain_preds=out2["chain_preds"],
        chain_resolutions=out2["chain_resolutions"],
        route_logits=out2["route_logits"],
        oracle_route_id=oracle,
        selected_cost=out2["selected_cost"],
        budget=out2["budget"],
        lambda_imitation=1.0,
    )
    m2.zero_grad(set_to_none=True)
    parts["loss"].backward()
    plan_g2 = any(
        p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
        for n, p in m2.named_parameters()
        if n.startswith("planner.")
    )
    print("planner_grad_with_imitation:", plan_g2)
    assert plan_g2, "planner route head must receive gradient under imitation"
    print("[ok] gradient audit passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
