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
    route_ce_loss,
)
from scripts.budget_f2f_synth_kwargs import synthetic_budget_f2f_kwargs

SYNTHETIC_MEAN = 100.0
SYNTHETIC_STD = 20.0


def synthetic_rescale_pair(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        pred * SYNTHETIC_STD + SYNTHETIC_MEAN,
        target * SYNTHETIC_STD + SYNTHETIC_MEAN,
    )


def _has_nonzero_grad(module_or_params, prefix: str | None = None) -> bool:
    if hasattr(module_or_params, "named_parameters"):
        iterator = module_or_params.named_parameters()
    else:
        iterator = module_or_params
    for n, p in iterator:
        if prefix is not None and not n.startswith(prefix):
            continue
        if p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0:
            return True
    return False


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
    failures: list[str] = []

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
        out["chain_preds"],
        out["chain_resolutions"],
        y,
        null_val=0.0,
        rescale_pair=synthetic_rescale_pair,
    )
    if not (loss.requires_grad and loss.ndim == 0 and torch.isfinite(loss)):
        failures.append("forced loss must be finite scalar with requires_grad")
    m.zero_grad(set_to_none=True)
    loss.backward()
    _grad_report(m, "forced_[3,6,12]_baseline_rawscale")

    # Three executed KASATemporalStep modules
    for i in range(3):
        step = m.backbone.temporal_steps[i]
        if not _has_nonzero_grad(step):
            failures.append(f"temporal_steps[{i}] missing finite nonzero grad")
    if m.backbone.progressive_spatial_modules is not None:
        for i, spat in enumerate(m.backbone.progressive_spatial_modules):
            if not _has_nonzero_grad(spat):
                failures.append(f"progressive_spatial_modules[{i}] missing grad")
    adapter = getattr(m.backbone, "forecast_state_adapter", None)
    if adapter is not None and not _has_nonzero_grad(adapter):
        failures.append("condition adapter missing finite nonzero grad")

    plan_g = _has_nonzero_grad(m, prefix="planner.")
    print("planner_grad_in_forced_mode:", plan_g)
    # Planner unused in forced mode: zero grad is allowed / expected.

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
        null_val=0.0,
        rescale_pair=synthetic_rescale_pair,
    )
    loss2 = parts["loss"]
    if not (loss2.requires_grad and loss2.ndim == 0 and torch.isfinite(loss2)):
        failures.append("imitation loss must be finite scalar with requires_grad")
    # route CE device must match final_pred
    l_ce = route_ce_loss(out2["route_logits"], oracle, reference=out2["pred"])
    if l_ce.device != out2["pred"].device:
        failures.append(
            f"route CE device {l_ce.device} != final_pred device {out2['pred'].device}"
        )
    if loss2.device != out2["pred"].device:
        failures.append("dynamic_fair total loss device mismatch vs final_pred")

    m2.zero_grad(set_to_none=True)
    loss2.backward()
    plan_g2 = _has_nonzero_grad(m2, prefix="planner.")
    print("planner_grad_with_imitation:", plan_g2)
    if not plan_g2:
        failures.append("planner route head must receive gradient under imitation")

    # Zero-route CE with reference stays on GPU/CPU of reference
    ref = torch.zeros((), device=out2["pred"].device)
    z = route_ce_loss(None, None, reference=ref)
    if z.device != ref.device:
        failures.append("route_ce_loss(None) ignored reference device")

    if failures:
        print("[FAIL] gradient audit:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("[ok] gradient audit passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
