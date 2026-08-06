"""Dynamic token-normalized loss for one-shot adaptive resolution F2F."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from basicts.archs.arch_zoo.ChainForecasting_arch.resolution_program_optimizer import (
    BudgetDualController,
    TeacherProgram,
    planner_imitation_loss_placeholder,
)


def matched_packed_token_mae(
    preds: list[torch.Tensor],
    targets: list[torch.Tensor],
    masks: list[torch.Tensor],
) -> torch.Tensor:
    """Global token-normalized MAE; masks mark valid tokens."""
    num = None
    den = None
    for pred, target, mask in zip(preds, targets, masks):
        if pred is None or target is None or mask is None:
            continue
        m = mask
        if m.ndim == pred.ndim - 1:
            m = m.unsqueeze(-1)
        m = m.to(pred.dtype)
        err = (pred - target).abs() * m
        n = err.sum()
        d = m.sum() * pred.shape[-1]
        num = n if num is None else num + n
        den = d if den is None else den + d
    if num is None:
        return torch.tensor(0.0)
    return num / den.clamp_min(1.0)


def one_shot_resolution_total_loss(
    intermediate_preds: list[torch.Tensor],
    intermediate_targets: list[torch.Tensor],
    intermediate_masks: list[torch.Tensor],
    final_pred: torch.Tensor,
    final_target: torch.Tensor,
    expected_optional_cost: torch.Tensor,
    optional_budget: torch.Tensor,
    dual: torch.Tensor,
    planner_out: dict[str, torch.Tensor] | None = None,
    teacher: TeacherProgram | None = None,
    imitation_coef: float = 0.0,
) -> dict[str, torch.Tensor]:
    l_inter = matched_packed_token_mae(
        intermediate_preds, intermediate_targets, intermediate_masks
    )
    if not torch.is_tensor(l_inter):
        l_inter = final_pred.new_zeros(())
    l_final = (final_pred - final_target).abs().mean()
    l_budget = dual.detach() * (expected_optional_cost.mean() - optional_budget.mean())
    l_imit = final_pred.new_zeros(())
    if planner_out is not None and imitation_coef > 0:
        l_imit = planner_imitation_loss_placeholder(planner_out, teacher)
    total = l_inter + l_final + l_budget + float(imitation_coef) * l_imit
    return {
        "loss": total,
        "l_intermediate": l_inter.detach() if torch.is_tensor(l_inter) else l_inter,
        "l_final": l_final.detach(),
        "l_budget": l_budget.detach(),
        "l_imitation": l_imit.detach() if torch.is_tensor(l_imit) else l_imit,
        "expected_optional_cost": expected_optional_cost.detach().mean(),
        "dual": dual.detach(),
    }


# Alias requested by task naming
DynamicResolutionTokenLoss = one_shot_resolution_total_loss
