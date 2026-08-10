"""Loss helpers for budget-conditioned adaptive F2F."""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn.functional as F

from basicts.archs.arch_zoo.ChainForecasting_arch.ChainForecasting_arch import (
    ChainForecasting,
)
from basicts.losses.forecast_state_token_mae import forecast_state_token_mae

RescalePair = Optional[
    Callable[
        [torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor],
    ]
]


def mid_token_mae(
    chain_preds: list[torch.Tensor],
    chain_resolutions: list[int],
    full_target: torch.Tensor,
    null_val: float = 0.0,
    rescale_pair: RescalePair = None,
) -> torch.Tensor:
    """Token MAE over intermediate stages only (exclude final H)."""
    if len(chain_preds) <= 1:
        return full_target.new_zeros(())
    preds = chain_preds[:-1]
    resolutions = chain_resolutions[:-1]
    targets = [ChainForecasting.pool_target(full_target, k) for k in resolutions]
    return forecast_state_token_mae(
        preds,
        targets,
        null_val=null_val,
        rescale_pair=rescale_pair,
    )


def baseline_compatible_token_mae(
    chain_preds: list[torch.Tensor],
    chain_resolutions: list[int],
    full_target: torch.Tensor,
    null_val: float = 0.0,
    rescale_pair: RescalePair = None,
) -> torch.Tensor:
    targets = [ChainForecasting.pool_target(full_target, k) for k in chain_resolutions]
    return forecast_state_token_mae(
        chain_preds,
        targets,
        null_val=null_val,
        rescale_pair=rescale_pair,
    )


def budget_violation(expected_cost: torch.Tensor, budget: torch.Tensor) -> torch.Tensor:
    return F.relu(expected_cost - budget).mean()


def route_ce_loss(
    route_logits: torch.Tensor | None,
    oracle_route_id: torch.Tensor | None,
    reference: torch.Tensor | None = None,
) -> torch.Tensor:
    if route_logits is None or oracle_route_id is None:
        if reference is not None:
            return reference.new_zeros(())
        if route_logits is not None:
            return route_logits.new_zeros(())
        if oracle_route_id is not None:
            return oracle_route_id.new_zeros(()).float()
        return torch.zeros((), dtype=torch.float32)
    labels = oracle_route_id.long().view(-1)
    if labels.numel() == 1 and route_logits.shape[0] > 1:
        labels = labels.expand(route_logits.shape[0])
    elif labels.numel() != route_logits.shape[0]:
        raise ValueError(
            f"oracle_route_id size {labels.numel()} != batch {route_logits.shape[0]}"
        )
    return F.cross_entropy(route_logits, labels)


def planner_imitation_loss(
    route_logits_masked: torch.Tensor | None,
    oracle: torch.Tensor | None,
    expected_cost: torch.Tensor | None,
    budget: torch.Tensor | None,
    lambda_imitation: float = 1.0,
    lambda_budget: float = 0.0,
) -> dict[str, torch.Tensor]:
    ref = expected_cost if expected_cost is not None else route_logits_masked
    l_ce = route_ce_loss(route_logits_masked, oracle, reference=ref)
    if expected_cost is None or budget is None or float(lambda_budget) == 0.0:
        l_bud = l_ce.new_zeros(())
    else:
        l_bud = budget_violation(expected_cost, budget)
    total = float(lambda_imitation) * l_ce + float(lambda_budget) * l_bud
    return {
        "loss": total,
        "L_total": total.detach(),
        "L_route_ce": l_ce.detach(),
        "L_budget": l_bud.detach() if torch.is_tensor(l_bud) else l_bud,
    }


def dynamic_fair_total_loss(
    final_pred: torch.Tensor,
    full_target: torch.Tensor,
    chain_preds: list[torch.Tensor],
    chain_resolutions: list[int],
    route_logits: torch.Tensor | None = None,
    oracle_route_id: torch.Tensor | None = None,
    selected_cost: torch.Tensor | None = None,
    expected_cost: torch.Tensor | None = None,
    budget: torch.Tensor | None = None,
    lambda_mid: float = 1.0,
    lambda_imitation: float = 1.0,
    lambda_budget: float = 0.0,
    null_val: float = 0.0,
    rescale_pair: RescalePair = None,
) -> dict[str, torch.Tensor]:
    final_target = full_target[..., : final_pred.shape[-1]]
    l_final = forecast_state_token_mae(
        [final_pred],
        [final_target],
        null_val=null_val,
        rescale_pair=rescale_pair,
    )
    l_mid = mid_token_mae(
        chain_preds,
        chain_resolutions,
        full_target,
        null_val=null_val,
        rescale_pair=rescale_pair,
    )
    l_ce = route_ce_loss(route_logits, oracle_route_id, reference=final_pred)
    cost_for_budget = expected_cost if expected_cost is not None else selected_cost
    if cost_for_budget is None or budget is None:
        l_bud = final_pred.new_zeros(())
    else:
        l_bud = budget_violation(cost_for_budget, budget)
    total = (
        l_final
        + float(lambda_mid) * l_mid
        + float(lambda_imitation) * l_ce
        + float(lambda_budget) * l_bud
    )
    return {
        "loss": total,
        "L_total": total.detach(),
        "L_final": l_final.detach(),
        "L_mid_token": l_mid.detach() if torch.is_tensor(l_mid) else l_mid,
        "L_route_ce": l_ce.detach() if torch.is_tensor(l_ce) else l_ce,
        "L_budget": l_bud.detach() if torch.is_tensor(l_bud) else l_bud,
    }
