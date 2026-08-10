"""Budget-constrained tolerance decision on refinement route scores."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from basicts.archs.arch_zoo.ChainForecasting_arch.budget_route_utils import (
    budget_from_intensity,
    budgets_from_intensity_tensor,
)
from basicts.archs.arch_zoo.ChainForecasting_arch.route_quality_decision import (
    feasible_mask_from_budget,
)


def select_routes_from_scores(
    route_scores: torch.Tensor,
    route_costs: torch.Tensor,
    eta: float | torch.Tensor,
    *,
    delta_abs: float = 0.05,
    eps: float = 1e-8,
) -> dict[str, Any]:
    """Cheapest-near-best on scores (higher score = better forecast gain).

    near_best = {r in F | score_r >= S_best - delta_abs}
    selected = argmin cost among near_best (tie-break: higher score)
    """
    if route_scores.ndim != 2:
        raise ValueError(f"route_scores must be [B,R], got {tuple(route_scores.shape)}")
    b, r = route_scores.shape
    device = route_scores.device
    dtype = route_scores.dtype
    costs = route_costs.to(device=device, dtype=dtype).reshape(-1)
    if costs.numel() != r:
        raise ValueError(f"route_costs size {costs.numel()} != R={r}")

    if torch.is_tensor(eta):
        etas = eta.to(device=device, dtype=dtype).reshape(-1)
        if etas.numel() == 1:
            etas = etas.expand(b)
        elif etas.numel() != b:
            raise ValueError(f"eta batch {etas.numel()} != B={b}")
        budget = budgets_from_intensity_tensor(etas, costs.detach().cpu().tolist()).to(
            device=device, dtype=dtype
        )
    else:
        bval = budget_from_intensity(float(eta), costs.detach().cpu().tolist())
        budget = torch.full((b,), bval, device=device, dtype=dtype)

    feas = feasible_mask_from_budget(costs, budget, eps=eps)
    neg_inf = route_scores.new_tensor(float("-inf"))
    masked = torch.where(feas, route_scores, neg_inf)
    best, _ = masked.max(dim=-1)
    near = feas & (route_scores >= (best.unsqueeze(-1) - float(delta_abs) - eps))
    # Prefer cheaper; among equal costs prefer higher score.
    big = route_scores.new_tensor(1e9)
    score = torch.where(
        near,
        costs.unsqueeze(0).expand(b, r) * 1e3 - route_scores,
        big,
    )
    selected = score.argmin(dim=-1)
    return {
        "selected_route_id": selected,
        "selected_cost": costs.gather(0, selected),
        "feasible_mask": feas,
        "near_best_mask": near,
        "budget": budget,
        "best_score": best,
        "route_scores": route_scores,
    }


def select_batch_routes_from_scores(
    route_scores: torch.Tensor,
    route_costs: torch.Tensor,
    eta: float | torch.Tensor,
    **kwargs,
) -> dict[str, Any]:
    mean_scores = route_scores.mean(dim=0, keepdim=True)
    if torch.is_tensor(eta):
        eta_use = eta.to(route_scores.dtype).reshape(-1).mean()
    else:
        eta_use = float(eta)
    decision = select_routes_from_scores(mean_scores, route_costs, eta_use, **kwargs)
    b = route_scores.shape[0]
    rid = int(decision["selected_route_id"][0].item())
    device = route_scores.device
    costs = route_costs.to(device=device, dtype=route_scores.dtype).reshape(-1)
    return {
        "selected_route_id": torch.full((b,), rid, device=device, dtype=torch.long),
        "selected_cost": costs[rid].expand(b),
        "feasible_mask": decision["feasible_mask"].expand(b, -1),
        "near_best_mask": decision["near_best_mask"].expand(b, -1),
        "budget": decision["budget"][0].expand(b),
        "best_score": decision["best_score"].expand(b),
        "route_scores": route_scores,
        "batch_route_id": rid,
        "batch_mean_scores": mean_scores.squeeze(0),
    }
