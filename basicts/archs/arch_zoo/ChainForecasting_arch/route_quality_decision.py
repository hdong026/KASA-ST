"""Budget-conditioned route decision from predicted route qualities.

eta defines computation budget only; it never enters quality estimation.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from basicts.archs.arch_zoo.ChainForecasting_arch.budget_route_utils import (
    budget_from_intensity,
    budgets_from_intensity_tensor,
)


def feasible_mask_from_budget(
    costs: torch.Tensor,
    budget: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return [B, R] feasible mask; cheapest route is always feasible."""
    costs = costs.reshape(-1)
    if budget.ndim == 0:
        budget = budget.reshape(1)
    b = budget.shape[0]
    r = costs.shape[0]
    feas = costs.unsqueeze(0).expand(b, r) <= (budget.reshape(b, 1) + float(eps))
    cheapest = int(costs.argmin().item())
    feas = feas | F.one_hot(
        torch.tensor(cheapest, device=costs.device), num_classes=r
    ).bool().unsqueeze(0)
    return feas


def select_route_ids_from_quality(
    predicted_losses: torch.Tensor,
    route_costs: torch.Tensor,
    eta: float | torch.Tensor,
    *,
    delta_abs: float = 0.05,
    delta_rel: float = 0.0,
    eps: float = 1e-8,
) -> dict[str, Any]:
    """Tolerance-aware cheapest-near-best selection.

    candidate_set = {r in F | L_hat_r <= L_best_hat + delta_abs + delta_rel*|L_best_hat|}
    r* = argmin_{r in candidate_set} C_r  (tie-break: lower predicted loss)
    """
    if predicted_losses.ndim != 2:
        raise ValueError(f"predicted_losses must be [B,R], got {tuple(predicted_losses.shape)}")
    b, r = predicted_losses.shape
    device = predicted_losses.device
    dtype = predicted_losses.dtype
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
    inf = predicted_losses.new_tensor(float("inf"))
    masked_losses = torch.where(feas, predicted_losses, inf)
    best_hat, _ = masked_losses.min(dim=-1)  # [B]
    tol = float(delta_abs) + float(delta_rel) * best_hat.abs()
    near_best = feas & (predicted_losses <= (best_hat.unsqueeze(-1) + tol.unsqueeze(-1) + eps))

    # Prefer cheapest among near-best; among equal costs prefer lower predicted loss.
    big = predicted_losses.new_tensor(1e9)
    score = torch.where(
        near_best,
        costs.unsqueeze(0).expand(b, r) * 1e3 + predicted_losses,
        big,
    )
    selected = score.argmin(dim=-1)
    selected_cost = costs.gather(0, selected)
    return {
        "selected_route_id": selected,
        "selected_cost": selected_cost,
        "feasible_mask": feas,
        "near_best_mask": near_best,
        "budget": budget,
        "best_hat_loss": best_hat,
        "predicted_route_losses": predicted_losses,
    }


def select_batch_route_from_quality(
    predicted_losses: torch.Tensor,
    route_costs: torch.Tensor,
    eta: float | torch.Tensor,
    **kwargs,
) -> dict[str, Any]:
    """Batch routing: mean predicted losses over batch, then one shared decision."""
    mean_losses = predicted_losses.mean(dim=0, keepdim=True)  # [1,R]
    if torch.is_tensor(eta):
        eta_use = eta.to(predicted_losses.dtype).reshape(-1).mean()
    else:
        eta_use = float(eta)
    decision = select_route_ids_from_quality(
        mean_losses, route_costs, eta_use, **kwargs
    )
    b = predicted_losses.shape[0]
    rid = int(decision["selected_route_id"][0].item())
    device = predicted_losses.device
    costs = route_costs.to(device=device, dtype=predicted_losses.dtype).reshape(-1)
    budget_val = decision["budget"][0]
    return {
        "selected_route_id": torch.full((b,), rid, device=device, dtype=torch.long),
        "selected_cost": costs[rid].expand(b),
        "feasible_mask": decision["feasible_mask"].expand(b, -1),
        "near_best_mask": decision["near_best_mask"].expand(b, -1),
        "budget": budget_val.expand(b),
        "best_hat_loss": decision["best_hat_loss"].expand(b),
        "predicted_route_losses": predicted_losses,
        "batch_route_id": rid,
        "batch_mean_predicted_losses": mean_losses.squeeze(0),
    }


def oracle_best_feasible_route(
    true_losses: torch.Tensor,
    route_costs: torch.Tensor,
    feasible_mask: torch.Tensor,
    *,
    delta_abs: float = 0.0,
    delta_rel: float = 0.0,
    eps: float = 1e-8,
) -> dict[str, torch.Tensor]:
    """Oracle selection on true losses under the same tolerance rule."""
    return _oracle_from_mask(
        true_losses, route_costs, feasible_mask, delta_abs, delta_rel, eps
    )


def _oracle_from_mask(
    true_losses: torch.Tensor,
    route_costs: torch.Tensor,
    feasible_mask: torch.Tensor,
    delta_abs: float,
    delta_rel: float,
    eps: float,
) -> dict[str, torch.Tensor]:
    b, r = true_losses.shape
    costs = route_costs.to(device=true_losses.device, dtype=true_losses.dtype).reshape(-1)
    inf = true_losses.new_tensor(float("inf"))
    masked = torch.where(feasible_mask, true_losses, inf)
    best, _ = masked.min(dim=-1)
    tol = float(delta_abs) + float(delta_rel) * best.abs()
    near = feasible_mask & (true_losses <= (best.unsqueeze(-1) + tol.unsqueeze(-1) + eps))
    big = true_losses.new_tensor(1e9)
    score = torch.where(near, costs.unsqueeze(0).expand(b, r) * 1e3 + true_losses, big)
    selected = score.argmin(dim=-1)
    return {
        "oracle_route_id": selected,
        "oracle_best_feasible_loss": best,
        "oracle_selected_loss": true_losses.gather(1, selected.unsqueeze(1)).squeeze(1),
        "near_best_mask": near,
    }


def route_regret(
    selected_true_loss: torch.Tensor,
    best_feasible_true_loss: torch.Tensor,
) -> torch.Tensor:
    return selected_true_loss - best_feasible_true_loss


def check_feasible_monotonicity(
    costs: list[float] | torch.Tensor,
    eta_grid: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Assert F(eta1) ⊆ F(eta2) for eta1 <= eta2 on a dense grid."""
    if torch.is_tensor(costs):
        c = costs.detach().float().cpu()
    else:
        c = torch.tensor(list(costs), dtype=torch.float32)
    if eta_grid is None:
        eta_grid = torch.linspace(0.0, 1.0, 101)
    prev = None
    violations = []
    for eta in eta_grid.tolist():
        bval = budget_from_intensity(float(eta), c.tolist())
        feas = feasible_mask_from_budget(c, torch.tensor([bval])).squeeze(0)
        if prev is not None:
            if not torch.all((~prev) | feas):
                violations.append(
                    {
                        "eta_prev": float(eta) - float(eta_grid[1] - eta_grid[0]),
                        "eta": float(eta),
                        "prev": prev.tolist(),
                        "curr": feas.tolist(),
                        "budget": bval,
                    }
                )
        prev = feas
    return {"ok": len(violations) == 0, "violations": violations}


def build_route_descriptor_tensors(
    routes: list[list[int]],
    horizon: int,
    costs: list[float] | torch.Tensor,
    device=None,
    dtype=torch.float32,
) -> dict[str, torch.Tensor]:
    """Static descriptors for arbitrary candidate routes (no hard-coded R=4)."""
    h = float(horizon)
    r = len(routes)
    max_stages = max((len(rt) for rt in routes), default=1)
    res_norm = torch.zeros(r, max_stages, dtype=dtype)
    stage_mask = torch.zeros(r, max_stages, dtype=torch.bool)
    jumps = torch.zeros(r, max_stages, dtype=dtype)
    stage_count = torch.zeros(r, dtype=dtype)
    first_res = torch.zeros(r, dtype=dtype)
    mean_res = torch.zeros(r, dtype=dtype)
    coverage = torch.zeros(r, dtype=dtype)
    for i, rt in enumerate(routes):
        stage_count[i] = float(len(rt))
        vals = [float(x) / h for x in rt]
        first_res[i] = vals[0]
        mean_res[i] = sum(vals) / len(vals)
        coverage[i] = sum(vals)
        for j, v in enumerate(vals):
            res_norm[i, j] = v
            stage_mask[i, j] = True
            if j == 0:
                jumps[i, j] = v
            else:
                jumps[i, j] = vals[j] - vals[j - 1]
    if torch.is_tensor(costs):
        cost_t = costs.detach().to(dtype=dtype).reshape(-1)
    else:
        cost_t = torch.tensor(list(costs), dtype=dtype)
    if device is not None:
        res_norm = res_norm.to(device)
        stage_mask = stage_mask.to(device)
        jumps = jumps.to(device)
        stage_count = stage_count.to(device)
        first_res = first_res.to(device)
        mean_res = mean_res.to(device)
        coverage = coverage.to(device)
        cost_t = cost_t.to(device)
    return {
        "res_norm": res_norm,
        "stage_mask": stage_mask,
        "jumps": jumps,
        "stage_count": stage_count,
        "first_res": first_res,
        "mean_res": mean_res,
        "coverage": coverage,
        "cost": cost_t,
        "max_stages": max_stages,
    }
