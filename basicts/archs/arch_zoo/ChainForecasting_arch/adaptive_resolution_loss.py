"""Dynamic token-normalized loss + budget dual utilities for pondering F2FNet."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


def matched_token_mae(
    preds: list[torch.Tensor],
    targets: list[torch.Tensor],
    masks: list[torch.Tensor],
) -> torch.Tensor:
    """Global token-normalized MAE over active coarse tokens (padding ignored)."""
    numerator = None
    denominator = None
    for pred, target, mask in zip(preds, targets, masks):
        # pred/target [B,T,S,C]; mask [B,T,S]
        m = mask.unsqueeze(-1).to(pred.dtype)
        abs_err = (pred - target).abs() * m
        num = abs_err.sum()
        den = m.sum() * pred.shape[-1]
        numerator = num if numerator is None else numerator + num
        denominator = den if denominator is None else denominator + den
    if numerator is None:
        raise ValueError("matched_token_mae requires at least one step")
    return numerator / denominator.clamp_min(1.0)


def halt_weighted_full_mae(
    full_candidates: list[torch.Tensor],
    halt_weights: torch.Tensor,
    full_target: torch.Tensor,
) -> torch.Tensor:
    """
    full_candidates: list length S of [B,H,N,C]
    halt_weights: [B,S]
    full_target: [B,H,N,C]

    Uses stop-gradient decoupling so forecast params and halt head both
    receive finite grads (coupled w(θ)·err(θ) paths can NaN through STE).
    """
    loss = full_target.new_zeros(())
    for si, cand in enumerate(full_candidates):
        w = halt_weights[:, si].view(-1, 1, 1, 1)
        err = (cand - full_target).abs()
        # Forecast path: frozen halt weights
        loss = loss + (w.detach() * err).mean()
        # Halt path: frozen errors
        loss = loss + (w * err.detach()).mean()
    return 0.5 * loss


def compute_step_cost(
    active_t: torch.Tensor,
    active_s: torch.Tensor,
    horizon: int,
    n_nodes: int,
) -> torch.Tensor:
    """Normalized compute proxy per sample."""
    # decoder ~ T*S, relations ~ S^2, overhead constant
    t = active_t.to(torch.float32)
    s = active_s.to(torch.float32)
    h = float(max(horizon, 1))
    n = float(max(n_nodes, 1))
    dec = (t * s) / (h * n)
    rel = (s * s) / (n * n)
    return dec + 0.25 * rel + 0.05


class BudgetDual(nn.Module):
    """Non-negative dual variable for primal-dual budget constraint."""

    def __init__(self, init_value: float = 0.1, lr: float = 0.01):
        super().__init__()
        self.raw = nn.Parameter(torch.tensor(float(init_value)).log())
        self.lr = float(lr)

    @property
    def value(self) -> torch.Tensor:
        return self.raw.exp()

    def dual_ascent_step(self, expected_cost: torch.Tensor, budget: torch.Tensor) -> None:
        """Gradient ascent on dual: maximize λ * (cost - budget)."""
        with torch.no_grad():
            gap = (expected_cost.detach() - budget.detach()).mean()
            # ascent on log-domain parameter
            self.raw.add_(self.lr * gap)


def dynamic_resolution_total_loss(
    matched_preds: list[torch.Tensor],
    matched_targets: list[torch.Tensor],
    matched_masks: list[torch.Tensor],
    full_candidates: list[torch.Tensor],
    halt_weights: torch.Tensor,
    full_target: torch.Tensor,
    expected_cost: torch.Tensor,
    budget: torch.Tensor,
    dual: torch.Tensor,
    halt_entropy: torch.Tensor | None = None,
    entropy_coef: float = 1e-3,
) -> dict[str, torch.Tensor]:
    l_matched = matched_token_mae(matched_preds, matched_targets, matched_masks)
    l_full = halt_weighted_full_mae(full_candidates, halt_weights, full_target)
    l_budget = dual.detach() * (expected_cost.mean() - budget.mean())
    total = l_matched + l_full + l_budget
    if halt_entropy is not None:
        total = total + float(entropy_coef) * (-halt_entropy.mean())
    return {
        "loss": total,
        "l_matched": l_matched.detach(),
        "l_full": l_full.detach(),
        "l_budget": l_budget.detach(),
        "expected_cost": expected_cost.detach().mean(),
        "dual": dual.detach(),
    }
