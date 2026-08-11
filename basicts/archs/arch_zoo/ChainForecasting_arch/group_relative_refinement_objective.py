"""GRPO-inspired group-relative trajectory objective for Plan B."""

from __future__ import annotations

from typing import Any

import torch


def terminal_route_reward(
    route_losses: torch.Tensor,
    route_costs: torch.Tensor,
    feasible_mask: torch.Tensor,
    *,
    delta_abs: float = 0.05,
    lambda_quality: float = 10.0,
    lambda_cost: float = 1.0,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute R(r) = -λq * q(r) - λc * C(r) for feasible routes.

    route_losses/costs/mask: [B,R]
    returns rewards [B,R] (infeasible filled with nan for safety)
    """
    inf = route_losses.new_tensor(float("inf"))
    masked = torch.where(feasible_mask, route_losses, inf)
    best, _ = masked.min(dim=-1)
    q = torch.clamp(route_losses - best.unsqueeze(-1) - float(delta_abs), min=0.0)
    costs = route_costs.reshape(1, -1).expand_as(route_losses)
    rewards = -float(lambda_quality) * q - float(lambda_cost) * costs
    rewards = torch.where(feasible_mask, rewards, route_losses.new_tensor(float("nan")))
    return rewards, {"best_loss": best, "quality_excess": q}


def group_relative_advantages(
    rewards: torch.Tensor,
    feasible_mask: torch.Tensor,
    *,
    eps: float = 1e-6,
    var_eps: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Per-sample group mean/std over feasible routes only."""
    b, r = rewards.shape
    adv = torch.zeros_like(rewards)
    zero_var = 0
    for i in range(b):
        m = feasible_mask[i]
        vals = rewards[i][m]
        if vals.numel() == 0:
            continue
        mu = vals.mean()
        sigma = vals.std(unbiased=False)
        if float(sigma.item()) < var_eps:
            zero_var += 1
            a = torch.zeros_like(vals)
        else:
            a = (vals - mu) / (sigma + eps)
        adv[i][m] = a
    return adv, {"zero_variance_groups": zero_var}


def clipped_trajectory_objective(
    logp_new: torch.Tensor,
    logp_old: torch.Tensor,
    advantages: torch.Tensor,
    *,
    clip_eps: float = 0.2,
) -> tuple[torch.Tensor, dict[str, float]]:
    """-mean(min(rho A, clip(rho) A)); trajectory-level ratio."""
    ratio = torch.exp(logp_new - logp_old)
    clipped = torch.clamp(ratio, 1.0 - float(clip_eps), 1.0 + float(clip_eps))
    obj1 = ratio * advantages
    obj2 = clipped * advantages
    loss = -torch.min(obj1, obj2).mean()
    return loss, {
        "ratio_mean": float(ratio.mean().item()),
        "ratio_min": float(ratio.min().item()),
        "ratio_max": float(ratio.max().item()),
        "adv_mean": float(advantages.mean().item()),
    }
