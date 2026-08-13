"""Exact full-information trajectory policy objective for Plan B-v2.

NOT original GRPO/GSPO. Uses exact terminal-route expected utility under
mean-centered (not group-std) advantages, with one global utility scale.
"""

from __future__ import annotations

from typing import Any

import torch

from basicts.archs.arch_zoo.ChainForecasting_arch.budget_route_utils import (
    budget_from_intensity,
)
from basicts.archs.arch_zoo.ChainForecasting_arch.group_relative_refinement_objective import (
    terminal_route_reward,
)
from basicts.archs.arch_zoo.ChainForecasting_arch.route_quality_decision import (
    feasible_mask_from_budget,
)
from basicts.archs.arch_zoo.ChainForecasting_arch.sequential_f2f_environment import (
    A0_DIRECT,
    A0_HALF,
    A0_QUARTER,
    A1_JUMP_FINAL,
    A1_REFINE_HALF,
)


def mean_centered_advantages(
    rewards: torch.Tensor,
    feasible_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """A = R - mean_{feasible}(R). Preserves margin magnitude. No /std."""
    b, r = rewards.shape
    adv = torch.zeros_like(rewards)
    margins = []
    for i in range(b):
        m = feasible_mask[i]
        vals = rewards[i][m]
        if vals.numel() == 0:
            continue
        mu = vals.mean()
        a = vals - mu
        adv[i][m] = a
        if vals.numel() >= 2:
            s = torch.sort(vals, descending=True).values
            margins.append(float((s[0] - s[1]).item()))
    return adv, {
        "mean_top1_top2_margin": float(sum(margins) / max(len(margins), 1)),
        "n_groups": b,
    }


def compute_global_utility_scale(
    centered: torch.Tensor,
    feasible_mask: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> float:
    """ONE global robust scale from TRAIN OOF nontrivial centered utilities.

    Uses MAD around median of nonzero-feasibility centered values:
        median(|A - median(A)|)
    """
    vals = centered[feasible_mask & torch.isfinite(centered)].detach().float().cpu()
    if vals.numel() == 0:
        return 1.0
    med = vals.median()
    mad = (vals - med).abs().median()
    scale = float(mad.item())
    return max(scale, float(eps))


def scale_advantages(
    centered: torch.Tensor,
    utility_scale: float,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    return centered / max(float(utility_scale), float(eps))


def unique_nontrivial_feasibility_regimes(
    route_costs: torch.Tensor | list[float],
    *,
    eta_grid: list[float] | None = None,
) -> list[dict[str, Any]]:
    """Derive unique feasible-route masks with >=2 routes from budget mapping.

    Does not hardcode route IDs — uses costs + budget_from_intensity.
    Trivial single-route regimes (eta~0/.25 for H12) are excluded.
    """
    if torch.is_tensor(route_costs):
        costs = [float(c) for c in route_costs.detach().cpu().tolist()]
        costs_t = route_costs.detach().float().cpu()
    else:
        costs = [float(c) for c in route_costs]
        costs_t = torch.tensor(costs, dtype=torch.float32)
    if eta_grid is None:
        eta_grid = [i / 100.0 for i in range(0, 101)]
    seen: dict[tuple[bool, ...], dict[str, Any]] = {}
    for eta in eta_grid:
        bval = budget_from_intensity(float(eta), costs)
        feas = feasible_mask_from_budget(costs_t, torch.tensor([bval])).squeeze(0)
        key = tuple(bool(x) for x in feas.tolist())
        n = int(feas.sum().item())
        if n < 2:
            continue
        if key not in seen:
            seen[key] = {
                "feasible_mask": feas.clone(),
                "n_feasible": n,
                "example_eta": float(eta),
                "budget": float(bval),
            }
    # Sort by n_feasible then example_eta for stability
    regimes = sorted(seen.values(), key=lambda d: (d["n_feasible"], d["example_eta"]))
    for i, reg in enumerate(regimes):
        reg["regime_id"] = i
        reg["name"] = f"F{i+1}"
    return regimes


def action_masks_from_feasible(
    feasible_routes: torch.Tensor,
    *,
    index_map: dict[str, int],
) -> dict[str, torch.Tensor]:
    """Hard action masks from terminal route feasibility (no eta in features)."""
    f = feasible_routes.bool()
    f_direct = bool(f[index_map["direct"]])
    f_half = bool(f[index_map["half"]])
    f_quarter = bool(f[index_map["quarter"]])
    f_prog = bool(f[index_map["progressive"]])
    m0 = torch.tensor(
        [f_direct, f_half, f_quarter or f_prog], dtype=torch.bool, device=f.device
    )
    m1 = torch.tensor([f_quarter, f_prog], dtype=torch.bool, device=f.device)
    return {"mask0": m0, "mask1": m1, "feasible_routes": f}


def exact_terminal_route_probs(
    log0: torch.Tensor,
    log1: torch.Tensor,
    mask0: torch.Tensor,
    mask1: torch.Tensor,
    *,
    index_map: dict[str, int],
    n_routes: int = 4,
) -> torch.Tensor:
    """Exact terminal probs without QUARTER double-counting.

    p([H])           = pi0(DIRECT)
    p([H/2,H])       = pi0(HALF)
    p([H/4,H])       = pi0(QUARTER) * pi1(JUMP)
    p([H/4,H/2,H])   = pi0(QUARTER) * pi1(REFINE)
    """
    b = log0.shape[0]
    device = log0.device
    p0 = log0.exp()
    p1 = log1.exp()
    if mask0.ndim == 1:
        mask0 = mask0.unsqueeze(0).expand(b, -1)
    if mask1.ndim == 1:
        mask1 = mask1.unsqueeze(0).expand(b, -1)

    probs = torch.zeros(b, n_routes, device=device, dtype=log0.dtype)
    # DIRECT / HALF
    probs[:, index_map["direct"]] = torch.where(
        mask0[:, A0_DIRECT], p0[:, A0_DIRECT], torch.zeros_like(p0[:, 0])
    )
    probs[:, index_map["half"]] = torch.where(
        mask0[:, A0_HALF], p0[:, A0_HALF], torch.zeros_like(p0[:, 0])
    )
    # QUARTER branch
    pq = torch.where(mask0[:, A0_QUARTER], p0[:, A0_QUARTER], torch.zeros_like(p0[:, 0]))
    jump_ok = mask1[:, A1_JUMP_FINAL]
    refine_ok = mask1[:, A1_REFINE_HALF]
    # If only one a1 allowed, renormalize pi1 over allowed (masked_log_softmax already does)
    probs[:, index_map["quarter"]] = torch.where(
        jump_ok, pq * p1[:, A1_JUMP_FINAL], torch.zeros_like(pq)
    )
    probs[:, index_map["progressive"]] = torch.where(
        refine_ok, pq * p1[:, A1_REFINE_HALF], torch.zeros_like(pq)
    )
    return probs


def exact_expected_utility(
    route_probs: torch.Tensor,
    advantages_scaled: torch.Tensor,
    feasible_mask: torch.Tensor,
) -> torch.Tensor:
    """J = sum_r p(r) * A_scaled(r) over feasible routes; mean over batch."""
    a = torch.where(feasible_mask, advantages_scaled, torch.zeros_like(advantages_scaled))
    p = torch.where(feasible_mask, route_probs, torch.zeros_like(route_probs))
    # renormalize numerically in case of tiny drift
    denom = p.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    p = p / denom
    return (p * a).sum(dim=-1).mean()


def terminal_entropy(
    route_probs: torch.Tensor,
    feasible_mask: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Entropy on feasible terminal support; safe for autograd (no log(0))."""
    p = torch.where(feasible_mask, route_probs, torch.zeros_like(route_probs))
    p = p / p.sum(dim=-1, keepdim=True).clamp_min(eps)
    p = p.clamp_min(eps)
    p = p / p.sum(dim=-1, keepdim=True)
    ent = -(p * p.log()).sum(dim=-1)
    return ent.mean()


def terminal_kl(
    p_ref: torch.Tensor,
    p_cur: torch.Tensor,
    feasible_mask: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """KL(p_ref || p_cur) on terminal distributions (feasible support)."""
    pref = torch.where(feasible_mask, p_ref, torch.zeros_like(p_ref))
    pcur = torch.where(feasible_mask, p_cur, torch.zeros_like(p_cur))
    pref = pref / pref.sum(dim=-1, keepdim=True).clamp_min(eps)
    pcur = pcur / pcur.sum(dim=-1, keepdim=True).clamp_min(eps)
    pref = pref.clamp_min(eps)
    pcur = pcur.clamp_min(eps)
    pref = pref / pref.sum(dim=-1, keepdim=True)
    pcur = pcur / pcur.sum(dim=-1, keepdim=True)
    kl = (pref * (pref.log() - pcur.log())).sum(dim=-1)
    return kl.mean()


def rewards_from_losses(
    route_losses: torch.Tensor,
    route_costs: torch.Tensor,
    feasible_mask: torch.Tensor,
    *,
    delta_abs: float = 0.05,
    lambda_quality: float = 10.0,
    lambda_cost: float = 1.0,
) -> torch.Tensor:
    """Same V1 reward — unchanged for V2-first-run."""
    rew, _ = terminal_route_reward(
        route_losses,
        route_costs,
        feasible_mask,
        delta_abs=delta_abs,
        lambda_quality=lambda_quality,
        lambda_cost=lambda_cost,
    )
    # replace nan with 0 for math safety (masked elsewhere)
    return torch.nan_to_num(rew, nan=0.0)
