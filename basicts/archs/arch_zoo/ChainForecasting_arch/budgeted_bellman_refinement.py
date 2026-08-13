"""Budgeted Bellman Forecast Refinement — finite-horizon Budgeted MDP utilities.

This is Plan B (Bellman), independent of Plan B-v1/v2 policy-gradient code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

from basicts.archs.arch_zoo.ChainForecasting_arch.budget_route_utils import (
    budget_from_intensity,
    default_candidate_routes,
    normalized_static_costs,
)


EPS_BUDGET = 1e-8


@dataclass(frozen=True)
class StageSemantics:
    """Semantic stage sizes for horizon H (must be divisible by 4)."""

    horizon: int
    q: int
    m: int
    f: int

    @classmethod
    def from_horizon(cls, horizon: int) -> "StageSemantics":
        h = int(horizon)
        if h % 4 != 0:
            raise ValueError(f"horizon must be divisible by 4, got {h}")
        return cls(horizon=h, q=h // 4, m=h // 2, f=h)

    def terminal_routes(self) -> dict[str, list[int]]:
        return {
            "D": [self.f],
            "M": [self.m, self.f],
            "Q": [self.q, self.f],
            "F": [self.q, self.m, self.f],
        }


@dataclass(frozen=True)
class AdditiveStageCosts:
    c_f: float
    c_m: float
    c_q: float
    route_costs: dict[str, float]
    C_min: float
    C_max: float

    @property
    def c_qmf(self) -> float:
        return float(self.c_q + self.c_m + self.c_f)


def derive_additive_stage_costs(
    horizon: int = 12,
    routes: Sequence[Sequence[int]] | None = None,
    route_costs: Sequence[float] | None = None,
    *,
    tol: float = 1e-6,
) -> AdditiveStageCosts:
    """Derive incremental stage costs from whole-route costs.

    Raises RuntimeError with ROUTE_COST_NOT_ADDITIVE if additivity fails.
    """
    sem = StageSemantics.from_horizon(horizon)
    if routes is None:
        routes = default_candidate_routes(horizon)
    if route_costs is None:
        route_costs = normalized_static_costs(list(routes), horizon)

    cmap: dict[tuple[int, ...], float] = {}
    for r, c in zip(routes, route_costs):
        cmap[tuple(int(x) for x in r)] = float(c)

    key_d = tuple(sem.terminal_routes()["D"])
    key_m = tuple(sem.terminal_routes()["M"])
    key_q = tuple(sem.terminal_routes()["Q"])
    key_f = tuple(sem.terminal_routes()["F"])
    for k in (key_d, key_m, key_q, key_f):
        if k not in cmap:
            raise KeyError(f"missing route cost for {list(k)}")

    c_f = cmap[key_d]
    c_m = cmap[key_m] - cmap[key_d]
    c_q = cmap[key_q] - cmap[key_d]
    summed = c_q + c_m + c_f
    if abs(summed - cmap[key_f]) > tol:
        raise RuntimeError(
            "ROUTE_COST_NOT_ADDITIVE: "
            f"c_q+c_m+c_f={summed} vs C[q,m,f]={cmap[key_f]}"
        )
    if not (c_q > 0 and c_m > 0 and c_f > 0):
        raise RuntimeError(
            f"ROUTE_COST_NOT_ADDITIVE: non-positive stage costs "
            f"c_q={c_q}, c_m={c_m}, c_f={c_f}"
        )

    named = {
        "D": cmap[key_d],
        "M": cmap[key_m],
        "Q": cmap[key_q],
        "F": cmap[key_f],
    }
    return AdditiveStageCosts(
        c_f=float(c_f),
        c_m=float(c_m),
        c_q=float(c_q),
        route_costs=named,
        C_min=float(min(named.values())),
        C_max=float(max(named.values())),
    )


def intensity_to_budget(eta: float, costs: AdditiveStageCosts) -> float:
    return float(costs.C_min + float(eta) * (costs.C_max - costs.C_min))


class BudgetedRefinementMDP:
    """Finite-horizon budgeted MDP over F2F refinement DAG."""

    def __init__(self, horizon: int = 12, stage_costs: AdditiveStageCosts | None = None):
        self.sem = StageSemantics.from_horizon(horizon)
        self.costs = stage_costs or derive_additive_stage_costs(horizon)
        self.horizon = int(horizon)

    def budget(self, eta: float) -> float:
        return intensity_to_budget(eta, self.costs)

    def min_finish_from_s0_after(self, action: str) -> float:
        """Minimum remaining finish cost after taking s0 action."""
        if action == "f":
            return 0.0
        if action == "m":
            return float(self.costs.c_f)
        if action == "q":
            # after q, can finish with f (cheaper than m then f)
            return float(self.costs.c_f)
        raise ValueError(action)

    def min_finish_from_sq_after(self, action: str) -> float:
        if action == "f":
            return 0.0
        if action == "m":
            return float(self.costs.c_f)
        raise ValueError(action)

    def action_cost_s0(self, action: str) -> float:
        return {"f": self.costs.c_f, "m": self.costs.c_m, "q": self.costs.c_q}[action]

    def action_cost_sq(self, action: str) -> float:
        return {"f": self.costs.c_f, "m": self.costs.c_m}[action]

    def s0_action_feasible(self, action: str, budget: float, *, eps: float = EPS_BUDGET) -> bool:
        c = self.action_cost_s0(action)
        return (c + self.min_finish_from_s0_after(action)) <= float(budget) + eps

    def sq_action_feasible(self, action: str, budget_remaining: float, *, eps: float = EPS_BUDGET) -> bool:
        c = self.action_cost_sq(action)
        return (c + self.min_finish_from_sq_after(action)) <= float(budget_remaining) + eps

    def s0_mask(self, budget: float) -> dict[str, bool]:
        return {a: self.s0_action_feasible(a, budget) for a in ("f", "m", "q")}

    def sq_mask(self, budget_remaining: float) -> dict[str, bool]:
        return {a: self.sq_action_feasible(a, budget_remaining) for a in ("f", "m")}

    def terminal_route_feasible(self, name: str, budget: float, *, eps: float = EPS_BUDGET) -> bool:
        """Exact terminal-route feasibility under recursive finish costs."""
        C = self.costs.route_costs[name]
        return float(C) <= float(budget) + eps

    def feasible_terminal_routes(self, budget: float) -> list[str]:
        return [n for n in ("D", "M", "Q", "F") if self.terminal_route_feasible(n, budget)]

    def feasible_terminal_routes_for_eta(self, eta: float) -> list[str]:
        return self.feasible_terminal_routes(self.budget(eta))

    def unique_nontrivial_budget_regimes(
        self, etas: Sequence[float] | None = None
    ) -> list[dict[str, Any]]:
        """Unique nontrivial feasible-terminal-route sets (exclude single-route)."""
        if etas is None:
            etas = [0.0, 0.25, 0.5, 0.75, 1.0]
        seen: dict[frozenset[str], dict[str, Any]] = {}
        for eta in etas:
            B = self.budget(float(eta))
            feas = frozenset(self.feasible_terminal_routes(B))
            if len(feas) <= 1:
                continue
            if feas not in seen:
                seen[feas] = {
                    "eta_example": float(eta),
                    "budget": float(B),
                    "feasible_terminals": sorted(feas),
                    "s0_mask": self.s0_mask(B),
                    "sq_mask_after_q": self.sq_mask(B - self.costs.c_q),
                }
        # stable order by budget
        return sorted(seen.values(), key=lambda d: d["budget"])


def centered_terminal_returns(
    losses_DMQF: torch.Tensor,
) -> torch.Tensor:
    """Baseline-centered returns preserving argmin(L) == argmax(g).

    losses: [..., 4] as (L_D, L_M, L_Q, L_F)
    returns g: [..., 4] with g_D=0, g_*=L_D-L_*
    """
    L = losses_DMQF
    L_D = L[..., 0:1]
    return torch.cat([torch.zeros_like(L_D), L_D - L[..., 1:]], dim=-1)


def assert_argmax_return_equals_argmin_loss(
    losses: torch.Tensor, returns: torch.Tensor | None = None
) -> bool:
    if returns is None:
        returns = centered_terminal_returns(losses)
    # break ties consistently via first index — check that every argmax g is a true min L
    flat_L = losses.reshape(-1, 4)
    flat_G = returns.reshape(-1, 4)
    for i in range(flat_L.shape[0]):
        L = flat_L[i]
        G = flat_G[i]
        min_l = float(L.min())
        max_g = float(G.max())
        # all argmax G must have L == min_l
        for j in range(4):
            if float(G[j]) >= max_g - 1e-12:
                if abs(float(L[j]) - min_l) > 1e-5:
                    return False
        # all argmin L must have G == max_g
        for j in range(4):
            if abs(float(L[j]) - min_l) <= 1e-5:
                if abs(float(G[j]) - max_g) > 1e-5:
                    return False
    return True


def losses_oracle_order_to_DMQF(
    losses_route_order: torch.Tensor,
    *,
    horizon: int = 12,
) -> torch.Tensor:
    """Map default candidate route order [D,M,Q,F] (already that order for H%4==0)."""
    # default_candidate_routes: [H], [H/2,H], [H/4,H], [H/4,H/2,H] == D,M,Q,F
    _ = horizon
    return losses_route_order


def exact_q1_targets(g: torch.Tensor) -> torch.Tensor:
    """g: [...,4]=(g_D,g_M,g_Q,g_F) -> targets [...,2]=(Q1_f, Q1_m)=(g_Q, g_F)."""
    return torch.stack([g[..., 2], g[..., 3]], dim=-1)


def exact_q0_targets(
    g: torch.Tensor,
    s0_mask: dict[str, bool] | torch.Tensor,
    sq_mask_after_q: dict[str, bool] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact Bellman Q0 targets for one budget regime.

    Returns:
        targets: [..., 3] for (f, m, q)
        valid:   [..., 3] bool mask of which heads are supervised
    """
    # broadcast scalars
    shape = g.shape[:-1]
    device = g.device
    dtype = g.dtype
    t = torch.zeros(*shape, 3, device=device, dtype=dtype)
    v = torch.zeros(*shape, 3, device=device, dtype=torch.bool)

    if isinstance(s0_mask, dict):
        mf, mm, mq = s0_mask["f"], s0_mask["m"], s0_mask["q"]
    else:
        mf = bool(s0_mask[..., 0].item()) if s0_mask.ndim else bool(s0_mask[0])
        mm = bool(s0_mask[..., 1].item()) if s0_mask.ndim else bool(s0_mask[1])
        mq = bool(s0_mask[..., 2].item()) if s0_mask.ndim else bool(s0_mask[2])

    # f always supervised when feasible (always is under nontrivial regimes)
    if mf:
        t[..., 0] = 0.0  # g_D
        v[..., 0] = True
    if mm:
        t[..., 1] = g[..., 1]  # g_M
        v[..., 1] = True
    if mq:
        # child feasibility after paying c_q
        if sq_mask_after_q is None:
            # infer from whether F terminal is feasible: need both f and m children
            # caller should pass sq_mask; default assume only f if F not in picture
            child_f = True
            child_m = False
        else:
            child_f = bool(sq_mask_after_q["f"])
            child_m = bool(sq_mask_after_q["m"])
        if child_f and child_m:
            t[..., 2] = torch.maximum(g[..., 2], g[..., 3])
        elif child_f:
            t[..., 2] = g[..., 2]
        elif child_m:
            t[..., 2] = g[..., 3]
        else:
            # q infeasible to finish — should not supervise
            v[..., 2] = False
            return t, v
        v[..., 2] = True
    return t, v


def global_return_scale_from_gains(
    gains: torch.Tensor,
    *,
    method: str = "iqr",
) -> float:
    """Robust scale from TRAIN OOF gains g_M,g_Q,g_F only (not g_D=0)."""
    x = gains.detach().float().reshape(-1)
    x = x[torch.isfinite(x)]
    if x.numel() == 0:
        return 1.0
    if method == "iqr":
        q75 = torch.quantile(x, 0.75)
        q25 = torch.quantile(x, 0.25)
        iqr = float(q75 - q25)
        scale = iqr / 1.349
        if scale < 1e-8:
            med = torch.median(x)
            mad = torch.median((x - med).abs())
            scale = float(mad) * 1.4826
    else:
        med = torch.median(x)
        mad = torch.median((x - med).abs())
        scale = float(mad) * 1.4826
    return float(max(scale, 1e-8))


def greedy_masked_argmax(q: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """q, mask: [B,A]; returns [B] action indices among feasible."""
    neg = torch.finfo(q.dtype).min / 4
    scored = torch.where(mask.bool(), q, neg)
    return scored.argmax(dim=-1)


def route_name_from_actions(a0: str, a1: str | None = None) -> str:
    if a0 == "f":
        return "D"
    if a0 == "m":
        return "M"
    if a0 == "q":
        if a1 == "f":
            return "Q"
        if a1 == "m":
            return "F"
        raise ValueError("q requires child action")
    raise ValueError(a0)


def semantic_to_route(name: str, horizon: int) -> list[int]:
    return StageSemantics.from_horizon(horizon).terminal_routes()[name]


def cost_audit_dict(horizon: int = 12) -> dict[str, Any]:
    try:
        costs = derive_additive_stage_costs(horizon)
        ok = True
        msg = "OK"
    except RuntimeError as e:
        return {
            "additive": False,
            "error": str(e),
            "ROUTE_COST_NOT_ADDITIVE": True,
        }
    mdp = BudgetedRefinementMDP(horizon, costs)
    feas_by_eta = {
        str(eta): mdp.feasible_terminal_routes_for_eta(eta)
        for eta in (0.0, 0.25, 0.5, 0.75, 1.0)
    }
    return {
        "additive": ok,
        "message": msg,
        "horizon": horizon,
        "c_f": costs.c_f,
        "c_m": costs.c_m,
        "c_q": costs.c_q,
        "route_costs": costs.route_costs,
        "C_min": costs.C_min,
        "C_max": costs.C_max,
        "feasibility_by_eta": feas_by_eta,
        "nontrivial_regimes": mdp.unique_nontrivial_budget_regimes(),
    }
