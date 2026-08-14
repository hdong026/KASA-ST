"""Exact expected policy objective over the finite trajectory set (no RL)."""

from __future__ import annotations

from typing import Optional

import torch

from basicts.archs.arch_zoo.ForecastTrajectory_arch.online_trajectory_policy import (
    OnlineTrajectoryPolicy,
)
from basicts.archs.arch_zoo.ForecastTrajectory_arch.trajectory_graph import (
    ForecastTrajectoryGraph,
)


PATH_PROB_ATOL = 1e-6


def token_normalized_mae(
    preds: list[torch.Tensor],
    targets: list[torch.Tensor],
    null_val: float = 0.0,
) -> torch.Tensor:
    """Sum of abs errors / total supervised tokens (no stage weights)."""
    numer = None
    denom = None
    for pred, target in zip(preds, targets):
        if float(null_val) == 0.0:
            mask = ~torch.isclose(
                target,
                torch.zeros_like(target),
                atol=5e-5,
                rtol=0.0,
            )
        else:
            mask = torch.ones_like(target, dtype=torch.bool)
        mask_f = mask.float()
        err = (pred - target).abs() * mask_f
        n = err.sum()
        d = mask_f.sum()
        numer = n if numer is None else numer + n
        denom = d if denom is None else denom + d
    if numer is None:
        raise ValueError("token_normalized_mae requires at least one pair")
    return numer / denom.clamp_min(1.0)


def masked_log_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    neg = torch.finfo(logits.dtype).min / 4
    return torch.log_softmax(logits.masked_fill(~mask, neg), dim=-1)


def exact_trajectory_probs(
    policy: OnlineTrajectoryPolicy,
    graph: ForecastTrajectoryGraph,
    h_history: torch.Tensor,
    z_by_prefix: dict[tuple[int, ...], Optional[torch.Tensor]],
    lam: torch.Tensor,
    remaining_init_norm: torch.Tensor,
    no_budget: torch.Tensor,
    edge_cost_ms: dict[tuple[int, int], float],
    min_finish_ms: dict[int, float],
    c_dense_ms: float,
    remaining_init_ms: Optional[torch.Tensor],
    extra_per_edge_ms: float = 0.0,
    feasible_tau_mask: Optional[torch.Tensor] = None,
) -> dict:
    """Compute p_φ(τ) for every terminal trajectory via prefix decision products.

    Returns dict with ``probs`` ``[B, n_tau]``, ``path_sum`` ``[B]``, ``valid`` bool.
    """
    trajectories = graph.terminal_trajectories()
    n_tau = len(trajectories)
    batch = h_history.shape[0]
    device = h_history.device
    dtype = h_history.dtype
    H = graph.H
    log_probs = h_history.new_zeros(batch, n_tau)
    feas = torch.ones(batch, n_tau, dtype=torch.bool, device=device)
    if feasible_tau_mask is not None:
        feas = feas & feasible_tau_mask.to(device=device, dtype=torch.bool)

    for ti, tau in enumerate(trajectories):
        s_prev = graph.START
        prefix: tuple[int, ...] = ()
        remaining_ms = None
        if remaining_init_ms is not None:
            remaining_ms = remaining_init_ms.to(device=device, dtype=dtype).reshape(batch)
        remaining_norm = remaining_init_norm.to(device=device, dtype=dtype).reshape(batch)
        logp = h_history.new_zeros(batch)
        for s_next in tau:
            z_cur = z_by_prefix.get(prefix)
            feasible_dest = None
            if remaining_ms is not None and (no_budget < 0.5).any():
                dests = []
                for cand in graph.successors(s_prev):
                    # per-sample feasibility uses batch min remaining? handle vectorized below
                    dests.append(cand)
                feasible_dest = dests
            out = policy(
                h_history=h_history,
                z_current=z_cur,
                s_current=s_prev,
                lam=lam,
                remaining_norm=remaining_norm,
                no_budget=no_budget,
                H=H,
                feasible_dest=None if remaining_ms is None else None,
            )
            # Hard budget mask per sample when budget is active.
            mask = out["mask"]
            if remaining_ms is not None:
                hard = mask.clone()
                for cand, idx in policy.state_to_index.items():
                    if not graph.is_legal_edge(s_prev, cand):
                        hard[:, idx] = False
                        continue
                    need = (
                        float(edge_cost_ms[(s_prev, cand)])
                        + float(extra_per_edge_ms)
                        + float(min_finish_ms[cand])
                    )
                    ok = remaining_ms >= (need - 1e-9)
                    use_hard = no_budget < 0.5
                    hard[:, idx] = torch.where(
                        use_hard, ok & hard[:, idx], hard[:, idx]
                    )
                mask = hard
                neg = torch.finfo(out["logits"].dtype).min / 4
                log_pi = torch.log_softmax(
                    out["logits"].masked_fill(~mask, neg), dim=-1
                )
            else:
                log_pi = out["log_probs"]
                mask = out["mask"]
            idx = policy.state_to_index[int(s_next)]
            step_ok = mask[:, idx]
            feas[:, ti] = feas[:, ti] & step_ok
            logp = logp + log_pi[:, idx]
            # consume edge
            edge_c = float(edge_cost_ms[(s_prev, s_next)]) + float(extra_per_edge_ms)
            if remaining_ms is not None:
                remaining_ms = remaining_ms - edge_c
                remaining_norm = remaining_ms / max(float(c_dense_ms), 1e-6)
            prefix = prefix + (int(s_next),)
            s_prev = int(s_next)
        log_probs[:, ti] = logp

    probs = torch.exp(log_probs)
    path_sum = probs.sum(dim=-1)
    good_rows = torch.isfinite(path_sum) & ((path_sum - 1.0).abs() <= PATH_PROB_ATOL)
    valid = bool(good_rows.all().item()) if batch else True
    return {
        "log_probs": log_probs,
        "probs": probs,
        "feas": feas,
        "path_sum": path_sum,
        "valid": valid,
        "max_abs_sum_err": float((path_sum - 1.0).abs().max().item()) if batch else 0.0,
    }


def exact_policy_loss(
    probs: torch.Tensor,
    feas: torch.Tensor,
    forecast_loss: torch.Tensor,
    cost_norm: torch.Tensor,
    lam: torch.Tensor,
) -> torch.Tensor:
    """L = mean_i sum_τ p(τ) [L(τ) + λ C_norm(τ)] over feasible τ."""
    if cost_norm.ndim == 1:
        cost_norm = cost_norm.view(1, -1)
    obj = forecast_loss + lam.reshape(-1, 1) * cost_norm
    weighted = probs * obj * feas.float()
    return weighted.sum(dim=-1).mean()
