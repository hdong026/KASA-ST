"""Composite losses for forecast-refinement gain learning."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from basicts.archs.arch_zoo.ChainForecasting_arch.forecast_refinement_routes import (
    route_scores_from_gains,
)


def _smooth_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.smooth_l1_loss(pred, target, reduction="mean")


def absolute_gain_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return _smooth_l1(pred, target)


def across_sample_centered_gain_loss(
    pred: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Center each gain dim across the batch (core anti-prior loss)."""
    pred_c = pred - pred.mean(dim=0, keepdim=True)
    tgt_c = target - target.mean(dim=0, keepdim=True)
    return _smooth_l1(pred_c, tgt_c)


def correlation_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    var_threshold: float = 1e-4,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Mean (1 - Pearson) over gain dims with sufficient true variance."""
    losses = []
    for k in range(pred.shape[-1]):
        x = pred[:, k]
        y = target[:, k]
        y_var = y.var(unbiased=False)
        if float(y_var.item()) < float(var_threshold):
            continue
        x = x - x.mean()
        y = y - y.mean()
        corr = (x * y).sum() / (x.norm() * y.norm() + eps)
        losses.append(1.0 - corr)
    if not losses:
        return pred.new_zeros(())
    return torch.stack(losses).mean()


def pairwise_route_ranking_loss(
    scores_hat: torch.Tensor,
    scores_true: torch.Tensor,
    *,
    rank_ignore_margin: float = 0.02,
    rank_temperature: float = 0.05,
    pair_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Imbalance-aware pairwise ranking on reconstructed route scores."""
    b, r = scores_hat.shape
    if r < 2:
        return scores_hat.new_zeros(())
    si = scores_hat.unsqueeze(-1)
    sj = scores_hat.unsqueeze(-2)
    ti = scores_true.unsqueeze(-1)
    tj = scores_true.unsqueeze(-2)
    d_hat = si - sj
    d_true = ti - tj
    tril = torch.tril(torch.ones(r, r, device=scores_hat.device, dtype=torch.bool), diagonal=-1)
    valid = tril.unsqueeze(0) & (d_true.abs() >= float(rank_ignore_margin))
    if not bool(valid.any()):
        return scores_hat.new_zeros(())
    sign = torch.sign(d_true)
    pair = F.softplus(-sign * d_hat / max(float(rank_temperature), 1e-6))
    mag = d_true.abs()
    mag = mag / (mag[valid].mean().detach() + 1e-6)
    weight = mag
    if pair_weights is not None:
        # pair_weights: [R,R] sign-independent frequency weights
        pw = pair_weights.to(device=scores_hat.device, dtype=scores_hat.dtype)
        weight = weight * pw.unsqueeze(0)
    return (pair * weight)[valid].mean()


def full_route_consistency_loss(
    g3_hat: torch.Tensor,
    g36_hat: torch.Tensor,
    full_true: torch.Tensor,
) -> torch.Tensor:
    return _smooth_l1(g3_hat + g36_hat, full_true)


def refinement_gain_total_loss(
    gains_hat: torch.Tensor,
    gains_true: torch.Tensor,
    *,
    index_map: dict[str, int],
    n_routes: int,
    lambda_abs: float = 0.25,
    lambda_center: float = 1.0,
    lambda_corr: float = 0.5,
    lambda_rank: float = 1.0,
    lambda_full: float = 0.5,
    rank_ignore_margin: float = 0.02,
    rank_temperature: float = 0.05,
    pair_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    l_abs = absolute_gain_loss(gains_hat, gains_true)
    l_center = across_sample_centered_gain_loss(gains_hat, gains_true)
    l_corr = correlation_loss(gains_hat, gains_true)
    scores_hat = route_scores_from_gains(
        gains_hat[:, 0],
        gains_hat[:, 1],
        gains_hat[:, 2],
        index_map=index_map,
        n_routes=n_routes,
    )
    scores_true = route_scores_from_gains(
        gains_true[:, 0],
        gains_true[:, 1],
        gains_true[:, 2],
        index_map=index_map,
        n_routes=n_routes,
    )
    l_rank = pairwise_route_ranking_loss(
        scores_hat,
        scores_true,
        rank_ignore_margin=rank_ignore_margin,
        rank_temperature=rank_temperature,
        pair_weights=pair_weights,
    )
    # full true = g3 + g36 by construction of targets
    full_true = gains_true[:, 0] + gains_true[:, 2]
    l_full = full_route_consistency_loss(gains_hat[:, 0], gains_hat[:, 2], full_true)
    total = (
        float(lambda_abs) * l_abs
        + float(lambda_center) * l_center
        + float(lambda_corr) * l_corr
        + float(lambda_rank) * l_rank
        + float(lambda_full) * l_full
    )
    parts = {
        "L_abs": float(l_abs.detach().item()),
        "L_center": float(l_center.detach().item()),
        "L_corr": float(l_corr.detach().item()),
        "L_rank": float(l_rank.detach().item()),
        "L_full": float(l_full.detach().item()),
        "total": float(total.detach().item()),
    }
    return total, parts


def compute_pair_imbalance_weights(
    scores_true: torch.Tensor,
    *,
    rank_ignore_margin: float = 0.02,
    max_pair_weight: float = 5.0,
    use_sqrt: bool = True,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Inverse-frequency weights over route-pair preference signs."""
    r = scores_true.shape[-1]
    pos = torch.zeros(r, r)
    neg = torch.zeros(r, r)
    for i in range(r):
        for j in range(i):
            d = scores_true[:, i] - scores_true[:, j]
            mask = d.abs() >= float(rank_ignore_margin)
            pos[i, j] = (d[mask] > 0).sum()
            neg[i, j] = (d[mask] < 0).sum()
            pos[j, i] = neg[i, j]
            neg[j, i] = pos[i, j]
    weights = torch.ones(r, r)
    report = {}
    for i in range(r):
        for j in range(i):
            p = float(pos[i, j].item())
            n = float(neg[i, j].item())
            total = p + n
            if total <= 0:
                w = 1.0
            else:
                # weight minority direction higher via harmonic-style factor
                freq_min = min(p, n) / total
                inv = 1.0 / max(freq_min, 1e-3)
                w = inv**0.5 if use_sqrt else inv
                w = min(w, float(max_pair_weight))
            weights[i, j] = w
            weights[j, i] = w
            report[f"{j}<{i}"] = {
                "pos_count": p,
                "neg_count": n,
                "weight": w,
            }
    return weights, report


@torch.no_grad()
def gain_diagnostics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    out: dict[str, float] = {
        "gain_abs_mae": float((pred - target).abs().mean().item()),
    }
    pred_c = pred - pred.mean(dim=0, keepdim=True)
    tgt_c = target - target.mean(dim=0, keepdim=True)
    out["gain_centered_mae"] = float((pred_c - tgt_c).abs().mean().item())
    names = ["g3", "g6", "g36"]
    for k, name in enumerate(names):
        x = pred[:, k]
        y = target[:, k]
        x0 = x - x.mean()
        y0 = y - y.mean()
        denom = x0.norm() * y0.norm() + 1e-6
        out[f"gain_pearson_{name}"] = float(((x0 * y0).sum() / denom).item())
        # Spearman via rank
        rx = x.argsort().argsort().float()
        ry = y.argsort().argsort().float()
        rx = rx - rx.mean()
        ry = ry - ry.mean()
        out[f"gain_spearman_{name}"] = float(
            ((rx * ry).sum() / (rx.norm() * ry.norm() + 1e-6)).item()
        )
        out[f"pred_{name}_mean"] = float(x.mean().item())
        out[f"pred_{name}_std"] = float(x.std(unbiased=False).item())
        out[f"true_{name}_mean"] = float(y.mean().item())
        out[f"true_{name}_std"] = float(y.std(unbiased=False).item())
        if out[f"pred_{name}_std"] < 0.1 * max(out[f"true_{name}_std"], 1e-6):
            out[f"GAIN_COLLAPSE_WARNING_{name}"] = 1.0
    return out
