"""Composite losses for route-quality estimation (no hard CE as main objective)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def _huber(pred: torch.Tensor, target: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    return F.smooth_l1_loss(pred, target, reduction="mean", beta=float(delta))


def absolute_quality_loss(
    pred: torch.Tensor, target: torch.Tensor, huber_delta: float = 1.0
) -> torch.Tensor:
    return _huber(pred, target, delta=huber_delta)


def centered_quality_loss(
    pred: torch.Tensor, target: torch.Tensor, huber_delta: float = 1.0
) -> torch.Tensor:
    pred_c = pred - pred.mean(dim=-1, keepdim=True)
    tgt_c = target - target.mean(dim=-1, keepdim=True)
    return _huber(pred_c, tgt_c, delta=huber_delta)


def pairwise_ranking_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    rank_ignore_margin: float = 0.02,
    tau_rank: float = 1.0,
) -> torch.Tensor:
    """Softplus pairwise ranking with magnitude-weighted pairs."""
    # pred/target: [B,R]
    b, r = pred.shape
    if r < 2:
        return pred.new_zeros(())
    # pairs (i,j) with i < j
    pi = pred.unsqueeze(-1)  # B,R,1
    pj = pred.unsqueeze(-2)  # B,1,R
    ti = target.unsqueeze(-1)
    tj = target.unsqueeze(-2)
    delta_hat = pj - pi  # L_j - L_i predicted
    delta_true = tj - ti
    tril = torch.tril(torch.ones(r, r, device=pred.device, dtype=torch.bool), diagonal=-1)
    mask = tril.unsqueeze(0).expand(b, r, r)
    abs_true = delta_true.abs()
    valid = mask & (abs_true >= float(rank_ignore_margin))
    if not bool(valid.any()):
        return pred.new_zeros(())
    sign = torch.sign(delta_true)
    # Prefer lower true loss: if L_j > L_i, we want Delta_hat > 0
    pair = F.softplus(-sign * delta_hat / max(float(tau_rank), 1e-6))
    # Weight by how distinguishable the pair is
    weight = abs_true.clamp(min=0.0)
    weight = weight / (weight[valid].mean().detach() + 1e-6)
    return (pair * weight)[valid].mean()


def listwise_ranking_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    """KL(p_true || p_hat) with p = softmax(-L / T)."""
    t = max(float(temperature), 1e-6)
    log_p_hat = F.log_softmax(-pred / t, dim=-1)
    p_true = F.softmax(-target / t, dim=-1)
    return F.kl_div(log_p_hat, p_true, reduction="batchmean")


def route_quality_total_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    lambda_abs: float = 0.25,
    lambda_center: float = 1.0,
    lambda_rank: float = 1.0,
    lambda_list: float = 0.25,
    rank_ignore_margin: float = 0.02,
    rank_temperature: float = 1.0,
    list_temperature: float = 1.0,
    huber_delta: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    l_abs = absolute_quality_loss(pred, target, huber_delta=huber_delta)
    l_center = centered_quality_loss(pred, target, huber_delta=huber_delta)
    l_rank = pairwise_ranking_loss(
        pred,
        target,
        rank_ignore_margin=rank_ignore_margin,
        tau_rank=rank_temperature,
    )
    l_list = listwise_ranking_loss(pred, target, temperature=list_temperature)
    total = (
        float(lambda_abs) * l_abs
        + float(lambda_center) * l_center
        + float(lambda_rank) * l_rank
        + float(lambda_list) * l_list
    )
    parts = {
        "L_abs": float(l_abs.detach().item()),
        "L_center": float(l_center.detach().item()),
        "L_rank": float(l_rank.detach().item()),
        "L_list": float(l_list.detach().item()),
        "total": float(total.detach().item()),
        "lambda_abs": float(lambda_abs),
        "lambda_center": float(lambda_center),
        "lambda_rank": float(lambda_rank),
        "lambda_list": float(lambda_list),
    }
    return total, parts


@torch.no_grad()
def route_quality_diagnostics(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    rank_ignore_margin: float = 0.02,
) -> dict[str, float]:
    """Diagnostic metrics (not training loss)."""
    mae = (pred - target).abs().mean().item()
    pred_c = pred - pred.mean(dim=-1, keepdim=True)
    tgt_c = target - target.mean(dim=-1, keepdim=True)
    centered_mae = (pred_c - tgt_c).abs().mean().item()
    # pairwise ranking accuracy
    b, r = pred.shape
    if r >= 2:
        pi = pred.unsqueeze(-1)
        pj = pred.unsqueeze(-2)
        ti = target.unsqueeze(-1)
        tj = target.unsqueeze(-2)
        delta_hat = pj - pi
        delta_true = tj - ti
        tril = torch.tril(torch.ones(r, r, device=pred.device, dtype=torch.bool), diagonal=-1)
        valid = tril.unsqueeze(0) & (delta_true.abs() >= float(rank_ignore_margin))
        if bool(valid.any()):
            agree = (torch.sign(delta_hat) == torch.sign(delta_true)) & valid
            pair_acc = agree.sum().float() / valid.sum().float()
            pair_acc_v = float(pair_acc.item())
        else:
            pair_acc_v = 1.0
    else:
        pair_acc_v = 1.0
    # Spearman via rank correlation of negated losses (higher preference)
    pref_pred = (-pred).argsort(dim=-1).argsort(dim=-1).float()
    pref_true = (-target).argsort(dim=-1).argsort(dim=-1).float()
    pref_pred = pref_pred - pref_pred.mean(dim=-1, keepdim=True)
    pref_true = pref_true - pref_true.mean(dim=-1, keepdim=True)
    num = (pref_pred * pref_true).sum(dim=-1)
    den = pref_pred.norm(dim=-1) * pref_true.norm(dim=-1) + 1e-6
    spearman = float((num / den).mean().item())
    top1 = float((pred.argmin(dim=-1) == target.argmin(dim=-1)).float().mean().item())
    return {
        "mae_route_loss": float(mae),
        "centered_mae": float(centered_mae),
        "pairwise_rank_acc": float(pair_acc_v),
        "spearman_route_rank": spearman,
        "best_route_top1_acc": top1,
    }


def summarize_regrets(regrets: torch.Tensor) -> dict[str, float]:
    x = regrets.detach().float().reshape(-1)
    if x.numel() == 0:
        return {"mean_route_regret": 0.0, "median_route_regret": 0.0, "p90_route_regret": 0.0}
    return {
        "mean_route_regret": float(x.mean().item()),
        "median_route_regret": float(x.median().item()),
        "p90_route_regret": float(torch.quantile(x, 0.9).item()),
    }
