"""Gradient surgery utilities for final-primary training with auxiliary objectives."""
from __future__ import annotations

from typing import Any

import torch
from torch import nn


def trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def flatten_grads(grads: list[torch.Tensor | None]) -> torch.Tensor:
    parts = [g.reshape(-1) for g in grads if g is not None]
    if not parts:
        device = next((g.device for g in grads if g is not None), None)
        return torch.zeros(0, device=device)
    return torch.cat(parts)


def unflatten_grads(flat: torch.Tensor, template: list[torch.Tensor | None]) -> list[torch.Tensor]:
    out: list[torch.Tensor] = []
    offset = 0
    for g in template:
        if g is None:
            continue
        numel = g.numel()
        out.append(flat[offset : offset + numel].view_as(g))
        offset += numel
    return out


def cosine_similarity_flat(g_aux: torch.Tensor, g_final: torch.Tensor, eps: float = 1e-8) -> float:
    if g_aux.numel() == 0 or g_final.numel() == 0:
        return float("nan")
    dot = torch.dot(g_aux, g_final)
    denom = g_aux.norm() * g_final.norm() + eps
    return float((dot / denom).detach().item())


def project_aux_grad_flat(
    g_aux: torch.Tensor,
    g_final: torch.Tensor,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, bool]:
    if g_aux.numel() == 0:
        return g_aux, False
    dot = torch.dot(g_aux, g_final)
    final_norm_sq = torch.dot(g_final, g_final) + eps
    if float(dot.item()) < 0.0:
        g_aux = g_aux - (dot / final_norm_sq) * g_final
        return g_aux, True
    return g_aux, False


def combine_primary_aux_grads(
    g_final: torch.Tensor,
    aux_grads: list[torch.Tensor],
    aux_grad_max_ratio: float = 0.2,
    eps: float = 1e-8,
) -> torch.Tensor:
    if not aux_grads:
        return g_final
    g_aux = torch.stack(aux_grads, dim=0).mean(dim=0)
    aux_norm = g_aux.norm()
    final_norm = g_final.norm()
    max_aux_norm = float(aux_grad_max_ratio) * final_norm
    if aux_norm > max_aux_norm + eps:
        g_aux = g_aux * (max_aux_norm / (aux_norm + eps))
    return g_final + g_aux


def _normalize_grads(
    raw: tuple[torch.Tensor | None, ...],
    params: list[nn.Parameter],
) -> list[torch.Tensor | None]:
    out: list[torch.Tensor | None] = []
    for param, grad in zip(params, raw):
        if grad is None:
            out.append(torch.zeros_like(param) if param.requires_grad else None)
        else:
            out.append(grad)
    return out


def _cosine_stat_key(aux_name: str) -> str:
    if aux_name == "L_T1":
        return "cos_g_T1_g_final"
    if aux_name == "L_T2":
        return "cos_g_T2_g_final"
    if aux_name == "L_G14":
        return "cos_G14_final"
    if aux_name == "L_G12":
        return "cos_G12_final"
    return f"cos_{aux_name}_g_final"


def compute_grad_surgery(
    losses: dict[str, torch.Tensor],
    params: list[nn.Parameter],
    aux_grad_max_ratio: float = 0.2,
    aux_loss_names: list[str] | None = None,
    eps: float = 1e-8,
) -> tuple[list[torch.Tensor], dict[str, Any]]:
    """Compute surgically combined gradients for L_final + projected aux grads."""
    if aux_loss_names is None:
        aux_loss_names = ["L_T1", "L_T2"]

    l_final = losses["L_final"]
    g_final_list = _normalize_grads(
        torch.autograd.grad(
            l_final,
            params,
            retain_graph=True,
            allow_unused=True,
            create_graph=False,
        ),
        params,
    )
    g_final_flat = flatten_grads(g_final_list)

    aux_proj_list: list[torch.Tensor] = []
    projected_flags: list[bool] = []
    stats: dict[str, Any] = {
        "L_final": float(l_final.detach().item()),
        "final_grad_norm": float(g_final_flat.norm().detach().item()),
    }

    for idx, aux_name in enumerate(aux_loss_names):
        l_aux = losses[aux_name]
        stats[aux_name] = float(l_aux.detach().item())
        retain = idx < len(aux_loss_names) - 1
        g_aux_list = _normalize_grads(
            torch.autograd.grad(
                l_aux,
                params,
                retain_graph=retain,
                allow_unused=True,
                create_graph=False,
            ),
            params,
        )
        g_aux_flat = flatten_grads(g_aux_list)
        cos_key = _cosine_stat_key(aux_name)
        stats[cos_key] = cosine_similarity_flat(g_aux_flat, g_final_flat, eps=eps)
        g_aux_proj, was_projected = project_aux_grad_flat(
            g_aux_flat.clone(),
            g_final_flat,
            eps=eps,
        )
        projected_flags.append(was_projected)
        aux_proj_list.append(g_aux_proj)

    g_combined_flat = combine_primary_aux_grads(
        g_final_flat,
        aux_proj_list,
        aux_grad_max_ratio=aux_grad_max_ratio,
        eps=eps,
    )

    aux_mean = (
        torch.stack(aux_proj_list, dim=0).mean(dim=0)
        if aux_proj_list
        else torch.zeros_like(g_final_flat)
    )
    stats["aux_grad_norm"] = float(aux_mean.norm().detach().item())
    stats["combined_grad_norm"] = float(g_combined_flat.norm().detach().item())
    stats["projected_rate"] = (
        float(sum(1 for flag in projected_flags if flag) / len(projected_flags))
        if projected_flags
        else 0.0
    )

    if "L_T1" in aux_loss_names:
        stats["cos_g_T1_g_final"] = stats.get("cos_g_T1_g_final", float("nan"))
        stats["T1_projected"] = projected_flags[aux_loss_names.index("L_T1")]
    if "L_T2" in aux_loss_names:
        stats["cos_g_T2_g_final"] = stats.get("cos_g_T2_g_final", float("nan"))
        stats["T2_projected"] = projected_flags[aux_loss_names.index("L_T2")]

    combined_list = unflatten_grads(g_combined_flat, g_final_list)
    return combined_list, stats
