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


def compute_grad_surgery(
    losses: dict[str, torch.Tensor],
    params: list[nn.Parameter],
    aux_grad_max_ratio: float = 0.2,
    eps: float = 1e-8,
) -> tuple[list[torch.Tensor], dict[str, Any]]:
    """Compute surgically combined gradients for L_final + projected aux grads."""
    l_final = losses["L_final"]
    l_t1 = losses["L_T1"]
    l_t2 = losses["L_T2"]

    g_final_list = torch.autograd.grad(
        l_final,
        params,
        retain_graph=True,
        allow_unused=True,
        create_graph=False,
    )
    g_t1_list = torch.autograd.grad(
        l_t1,
        params,
        retain_graph=True,
        allow_unused=True,
        create_graph=False,
    )
    g_t2_list = torch.autograd.grad(
        l_t2,
        params,
        retain_graph=False,
        allow_unused=True,
        create_graph=False,
    )

    def _normalize(raw: tuple[torch.Tensor | None, ...]) -> list[torch.Tensor | None]:
        out: list[torch.Tensor | None] = []
        for param, grad in zip(params, raw):
            if grad is None:
                out.append(torch.zeros_like(param) if param.requires_grad else None)
            else:
                out.append(grad)
        return out

    g_final_list = _normalize(g_final_list)
    g_t1_list = _normalize(g_t1_list)
    g_t2_list = _normalize(g_t2_list)

    g_final_flat = flatten_grads(g_final_list)
    g_t1_flat = flatten_grads(g_t1_list)
    g_t2_flat = flatten_grads(g_t2_list)

    cos_t1 = cosine_similarity_flat(g_t1_flat, g_final_flat, eps=eps)
    cos_t2 = cosine_similarity_flat(g_t2_flat, g_final_flat, eps=eps)

    g_t1_proj, t1_projected = project_aux_grad_flat(g_t1_flat.clone(), g_final_flat, eps=eps)
    g_t2_proj, t2_projected = project_aux_grad_flat(g_t2_flat.clone(), g_final_flat, eps=eps)

    g_combined_flat = combine_primary_aux_grads(
        g_final_flat,
        [g_t1_proj, g_t2_proj],
        aux_grad_max_ratio=aux_grad_max_ratio,
        eps=eps,
    )

    aux_mean = torch.stack([g_t1_proj, g_t2_proj], dim=0).mean(dim=0)
    stats = {
        "L_final": float(l_final.detach().item()),
        "L_T1": float(l_t1.detach().item()),
        "L_T2": float(l_t2.detach().item()),
        "cos_g_T1_g_final": cos_t1,
        "cos_g_T2_g_final": cos_t2,
        "T1_projected": t1_projected,
        "T2_projected": t2_projected,
        "final_grad_norm": float(g_final_flat.norm().detach().item()),
        "aux_grad_norm": float(aux_mean.norm().detach().item()),
        "combined_grad_norm": float(g_combined_flat.norm().detach().item()),
    }

    combined_list = unflatten_grads(g_combined_flat, g_final_list)
    return combined_list, stats
