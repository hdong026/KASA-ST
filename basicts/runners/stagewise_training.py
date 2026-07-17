"""Stagewise training utilities for GR7 (temporal chain + graph resolution)."""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

STAGE_ORDER = ["T1", "T2", "T3", "S14", "S12", "S1", "FT"]
TEMPORAL_STAGE_MAP = {"T1": 0, "T2": 1, "T3": 2}
SPATIAL_STAGE_MAP = {"S14": 0, "S12": 1, "S1": 2}
PREV_STAGE_CKPT = {
    "T1": None,
    "T2": "T1.pt",
    "T3": "T2.pt",
    "S14": "T3.pt",
    "S12": "S14.pt",
    "S1": "S12.pt",
    "FT": "S1.pt",
}
STAGE_LOSS_NAMES = {
    "T1": "L_T1",
    "T2": "L_T2",
    "T3": "L_T3",
    "S14": "L_S14",
    "S12": "L_S12",
    "S1": "L_S1",
    "FT": "L_FT",
}


def temporal_downsample_target(target: torch.Tensor, h: int) -> torch.Tensor:
    """Downsample full future target [B,H,N,C] to [B,h,N,C] via horizon avg-pool."""
    from basicts.archs.arch_zoo.ChainForecasting_arch.ChainForecasting_arch import ChainForecasting

    return ChainForecasting.pool_target(target, int(h))


def _project_nodes(node_x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    return torch.einsum("mn,btnc->btmc", p, node_x)


def _shared_temporal_param_names() -> tuple[str, ...]:
    return ("td_codebook", "dw_codebook", "spa_codebook")


def set_trainable_by_stage(
    model: nn.Module,
    stage: str,
    freeze_previous: bool = True,
) -> dict[str, Any]:
    """Freeze/unfreeze modules for a stagewise training step."""
    stage = str(stage).upper()
    if stage not in STAGE_ORDER:
        raise ValueError(f"Unknown stagewise stage: {stage}")

    for param in model.parameters():
        param.requires_grad = False

    trainable_names: list[str] = []

    if stage == "FT":
        for param in model.parameters():
            param.requires_grad = True
        trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
        frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return {
            "stage": stage,
            "trainable_names": trainable_names,
            "trainable_count": trainable,
            "frozen_count": frozen,
        }

    if stage in TEMPORAL_STAGE_MAP:
        active_idx = TEMPORAL_STAGE_MAP[stage]
        if hasattr(model, "temporal_steps"):
            for step_idx, step in enumerate(model.temporal_steps):
                enable = step_idx == active_idx
                for param in step.parameters():
                    param.requires_grad = enable
                if enable:
                    trainable_names.append(f"temporal_steps.{step_idx}")
        if active_idx == 0:
            for name in _shared_temporal_param_names():
                if hasattr(model, name):
                    tensor = getattr(model, name)
                    if isinstance(tensor, nn.Parameter):
                        tensor.requires_grad = True
                        trainable_names.append(name)

    if stage in SPATIAL_STAGE_MAP and getattr(model, "graph_resolution_stack", None) is not None:
        active_idx = SPATIAL_STAGE_MAP[stage]
        stack = model.graph_resolution_stack
        for stage_idx, module in enumerate(stack.spatial_modules):
            enable = stage_idx == active_idx
            for param in module.parameters():
                param.requires_grad = enable
            if enable:
                trainable_names.append(f"graph_resolution_stack.spatial_modules.{stage_idx}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return {
        "stage": stage,
        "trainable_names": trainable_names,
        "trainable_count": trainable,
        "frozen_count": frozen,
        "freeze_previous": freeze_previous,
    }


def default_stage_ckpt_path(
    ckpt_root: str,
    horizon: int,
    seed: int,
    stage: str,
) -> str:
    return f"{ckpt_root.rstrip('/')}/H{horizon}/seed{seed}/{stage}.pt"


def resolve_load_checkpoint(
    stage: str,
    ckpt_root: str,
    horizon: int,
    seed: int,
    explicit: str | None = None,
) -> str | None:
    if explicit:
        return explicit
    prev_file = PREV_STAGE_CKPT.get(str(stage).upper())
    if prev_file is None:
        return None
    return f"{ckpt_root.rstrip('/')}/H{horizon}/seed{seed}/{prev_file}"


def compute_stagewise_loss(
    stage: str,
    out: dict[str, Any],
    real_value: torch.Tensor,
    chain_lengths: list[int],
    raw_loss_fn,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute stage-specific loss from forward intermediates."""
    stage = str(stage).upper()
    parts: dict[str, float] = {}

    if stage == "T1":
        pred = out["pred_T_low"]
        target_h = temporal_downsample_target(real_value, chain_lengths[0])
        loss = raw_loss_fn(pred, target_h)
        parts["L_T1"] = float(loss.detach().item())
        return loss, parts

    if stage == "T2":
        pred = out["pred_T_mid"]
        target_h = temporal_downsample_target(real_value, chain_lengths[1])
        loss = raw_loss_fn(pred, target_h)
        parts["L_T2"] = float(loss.detach().item())
        return loss, parts

    if stage == "T3":
        pred = out["pred_T_full"]
        loss = raw_loss_fn(pred, real_value)
        parts["L_T3"] = float(loss.detach().item())
        return loss, parts

    if stage in {"S14", "S12", "S1"}:
        diag = out.get("graph_resolution_diagnostics") or {}
        cluster_residuals = diag.get("cluster_residuals") or []
        node_before_preds = diag.get("node_before_preds") or []
        projection_matrices = diag.get("graph_projection_matrices") or []
        lifted_residuals = diag.get("lifted_residuals") or []
        stage_idx = SPATIAL_STAGE_MAP[stage]

        if stage_idx >= len(cluster_residuals):
            raise RuntimeError(f"Missing spatial diagnostics for stage {stage}")

        if stage == "S1":
            r_n = lifted_residuals[stage_idx]
            y_prev = node_before_preds[stage_idx]
            target_r = real_value - y_prev.detach()
            loss = raw_loss_fn(r_n, target_r)
            parts["L_S1"] = float(loss.detach().item())
            return loss, parts

        r_c = cluster_residuals[stage_idx]
        p = projection_matrices[stage_idx]
        y_prev = node_before_preds[stage_idx]
        target_c = _project_nodes(real_value, p)
        target_r = target_c - _project_nodes(y_prev.detach(), p)
        loss = raw_loss_fn(r_c, target_r)
        key = "L_S14" if stage == "S14" else "L_S12"
        parts[key] = float(loss.detach().item())
        return loss, parts

    if stage == "FT":
        loss = raw_loss_fn(out["pred"], real_value)
        parts["L_FT"] = float(loss.detach().item())
        parts["L_final"] = float(loss.detach().item())
        return loss, parts

    raise ValueError(f"Unsupported stagewise stage for loss: {stage}")
