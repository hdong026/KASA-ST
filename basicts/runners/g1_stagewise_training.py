"""Stagewise training utilities for G1_final_adaptive (T1 -> T2 -> T3 -> S1)."""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

G1_STAGE_ORDER = ["T1", "T2", "T3", "S1"]
G1_STAGE_CHOICES = list(G1_STAGE_ORDER)

G1_PREV_STAGE_CKPT: dict[str, str | None] = {
    "T1": None,
    "T2": "T1_best.pt",
    "T3": "T2_best.pt",
    "S1": "T3_best.pt",
}

G1_STAGE_LOSS_NAMES = {
    "T1": "L_T1",
    "T2": "L_T2",
    "T3": "L_T3",
    "S1": "L_S1",
}


def stage_best_ckpt_name(stage: str) -> str:
    return f"{str(stage).upper()}_best.pt"


def temporal_downsample_target(target: torch.Tensor, h: int) -> torch.Tensor:
    """Downsample full future target [B,H,N,C] to [B,h,N,C] via horizon avg-pool."""
    from basicts.archs.arch_zoo.ChainForecasting_arch.ChainForecasting_arch import ChainForecasting

    return ChainForecasting.pool_target(target, int(h))


def _shared_temporal_param_names() -> tuple[str, ...]:
    return ("td_codebook", "dw_codebook", "spa_codebook")


def set_trainable_by_stage(
    model: nn.Module,
    stage: str,
    freeze_previous: bool = True,
    train_shared_temporal: bool = True,
) -> dict[str, Any]:
    """Freeze/unfreeze modules for G1 stagewise training (T1/T2/T3/S1 only)."""
    stage = str(stage).upper()
    if stage not in G1_STAGE_CHOICES:
        raise ValueError(f"Unknown G1 stagewise stage: {stage}")

    for param in model.parameters():
        param.requires_grad = False

    trainable_names: list[str] = []
    temporal_map = {"T1": 0, "T2": 1, "T3": 2}

    if stage in temporal_map:
        active_idx = temporal_map[stage]
        if hasattr(model, "temporal_steps"):
            for step_idx, step in enumerate(model.temporal_steps):
                enable = step_idx == active_idx
                for param in step.parameters():
                    param.requires_grad = enable
                if enable:
                    trainable_names.append(f"temporal_steps.{step_idx}")
        if train_shared_temporal:
            for name in _shared_temporal_param_names():
                if hasattr(model, name):
                    tensor = getattr(model, name)
                    if isinstance(tensor, nn.Parameter):
                        tensor.requires_grad = True
                        trainable_names.append(name)

    if stage == "S1":
        spatial = getattr(model, "spatial_module", None)
        if spatial is None:
            raise RuntimeError("G1_stagewise S1 requires model.spatial_module (final adaptive spatial)")
        for param in spatial.parameters():
            param.requires_grad = True
        trainable_names.append("spatial_module")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return {
        "stage": stage,
        "trainable_names": trainable_names,
        "trainable_count": trainable,
        "frozen_count": frozen,
        "freeze_previous": freeze_previous,
        "train_shared_temporal": train_shared_temporal,
    }


def default_stage_ckpt_path(
    ckpt_root: str,
    horizon: int,
    seed: int,
    stage: str,
) -> str:
    return f"{ckpt_root.rstrip('/')}/H{horizon}/seed{seed}/{stage_best_ckpt_name(stage)}"


def resolve_load_checkpoint(
    stage: str,
    ckpt_root: str,
    horizon: int,
    seed: int,
    explicit: str | None = None,
) -> str | None:
    if explicit:
        return explicit
    prev_file = G1_PREV_STAGE_CKPT.get(str(stage).upper())
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
    """Compute G1 stage-specific loss from forward intermediates."""
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

    if stage == "S1":
        pred = out.get("pred_final")
        if pred is None:
            pred = out.get("pred")
        if pred is None:
            raise RuntimeError("G1_stagewise S1 forward missing pred_final / pred")
        loss = raw_loss_fn(pred, real_value)
        parts["L_S1"] = float(loss.detach().item())
        return loss, parts

    raise ValueError(f"Unsupported G1 stagewise stage for loss: {stage}")
