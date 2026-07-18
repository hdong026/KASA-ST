"""Stagewise training utilities for GR7 (temporal chain + graph resolution)."""
from __future__ import annotations

import logging
from typing import Any

import torch
from torch import nn

DEFAULT_STAGE_ORDER = ["T1", "T2", "T3", "S14", "S12", "S1"]
STAGE_SEQUENCES: dict[str, list[str]] = {
    "full": ["T1", "T2", "T3", "S14", "S12", "S1"],
    "no_s14": ["T1", "T2", "T3", "S12", "S1"],
    "final_spatial_only": ["T1", "T2", "T3", "S1"],
}
# Kept for manual `--stage FT` compatibility only; excluded from --run_stagewise_all.
ALL_STAGE_CHOICES = DEFAULT_STAGE_ORDER + ["FT"]
STAGE_ORDER = DEFAULT_STAGE_ORDER

TEMPORAL_STAGE_MAP = {"T1": 0, "T2": 1, "T3": 2}
SPATIAL_STAGE_MAP = {"S14": 0, "S12": 1, "S1": 2}
PREV_STAGE_CKPT_BY_SEQUENCE: dict[str, dict[str, str | None]] = {
    "full": {
        "T1": None,
        "T2": "T1_best.pt",
        "T3": "T2_best.pt",
        "S14": "T3_best.pt",
        "S12": "S14_best.pt",
        "S1": "S12_best.pt",
    },
    "no_s14": {
        "T1": None,
        "T2": "T1_best.pt",
        "T3": "T2_best.pt",
        "S12": "T3_best.pt",
        "S1": "S12_best.pt",
    },
    "final_spatial_only": {
        "T1": None,
        "T2": "T1_best.pt",
        "T3": "T2_best.pt",
        "S1": "T3_best.pt",
    },
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


def stage_best_ckpt_name(stage: str) -> str:
    return f"{str(stage).upper()}_best.pt"


def temporal_downsample_target(target: torch.Tensor, h: int) -> torch.Tensor:
    """Downsample full future target [B,H,N,C] to [B,h,N,C] via horizon avg-pool."""
    from basicts.archs.arch_zoo.ChainForecasting_arch.ChainForecasting_arch import ChainForecasting

    return ChainForecasting.pool_target(target, int(h))


def _project_nodes(node_x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    return torch.einsum("mn,btnc->btmc", p, node_x)


def _shared_temporal_param_names() -> tuple[str, ...]:
    return ("td_codebook", "dw_codebook", "spa_codebook")


def resolve_spatial_stage_idx(stage: str, sequence: str, num_spatial_modules: int) -> int:
    """Map stagewise spatial stage name to module index in graph_resolution_stack."""
    stage = str(stage).upper()
    seq = str(sequence).lower()
    if stage == "S1" and seq == "final_spatial_only":
        return 0
    if stage not in SPATIAL_STAGE_MAP:
        raise ValueError(f"Unknown spatial stagewise stage: {stage}")
    idx = SPATIAL_STAGE_MAP[stage]
    if idx >= num_spatial_modules:
        raise ValueError(
            f"Spatial stage {stage} index {idx} out of range for {num_spatial_modules} modules"
        )
    return idx


def set_trainable_by_stage(
    model: nn.Module,
    stage: str,
    freeze_previous: bool = True,
    train_shared_temporal: bool = True,
    sequence: str = "full",
) -> dict[str, Any]:
    """Freeze/unfreeze modules for a stagewise training step."""
    stage = str(stage).upper()
    if stage not in ALL_STAGE_CHOICES:
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
            "train_shared_temporal": train_shared_temporal,
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
        if train_shared_temporal:
            for name in _shared_temporal_param_names():
                if hasattr(model, name):
                    tensor = getattr(model, name)
                    if isinstance(tensor, nn.Parameter):
                        tensor.requires_grad = True
                        trainable_names.append(name)

    if stage in SPATIAL_STAGE_MAP and getattr(model, "graph_resolution_stack", None) is not None:
        stack = model.graph_resolution_stack
        num_modules = len(stack.spatial_modules)
        active_idx = resolve_spatial_stage_idx(stage, sequence, num_modules)
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
    sequence: str = "full",
    explicit: str | None = None,
) -> str | None:
    if explicit:
        return explicit
    seq = sequence if sequence in PREV_STAGE_CKPT_BY_SEQUENCE else "full"
    prev_file = PREV_STAGE_CKPT_BY_SEQUENCE[seq].get(str(stage).upper())
    if prev_file is None:
        return None
    return f"{ckpt_root.rstrip('/')}/H{horizon}/seed{seed}/{prev_file}"


def _stage_alpha(
    stage: str,
    stage_idx: int,
    diag: dict[str, Any],
    fallback_alphas: list[float] | None = None,
) -> float:
    alphas = diag.get("graph_resolution_alphas") or fallback_alphas or [0.03, 0.06, 0.10]
    if stage_idx < len(alphas):
        return float(alphas[stage_idx])
    return float(alphas[-1])


def compute_stagewise_loss(
    stage: str,
    out: dict[str, Any],
    real_value: torch.Tensor,
    chain_lengths: list[int],
    raw_loss_fn,
    sequence: str = "full",
    logger: logging.Logger | None = None,
    graph_resolution_alphas: list[float] | None = None,
    log_alpha_once: bool = False,
    _alpha_logged: set[str] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute stage-specific loss from forward intermediates."""
    stage = str(stage).upper()
    parts: dict[str, float] = {}
    alpha_logged = _alpha_logged if _alpha_logged is not None else set()

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
        num_spatial = max(
            len(cluster_residuals),
            len(lifted_residuals),
            len(node_before_preds),
            1,
        )
        stage_idx = resolve_spatial_stage_idx(stage, sequence, num_spatial)
        alpha_r = _stage_alpha(stage, stage_idx, diag, graph_resolution_alphas)

        if stage_idx >= len(cluster_residuals) and stage != "S1":
            raise RuntimeError(f"Missing spatial diagnostics for stage {stage}")

        log_key = f"{stage}:{stage_idx}"
        if log_alpha_once and log_key not in alpha_logged and logger is not None:
            logger.info(
                "[GR7_stagewise loss] stage=%s alpha_r=%.4f residual_scaled_by_alpha=True",
                stage,
                alpha_r,
            )
            alpha_logged.add(log_key)

        if stage == "S1":
            y_prev = out.get("pred_T_full")
            if y_prev is None and stage_idx < len(node_before_preds):
                y_prev = node_before_preds[stage_idx]
            if stage_idx < len(lifted_residuals) and lifted_residuals[stage_idx] is not None:
                scaled_r_n = alpha_r * lifted_residuals[stage_idx]
                target_r = real_value - y_prev.detach()
                loss = raw_loss_fn(scaled_r_n, target_r)
            else:
                loss = raw_loss_fn(out["pred"], real_value)
            parts["L_S1"] = float(loss.detach().item())
            return loss, parts

        residual_idx = stage_idx
        proj_idx = stage_idx
        if stage == "S12" and sequence == "no_s14":
            residual_idx = -1
            proj_idx = -1
        r_c = cluster_residuals[residual_idx]
        scaled_r_c = alpha_r * r_c
        p = projection_matrices[proj_idx]
        if stage == "S12" and sequence == "no_s14":
            y_prev = out.get("pred_T_full")
            if y_prev is None:
                raise RuntimeError("pred_T_full missing for no_s14 S12 loss")
        else:
            y_prev = node_before_preds[stage_idx]
        target_c = _project_nodes(real_value, p)
        target_r = target_c - _project_nodes(y_prev.detach(), p)
        loss = raw_loss_fn(scaled_r_c, target_r)
        key = "L_S14" if stage == "S14" else "L_S12"
        parts[key] = float(loss.detach().item())
        return loss, parts

    if stage == "FT":
        loss = raw_loss_fn(out["pred"], real_value)
        parts["L_FT"] = float(loss.detach().item())
        parts["L_final"] = float(loss.detach().item())
        return loss, parts

    raise ValueError(f"Unsupported stagewise stage for loss: {stage}")
