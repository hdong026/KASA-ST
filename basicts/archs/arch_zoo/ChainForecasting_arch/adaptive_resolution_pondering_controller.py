"""Sample-wise temporal/spatial split + halt controller for pondering F2FNet."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


def hard_gumbel_sigmoid(
    logits: torch.Tensor,
    tau: float = 1.0,
    hard: bool = True,
    threshold: float = 0.5,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Binary decisions with straight-through estimator."""
    u = torch.rand_like(logits).clamp(eps, 1.0 - eps)
    g = -torch.log(-torch.log(u))
    y_soft = torch.sigmoid((logits + g) / max(float(tau), 1e-4))
    if not hard:
        return y_soft
    y_hard = (y_soft > threshold).to(dtype=logits.dtype)
    return y_hard + (y_soft - y_soft.detach())


class AdaptiveResolutionPonderingController(nn.Module):
    """Per-unit split heads + sample halt head (shared parameters)."""

    def __init__(
        self,
        unit_feat_dim: int = 12,
        global_feat_dim: int = 16,
        hidden_dim: int = 32,
        split_bias_init: float = 1.5,
        halt_bias_init: float = -2.0,
        temperature: float = 1.0,
        split_threshold: float = 0.5,
        halt_threshold: float = 0.5,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.temperature = float(temperature)
        self.split_threshold = float(split_threshold)
        self.halt_threshold = float(halt_threshold)

        self.temporal_mlp = nn.Sequential(
            nn.Linear(unit_feat_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.spatial_mlp = nn.Sequential(
            nn.Linear(unit_feat_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.halt_mlp = nn.Sequential(
            nn.Linear(global_feat_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.constant_(self.temporal_mlp[-1].bias, float(split_bias_init))
        nn.init.constant_(self.spatial_mlp[-1].bias, float(split_bias_init))
        nn.init.constant_(self.halt_mlp[-1].bias, float(halt_bias_init))
        nn.init.zeros_(self.temporal_mlp[-1].weight)
        nn.init.zeros_(self.spatial_mlp[-1].weight)
        nn.init.zeros_(self.halt_mlp[-1].weight)

    def set_temperature(self, tau: float) -> None:
        self.temperature = float(tau)

    def forward(
        self,
        temporal_unit_feats: torch.Tensor,
        spatial_unit_feats: torch.Tensor,
        temporal_valid_mask: torch.Tensor,
        spatial_valid_mask: torch.Tensor,
        temporal_leaf_mask: torch.Tensor,
        spatial_leaf_mask: torch.Tensor,
        global_feats: torch.Tensor,
        halted: torch.Tensor,
        deterministic: bool | None = None,
    ) -> dict[str, Any]:
        """
        Args:
            temporal_unit_feats: [B, Tpad, F]
            spatial_unit_feats: [B, Spad, F]
            *_valid_mask: [B, *pad] active frontier slots
            *_leaf_mask: [B, *pad] whether slot is leaf (cannot split)
            global_feats: [B, G]
            halted: [B]
        """
        if deterministic is None:
            deterministic = not self.training

        t_logits = self.temporal_mlp(temporal_unit_feats).squeeze(-1)  # [B,T]
        s_logits = self.spatial_mlp(spatial_unit_feats).squeeze(-1)
        h_logits = self.halt_mlp(global_feats).squeeze(-1)  # [B]

        # Mask invalid / leaf / halted
        t_allow = (temporal_valid_mask > 0.5) & (temporal_leaf_mask < 0.5)
        s_allow = (spatial_valid_mask > 0.5) & (spatial_leaf_mask < 0.5)
        t_logits = t_logits.masked_fill(~t_allow, -1e4)
        s_logits = s_logits.masked_fill(~s_allow, -1e4)
        h_logits = h_logits.masked_fill(halted, 1e4)

        t_prob = torch.sigmoid(t_logits)
        s_prob = torch.sigmoid(s_logits)
        h_prob = torch.sigmoid(h_logits)

        if deterministic:
            t_hard = (t_prob > self.split_threshold).to(t_prob.dtype) * t_allow.to(t_prob.dtype)
            s_hard = (s_prob > self.split_threshold).to(s_prob.dtype) * s_allow.to(s_prob.dtype)
            h_hard = (h_prob > self.halt_threshold).to(h_prob.dtype)
        else:
            t_hard = hard_gumbel_sigmoid(
                t_logits, tau=self.temperature, hard=True, threshold=self.split_threshold
            ) * t_allow.to(t_prob.dtype)
            s_hard = hard_gumbel_sigmoid(
                s_logits, tau=self.temperature, hard=True, threshold=self.split_threshold
            ) * s_allow.to(s_prob.dtype)
            h_hard = hard_gumbel_sigmoid(
                h_logits, tau=self.temperature, hard=True, threshold=self.halt_threshold
            )

        # Halted samples: no splits
        alive = (~halted).to(t_hard.dtype).unsqueeze(-1)
        t_hard = t_hard * alive
        s_hard = s_hard * alive

        return {
            "temporal_split_logits": t_logits,
            "spatial_split_logits": s_logits,
            "halt_logits": h_logits,
            "temporal_split_prob": t_prob,
            "spatial_split_prob": s_prob,
            "halt_prob": h_prob,
            "temporal_split_hard": t_hard,
            "spatial_split_hard": s_hard,
            "halt_hard": h_hard,
            "temporal_allow": t_allow,
            "spatial_allow": s_allow,
        }


def enforce_progress_or_halt(
    ctrl_out: dict[str, torch.Tensor],
    fully_refined: torch.Tensor,
    halted: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Safety: force halt if refined; else force at least one split if none selected."""
    t_hard = ctrl_out["temporal_split_hard"].clone()
    s_hard = ctrl_out["spatial_split_hard"].clone()
    h_hard = ctrl_out["halt_hard"].clone()
    t_allow = ctrl_out["temporal_allow"]
    s_allow = ctrl_out["spatial_allow"]
    t_prob = ctrl_out["temporal_split_prob"]
    s_prob = ctrl_out["spatial_split_prob"]

    b = h_hard.shape[0]
    for bi in range(b):
        if bool(halted[bi]) or bool(h_hard[bi] > 0.5):
            t_hard[bi].zero_()
            s_hard[bi].zero_()
            h_hard[bi] = 1.0
            continue
        if bool(fully_refined[bi]):
            t_hard[bi].zero_()
            s_hard[bi].zero_()
            h_hard[bi] = 1.0
            continue
        any_split = bool((t_hard[bi] > 0.5).any() or (s_hard[bi] > 0.5).any())
        if not any_split:
            # Force highest-probability valid split (STE-friendly overwrite)
            t_scores = t_prob[bi].masked_fill(~t_allow[bi], -1.0)
            s_scores = s_prob[bi].masked_fill(~s_allow[bi], -1.0)
            t_max = float(t_scores.max()) if t_allow[bi].any() else -1.0
            s_max = float(s_scores.max()) if s_allow[bi].any() else -1.0
            if t_max < 0 and s_max < 0:
                h_hard[bi] = 1.0
            elif t_max >= s_max:
                idx = int(t_scores.argmax())
                # Keep gradient path through soft probability at forced index
                t_hard[bi, idx] = 1.0 + (t_prob[bi, idx] - t_prob[bi, idx].detach())
            else:
                idx = int(s_scores.argmax())
                s_hard[bi, idx] = 1.0 + (s_prob[bi, idx] - s_prob[bi, idx].detach())
    ctrl_out = dict(ctrl_out)
    ctrl_out["temporal_split_hard"] = t_hard
    ctrl_out["spatial_split_hard"] = s_hard
    ctrl_out["halt_hard"] = h_hard
    return ctrl_out


def map_slot_splits_to_tree_mask(
    slot_split: torch.Tensor,
    frontier_mask: torch.Tensor,
) -> torch.Tensor:
    """Map padded slot splits [B, max] onto tree-node mask [B, n_tree]."""
    b, n_tree = frontier_mask.shape
    out = torch.zeros_like(frontier_mask)
    for bi in range(b):
        ids = torch.nonzero(frontier_mask[bi] > 0.5, as_tuple=False).flatten()
        for slot, nid in enumerate(ids.tolist()):
            if slot >= slot_split.shape[1]:
                break
            out[bi, nid] = slot_split[bi, slot]
    return out
