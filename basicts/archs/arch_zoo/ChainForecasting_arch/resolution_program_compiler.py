"""Compile one-shot planner logits into nested legal resolution programs."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from basicts.archs.arch_zoo.ChainForecasting_arch.one_shot_resolution_hierarchy import (
    apply_batched_splits,
    frontier_active_counts,
    full_leaf_frontier_mask,
    root_frontier_mask,
)
from basicts.archs.arch_zoo.ChainForecasting_arch.one_shot_resolution_planner import (
    MAX_OPTIONAL_INTERMEDIATE_STEPS,
)


class ResolutionProgramCompiler:
    """Batched compiler: root → 0..K intermediate frontiers → mandatory full leaves."""

    def __init__(
        self,
        temporal_tree,
        spatial_tree,
        left_t: torch.Tensor,
        right_t: torch.Tensor,
        leaf_t: torch.Tensor,
        left_s: torch.Tensor,
        right_s: torch.Tensor,
        leaf_s: torch.Tensor,
        k_steps: int = MAX_OPTIONAL_INTERMEDIATE_STEPS,
        split_threshold: float = 0.5,
        continue_threshold: float = 0.5,
    ):
        self.temporal_tree = temporal_tree
        self.spatial_tree = spatial_tree
        self.left_t = left_t
        self.right_t = right_t
        self.leaf_t = leaf_t
        self.left_s = left_s
        self.right_s = right_s
        self.leaf_s = leaf_s
        self.k_steps = int(k_steps)
        self.split_threshold = float(split_threshold)
        self.continue_threshold = float(continue_threshold)

    def _hard_splits(
        self,
        logits: torch.Tensor,
        frontier: torch.Tensor,
        is_leaf: torch.Tensor,
        budget_scale: torch.Tensor,
        deterministic: bool,
    ) -> torch.Tensor:
        """Select splits among currently active non-leaves; STE when training."""
        allow = (frontier > 0.5) & (~is_leaf.unsqueeze(0))
        prob = torch.sigmoid(logits)
        # Budget soft gating: lower remaining budget → lower split tendency
        gated = prob * budget_scale.unsqueeze(-1).clamp(0.0, 1.0)
        if deterministic:
            hard = (gated > self.split_threshold).to(logits.dtype) * allow.to(logits.dtype)
        else:
            # Straight-through Bernoulli-like
            soft = gated
            hard = (soft > self.split_threshold).to(logits.dtype)
            hard = hard + (soft - soft.detach())
            hard = hard * allow.to(logits.dtype)
        return hard

    def compile(
        self,
        planner_out: dict[str, torch.Tensor],
        optional_budget: torch.Tensor,
        thinking_intensity: float | torch.Tensor,
        deterministic: bool = True,
    ) -> dict[str, Any]:
        t_logits = planner_out["temporal_split_logits"]  # [B,K,T]
        s_logits = planner_out["spatial_split_logits"]
        c_logits = planner_out["continue_logits"]  # [B,K]
        b = t_logits.shape[0]
        device = t_logits.device
        dtype = t_logits.dtype
        k = self.k_steps

        t_front = root_frontier_mask(b, self.temporal_tree, device, dtype)
        s_front = root_frontier_mask(b, self.spatial_tree, device, dtype)
        remaining = optional_budget.to(device=device, dtype=dtype).clone()
        if remaining.ndim == 0:
            remaining = remaining.expand(b).clone()

        still_planning = torch.ones(b, device=device, dtype=torch.bool)
        stage_valid = []
        stage_t_frontiers = []
        stage_s_frontiers = []
        stage_types = []
        stage_costs = []
        continue_probs = torch.sigmoid(c_logits)

        left_t = self.left_t.to(device)
        right_t = self.right_t.to(device)
        leaf_t = self.leaf_t.to(device)
        left_s = self.left_s.to(device)
        right_s = self.right_s.to(device)
        leaf_s = self.leaf_s.to(device)

        cum_cost = torch.zeros(b, device=device, dtype=dtype)

        for ki in range(k):
            cont_prob = continue_probs[:, ki]
            if deterministic:
                do_cont = (cont_prob > self.continue_threshold) & still_planning & (remaining > 1e-6)
            else:
                # STE continue
                soft = cont_prob
                hard_c = (soft > self.continue_threshold).to(dtype)
                hard_c = hard_c + (soft - soft.detach())
                do_cont = (hard_c > 0.5) & still_planning & (remaining > 1e-6)

            # Budget scale for split gating
            scale = (remaining / optional_budget.clamp_min(1e-6)).clamp(0.0, 1.0)
            # Only samples that continue get splits applied for this stage frontier
            t_split = self._hard_splits(
                t_logits[:, ki], t_front, leaf_t, scale, deterministic
            )
            s_split = self._hard_splits(
                s_logits[:, ki], s_front, leaf_s, scale, deterministic
            )
            # Zero splits for stopped samples
            alive = do_cont.to(dtype).unsqueeze(-1)
            t_split = t_split * alive
            s_split = s_split * alive

            # Safety: continuing samples must change resolution at least once (T or S)
            t_allow = (t_front > 0.5) & (~leaf_t.unsqueeze(0))
            s_allow = (s_front > 0.5) & (~leaf_s.unsqueeze(0))
            need = do_cont & (t_split.sum(-1) + s_split.sum(-1) < 0.5) & (
                t_allow.any(-1) | s_allow.any(-1)
            )
            t_prob = torch.sigmoid(t_logits[:, ki]).masked_fill(~t_allow, -1.0)
            s_prob = torch.sigmoid(s_logits[:, ki]).masked_fill(~s_allow, -1.0)
            t_best = t_prob.max(dim=-1).values
            s_best = s_prob.max(dim=-1).values
            prefer_t = t_best >= s_best
            t_idx = t_prob.argmax(dim=-1)
            s_idx = s_prob.argmax(dim=-1)
            t_force = F.one_hot(t_idx, num_classes=t_split.shape[-1]).to(dtype)
            s_force = F.one_hot(s_idx, num_classes=s_split.shape[-1]).to(dtype)
            t_split = torch.where(
                (need & prefer_t).unsqueeze(-1),
                t_force * t_allow.to(dtype),
                t_split,
            )
            s_split = torch.where(
                (need & (~prefer_t)).unsqueeze(-1),
                s_force * s_allow.to(dtype),
                s_split,
            )

            t_front = apply_batched_splits(t_front, t_split, left_t, right_t, leaf_t)
            s_front = apply_batched_splits(s_front, s_split, left_s, right_s, leaf_s)

            t_count = frontier_active_counts(t_front)
            s_count = frontier_active_counts(s_front)
            # Cost proxy
            h = float(self.temporal_tree.horizon)
            n = float(self.spatial_tree.n_nodes)
            cost = (t_count * s_count) / (h * n) + 0.05
            cost = cost * do_cont.to(dtype)
            cum_cost = cum_cost + cost
            remaining = (optional_budget - cum_cost).clamp_min(0.0)

            # Stage type labels (tensor codes): 0 temporal, 1 spatial, 2 joint, 3 skip
            t_changed = (t_split > 0.5).any(dim=-1)
            s_changed = (s_split > 0.5).any(dim=-1)
            typ = torch.full((b,), 3, device=device, dtype=torch.long)
            typ = torch.where(do_cont & t_changed & s_changed, torch.full_like(typ, 2), typ)
            typ = torch.where(do_cont & t_changed & (~s_changed), torch.full_like(typ, 0), typ)
            typ = torch.where(do_cont & (~t_changed) & s_changed, torch.full_like(typ, 1), typ)

            stage_valid.append(do_cont)
            stage_t_frontiers.append(t_front.clone())
            stage_s_frontiers.append(s_front.clone())
            stage_types.append(typ)
            stage_costs.append(cost)

            still_planning = still_planning & do_cont

        # Mandatory full-resolution final
        t_final = full_leaf_frontier_mask(b, self.temporal_tree, device, dtype)
        s_final = full_leaf_frontier_mask(b, self.spatial_tree, device, dtype)

        return {
            "stage_valid": torch.stack(stage_valid, dim=1),  # [B,K] bool
            "temporal_frontiers": torch.stack(stage_t_frontiers, dim=1),  # [B,K,Tnodes]
            "spatial_frontiers": torch.stack(stage_s_frontiers, dim=1),
            "stage_types": torch.stack(stage_types, dim=1),  # [B,K]
            "stage_costs": torch.stack(stage_costs, dim=1),
            "cumulative_optional_cost": cum_cost,
            "remaining_budget": remaining,
            "continue_probs": continue_probs,
            "final_temporal_frontier": t_final,
            "final_spatial_frontier": s_final,
            "intermediate_stage_count": torch.stack(stage_valid, dim=1).to(dtype).sum(dim=1),
        }
