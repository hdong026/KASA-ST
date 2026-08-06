"""Teacher resolution program optimizer — training-only, never in inference forward."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import nn

from basicts.archs.arch_zoo.ChainForecasting_arch.one_shot_resolution_planner import (
    MAX_OPTIONAL_INTERMEDIATE_STEPS,
)


Backend = Literal["proxy_greedy", "bounded_beam_search"]


@dataclass
class TeacherProgram:
    temporal_frontiers: torch.Tensor  # [B,K,Tnodes]
    spatial_frontiers: torch.Tensor
    stage_valid: torch.Tensor
    notes: str = ""


class ResolutionProgramOptimizer:
    """Offline teacher program generator for future Stage-B imitation.

    Must NOT be registered as a submodule of the inference model.
    """

    def __init__(
        self,
        temporal_tree,
        spatial_tree,
        backend: Backend = "proxy_greedy",
        k_steps: int = MAX_OPTIONAL_INTERMEDIATE_STEPS,
        beam_width: int = 2,
    ):
        self.temporal_tree = temporal_tree
        self.spatial_tree = spatial_tree
        self.backend = backend
        self.k_steps = int(k_steps)
        self.beam_width = int(min(beam_width, 4))  # hard small cap

    def generate_teacher_programs(
        self,
        history: torch.Tensor,
        target: torch.Tensor,
        optional_budget: torch.Tensor,
        leaf_cover_t: torch.Tensor,
        leaf_cover_s: torch.Tensor,
    ) -> TeacherProgram:
        if self.backend == "proxy_greedy":
            return self._proxy_greedy(history, target, optional_budget, leaf_cover_t, leaf_cover_s)
        if self.backend == "bounded_beam_search":
            # Interface only for this task — falls back to proxy (no actual beam exec required)
            return self._proxy_greedy(history, target, optional_budget, leaf_cover_t, leaf_cover_s)
        raise ValueError(f"Unknown backend: {self.backend}")

    def _proxy_greedy(
        self,
        history: torch.Tensor,
        target: torch.Tensor,
        optional_budget: torch.Tensor,
        leaf_cover_t: torch.Tensor,
        leaf_cover_s: torch.Tensor,
    ) -> TeacherProgram:
        """Low-cost proxy: split high-heterogeneity non-leaf regions within budget."""
        from basicts.archs.arch_zoo.ChainForecasting_arch.one_shot_resolution_hierarchy import (
            apply_batched_splits,
            build_tree_child_tables,
            frontier_active_counts,
            root_frontier_mask,
        )

        b = history.shape[0]
        device = history.device
        dtype = history.dtype
        t_front = root_frontier_mask(b, self.temporal_tree, device, dtype)
        s_front = root_frontier_mask(b, self.spatial_tree, device, dtype)
        left_t, right_t, leaf_t = build_tree_child_tables(self.temporal_tree)
        left_s, right_s, leaf_s = build_tree_child_tables(self.spatial_tree)
        left_t, right_t, leaf_t = left_t.to(device), right_t.to(device), leaf_t.to(device)
        left_s, right_s, leaf_s = left_s.to(device), right_s.to(device), leaf_s.to(device)

        # Target temporal heterogeneity proxy
        # target [B,H,N,C]
        t_var = target.var(dim=2, unbiased=False).mean(dim=-1)  # [B,H]
        # Map leaf variance to tree nodes via cover mean
        cover_t = leaf_cover_t.to(device=device, dtype=dtype)
        node_t_score = (cover_t @ t_var.unsqueeze(-1)).squeeze(-1) / cover_t.sum(-1).clamp_min(1.0).unsqueeze(0)
        # [B, Tnodes]
        s_var = target.var(dim=1, unbiased=False).mean(dim=-1)  # [B,N]
        cover_s = leaf_cover_s.to(device=device, dtype=dtype)
        node_s_score = (cover_s @ s_var.unsqueeze(-1)).squeeze(-1) / cover_s.sum(-1).clamp_min(1.0).unsqueeze(0)

        stage_t, stage_s, stage_v = [], [], []
        remaining = optional_budget.to(device=device, dtype=dtype).clone()
        if remaining.ndim == 0:
            remaining = remaining.expand(b).clone()
        still = torch.ones(b, device=device, dtype=torch.bool)
        h = float(self.temporal_tree.horizon)
        n = float(self.spatial_tree.n_nodes)
        cum = torch.zeros(b, device=device, dtype=dtype)

        for _ in range(self.k_steps):
            do = still & (remaining > 1e-6)
            # Split highest-scoring active non-leaf if beneficial
            allow_t = (t_front > 0.5) & (~leaf_t.unsqueeze(0))
            allow_s = (s_front > 0.5) & (~leaf_s.unsqueeze(0))
            score_t = node_t_score.masked_fill(~allow_t, -1.0)
            score_s = node_s_score.masked_fill(~allow_s, -1.0)
            t_split = torch.zeros_like(t_front)
            s_split = torch.zeros_like(s_front)
            # Pick top-1 temporal and/or spatial by comparing best scores
            t_best = score_t.max(dim=-1).values
            s_best = score_s.max(dim=-1).values
            t_idx = score_t.argmax(dim=-1)
            s_idx = score_s.argmax(dim=-1)
            use_t = do & (t_best >= 0) & (t_best >= s_best * 0.8)
            use_s = do & (s_best >= 0) & (s_best >= t_best * 0.8)
            t_split.scatter_(1, t_idx.unsqueeze(-1), use_t.to(dtype).unsqueeze(-1))
            s_split.scatter_(1, s_idx.unsqueeze(-1), use_s.to(dtype).unsqueeze(-1))
            t_front = apply_batched_splits(t_front, t_split, left_t, right_t, leaf_t)
            s_front = apply_batched_splits(s_front, s_split, left_s, right_s, leaf_s)
            cost = (
                frontier_active_counts(t_front) * frontier_active_counts(s_front)
            ) / (h * n) + 0.05
            cost = cost * do.to(dtype)
            cum = cum + cost
            remaining = (optional_budget - cum).clamp_min(0.0)
            stage_t.append(t_front.clone())
            stage_s.append(s_front.clone())
            stage_v.append(do)
            # Stop if cost spent or no splits
            still = do & ((t_split + s_split).sum(-1) > 0) & (remaining > 1e-6)

        return TeacherProgram(
            temporal_frontiers=torch.stack(stage_t, dim=1),
            spatial_frontiers=torch.stack(stage_s, dim=1),
            stage_valid=torch.stack(stage_v, dim=1),
            notes=f"backend={self.backend}",
        )


def planner_imitation_loss_placeholder(
    planner_out: dict[str, torch.Tensor],
    teacher: TeacherProgram | None,
) -> torch.Tensor:
    """Future Stage-B interface; returns zero when teacher is None."""
    ref = planner_out["continue_logits"]
    if teacher is None:
        return ref.new_zeros(())
    # Simple BCE on continue vs teacher stage_valid
    target = teacher.stage_valid.to(ref.dtype)
    return torch.nn.functional.binary_cross_entropy_with_logits(ref, target)


class BudgetDualController(nn.Module):
    """Non-negative dual interface for future primal-dual training (no step in this task)."""

    def __init__(self, init_value: float = 0.1, lr: float = 0.01):
        super().__init__()
        self.raw = nn.Parameter(torch.tensor(float(init_value)).log())
        self.lr = float(lr)

    @property
    def value(self) -> torch.Tensor:
        return self.raw.exp()

    def dual_ascent_step(self, expected_cost: torch.Tensor, budget: torch.Tensor) -> None:
        """Interface only — callers in this task must not invoke during tests that forbid updates.
        Implementation exists for future training."""
        with torch.no_grad():
            gap = (expected_cost.detach() - budget.detach()).mean()
            self.raw.add_(self.lr * gap)
