"""Group-Relative Sequential Forecast Refinement Policy (Plan B).

eta / budget NEVER enter policy features — only hard action masks.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from basicts.archs.arch_zoo.ChainForecasting_arch.sequential_f2f_environment import (
    A0_DIRECT,
    A0_HALF,
    A0_QUARTER,
    A1_JUMP_FINAL,
    A1_REFINE_HALF,
)


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.net(x))


class GroupRelativeRefinementPolicy(nn.Module):
    """Two-stage policy over pre-route context + explicit coarse forecast Z_q."""

    def __init__(self, context_dim: int, zq_dim: int = 1, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.context_dim = int(context_dim)
        self.s0_proj = nn.Sequential(
            nn.Linear(context_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            ResidualBlock(hidden, dropout),
            ResidualBlock(hidden, dropout),
        )
        self.policy0 = nn.Linear(hidden, 3)  # DIRECT, HALF, QUARTER
        self.zq_pool = nn.Sequential(
            nn.Linear(zq_dim, 64),
            nn.GELU(),
            nn.Linear(64, 64),
        )
        self.s0_for_s1 = nn.Linear(hidden, 128)
        self.policy1 = nn.Sequential(
            nn.Linear(128 + 64, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 2),  # JUMP_FINAL, REFINE_HALF
        )

    def encode_s0(self, context: torch.Tensor) -> torch.Tensor:
        # context: [B, D] pooled pre-route feature
        return self.s0_proj(context)

    def logits0(self, s0: torch.Tensor) -> torch.Tensor:
        return self.policy0(s0)

    def logits1(self, s0: torch.Tensor, zq_pooled: torch.Tensor) -> torch.Tensor:
        return self.policy1(torch.cat([self.s0_for_s1(s0), self.zq_pool(zq_pooled)], dim=-1))

    def pool_zq(self, zq: torch.Tensor) -> torch.Tensor:
        """Pool explicit coarse forecast Z_q [B,T,N,C] -> [B, C] (default C=1)."""
        if zq.ndim == 4:
            return zq.mean(dim=(1, 2))  # B,C
        if zq.ndim == 3:
            return zq.mean(dim=1)
        return zq

    def masked_log_softmax(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        neg = torch.finfo(logits.dtype).min
        masked = logits.masked_fill(~mask, neg)
        return F.log_softmax(masked, dim=-1)

    def trajectory_logprob(
        self,
        s0: torch.Tensor,
        zq: torch.Tensor | None,
        a0: torch.Tensor,
        a1: torch.Tensor | None,
        mask0: torch.Tensor,
        mask1: torch.Tensor | None,
    ) -> torch.Tensor:
        """log P(route) for a batch of trajectories."""
        log0 = self.masked_log_softmax(self.logits0(s0), mask0)
        lp = log0.gather(1, a0.view(-1, 1)).squeeze(1)
        needs_a1 = a0 == A0_QUARTER
        if needs_a1.any():
            if zq is None or a1 is None or mask1 is None:
                raise ValueError("quarter trajectories require Z_q, a1, mask1")
            zqp = self.pool_zq(zq)
            log1 = self.masked_log_softmax(self.logits1(s0, zqp), mask1)
            lp = lp + torch.where(
                needs_a1, log1.gather(1, a1.view(-1, 1)).squeeze(1), torch.zeros_like(lp)
            )
        return lp

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
