"""Plan B-v2 sequential forecast refinement policy with structured states.

Exact full-information trajectory policy — NOT original GRPO/GSPO.
Independent weights from Plan A gain controller (idea reuse only).
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from basicts.archs.arch_zoo.ChainForecasting_arch.exact_trajectory_policy_objective import (
    action_masks_from_feasible,
    exact_terminal_route_probs,
)
from basicts.archs.arch_zoo.ChainForecasting_arch.sequential_f2f_environment import (
    A0_DIRECT,
    A0_HALF,
    A0_QUARTER,
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


class LearnedQueryPool(nn.Module):
    """Lightweight cross-node learned-query attention (Plan-A idea, own weights)."""

    def __init__(
        self,
        dim: int,
        num_queries: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_queries = int(num_queries)
        self.queries = nn.Parameter(torch.randn(self.num_queries, dim) * 0.02)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=int(num_heads), dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, node_tokens: torch.Tensor) -> torch.Tensor:
        # node_tokens: [B,N,D]
        b = node_tokens.shape[0]
        q = self.queries.unsqueeze(0).expand(b, -1, -1)
        upd, _ = self.attn(q, node_tokens, node_tokens, need_weights=False)
        q = self.norm(q + upd)
        return q.reshape(b, -1)  # [B, Q*D]


class StructuredState0Encoder(nn.Module):
    """H_shared [B,M,N,D] -> hidden [B,256] preserving temporal+node structure."""

    def __init__(
        self,
        input_dim: int,
        node_proj: int = 128,
        hidden: int = 256,
        num_queries: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.node_proj_dim = int(node_proj)
        self.hidden = int(hidden)
        self.node_proj = nn.Sequential(
            nn.Linear(3 * int(input_dim), node_proj),
            nn.GELU(),
            nn.LayerNorm(node_proj),
        )
        self.pool = LearnedQueryPool(node_proj, num_queries, num_heads, dropout)
        self.trunk_in = nn.Sequential(
            nn.Linear(num_queries * node_proj, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.trunk = nn.Sequential(
            ResidualBlock(hidden, dropout),
            ResidualBlock(hidden, dropout),
        )

    def temporal_node_descriptors(self, h_shared: torch.Tensor) -> torch.Tensor:
        """Per-node [mean, last, abs_variation] -> [B,N,3D]."""
        if h_shared.ndim != 4:
            raise ValueError(f"H_shared must be [B,M,N,D], got {tuple(h_shared.shape)}")
        h_mean = h_shared.mean(dim=1)
        h_last = h_shared[:, -1]
        if h_shared.shape[1] > 1:
            h_abs = (h_shared[:, 1:] - h_shared[:, :-1]).abs().mean(dim=1)
        else:
            h_abs = torch.zeros_like(h_mean)
        return torch.cat([h_mean, h_last, h_abs], dim=-1)  # B,N,3D

    def forward(self, h_shared: torch.Tensor) -> dict[str, torch.Tensor]:
        u0 = self.temporal_node_descriptors(h_shared)  # B,N,3D
        nodes = self.node_proj(u0)  # B,N,128
        pooled = self.pool(nodes)
        hidden = self.trunk(self.trunk_in(pooled))
        return {
            "u0": u0,
            "node_tokens": nodes,
            "state0_hidden": hidden,
        }


class StructuredZqEncoder(nn.Module):
    """Explicit Z_q [B,T,N,C] -> structured hidden (NOT scalar mean)."""

    def __init__(
        self,
        z_channels: int = 1,
        node_proj: int = 64,
        out_dim: int = 128,
        num_queries: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.z_channels = int(z_channels)
        # 5 descriptors × C
        in_dim = 5 * int(z_channels)
        self.node_proj = nn.Sequential(
            nn.Linear(in_dim, node_proj),
            nn.GELU(),
            nn.LayerNorm(node_proj),
        )
        self.pool = LearnedQueryPool(node_proj, num_queries, num_heads, dropout)
        # also keep mean+std of node tokens as light spatial summary
        self.out = nn.Sequential(
            nn.Linear(num_queries * node_proj + 2 * node_proj, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim),
            ResidualBlock(out_dim, dropout),
        )

    def node_descriptors(self, zq: torch.Tensor) -> torch.Tensor:
        """Per-node temporal mean/last/slope/abs_var/std -> [B,N,5C]."""
        if zq.ndim != 4:
            raise ValueError(f"Z_q must be [B,T,N,C], got {tuple(zq.shape)}")
        b, t, n, c = zq.shape
        z_mean = zq.mean(dim=1)
        z_last = zq[:, -1]
        z_std = zq.std(dim=1, unbiased=False)
        if t > 1:
            z_abs = (zq[:, 1:] - zq[:, :-1]).abs().mean(dim=1)
            # slope via centered time regression
            tt = torch.linspace(0, 1, t, device=zq.device, dtype=zq.dtype).view(1, t, 1, 1)
            tt = tt - tt.mean()
            z_slope = ((zq - z_mean.unsqueeze(1)) * tt).sum(dim=1) / (
                (tt * tt).sum() + 1e-8
            )
        else:
            z_abs = torch.zeros_like(z_mean)
            z_slope = torch.zeros_like(z_mean)
        return torch.cat([z_mean, z_last, z_slope, z_abs, z_std], dim=-1)

    def forward(self, zq: torch.Tensor) -> dict[str, torch.Tensor]:
        desc = self.node_descriptors(zq)  # B,N,5C — NEVER scalar-reduce first
        nodes = self.node_proj(desc)
        pooled = self.pool(nodes)
        spat = torch.cat([nodes.mean(dim=1), nodes.std(dim=1, unbiased=False)], dim=-1)
        hidden = self.out(torch.cat([pooled, spat], dim=-1))
        return {
            "zq_node_descriptors": desc,
            "zq_node_tokens": nodes,
            "zq_hidden": hidden,
        }


class GroupRelativeRefinementPolicyV2(nn.Module):
    """Two-stage policy with structured state0/state1 and zero-init action heads."""

    def __init__(
        self,
        h_dim: int,
        z_channels: int = 1,
        hidden: int = 256,
        node_proj0: int = 128,
        zq_out: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.h_dim = int(h_dim)
        self.hidden = int(hidden)
        self.state0 = StructuredState0Encoder(
            input_dim=h_dim, node_proj=node_proj0, hidden=hidden, dropout=dropout
        )
        self.state1_zq = StructuredZqEncoder(
            z_channels=z_channels, node_proj=64, out_dim=zq_out, dropout=dropout
        )
        self.s0_for_s1 = nn.Linear(hidden, 128)
        self.policy1_mlp = nn.Sequential(
            nn.Linear(128 + zq_out, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            ResidualBlock(256, dropout),
            nn.Linear(256, 128),
            nn.GELU(),
        )
        # Zero-init action heads — uniform before masks
        self.policy0 = nn.Linear(hidden, 3)
        self.policy1 = nn.Linear(128, 2)
        nn.init.zeros_(self.policy0.weight)
        nn.init.zeros_(self.policy0.bias)
        nn.init.zeros_(self.policy1.weight)
        nn.init.zeros_(self.policy1.bias)

    def encode_state0(self, h_shared: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.state0(h_shared)

    def encode_zq(self, zq: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.state1_zq(zq)

    def logits0(self, state0_hidden: torch.Tensor) -> torch.Tensor:
        return self.policy0(state0_hidden)

    def logits1(self, state0_hidden: torch.Tensor, zq_hidden: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.s0_for_s1(state0_hidden), zq_hidden], dim=-1)
        return self.policy1(self.policy1_mlp(x))

    def masked_log_softmax(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if mask.ndim == 1:
            mask = mask.unsqueeze(0).expand(logits.shape[0], -1)
        neg = torch.finfo(logits.dtype).min
        masked = logits.masked_fill(~mask, neg)
        return F.log_softmax(masked, dim=-1)

    def forward_terminal_probs(
        self,
        h_shared: torch.Tensor,
        zq: torch.Tensor | None,
        feasible_routes: torch.Tensor,
        index_map: dict[str, int],
    ) -> dict[str, Any]:
        """Compute exact terminal route probabilities for a feasibility regime."""
        s0 = self.encode_state0(h_shared)
        hidden = s0["state0_hidden"]
        masks = action_masks_from_feasible(feasible_routes, index_map=index_map)
        m0 = masks["mask0"].to(hidden.device)
        m1 = masks["mask1"].to(hidden.device)
        log0 = self.masked_log_softmax(self.logits0(hidden), m0)
        # Z_q only needed if QUARTER is allowed
        if bool(m0[A0_QUARTER].item()) if m0.ndim == 1 else bool(m0[:, A0_QUARTER].any().item()):
            if zq is None:
                raise ValueError("Z_q required when QUARTER is feasible")
            zenc = self.encode_zq(zq)
            log1 = self.masked_log_softmax(self.logits1(hidden, zenc["zq_hidden"]), m1)
            zq_info = zenc
        else:
            # dummy uniform log1 (masked to zeros contribution)
            b = hidden.shape[0]
            log1 = torch.zeros(b, 2, device=hidden.device)
            log1 = log1 - torch.log(torch.tensor(2.0, device=hidden.device))
            zq_info = {}
        probs = exact_terminal_route_probs(
            log0, log1, m0, m1, index_map=index_map, n_routes=int(feasible_routes.numel())
        )
        return {
            "route_probs": probs,
            "log0": log0,
            "log1": log1,
            "mask0": m0,
            "mask1": m1,
            "state0_hidden": hidden,
            **s0,
            **zq_info,
        }

    def select_actions_deterministic(
        self,
        h_shared: torch.Tensor,
        zq: torch.Tensor | None,
        feasible_routes: torch.Tensor,
        index_map: dict[str, int],
    ) -> dict[str, torch.Tensor]:
        out = self.forward_terminal_probs(h_shared, zq, feasible_routes, index_map)
        route_ids = out["route_probs"].argmax(dim=-1)
        a0 = out["log0"].argmax(dim=-1)
        a1 = out["log1"].argmax(dim=-1)
        return {"route_ids": route_ids, "a0": a0, "a1": a1, **out}

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
