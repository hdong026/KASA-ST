"""Q0 / Q1 networks for Budgeted Bellman Forecast Refinement.

Outputs are scalar Q-values — NO softmax / entropy / probabilities.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


class DepthwiseTemporalPath(nn.Module):
    def __init__(self, in_ch: int, width: int, kernel: int = 3):
        super().__init__()
        self.proj = nn.Linear(in_ch, width)
        self.dw = nn.Conv1d(width, width, kernel_size=kernel, padding=kernel // 2, groups=width)
        self.act = nn.GELU()
        self.ln = nn.LayerNorm(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,P,N,C] -> node tokens via temporal path
        b, p, n, c = x.shape
        h = self.proj(x)  # [B,P,N,W]
        h = h.permute(0, 2, 3, 1).reshape(b * n, h.shape[-1], p)  # [BN,W,P]
        h = self.dw(h)
        h = h.permute(0, 2, 1)  # [BN,P,W]
        h = self.act(h)
        h = self.ln(h)
        # pool over time
        h = h.mean(dim=1)  # [BN,W]
        return h.view(b, n, -1)


class LearnedQueryPool(nn.Module):
    def __init__(self, dim: int, n_queries: int = 4, n_heads: int = 4, out_dim: int | None = None):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(n_queries, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.out = nn.Linear(n_queries * dim, out_dim or dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [B,N,D]
        b = tokens.shape[0]
        q = self.queries.unsqueeze(0).expand(b, -1, -1)
        out, _ = self.attn(q, tokens, tokens, need_weights=False)
        return self.out(out.reshape(b, -1))


class HistoryEncoder(nn.Module):
    """Encode raw history X [B,P,N,Cx] without global pooling collapse."""

    def __init__(
        self,
        in_ch: int = 4,
        node_width: int = 96,
        out_dim: int = 256,
        n_queries: int = 4,
        n_heads: int = 4,
        active_channels: tuple[int, ...] = (0, 1, 2, 3),
    ):
        super().__init__()
        self.active_channels = tuple(active_channels)
        c = len(self.active_channels)
        self.stats_proj = nn.Linear(3, node_width)  # mean, last, abs-var
        self.temp_path = DepthwiseTemporalPath(c, node_width)
        self.fuse = nn.Sequential(
            nn.Linear(2 * node_width, node_width),
            nn.GELU(),
            nn.LayerNorm(node_width),
        )
        self.pool = LearnedQueryPool(node_width, n_queries=n_queries, n_heads=n_heads, out_dim=out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,P,N,C]
        x = x[..., list(self.active_channels)]
        # per-node temporal stats on channel-0 traffic primarily + mean over channels
        traffic = x[..., 0]  # [B,P,N]
        t_mean = traffic.mean(dim=1)
        t_last = traffic[:, -1, :]
        t_abs = (traffic[:, 1:, :] - traffic[:, :-1, :]).abs().mean(dim=1)
        stats = torch.stack([t_mean, t_last, t_abs], dim=-1)  # [B,N,3]
        s = self.stats_proj(stats)
        t = self.temp_path(x)
        tokens = self.fuse(torch.cat([s, t], dim=-1))
        return self.pool(tokens)  # [B,out_dim]


class ZqEncoder(nn.Module):
    def __init__(self, in_ch: int = 1, node_width: int = 64, out_dim: int = 128, n_queries: int = 4, n_heads: int = 4):
        super().__init__()
        self.desc_proj = nn.Linear(5, node_width)  # mean,last,slope,absvar,std
        self.pool = LearnedQueryPool(node_width, n_queries=n_queries, n_heads=n_heads, out_dim=out_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B,q,N,Cy] — use channel 0
        if z.dim() == 4:
            z0 = z[..., 0]
        else:
            z0 = z
        # [B,q,N]
        mean = z0.mean(dim=1)
        last = z0[:, -1, :]
        if z0.shape[1] >= 2:
            slope = (z0[:, -1, :] - z0[:, 0, :]) / max(z0.shape[1] - 1, 1)
            absvar = (z0[:, 1:, :] - z0[:, :-1, :]).abs().mean(dim=1)
        else:
            slope = torch.zeros_like(mean)
            absvar = torch.zeros_like(mean)
        std = z0.std(dim=1, unbiased=False)
        desc = torch.stack([mean, last, slope, absvar, std], dim=-1)
        tokens = self.desc_proj(desc)
        return self.pool(tokens)


class Q0Net(nn.Module):
    """Outputs Q0_f, Q0_m, Q0_q (no softmax)."""

    def __init__(
        self,
        in_ch: int = 4,
        history_dim: int = 256,
        node_width: int = 96,
        active_channels: tuple[int, ...] = (0, 1, 2, 3),
    ):
        super().__init__()
        self.encoder = HistoryEncoder(
            in_ch=in_ch,
            node_width=node_width,
            out_dim=history_dim,
            active_channels=active_channels,
        )
        self.head = nn.Sequential(
            nn.Linear(history_dim + 1 + 3, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 3),
        )
        # zero-init last layer for stable start
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        budget_norm: torch.Tensor,
        s0_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.encoder(x)
        if budget_norm.dim() == 1:
            budget_norm = budget_norm.unsqueeze(-1)
        if s0_mask is None:
            mask_emb = torch.ones(x.shape[0], 3, device=x.device, dtype=x.dtype)
        else:
            mask_emb = s0_mask.float()
        inp = torch.cat([h, budget_norm, mask_emb], dim=-1)
        return self.head(inp)  # [B,3]


class Q1Net(nn.Module):
    """Outputs Q1_f, Q1_m (no softmax). Independent history encoder from Q0."""

    def __init__(
        self,
        in_ch: int = 4,
        history_dim: int = 256,
        zq_dim: int = 128,
        node_width: int = 96,
        active_channels: tuple[int, ...] = (0, 1, 2, 3),
    ):
        super().__init__()
        self.hist = HistoryEncoder(
            in_ch=in_ch,
            node_width=node_width,
            out_dim=history_dim,
            active_channels=active_channels,
        )
        self.zq = ZqEncoder(out_dim=zq_dim)
        self.head = nn.Sequential(
            nn.Linear(history_dim + zq_dim + 1 + 2, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 2),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        z_q: torch.Tensor,
        budget_norm: torch.Tensor,
        sq_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.hist(x)
        z = self.zq(z_q)
        if budget_norm.dim() == 1:
            budget_norm = budget_norm.unsqueeze(-1)
        if sq_mask is None:
            mask_emb = torch.ones(x.shape[0], 2, device=x.device, dtype=x.dtype)
        else:
            mask_emb = sq_mask.float()
        inp = torch.cat([h, z, budget_norm, mask_emb], dim=-1)
        return self.head(inp)  # [B,2]


class BellmanRefinementRouter(nn.Module):
    """Container for Q0+Q1 used at inference."""

    def __init__(self, q0: Q0Net, q1: Q1Net, *, global_return_scale: float, c_max: float):
        super().__init__()
        self.q0 = q0
        self.q1 = q1
        self.global_return_scale = float(global_return_scale)
        self.c_max = float(c_max)

    def normalize_budget(self, b: torch.Tensor | float) -> torch.Tensor:
        if not torch.is_tensor(b):
            b = torch.tensor([float(b)], dtype=torch.float32)
        return (b.float() / max(self.c_max, 1e-8)).view(-1, 1)
