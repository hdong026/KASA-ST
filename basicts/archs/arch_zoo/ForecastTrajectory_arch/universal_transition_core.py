"""Shared resolution-conditioned forecast transition core.

ONE parameter set is used for every legal edge, including START→s and s→H.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn

from basicts.archs.arch_zoo.ChainForecasting_arch.kasa_temporal_step import interpolate_forecast
from basicts.archs.arch_zoo.ForecastTrajectory_arch.resolution_conditioner import (
    ResolutionConditioner,
)


def destination_positions(s_next: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Normalized future bin centers ``u_j = (j + 0.5) / s_next``."""
    s = int(s_next)
    j = torch.arange(s, device=device, dtype=dtype)
    return (j + 0.5) / float(s)


class DestinationQueryBuilder(nn.Module):
    """Fourier positional queries of variable destination length, shared weights."""

    def __init__(self, d_model: int, n_freq: int = 8, node_size: int = 307, d_spa: int = 32):
        super().__init__()
        self.d_model = int(d_model)
        self.n_freq = int(n_freq)
        self.node_size = int(node_size)
        self.fourier_dim = 2 * self.n_freq
        self.pos_proj = nn.Linear(self.fourier_dim, d_model)
        self.spa_proj = nn.Linear(d_spa, d_model)
        self.spa_codebook = nn.Parameter(torch.empty(node_size, d_spa))
        nn.init.xavier_uniform_(self.spa_codebook)
        self.start_token = nn.Parameter(torch.zeros(d_model))

    def fourier(self, u: torch.Tensor) -> torch.Tensor:
        freqs = (2.0 * math.pi) * torch.arange(
            1, self.n_freq + 1, device=u.device, dtype=u.dtype
        )
        ang = u.unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)

    def forward(
        self,
        s_next: int,
        batch_size: int,
        num_nodes: int,
        s_prev: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        u = destination_positions(s_next, device, dtype)
        feat = self.fourier(u)
        pos = self.pos_proj(feat)  # [s, D]
        spa = self.spa_proj(self.spa_codebook[:num_nodes].to(dtype=dtype))  # [N, D]
        queries = pos.unsqueeze(0).unsqueeze(2) + spa.unsqueeze(0).unsqueeze(0)
        queries = queries.expand(batch_size, -1, -1, -1).contiguous()
        if int(s_prev) == 0:
            queries = queries + self.start_token.view(1, 1, 1, -1)
        return queries


class StateConditionedTemporalStep(nn.Module):
    """Shared temporal residual block with dynamic destination queries + FiLM.

    This is the KASA TemporalStep *principle* (history tokens → destination
    residual) without per-horizon projection weights.
    """

    def __init__(self, d_model: int, n_heads: int = 4, cy: int = 1):
        super().__init__()
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.cy = int(cy)
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        self.head_dim = self.d_model // self.n_heads
        self.z_proj = nn.Linear(cy, d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.delta_head = nn.Linear(d_model, cy)
        # Zero-init residual so Z_next ≈ Z_bar at start.
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)
        self.hist_skip = nn.Linear(1, cy)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, N, D] -> [B, N, H, T, Dh]
        b, t, n, _ = x.shape
        x = x.view(b, t, n, self.n_heads, self.head_dim)
        return x.permute(0, 2, 3, 1, 4).contiguous()

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, H, T, Dh] -> [B, T, N, D]
        b, n, h, t, dh = x.shape
        x = x.permute(0, 3, 1, 2, 4).contiguous()
        return x.view(b, t, n, h * dh)

    def cross_attend(
        self,
        queries: torch.Tensor,
        history_tokens: torch.Tensor,
    ) -> torch.Tensor:
        q = self._split_heads(self.q_proj(queries))
        k = self._split_heads(self.k_proj(history_tokens))
        v = self._split_heads(self.v_proj(history_tokens))
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-1, -2)) * scale
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        return self.out_proj(self._merge_heads(out))

    def forward(
        self,
        history_tokens: torch.Tensor,
        z_bar: torch.Tensor,
        queries: torch.Tensor,
        gamma: torch.Tensor,
        beta: torch.Tensor,
        history_flow: torch.Tensor,
    ) -> torch.Tensor:
        """Predict residual Δ and return ``Z_bar + Δ``.

        Args:
            history_tokens: ``[B, P, N, D]``
            z_bar: ``[B, s_next, N, Cy]``
            queries: ``[B, s_next, N, D]``
            gamma, beta: ``[B, D]``
            history_flow: ``[B, L, N]``
        """
        z_emb = self.z_proj(z_bar)
        fused = queries + z_emb
        fused = ResolutionConditioner.apply_film(fused, gamma, beta)
        attended = self.cross_attend(fused, history_tokens)
        h = self.norm1(fused + attended)
        h = ResolutionConditioner.apply_film(h, gamma, beta)
        h = self.norm2(h + self.ffn(h))
        h = ResolutionConditioner.apply_film(h, gamma, beta)
        delta = self.delta_head(h)
        # Resolution-agnostic linear residual from history flow.
        hist = history_flow.unsqueeze(-1)
        hist_up = interpolate_forecast(hist, int(z_bar.shape[1]))
        delta = delta + self.hist_skip(hist_up)
        return z_bar + delta


class UniversalTransitionCore(nn.Module):
    """Single shared F_θ called for every legal edge."""

    def __init__(
        self,
        d_model: int = 64,
        cond_dim: int = 64,
        n_heads: int = 4,
        cy: int = 1,
        node_size: int = 307,
        n_freq: int = 8,
        d_spa: int = 32,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.cy = int(cy)
        self.conditioner = ResolutionConditioner(
            cond_dim=cond_dim, hidden_dim=cond_dim, film_dim=d_model
        )
        self.query_builder = DestinationQueryBuilder(
            d_model=d_model, n_freq=n_freq, node_size=node_size, d_spa=d_spa
        )
        self.temporal = StateConditionedTemporalStep(
            d_model=d_model, n_heads=n_heads, cy=cy
        )

    def forward(
        self,
        history_tokens: torch.Tensor,
        z_prev: Optional[torch.Tensor],
        s_prev: int,
        s_next: int,
        H: int,
        history_flow: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, _, num_nodes, _ = history_tokens.shape
        device = history_tokens.device
        dtype = history_tokens.dtype
        s_prev = int(s_prev)
        s_next = int(s_next)
        if s_prev > 0:
            if z_prev is None:
                raise ValueError("z_prev is required when s_prev > 0")
            if int(z_prev.shape[1]) != s_prev:
                raise ValueError(
                    f"Z_prev time dim {z_prev.shape[1]} != s_prev={s_prev}"
                )
            z_bar = interpolate_forecast(z_prev, s_next)
        else:
            z_bar = history_tokens.new_zeros(batch, s_next, num_nodes, self.cy)

        e, gamma, beta = self.conditioner(
            s_prev, s_next, H, batch, device, dtype
        )
        queries = self.query_builder(
            s_next, batch, num_nodes, s_prev, device, dtype
        )
        z_next = self.temporal(
            history_tokens, z_bar, queries, gamma, beta, history_flow
        )
        return z_next, e
