"""Forecast Refinement Gain Controller over shared F2F representations.

Semantic (signed raw-physical MAE gains; no sigmoid/softmax):

    g3_hat  ≈ L_[H] - L_[H/4,H]     benefit of adding H/4 forecast
    g6_hat  ≈ L_[H] - L_[H/2,H]     benefit of adding H/2 forecast
    g36_hat ≈ L_[H/4,H] - L_[H/4,H/2,H]  extra benefit after H/4

Positive => refinement expected to help; negative => expected to hurt.
``eta`` / budget must NEVER enter this module.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int = 256, dropout: float = 0.1):
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


class ForecastRefinementGainController(nn.Module):
    """Compact controller on pre-route shared forecasting features.

    Expects ``H_shared`` with shape ``[B, T, N, D]`` (temporal patches × nodes).
    """

    def __init__(
        self,
        input_dim: int,
        controller_dim: int = 128,
        pooling_queries: int = 4,
        num_heads: int = 4,
        trunk_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.controller_dim = int(controller_dim)
        self.pooling_queries = int(pooling_queries)
        self.input_proj = nn.Sequential(
            nn.Linear(int(input_dim) * 3, self.controller_dim),
            nn.GELU(),
            nn.LayerNorm(self.controller_dim),
        )
        self.queries = nn.Parameter(
            torch.randn(self.pooling_queries, self.controller_dim) * 0.02
        )
        self.pool_attn = nn.MultiheadAttention(
            embed_dim=self.controller_dim,
            num_heads=int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.pool_norm = nn.LayerNorm(self.controller_dim)
        self.trunk_in = nn.Sequential(
            nn.Linear(self.pooling_queries * self.controller_dim, trunk_dim),
            nn.GELU(),
            nn.LayerNorm(trunk_dim),
        )
        self.trunk = nn.Sequential(
            ResidualMLPBlock(trunk_dim, dropout=dropout),
            ResidualMLPBlock(trunk_dim, dropout=dropout),
        )
        self.head_g3 = self._make_head(trunk_dim)
        self.head_g6 = self._make_head(trunk_dim)
        self.head_g36 = self._make_head(trunk_dim)

    @staticmethod
    def _make_head(dim: int) -> nn.Module:
        head = nn.Sequential(
            nn.Linear(dim, 128),
            nn.GELU(),
            nn.Linear(128, 1, bias=False),
        )
        nn.init.zeros_(head[-1].weight)
        return head

    def pool_nodes(self, node_feat: torch.Tensor) -> torch.Tensor:
        # node_feat: [B,N,D_ctrl]
        b = node_feat.shape[0]
        q = self.queries.unsqueeze(0).expand(b, -1, -1)
        upd, _ = self.pool_attn(q, node_feat, node_feat, need_weights=False)
        q = self.pool_norm(q + upd)
        return q.reshape(b, -1)

    def forward(self, h_shared: torch.Tensor) -> dict[str, Any]:
        """
        Args:
            h_shared: [B, T, N, D] shared forecasting representation.
                Must NOT include eta/budget.
        Returns:
            predicted_gains: [B,3] = (g3, g6, g36)
        """
        if h_shared.ndim != 4:
            raise ValueError(
                f"H_shared must be [B,T,N,D], got {tuple(h_shared.shape)}"
            )
        # Temporal pooling over patch axis T
        t_mean = h_shared.mean(dim=1)
        t_last = h_shared[:, -1, :, :]
        if h_shared.shape[1] > 1:
            t_var = (h_shared[:, 1:, :, :] - h_shared[:, :-1, :, :]).abs().mean(dim=1)
        else:
            t_var = torch.zeros_like(t_mean)
        node = self.input_proj(torch.cat([t_mean, t_last, t_var], dim=-1))  # B,N,D
        z = self.trunk(self.trunk_in(self.pool_nodes(node)))
        g3 = self.head_g3(z).squeeze(-1)
        g6 = self.head_g6(z).squeeze(-1)
        g36 = self.head_g36(z).squeeze(-1)
        gains = torch.stack([g3, g6, g36], dim=-1)
        return {
            "predicted_gains": gains,
            "g3_hat": g3,
            "g6_hat": g6,
            "g36_hat": g36,
            "controller_embedding": z,
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
