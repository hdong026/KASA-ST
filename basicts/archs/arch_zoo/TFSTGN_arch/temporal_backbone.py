"""Module 4: Temporal Backbone — BiGRU (default) with optional Mamba fallback."""

import torch
import torch.nn as nn


class BiGRUBackbone(nn.Module):
    """Per-node bidirectional GRU with residual LayerNorm fusion."""

    def __init__(self, hidden_dim: int, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        gru_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=gru_dropout,
        )
        self.fuse = nn.Linear(2 * hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [B, T_prime, N, D]
        Returns:
            h: [B, T_prime, N, D]
        """
        b, t_prime, n, d = z.shape
        z_seq = z.permute(0, 2, 1, 3).reshape(b * n, t_prime, d)  # [B*N, T', D]
        h, _ = self.gru(z_seq)  # [B*N, T', 2D]
        h = self.fuse(h)  # [B*N, T', D]
        h = self.norm(h + z_seq)
        return h.reshape(b, n, t_prime, d).permute(0, 2, 1, 3)  # [B, T', N, D]


class OptionalMambaBackbone(nn.Module):
    """Optional Mamba backbone; falls back to BiGRU when mamba_ssm is unavailable."""

    def __init__(self, hidden_dim: int, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self._fallback = BiGRUBackbone(hidden_dim, num_layers, dropout)
        self._mamba = None
        try:
            from mamba_ssm import Mamba  # type: ignore

            self._mamba = nn.ModuleList(
                [Mamba(d_model=hidden_dim) for _ in range(num_layers)]
            )
            self.norm = nn.LayerNorm(hidden_dim)
        except Exception:
            self._mamba = None

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if self._mamba is None:
            return self._fallback(z)

        b, t_prime, n, d = z.shape
        z_seq = z.permute(0, 2, 1, 3).reshape(b * n, t_prime, d)
        h = z_seq
        for layer in self._mamba:
            h = layer(h)
        h = self.norm(h + z_seq)
        return h.reshape(b, n, t_prime, d).permute(0, 2, 1, 3)


def build_temporal_backbone(
    name: str,
    hidden_dim: int,
    num_layers: int = 2,
    dropout: float = 0.1,
) -> nn.Module:
    if name.lower() == "mamba":
        return OptionalMambaBackbone(hidden_dim, num_layers, dropout)
    return BiGRUBackbone(hidden_dim, num_layers, dropout)
