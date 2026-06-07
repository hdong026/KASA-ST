"""Multi-horizon decoder for the TF spatial refinement branch."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HorizonDecoder(nn.Module):
    """Pool STFT frames, then predict all horizon steps with a shared head."""

    def __init__(
        self,
        hidden_dim: int,
        output_len: int,
        target_dim: int = 1,
        t_prime: int = 7,
        dropout: float = 0.1,
        mode: str = "shared",
    ):
        super().__init__()
        self.output_len = output_len
        self.target_dim = target_dim
        self.mode = mode
        self.time_weights = nn.Parameter(torch.zeros(t_prime))

        mid_dim = max(hidden_dim // 2, 16)
        if mode == "multi_head":
            self.trunk = nn.Sequential(
                nn.Linear(hidden_dim, mid_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.step_heads = nn.ModuleList(
                [nn.Linear(mid_dim, target_dim) for _ in range(output_len)]
            )
            self.head = None
        else:
            self.trunk = nn.Sequential(
                nn.Linear(hidden_dim, mid_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.head = nn.Linear(mid_dim, output_len * target_dim)
            self.step_heads = None

    def _pool_frames(self, h: torch.Tensor) -> torch.Tensor:
        t_prime = h.shape[1]
        if t_prime != self.time_weights.shape[0]:
            weights = F.interpolate(
                self.time_weights.view(1, 1, -1),
                size=t_prime,
                mode="linear",
                align_corners=False,
            ).view(t_prime)
            weights = F.softmax(weights, dim=0)
        else:
            weights = F.softmax(self.time_weights, dim=0)
        return (h * weights.view(1, t_prime, 1, 1)).sum(dim=1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: [B, T_prime, N, D]
        Returns:
            [B, output_len, N, target_dim]
        """
        h_pool = self._pool_frames(h)
        h_shared = self.trunk(h_pool)

        if self.mode == "multi_head":
            steps = [head(h_shared) for head in self.step_heads]
            return torch.stack(steps, dim=1)

        b, n, _ = h_shared.shape
        out = self.head(h_shared)
        return out.reshape(b, n, self.output_len, self.target_dim).permute(0, 2, 1, 3)
