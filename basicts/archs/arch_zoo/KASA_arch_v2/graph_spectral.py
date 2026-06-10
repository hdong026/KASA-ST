"""Online graph spectral calibration for KASA v2."""

from __future__ import annotations

import torch
import torch.nn as nn


class GraphSpectralCalibration(nn.Module):
    """Apply graph spectral low/high calibration on post-spatial predictions."""

    def __init__(self, **model_args):
        super().__init__()
        self.mode = str(model_args.get("graph_spectral_calibration_mode", "none")).lower()
        self.k = int(model_args.get("graph_spectral_k", 32))
        self.hidden = int(model_args.get("graph_spectral_hidden", 16))
        self.dropout_rate = float(model_args.get("graph_spectral_dropout", 0.0))
        self.gamma_init = float(model_args.get("graph_spectral_gamma_init", 0.0))
        self.eps = 1e-8

        if self.mode == "low":
            mlp_in = 1
        elif self.mode in {"low_high", "residual_safe_low_high"}:
            mlp_in = 2
        else:
            mlp_in = 0

        if mlp_in > 0:
            self.mlp = nn.Sequential(
                nn.Linear(mlp_in, self.hidden),
                nn.SiLU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.hidden, 1),
            )
            self.gamma = nn.Parameter(torch.tensor(self.gamma_init, dtype=torch.float32))

        self._cached_adj_key = None
        self._cached_p_low = None

    def _adj_key(self, adj: torch.Tensor) -> tuple:
        if adj.dim() == 2:
            return ("global", adj.shape[0], float(adj.sum().item()))
        return ("batch", adj.shape[0], adj.shape[1], float(adj.sum().item()))

    def _projector_low(self, adaptive_adj: torch.Tensor) -> torch.Tensor:
        if adaptive_adj.dim() == 3:
            a = adaptive_adj[0]
        else:
            a = adaptive_adj

        cache_key = self._adj_key(a)
        if self._cached_adj_key == cache_key and self._cached_p_low is not None:
            return self._cached_p_low

        a_sym = 0.5 * (a + a.transpose(0, 1))
        n = a_sym.shape[0]
        a_sym = a_sym + self.eps * torch.eye(n, device=a.device, dtype=a.dtype)

        deg = a_sym.sum(dim=1)
        d_inv_sqrt = torch.pow(deg + self.eps, -0.5)
        d_mat = torch.diag(d_inv_sqrt)
        lap = torch.eye(n, device=a.device, dtype=a.dtype) - d_mat @ a_sym @ d_mat

        _, eigvecs = torch.linalg.eigh(lap)
        k_eff = min(self.k, n)
        u_low = eigvecs[:, :k_eff]
        p_low = u_low @ u_low.transpose(0, 1)

        if adaptive_adj.dim() == 2:
            self._cached_adj_key = cache_key
            self._cached_p_low = p_low

        return p_low

    def forward(self, y: torch.Tensor, adaptive_adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y: [B, H, N, 1]
            adaptive_adj: [N, N] or [B, N, N]
        """
        if self.mode == "none":
            return y

        p_low = self._projector_low(adaptive_adj)
        y_flat = y.squeeze(-1)
        y_low = torch.einsum("nm,bhm->bhn", p_low, y_flat).unsqueeze(-1)
        y_high = y - y_low

        if self.mode == "low":
            feat = y_low
        else:
            feat = torch.cat([y_low, y_high], dim=-1)

        delta = self.mlp(feat)
        return y + self.gamma * delta
