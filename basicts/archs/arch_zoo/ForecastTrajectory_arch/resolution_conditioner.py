"""Continuous resolution-transition conditioning (FiLM), shared across all edges."""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn


def resolution_features(
    s_prev: int,
    s_next: int,
    H: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Four continuous features describing ``s_prev -> s_next``.

    ``s_prev`` may be START=0.
    """
    sp = float(s_prev)
    sn = float(s_next)
    h = float(H)
    r1 = sp / h
    r2 = sn / h
    r3 = (sn - sp) / h
    r4 = math.log((sn + 1.0) / (sp + 1.0))
    return torch.tensor([r1, r2, r3, r4], device=device, dtype=dtype)


class ResolutionConditioner(nn.Module):
    """MLP that maps resolution features to a conditioning vector and FiLM params."""

    def __init__(self, cond_dim: int = 64, hidden_dim: int = 64, film_dim: int = 64):
        super().__init__()
        self.cond_dim = int(cond_dim)
        self.film_dim = int(film_dim)
        self.mlp = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, cond_dim),
            nn.SiLU(),
        )
        self.film = nn.Linear(cond_dim, 2 * film_dim)
        # Identity FiLM at init: gamma=1, beta=0.
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        with torch.no_grad():
            self.film.bias[:film_dim].fill_(1.0)

    def features(
        self,
        s_prev: int,
        s_next: int,
        H: int,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        feat = resolution_features(s_prev, s_next, H, device, dtype)
        return feat.unsqueeze(0).expand(batch_size, -1).contiguous()

    def forward(
        self,
        s_prev: int,
        s_next: int,
        H: int,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        raw_features: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(e_transition, gamma, beta)``.

        ``e_transition``: ``[B, cond_dim]``
        ``gamma, beta``: ``[B, film_dim]``
        """
        if raw_features is None:
            raw_features = self.features(s_prev, s_next, H, batch_size, device, dtype)
        e = self.mlp(raw_features)
        gb = self.film(e)
        gamma, beta = gb.chunk(2, dim=-1)
        return e, gamma, beta

    @staticmethod
    def apply_film(hidden: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """FiLM: ``hidden`` is ``[B, ..., D]``; gamma/beta are ``[B, D]``."""
        while gamma.ndim < hidden.ndim:
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)
        return gamma * hidden + beta
