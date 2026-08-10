"""Lightweight one-shot route planner over a fixed candidate pool."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


def history_route_features(history: torch.Tensor) -> torch.Tensor:
    """Build planner features from history target channel [B,P,N,C] → [B,F].

    Uses channel 0 (traffic). Avoids a single global mean over (time,node).
    """
    x = history[..., 0]  # [B,P,N]
    b, p, n = x.shape
    # Per-node stats then pool
    t_mean = x.mean(dim=1)  # [B,N]
    t_std = x.std(dim=1, unbiased=False)
    last = x[:, -1, :]
    first = x[:, 0, :]
    slope = last - first
    d1 = (x[:, 1:, :] - x[:, :-1, :]).abs()
    mean_abs_d1 = d1.mean(dim=1)
    max_abs_d1 = d1.max(dim=1).values
    mid = max(p // 2, 1)
    recent = x[:, mid:, :].mean(dim=1)
    early = x[:, :mid, :].mean(dim=1)
    recent_early = recent - early
    node_disp = t_mean.std(dim=-1, unbiased=False)  # [B]
    vol = t_std.mean(dim=-1)

    def _pool(v: torch.Tensor) -> torch.Tensor:
        return torch.stack([v.mean(-1), v.std(-1, unbiased=False), v.max(-1).values], dim=-1)

    parts = [
        _pool(t_mean),
        _pool(t_std),
        _pool(last),
        _pool(slope),
        _pool(mean_abs_d1),
        _pool(max_abs_d1),
        _pool(recent_early),
        node_disp.unsqueeze(-1),
        vol.unsqueeze(-1),
    ]
    # Optional TOD/DOW if present
    if history.shape[-1] >= 3:
        tod = history[..., 1].mean(dim=(1, 2), keepdim=False).unsqueeze(-1)
        dow = history[..., 2].mean(dim=(1, 2), keepdim=False).unsqueeze(-1)
        parts.extend([tod, dow])
    return torch.cat(parts, dim=-1)


class BudgetRoutePlanner(nn.Module):
    """Scores a fixed set of routes; discrete selection via feasibility + argmax."""

    def __init__(self, feat_dim: int, n_routes: int, hidden_dim: int = 64):
        super().__init__()
        self.n_routes = int(n_routes)
        self.net = nn.Sequential(
            nn.Linear(feat_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, n_routes),
        )

    def forward(
        self,
        history: torch.Tensor,
        intensity: float | torch.Tensor,
        route_costs: torch.Tensor,
        budget: torch.Tensor,
        deterministic: bool = True,
    ) -> dict[str, Any]:
        """
        Args:
            history: [B,P,N,C]
            route_costs: [R] or [B,R]
            budget: [B] or scalar
        """
        feats = history_route_features(history)
        b = feats.shape[0]
        device = feats.device
        dtype = feats.dtype
        if not torch.is_tensor(intensity):
            eta = feats.new_full((b, 1), float(intensity))
        else:
            eta = intensity.to(device=device, dtype=dtype).reshape(b, 1)
        logits = self.net(torch.cat([feats, eta], dim=-1))  # [B,R]

        costs = route_costs.to(device=device, dtype=dtype)
        if costs.ndim == 1:
            costs = costs.view(1, -1).expand(b, -1)
        bud = budget.to(device=device, dtype=dtype)
        if bud.ndim == 0:
            bud = bud.expand(b)
        feasible = costs <= bud.unsqueeze(-1)  # [B,R]
        # Always keep cheapest route feasible
        cheapest = costs.argmin(dim=-1)
        feasible = feasible | F.one_hot(cheapest, num_classes=self.n_routes).bool()

        masked_logits = logits.masked_fill(~feasible, -1e9)
        probs = F.softmax(masked_logits, dim=-1)
        expected_cost = (probs * costs).sum(-1)
        if deterministic:
            selected = masked_logits.argmax(dim=-1)
        else:
            selected = torch.multinomial(probs.clamp_min(1e-8), 1).squeeze(-1)
        # Batch mode: use majority / first sample's choice for shared route
        # (caller may override). Per-sample ids still returned.
        selected_cost = costs.gather(1, selected.unsqueeze(-1)).squeeze(-1)
        return {
            "route_logits": logits,
            "masked_route_logits": masked_logits,
            "route_probs": probs,
            "feasible_mask": feasible,
            "selected_route_id": selected,
            "selected_cost": selected_cost,
            "expected_cost": expected_cost,
            "budget": bud,
            "features": feats,
        }
