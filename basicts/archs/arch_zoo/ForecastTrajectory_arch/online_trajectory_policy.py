"""Lightweight online trajectory policy π_φ(s_next | h_X, Z, s, λ, remaining budget)."""

from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch import nn

from basicts.archs.arch_zoo.ForecastTrajectory_arch.resolution_conditioner import (
    resolution_features,
)
from basicts.archs.arch_zoo.ForecastTrajectory_arch.trajectory_graph import (
    ForecastTrajectoryGraph,
)


def forecast_descriptors(z: Optional[torch.Tensor]) -> torch.Tensor:
    """Lightweight Z descriptors: mean, std, last, slope, abs-variation.

    Returns ``[B, 5]`` (channel-averaged).
    """
    if z is None:
        raise ValueError("forecast_descriptors requires a tensor or use zeros_like_batch")
    # z: [B, S, N, Cy]
    flat = z.mean(dim=2)  # [B, S, Cy]
    if flat.shape[-1] > 1:
        flat = flat.mean(dim=-1, keepdim=True)
    seq = flat.squeeze(-1)  # [B, S]
    mean = seq.mean(dim=1)
    std = seq.std(dim=1, unbiased=False)
    last = seq[:, -1]
    if seq.shape[1] >= 2:
        slope = seq[:, -1] - seq[:, 0]
        abs_var = (seq[:, 1:] - seq[:, :-1]).abs().mean(dim=1)
    else:
        slope = torch.zeros_like(mean)
        abs_var = torch.zeros_like(mean)
    return torch.stack([mean, std, last, slope, abs_var], dim=-1)


class OnlineTrajectoryPolicy(nn.Module):
    """Masked softmax over configured destination states. Target: < 200k params."""

    def __init__(
        self,
        graph: ForecastTrajectoryGraph,
        d_history: int = 64,
        hidden_dim: int = 128,
        n_z_desc: int = 5,
    ):
        super().__init__()
        self.graph = graph
        self.dest_states = list(graph.states)
        self.n_dest = len(self.dest_states)
        self.state_to_index = {int(s): i for i, s in enumerate(self.dest_states)}
        self.d_history = int(d_history)
        self.hidden_dim = int(hidden_dim)

        self.hist_mlp = nn.Sequential(
            nn.Linear(d_history, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.z_mlp = nn.Sequential(
            nn.Linear(n_z_desc, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.z_node_query = nn.Parameter(torch.zeros(d_history))
        self.z_node_proj = nn.Linear(1, d_history)
        ctx_dim = 4 + 3  # resolution features + (lambda, b_norm, no_budget)
        self.ctx_mlp = nn.Sequential(
            nn.Linear(ctx_dim, hidden_dim),
            nn.SiLU(),
        )
        fuse_in = hidden_dim * 3
        self.fuse = nn.Sequential(
            nn.Linear(fuse_in, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.n_dest),
        )
        n_params = sum(p.numel() for p in self.parameters())
        print(f"[ForecastTrajectory] policy parameter count = {n_params}")
        if n_params >= 200_000:
            print(
                f"[ForecastTrajectory] WARNING: policy params {n_params} >= 200k target"
            )

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def zeros_z_desc(self, batch_size: int, device, dtype) -> torch.Tensor:
        return torch.zeros(batch_size, 5, device=device, dtype=dtype)

    def encode_z(self, z: Optional[torch.Tensor], batch_size: int, device, dtype) -> torch.Tensor:
        if z is None:
            return self.zeros_z_desc(batch_size, device, dtype)
        desc = forecast_descriptors(z)
        # learned node pooling on channel-0
        z0 = z[..., :1]
        node_h = self.z_node_proj(z0)  # [B,S,N,D]
        scores = (node_h * self.z_node_query.view(1, 1, 1, -1)).sum(dim=-1)
        w = torch.softmax(scores.mean(dim=1), dim=-1)  # [B, N]
        pooled = (z0.mean(dim=1).squeeze(-1) * w).sum(dim=-1, keepdim=True)
        # fold pooled into mean channel (already have mean); keep desc as-is
        del pooled
        return desc

    def legal_mask(
        self,
        s_current: int,
        batch_size: int,
        device,
        feasible_dest: Optional[Sequence[int]] = None,
    ) -> torch.Tensor:
        legal = torch.zeros(self.n_dest, dtype=torch.bool, device=device)
        succ = set(self.graph.successors(int(s_current)))
        if feasible_dest is not None:
            succ = succ.intersection({int(s) for s in feasible_dest})
        for s, idx in self.state_to_index.items():
            if s in succ:
                legal[idx] = True
        return legal.unsqueeze(0).expand(batch_size, -1).contiguous()

    def forward(
        self,
        h_history: torch.Tensor,
        z_current: Optional[torch.Tensor],
        s_current: int,
        lam: torch.Tensor,
        remaining_norm: torch.Tensor,
        no_budget: torch.Tensor,
        H: int,
        feasible_dest: Optional[Sequence[int]] = None,
    ) -> dict:
        """Return logits and masked probabilities over destination states.

        Args:
            h_history: ``[B, D]`` pooled shared history
            z_current: ``[B, s, N, Cy]`` or None at START
            s_current: int
            lam: ``[B]`` or ``[B, 1]``
            remaining_norm: ``[B]`` remaining budget / C_dense
            no_budget: ``[B]`` 1 if no hard budget
        """
        batch = h_history.shape[0]
        device = h_history.device
        dtype = h_history.dtype
        if lam.ndim > 1:
            lam = lam.reshape(batch)
        if remaining_norm.ndim > 1:
            remaining_norm = remaining_norm.reshape(batch)
        if no_budget.ndim > 1:
            no_budget = no_budget.reshape(batch)

        z_desc = self.encode_z(z_current, batch, device, dtype)
        h = self.hist_mlp(h_history)
        z_h = self.z_mlp(z_desc)
        feat = resolution_features(int(s_current), H, H, device, dtype)
        feat = feat.unsqueeze(0).expand(batch, -1)
        ctx = torch.cat(
            [
                feat,
                lam.unsqueeze(-1).to(dtype),
                remaining_norm.unsqueeze(-1).to(dtype),
                no_budget.unsqueeze(-1).to(dtype),
            ],
            dim=-1,
        )
        ctx_h = self.ctx_mlp(ctx)
        logits = self.fuse(torch.cat([h, z_h, ctx_h], dim=-1))
        mask = self.legal_mask(s_current, batch, device, feasible_dest=feasible_dest)
        neg_large = torch.finfo(logits.dtype).min / 4
        masked_logits = logits.masked_fill(~mask, neg_large)
        # If a row is fully masked, keep uniform over configured dest to avoid NaN;
        # caller should treat this as PATH_PROBABILITY_INVALID.
        all_masked = ~mask.any(dim=-1)
        if all_masked.any():
            masked_logits = masked_logits.clone()
            masked_logits[all_masked] = 0.0
        log_probs = torch.log_softmax(masked_logits, dim=-1)
        probs = torch.softmax(masked_logits, dim=-1)
        probs = probs * mask.float()
        denom = probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        probs = probs / denom
        return {
            "logits": logits,
            "masked_logits": masked_logits,
            "log_probs": log_probs,
            "probs": probs,
            "mask": mask,
            "all_masked": all_masked,
        }
