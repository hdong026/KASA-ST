"""Shared resolution-conditioned forecast cell (dynamic T/S active units)."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from basicts.archs.arch_zoo.ChainForecasting_arch.adaptive_resolution_hierarchy import (
    SpatialResolutionTree,
    TemporalResolutionTree,
)


class ResolutionConditionAdapter(nn.Module):
    """Condition-only residual adapter (zero-init out). Never edits supervised forecast."""

    def __init__(self, feat_dim: int = 8, hidden_dim: int = 16, epsilon: float = 0.02):
        super().__init__()
        self.epsilon = float(epsilon)
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(
        self,
        current: torch.Tensor,
        previous_aligned: torch.Tensor,
        meta: torch.Tensor,
    ) -> torch.Tensor:
        """
        current/previous_aligned: [B,T,S,1]
        meta: [B,T,S,F-2] extra channels (or broadcastable)
        """
        delta = current - previous_aligned
        x = torch.cat([current, previous_aligned, delta, meta], dim=-1)
        raw = self.mlp(x)
        scale = current.detach().abs().mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-3)
        return current + self.epsilon * torch.tanh(raw) * scale


class ResolutionConditionedForecastCell(nn.Module):
    """Query-based temporal decoder + adaptive spatial relations on active units."""

    def __init__(
        self,
        history_len: int,
        input_dim: int,
        hidden_dim: int = 32,
        output_dim: int = 1,
        ponder_embed_dim: int = 8,
    ):
        super().__init__()
        self.history_len = int(history_len)
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.history_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.time_query_mlp = nn.Sequential(
            nn.Linear(6 + ponder_embed_dim + 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=4, batch_first=True
        )
        self.direct_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.spatial_src = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.spatial_dst = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.spatial_mix = nn.Parameter(torch.tensor(0.1))
        self.ponder_embed = nn.Embedding(64, ponder_embed_dim)
        self.cluster_proj = nn.Linear(hidden_dim, hidden_dim)

    def _time_queries(
        self,
        temporal_tree: TemporalResolutionTree,
        temporal_frontier_mask: torch.Tensor,
        t_max: int,
        ponder_step: int,
        thinking_intensity: float,
        remaining_budget: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return queries [B,Tmax,Hdim], meta [B,Tmax,6], leaf_mask [B,Tmax]."""
        b = temporal_frontier_mask.shape[0]
        device = temporal_frontier_mask.device
        h = temporal_tree.horizon
        queries = torch.zeros(b, t_max, self.hidden_dim, device=device)
        meta = torch.zeros(b, t_max, 6, device=device)
        leaf_mask = torch.zeros(b, t_max, device=device)
        step_e = self.ponder_embed(
            torch.tensor(min(ponder_step, 63), device=device, dtype=torch.long)
        )
        for bi in range(b):
            ids = torch.nonzero(temporal_frontier_mask[bi] > 0.5, as_tuple=False).flatten()
            rem = float(remaining_budget[bi].detach().item())
            for slot, nid in enumerate(ids.tolist()):
                if slot >= t_max:
                    break
                node = temporal_tree.nodes[nid]
                start = node.start / h
                end = node.end / h
                center = 0.5 * (start + end)
                width = end - start
                depth = node.depth / max(temporal_tree.depth, 1)
                feats = torch.tensor(
                    [start, end, center, width, depth, float(node.is_leaf)],
                    device=device,
                )
                meta[bi, slot] = feats
                leaf_mask[bi, slot] = 1.0 if node.is_leaf else 0.0
                q_in = torch.cat(
                    [
                        feats,
                        step_e,
                        torch.tensor(
                            [thinking_intensity, rem], device=device
                        ),
                    ]
                )
                queries[bi, slot] = self.time_query_mlp(q_in)
        return queries, meta, leaf_mask

    def forward(
        self,
        history: torch.Tensor,
        temporal_tree: TemporalResolutionTree,
        spatial_tree: SpatialResolutionTree,
        temporal_frontier_mask: torch.Tensor,
        spatial_frontier_mask: torch.Tensor,
        p_t: torch.Tensor,
        p_s: torch.Tensor,
        t_active_mask: torch.Tensor,
        s_active_mask: torch.Tensor,
        previous_condition: torch.Tensor | None,
        ponder_step: int,
        thinking_intensity: float,
        remaining_budget: torch.Tensor,
    ) -> dict[str, Any]:
        """
        history: [B,P,N,Cx]
        previous_condition: [B,Tmax,Smax,Cy] or None
        """
        b, p, n, cx = history.shape
        t_max = p_t.shape[1]
        s_max = p_s.shape[1]
        device = history.device

        # Encode history nodes then pool to active spatial units: [B,P,S,H]
        h_enc = self.history_encoder(history)  # [B,P,N,H]
        # Aggregate over members via P_s: [B,S,N] @ [B,P,N,H] -> [B,P,S,H]
        h_space = torch.einsum("bsn,bpnh->bpsh", p_s, h_enc)
        # Mean over history time as KV base: [B,S,H]
        kv = h_space.mean(dim=1)
        kv = self.cluster_proj(kv)

        queries, t_meta, t_leaf = self._time_queries(
            temporal_tree,
            temporal_frontier_mask,
            t_max,
            ponder_step,
            thinking_intensity,
            remaining_budget,
        )

        # Cross-attention: Q time queries, K/V spatial clusters (shared across queries)
        # Expand KV per batch already [B,S,H]
        attn_mask = s_active_mask < 0.5  # True = ignore
        # MHA expects key_padding_mask [B,S]
        attn_out, _ = self.cross_attn(
            queries, kv, kv, key_padding_mask=attn_mask, need_weights=False
        )
        # Broadcast spatial context into each time query and decode
        # Pair each (t,s): concat query_t and kv_s
        q_exp = queries.unsqueeze(2).expand(b, t_max, s_max, self.hidden_dim)
        kv_exp = kv.unsqueeze(1).expand(b, t_max, s_max, self.hidden_dim)
        paired = torch.cat([q_exp, kv_exp], dim=-1)
        coarse = self.direct_head(paired)  # [B,T,S,Cy]

        # Adaptive spatial relations among active clusters, applied per time
        src = F.normalize(self.spatial_src(kv), dim=-1)
        dst = F.normalize(self.spatial_dst(kv), dim=-1)
        logits = torch.matmul(src, dst.transpose(1, 2)) / math.sqrt(self.hidden_dim)
        logits = logits.masked_fill(attn_mask.unsqueeze(1), -1e4)
        logits = logits.masked_fill(attn_mask.unsqueeze(2), -1e4)
        rel = torch.softmax(logits, dim=-1)  # [B,S,S]
        # Propagate over spatial dim of coarse
        c = coarse.squeeze(-1) if coarse.shape[-1] == 1 else coarse.mean(dim=-1)
        # c: [B,T,S]
        c_ref = torch.einsum("bsj,btj->bts", rel, c)
        alpha = torch.sigmoid(self.spatial_mix)
        c_mix = c + alpha * (c_ref - c)
        coarse = c_mix.unsqueeze(-1)

        # Mask inactive slots
        slot_mask = t_active_mask.unsqueeze(-1) * s_active_mask.unsqueeze(1)
        coarse = coarse * slot_mask.unsqueeze(-1)

        # Spatial leaf mask for controller
        s_leaf = torch.zeros(b, s_max, device=device)
        for bi in range(b):
            ids = torch.nonzero(spatial_frontier_mask[bi] > 0.5, as_tuple=False).flatten()
            for slot, nid in enumerate(ids.tolist()):
                if slot >= s_max:
                    break
                s_leaf[bi, slot] = 1.0 if spatial_tree.nodes[nid].is_leaf else 0.0

        cell_features = {
            "attn_out": attn_out,
            "cluster_kv": kv,
            "temporal_meta": t_meta,
            "temporal_leaf_mask": t_leaf,
            "spatial_leaf_mask": s_leaf,
            "relation": rel,
            "slot_mask": slot_mask,
        }
        if previous_condition is not None:
            cell_features["previous_condition"] = previous_condition
        return {
            "coarse_forecast": coarse,
            "cell_features": cell_features,
        }


def build_unit_features(
    coarse: torch.Tensor,
    active_mask_t: torch.Tensor,
    active_mask_s: torch.Tensor,
    t_meta: torch.Tensor,
    s_sizes: torch.Tensor,
    s_depths: torch.Tensor,
    previous: torch.Tensor | None,
    remaining_budget: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct per-unit features for split heads.

    Returns:
        temporal_feats [B,T,12], spatial_feats [B,S,12]
    """
    b, t, s, _ = coarse.shape
    device = coarse.device
    mean_ts = coarse.mean(dim=-1)  # [B,T,S]
    # Temporal feats: pool over active spatial
    s_w = active_mask_s.unsqueeze(1).clamp_min(0)
    t_mean = (mean_ts * s_w).sum(dim=2) / s_w.sum(dim=2).clamp_min(1.0)
    t_std = ((mean_ts - t_mean.unsqueeze(-1)).pow(2) * s_w).sum(dim=2)
    t_std = (t_std / s_w.sum(dim=2).clamp_min(1.0)).sqrt()
    t_var_s = mean_ts.std(dim=2, unbiased=False)
    if previous is None:
        t_disc = torch.zeros_like(t_mean)
    else:
        t_disc = (coarse - previous).abs().mean(dim=(2, 3))
    rem = remaining_budget.view(b, 1).expand(b, t)
    zeros = torch.zeros(b, t, device=device)
    temporal_feats = torch.stack(
        [
            t_meta[..., 0],
            t_meta[..., 1],
            t_meta[..., 3],
            t_meta[..., 4],
            t_mean,
            t_std,
            t_var_s,
            t_disc,
            rem,
            active_mask_t,
            zeros,
            zeros,
        ],
        dim=-1,
    )

    # Spatial feats: pool over active temporal
    t_w = active_mask_t.unsqueeze(-1).clamp_min(0)
    s_mean = (mean_ts * t_w).sum(dim=1) / t_w.sum(dim=1).clamp_min(1.0)
    s_std = ((mean_ts - s_mean.unsqueeze(1)).pow(2) * t_w).sum(dim=1)
    s_std = (s_std / t_w.sum(dim=1).clamp_min(1.0)).sqrt()
    s_var_t = mean_ts.std(dim=1, unbiased=False)
    if previous is None:
        s_disc = torch.zeros_like(s_mean)
    else:
        s_disc = (coarse - previous).abs().mean(dim=(1, 3))
    rem_s = remaining_budget.view(b, 1).expand(b, s)
    spatial_feats = torch.stack(
        [
            s_sizes,
            s_depths,
            s_mean,
            s_std,
            s_var_t,
            s_disc,
            rem_s,
            active_mask_s,
            torch.zeros(b, s, device=device),
            torch.zeros(b, s, device=device),
            torch.zeros(b, s, device=device),
            torch.zeros(b, s, device=device),
        ],
        dim=-1,
    )
    return temporal_feats, spatial_feats


def build_spatial_slot_meta(
    spatial_tree: SpatialResolutionTree,
    spatial_frontier_mask: torch.Tensor,
    s_max: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    b = spatial_frontier_mask.shape[0]
    device = spatial_frontier_mask.device
    sizes = torch.zeros(b, s_max, device=device)
    depths = torch.zeros(b, s_max, device=device)
    n = max(spatial_tree.n_nodes, 1)
    dmax = max(spatial_tree.depth, 1)
    for bi in range(b):
        ids = torch.nonzero(spatial_frontier_mask[bi] > 0.5, as_tuple=False).flatten()
        for slot, nid in enumerate(ids.tolist()):
            if slot >= s_max:
                break
            node = spatial_tree.nodes[nid]
            sizes[bi, slot] = len(node.original_node_indices) / n
            depths[bi, slot] = node.depth / dmax
    return sizes, depths
