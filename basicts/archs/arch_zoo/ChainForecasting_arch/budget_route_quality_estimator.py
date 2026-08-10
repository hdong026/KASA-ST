"""Sample-aware Route Quality Estimator (no eta in the quality path)."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from basicts.archs.arch_zoo.ChainForecasting_arch.route_quality_decision import (
    build_route_descriptor_tensors,
)


class TemporalNodeEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        dim_feedforward: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_len: int = 64,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.input_proj = nn.Linear(int(input_dim), self.d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, int(max_len), self.d_model))
        nn.init.normal_(self.pos_emb, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(nhead),
            dim_feedforward=int(dim_feedforward),
            dropout=float(dropout),
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(num_layers))
        self.fuse = nn.Sequential(
            nn.Linear(self.d_model * 3, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, P, N, C]
        b, p, n, c = x.shape
        h = self.input_proj(x)  # B,P,N,d
        h = h.permute(0, 2, 1, 3).reshape(b * n, p, self.d_model)
        h = h + self.pos_emb[:, :p, :]
        h = self.encoder(h)  # BN,P,d
        last = h[:, -1, :]
        mean = h.mean(dim=1)
        std = h.std(dim=1, unbiased=False)
        node = self.fuse(torch.cat([last, mean, std], dim=-1))
        return node.view(b, n, self.d_model)


class SpatialQueryPool(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        num_queries: int = 4,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.num_queries = int(num_queries)
        self.queries = nn.Parameter(torch.randn(self.num_queries, self.d_model) * 0.02)
        self.layers = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=self.d_model,
                    num_heads=int(nhead),
                    dropout=float(dropout),
                    batch_first=True,
                )
                for _ in range(int(num_layers))
            ]
        )
        self.norms = nn.ModuleList(
            [nn.LayerNorm(self.d_model) for _ in range(int(num_layers))]
        )
        self.out = nn.Sequential(
            nn.Linear(self.num_queries * self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
        )

    def forward(self, node_emb: torch.Tensor) -> torch.Tensor:
        # node_emb: [B,N,d]
        b = node_emb.shape[0]
        q = self.queries.unsqueeze(0).expand(b, -1, -1)
        for attn, norm in zip(self.layers, self.norms):
            upd, _ = attn(q, node_emb, node_emb, need_weights=False)
            q = norm(q + upd)
        return self.out(q.reshape(b, -1))


class StatisticalDifficultyBranch(nn.Module):
    """Explicit history statistics; channel-agnostic (not hard-coded C=1)."""

    def __init__(self, out_dim: int = 128):
        super().__init__()
        # per-channel stats (~12) + node-dispersion / high-change (~4) = 16 slots before expand
        self.stat_dim = 16
        self.mlp = nn.Sequential(
            nn.Linear(self.stat_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,P,N,C] — use all channels, reduce over N and C for sample-level stats
        b, p, n, c = x.shape
        # Primary flow-like signal: channel 0 if present, else mean over C
        flow = x[..., 0] if c >= 1 else x.mean(dim=-1)  # B,P,N
        # Channel-aggregated tensor for multi-channel awareness
        x_mean_c = x.mean(dim=-1)  # B,P,N

        t_mean = flow.mean(dim=1)  # B,N
        t_std = flow.std(dim=1, unbiased=False)
        last = flow[:, -1, :]
        first = flow[:, 0, :]
        # linear slope via normalized index regression proxy
        t = torch.linspace(0, 1, p, device=x.device, dtype=x.dtype).view(1, p, 1)
        t_c = t - t.mean()
        slope = ((flow - t_mean.unsqueeze(1)) * t_c).sum(dim=1) / (
            (t_c**2).sum() + 1e-6
        )
        diff = flow[:, 1:, :] - flow[:, :-1, :]
        mad = diff.abs().mean(dim=1)
        max_ad = diff.abs().amax(dim=1)
        half = max(p // 2, 1)
        recent = flow[:, -half:, :].mean(dim=1)
        early = flow[:, :half, :].mean(dim=1)
        recent_early = recent - early
        node_disp = t_mean.std(dim=-1, unbiased=False)  # B
        vol = t_std.mean(dim=-1)
        high_change = (mad > mad.mean(dim=-1, keepdim=True)).float().mean(dim=-1)
        multi_ch_vol = x_mean_c.std(dim=1, unbiased=False).mean(dim=-1)

        def _pool(v: torch.Tensor) -> torch.Tensor:
            return v.mean(dim=-1)

        feats = torch.stack(
            [
                _pool(t_mean),
                _pool(t_std),
                _pool(last),
                _pool(first),
                _pool(slope),
                _pool(mad),
                _pool(max_ad),
                _pool(recent),
                _pool(early),
                _pool(recent_early),
                node_disp,
                vol,
                high_change,
                multi_ch_vol,
                flow.amax(dim=(1, 2)),
                flow.amin(dim=(1, 2)),
            ],
            dim=-1,
        )
        return self.mlp(feats)


class SampleFusion(nn.Module):
    def __init__(self, d_model: int, stat_dim: int, sample_dim: int = 256):
        super().__init__()
        self.norm = nn.LayerNorm(d_model + stat_dim)
        self.mlp = nn.Sequential(
            nn.Linear(d_model + stat_dim, sample_dim),
            nn.GELU(),
            nn.Linear(sample_dim, sample_dim),
        )
        self.res = nn.Sequential(
            nn.LayerNorm(sample_dim),
            nn.Linear(sample_dim, sample_dim),
            nn.GELU(),
            nn.Linear(sample_dim, sample_dim),
        )

    def forward(self, spatial: torch.Tensor, stats: torch.Tensor) -> torch.Tensor:
        z = self.mlp(self.norm(torch.cat([spatial, stats], dim=-1)))
        return z + self.res(z)


class RouteSequenceEncoder(nn.Module):
    """Generic route encoder for arbitrary candidate pools."""

    def __init__(self, route_embedding_dim: int = 64, hidden: int = 64):
        super().__init__()
        self.route_embedding_dim = int(route_embedding_dim)
        self.token_proj = nn.Linear(3, hidden)  # res_norm, jump, stage_frac
        self.gru = nn.GRU(
            input_size=hidden,
            hidden_size=hidden,
            num_layers=1,
            batch_first=True,
        )
        self.out = nn.Sequential(
            nn.Linear(hidden + 4, route_embedding_dim),
            nn.GELU(),
            nn.Linear(route_embedding_dim, route_embedding_dim),
        )

    def forward(self, descriptors: dict[str, torch.Tensor]) -> torch.Tensor:
        res = descriptors["res_norm"]  # R,S
        jumps = descriptors["jumps"]
        mask = descriptors["stage_mask"]
        r, s = res.shape
        stage_idx = torch.arange(s, device=res.device, dtype=res.dtype).view(1, s) / max(
            s - 1, 1
        )
        stage_idx = stage_idx.expand(r, s)
        tokens = torch.stack([res, jumps, stage_idx], dim=-1)  # R,S,3
        h = self.token_proj(tokens)
        lengths = mask.sum(dim=-1).clamp(min=1)
        # pack
        packed = nn.utils.rnn.pack_padded_sequence(
            h,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hn = self.gru(packed)
        seq = hn[-1]  # R,hidden
        static = torch.stack(
            [
                descriptors["stage_count"],
                descriptors["first_res"],
                descriptors["mean_res"],
                descriptors["cost"],
            ],
            dim=-1,
        )
        return self.out(torch.cat([seq, static], dim=-1))  # R, Er


class RouteConditionedQualityHead(nn.Module):
    """Shared route-conditioned scorer: L_hat = d_hat(X) + delta_hat(X,r)."""

    def __init__(
        self,
        sample_dim: int = 256,
        route_dim: int = 64,
        hidden: int = 256,
    ):
        super().__init__()
        self.route_proj = nn.Linear(route_dim, sample_dim)
        self.difficulty = nn.Sequential(
            nn.Linear(sample_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        pair_in = sample_dim + sample_dim + sample_dim + 2
        self.delta = nn.Sequential(
            nn.Linear(pair_in, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        z_x: torch.Tensor,
        e_r: torch.Tensor,
        costs: torch.Tensor,
        stage_count: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # z_x: [B,D], e_r: [R,Er]
        b = z_x.shape[0]
        r = e_r.shape[0]
        d_hat = self.difficulty(z_x).squeeze(-1)  # B
        er = self.route_proj(e_r)  # R,D
        zx = z_x.unsqueeze(1).expand(b, r, -1)
        er_b = er.unsqueeze(0).expand(b, r, -1)
        interact = zx * er_b
        cost = costs.reshape(1, r, 1).expand(b, r, 1)
        stages = stage_count.reshape(1, r, 1).expand(b, r, 1)
        pair = torch.cat([zx, er_b, interact, cost, stages], dim=-1)
        delta = self.delta(pair).squeeze(-1)  # B,R
        l_hat = d_hat.unsqueeze(-1) + delta
        return l_hat, d_hat, delta


class RouteQualityEstimator(nn.Module):
    """Estimate per-route forecasting MAE from history X only (no eta)."""

    def __init__(
        self,
        input_dim: int = 4,
        d_model: int = 128,
        temporal_layers: int = 2,
        nhead: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        spatial_query_count: int = 4,
        spatial_layers: int = 2,
        sample_embedding_dim: int = 256,
        route_embedding_dim: int = 64,
        stat_dim: int = 128,
        max_len: int = 64,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.sample_embedding_dim = int(sample_embedding_dim)
        self.route_embedding_dim = int(route_embedding_dim)
        self.temporal = TemporalNodeEncoder(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            num_layers=temporal_layers,
            dropout=dropout,
            max_len=max_len,
        )
        self.spatial = SpatialQueryPool(
            d_model=d_model,
            num_queries=spatial_query_count,
            nhead=nhead,
            num_layers=spatial_layers,
            dropout=dropout,
        )
        self.stats = StatisticalDifficultyBranch(out_dim=stat_dim)
        self.fusion = SampleFusion(d_model, stat_dim, sample_embedding_dim)
        self.route_encoder = RouteSequenceEncoder(route_embedding_dim=route_embedding_dim)
        self.head = RouteConditionedQualityHead(
            sample_dim=sample_embedding_dim,
            route_dim=route_embedding_dim,
            hidden=256,
        )

    def encode_sample(self, history: torch.Tensor) -> torch.Tensor:
        node = self.temporal(history)
        spatial = self.spatial(node)
        stats = self.stats(history)
        return self.fusion(spatial, stats)

    def forward(
        self,
        history: torch.Tensor,
        routes: list[list[int]],
        route_costs: torch.Tensor | list[float],
        horizon: int,
    ) -> dict[str, Any]:
        """Return predicted raw-scale route MAEs shaped [B, R].

        Intentionally has no ``eta`` / budget argument.
        """
        if history.ndim != 4:
            raise ValueError(f"history must be [B,P,N,C], got {tuple(history.shape)}")
        descriptors = build_route_descriptor_tensors(
            routes,
            horizon=horizon,
            costs=route_costs,
            device=history.device,
            dtype=history.dtype,
        )
        z_x = self.encode_sample(history)
        e_r = self.route_encoder(descriptors)
        l_hat, d_hat, delta = self.head(
            z_x, e_r, descriptors["cost"], descriptors["stage_count"]
        )
        return {
            "predicted_route_losses": l_hat,
            "sample_difficulty": d_hat,
            "route_residuals": delta,
            "sample_embedding": z_x,
            "route_embeddings": e_r,
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
