"""Spatio-Temporal Forecast-State Flow (ST-FSF)."""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .spectral_ops import (
    compute_graph_fourier_basis,
    lift_native_to_full,
    load_adj_matrix,
    native_graph_coeff,
    st_project,
    temporal_pool,
)


class LightSTBlock(nn.Module):
    def __init__(self, hidden_dim: int, node_size: int, topk: int = 20, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.topk = topk

        self.temp_dw = nn.Conv2d(
            hidden_dim,
            hidden_dim,
            kernel_size=(3, 1),
            padding=(1, 0),
            groups=hidden_dim,
        )
        self.temp_pw = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)

        self.node_emb1 = nn.Parameter(torch.randn(node_size, hidden_dim) * 0.02)
        self.node_emb2 = nn.Parameter(torch.randn(node_size, hidden_dim) * 0.02)

        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def _adj(self, device, dtype):
        score = torch.relu(self.node_emb1 @ self.node_emb2.T)
        if self.topk is not None and 0 < self.topk < score.size(-1):
            val, idx = torch.topk(score, self.topk, dim=-1)
            mask = torch.full_like(score, float("-inf"))
            mask.scatter_(dim=-1, index=idx, src=val)
            score = mask
        return torch.softmax(score, dim=-1).to(device=device, dtype=dtype)

    def _temporal_mix(self, h: torch.Tensor) -> torch.Tensor:
        x = h.permute(0, 3, 1, 2)
        x = self.temp_pw(self.temp_dw(x))
        return x.permute(0, 2, 3, 1)

    def _spatial_mix(self, h: torch.Tensor) -> torch.Tensor:
        adj = self._adj(h.device, h.dtype)
        return torch.einsum("nm,btmd->btnd", adj, h)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        temp = self._temporal_mix(h)
        spa = self._spatial_mix(h)
        joint = self._spatial_mix(temp)
        out = self.fuse(torch.cat([temp, spa, joint], dim=-1))
        return self.norm(h + out)


class LightSTContextEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        node_size: int,
        num_layers: int = 2,
        topk: int = 20,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [
                LightSTBlock(hidden_dim, node_size, topk=topk, dropout=dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, history_data: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(history_data)
        for block in self.blocks:
            h = block(h)
        return h.mean(dim=1)


class CoarseSpectralForecastHead(nn.Module):
    def __init__(self, hidden_dim: int, r1: int, q1: int):
        super().__init__()
        self.r1 = r1
        self.q1 = q1
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, r1),
        )

    def forward(self, context: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        vq = v[:, : self.q1].to(device=context.device, dtype=context.dtype)
        context_q = torch.einsum("nq,bnd->bqd", vq, context)
        z = self.head(context_q)
        return z.permute(0, 2, 1).unsqueeze(-1)


class STForecastStateVectorField(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        node_size: int,
        stage_count: int = 4,
        topk: int = 20,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.topk = topk

        self.state_proj = nn.Linear(1, hidden_dim)
        self.context_proj = nn.Linear(hidden_dim, hidden_dim)
        self.tau_proj = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.stage_emb = nn.Embedding(stage_count, hidden_dim)

        self.temp_dw = nn.Conv2d(
            hidden_dim,
            hidden_dim,
            kernel_size=(3, 1),
            padding=(1, 0),
            groups=hidden_dim,
        )
        self.temp_pw = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)

        self.node_emb1 = nn.Parameter(torch.randn(node_size, hidden_dim) * 0.02)
        self.node_emb2 = nn.Parameter(torch.randn(node_size, hidden_dim) * 0.02)

        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def _adj(self, device, dtype):
        score = torch.relu(self.node_emb1 @ self.node_emb2.T)
        if self.topk is not None and 0 < self.topk < score.size(-1):
            val, idx = torch.topk(score, self.topk, dim=-1)
            mask = torch.full_like(score, float("-inf"))
            mask.scatter_(dim=-1, index=idx, src=val)
            score = mask
        return torch.softmax(score, dim=-1).to(device=device, dtype=dtype)

    def _temporal_mix(self, h: torch.Tensor) -> torch.Tensor:
        x = h.permute(0, 3, 1, 2)
        x = self.temp_pw(self.temp_dw(x))
        return x.permute(0, 2, 3, 1)

    def _spatial_mix(self, h: torch.Tensor) -> torch.Tensor:
        adj = self._adj(h.device, h.dtype)
        return torch.einsum("nm,bfmd->bfnd", adj, h)

    def forward(
        self,
        state: torch.Tensor,
        tau: torch.Tensor | float,
        context: torch.Tensor,
        stage_id: int,
    ) -> torch.Tensor:
        b, f_len, _, _ = state.shape

        h_state = self.state_proj(state)
        h_ctx = self.context_proj(context).unsqueeze(1).expand(-1, f_len, -1, -1)

        if not torch.is_tensor(tau):
            tau = torch.full((b, 1), float(tau), device=state.device, dtype=state.dtype)
        elif tau.dim() == 0:
            tau = tau.reshape(1, 1).expand(b, 1).to(device=state.device, dtype=state.dtype)
        elif tau.dim() == 1:
            tau = tau[:, None].to(device=state.device, dtype=state.dtype)

        h_tau = self.tau_proj(tau).view(b, 1, 1, self.hidden_dim)
        sid = torch.full((b,), stage_id, device=state.device, dtype=torch.long)
        h_stage = self.stage_emb(sid).view(b, 1, 1, self.hidden_dim)

        h = h_state + h_ctx + h_tau + h_stage
        temp = self._temporal_mix(h)
        spa = self._spatial_mix(h)
        joint = self._spatial_mix(temp)

        fused = self.fuse(torch.cat([h, temp, spa, joint], dim=-1))
        fused = self.norm(h + fused)
        return self.alpha * self.out(fused)


class STForecastStateFlow(nn.Module):
    """Spatio-temporal forecast-state flow chain in target coordinates."""

    @staticmethod
    def build_stage_specs(
        output_len: int,
        node_size: int,
        q_ratio_1: float = 0.25,
        q_ratio_2: float = 0.50,
        q_list_override: list[int] | None = None,
        direct: bool = False,
    ) -> list[tuple[int, int]]:
        if direct:
            return [(output_len, node_size)]
        if output_len == 12:
            r_list = [3, 6, 12]
        elif output_len == 24:
            r_list = [6, 12, 24]
        elif output_len == 48:
            r_list = [12, 24, 48]
        else:
            r_list = [max(output_len // 4, 1), max(output_len // 2, 1), output_len]

        if q_list_override is not None:
            q_list = list(q_list_override)
        else:
            q1 = math.ceil(node_size * q_ratio_1)
            q2 = math.ceil(node_size * q_ratio_2)
            q_list = [q1, q2, node_size]

        if len(r_list) != len(q_list):
            raise ValueError(f"r_list length {len(r_list)} != q_list length {len(q_list)}")
        return list(zip(r_list, q_list))

    def __init__(self, **model_args):
        super().__init__()
        node_size = model_args["node_size"]
        input_len = model_args["input_len"]
        output_len = model_args["output_len"]
        input_dim = model_args.get("input_dim", 4)
        hidden_dim = model_args.get("hidden_dim", 64)
        num_encoder_layers = model_args.get("num_encoder_layers", 2)
        topk = model_args.get("topk", 20)
        dropout = model_args.get("dropout", 0.1)
        num_flow_steps = model_args.get("num_flow_steps", 1)
        fm_sigma = model_args.get("fm_sigma", 0.01)
        q_ratio_1 = model_args.get("q_ratio_1", 0.25)
        q_ratio_2 = model_args.get("q_ratio_2", 0.50)
        use_projection_after_flow = model_args.get("use_projection_after_flow", True)
        adj_mx_path = model_args.get("adj_mx_path")
        q_list_override = model_args.get("q_list_override")
        direct = model_args.get("direct_forecast", False)

        self.node_size = node_size
        self.input_len = input_len
        self.output_len = output_len
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_flow_steps = num_flow_steps
        self.fm_sigma = fm_sigma
        self.use_projection_after_flow = use_projection_after_flow

        self.stage_specs = self.build_stage_specs(
            output_len=output_len,
            node_size=node_size,
            q_ratio_1=q_ratio_1,
            q_ratio_2=q_ratio_2,
            q_list_override=q_list_override,
            direct=direct,
        )

        adj = load_adj_matrix(adj_mx_path, node_size)
        graph_basis = compute_graph_fourier_basis(adj)
        self.register_buffer("graph_basis", graph_basis)

        self.encoder = LightSTContextEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            node_size=node_size,
            num_layers=num_encoder_layers,
            topk=topk,
            dropout=dropout,
        )

        r1, q1 = self.stage_specs[0]
        self.coarse_head = CoarseSpectralForecastHead(hidden_dim, r1, q1)

        num_transitions = max(len(self.stage_specs) - 1, 0)
        self.vector_fields = nn.ModuleList(
            [
                STForecastStateVectorField(
                    hidden_dim=hidden_dim,
                    node_size=node_size,
                    stage_count=max(num_transitions, 1),
                    topk=topk,
                    dropout=dropout,
                )
                for _ in range(num_transitions)
            ]
        )

    @staticmethod
    def st_target(
        future_target: torch.Tensor,
        v: torch.Tensor,
        r: int,
        q: int,
        full_len: int,
    ) -> torch.Tensor:
        return st_project(future_target[..., :1], v, r, q, full_len)

    @staticmethod
    def native_target(
        future_target: torch.Tensor,
        v: torch.Tensor,
        r: int,
        q: int,
    ) -> torch.Tensor:
        pooled = temporal_pool(future_target[..., :1], r)
        return native_graph_coeff(pooled, v, q)

    def _should_project(self, r_next: int, q_next: int) -> bool:
        if not self.use_projection_after_flow:
            return False
        return not (r_next == self.output_len and q_next >= self.node_size)

    def forward(
        self,
        history_data: torch.Tensor,
        future_data: torch.Tensor | None = None,
        return_all: bool = False,
        train: bool = False,
        **kwargs,
    ):
        context = self.encoder(history_data)
        f_len = self.output_len
        v = self.graph_basis

        target = future_data[..., :1] if future_data is not None else None

        r1, q1 = self.stage_specs[0]
        native0 = self.coarse_head(context, v)
        s = lift_native_to_full(native0, v, f_len)
        if self._should_project(r1, q1):
            s = st_project(s, v, r1, q1, f_len)

        states = [s]
        native_states = [native0]
        fm_items: list[dict[str, torch.Tensor]] = []
        fm_losses: list[torch.Tensor] = []

        for k in range(len(self.stage_specs) - 1):
            r_cur, q_cur = self.stage_specs[k]
            r_next, q_next = self.stage_specs[k + 1]

            s_next = s
            dt = 1.0 / float(self.num_flow_steps)
            for step in range(self.num_flow_steps):
                tau = step * dt
                vel = self.vector_fields[k](s_next, tau, context, stage_id=k)
                s_next = s_next + dt * vel

            if self._should_project(r_next, q_next):
                s_next = st_project(s_next, v, r_next, q_next, f_len)

            s = s_next
            states.append(s)
            native_states.append(native_graph_coeff(temporal_pool(s, r_next), v, q_next))

            if target is not None and train:
                sa = st_project(target, v, r_cur, q_cur, f_len)
                sb = st_project(target, v, r_next, q_next, f_len)
                tau = torch.rand(target.size(0), device=target.device, dtype=target.dtype)
                tau_view = tau.view(-1, 1, 1, 1)
                noise = torch.randn_like(sa) * self.fm_sigma
                st = (1.0 - tau_view) * sa + tau_view * sb + noise
                vel_target = sb - sa
                vel_pred = self.vector_fields[k](st, tau, context, stage_id=k)
                fm_items.append({"vel_pred": vel_pred, "vel_target": vel_target})
                fm_losses.append(torch.mean(torch.abs(vel_pred - vel_target)))

        pred = states[-1]

        if return_all:
            return {
                "pred": pred,
                "states": states,
                "native_states": native_states,
                "fm_items": fm_items,
                "fm_losses": fm_losses,
                "stage_specs": self.stage_specs,
            }
        return pred
