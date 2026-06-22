"""Forecast-State Flow Chain (FSF): rectified-flow refinement over forecast states."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def temporal_resize(x: torch.Tensor, target_len: int) -> torch.Tensor:
    """Linear upsample along time: [B, T, N, C] -> [B, target_len, N, C]."""
    b, t, n, c = x.shape
    y = x.permute(0, 2, 3, 1).reshape(b * n, c, t)
    y = F.interpolate(y, size=target_len, mode="linear", align_corners=False)
    return y.reshape(b, n, c, target_len).permute(0, 3, 1, 2)


def temporal_pool(y: torch.Tensor, target_len: int) -> torch.Tensor:
    """Downsample along time: [B, F, N, C] -> [B, target_len, N, C]."""
    b, f_len, n, c = y.shape
    if f_len == target_len:
        return y
    if f_len % target_len == 0:
        g = f_len // target_len
        return y.reshape(b, target_len, g, n, c).mean(dim=2)
    z = y.permute(0, 2, 3, 1).reshape(b * n, c, f_len)
    z = F.adaptive_avg_pool1d(z, target_len)
    return z.reshape(b, n, c, target_len).permute(0, 3, 1, 2)


def adaptive_adj(node_emb1: torch.Tensor, node_emb2: torch.Tensor, topk: int | None = None) -> torch.Tensor:
    score = torch.relu(node_emb1 @ node_emb2.T)
    if topk is not None and topk > 0 and topk < score.size(-1):
        val, idx = torch.topk(score, topk, dim=-1)
        mask = torch.full_like(score, float("-inf"))
        mask.scatter_(dim=-1, index=idx, src=val)
        score = mask
    return torch.softmax(score, dim=-1)


def spatial_mix(h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
    """h: [B, T, N, D], adj: [N, N] -> [B, T, N, D]."""
    return torch.einsum("nm,btmd->btnd", adj, h)


def temporal_mix(h: torch.Tensor, conv_dw: nn.Conv2d, conv_pw: nn.Conv2d) -> torch.Tensor:
    """h: [B, T, N, D] -> [B, T, N, D]."""
    x = h.permute(0, 3, 1, 2)
    x = conv_pw(conv_dw(x))
    return x.permute(0, 2, 3, 1)


class LightSTContextEncoder(nn.Module):
    """Lightweight joint spatio-temporal context encoder."""

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
        self.hidden_dim = hidden_dim
        self.topk = topk
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.node_emb1 = nn.Parameter(torch.randn(node_size, hidden_dim) * 0.02)
        self.node_emb2 = nn.Parameter(torch.randn(node_size, hidden_dim) * 0.02)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "temp_dw": nn.Conv2d(
                            hidden_dim,
                            hidden_dim,
                            kernel_size=(3, 1),
                            padding=(1, 0),
                            groups=hidden_dim,
                        ),
                        "temp_pw": nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
                        "joint_dw": nn.Conv2d(
                            hidden_dim,
                            hidden_dim,
                            kernel_size=(3, 1),
                            padding=(1, 0),
                            groups=hidden_dim,
                        ),
                        "joint_pw": nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
                        "fuse": nn.Sequential(
                            nn.Linear(hidden_dim * 3, hidden_dim),
                            nn.GELU(),
                            nn.Dropout(dropout),
                            nn.Linear(hidden_dim, hidden_dim),
                        ),
                        "norm": nn.LayerNorm(hidden_dim),
                    }
                )
            )

    def _adj(self) -> torch.Tensor:
        return adaptive_adj(self.node_emb1, self.node_emb2, self.topk)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, H, N, C] -> context [B, N, D]."""
        h = self.input_proj(x)
        adj = self._adj()
        for layer in self.layers:
            temp = temporal_mix(h, layer["temp_dw"], layer["temp_pw"])
            spa = spatial_mix(h, adj)
            xj = temporal_mix(h, layer["joint_dw"], layer["joint_pw"])
            joint = spatial_mix(xj, adj)
            fused = layer["fuse"](torch.cat([temp, spa, joint], dim=-1))
            h = layer["norm"](h + fused)
        context = (h.mean(dim=1) + h.amax(dim=1)) * 0.5
        return context


class ForecastStateVectorField(nn.Module):
    """Vector field for rectified-flow transition between forecast states."""

    def __init__(
        self,
        hidden_dim: int,
        node_size: int,
        stage_dim: int = 16,
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
        self.stage_emb = nn.Embedding(8, hidden_dim)

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

        self.joint_dw = nn.Conv2d(
            hidden_dim,
            hidden_dim,
            kernel_size=(3, 1),
            padding=(1, 0),
            groups=hidden_dim,
        )
        self.joint_pw = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)

        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def _adj(self) -> torch.Tensor:
        score = torch.relu(self.node_emb1 @ self.node_emb2.T)
        if self.topk is not None and self.topk > 0 and self.topk < score.size(-1):
            val, idx = torch.topk(score, self.topk, dim=-1)
            mask = torch.full_like(score, float("-inf"))
            mask.scatter_(dim=-1, index=idx, src=val)
            score = mask
        return torch.softmax(score, dim=-1)

    def forward(
        self,
        state: torch.Tensor,
        tau: torch.Tensor | float,
        context: torch.Tensor,
        stage_id: int,
    ) -> torch.Tensor:
        b, f_len, n, _ = state.shape

        h_state = self.state_proj(state)
        h_ctx = self.context_proj(context).unsqueeze(1).expand(-1, f_len, -1, -1)

        if not torch.is_tensor(tau):
            tau = torch.full((b, 1), float(tau), device=state.device, dtype=state.dtype)
        elif tau.dim() == 0:
            tau = tau.reshape(1, 1).expand(b, 1)
        elif tau.dim() == 1:
            tau = tau[:, None]

        h_tau = self.tau_proj(tau).view(b, 1, 1, self.hidden_dim)
        h_stage = self.stage_emb(
            torch.full((b,), stage_id, device=state.device, dtype=torch.long)
        ).view(b, 1, 1, self.hidden_dim)

        h = h_state + h_ctx + h_tau + h_stage

        x = h.permute(0, 3, 1, 2)
        temp = self.temp_pw(self.temp_dw(x)).permute(0, 2, 3, 1)

        adj = self._adj()
        spa = torch.einsum("nm,bfmd->bfnd", adj, h)

        xj = self.joint_pw(self.joint_dw(x)).permute(0, 2, 3, 1)
        joint = torch.einsum("nm,bfmd->bfnd", adj, xj)

        fused = self.fuse(torch.cat([h, temp, spa, joint], dim=-1))
        fused = self.norm(h + fused)

        return self.alpha * self.out(fused)


class ForecastStateFlow(nn.Module):
    """Forecast-State Flow Chain: coarse native forecast + rectified-flow refinement."""

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

        chain_lengths = list(model_args.get("chain_lengths", [3, 6, 12]))
        if chain_lengths[-1] != output_len:
            raise ValueError(
                f"chain_lengths[-1]={chain_lengths[-1]} must equal output_len={output_len}"
            )

        self.node_size = node_size
        self.input_len = input_len
        self.output_len = output_len
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.chain_lengths = chain_lengths
        self.num_flow_steps = num_flow_steps
        self.fm_sigma = fm_sigma

        self.encoder = LightSTContextEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            node_size=node_size,
            num_layers=num_encoder_layers,
            topk=topk,
            dropout=dropout,
        )

        r1 = chain_lengths[0]
        self.coarse_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, r1),
        )

        num_transitions = max(len(chain_lengths) - 1, 0)
        self.vector_fields = nn.ModuleList(
            [
                ForecastStateVectorField(
                    hidden_dim=hidden_dim,
                    node_size=node_size,
                    topk=topk,
                    dropout=dropout,
                )
                for _ in range(num_transitions)
            ]
        )

    @staticmethod
    def pool_target(future_target: torch.Tensor, target_len: int) -> torch.Tensor:
        return temporal_pool(future_target[..., :1], target_len)

    @staticmethod
    def state_target(future_target: torch.Tensor, target_len: int, full_len: int) -> torch.Tensor:
        pooled = temporal_pool(future_target[..., :1], target_len)
        return temporal_resize(pooled, full_len)

    def predict_native(self, context: torch.Tensor, r: int) -> torch.Tensor:
        """context [B, N, D] -> native forecast [B, r, N, 1]."""
        z = self.coarse_head(context)
        z = z.transpose(1, 2).unsqueeze(-1)
        return z

    def euler_refine(
        self,
        s: torch.Tensor,
        context: torch.Tensor,
        stage_id: int,
        num_steps: int | None = None,
    ) -> torch.Tensor:
        steps = num_steps if num_steps is not None else self.num_flow_steps
        dt = 1.0 / steps
        for step in range(steps):
            tau = step * dt
            v = self.vector_fields[stage_id](s, tau, context, stage_id=stage_id)
            s = s + dt * v
        return s

    def forward(
        self,
        history_data: torch.Tensor,
        future_data: torch.Tensor | None = None,
        return_all: bool = False,
        train: bool = False,
        **kwargs,
    ):
        context = self.encoder(history_data)

        r_list = self.chain_lengths
        f_len = self.output_len

        z0_native = self.predict_native(context, r_list[0])
        s = temporal_resize(z0_native, f_len)

        states = [s]
        native_states = [z0_native]
        fm_items: list[dict[str, torch.Tensor]] = []

        for k in range(len(r_list) - 1):
            s = self.euler_refine(
                s=s,
                context=context,
                stage_id=k,
                num_steps=self.num_flow_steps,
            )
            states.append(s)
            native_states.append(temporal_pool(s, r_list[k + 1]))

            if future_data is not None and train:
                target = future_data[..., :1]
                sa = temporal_resize(temporal_pool(target, r_list[k]), f_len)
                sb = temporal_resize(temporal_pool(target, r_list[k + 1]), f_len)

                tau = torch.rand(target.size(0), device=target.device)
                noise = torch.randn_like(sa) * self.fm_sigma
                st = (
                    (1 - tau.view(-1, 1, 1, 1)) * sa
                    + tau.view(-1, 1, 1, 1) * sb
                    + noise
                )
                vel_target = sb - sa
                vel_pred = self.vector_fields[k](st, tau, context, stage_id=k)
                fm_items.append({"vel_pred": vel_pred, "vel_target": vel_target})

        pred = states[-1]

        if return_all:
            return {
                "pred": pred,
                "states": states,
                "native_states": native_states,
                "fm_items": fm_items,
            }

        return pred
