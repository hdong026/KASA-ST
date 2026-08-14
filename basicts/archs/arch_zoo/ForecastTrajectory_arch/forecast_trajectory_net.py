"""ForecastTrajectoryNet: shared history encoder + one universal transition core."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Optional, Sequence

import torch
from torch import nn

from basicts.archs.arch_zoo.ChainForecasting_arch.gcn import ABCDSpatialModule
from basicts.archs.arch_zoo.ChainForecasting_arch.mlp import MultiLayerPerceptron
from basicts.archs.arch_zoo.ForecastTrajectory_arch.trajectory_graph import (
    ForecastTrajectoryGraph,
)
from basicts.archs.arch_zoo.ForecastTrajectory_arch.universal_transition_core import (
    UniversalTransitionCore,
)


@dataclass
class HistoryState:
    tokens: torch.Tensor  # [B, P, N, D]
    pooled: torch.Tensor  # [B, D]  (node+time pooled, for policy)
    pooled_nodes: torch.Tensor  # [B, N, D]
    history_flow: torch.Tensor  # [B, L, N]
    history_data: torch.Tensor  # [B, L, N, C]


class SharedHistoryEncoder(nn.Module):
    """KASA-style patch + downsample encoder WITHOUT horizon-specific heads.

    Produces history tokens reused by every subsequent transition.
    """

    def __init__(
        self,
        input_len: int,
        input_dim: int,
        patch_len: int,
        stride: int,
        node_size: int,
        td_size: int,
        dw_size: int,
        d_d: int,
        d_td: int,
        d_dw: int,
        d_spa: int,
        d_model: int,
        num_layer: int,
        if_time_in_day: bool = True,
        if_day_in_week: bool = True,
        if_spatial: bool = True,
    ):
        super().__init__()
        self.input_len = int(input_len)
        self.input_dim = int(input_dim)
        self.patch_len = int(patch_len)
        self.stride = int(stride)
        self.node_size = int(node_size)
        self.td_size = int(td_size)
        self.dw_size = int(dw_size)
        self.d_d = int(d_d)
        self.d_model = int(d_model)
        self.if_time_in_day = bool(if_time_in_day)
        self.if_day_in_week = bool(if_day_in_week)
        self.if_spatial = bool(if_spatial)

        self.td_codebook = None
        self.dw_codebook = None
        self.spa_codebook = None
        if self.if_time_in_day:
            self.td_codebook = nn.Parameter(torch.empty(self.td_size, d_td))
            nn.init.xavier_uniform_(self.td_codebook)
        if self.if_day_in_week:
            self.dw_codebook = nn.Parameter(torch.empty(self.dw_size, d_dw))
            nn.init.xavier_uniform_(self.dw_codebook)
        if self.if_spatial:
            self.spa_codebook = nn.Parameter(torch.empty(self.node_size, d_spa))
            nn.init.xavier_uniform_(self.spa_codebook)

        self.patch_embed = nn.Conv2d(
            in_channels=self.input_dim * self.patch_len,
            out_channels=d_d,
            kernel_size=(1, 1),
            bias=True,
        )
        self.down_embed = nn.Conv2d(
            in_channels=self.input_dim * self.patch_len,
            out_channels=d_d,
            kernel_size=(1, 1),
            bias=True,
        )
        token_dim = d_d + d_spa * int(self.if_spatial)
        self.patch_data_mlp = nn.Sequential(
            *[MultiLayerPerceptron(d_d, d_d) for _ in range(num_layer)]
        )
        self.down_data_mlp = nn.Sequential(
            *[MultiLayerPerceptron(d_d, d_d) for _ in range(num_layer)]
        )
        self.patch_spatial_mlp = nn.Sequential(
            *[MultiLayerPerceptron(token_dim, token_dim) for _ in range(num_layer)]
        )
        self.down_spatial_mlp = nn.Sequential(
            *[MultiLayerPerceptron(token_dim, token_dim) for _ in range(num_layer)]
        )
        self.token_proj = nn.Linear(token_dim, d_model)

    def _embed_branch(self, patch_input: torch.Tensor, embed: nn.Module) -> torch.Tensor:
        # patch_input: [B, M, P, N, C]  -> concat C over P -> [B, M, P*C, N]
        data_channels = [patch_input[..., i] for i in range(self.input_dim)]
        data_emb_input = torch.cat(data_channels, dim=2)
        data_emb = embed(data_emb_input.permute(0, 2, 1, 3)).permute(0, 2, 3, 1)
        return data_emb  # [B, M, N, d_d]

    def _encode_branch(
        self,
        patch_input: torch.Tensor,
        embed: nn.Module,
        data_mlp: nn.Module,
        spatial_mlp: nn.Module,
    ) -> torch.Tensor:
        batch_size, num, _, num_nodes, _ = patch_input.shape
        data_emb = self._embed_branch(patch_input, embed)
        data_emb = data_mlp(data_emb.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        if self.if_spatial:
            spatial_emb = (
                self.spa_codebook.unsqueeze(0)
                .expand(batch_size, -1, -1)
                .unsqueeze(1)
                .expand(-1, num, -1, -1)
            )
            hidden = torch.cat((data_emb, spatial_emb), dim=-1)
        else:
            hidden = data_emb
        hidden = spatial_mlp(hidden.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        return hidden  # [B, M, N, token_dim]

    def forward(self, history_data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = history_data[..., : self.input_dim]
        in_len_add = ceil(1.0 * self.input_len / self.stride) * self.stride - self.input_len
        if in_len_add:
            x_aug = torch.cat((x[:, -1:, :, :].expand(-1, in_len_add, -1, -1), x), dim=1)
        else:
            x_aug = x
        downsamp_input = torch.stack(
            [x_aug[:, i :: self.stride, :, :] for i in range(self.stride)], dim=1
        )
        patch_input = x_aug.unfold(
            dimension=1, size=self.patch_len, step=self.patch_len
        ).permute(0, 1, 4, 2, 3)

        patch_h = self._encode_branch(
            patch_input, self.patch_embed, self.patch_data_mlp, self.patch_spatial_mlp
        )
        down_h = self._encode_branch(
            downsamp_input, self.down_embed, self.down_data_mlp, self.down_spatial_mlp
        )
        tokens = torch.cat([patch_h, down_h], dim=1)
        tokens = self.token_proj(tokens)
        return tokens, history_data[..., 0]


class ForecastTrajectoryNet(nn.Module):
    """Universal state-conditioned forecast transition model.

    API:
        history = model.prepare_history(X)
        Z = model.transition(history, Z_prev, s_prev, s_next)
        states = model.rollout(X, trajectory)
    """

    def __init__(self, **model_args):
        super().__init__()
        self.node_size = int(model_args.get("node_size", 307))
        self.input_len = int(model_args.get("input_len", 12))
        self.output_len = int(model_args.get("output_len", 12))
        self.input_dim = int(model_args.get("input_dim", 4))
        self.output_dim = int(model_args.get("output_dim", 1))
        self.patch_len = int(model_args.get("patch_len", 3))
        self.stride = int(model_args.get("stride", 4))
        self.td_size = int(model_args.get("td_size", 288))
        self.dw_size = int(model_args.get("dw_size", 7))
        self.d_td = int(model_args.get("d_td", 32))
        self.d_dw = int(model_args.get("d_dw", 32))
        self.d_d = int(model_args.get("d_d", 32))
        self.d_spa = int(model_args.get("d_spa", 32))
        self.d_model = int(model_args.get("d_model", 64))
        self.cond_dim = int(model_args.get("cond_dim", 64))
        self.num_layer = int(model_args.get("num_layer", 2))
        self.n_heads = int(model_args.get("n_heads", 4))
        self.if_time_in_day = bool(model_args.get("if_time_in_day", True))
        self.if_day_in_week = bool(model_args.get("if_day_in_week", True))
        self.if_spatial = bool(model_args.get("if_spatial", True))
        states = list(model_args.get("states", [2, 3, 4, 6, 12]))
        self.graph = ForecastTrajectoryGraph(H=self.output_len, states=states)
        self.states = list(self.graph.states)

        self.history_encoder = SharedHistoryEncoder(
            input_len=self.input_len,
            input_dim=min(3, self.input_dim),
            patch_len=self.patch_len,
            stride=self.stride,
            node_size=self.node_size,
            td_size=self.td_size,
            dw_size=self.dw_size,
            d_d=self.d_d,
            d_td=self.d_td,
            d_dw=self.d_dw,
            d_spa=self.d_spa,
            d_model=self.d_model,
            num_layer=self.num_layer,
            if_time_in_day=self.if_time_in_day,
            if_day_in_week=self.if_day_in_week,
            if_spatial=self.if_spatial,
        )
        self.transition_core = UniversalTransitionCore(
            d_model=self.d_model,
            cond_dim=self.cond_dim,
            n_heads=self.n_heads,
            cy=self.output_dim,
            node_size=self.node_size,
            n_freq=int(model_args.get("n_freq", 8)),
            d_spa=self.d_spa,
        )
        spatial_kwargs = {
            "spatial_scheme": model_args.get("spatial_scheme", "C"),
            "adj_mx_path": model_args.get("adj_mx_path"),
            "use_gcn": False,
            "use_dynamic_spatial": False,
            "use_adaptive_adj": True,
            "adp_hidden_dim": int(model_args.get("adp_hidden_dim", 32)),
            "adp_topk": int(model_args.get("adp_topk", 20)),
            "adp_tau": float(model_args.get("adp_tau", 0.5)),
            "adp_alpha": float(model_args.get("adp_alpha", 0.1)),
            "use_hybrid_graph": True,
            "hybrid_alpha": float(model_args.get("hybrid_alpha", 0.2)),
            "post_spatial_mode": model_args.get("post_spatial_mode", "adaptive_only"),
        }
        self.spatial_module = ABCDSpatialModule(
            node_size=self.node_size,
            input_len=self.input_len,
            d_spa=self.d_spa,
            if_spatial=self.if_spatial,
            **spatial_kwargs,
        )
        self._history_encode_count = 0
        n_trans = self.transition_parameter_count()
        ids = self.transition_parameter_ids()
        print(
            "[ForecastTrajectory] unique transition parameter count = "
            f"{n_trans}  unique_param_object_ids = {len(set(ids))}"
        )
        if len(ids) != len(set(ids)):
            raise RuntimeError("transition parameter object IDs are not unique")

    def transition_parameter_count(self) -> int:
        return sum(p.numel() for p in self.transition_core.parameters())

    def transition_parameter_ids(self) -> list[int]:
        return [id(p) for p in self.transition_core.parameters()]

    def history_parameter_count(self) -> int:
        return sum(p.numel() for p in self.history_encoder.parameters())

    def reset_history_encode_count(self) -> None:
        self._history_encode_count = 0

    @property
    def history_encode_count(self) -> int:
        return int(self._history_encode_count)

    def prepare_history(self, X: torch.Tensor) -> HistoryState:
        self._history_encode_count += 1
        tokens, history_flow = self.history_encoder(X)
        pooled_nodes = tokens.mean(dim=1)
        pooled = pooled_nodes.mean(dim=1)
        return HistoryState(
            tokens=tokens,
            pooled=pooled,
            pooled_nodes=pooled_nodes,
            history_flow=history_flow,
            history_data=X,
        )

    def transition(
        self,
        history: HistoryState,
        Z_prev: Optional[torch.Tensor],
        s_prev: int,
        s_next: int,
    ) -> torch.Tensor:
        s_prev = int(s_prev)
        s_next = int(s_next)
        if not self.graph.is_legal_edge(s_prev, s_next):
            raise ValueError(f"illegal transition {s_prev} -> {s_next}")
        z_next, _e = self.transition_core(
            history_tokens=history.tokens,
            z_prev=Z_prev,
            s_prev=s_prev,
            s_next=s_next,
            H=self.graph.H,
            history_flow=history.history_flow,
        )
        z_next = self.spatial_module.refine_prediction(z_next, history.history_flow)
        return z_next

    def rollout(
        self,
        X: torch.Tensor,
        trajectory: Sequence[int],
        history: Optional[HistoryState] = None,
    ) -> dict[int, torch.Tensor]:
        tau = [int(s) for s in trajectory]
        if not tau or tau[-1] != self.graph.H:
            raise ValueError(f"trajectory must terminate at H={self.graph.H}, got {tau}")
        if history is None:
            history = self.prepare_history(X)
        out: dict[int, torch.Tensor] = {}
        z_prev = None
        s_prev = self.graph.START
        for s_next in tau:
            z_prev = self.transition(history, z_prev, s_prev, s_next)
            out[s_next] = z_prev
            s_prev = s_next
        return out

    def forward(
        self,
        history_data: torch.Tensor,
        trajectory: Optional[Sequence[int]] = None,
        **_kwargs,
    ) -> torch.Tensor:
        tau = list(trajectory) if trajectory is not None else [self.graph.H]
        states = self.rollout(history_data, tau)
        return states[self.graph.H]
