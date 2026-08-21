"""Capacity-preserving recurrent F2F reasoning built from the original KASA unit.

The model deliberately has one forecasting core.  Every explicit target-space
forecast is produced by calling that same core again with an updated memory
which is itself derived only from previously emitted forecasts.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, log
from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from basicts.archs.arch_zoo.ChainForecasting_arch.downsamp_emb import DownsampEncoder
from basicts.archs.arch_zoo.ChainForecasting_arch.gcn import ABCDSpatialModule
from basicts.archs.arch_zoo.ChainForecasting_arch.kasa_temporal_step import (
    interpolate_forecast,
)
from basicts.archs.arch_zoo.ChainForecasting_arch.patch_emb import PatchEncoder


def pool_forecast(forecast: torch.Tensor, target_len: int) -> torch.Tensor:
    """Average-pool ``[B,H,N,C]`` to an explicit ``[B,target_len,N,C]`` state."""
    batch, horizon, nodes, channels = forecast.shape
    target_len = int(target_len)
    if horizon % target_len == 0:
        group = horizon // target_len
        return forecast.reshape(batch, target_len, group, nodes, channels).mean(dim=2)
    x = forecast.permute(0, 2, 3, 1).reshape(batch * nodes, channels, horizon)
    x = F.adaptive_avg_pool1d(x, target_len)
    return x.reshape(batch, nodes, channels, target_len).permute(0, 3, 1, 2)


class ResolutionConditioner(nn.Module):
    """Small shared source/destination conditioner; it is not an edge network."""

    def __init__(
        self,
        resolutions: Sequence[int],
        embedding_dim: int,
        condition_channels: int,
    ) -> None:
        super().__init__()
        values = (0, *sorted({int(value) for value in resolutions}))
        self.values = values
        self.index = {value: idx for idx, value in enumerate(values)}
        self.src_embedding = nn.Embedding(len(values), embedding_dim)
        self.dst_embedding = nn.Embedding(len(values), embedding_dim)
        self.continuous = nn.Sequential(
            nn.Linear(4, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.fuse = nn.Sequential(
            nn.Linear(3 * embedding_dim, 2 * embedding_dim),
            nn.SiLU(),
            nn.Linear(2 * embedding_dim, embedding_dim),
            nn.SiLU(),
        )
        self.to_planes = nn.Linear(embedding_dim, condition_channels)
        self.to_branch_scale = nn.Linear(embedding_dim, 3)
        nn.init.zeros_(self.to_branch_scale.weight)
        nn.init.zeros_(self.to_branch_scale.bias)

    def forward(
        self,
        current_resolution: int,
        next_resolution: int,
        horizon: int,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        current_resolution = int(current_resolution)
        next_resolution = int(next_resolution)
        if current_resolution not in self.index or next_resolution not in self.index:
            raise ValueError(
                f"unsupported resolution transition {current_resolution}->{next_resolution}; "
                f"known resolutions are {self.values}"
            )
        src_idx = torch.full(
            (batch_size,), self.index[current_resolution], device=device, dtype=torch.long
        )
        dst_idx = torch.full(
            (batch_size,), self.index[next_resolution], device=device, dtype=torch.long
        )
        continuous = torch.tensor(
            [
                current_resolution / float(horizon),
                next_resolution / float(horizon),
                (next_resolution - current_resolution) / float(horizon),
                log((next_resolution + 1.0) / (current_resolution + 1.0)),
            ],
            device=device,
            dtype=dtype,
        ).view(1, 4).expand(batch_size, -1)
        code = self.fuse(
            torch.cat(
                (
                    self.src_embedding(src_idx),
                    self.dst_embedding(dst_idx),
                    self.continuous(continuous),
                ),
                dim=-1,
            )
        )
        planes = self.to_planes(code)
        # Original KASA sums its three branches.  Conditioning is initialized at
        # exactly that behavior and is constrained to a lightweight modulation.
        branch_scale = 1.0 + 0.10 * torch.tanh(self.to_branch_scale(code))
        return code, planes, branch_scale


class ForecastTraceMemory(nn.Module):
    """Compact recurrent memory updated exclusively from explicit forecasts."""

    def __init__(self, resolution_dim: int, memory_dim: int, context_channels: int):
        super().__init__()
        self.memory_dim = int(memory_dim)
        update_dim = 1 + int(resolution_dim)
        self.update_gate = nn.Linear(update_dim + memory_dim, memory_dim)
        self.reset_gate = nn.Linear(update_dim + memory_dim, memory_dim)
        self.candidate = nn.Linear(update_dim + memory_dim, memory_dim)
        self.to_context = nn.Sequential(
            nn.LayerNorm(memory_dim),
            nn.Linear(memory_dim, context_channels),
        )

    def initial(self, history: torch.Tensor, horizon: int) -> torch.Tensor:
        return history.new_zeros(
            history.shape[0], int(horizon), history.shape[2], self.memory_dim
        )

    def update(
        self,
        memory: torch.Tensor,
        explicit_forecast: torch.Tensor,
        resolution_code: torch.Tensor,
        horizon: int,
    ) -> torch.Tensor:
        z_canvas = interpolate_forecast(explicit_forecast, int(horizon))
        code = resolution_code[:, None, None, :].expand(
            -1, int(horizon), z_canvas.shape[2], -1
        )
        update_input = torch.cat((z_canvas, code), dim=-1)
        combined = torch.cat((update_input, memory), dim=-1)
        update_gate = torch.sigmoid(self.update_gate(combined))
        reset_gate = torch.sigmoid(self.reset_gate(combined))
        candidate = torch.tanh(
            self.candidate(torch.cat((update_input, reset_gate * memory), dim=-1))
        )
        return (1.0 - update_gate) * memory + update_gate * candidate

    def context(self, memory: torch.Tensor, input_len: int) -> torch.Tensor:
        context = self.to_context(memory)
        if context.shape[1] != int(input_len):
            context = interpolate_forecast(context, int(input_len))
        return context


class SharedKASAReasoningCore(nn.Module):
    """One enlarged version of the original patch/downsample/linear KASA step."""

    def __init__(
        self,
        *,
        node_size: int,
        input_len: int,
        output_len: int,
        patch_len: int,
        stride: int,
        td_size: int,
        dw_size: int,
        d_d: int,
        d_td: int,
        d_dw: int,
        d_spa: int,
        num_layer: int,
        context_channels: int,
        condition_channels: int,
        patch_data_input_mode: str,
        patch_embedding_mode: str,
        patch_feature_dim: Optional[int],
    ) -> None:
        super().__init__()
        self.node_size = int(node_size)
        self.input_len = int(input_len)
        self.output_len = int(output_len)
        self.patch_len = int(patch_len)
        self.stride = int(stride)
        self.context_channels = int(context_channels)
        self.condition_channels = int(condition_channels)
        self.encoder_input_dim = 3 + 1 + context_channels + condition_channels

        self.td_codebook = nn.Parameter(torch.empty(td_size, d_td))
        self.dw_codebook = nn.Parameter(torch.empty(dw_size, d_dw))
        self.spa_codebook = nn.Parameter(torch.empty(node_size, d_spa))
        nn.init.xavier_uniform_(self.td_codebook)
        nn.init.xavier_uniform_(self.dw_codebook)
        nn.init.xavier_uniform_(self.spa_codebook)

        encoder_kwargs = dict(
            td_size=td_size,
            dw_size=dw_size,
            td_codebook=self.td_codebook,
            dw_codebook=self.dw_codebook,
            spa_codebook=self.spa_codebook,
            if_time_in_day=True,
            if_day_in_week=True,
            if_spatial=True,
            input_dim=self.encoder_input_dim,
            patch_len=patch_len,
            stride=stride,
            d_d=d_d,
            d_td=d_td,
            d_dw=d_dw,
            d_spa=d_spa,
            output_len=output_len,
            num_layer=num_layer,
        )
        self.patch_encoder = PatchEncoder(
            patch_data_input_mode=patch_data_input_mode,
            patch_embedding_mode=patch_embedding_mode,
            patch_feature_dim=patch_feature_dim,
            **encoder_kwargs,
        )
        self.downsample_encoder = DownsampEncoder(**encoder_kwargs)
        self.linear_residual = nn.Conv2d(input_len, output_len, kernel_size=(1, 1))
        self.call_count = 0
        self._initialize_recurrent_forecast_heads()

    def _initialize_recurrent_forecast_heads(self) -> None:
        """Start repeated reasoning near stable last-observation persistence.

        Unlike the original independently-called stages, one bad shared call is
        fed back into the next call.  Small learned KASA heads plus an exact
        persistence linear path avoid early recurrent amplification while all
        components remain trainable from the first minibatch.
        """
        for encoder in (self.patch_encoder, self.downsample_encoder):
            nn.init.normal_(encoder.projection1.weight, mean=0.0, std=1e-3)
            nn.init.zeros_(encoder.projection1.bias)
        nn.init.zeros_(self.linear_residual.weight)
        nn.init.zeros_(self.linear_residual.bias)
        with torch.no_grad():
            self.linear_residual.weight[:, self.input_len - 1, 0, 0] = 1.0

    def reset_call_count(self) -> None:
        self.call_count = 0

    def forward(
        self,
        history_data: torch.Tensor,
        latest_forecast: Optional[torch.Tensor],
        memory_context: torch.Tensor,
        resolution_planes: torch.Tensor,
        branch_scale: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        self.call_count += 1
        batch, _, nodes, _ = history_data.shape
        if latest_forecast is None:
            latest = history_data.new_zeros(batch, self.input_len, nodes, 1)
        else:
            latest = interpolate_forecast(latest_forecast, self.input_len)
        planes = resolution_planes[:, None, None, :].expand(
            -1, self.input_len, nodes, -1
        )
        step_input = torch.cat(
            (history_data[..., :3], latest, memory_context, planes), dim=-1
        )
        if step_input.shape[-1] != self.encoder_input_dim:
            raise RuntimeError(
                f"reasoning input has {step_input.shape[-1]} channels, "
                f"expected {self.encoder_input_dim}"
            )

        in_len_add = ceil(self.input_len / self.stride) * self.stride - self.input_len
        if in_len_add:
            augmented = torch.cat(
                (step_input[:, -1:].expand(-1, in_len_add, -1, -1), step_input), dim=1
            )
        else:
            augmented = step_input
        downsample_input = torch.stack(
            [augmented[:, idx :: self.stride] for idx in range(self.stride)], dim=1
        )
        patch_input = augmented.unfold(
            dimension=1, size=self.patch_len, step=self.patch_len
        ).permute(0, 1, 4, 2, 3)

        patch = self.patch_encoder(patch_input, spatial_codebook=self.spa_codebook)
        downsample = self.downsample_encoder(
            downsample_input, spatial_codebook=self.spa_codebook
        )
        linear = self.linear_residual(history_data[..., 0:1])
        scales = [branch_scale[:, idx].view(batch, 1, 1, 1) for idx in range(3)]
        forecast_canvas = scales[0] * patch + scales[1] * downsample + scales[2] * linear
        return forecast_canvas, {
            "patch": patch,
            "downsample": downsample,
            "linear": linear,
            "branch_scale": branch_scale,
        }


@dataclass
class ForecastReasoningState:
    memory: torch.Tensor
    latest_forecast: Optional[torch.Tensor]
    current_resolution: int
    forecasts: tuple[torch.Tensor, ...]
    resolutions: tuple[int, ...]


class F2FCoTNet(nn.Module):
    """Repeated shared KASA forecasting with explicit forecast reasoning states."""

    def __init__(self, **model_args) -> None:
        super().__init__()
        self.node_size = int(model_args.get("node_size", 307))
        self.input_len = int(model_args.get("input_len", 12))
        self.output_len = int(model_args.get("output_len", 12))
        self.patch_len = int(model_args.get("patch_len", 3))
        self.stride = int(model_args.get("stride", 4))
        self.resolutions = tuple(
            sorted({int(value) for value in model_args.get("resolutions", [2, 3, 4, 6, 12])})
        )
        if self.output_len not in self.resolutions:
            raise ValueError("resolutions must contain output_len")
        if any(value <= 0 or value > self.output_len for value in self.resolutions):
            raise ValueError(f"invalid resolutions: {self.resolutions}")

        resolution_dim = int(model_args.get("resolution_dim", 32))
        context_channels = int(model_args.get("context_channels", 4))
        condition_channels = int(model_args.get("condition_channels", 8))
        memory_dim = int(model_args.get("memory_dim", 16))
        d_spa = int(model_args.get("d_spa", 64))

        self.resolution_conditioner = ResolutionConditioner(
            self.resolutions, resolution_dim, condition_channels
        )
        self.trace_memory = ForecastTraceMemory(
            resolution_dim=resolution_dim,
            memory_dim=memory_dim,
            context_channels=context_channels,
        )
        self.reasoning_core = SharedKASAReasoningCore(
            node_size=self.node_size,
            input_len=self.input_len,
            output_len=self.output_len,
            patch_len=self.patch_len,
            stride=self.stride,
            td_size=int(model_args.get("td_size", 288)),
            dw_size=int(model_args.get("dw_size", 7)),
            d_d=int(model_args.get("d_d", 64)),
            d_td=int(model_args.get("d_td", 48)),
            d_dw=int(model_args.get("d_dw", 48)),
            d_spa=d_spa,
            num_layer=int(model_args.get("num_layer", 4)),
            context_channels=context_channels,
            condition_channels=condition_channels,
            patch_data_input_mode=str(model_args.get("patch_data_input_mode", "all")),
            patch_embedding_mode=str(model_args.get("patch_embedding_mode", "serial_concat")),
            patch_feature_dim=model_args.get("patch_feature_dim"),
        )
        self.spatial = ABCDSpatialModule(
            node_size=self.node_size,
            input_len=self.input_len,
            d_spa=d_spa,
            if_spatial=True,
            spatial_scheme=str(model_args.get("spatial_scheme", "C")),
            adj_mx_path=model_args.get("adj_mx_path"),
            use_gcn=bool(model_args.get("use_gcn", True)),
            gcn_hidden_dim=int(model_args.get("gcn_hidden_dim", 64)),
            use_dynamic_spatial=bool(model_args.get("use_dynamic_spatial", True)),
            dyn_hidden_dim=int(model_args.get("dyn_hidden_dim", 64)),
            dyn_topk=int(model_args.get("dyn_topk", 20)),
            dyn_tau=float(model_args.get("dyn_tau", 0.5)),
            dyn_static_weight=float(model_args.get("dyn_static_weight", 0.2)),
            use_adaptive_adj=bool(model_args.get("use_adaptive_adj", True)),
            adp_hidden_dim=int(model_args.get("adp_hidden_dim", 32)),
            adp_topk=int(model_args.get("adp_topk", 20)),
            adp_tau=float(model_args.get("adp_tau", 0.5)),
            use_hybrid_graph=bool(model_args.get("use_hybrid_graph", True)),
            hybrid_alpha=float(model_args.get("hybrid_alpha", 0.2)),
            post_spatial_mode=str(model_args.get("post_spatial_mode", "adaptive_only")),
        )

    def begin_reasoning(self, history_data: torch.Tensor) -> ForecastReasoningState:
        return ForecastReasoningState(
            memory=self.trace_memory.initial(history_data, self.output_len),
            latest_forecast=None,
            current_resolution=0,
            forecasts=(),
            resolutions=(),
        )

    def reason_step(
        self,
        history_data: torch.Tensor,
        state: ForecastReasoningState,
        next_resolution: int,
    ) -> tuple[ForecastReasoningState, dict]:
        next_resolution = int(next_resolution)
        current = int(state.current_resolution)
        if next_resolution not in self.resolutions:
            raise ValueError(f"unsupported next resolution {next_resolution}")
        if next_resolution <= current:
            raise ValueError(f"reasoning resolutions must increase: {current}->{next_resolution}")
        batch = history_data.shape[0]
        code, planes, branch_scale = self.resolution_conditioner(
            current,
            next_resolution,
            self.output_len,
            batch,
            history_data.device,
            history_data.dtype,
        )
        memory_context = self.trace_memory.context(state.memory, self.input_len)
        canvas, branches = self.reasoning_core(
            history_data,
            state.latest_forecast,
            memory_context,
            planes,
            branch_scale,
        )
        explicit = pool_forecast(canvas, next_resolution)
        explicit = self.spatial.refine_prediction(explicit, history_data[..., 0])
        updated_memory = self.trace_memory.update(
            state.memory, explicit, code, self.output_len
        )
        new_state = ForecastReasoningState(
            memory=updated_memory,
            latest_forecast=explicit,
            current_resolution=next_resolution,
            forecasts=(*state.forecasts, explicit),
            resolutions=(*state.resolutions, next_resolution),
        )
        return new_state, {
            "forecast": explicit,
            "forecast_canvas": canvas,
            "memory": updated_memory,
            "resolution_code": code,
            "branches": branches,
        }

    def rollout(self, history_data: torch.Tensor, trajectory: Sequence[int]) -> dict:
        route = tuple(int(value) for value in trajectory)
        if not route or route[-1] != self.output_len:
            raise ValueError(f"trajectory must be nonempty and end at {self.output_len}: {route}")
        self.reasoning_core.reset_call_count()
        state = self.begin_reasoning(history_data)
        by_resolution: dict[int, torch.Tensor] = {}
        steps = []
        for next_resolution in route:
            state, diagnostics = self.reason_step(history_data, state, next_resolution)
            by_resolution[next_resolution] = diagnostics["forecast"]
            steps.append(diagnostics)
        return {
            "pred": state.latest_forecast,
            "forecasts": tuple(state.forecasts),
            "resolutions": tuple(state.resolutions),
            "by_resolution": by_resolution,
            "state": state,
            "steps": tuple(steps),
            "reasoning_calls": self.reasoning_core.call_count,
        }

    def forward(
        self,
        history_data: torch.Tensor,
        trajectory: Sequence[int] = (3, 6, 12),
        return_all: bool = False,
        **_: object,
    ):
        output = self.rollout(history_data, trajectory)
        return output if return_all else output["pred"]

    @staticmethod
    def _count(module: nn.Module) -> int:
        return sum(parameter.numel() for parameter in module.parameters())

    def parameter_breakdown(self) -> dict[str, int | float | bool]:
        total = self._count(self)
        core = self._count(self.reasoning_core)
        resolution = self._count(self.resolution_conditioner)
        context = self._count(self.trace_memory)
        spatial = self._count(self.spatial)
        return {
            "total": total,
            "shared_kasa_reasoning_core": core,
            "forecast_trace_memory": context,
            "resolution_conditioning": resolution,
            "spatial": spatial,
            "accounted_total": core + resolution + context + spatial,
            "one_shared_reasoning_core": True,
            "per_edge_forecasting_networks": False,
        }

    def resolution_specific_parameter_count(self) -> int:
        """Only embedding rows vary by resolution; all projections are shared."""
        return (
            self.resolution_conditioner.src_embedding.weight.numel()
            + self.resolution_conditioner.dst_embedding.weight.numel()
        )
