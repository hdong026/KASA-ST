"""Resolution-native shared KASA forecast reasoning.

This is an independent successor to F2FCoT.  It deliberately does not import or
modify the protected F2FCoT model implementation.  KASA patch/downsample trunks
encode history once.  A single shared target-grid reasoner is then reused at
three genuinely different active future lengths: 3, 6, and 12.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log, sqrt
from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from basicts.archs.arch_zoo.ChainForecasting_arch.mlp import MultiLayerPerceptron
from basicts.archs.arch_zoo.ForecastTrajectoryV2_arch.kasa_token_encoder import (
    KASATokenDownsampEncoder,
    KASATokenPatchEncoder,
    _prepare_patch_inputs,
)


FIXED_ROUTE = (3, 6, 12)


def temporal_mean_pool(value: torch.Tensor, target_len: int) -> torch.Tensor:
    """Mean-pool ``[B,T,N,C]`` over a nested temporal partition."""
    batch, source_len, nodes, channels = value.shape
    target_len = int(target_len)
    if source_len == target_len:
        return value
    if source_len % target_len != 0:
        raise ValueError(
            f"resolution {source_len} is not a nested refinement of {target_len}"
        )
    group = source_len // target_len
    return value.reshape(batch, target_len, group, nodes, channels).mean(dim=2)


def repeat_to_resolution(value: torch.Tensor, target_len: int) -> torch.Tensor:
    """Piecewise-constant right inverse of :func:`temporal_mean_pool`."""
    source_len = int(value.shape[1])
    target_len = int(target_len)
    if source_len == target_len:
        return value
    if target_len % source_len != 0:
        raise ValueError(
            f"resolution {source_len} does not divide target resolution {target_len}"
        )
    return value.repeat_interleave(target_len // source_len, dim=1)


def _cross_attend(query: torch.Tensor, evidence: torch.Tensor) -> torch.Tensor:
    """Per-node attention: ``[B,R,N,D] x [B,K,N,D] -> [B,R,N,D]``."""
    scale = 1.0 / sqrt(float(query.shape[-1]))
    logits = torch.einsum("brnd,bknd->bnrk", query, evidence) * scale
    weights = torch.softmax(logits, dim=-1)
    return torch.einsum("bnrk,bknd->brnd", weights, evidence)


@dataclass
class HistoryEvidence:
    tokens: torch.Tensor
    history_data: torch.Tensor
    history_flow: torch.Tensor
    td_codebook: torch.Tensor
    dw_codebook: torch.Tensor
    spa_codebook: torch.Tensor


@dataclass
class ResolutionNativeReasoningState:
    evidence: HistoryEvidence
    latest_forecast: Optional[torch.Tensor]
    current_resolution: int
    forecasts: tuple[torch.Tensor, ...]
    resolutions: tuple[int, ...]


class SharedKASAHistoryEvidenceEncoder(nn.Module):
    """High-capacity KASA patch/downsample trunks evaluated once per sample."""

    def __init__(
        self,
        *,
        node_size: int,
        input_len: int,
        patch_len: int,
        stride: int,
        td_size: int,
        dw_size: int,
        d_d: int,
        d_td: int,
        d_dw: int,
        d_spa: int,
        num_layer: int,
        patch_embedding_mode: str,
        patch_feature_dim: Optional[int],
    ) -> None:
        super().__init__()
        self.input_len = int(input_len)
        self.patch_len = int(patch_len)
        self.stride = int(stride)
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
            input_dim=3,
            patch_len=patch_len,
            stride=stride,
            d_d=d_d,
            d_td=d_td,
            d_dw=d_dw,
            d_spa=d_spa,
            output_len=1,
            num_layer=num_layer,
        )
        self.patch_encoder = KASATokenPatchEncoder(
            patch_data_input_mode="all",
            patch_embedding_mode=patch_embedding_mode,
            patch_feature_dim=patch_feature_dim,
            **encoder_kwargs,
        )
        self.downsample_encoder = KASATokenDownsampEncoder(**encoder_kwargs)
        # The token path stops before projection1.  Remove these unused horizon
        # heads so parameter counts cannot hide a full-horizon forecast head.
        self.patch_encoder.projection1 = nn.Identity()
        self.downsample_encoder.projection1 = nn.Identity()
        self.token_dim = int(self.patch_encoder.hidden_dim + d_spa)
        self.encode_count = 0

    def reset_encode_count(self) -> None:
        self.encode_count = 0

    def forward(self, history_data: torch.Tensor) -> HistoryEvidence:
        self.encode_count += 1
        model_input = history_data[..., :3]
        patch_input, downsample_input = _prepare_patch_inputs(
            model_input, self.input_len, self.patch_len, self.stride
        )
        patch_tokens = self.patch_encoder.forward_tokens(
            patch_input, spatial_codebook=self.spa_codebook
        )
        downsample_tokens = self.downsample_encoder.forward_tokens(
            downsample_input, spatial_codebook=self.spa_codebook
        )
        return HistoryEvidence(
            tokens=torch.cat((patch_tokens, downsample_tokens), dim=1),
            history_data=history_data,
            history_flow=history_data[..., 0],
            td_codebook=self.td_codebook,
            dw_codebook=self.dw_codebook,
            spa_codebook=self.spa_codebook,
        )


class ResolutionTransitionConditioner(nn.Module):
    """Small shared transition conditioner; no per-resolution forecasting nets."""

    def __init__(self, resolutions: Sequence[int], embedding_dim: int, d_model: int):
        super().__init__()
        values = (0, *tuple(int(value) for value in resolutions))
        self.index = {value: index for index, value in enumerate(values)}
        self.src_embedding = nn.Embedding(len(values), embedding_dim)
        self.dst_embedding = nn.Embedding(len(values), embedding_dim)
        self.continuous = nn.Sequential(
            nn.Linear(4, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.fuse = nn.Sequential(
            nn.Linear(embedding_dim * 3, embedding_dim * 2),
            nn.SiLU(),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.SiLU(),
        )
        self.to_film = nn.Linear(embedding_dim, d_model * 2)
        self.to_branch_scale = nn.Linear(embedding_dim, 3)
        self.to_low_frequency_gain = nn.Linear(embedding_dim, 1)
        nn.init.zeros_(self.to_film.weight)
        nn.init.zeros_(self.to_film.bias)
        with torch.no_grad():
            self.to_film.bias[:d_model].fill_(1.0)
        nn.init.zeros_(self.to_branch_scale.weight)
        nn.init.zeros_(self.to_branch_scale.bias)
        nn.init.zeros_(self.to_low_frequency_gain.weight)
        nn.init.zeros_(self.to_low_frequency_gain.bias)

    def forward(
        self,
        current_resolution: int,
        next_resolution: int,
        horizon: int,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        current_resolution = int(current_resolution)
        next_resolution = int(next_resolution)
        src_index = torch.full(
            (batch_size,), self.index[current_resolution], device=device, dtype=torch.long
        )
        dst_index = torch.full(
            (batch_size,), self.index[next_resolution], device=device, dtype=torch.long
        )
        continuous = torch.tensor(
            (
                current_resolution / float(horizon),
                next_resolution / float(horizon),
                (next_resolution - current_resolution) / float(horizon),
                log((next_resolution + 1.0) / (current_resolution + 1.0)),
            ),
            device=device,
            dtype=dtype,
        ).view(1, 4).expand(batch_size, -1)
        code = self.fuse(
            torch.cat(
                (
                    self.src_embedding(src_index),
                    self.dst_embedding(dst_index),
                    self.continuous(continuous),
                ),
                dim=-1,
            )
        )
        gamma, beta = self.to_film(code).chunk(2, dim=-1)
        return {
            "code": code,
            "gamma": gamma,
            "beta": beta,
            "branch_scale": 1.0
            + 0.10 * torch.tanh(self.to_branch_scale(code)),
            "low_frequency_gain": torch.sigmoid(
                self.to_low_frequency_gain(code)
            ).view(batch_size, 1, 1, 1),
        }


class ResolutionNativeTemporalBlock(nn.Module):
    """Variable-length temporal mixing plus a KASA residual MLP."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.depthwise_temporal = nn.Conv2d(
            d_model,
            d_model,
            kernel_size=(3, 1),
            padding=(1, 0),
            groups=d_model,
        )
        self.temporal_gate = nn.Parameter(torch.zeros(1))
        self.mlp = MultiLayerPerceptron(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        channels_first = value.permute(0, 3, 1, 2)
        mixed = self.depthwise_temporal(channels_first)
        channels_first = channels_first + torch.tanh(self.temporal_gate) * mixed
        channels_first = self.mlp(channels_first)
        value = channels_first.permute(0, 2, 3, 1)
        return self.norm(value)


class SharedLatentSpatialMixer(nn.Module):
    """One shared adaptive graph mixer on active target tokens."""

    def __init__(
        self,
        node_size: int,
        d_model: int,
        graph_rank: int,
        adjacency_dim: int,
        topk: int,
        condition_dim: int,
    ) -> None:
        super().__init__()
        self.topk = int(topk)
        self.adaptive_src = nn.Parameter(torch.empty(node_size, adjacency_dim))
        self.adaptive_dst = nn.Parameter(torch.empty(node_size, adjacency_dim))
        nn.init.xavier_uniform_(self.adaptive_src)
        nn.init.xavier_uniform_(self.adaptive_dst)
        self.in_proj = nn.Linear(d_model, graph_rank, bias=False)
        self.out_proj = nn.Linear(graph_rank, d_model, bias=False)
        self.gate = nn.Linear(condition_dim, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -1.38629436112)  # sigmoid = 0.2
        self.norm = nn.LayerNorm(d_model)
        self.last_active_temporal_length = 0

    def _adjacency(self) -> torch.Tensor:
        src = F.normalize(self.adaptive_src, p=2, dim=-1)
        dst = F.normalize(self.adaptive_dst, p=2, dim=-1)
        logits = torch.matmul(src, dst.transpose(0, 1)) / sqrt(float(src.shape[-1]))
        topk = min(max(self.topk, 1), logits.shape[-1])
        if topk < logits.shape[-1]:
            indices = torch.topk(logits, k=topk, dim=-1).indices
            keep = torch.zeros_like(logits, dtype=torch.bool)
            keep.scatter_(1, indices, True)
            logits = logits.masked_fill(~keep, float("-inf"))
        return torch.softmax(logits, dim=-1)

    def forward(self, value: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        self.last_active_temporal_length = int(value.shape[1])
        adjacency = self._adjacency()
        low_rank = self.in_proj(value)
        propagated = torch.einsum("ij,btjr->btir", adjacency, low_rank)
        delta = self.out_proj(propagated - low_rank)
        gate = torch.sigmoid(self.gate(condition)).view(-1, 1, 1, 1)
        return self.norm(value + gate * delta)


class ResolutionNativeSharedKASARefiner(nn.Module):
    """The only repeatedly-called main reasoner."""

    def __init__(
        self,
        *,
        node_size: int,
        horizon: int,
        token_dim: int,
        td_size: int,
        dw_size: int,
        d_td: int,
        d_dw: int,
        d_spa: int,
        num_layer: int,
        condition_dim: int,
        graph_rank: int,
        adjacency_dim: int,
        spatial_topk: int,
    ) -> None:
        super().__init__()
        self.horizon = int(horizon)
        self.d_model = int(token_dim)
        self.td_size = int(td_size)
        self.dw_size = int(dw_size)
        self.d_td = int(d_td)
        self.d_dw = int(d_dw)
        self.conditioner = ResolutionTransitionConditioner(
            FIXED_ROUTE, condition_dim, self.d_model
        )
        self.position_proj = nn.Linear(12, self.d_model)
        self.td_proj = nn.Linear(d_td, self.d_model, bias=False)
        self.dw_proj = nn.Linear(d_dw, self.d_model, bias=False)
        self.spa_proj = nn.Linear(d_spa, self.d_model, bias=False)
        self.query_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.patch_value_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.downsample_value_proj = nn.Linear(
            self.d_model, self.d_model, bias=False
        )
        self.previous_proj = nn.Linear(1, self.d_model)
        self.persistence_proj = nn.Linear(1, self.d_model)
        self.input_norm = nn.LayerNorm(self.d_model)
        self.temporal_blocks = nn.ModuleList(
            ResolutionNativeTemporalBlock(self.d_model)
            for _ in range(int(num_layer))
        )
        self.spatial_mixer = SharedLatentSpatialMixer(
            node_size=node_size,
            d_model=self.d_model,
            graph_rank=graph_rank,
            adjacency_dim=adjacency_dim,
            topk=spatial_topk,
            condition_dim=condition_dim,
        )
        self.output_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.SiLU(),
            nn.Dropout(0.10),
            nn.Linear(self.d_model // 2, 1),
        )
        nn.init.normal_(self.output_head[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.output_head[-1].bias)
        self.call_count = 0
        self.last_active_future_length = 0
        self.created_full_horizon_canvas = False

    def reset_diagnostics(self) -> None:
        self.call_count = 0
        self.last_active_future_length = 0
        self.created_full_horizon_canvas = False

    def _target_queries(
        self, evidence: HistoryEvidence, resolution: int
    ) -> torch.Tensor:
        batch, _, nodes, _ = evidence.history_data.shape
        device = evidence.history_data.device
        dtype = evidence.history_data.dtype
        index = torch.arange(resolution, device=device, dtype=dtype)
        start = index / float(resolution)
        end = (index + 1.0) / float(resolution)
        center = 0.5 * (start + end)
        width = end - start
        frequencies = (
            2.0
            * torch.pi
            * torch.arange(1, 5, device=device, dtype=dtype)
        )
        positional = torch.cat(
            (
                start[:, None],
                end[:, None],
                center[:, None],
                width[:, None],
                torch.sin(center[:, None] * frequencies[None]),
                torch.cos(center[:, None] * frequencies[None]),
            ),
            dim=-1,
        )
        query = self.position_proj(positional).view(
            1, resolution, 1, self.d_model
        ).expand(batch, -1, nodes, -1)

        last_tod = evidence.history_data[:, -1, :, 1]
        center_offset = (index + 0.5) * (self.horizon / float(resolution))
        tod_index = (
            (last_tod[:, None, :] * self.td_size)
            + center_offset.view(1, resolution, 1)
        ).long() % self.td_size
        day_index = (
            evidence.history_data[:, -1, :, 2]
            .long()
            .clamp(0, self.dw_size - 1)
        )
        td_embedding = evidence.td_codebook[tod_index]
        dw_embedding = evidence.dw_codebook[day_index][
            :, None, :, :
        ].expand(-1, resolution, -1, -1)
        spatial_embedding = evidence.spa_codebook[
            None, None, :, :
        ].expand(batch, resolution, -1, -1)
        return (
            query
            + self.td_proj(td_embedding)
            + self.dw_proj(dw_embedding)
            + self.spa_proj(spatial_embedding)
        )

    def forward(
        self,
        evidence: HistoryEvidence,
        previous_forecast: Optional[torch.Tensor],
        current_resolution: int,
        next_resolution: int,
    ) -> tuple[torch.Tensor, dict]:
        self.call_count += 1
        next_resolution = int(next_resolution)
        self.last_active_future_length = next_resolution
        batch, token_count, nodes, _ = evidence.tokens.shape
        if token_count % 2 != 0:
            raise RuntimeError(f"expected paired KASA branches, got {token_count} tokens")

        condition = self.conditioner(
            current_resolution,
            next_resolution,
            self.horizon,
            batch,
            evidence.tokens.device,
            evidence.tokens.dtype,
        )
        query = self._target_queries(evidence, next_resolution)
        query_for_attention = self.query_proj(query)
        branch_tokens = token_count // 2
        patch_evidence = self.patch_value_proj(
            evidence.tokens[:, :branch_tokens]
        )
        downsample_evidence = self.downsample_value_proj(
            evidence.tokens[:, branch_tokens:]
        )
        patch_context = _cross_attend(query_for_attention, patch_evidence)
        downsample_context = _cross_attend(
            query_for_attention, downsample_evidence
        )

        last_flow = evidence.history_flow[:, -1:, :, None]
        persistence = last_flow.expand(-1, next_resolution, -1, -1)
        if previous_forecast is None:
            anchor = persistence
            previous_context = torch.zeros_like(query)
        else:
            anchor = repeat_to_resolution(previous_forecast, next_resolution)
            previous_context = self.previous_proj(anchor)
        linear_context = self.persistence_proj(persistence)
        branch_scale = condition["branch_scale"]
        hidden = (
            query
            + previous_context
            + branch_scale[:, 0, None, None, None] * patch_context
            + branch_scale[:, 1, None, None, None] * downsample_context
            + branch_scale[:, 2, None, None, None] * linear_context
        )
        gamma = condition["gamma"][:, None, None, :]
        beta = condition["beta"][:, None, None, :]
        hidden = self.input_norm(gamma * hidden + beta)
        for block in self.temporal_blocks:
            hidden = block(hidden)
        hidden = self.spatial_mixer(hidden, condition["code"])
        raw_correction = self.output_head(hidden)

        if previous_forecast is None:
            forecast = anchor + raw_correction
            low_frequency = raw_correction
            detail = torch.zeros_like(raw_correction)
            corrected_parent = None
        else:
            parent_correction = temporal_mean_pool(
                raw_correction, current_resolution
            )
            low_frequency = repeat_to_resolution(
                parent_correction, next_resolution
            )
            detail = raw_correction - low_frequency
            low_gain = condition["low_frequency_gain"]
            forecast = anchor + low_gain * low_frequency + detail
            corrected_parent = previous_forecast + low_gain * parent_correction

        if forecast.shape[1] != next_resolution:
            raise RuntimeError(
                f"native reasoner emitted {forecast.shape[1]} tokens for r={next_resolution}"
            )
        if next_resolution < self.horizon and forecast.shape[1] == self.horizon:
            self.created_full_horizon_canvas = True
            raise RuntimeError("an intermediate full-horizon forecast canvas was created")
        return forecast, {
            "active_future_length": next_resolution,
            "active_hidden_shape": tuple(hidden.shape),
            # Target-free latent state for downstream adaptive controllers.
            # This is the same tensor already used by the frozen reasoner; it
            # does not alter forecasts, parameters, or checkpoint semantics.
            "active_hidden": hidden,
            "forecast_shape": tuple(forecast.shape),
            "anchor": anchor,
            "raw_correction": raw_correction,
            "low_frequency_correction": low_frequency,
            "detail_correction": detail,
            "corrected_parent": corrected_parent,
            "branch_scale": branch_scale,
            "low_frequency_gain": condition["low_frequency_gain"],
        }


class F2FCoTResolutionNativeV1Net(nn.Module):
    """Fixed 3->6->12 resolution-native shared KASA CoT model."""

    def __init__(self, **model_args) -> None:
        super().__init__()
        self.node_size = int(model_args.get("node_size", 307))
        self.input_len = int(model_args.get("input_len", 12))
        self.output_len = int(model_args.get("output_len", 12))
        if self.output_len != 12:
            raise ValueError("ResolutionNativeV1 currently formalizes only H=12")
        self.patch_len = int(model_args.get("patch_len", 3))
        self.stride = int(model_args.get("stride", 4))
        td_size = int(model_args.get("td_size", 288))
        dw_size = int(model_args.get("dw_size", 7))
        d_d = int(model_args.get("d_d", 64))
        d_td = int(model_args.get("d_td", 48))
        d_dw = int(model_args.get("d_dw", 48))
        d_spa = int(model_args.get("d_spa", 64))
        evidence_layers = int(model_args.get("evidence_num_layer", 2))
        reasoner_layers = int(model_args.get("reasoner_num_layer", 2))

        self.evidence_encoder = SharedKASAHistoryEvidenceEncoder(
            node_size=self.node_size,
            input_len=self.input_len,
            patch_len=self.patch_len,
            stride=self.stride,
            td_size=td_size,
            dw_size=dw_size,
            d_d=d_d,
            d_td=d_td,
            d_dw=d_dw,
            d_spa=d_spa,
            num_layer=evidence_layers,
            patch_embedding_mode=str(
                model_args.get("patch_embedding_mode", "serial_concat")
            ),
            patch_feature_dim=model_args.get("patch_feature_dim"),
        )
        self.reasoner = ResolutionNativeSharedKASARefiner(
            node_size=self.node_size,
            horizon=self.output_len,
            token_dim=self.evidence_encoder.token_dim,
            td_size=td_size,
            dw_size=dw_size,
            d_td=d_td,
            d_dw=d_dw,
            d_spa=d_spa,
            num_layer=reasoner_layers,
            condition_dim=int(model_args.get("resolution_dim", 32)),
            graph_rank=int(model_args.get("graph_rank", 16)),
            adjacency_dim=int(model_args.get("adp_hidden_dim", 32)),
            spatial_topk=int(model_args.get("adp_topk", 20)),
        )

    @staticmethod
    def _validate_transition(current_resolution: int, next_resolution: int) -> None:
        legal = {(0, 3), (3, 6), (6, 12)}
        transition = (int(current_resolution), int(next_resolution))
        if transition not in legal:
            raise ValueError(
                f"ResolutionNativeV1 supports only 0->3->6->12, got "
                f"{transition[0]}->{transition[1]}"
            )

    def begin_reasoning(
        self, history_data: torch.Tensor
    ) -> ResolutionNativeReasoningState:
        evidence = self.evidence_encoder(history_data)
        return ResolutionNativeReasoningState(
            evidence=evidence,
            latest_forecast=None,
            current_resolution=0,
            forecasts=(),
            resolutions=(),
        )

    def reason_step(
        self,
        history_data: torch.Tensor,
        state: ResolutionNativeReasoningState,
        next_resolution: int,
    ) -> tuple[ResolutionNativeReasoningState, dict]:
        del history_data  # evidence was encoded once in begin_reasoning
        self._validate_transition(state.current_resolution, int(next_resolution))
        forecast, diagnostics = self.reasoner(
            state.evidence,
            state.latest_forecast,
            state.current_resolution,
            int(next_resolution),
        )
        next_state = ResolutionNativeReasoningState(
            evidence=state.evidence,
            latest_forecast=forecast,
            current_resolution=int(next_resolution),
            forecasts=(*state.forecasts, forecast),
            resolutions=(*state.resolutions, int(next_resolution)),
        )
        return next_state, {"forecast": forecast, **diagnostics}

    def rollout(
        self, history_data: torch.Tensor, trajectory: Sequence[int] = FIXED_ROUTE
    ) -> dict:
        route = tuple(int(value) for value in trajectory)
        if route != FIXED_ROUTE:
            raise ValueError(
                f"ResolutionNativeV1 first experiment is fixed to {FIXED_ROUTE}, got {route}"
            )
        self.evidence_encoder.reset_encode_count()
        self.reasoner.reset_diagnostics()
        state = self.begin_reasoning(history_data)
        steps = []
        by_resolution = {}
        for next_resolution in route:
            state, diagnostics = self.reason_step(
                history_data, state, next_resolution
            )
            steps.append(diagnostics)
            by_resolution[next_resolution] = diagnostics["forecast"]
        return {
            "pred": state.latest_forecast,
            "forecasts": tuple(state.forecasts),
            "resolutions": tuple(state.resolutions),
            "by_resolution": by_resolution,
            "state": state,
            "steps": tuple(steps),
            "reasoning_calls": self.reasoner.call_count,
            "history_encode_count": self.evidence_encoder.encode_count,
            "created_full_horizon_canvas": self.reasoner.created_full_horizon_canvas,
        }

    def forward(
        self,
        history_data: torch.Tensor,
        trajectory: Sequence[int] = FIXED_ROUTE,
        return_all: bool = False,
        **_: object,
    ):
        output = self.rollout(history_data, trajectory)
        return output if return_all else output["pred"]

    def parameter_breakdown(self) -> dict[str, int | bool]:
        count = lambda module: sum(parameter.numel() for parameter in module.parameters())
        return {
            "total": count(self),
            "history_evidence_encoder": count(self.evidence_encoder),
            "shared_resolution_native_reasoner": count(self.reasoner),
            "one_shared_reasoner": True,
            "per_resolution_forecasting_networks": False,
            "fixed_horizon_forecast_head": False,
        }

    def shared_reasoner_parameter_ids(self) -> tuple[int, ...]:
        return tuple(id(parameter) for parameter in self.reasoner.parameters())

    def warm_start_from_f2f_cot(self, source_state: dict[str, torch.Tensor]) -> dict:
        """Copy shape-compatible KASA trunk/spatial weights from protected F2FCoT."""
        own_state = self.state_dict()
        loaded = {}
        copied_groups: dict[str, int] = {
            "codebooks": 0,
            "patch_trunk": 0,
            "downsample_trunk": 0,
            "reasoner_mlp": 0,
            "adaptive_graph": 0,
        }

        direct_prefixes = (
            (
                "evidence_encoder.td_codebook",
                "reasoning_core.td_codebook",
                "codebooks",
            ),
            (
                "evidence_encoder.dw_codebook",
                "reasoning_core.dw_codebook",
                "codebooks",
            ),
            (
                "evidence_encoder.spa_codebook",
                "reasoning_core.spa_codebook",
                "codebooks",
            ),
        )
        for destination, source, group in direct_prefixes:
            if source in source_state and own_state[destination].shape == source_state[source].shape:
                loaded[destination] = source_state[source]
                copied_groups[group] += own_state[destination].numel()

        for destination_prefix, source_prefix, group in (
            (
                "evidence_encoder.patch_encoder.",
                "reasoning_core.patch_encoder.",
                "patch_trunk",
            ),
            (
                "evidence_encoder.downsample_encoder.",
                "reasoning_core.downsample_encoder.",
                "downsample_trunk",
            ),
        ):
            for destination, tensor in own_state.items():
                if not destination.startswith(destination_prefix):
                    continue
                suffix = destination[len(destination_prefix) :]
                source = source_prefix + suffix
                if source in source_state and tensor.shape == source_state[source].shape:
                    loaded[destination] = source_state[source]
                    copied_groups[group] += tensor.numel()

        for block_index in range(len(self.reasoner.temporal_blocks)):
            destination_prefix = (
                f"reasoner.temporal_blocks.{block_index}.mlp."
            )
            source_prefix = (
                f"reasoning_core.patch_encoder.temporal_encoder.{block_index}."
            )
            for destination, tensor in own_state.items():
                if destination.startswith(destination_prefix):
                    source = source_prefix + destination[len(destination_prefix) :]
                    if source in source_state and tensor.shape == source_state[source].shape:
                        loaded[destination] = source_state[source]
                        copied_groups["reasoner_mlp"] += tensor.numel()

        for name in ("adaptive_src", "adaptive_dst"):
            destination = f"reasoner.spatial_mixer.{name}"
            source = f"spatial.{name}"
            if source in source_state and own_state[destination].shape == source_state[source].shape:
                loaded[destination] = source_state[source]
                copied_groups["adaptive_graph"] += own_state[destination].numel()

        self.load_state_dict(loaded, strict=False)
        return {
            "copied_tensors": len(loaded),
            "copied_parameters": int(sum(tensor.numel() for tensor in loaded.values())),
            "copied_groups": copied_groups,
            "source_tensors": len(source_state),
        }
