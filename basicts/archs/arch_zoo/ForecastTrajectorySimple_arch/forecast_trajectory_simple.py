"""A small trajectory graph that preserves the canonical F2F computation.

The original ChainForecasting model remains a complete submodule. Its native
START -> 3 -> 6 -> 12 execution is never reconstructed: the exact original
forward is called for that trajectory. Only graph edges which the original
model does not define receive new forecasting modules.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import torch
from torch import nn

from basicts.archs.arch_zoo.ChainForecasting_arch.ChainForecasting_arch import (
    ChainForecasting,
)
from basicts.archs.arch_zoo.ChainForecasting_arch.kasa_temporal_step import interpolate_forecast
from .forecast_transition_bridge import ForecastTransitionBridge

Resolution = Optional[int]
Edge = tuple[Resolution, int]
Trajectory = tuple[int, ...]


def _as_trajectory(values: Sequence[int]) -> Trajectory:
    return tuple(int(value) for value in values)


def _edge_key(source: Resolution, target: int) -> str:
    return f"start__{target}" if source is None else f"{source}__{target}"


class ForecastTrajectorySimple(nn.Module):
    """Extend a canonical F2F model with explicit missing-edge bridges.

    Parameters are the normal ChainForecasting model arguments plus:

    trajectories:
        Trajectories that define the graph to instantiate. Every trajectory
        must end at output_len and be strictly increasing.
    freeze_f2f:
        Must be True. It is accepted only to make the invariant explicit in
        configs. The complete canonical model is always frozen; bridge
        parameters remain trainable.
    bridge_num_layer:
        Optional KASA depth for new edges. By default it equals the canonical
        temporal-step depth.

    The default forward executes the canonical trajectory. Use trajectory for
    one route shared by a batch, or trajectories for one route per sample. No
    route is selected internally.
    """

    DEFAULT_TRAJECTORIES: tuple[Trajectory, ...] = (
        (3, 6, 12),
        (3, 12),
        (2, 4, 12),
        (4, 12),
        (3, 4, 6, 12),
    )

    def __init__(self, **model_args):
        super().__init__()
        args = dict(model_args)
        raw_trajectories = args.pop("trajectories", self.DEFAULT_TRAJECTORIES)
        if not bool(args.pop("freeze_f2f", True)):
            raise ValueError(
                "ForecastTrajectorySimple requires freeze_f2f=True so bridge "
                "training cannot change the mature canonical F2F."
            )
        self.freeze_f2f = True
        bridge_num_layer = int(args.pop("bridge_num_layer", args.get("num_layer", 2)))
        bridge_correction_limit = float(args.pop("bridge_correction_limit", 2.0))

        # Building canonical F2F first also preserves seeded initialization.
        self.f2f = ChainForecasting(**args)
        self.output_len = int(self.f2f.output_len)
        self.canonical_trajectory = _as_trajectory(self.f2f.chain_lengths)
        self._validate_canonical_model()

        trajectories = [_as_trajectory(route) for route in raw_trajectories]
        if self.canonical_trajectory not in trajectories:
            trajectories.insert(0, self.canonical_trajectory)
        self.trajectories = tuple(dict.fromkeys(trajectories))
        for route in self.trajectories:
            self._validate_trajectory(route)

        self.native_edges: dict[Edge, int] = {
            (None, self.canonical_trajectory[0]): 0,
            **{
                (self.canonical_trajectory[index - 1], resolution): index
                for index, resolution in enumerate(self.canonical_trajectory[1:], start=1)
            },
        }
        self.bridge_edges = tuple(self._missing_edges(self.trajectories))
        self.bridges = nn.ModuleDict(
            {
                _edge_key(source, target): self._make_bridge(
                    source, target, bridge_num_layer, bridge_correction_limit
                )
                for source, target in self.bridge_edges
            }
        )
        self.freeze_canonical_model()

    def _validate_canonical_model(self) -> None:
        if self.canonical_trajectory != (3, 6, 12):
            raise ValueError(
                "ForecastTrajectorySimple requires canonical chain_lengths=[3, 6, 12], "
                f"got {list(self.canonical_trajectory)}"
            )
        if self.f2f.architecture_mode != "chain":
            raise ValueError("The canonical F2F must use architecture_mode='chain'.")
        if self.f2f.propagation_mode != "forecast_state":
            raise ValueError("The canonical F2F must propagate explicit forecast states.")
        if not self.f2f.use_prev_condition:
            raise ValueError("The canonical F2F must use previous-forecast conditioning.")
        if self.f2f.spatial_placement not in {"final", "none"}:
            raise ValueError(
                "Alternate trajectories support the verified spatial_placement='final' "
                f"(or 'none') semantics; got {self.f2f.spatial_placement!r}."
            )

    def _validate_trajectory(self, trajectory: Trajectory) -> None:
        if not trajectory:
            raise ValueError("A trajectory cannot be empty.")
        if trajectory[-1] != self.output_len:
            raise ValueError(
                f"Trajectory {list(trajectory)} must end at output_len={self.output_len}."
            )
        if any(value <= 0 for value in trajectory):
            raise ValueError(f"Trajectory resolutions must be positive: {list(trajectory)}")
        if any(left >= right for left, right in zip(trajectory, trajectory[1:])):
            raise ValueError(
                f"Trajectory resolutions must be strictly increasing: {list(trajectory)}"
            )

    def _missing_edges(self, trajectories: Iterable[Trajectory]) -> list[Edge]:
        native = {
            (None, self.canonical_trajectory[0]),
            *zip(self.canonical_trajectory[:-1], self.canonical_trajectory[1:]),
        }
        missing: list[Edge] = []
        seen: set[Edge] = set()
        for trajectory in trajectories:
            source: Resolution = None
            for target in trajectory:
                edge = (source, target)
                if edge not in native and edge not in seen:
                    missing.append(edge)
                    seen.add(edge)
                source = target
        return missing

    def _make_bridge(
        self,
        source: Resolution,
        target: int,
        num_layer: int,
        correction_limit: float,
    ) -> ForecastTransitionBridge:
        f2f = self.f2f
        return ForecastTransitionBridge(
            source_resolution=source,
            target_resolution=target,
            correction_limit=correction_limit,
            input_len=f2f.input_len,
            patch_len=f2f.patch_len,
            stride=f2f.stride,
            td_size=f2f.td_size,
            dw_size=f2f.dw_size,
            td_codebook=f2f.td_codebook,
            dw_codebook=f2f.dw_codebook,
            spa_codebook=f2f.spa_codebook,
            if_time_in_day=f2f.if_time_in_day,
            if_day_in_week=f2f.if_day_in_week,
            if_spatial=f2f.if_spatial,
            d_d=f2f.d_d,
            d_td=f2f.d_td,
            d_dw=f2f.d_dw,
            d_spa=f2f.d_spa,
            num_layer=num_layer,
            use_patch_branch=f2f.use_patch_branch,
            use_downsample_branch=f2f.use_downsample_branch,
            use_linear_residual_branch=f2f.use_linear_residual_branch,
            patch_data_input_mode=f2f.patch_data_input_mode,
            patch_embedding_mode=f2f.patch_embedding_mode,
            patch_feature_dim=f2f.patch_feature_dim,
        )

    def freeze_canonical_model(self) -> None:
        """Freeze every original F2F parameter, including shared codebooks."""
        self.freeze_f2f = True
        self.f2f.requires_grad_(False)
        self.f2f.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.f2f.eval()
        return self

    def bridge_parameters(self):
        """Yield only trainable bridge parameters, without duplicate codebooks."""
        canonical_parameter_ids = {id(parameter) for parameter in self.f2f.parameters()}
        seen: set[int] = set()
        for parameter in self.bridges.parameters():
            if (
                parameter.requires_grad
                and id(parameter) not in canonical_parameter_ids
                and id(parameter) not in seen
            ):
                seen.add(id(parameter))
                yield parameter

    @staticmethod
    def _extract_state_dict(checkpoint: Mapping) -> Mapping[str, torch.Tensor]:
        state = checkpoint
        for key in ("model_state_dict", "state_dict", "model"):
            value = state.get(key) if isinstance(state, Mapping) else None
            if isinstance(value, Mapping):
                state = value
                break
        if not isinstance(state, Mapping):
            raise TypeError("Checkpoint does not contain a model state dictionary.")
        return state

    def load_pretrained_f2f(
        self,
        checkpoint: str | Path | Mapping,
        *,
        strict: bool = True,
        map_location: str | torch.device = "cpu",
    ):
        """Load a mature canonical F2F checkpoint into the intact submodel."""
        if isinstance(checkpoint, (str, Path)):
            checkpoint = torch.load(checkpoint, map_location=map_location)
        state = self._extract_state_dict(checkpoint)
        cleaned = {}
        for raw_key, value in state.items():
            key = str(raw_key)
            if key.startswith("module."):
                key = key[len("module.") :]
            if key.startswith("f2f."):
                key = key[len("f2f.") :]
            cleaned[key] = value
        result = self.f2f.load_state_dict(cleaned, strict=strict)
        self.freeze_canonical_model()
        return result

    def _native_step(
        self,
        history_data: torch.Tensor,
        source: Resolution,
        target: int,
        previous: Optional[torch.Tensor],
    ) -> torch.Tensor:
        step_index = self.native_edges[(source, target)]
        previous_for_step = (
            None if previous is None else interpolate_forecast(previous, target)
        )
        return self.f2f.temporal_steps[step_index](
            history_data,
            prev_forecast=previous_for_step,
            spatial_codebook=self.f2f._spatial_codebook(),
        )

    def _bridge_step(
        self,
        history_data: torch.Tensor,
        source: Resolution,
        target: int,
        previous: Optional[torch.Tensor],
    ) -> torch.Tensor:
        bridge = self.bridges[_edge_key(source, target)]
        return bridge(
            history_data,
            previous_forecast=previous,
            spatial_codebook=self.f2f._spatial_codebook(),
        )

    def execute_transition(
        self,
        history_data: torch.Tensor,
        source: Resolution,
        target: int,
        previous_forecast: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Execute one online graph edge and return its explicit forecast state.

        This is the primitive used by sequential policies. It does not choose
        an action and it never reads a future target. Native and bridge edges
        are exactly the same modules used by :meth:`execute_trajectory`.
        """
        edge = (source, int(target))
        if edge in self.native_edges:
            return self._native_step(
                history_data, source, int(target), previous_forecast
            )
        key = _edge_key(source, int(target))
        if key not in self.bridges:
            raise ValueError(f"Unavailable forecast transition {edge}.")
        return self._bridge_step(
            history_data, source, int(target), previous_forecast
        )

    def finalize_forecast(
        self, final_temporal_forecast: torch.Tensor, history_data: torch.Tensor
    ) -> torch.Tensor:
        """Apply the unchanged final spatial refinement, when configured."""
        if self.f2f.spatial_placement == "final":
            return self.f2f._apply_spatial_refine(
                final_temporal_forecast, history_data
            )
        return final_temporal_forecast

    def _canonical_forward(self, history_data: torch.Tensor) -> dict:
        # Central invariant: no reconstructed graph arithmetic on this path.
        original = self.f2f(history_data, return_all=True)
        result = dict(original)
        result.update(
            {
                "trajectory": self.canonical_trajectory,
                "trajectory_edges": ((None, 3), (3, 6), (6, 12)),
                "edge_types": ("native", "native", "native"),
                "state_forecasts": {
                    resolution: forecast
                    for resolution, forecast in zip(
                        self.canonical_trajectory, original["temporal_preds"]
                    )
                },
            }
        )
        return result

    def execute_trajectory(
        self,
        history_data: torch.Tensor,
        trajectory: Sequence[int],
    ) -> dict:
        """Execute one explicit trajectory for every sample in history_data."""
        route = _as_trajectory(trajectory)
        self._validate_trajectory(route)
        if route == self.canonical_trajectory:
            return self._canonical_forward(history_data)

        source: Resolution = None
        previous = None
        states: dict[int, torch.Tensor] = {}
        edges: list[Edge] = []
        edge_types: list[str] = []
        for target in route:
            edge = (source, target)
            if edge in self.native_edges:
                current = self.execute_transition(
                    history_data, source, target, previous
                )
                edge_type = "native"
            else:
                key = _edge_key(source, target)
                if key not in self.bridges:
                    raise ValueError(
                        f"Trajectory {list(route)} uses unavailable edge {edge}. "
                        "Add a trajectory containing this edge when constructing the model."
                    )
                current = self.execute_transition(
                    history_data, source, target, previous
                )
                edge_type = "bridge"
            states[target] = current
            edges.append(edge)
            edge_types.append(edge_type)
            previous = current
            source = target

        raw_final = previous
        assert raw_final is not None
        prediction = self.finalize_forecast(raw_final, history_data)

        return {
            "pred": prediction,
            "final_temporal_pred": raw_final,
            "trajectory": route,
            "trajectory_edges": tuple(edges),
            "edge_types": tuple(edge_types),
            "state_forecasts": states,
            "chain_preds": [states[value] for value in route],
            "temporal_preds": [states[value] for value in route],
            "temporal_stage_preds": [states[value] for value in route],
            "chain_lengths": list(route),
            "spatial_stage_preds": [prediction]
            if self.f2f.spatial_placement == "final"
            else [],
        }

    def execute_sample_trajectories(
        self,
        history_data: torch.Tensor,
        trajectories: Sequence[Sequence[int]],
    ) -> dict:
        """Execute one trajectory per sample, grouping equal routes efficiently."""
        if len(trajectories) != history_data.shape[0]:
            raise ValueError(
                f"Got {len(trajectories)} trajectories for batch size "
                f"{history_data.shape[0]}."
            )
        groups: dict[Trajectory, list[int]] = defaultdict(list)
        normalized: list[Trajectory] = []
        for sample_index, values in enumerate(trajectories):
            route = _as_trajectory(values)
            self._validate_trajectory(route)
            normalized.append(route)
            groups[route].append(sample_index)

        grouped_predictions = []
        grouped_indices = []
        group_results = {}
        for route, indices in groups.items():
            index = torch.as_tensor(indices, device=history_data.device, dtype=torch.long)
            result = self.execute_trajectory(history_data.index_select(0, index), route)
            grouped_predictions.append(result["pred"])
            grouped_indices.append(index)
            group_results[route] = result

        concatenated_predictions = torch.cat(grouped_predictions, dim=0)
        concatenated_indices = torch.cat(grouped_indices, dim=0)
        order = torch.argsort(concatenated_indices)
        return {
            "pred": concatenated_predictions.index_select(0, order),
            "trajectories": tuple(normalized),
            "route_groups": {route: tuple(indices) for route, indices in groups.items()},
            "group_results": group_results,
        }

    def forward(
        self,
        history_data: torch.Tensor,
        future_data: torch.Tensor = None,
        batch_seen: int = 0,
        epoch: int = 0,
        train: bool = False,
        return_all: bool = False,
        return_intermediates: bool = False,
        trajectory: Optional[Sequence[int]] = None,
        trajectories: Optional[Sequence[Sequence[int]]] = None,
        **kwargs,
    ):
        del future_data, batch_seen, epoch, train, kwargs
        if trajectory is not None and trajectories is not None:
            raise ValueError("Pass either trajectory or trajectories, not both.")
        if trajectories is not None:
            result = self.execute_sample_trajectories(history_data, trajectories)
        else:
            route = self.canonical_trajectory if trajectory is None else trajectory
            result = self.execute_trajectory(history_data, route)
        if return_all or return_intermediates:
            return result
        return result["pred"]
