"""Route-complete nested-resolution programs for frozen ResolutionNative V1.

The forecasting architecture remains one history encoder and one shared
reasoner.  This successor only expands transition conditioning to resolutions
2 and 4, generalizes the legal program, and adds exact shared-prefix
executors.  Protected V1 modules and checkpoints are never modified.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Mapping, Sequence

import torch

from .f2f_cot_resolution_native_v1 import (
    F2FCoTResolutionNativeV1Net,
    ResolutionNativeReasoningState,
    ResolutionTransitionConditioner,
)


SUPPORTED_RESOLUTIONS = (2, 3, 4, 6, 12)
CANONICAL_ROUTE = (3, 6, 12)
ROUTES = (
    (12,),
    (2, 12),
    (2, 4, 12),
    (2, 6, 12),
    (3, 12),
    (3, 6, 12),
    (4, 12),
    (6, 12),
)
LEGAL_EDGES = (
    (0, 2),
    (0, 3),
    (0, 4),
    (0, 6),
    (0, 12),
    (2, 4),
    (2, 6),
    (2, 12),
    (3, 6),
    (3, 12),
    (4, 12),
    (6, 12),
)

_LEGACY_CONDITIONER_VALUES = (0, 3, 6, 12)
_ROUTE_COMPLETE_CONDITIONER_VALUES = (0, *SUPPORTED_RESOLUTIONS)
_EMBEDDING_KEYS = (
    "reasoner.conditioner.src_embedding.weight",
    "reasoner.conditioner.dst_embedding.weight",
)


def _interpolate_embedding_row(
    embedding: torch.Tensor,
    left_value: int,
    value: int,
    right_value: int,
) -> torch.Tensor:
    """Linearly interpolate one semantic resolution row."""
    left_index = _LEGACY_CONDITIONER_VALUES.index(int(left_value))
    right_index = _LEGACY_CONDITIONER_VALUES.index(int(right_value))
    fraction = float(value - left_value) / float(right_value - left_value)
    return embedding[left_index] * (1.0 - fraction) + embedding[right_index] * fraction


def expand_v1_conditioner_embedding(embedding: torch.Tensor) -> torch.Tensor:
    """Map V1 rows ``0/3/6/12`` to ``0/2/3/4/6/12`` deterministically."""
    if embedding.ndim != 2 or embedding.shape[0] != len(
        _LEGACY_CONDITIONER_VALUES
    ):
        raise ValueError(
            "expected a V1 conditioner embedding with rows (0, 3, 6, 12), "
            f"got shape {tuple(embedding.shape)}"
        )
    rows = {
        value: embedding[index]
        for index, value in enumerate(_LEGACY_CONDITIONER_VALUES)
    }
    rows[2] = _interpolate_embedding_row(embedding, 0, 2, 3)
    rows[4] = _interpolate_embedding_row(embedding, 3, 4, 6)
    return torch.stack(
        [rows[value] for value in _ROUTE_COMPLETE_CONDITIONER_VALUES], dim=0
    )


def _state_is_unchanged(
    state: ResolutionNativeReasoningState,
    snapshot: tuple[object, ...],
) -> bool:
    return (
        state.evidence is snapshot[0]
        and state.latest_forecast is snapshot[1]
        and state.forecasts is snapshot[2]
        and state.resolutions is snapshot[3]
        and state.current_resolution == snapshot[4]
    )


class F2FCoTResolutionNativeV1RouteCompleteNet(F2FCoTResolutionNativeV1Net):
    """One shared ResolutionNative V1 reasoner over the complete nested DAG."""

    def __init__(self, **model_args) -> None:
        super().__init__(**model_args)
        legacy_conditioner = self.reasoner.conditioner
        embedding_dim = int(legacy_conditioner.src_embedding.embedding_dim)
        expanded_conditioner = ResolutionTransitionConditioner(
            SUPPORTED_RESOLUTIONS,
            embedding_dim,
            self.reasoner.d_model,
        )
        legacy_state = legacy_conditioner.state_dict()
        expanded_state = OrderedDict()
        for name, destination in expanded_conditioner.state_dict().items():
            source = legacy_state[name]
            if name in ("src_embedding.weight", "dst_embedding.weight"):
                expanded_state[name] = expand_v1_conditioner_embedding(source)
            else:
                if source.shape != destination.shape:
                    raise RuntimeError(
                        f"conditioner incompatibility at {name}: "
                        f"{tuple(source.shape)} != {tuple(destination.shape)}"
                    )
                expanded_state[name] = source
        expanded_conditioner.load_state_dict(expanded_state, strict=True)
        self.reasoner.conditioner = expanded_conditioner

    @staticmethod
    def _validate_transition(current_resolution: int, next_resolution: int) -> None:
        current = int(current_resolution)
        nxt = int(next_resolution)
        if nxt not in SUPPORTED_RESOLUTIONS:
            raise ValueError(f"unsupported next resolution {nxt}")
        if current == 0:
            return
        if current not in SUPPORTED_RESOLUTIONS:
            raise ValueError(f"unsupported current resolution {current}")
        if nxt <= current or nxt % current != 0:
            raise ValueError(
                "only strictly increasing nested refinements are legal, "
                f"got {current}->{nxt}"
            )

    @classmethod
    def _validate_complete_route(cls, trajectory: Sequence[int]) -> tuple[int, ...]:
        route = tuple(int(value) for value in trajectory)
        if not route:
            raise ValueError("a complete route must be nonempty")
        current = 0
        for nxt in route:
            cls._validate_transition(current, nxt)
            current = nxt
        if current != 12:
            raise ValueError(f"a complete route must finish at 12, got {route}")
        if route not in ROUTES:
            raise RuntimeError(f"nested route enumeration is incomplete for {route}")
        return route

    def _advance(
        self,
        history_data: torch.Tensor,
        state: ResolutionNativeReasoningState,
        trajectory: Sequence[int],
    ) -> tuple[ResolutionNativeReasoningState, tuple[dict, ...]]:
        steps = []
        for next_resolution in tuple(int(value) for value in trajectory):
            state, diagnostics = self.reason_step(
                history_data, state, next_resolution
            )
            steps.append(diagnostics)
        return state, tuple(steps)

    def _format_output(
        self,
        state: ResolutionNativeReasoningState,
        steps: Sequence[dict],
        *,
        reasoning_calls: int,
    ) -> dict:
        return {
            "pred": state.latest_forecast,
            "forecasts": tuple(state.forecasts),
            "resolutions": tuple(state.resolutions),
            "by_resolution": {
                int(resolution): forecast
                for resolution, forecast in zip(state.resolutions, state.forecasts)
            },
            "state": state,
            "steps": tuple(steps),
            "reasoning_calls": int(reasoning_calls),
            "history_encode_count": int(self.evidence_encoder.encode_count),
            "created_full_horizon_canvas": (
                self.reasoner.created_full_horizon_canvas
            ),
        }

    def rollout(
        self,
        history_data: torch.Tensor,
        trajectory: Sequence[int] = (3, 6, 12),
    ) -> dict:
        """Execute any one of the eight complete nested routes."""
        route = self._validate_complete_route(trajectory)
        self.evidence_encoder.reset_encode_count()
        self.reasoner.reset_diagnostics()
        state = self.begin_reasoning(history_data)
        state, steps = self._advance(history_data, state, route)
        return self._format_output(
            state,
            steps,
            reasoning_calls=self.reasoner.call_count,
        )

    def continue_from(
        self,
        history_data: torch.Tensor,
        state: ResolutionNativeReasoningState,
        trajectory: Sequence[int],
    ) -> dict:
        """Purely continue an executed state without encoding or mutating it."""
        continuation = tuple(int(value) for value in trajectory)
        if not continuation:
            raise ValueError("continuation trajectory must be nonempty")
        current = int(state.current_resolution)
        for nxt in continuation:
            self._validate_transition(current, nxt)
            current = nxt

        snapshot = (
            state.evidence,
            state.latest_forecast,
            state.forecasts,
            state.resolutions,
            state.current_resolution,
        )
        start_calls = int(self.reasoner.call_count)
        final_state, steps = self._advance(history_data, state, continuation)
        prefix_length = len(state.forecasts)
        prefix_forecasts_preserved = all(
            final_state.forecasts[index] is forecast
            for index, forecast in enumerate(state.forecasts)
        )
        output = self._format_output(
            final_state,
            steps,
            reasoning_calls=self.reasoner.call_count,
        )
        output.update(
            {
                "continuation_calls": self.reasoner.call_count - start_calls,
                "continuation_start_state": state,
                "identity_diagnostics": {
                    "input_state_unchanged": _state_is_unchanged(state, snapshot),
                    "evidence_same_object": final_state.evidence is state.evidence,
                    "prefix_forecasts_same_objects": prefix_forecasts_preserved,
                    "prefix_forecast_count": prefix_length,
                },
            }
        )
        return output

    def rollout_shared_prefix(
        self,
        history_data: torch.Tensor,
        prefix: Sequence[int],
        continuations: Sequence[Sequence[int]],
    ) -> dict:
        """Execute one prefix once and fork any number of exact continuations."""
        prefix_route = tuple(int(value) for value in prefix)
        continuation_routes = tuple(
            tuple(int(value) for value in route) for route in continuations
        )
        if not prefix_route:
            raise ValueError("shared prefix must be nonempty")
        if not continuation_routes or any(not route for route in continuation_routes):
            raise ValueError("at least one nonempty continuation is required")
        if len(set(continuation_routes)) != len(continuation_routes):
            raise ValueError("continuations must be unique")
        for continuation in continuation_routes:
            self._validate_complete_route((*prefix_route, *continuation))

        self.evidence_encoder.reset_encode_count()
        self.reasoner.reset_diagnostics()
        start = self.begin_reasoning(history_data)
        prefix_state, prefix_steps = self._advance(
            history_data, start, prefix_route
        )
        prefix_output = self._format_output(
            prefix_state,
            prefix_steps,
            reasoning_calls=len(prefix_route),
        )
        prefix_snapshot = (
            prefix_state.evidence,
            prefix_state.latest_forecast,
            prefix_state.forecasts,
            prefix_state.resolutions,
            prefix_state.current_resolution,
        )
        branches = OrderedDict(
            (
                continuation,
                self.continue_from(history_data, prefix_state, continuation),
            )
            for continuation in continuation_routes
        )
        prefix_forecast = prefix_state.latest_forecast
        diagnostics = {
            "history_encode_count": int(self.evidence_encoder.encode_count),
            "all_start_from_prefix_state_object": all(
                branch["continuation_start_state"] is prefix_state
                for branch in branches.values()
            ),
            "all_share_evidence_object": all(
                branch["state"].evidence is prefix_state.evidence
                for branch in branches.values()
            ),
            "all_share_prefix_forecast_object": all(
                branch["forecasts"][len(prefix_route) - 1] is prefix_forecast
                for branch in branches.values()
            ),
            "prefix_state_unchanged": _state_is_unchanged(
                prefix_state, prefix_snapshot
            ),
            "prefix_reasoning_calls": len(prefix_route),
            "total_reasoning_calls": int(self.reasoner.call_count),
        }
        return {
            "prefix": prefix_output,
            "continuations": branches,
            "identity_diagnostics": diagnostics,
        }

    def rollout_all_routes(self, history_data: torch.Tensor) -> dict:
        """Evaluate all eight terminal paths through one exact prefix tree."""
        self.evidence_encoder.reset_encode_count()
        self.reasoner.reset_diagnostics()
        root_state = self.begin_reasoning(history_data)
        states: OrderedDict[tuple[int, ...], ResolutionNativeReasoningState] = (
            OrderedDict({(): root_state})
        )
        edge_steps: OrderedDict[tuple[int, ...], dict] = OrderedDict()

        for route in ROUTES:
            parent = ()
            for next_resolution in route:
                child = (*parent, next_resolution)
                if child not in states:
                    child_state, diagnostics = self.reason_step(
                        history_data, states[parent], next_resolution
                    )
                    states[child] = child_state
                    edge_steps[child] = diagnostics
                parent = child

        route_prefix_states = OrderedDict()
        outputs = OrderedDict()
        for route in ROUTES:
            prefixes = tuple(route[:index] for index in range(1, len(route) + 1))
            path_states = tuple(states[prefix] for prefix in prefixes)
            route_prefix_states[route] = path_states
            steps = tuple(edge_steps[prefix] for prefix in prefixes)
            final_state = states[route]
            outputs[route] = self._format_output(
                final_state,
                steps,
                reasoning_calls=len(route),
            )

        shared_prefixes = OrderedDict()
        for prefix, prefix_state in states.items():
            consumers = tuple(
                route
                for route in ROUTES
                if len(route) >= len(prefix) and route[: len(prefix)] == prefix
            )
            if len(consumers) < 2:
                continue
            if prefix:
                prefix_index = len(prefix) - 1
                forecast_identity = all(
                    outputs[route]["forecasts"][prefix_index]
                    is prefix_state.latest_forecast
                    for route in consumers
                )
                state_identity = all(
                    route_prefix_states[route][prefix_index] is prefix_state
                    for route in consumers
                )
            else:
                forecast_identity = True
                state_identity = states[()] is root_state
            shared_prefixes[prefix] = {
                "consumer_routes": consumers,
                "state_same_object": state_identity,
                "forecast_same_object": forecast_identity,
            }

        naive_calls = sum(len(route) for route in ROUTES)
        identity_diagnostics = {
            "history_encode_count": int(self.evidence_encoder.encode_count),
            "all_states_share_evidence_object": all(
                state.evidence is root_state.evidence for state in states.values()
            ),
            "all_route_prefixes_use_tree_state_objects": all(
                route_prefix_states[route][index] is states[route[: index + 1]]
                for route in ROUTES
                for index in range(len(route))
            ),
            "all_route_prefixes_use_tree_forecast_objects": all(
                outputs[route]["forecasts"][index]
                is states[route[: index + 1]].latest_forecast
                for route in ROUTES
                for index in range(len(route))
            ),
            "shared_prefixes": shared_prefixes,
            "unique_reasoning_calls": int(self.reasoner.call_count),
            "naive_reasoning_calls": naive_calls,
            "saved_reasoning_calls": naive_calls - int(self.reasoner.call_count),
        }
        return {
            "routes": outputs,
            "states": states,
            "edge_steps": edge_steps,
            "route_prefix_states": route_prefix_states,
            "identity_diagnostics": identity_diagnostics,
            "history_encode_count": int(self.evidence_encoder.encode_count),
            "reasoning_calls": int(self.reasoner.call_count),
            "created_full_horizon_canvas": (
                self.reasoner.created_full_horizon_canvas
            ),
        }

    def rollout_all_routes_shared(
        self,
        history_data: torch.Tensor,
        routes: Sequence[Sequence[int]] = ROUTES,
    ) -> dict:
        """Compatibility entry point for TRAIN/VALID route-cache pipelines.

        The complete shared tree is always executed so every requested route
        reads the exact same prefix objects.  Restricting ``routes`` only
        filters returned terminal outputs; it never triggers independent
        recomputation.
        """
        requested = tuple(
            self._validate_complete_route(route) for route in routes
        )
        complete = self.rollout_all_routes(history_data)
        if requested == ROUTES:
            return complete
        complete = dict(complete)
        complete["routes"] = OrderedDict(
            (route, complete["routes"][route]) for route in requested
        )
        return complete

    def load_v1_state_dict(
        self, source_state: Mapping[str, torch.Tensor]
    ) -> dict[str, object]:
        """Strictly load a formal/shared-prefix V1 state without mutating it."""
        own_state = self.state_dict()
        source_keys = set(source_state)
        destination_keys = set(own_state)
        missing = sorted(destination_keys - source_keys)
        unexpected = sorted(source_keys - destination_keys)
        if missing or unexpected:
            raise RuntimeError(
                "V1 state_dict keys are not strictly compatible: "
                f"missing={missing}, unexpected={unexpected}"
            )

        mapped = OrderedDict()
        for name, destination in own_state.items():
            source = source_state[name]
            if not isinstance(source, torch.Tensor):
                raise TypeError(f"state_dict entry {name} is not a tensor")
            if name in _EMBEDDING_KEYS:
                expanded = expand_v1_conditioner_embedding(source)
                if expanded.shape != destination.shape:
                    raise RuntimeError(
                        f"expanded shape mismatch at {name}: "
                        f"{tuple(expanded.shape)} != {tuple(destination.shape)}"
                    )
                mapped[name] = expanded
            else:
                if source.shape != destination.shape:
                    raise RuntimeError(
                        f"strict shape mismatch at {name}: "
                        f"{tuple(source.shape)} != {tuple(destination.shape)}"
                    )
                mapped[name] = source
        super().load_state_dict(mapped, strict=True)
        return {
            "source_format": "formal_or_shared_prefix_v1",
            "strict_non_embedding_compatibility": True,
            "copied_embedding_rows": (0, 3, 6, 12),
            "interpolated_embedding_rows": (2, 4),
            "loaded_tensor_count": len(mapped),
        }

    load_from_v1_state_dict = load_v1_state_dict


# Stable aliases used by the independent continuation/policy scripts.  The V1
# name remains canonical and makes the protected lineage explicit.
F2FCoTResolutionNativeRouteCompleteNet = (
    F2FCoTResolutionNativeV1RouteCompleteNet
)

