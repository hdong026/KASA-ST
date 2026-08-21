"""Diagnostic shared-prefix wrapper around frozen ResolutionNative V1.

This file does not change the forecasting backbone.  It adds no parameters.
It only:

1. Relaxes the *legal program* so a nested skip ``3 -> 12`` can execute.
2. Provides an explicit shared-prefix fork: ``X -> Z3`` is computed once,
   then ``3 -> 12`` and ``3 -> 6 -> 12`` continue from that exact state.

The frozen formal V1 model trained only ``{(0, 3), (3, 6), (6, 12)}``.
``3 -> 12`` is architecturally encodable (source/destination embeddings for
3 and 12 already exist, and ``12 % 3 == 0``), but it was never a trained
transition.  Zero-shot ``3 -> 12`` is therefore documentation, not a
scientific comparison, until the same shared reasoner is continuation-trained
on both routes.
"""

from __future__ import annotations

from typing import Sequence

import torch

from .f2f_cot_resolution_native_v1 import (
    FIXED_ROUTE,
    F2FCoTResolutionNativeV1Net,
    HistoryEvidence,
    ResolutionNativeReasoningState,
)


LEGAL_ROUTES = ((3, 6, 12), (3, 12))
SHORT_CONTINUATION = (12,)
LONG_CONTINUATION = (6, 12)


def clone_history_evidence(evidence: HistoryEvidence) -> HistoryEvidence:
    """Copy writable evidence tensors.  Codebooks remain shared views."""
    return HistoryEvidence(
        tokens=evidence.tokens.clone(),
        history_data=evidence.history_data.clone(),
        history_flow=evidence.history_flow.clone(),
        td_codebook=evidence.td_codebook,
        dw_codebook=evidence.dw_codebook,
        spa_codebook=evidence.spa_codebook,
    )


def clone_reasoning_state(
    state: ResolutionNativeReasoningState,
) -> ResolutionNativeReasoningState:
    """Copy a reasoning state without sharing writable forecast storage."""
    return ResolutionNativeReasoningState(
        evidence=clone_history_evidence(state.evidence),
        latest_forecast=(
            None if state.latest_forecast is None else state.latest_forecast.clone()
        ),
        current_resolution=int(state.current_resolution),
        forecasts=tuple(forecast.clone() for forecast in state.forecasts),
        resolutions=tuple(int(value) for value in state.resolutions),
    )


class F2FCoTResolutionNativeV1SharedPrefixNet(F2FCoTResolutionNativeV1Net):
    """Same frozen V1 modules; legal nested skip ``3->12`` and shared-prefix fork.

    Parameter tensors are exactly those of :class:`F2FCoTResolutionNativeV1Net`.
    A formal V1 ``state_dict`` therefore loads with ``strict=True``.
    """

    @staticmethod
    def _validate_transition(current_resolution: int, next_resolution: int) -> None:
        current = int(current_resolution)
        nxt = int(next_resolution)
        if nxt not in (3, 6, 12):
            raise ValueError(f"unsupported next resolution {nxt}")
        if current == 0:
            return
        if nxt <= current or nxt % current != 0:
            raise ValueError(
                f"only nested increasing refinements are legal, got {current}->{nxt}"
            )

    def rollout(
        self, history_data: torch.Tensor, trajectory: Sequence[int] = FIXED_ROUTE
    ) -> dict:
        route = tuple(int(value) for value in trajectory)
        if route not in LEGAL_ROUTES:
            raise ValueError(
                f"shared-prefix diagnostic allows only {LEGAL_ROUTES}, got {route}"
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

    def continue_from(
        self,
        history_data: torch.Tensor,
        state: ResolutionNativeReasoningState,
        trajectory: Sequence[int],
    ) -> dict:
        """Resume from an already executed prefix.  Does not reset call counts."""
        route = tuple(int(value) for value in trajectory)
        if not route:
            raise ValueError("continuation trajectory must be nonempty")
        start_calls = int(self.reasoner.call_count)
        steps = []
        by_resolution = {
            int(resolution): forecast
            for resolution, forecast in zip(state.resolutions, state.forecasts)
        }
        for next_resolution in route:
            state, diagnostics = self.reason_step(
                history_data, state, next_resolution
            )
            steps.append(diagnostics)
            by_resolution[int(next_resolution)] = diagnostics["forecast"]
        return {
            "pred": state.latest_forecast,
            "forecasts": tuple(state.forecasts),
            "resolutions": tuple(state.resolutions),
            "by_resolution": by_resolution,
            "state": state,
            "steps": tuple(steps),
            "reasoning_calls": self.reasoner.call_count,
            "continuation_calls": self.reasoner.call_count - start_calls,
            "history_encode_count": self.evidence_encoder.encode_count,
            "created_full_horizon_canvas": self.reasoner.created_full_horizon_canvas,
        }

    def rollout_shared_prefix_pair(
        self,
        history_data: torch.Tensor,
        prefix_resolution: int = 3,
        short_continuation: Sequence[int] = SHORT_CONTINUATION,
        long_continuation: Sequence[int] = LONG_CONTINUATION,
        clone_prefix: bool = False,
    ) -> dict:
        """Execute ``X -> Z_r`` once, then fork two continuations from that state.

        By default both branches receive the identical
        :class:`ResolutionNativeReasoningState` object.  ``reason_step`` is
        functionally pure, so they read the same executed ``Z3`` tensor and the
        same post-Z3 evidence/context.  Independent forwards that merely both
        happen to produce a Z3 are never used.
        """
        prefix_resolution = int(prefix_resolution)
        short_route = tuple(int(value) for value in short_continuation)
        long_route = tuple(int(value) for value in long_continuation)
        if short_route[-1] != self.output_len or long_route[-1] != self.output_len:
            raise ValueError("both continuations must finish at the output horizon")

        self.evidence_encoder.reset_encode_count()
        self.reasoner.reset_diagnostics()
        start = self.begin_reasoning(history_data)
        if int(self.evidence_encoder.encode_count) != 1:
            raise RuntimeError("history evidence must be encoded exactly once")
        prefix_state, prefix_diagnostics = self.reason_step(
            history_data, start, prefix_resolution
        )
        if int(self.reasoner.call_count) != 1:
            raise RuntimeError("shared prefix must be exactly one reasoner call")

        short_start = (
            clone_reasoning_state(prefix_state) if clone_prefix else prefix_state
        )
        long_start = (
            clone_reasoning_state(prefix_state) if clone_prefix else prefix_state
        )
        short = self.continue_from(history_data, short_start, short_route)
        long = self.continue_from(history_data, long_start, long_route)

        z3_short = short["forecasts"][0]
        z3_long = long["forecasts"][0]
        prefix_z3 = prefix_state.latest_forecast
        if prefix_z3 is None:
            raise RuntimeError("prefix forecast is missing")
        shared = {
            "prefix_resolution": prefix_resolution,
            "short_route": [prefix_resolution, *short_route],
            "long_route": [prefix_resolution, *long_route],
            "clone_prefix": bool(clone_prefix),
            "z3_same_object": z3_short is z3_long,
            "z3_is_prefix_object": (z3_short is prefix_z3) and (z3_long is prefix_z3),
            "z3_torch_equal": bool(torch.equal(z3_short, z3_long)),
            "evidence_same_object": short_start.evidence is long_start.evidence,
            "latest_forecast_same_object": (
                short_start.latest_forecast is long_start.latest_forecast
            ),
            "current_resolution_at_fork": int(prefix_state.current_resolution),
            "prefix_reasoner_calls": 1,
            "prefix_history_encode_count": int(self.evidence_encoder.encode_count),
            "short_calls": 1 + len(short_route),
            "long_calls": 1 + len(long_route),
            "extra_reasoning_calls": len(long_route) - len(short_route),
        }
        return {
            "prefix": {
                "state": prefix_state,
                "forecast": prefix_z3,
                "diagnostics": prefix_diagnostics,
                "reasoning_calls": 1,
            },
            "short": short,
            "long": long,
            "shared_prefix": shared,
            "pred_short": short["pred"],
            "pred_long": long["pred"],
        }
