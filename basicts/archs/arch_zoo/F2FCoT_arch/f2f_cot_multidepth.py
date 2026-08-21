"""Multi-depth extension of :mod:`f2f_cot` with explicit forecast refinement.

This class deliberately adds no forecasting parameters.  It changes only the
legal recurrent program: the shared KASA reasoner may be called again at the
current resolution, producing another explicit target-space forecast state.
Decreasing-resolution transitions remain forbidden.

The shared-prefix fork used by the continuation-depth study is also defined
here.  Both ``3 -> 12`` and ``3 -> 6 -> 12`` continue from the *same executed*
``Z_3`` tensor and the same ForecastTraceMemory, not from two independently
recomputed prefixes.
"""

from __future__ import annotations

from typing import Sequence

import torch

from .f2f_cot import F2FCoTNet, ForecastReasoningState, pool_forecast


def clone_reasoning_state(state: ForecastReasoningState) -> ForecastReasoningState:
    """Copy a reasoning state without sharing writeable tensor storage."""
    return ForecastReasoningState(
        memory=state.memory.clone(),
        latest_forecast=(
            None if state.latest_forecast is None else state.latest_forecast.clone()
        ),
        current_resolution=int(state.current_resolution),
        forecasts=tuple(forecast.clone() for forecast in state.forecasts),
        resolutions=tuple(int(value) for value in state.resolutions),
    )


class F2FCoTMultiDepthNet(F2FCoTNet):
    """One shared F2F/KASA core supporting resolution moves and refinements.

    A repeated resolution is represented positionally in ``forecasts`` and
    ``resolutions`` (for example, ``Z_12^(1), Z_12^(2)``).  The recurrent
    context changes after every emitted forecast, while the underlying model
    and its resolution conditioner remain exactly shared.
    """

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
        if next_resolution < current:
            raise ValueError(
                f"reasoning resolutions cannot decrease: {current}->{next_resolution}"
            )
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
            "is_same_resolution_refinement": next_resolution == current,
        }

    def rollout(self, history_data: torch.Tensor, trajectory: Sequence[int]) -> dict:
        """Roll out a step-indexed trace without collapsing repeated states."""
        route = tuple(int(value) for value in trajectory)
        if not route or route[-1] != self.output_len:
            raise ValueError(
                f"trajectory must be nonempty and end at {self.output_len}: {route}"
            )
        self.reasoning_core.reset_call_count()
        state = self.begin_reasoning(history_data)
        steps = []
        for step_index, next_resolution in enumerate(route):
            state, diagnostics = self.reason_step(history_data, state, next_resolution)
            diagnostics = dict(diagnostics)
            diagnostics["step_index"] = step_index
            diagnostics["resolution"] = next_resolution
            steps.append(diagnostics)
        return {
            "pred": state.latest_forecast,
            "forecasts": tuple(state.forecasts),
            "resolutions": tuple(state.resolutions),
            "state": state,
            "steps": tuple(steps),
            "reasoning_calls": self.reasoning_core.call_count,
        }

    def continue_from(
        self,
        history_data: torch.Tensor,
        state: ForecastReasoningState,
        trajectory: Sequence[int],
    ) -> dict:
        """Resume reasoning from an already executed prefix.  Does not reset calls."""
        route = tuple(int(value) for value in trajectory)
        if not route:
            raise ValueError("continuation trajectory must be nonempty")
        start_calls = int(self.reasoning_core.call_count)
        steps = []
        for next_resolution in route:
            state, diagnostics = self.reason_step(history_data, state, next_resolution)
            steps.append(diagnostics)
        return {
            "pred": state.latest_forecast,
            "forecasts": tuple(state.forecasts),
            "resolutions": tuple(state.resolutions),
            "state": state,
            "steps": tuple(steps),
            "reasoning_calls": self.reasoning_core.call_count,
            "continuation_calls": self.reasoning_core.call_count - start_calls,
        }

    def rollout_shared_prefix_pair(
        self,
        history_data: torch.Tensor,
        prefix_resolution: int = 3,
        short_continuation: Sequence[int] = (12,),
        long_continuation: Sequence[int] = (6, 12),
        clone_prefix: bool = False,
    ) -> dict:
        """Execute ``X -> Z_r`` once, then fork two continuations from that state.

        By default the two branches receive the identical ``ForecastReasoningState``
        object.  ``reason_step`` is functionally pure, so they read the same
        ``Z_3`` tensor and the same memory.  Set ``clone_prefix=True`` only if a
        caller must isolate later in-place mutation; equality is then by value
        rather than by object identity.
        """
        prefix_resolution = int(prefix_resolution)
        short_route = tuple(int(value) for value in short_continuation)
        long_route = tuple(int(value) for value in long_continuation)
        if short_route[-1] != self.output_len or long_route[-1] != self.output_len:
            raise ValueError("both continuations must finish at the output horizon")

        self.reasoning_core.reset_call_count()
        start = self.begin_reasoning(history_data)
        prefix_state, prefix_diagnostics = self.reason_step(
            history_data, start, prefix_resolution
        )
        if int(self.reasoning_core.call_count) != 1:
            raise RuntimeError("shared prefix must be exactly one shared-core call")

        short_start = clone_reasoning_state(prefix_state) if clone_prefix else prefix_state
        long_start = clone_reasoning_state(prefix_state) if clone_prefix else prefix_state
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
            "memory_torch_equal_at_prefix": bool(
                torch.equal(short_start.memory, long_start.memory)
            ),
            "prefix_memory_same_object": short_start.memory is long_start.memory,
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

