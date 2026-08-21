"""Structural tests for the independent resolution-native F2FCoT V1."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basicts.archs.arch_zoo.F2FCoTResolutionNative_arch import (
    F2FCoTResolutionNativeV1Net,
)
from basicts.archs.arch_zoo.F2FCoTResolutionNative_arch.f2f_cot_resolution_native_v1 import (
    temporal_mean_pool,
)


def sample(device: torch.device) -> torch.Tensor:
    history = torch.randn(3, 12, 7, 4, device=device)
    history[..., 1] = torch.rand_like(history[..., 1])
    history[..., 2] = torch.randint(
        0, 7, history[..., 2].shape, device=device
    ).to(history.dtype)
    return history


def small_model(device: torch.device) -> F2FCoTResolutionNativeV1Net:
    return F2FCoTResolutionNativeV1Net(
        node_size=7,
        d_d=8,
        d_td=4,
        d_dw=4,
        d_spa=8,
        evidence_num_layer=1,
        reasoner_num_layer=1,
        graph_rank=4,
        adp_hidden_dim=4,
        adp_topk=3,
    ).to(device)


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(17)
    model = small_model(device)
    history = sample(device)
    reasoner_ids_before = model.shared_reasoner_parameter_ids()
    output = model.rollout(history, (3, 6, 12))
    reasoner_ids_after = model.shared_reasoner_parameter_ids()

    assert [tuple(value.shape) for value in output["forecasts"]] == [
        (3, 3, 7, 1),
        (3, 6, 7, 1),
        (3, 12, 7, 1),
    ]
    assert [step["active_future_length"] for step in output["steps"]] == [3, 6, 12]
    assert [step["active_hidden_shape"][1] for step in output["steps"]] == [3, 6, 12]
    assert all("forecast_canvas" not in step for step in output["steps"])
    assert output["created_full_horizon_canvas"] is False
    assert output["history_encode_count"] == 1
    assert output["reasoning_calls"] == 3
    assert reasoner_ids_before == reasoner_ids_after
    assert len(reasoner_ids_before) == len(set(reasoner_ids_before))
    assert model.parameter_breakdown()["one_shared_reasoner"] is True
    assert model.parameter_breakdown()["per_resolution_forecasting_networks"] is False

    # The only target head is token-wise 1-D output; there is no H=12 head.
    assert model.reasoner.output_head[-1].out_features == 1
    assert not any(
        isinstance(module, nn.Conv2d) and module.out_channels == 12
        for module in model.reasoner.modules()
    )

    z3, z6, z12 = output["forecasts"]
    corrected_z3 = output["steps"][1]["corrected_parent"]
    corrected_z6 = output["steps"][2]["corrected_parent"]
    torch.testing.assert_close(temporal_mean_pool(z6, 3), corrected_z3)
    torch.testing.assert_close(temporal_mean_pool(z12, 6), corrected_z6)

    # Explicit previous forecast is causally necessary: shuffling it changes Z6.
    state = model.begin_reasoning(history)
    state, _ = model.reason_step(history, state, 3)
    full_state, _ = model.reason_step(history, state, 6)
    shuffled = replace(
        state, latest_forecast=state.latest_forecast.roll(shifts=1, dims=0)
    )
    shuffled_state, _ = model.reason_step(history, shuffled, 6)
    assert not torch.allclose(
        full_state.latest_forecast, shuffled_state.latest_forecast
    )

    loss = sum(value.abs().mean() for value in output["forecasts"])
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.reasoner.parameters())
    print(
        {
            "device": str(device),
            "forecast_shapes": [tuple(value.shape) for value in output["forecasts"]],
            "hidden_lengths": [
                step["active_hidden_shape"][1] for step in output["steps"]
            ],
            "history_encode_count": output["history_encode_count"],
            "reasoning_calls": output["reasoning_calls"],
            "parameters": model.parameter_breakdown(),
            "passed": True,
        }
    )


if __name__ == "__main__":
    main()
