"""Structural tests for the ResolutionNative V1 shared-prefix diagnostic wrapper."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basicts.archs.arch_zoo.F2FCoTResolutionNative_arch.f2f_cot_resolution_native_v1 import (
    F2FCoTResolutionNativeV1Net,
)
from basicts.archs.arch_zoo.F2FCoTResolutionNative_arch.f2f_cot_resolution_native_v1_shared_prefix import (
    F2FCoTResolutionNativeV1SharedPrefixNet,
)


def sample(device: torch.device) -> torch.Tensor:
    history = torch.randn(2, 12, 7, 4, device=device)
    history[..., 1] = torch.rand_like(history[..., 1])
    history[..., 2] = torch.randint(
        0, 7, history[..., 2].shape, device=device
    ).to(history.dtype)
    return history


def small_parent(device: torch.device) -> F2FCoTResolutionNativeV1Net:
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


def small_wrapper(device: torch.device) -> F2FCoTResolutionNativeV1SharedPrefixNet:
    return F2FCoTResolutionNativeV1SharedPrefixNet(
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
    torch.manual_seed(19)
    history = sample(device)

    frozen = small_parent(device)
    raised = False
    try:
        frozen.rollout(history, (3, 12))
    except ValueError as error:
        raised = "fixed to" in str(error) or "0->3->6->12" in str(error)
        if not raised:
            raise
    assert raised, "frozen V1 must reject the untrained 3->12 route"

    model = small_wrapper(device)
    model.load_state_dict(frozen.state_dict(), strict=True)
    parent_count = sum(parameter.numel() for parameter in frozen.parameters())
    wrapper_count = sum(parameter.numel() for parameter in model.parameters())
    assert parent_count == wrapper_count
    assert model.parameter_breakdown()["total"] == parent_count

    paired = model.rollout_shared_prefix_pair(history)
    identity = paired["shared_prefix"]
    assert identity["z3_same_object"] is True
    assert identity["z3_is_prefix_object"] is True
    assert identity["z3_torch_equal"] is True
    assert identity["evidence_same_object"] is True
    assert identity["latest_forecast_same_object"] is True
    assert identity["current_resolution_at_fork"] == 3
    assert identity["prefix_history_encode_count"] == 1
    assert identity["short_calls"] == 2
    assert identity["long_calls"] == 3
    assert identity["extra_reasoning_calls"] == 1
    assert paired["short"]["forecasts"][0] is paired["prefix"]["forecast"]
    assert paired["long"]["forecasts"][0] is paired["prefix"]["forecast"]
    assert paired["pred_short"].shape[1] == 12
    assert paired["pred_long"].shape[1] == 12
    assert paired["long"]["forecasts"][1].shape[1] == 6
    assert torch.equal(paired["short"]["forecasts"][0], paired["long"]["forecasts"][0])

    short = model.rollout(history, (3, 12))
    long = model.rollout(history, (3, 6, 12))
    assert short["resolutions"] == (3, 12)
    assert long["resolutions"] == (3, 6, 12)
    # Independent rollouts recompute Z3; they must not be used as the comparison.
    assert short["forecasts"][0] is not long["forecasts"][0]

    cloned = model.rollout_shared_prefix_pair(history, clone_prefix=True)
    assert cloned["shared_prefix"]["z3_same_object"] is False
    assert cloned["shared_prefix"]["z3_torch_equal"] is True
    print("shared-prefix structural tests passed")


if __name__ == "__main__":
    main()
