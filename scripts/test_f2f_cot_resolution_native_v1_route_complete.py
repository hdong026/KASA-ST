"""Structural tests for ResolutionNative V1 route-complete execution."""

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
from basicts.archs.arch_zoo.F2FCoTResolutionNative_arch.f2f_cot_resolution_native_v1_route_complete import (
    LEGAL_EDGES,
    ROUTES,
    SUPPORTED_RESOLUTIONS,
    F2FCoTResolutionNativeV1RouteCompleteNet,
)
from basicts.archs.arch_zoo.F2FCoTResolutionNative_arch.f2f_cot_resolution_native_v1_shared_prefix import (
    F2FCoTResolutionNativeV1SharedPrefixNet,
)


MODEL_ARGS = {
    "node_size": 4,
    "d_d": 8,
    "d_td": 4,
    "d_dw": 4,
    "d_spa": 8,
    "evidence_num_layer": 1,
    "reasoner_num_layer": 1,
    "resolution_dim": 8,
    "graph_rank": 4,
    "adp_hidden_dim": 4,
    "adp_topk": 2,
}
EMBEDDING_KEYS = (
    "reasoner.conditioner.src_embedding.weight",
    "reasoner.conditioner.dst_embedding.weight",
)
LEGACY_VALUES = (0, 3, 6, 12)
EXPANDED_VALUES = (0, 2, 3, 4, 6, 12)


def sample(device: torch.device) -> torch.Tensor:
    history = torch.randn(1, 12, MODEL_ARGS["node_size"], 4, device=device)
    history[..., 1] = torch.rand_like(history[..., 1])
    history[..., 2] = torch.randint(
        0, 7, history[..., 2].shape, device=device
    ).to(history.dtype)
    return history


def formal_model(device: torch.device) -> F2FCoTResolutionNativeV1Net:
    return F2FCoTResolutionNativeV1Net(**MODEL_ARGS).to(device).eval()


def shared_prefix_model(
    device: torch.device,
) -> F2FCoTResolutionNativeV1SharedPrefixNet:
    return F2FCoTResolutionNativeV1SharedPrefixNet(**MODEL_ARGS).to(device).eval()


def route_complete_model(
    device: torch.device,
) -> F2FCoTResolutionNativeV1RouteCompleteNet:
    return F2FCoTResolutionNativeV1RouteCompleteNet(**MODEL_ARGS).to(device).eval()


def assert_checkpoint_mapping(device: torch.device) -> None:
    formal = formal_model(device)
    source = formal.state_dict()
    source_snapshot = {name: value.clone() for name, value in source.items()}
    source_parameter_ids = tuple(id(parameter) for parameter in formal.parameters())
    target = route_complete_model(device)
    report = target.load_v1_state_dict(source)
    target_state = target.state_dict()

    assert report["strict_non_embedding_compatibility"] is True
    assert report["copied_embedding_rows"] == (0, 3, 6, 12)
    assert report["interpolated_embedding_rows"] == (2, 4)
    assert set(source) == set(target_state)
    for name, source_value in source.items():
        torch.testing.assert_close(source_value, source_snapshot[name], rtol=0, atol=0)
        if name not in EMBEDDING_KEYS:
            assert source_value.shape == target_state[name].shape
            torch.testing.assert_close(
                target_state[name], source_value, rtol=0, atol=0
            )

    for name in EMBEDDING_KEYS:
        legacy = source[name]
        expanded = target_state[name]
        assert legacy.shape[0] == 4
        assert expanded.shape[0] == 6
        for value in LEGACY_VALUES:
            torch.testing.assert_close(
                expanded[EXPANDED_VALUES.index(value)],
                legacy[LEGACY_VALUES.index(value)],
                rtol=0,
                atol=0,
            )
        torch.testing.assert_close(
            expanded[EXPANDED_VALUES.index(2)],
            legacy[LEGACY_VALUES.index(0)] / 3.0
            + legacy[LEGACY_VALUES.index(3)] * (2.0 / 3.0),
        )
        torch.testing.assert_close(
            expanded[EXPANDED_VALUES.index(4)],
            legacy[LEGACY_VALUES.index(3)] * (2.0 / 3.0)
            + legacy[LEGACY_VALUES.index(6)] / 3.0,
        )

    assert source_parameter_ids == tuple(id(parameter) for parameter in formal.parameters())
    for name, value in formal.state_dict().items():
        torch.testing.assert_close(value, source_snapshot[name], rtol=0, atol=0)

    shared = shared_prefix_model(device)
    shared_source = shared.state_dict()
    shared_snapshot = {
        name: value.clone() for name, value in shared_source.items()
    }
    shared_target = route_complete_model(device)
    shared_report = shared_target.load_from_v1_state_dict(shared_source)
    assert shared_report["source_format"] == "formal_or_shared_prefix_v1"
    for name, value in shared_source.items():
        torch.testing.assert_close(value, shared_snapshot[name], rtol=0, atol=0)


def assert_transition_program() -> None:
    assert SUPPORTED_RESOLUTIONS == (2, 3, 4, 6, 12)
    assert ROUTES == (
        (12,),
        (2, 12),
        (2, 4, 12),
        (2, 6, 12),
        (3, 12),
        (3, 6, 12),
        (4, 12),
        (6, 12),
    )
    for current, nxt in LEGAL_EDGES:
        F2FCoTResolutionNativeV1RouteCompleteNet._validate_transition(current, nxt)

    illegal_edges = (
        (0, 5),
        (2, 2),
        (2, 3),
        (3, 4),
        (4, 6),
        (6, 4),
        (12, 12),
    )
    for current, nxt in illegal_edges:
        try:
            F2FCoTResolutionNativeV1RouteCompleteNet._validate_transition(
                current, nxt
            )
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal edge {current}->{nxt} was accepted")


def assert_generic_rollout(
    model: F2FCoTResolutionNativeV1RouteCompleteNet,
    history: torch.Tensor,
) -> None:
    for route in ROUTES:
        output = model.rollout(history, route)
        assert output["resolutions"] == route
        assert output["history_encode_count"] == 1
        assert output["reasoning_calls"] == len(route)
        assert output["created_full_horizon_canvas"] is False
        for resolution, forecast, step in zip(
            route, output["forecasts"], output["steps"]
        ):
            assert forecast.shape[1] == resolution
            assert step["active_future_length"] == resolution
            assert step["active_hidden_shape"][1] == resolution
            assert step["forecast_shape"][1] == resolution
            assert "forecast_canvas" not in step


def assert_shared_prefix_and_purity(
    model: F2FCoTResolutionNativeV1RouteCompleteNet,
    history: torch.Tensor,
) -> None:
    forked = model.rollout_shared_prefix(
        history,
        prefix=(2,),
        continuations=((12,), (4, 12), (6, 12)),
    )
    identity = forked["identity_diagnostics"]
    assert identity["history_encode_count"] == 1
    assert identity["all_start_from_prefix_state_object"] is True
    assert identity["all_share_evidence_object"] is True
    assert identity["all_share_prefix_forecast_object"] is True
    assert identity["prefix_state_unchanged"] is True
    assert identity["prefix_reasoning_calls"] == 1
    assert identity["total_reasoning_calls"] == 6

    prefix_state = forked["prefix"]["state"]
    prefix_forecast = prefix_state.latest_forecast
    for branch in forked["continuations"].values():
        assert branch["continuation_start_state"] is prefix_state
        assert branch["forecasts"][0] is prefix_forecast
        assert branch["identity_diagnostics"]["input_state_unchanged"] is True
        assert branch["identity_diagnostics"]["evidence_same_object"] is True
        assert branch["identity_diagnostics"][
            "prefix_forecasts_same_objects"
        ] is True


def assert_shared_tree(
    model: F2FCoTResolutionNativeV1RouteCompleteNet,
    history: torch.Tensor,
) -> None:
    reasoner_ids_before = model.shared_reasoner_parameter_ids()
    tree = model.rollout_all_routes(history)
    reasoner_ids_after = model.shared_reasoner_parameter_ids()
    identity = tree["identity_diagnostics"]

    assert tuple(tree["routes"]) == ROUTES
    assert tree["history_encode_count"] == 1
    assert tree["reasoning_calls"] == 15
    assert tree["created_full_horizon_canvas"] is False
    assert identity["history_encode_count"] == 1
    assert identity["all_states_share_evidence_object"] is True
    assert identity["all_route_prefixes_use_tree_state_objects"] is True
    assert identity["all_route_prefixes_use_tree_forecast_objects"] is True
    assert identity["unique_reasoning_calls"] == 15
    assert identity["naive_reasoning_calls"] == 18
    assert identity["saved_reasoning_calls"] == 3
    assert set(identity["shared_prefixes"]) == {(), (2,), (3,)}
    assert all(
        item["state_same_object"] and item["forecast_same_object"]
        for item in identity["shared_prefixes"].values()
    )

    exercised_edges = {
        (
            tree["states"][path[:-1]].current_resolution,
            path[-1],
        )
        for path in tree["edge_steps"]
    }
    assert exercised_edges == set(LEGAL_EDGES)
    for route, output in tree["routes"].items():
        assert output["resolutions"] == route
        assert output["pred"].shape[1] == 12
        for resolution, forecast, step in zip(
            route, output["forecasts"], output["steps"]
        ):
            assert forecast.shape[1] == resolution
            assert step["active_future_length"] == resolution
            assert step["active_hidden_shape"][1] == resolution
            assert step["forecast_shape"][1] == resolution
            assert "forecast_canvas" not in step

    assert reasoner_ids_before == reasoner_ids_after
    assert len(reasoner_ids_before) == len(set(reasoner_ids_before))
    breakdown = model.parameter_breakdown()
    assert breakdown["one_shared_reasoner"] is True
    assert breakdown["per_resolution_forecasting_networks"] is False
    assert not hasattr(model, "route_nets")
    assert not any(name.startswith("route_nets.") for name, _ in model.named_modules())


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(23)
    assert_transition_program()
    assert_checkpoint_mapping(device)
    model = route_complete_model(device)
    history = sample(device)
    with torch.no_grad():
        assert_generic_rollout(model, history)
        assert_shared_prefix_and_purity(model, history)
        assert_shared_tree(model, history)
    print(
        {
            "device": str(device),
            "supported_resolutions": SUPPORTED_RESOLUTIONS,
            "legal_edges": len(LEGAL_EDGES),
            "routes": len(ROUTES),
            "tree_reasoning_calls": 15,
            "history_encode_count": 1,
            "passed": True,
        }
    )


if __name__ == "__main__":
    main()

