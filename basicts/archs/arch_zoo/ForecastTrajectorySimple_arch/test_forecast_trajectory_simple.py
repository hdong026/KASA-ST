"""CPU regression checks for ForecastTrajectorySimple."""

from __future__ import annotations

from collections import OrderedDict

import torch

from basicts.archs.arch_zoo.ChainForecasting_arch.kasa_temporal_step import (
    interpolate_forecast,
)
from .forecast_trajectory_simple import ForecastTrajectorySimple
from .latency import profile_trajectory_latency
from .objectives import headroom_from_predictions, quality_latency_objective
from .online_rl_policy import OnlineResolutionActorCritic
from .progressive_selector import (
    ProgressiveTrajectorySelector,
    decision_targets,
    forecast_state_features,
    history_state_features,
)
from .sequential_budget_policy import (
    TRAJECTORIES,
    SequentialBudgetPolicy,
    compact_forecast_state_features,
    compact_history_state_features,
    compact_online_features,
    explicit_forecast_state_features,
    feasible_action_mask,
)
from .run_online_sequential_rl import rollout_online_batch


def _model_args() -> dict:
    return {
        "node_size": 5,
        "input_len": 12,
        "output_len": 12,
        "input_dim": 3,
        "patch_len": 3,
        "stride": 4,
        "td_size": 24,
        "dw_size": 7,
        "d_td": 4,
        "d_dw": 4,
        "d_d": 4,
        "d_spa": 4,
        "if_time_in_day": True,
        "if_day_in_week": True,
        "if_spatial": False,
        "num_layer": 1,
        "spatial_scheme": "C",
        "use_gcn": False,
        "use_dynamic_spatial": False,
        "use_adaptive_adj": False,
        "use_hybrid_graph": False,
        "use_patch_branch": True,
        "use_downsample_branch": True,
        "use_linear_residual_branch": True,
        "patch_embedding_mode": "serial_concat",
        "patch_data_input_mode": "all",
        "spatial_placement": "none",
        "use_prev_condition": True,
        "chain_lengths": [3, 6, 12],
        "chain_loss_weights": [0.2, 0.3, 1.0],
        "trajectories": [[3, 6, 12], [3, 12], [3, 4, 6, 12]],
        "freeze_f2f": True,
    }


def _history(batch_size: int = 3) -> torch.Tensor:
    history = torch.randn(batch_size, 12, 5, 3)
    history[..., 1] = torch.rand(batch_size, 12, 5)
    history[..., 2] = torch.randint(0, 7, (batch_size, 12, 5)).float()
    return history


def test_canonical_is_exact_original_forward():
    torch.manual_seed(7)
    model = ForecastTrajectorySimple(**_model_args()).eval()
    history = _history()
    with torch.no_grad():
        expected = model.f2f(history)
        actual = model(history, trajectory=[3, 6, 12])
    assert torch.equal(actual, expected)


def test_bridges_and_sample_specific_execution():
    torch.manual_seed(11)
    model = ForecastTrajectorySimple(**_model_args()).eval()
    history = _history()
    route_a = (3, 6, 12)
    route_b = (3, 12)
    route_c = (3, 4, 6, 12)
    with torch.no_grad():
        trace = model(history, trajectory=route_c, return_all=True)
        assert trace["edge_types"] == ("native", "bridge", "bridge", "native")
        assert tuple(trace["pred"].shape) == (3, 12, 5, 1)

        assigned = [route_a, route_b, route_c]
        grouped = model(history, trajectories=assigned)
        expected = torch.cat(
            [
                model(history[index : index + 1], trajectory=route)
                for index, route in enumerate(assigned)
            ],
            dim=0,
        )
    assert torch.allclose(grouped, expected, rtol=0.0, atol=1e-6)


def test_new_bridge_starts_as_explicit_forecast_anchor():
    torch.manual_seed(12)
    model = ForecastTrajectorySimple(**_model_args()).eval()
    history = _history(2)
    with torch.no_grad():
        direct = model(history, trajectory=(3, 12), return_all=True)
        multi = model(history, trajectory=(3, 4, 6, 12), return_all=True)
    expected_direct = interpolate_forecast(direct["state_forecasts"][3], 12)
    assert torch.equal(direct["state_forecasts"][12], expected_direct)
    expected_z4 = interpolate_forecast(multi["state_forecasts"][3], 4)
    expected_z6 = interpolate_forecast(expected_z4, 6)
    assert torch.equal(multi["state_forecasts"][4], expected_z4)
    assert torch.equal(multi["state_forecasts"][6], expected_z6)


def test_frozen_boundary_and_bridge_gradient():
    torch.manual_seed(13)
    model = ForecastTrajectorySimple(**_model_args()).train()
    history = _history(2)
    prediction = model(history, trajectory=[3, 12])
    prediction.square().mean().backward()
    assert all(parameter.grad is None for parameter in model.f2f.parameters())
    bridge_grads = [
        parameter.grad for parameter in model.bridge_parameters() if parameter.grad is not None
    ]
    assert bridge_grads
    assert sum(float(grad.abs().sum()) for grad in bridge_grads) > 0.0
    assert not model.f2f.training


def test_headroom_latency_objective_and_profiler():
    target = torch.ones(3, 12, 2, 1)
    predictions = OrderedDict(
        [
            ((3, 6, 12), target + torch.tensor([0.1, 0.5, 0.4])[:, None, None, None]),
            ((3, 12), target + torch.tensor([0.3, 0.2, 0.1])[:, None, None, None]),
        ]
    )
    report = headroom_from_predictions(predictions, target, null_val=0.0)
    assert report["oracle_route_counts"].tolist() == [1, 2]
    assert report["oracle_mae"] < report["best_fixed_mae"]

    tradeoff = quality_latency_objective(
        report["per_sample_route_mae"],
        route_times=[2.0, 1.0],
        preference_lambda=0.1,
        latency_ceiling=2.0,
    )
    assert tuple(tradeoff["selected_route_index"].shape) == (3,)

    torch.manual_seed(17)
    model = ForecastTrajectorySimple(**_model_args()).eval()
    measured = profile_trajectory_latency(
        model, _history(1), [(3, 6, 12)], warmup=0, repeats=1
    )
    assert measured[(3, 6, 12)]["median_ms"] > 0.0


def test_progressive_selector_uses_only_online_states_and_maps_routes():
    history = _history(3)
    z3 = torch.randn(3, 3, 5, 1)
    history_features = history_state_features(history)
    z3_features = forecast_state_features(history, z3)
    selector = ProgressiveTrajectorySelector(
        history_features.shape[1], z3_features.shape[1], hidden_dim=16, dropout=0.0
    ).eval()
    # Force: sample-independent START->3, followed by canonical 3->6.
    with torch.no_grad():
        for parameter in selector.parameters():
            parameter.zero_()
        selector.initial_target_mean.copy_(torch.tensor([1.0, 1.0]))
        selector.z3_target_mean.copy_(torch.tensor([1.0, 1.0]))
        selected = selector.select_route_indices(history_features, z3_features)
    assert selected.tolist() == [0, 0, 0]
    assert history_features.shape[0] == z3_features.shape[0] == 3


def test_decision_targets_are_actual_cost_differences():
    losses = torch.tensor(
        [[1.0, 1.2, 0.9, 1.4, 0.8], [1.0, 0.7, 1.3, 1.2, 1.1]]
    )
    initial, after_z3 = decision_targets(losses)
    assert torch.allclose(initial, torch.tensor([[-0.1, 0.4], [0.3, 0.2]]))
    assert torch.allclose(after_z3, torch.tensor([[0.2, -0.2], [-0.3, 0.1]]))


def test_sequential_graph_enumerates_paths_instead_of_route_classes():
    assert set(TRAJECTORIES) == {
        (2, 4, 6, 12), (2, 4, 12),
        (3, 4, 6, 12), (3, 4, 12), (3, 6, 12), (3, 12),
        (4, 6, 12), (4, 12),
    }
    costs = {route: float(len(route)) for route in TRAJECTORIES}
    mask = feasible_action_mask((3,), (4, 6, 12), 2.0, costs)
    assert mask.tolist() == [False, False, True]


def test_online_transition_sequence_is_exact_canonical():
    torch.manual_seed(21)
    model = ForecastTrajectorySimple(**_model_args()).eval()
    history = _history(2)
    with torch.no_grad():
        expected = model.f2f(history)
        z3 = model.execute_transition(history, None, 3, None)
        z6 = model.execute_transition(history, 3, 6, z3)
        z12 = model.execute_transition(history, 6, 12, z6)
        actual = model.finalize_forecast(z12, history)
    assert torch.equal(actual, expected)


def test_budget_policy_accepts_only_current_explicit_state():
    history = _history(2)
    z3 = torch.randn(2, 3, 5, 1)
    history_features = history_state_features(history)
    state_features = explicit_forecast_state_features(history, z3)
    policy = SequentialBudgetPolicy(node_count=5, hidden_dim=32).eval()
    budget = torch.tensor([5.0, 8.0])
    with torch.no_grad():
        start_logits = policy.hard_logits(None, history_features, budget)
        state_logits = policy.hard_logits(3, state_features, budget)
    assert start_logits.shape == (2, 3)
    assert state_logits.shape == (2, 3)


def test_compact_online_features_are_exact_legacy_pooling():
    history = _history(2)
    z3 = torch.randn(2, 3, 5, 1)
    expanded_history = history_state_features(history)
    expanded_state = explicit_forecast_state_features(history, z3)
    assert torch.equal(
        compact_online_features(expanded_history, 5, False),
        compact_history_state_features(history),
    )
    assert torch.equal(
        compact_online_features(expanded_state, 5, True),
        compact_forecast_state_features(history, z3),
    )


def test_online_rl_sampled_actions_execute_real_transitions():
    torch.manual_seed(31)
    args = _model_args()
    args["trajectories"] = [list(route) for route in TRAJECTORIES]
    model = ForecastTrajectorySimple(**args).eval()
    policy = OnlineResolutionActorCritic(
        node_count=5, hidden_dim=16, dropout=0.0
    ).eval()
    policy.set_budget_range(1.0, 8.0)
    history = _history(4)
    target = torch.randn(4, 12, 5, 1)
    route_costs = {route: float(len(route) + 1) for route in TRAJECTORIES}
    prefix_costs = {
        (): 0.0,
        (3,): 1.0,
        (4,): 1.0,
        (2, 4): 2.0,
        (3, 4): 2.0,
    }
    rollout = rollout_online_batch(
        model,
        policy,
        history,
        target,
        0.0,
        1.0,
        8.0,
        route_costs,
        prefix_costs,
        sample_actions=True,
        hard_cap=False,
    )
    assert rollout["sampled_decisions"] > 0
    assert (
        rollout["sampled_decisions"]
        == rollout["chosen_transition_executions"]
    )
    assert all(route[-1] == 12 for route in rollout["routes"])
    assert sum(rollout["executed_edges"].values()) >= len(history)
