"""Unit/structural tests for the exact ResolutionNative eight-route policy."""

from __future__ import annotations

import json
import sys
import tempfile
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basicts.archs.arch_zoo.F2FCoTResolutionNative_arch.resolution_native_route_complete_policy import (
    ACTION_INDEX,
    LEGAL_NEXT,
    PREFIXES,
    PREFIX_INDEX,
    ROUTES,
    PrimalDualBudgetController,
    ResolutionNativeExactRouter,
    exact_full_information_objective,
    legal_action_mask,
    load_actual_route_flops,
    load_route_analysis_cache,
    mean_centered_advantages,
    negative_control_states,
    normalize_actual_flops,
    require_passing_gate,
    robust_global_route_margin_scale,
)
from scripts.train_resolution_native_route_complete_policy import train


def test_dag_and_hard_masks() -> None:
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
    expected = {
        (): (2, 3, 4, 6, 12),
        (2,): (4, 6, 12),
        (2, 4): (12,),
        (2, 6): (12,),
        (3,): (6, 12),
        (3, 6): (12,),
        (4,): (12,),
        (6,): (12,),
    }
    assert LEGAL_NEXT == expected
    for prefix, allowed in expected.items():
        mask = legal_action_mask(prefix)
        actual = tuple(
            resolution
            for resolution, keep in zip((2, 3, 4, 6, 12), mask.tolist())
            if keep
        )
        assert actual == allowed


def test_route_probability_factorization_and_normalization() -> None:
    torch.manual_seed(7)
    model = ResolutionNativeExactRouter(11, hidden_dim=19, dropout=0.0).eval()
    states = torch.randn(5, len(PREFIXES), 11)
    budgets = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])
    output = model.terminal_distribution(states, budgets)
    probs = output["route_probs"]
    action = {
        prefix: values.exp()
        for prefix, values in output["action_log_probs"].items()
    }
    manual = []
    for route in ROUTES:
        prefix: tuple[int, ...] = ()
        product = torch.ones(states.shape[0])
        for next_resolution in route:
            product = product * action[prefix][:, ACTION_INDEX[next_resolution]]
            prefix = (*prefix, next_resolution)
        manual.append(product)
    manual_t = torch.stack(manual, dim=-1)
    torch.testing.assert_close(output["raw_route_action_products"], manual_t)
    torch.testing.assert_close(
        probs, manual_t / manual_t.sum(dim=-1, keepdim=True)
    )
    torch.testing.assert_close(probs.sum(dim=-1), torch.ones(states.shape[0]))
    torch.testing.assert_close(
        output["route_normalizer"], torch.ones(states.shape[0], 1), atol=1e-6, rtol=1e-6
    )
    for prefix, values in action.items():
        mask = legal_action_mask(prefix)
        assert torch.equal(values[:, ~mask], torch.zeros_like(values[:, ~mask]))
        torch.testing.assert_close(values[:, mask].sum(dim=-1), torch.ones(states.shape[0]))


def test_global_scaling_preserves_route_margins() -> None:
    losses = torch.tensor(
        [
            [1.00, 1.10, 1.20, 1.30, 1.40, 1.50, 1.60, 1.70],
            [2.00, 2.02, 2.04, 2.06, 2.08, 2.10, 2.12, 2.14],
            [3.00, 3.50, 4.00, 4.50, 5.00, 5.50, 6.00, 6.50],
        ]
    )
    scale = robust_global_route_margin_scale(losses, split="train")
    scaled = mean_centered_advantages(-losses / scale)
    for row in range(losses.shape[0]):
        original_margin = losses[row, 7] - losses[row, 0]
        scaled_margin = scaled[row, 0] - scaled[row, 7]
        torch.testing.assert_close(scaled_margin, original_margin / scale)
    ratio_before = (losses[2, 7] - losses[2, 0]) / (
        losses[1, 7] - losses[1, 0]
    )
    ratio_after = (scaled[2, 0] - scaled[2, 7]) / (
        scaled[1, 0] - scaled[1, 7]
    )
    torch.testing.assert_close(ratio_after, ratio_before)
    try:
        robust_global_route_margin_scale(losses, split="valid")
    except ValueError as error:
        assert "TRAIN only" in str(error)
    else:
        raise AssertionError("VALID was incorrectly accepted for global scale fitting")


def test_dual_update_and_expected_budget_accounting() -> None:
    controller = PrimalDualBudgetController(
        [0.25, 0.75], learning_rate=0.2, max_lambda=3.0
    )
    updated = controller.update(torch.tensor([0.50, 0.50]))
    torch.testing.assert_close(updated, torch.tensor([0.05, 0.00]))
    updated = controller.update(torch.tensor([0.00, 1.00]))
    torch.testing.assert_close(updated, torch.tensor([0.00, 0.05]))

    raw_flops = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0])
    costs = normalize_actual_flops(raw_flops)
    route_probs = torch.zeros(2, len(ROUTES))
    route_probs[0, 0] = 0.25
    route_probs[0, 7] = 0.75
    route_probs[1, 0] = 1.0
    losses = torch.ones_like(route_probs)
    _, details = exact_full_information_objective(
        route_probs,
        losses,
        costs,
        torch.tensor([0.50, 0.25]),
        torch.tensor([1.0, 1.0]),
        global_margin_scale=1.0,
        reference_probs=None,
        kl_coefficient=0.0,
        entropy_coefficient_value=0.0,
    )
    torch.testing.assert_close(
        details["expected_cost_per_sample"], torch.tensor([0.75, 0.00])
    )
    torch.testing.assert_close(details["cost_violation"], torch.tensor(0.0))


def test_trajectory_kl_is_wired_into_gradient() -> None:
    logits = torch.zeros(3, len(ROUTES), requires_grad=True)
    probs = torch.softmax(logits, dim=-1)
    reference = torch.tensor(
        [[0.70, 0.10, 0.05, 0.05, 0.025, 0.025, 0.025, 0.025]]
    ).expand(3, -1)
    loss, details = exact_full_information_objective(
        probs,
        torch.ones_like(probs),
        torch.linspace(0.0, 1.0, len(ROUTES)),
        torch.full((3,), 0.5),
        torch.zeros(3),
        global_margin_scale=1.0,
        reference_probs=reference,
        kl_coefficient=1.0,
        entropy_coefficient_value=0.0,
    )
    assert float(details["kl"].detach()) > 0.0
    loss.backward()
    assert logits.grad is not None
    assert float(logits.grad.abs().sum()) > 0.0


def test_cache_contract_gate_and_negative_controls() -> None:
    rng = np.random.default_rng(11)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        metadata = {
            "split": "train",
            "routes": [list(route) for route in ROUTES],
            "feature_names": [
                "evidence_mean",
                "history_variation",
                "Zr_temporal_slope",
                "Zr_spatial_dispersion",
            ],
            "feature_groups": {"Zr_temporal": [2], "Zr_spatial": [3]},
            "cost_unit": "actual profiler FLOPs",
            "cost_is_proxy": False,
            "contains_target_features": False,
        }
        cache_path = root / "train.npz"
        np.savez(
            cache_path,
            state_features=rng.normal(size=(9, len(PREFIXES), 4)).astype(np.float32),
            route_losses=np.abs(rng.normal(size=(9, len(ROUTES)))).astype(np.float32),
            route_flops=np.linspace(1e6, 8e6, len(ROUTES), dtype=np.float32),
            sample_ids=np.arange(9),
            split=np.asarray("train"),
            metadata_json=np.asarray(json.dumps(metadata)),
        )
        cache = load_route_analysis_cache(cache_path, expected_split="train")
        assert cache.states.shape == (9, len(PREFIXES), 4)
        torch.testing.assert_close(cache.normalized_flops[[0, -1]], torch.tensor([0.0, 1.0]))

        cost_profile_path = root / "cost_profile.json"
        cost_profile_path.write_text(
            json.dumps(
                {
                    "primary_cost": "normalized torch.profiler FLOPs",
                    "routes": {
                        "-".join(map(str, route)): {
                            "flops": float((index + 1) * 1e6)
                        }
                        for index, route in enumerate(ROUTES)
                    },
                }
            ),
            encoding="utf-8",
        )
        external_flops = load_actual_route_flops(cost_profile_path)
        native_cache_path = root / "native_valid.npz"
        native_metadata = {
            "split": "VALID",
            "routes": [list(route) for route in ROUTES],
            "route_feature_shape": [5, len(PREFIXES), 61],
            "uses_target_in_features": False,
        }
        np.savez(
            native_cache_path,
            route_features=rng.normal(size=(5, len(PREFIXES), 61)).astype(np.float32),
            decision_features=rng.normal(size=(5, 3, 61)).astype(np.float32),
            mae=np.abs(rng.normal(size=(5, len(ROUTES)))).astype(np.float32),
            indices=np.arange(5),
            metadata_json=np.asarray(json.dumps(native_metadata)),
        )
        native_cache = load_route_analysis_cache(
            native_cache_path,
            expected_split="valid",
            route_flops=external_flops,
        )
        assert native_cache.states.shape == (5, len(PREFIXES), 61)
        torch.testing.assert_close(native_cache.route_flops, external_flops)

        states = cache.states
        shuffled = negative_control_states(
            states,
            "shuffle",
            generator=torch.Generator().manual_seed(3),
        )
        assert not torch.equal(shuffled, states)
        no_zr = negative_control_states(
            states, "no_zr", zr_feature_indices=[2, 3]
        )
        assert bool((no_zr[:, :, 2:4] == 0).all())
        torch.testing.assert_close(no_zr[:, :, :2], states[:, :, :2])

        failed_gate = root / "failed_gate.json"
        failed_gate.write_text(
            json.dumps(
                {
                    "passed": False,
                    "oracle_gate": {"passed": True},
                    "observability_gate": {"passed": False},
                    "test_used": False,
                }
            ),
            encoding="utf-8",
        )
        try:
            require_passing_gate(failed_gate)
        except RuntimeError as error:
            assert "NOT TRAINED" in str(error)
        else:
            raise AssertionError("failed gate was incorrectly accepted")

        native_gate = root / "native_gate.json"
        native_gate.write_text(
            json.dumps(
                {
                    "gate": {
                        "gates": {
                            "VALID_oracle_headroom_at_least_0.03": True,
                            "probe_recovers_at_least_25pct": True,
                            "probe_discrimination_nontrivial": True,
                            "proceed_to_policy_learning": True,
                        }
                    },
                    "test": None,
                    "policy_trained": False,
                }
            ),
            encoding="utf-8",
        )
        assert require_passing_gate(native_gate)["policy_trained"] is False


def test_end_to_end_policy_training_smoke() -> None:
    rng = np.random.default_rng(37)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        route_flops = np.asarray(
            [1.0e6, 2.0e6, 4.0e6, 4.5e6, 3.0e6, 5.5e6, 3.5e6, 5.0e6],
            dtype=np.float32,
        )

        def write_cache(split: str, samples: int) -> Path:
            states = rng.normal(size=(samples, len(PREFIXES), 6)).astype(np.float32)
            signal = states[:, 0, 0:1]
            route_bias = np.linspace(0.0, 0.14, len(ROUTES), dtype=np.float32)[None]
            losses = (
                1.0
                + route_bias
                + 0.04 * signal * np.asarray(
                    [[-1.0, 0.8, -0.5, 0.6, -0.2, 0.4, 0.2, -0.8]],
                    dtype=np.float32,
                )
            )
            metadata = {
                "split": split,
                "routes": [list(route) for route in ROUTES],
                "feature_names": [
                    "evidence_mean",
                    "history_variation",
                    "Zr_slope",
                    "Zr_spatial",
                    "current_detail",
                    "branch_scale",
                ],
                "feature_groups": {"Zr_temporal": [2], "Zr_spatial": [3]},
                "cost_unit": "actual profiler FLOPs",
                "cost_is_proxy": False,
                "contains_target_features": False,
                "forecaster_checkpoint": "synthetic-frozen-route-complete",
            }
            path = root / f"{split}.npz"
            np.savez(
                path,
                state_features=states,
                route_losses=losses.astype(np.float32),
                route_flops=route_flops,
                sample_ids=np.arange(samples),
                split=np.asarray(split),
                metadata_json=np.asarray(json.dumps(metadata)),
            )
            return path

        train_cache = write_cache("train", 12)
        valid_cache = write_cache("valid", 7)
        gate_path = root / "gate.json"
        gate_path.write_text(
            json.dumps(
                {
                    "passed": True,
                    "oracle_gate": {"passed": True},
                    "observability_gate": {"passed": True},
                    "test_used": False,
                }
            ),
            encoding="utf-8",
        )
        output_dir = root / "output"
        report = train(
            Namespace(
                gate_report=str(gate_path),
                train_cache=str(train_cache),
                valid_cache=str(valid_cache),
                output_dir=str(output_dir),
                budgets="0.25,0.65",
                gpu=-1,
                hidden_dim=12,
                dropout=0.0,
                learning_rate=1e-3,
                weight_decay=0.0,
                dual_learning_rate=0.2,
                max_lambda=10.0,
                epochs=1,
                batch_size=16,
                eval_batch_size=16,
                entropy_coefficient=0.01,
                entropy_anneal_fraction=0.25,
                kl_coefficient=0.02,
                gradient_clip=5.0,
                selection_violation_penalty=10.0,
                reference_update_epochs=1,
                supervised_epochs=1,
                seed=5,
            )
        )
        assert report["selection"]["split"] == "VALID"
        assert report["selection"]["TEST_loaded"] is False
        assert report["objective"]["exact_expected_trajectory_utility"] is True
        assert report["objective"]["group_std_division"] is False
        assert report["negative_controls"]["no_Zr"]
        assert (output_dir / "best_valid_policy.pt").is_file()
        assert (output_dir / "final_valid_report.json").is_file()


def main() -> None:
    tests = [
        test_dag_and_hard_masks,
        test_route_probability_factorization_and_normalization,
        test_global_scaling_preserves_route_margins,
        test_dual_update_and_expected_budget_accounting,
        test_trajectory_kl_is_wired_into_gradient,
        test_cache_contract_gate_and_negative_controls,
        test_end_to_end_policy_training_smoke,
    ]
    for test in tests:
        test()
        print(f"[PASS] {test.__name__}")
    print({"tests": len(tests), "routes": len(ROUTES), "passed": True})


if __name__ == "__main__":
    main()
