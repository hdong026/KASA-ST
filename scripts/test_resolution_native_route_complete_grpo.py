#!/usr/bin/env python3
"""Structural test for the complete-DAG GRPO environment and objective."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basicts.archs.arch_zoo.F2FCoTResolutionNative_arch.f2f_cot_resolution_native_v1_route_complete import F2FCoTResolutionNativeV1RouteCompleteNet, ROUTES
from basicts.archs.arch_zoo.F2FCoTResolutionNative_arch.full_dag_constrained_grpo import FullDAGBudgetRouter, FrozenCompleteDAGEnvironment, LEGAL_NEXT, leave_one_out_advantages
from scripts.f2f_cot_resolution_native_v1_experiment import model_args
from scripts.f2f_cot_runtime import make_loader, select_batch


CHECKPOINT = ROOT / "checkpoints/PEMS04/H12/f2f_cot_resolution_native_route_complete_continuation/continuation_c60_seed1/route_complete_continuation_best_valid.pt"


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    feature_dim = 61
    policy = FullDAGBudgetRouter(feature_dim).to(device)
    costs = torch.tensor([0.0, .2846756, .7413687, .9138184, .3708572, 1.0, .4570388, .6294021], device=device)
    features = torch.randn(4, feature_dim, device=device)
    for prefix in LEGAL_NEXT:
        _, logp, probs, mask = policy.logits_and_probs(features, prefix, torch.full((4,), .75, device=device), costs)
        assert torch.isfinite(logp[mask]).all() and torch.allclose(probs.sum(1), torch.ones(4, device=device))
        if bool((~mask).any()):
            assert float(probs[~mask].abs().max()) == 0.0
    returns = torch.randn(4, 6, device=device)
    advantages = leave_one_out_advantages(returns)
    assert torch.allclose(advantages.sum(1), torch.zeros(4, device=device), atol=1e-6)

    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    forecaster = F2FCoTResolutionNativeV1RouteCompleteNet(**checkpoint["model_args"]).to(device)
    forecaster.load_state_dict(checkpoint["model_state_dict"], strict=True)
    environment = FrozenCompleteDAGEnvironment(forecaster)
    history, _, _ = select_batch(next(iter(make_loader("valid", 2, False, 0))), device)
    observation = environment.begin(history)
    generator = torch.Generator(device=device).manual_seed(17)
    trajectories, diagnostics = environment.sample(observation, torch.full((history.shape[0],), .75, device=device), costs, policy, 3, .1, generator)
    assert trajectories.predictions.shape[:2] == (history.shape[0] * 3, 12)
    assert bool(((trajectories.route_ids >= 0) & (trajectories.route_ids < len(ROUTES))).all())
    assert diagnostics["history_encode_count"] == 1.0
    assert diagnostics["action_count"] >= float(history.shape[0] * 3)
    print({"routes": len(ROUTES), "prefixes": len(LEGAL_NEXT), "feature_dim": feature_dim, "sampled_trajectories": int(trajectories.route_ids.numel()), "history_encode_count": diagnostics["history_encode_count"], "passed": True})


if __name__ == "__main__":
    main()
