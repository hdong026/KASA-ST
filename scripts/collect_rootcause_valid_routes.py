#!/usr/bin/env python3
"""Collect VALID per-sample routes for PlanA / B-v2 / Bellman (inference-only)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from basicts.archs.arch_zoo.ChainForecasting_arch.budget_route_utils import default_candidate_routes
from basicts.archs.arch_zoo.ChainForecasting_arch.budgeted_bellman_refinement import (
    BudgetedRefinementMDP,
)
from basicts.archs.arch_zoo.ChainForecasting_arch.group_relative_refinement_policy_v2 import (
    GroupRelativeRefinementPolicyV2,
)
from basicts.archs.arch_zoo.ChainForecasting_arch.plan_b_v2_state_cache import load_supernet_strict
from basicts.archs.arch_zoo.ChainForecasting_arch.sequential_f2f_environment import (
    SequentialF2FEnvironment,
)
from basicts.data.indexed_timeseries_dataset import IndexedTimeSeriesForecastingDataset

DATA = "datasets/PEMS04/data_in12_out12.pkl"
INDEX = "datasets/PEMS04/index_in12_out12.pkl"
STABLE = (
    "checkpoints/PEMS04/H12/budget_f2f/"
    "supernet_eta0p50_dynamic_fair_rawscale_loss_v2_60f53aa1c6/seed1/"
    "b5678fda5e8d94ed028c6c8bb073461d/BudgetConditionedAdaptiveF2FNet_best_val_MAE.pt"
)
ETAS = [0.5, 0.75, 1.0]


def collate(batch):
    futs, hists, sis = zip(*batch)
    return torch.stack(hists, 0), list(sis)


@torch.no_grad()
def collect_plana(device, loader):
    from scripts.train_forecast_refinement_controller import _build_model, _load_supernet

    class _Args:
        horizon = 12
        controller_dim = 128
        pooling_queries = 4
        delta_abs = 0.05
        route_cost_file = None
        cfg = (
            "checkpoints/PEMS04/H12/budget_f2f/"
            "supernet_eta0p50_dynamic_fair_rawscale_loss_v2_60f53aa1c6/seed1/"
            "b5678fda5e8d94ed028c6c8bb073461d/"
            "H12_supernet_eta0p50_dynamic_fair_rawscale_loss_v2_60f53aa1c6_seed1.py"
        )

    routes = default_candidate_routes(12)
    model = _build_model(_Args(), routes, device)
    _load_supernet(model, Path(STABLE))
    blob = torch.load(
        "checkpoints/PEMS04/H12/budget_f2f/crossfit_refinement_controller/"
        "refinement_controller_best_val_regret.pt",
        map_location="cpu",
    )
    state = blob.get("controller_state_dict") or blob.get("model_state_dict") or blob
    if any(str(k).startswith("gain_controller.") for k in state):
        model.load_state_dict(state, strict=False)
    else:
        model.gain_controller.load_state_dict(state, strict=False)
    model.eval()
    out = {}
    for eta in ETAS:
        sels = []
        for hist, _sis in loader:
            hist = hist.to(device)
            pred = model(
                hist,
                None,
                train=False,
                return_all=True,
                inference_intensity_override=float(eta),
            )
            rid = pred.get("executed_route_id", pred["selected_route_id"])
            sels.extend(int(x) for x in rid.detach().cpu().view(-1).tolist())
        out[str(eta)] = np.asarray(sels, dtype=np.int64)
        print("PlanA", eta, "hist", dict(zip(*np.unique(out[str(eta)], return_counts=True))))
    return out


@torch.no_grad()
def collect_bv2(device, loader, supernet):
    from scripts.eval_plan_b_v2 import PlanBV2EvalNet, load_policy_v2

    env = SequentialF2FEnvironment(supernet)
    probe = torch.zeros(1, 12, 307, 4, device=device)
    h = supernet.extract_pre_route_context(probe, detach=True)
    z = env.execute_quarter_prefix(probe)["Z_q"]
    policy, _ = load_policy_v2(
        Path("checkpoints/PEMS04/H12/budget_f2f/plan_b_v2_exact_policy.pt"),
        int(h.shape[-1]),
        int(z.shape[-1]),
        device,
    )
    runner = PlanBV2EvalNet(supernet, policy, env).to(device)
    out = {}
    for eta in ETAS:
        sels = []
        runner.set_eta(eta)
        for hist, _ in loader:
            o = runner(history_data=hist.to(device), return_all=True)
            sels.extend(int(x) for x in o["executed_route_id"].cpu().view(-1).tolist())
        out[str(eta)] = np.asarray(sels, dtype=np.int64)
        print("Bv2", eta, "hist", dict(zip(*np.unique(out[str(eta)], return_counts=True))))
    return out


@torch.no_grad()
def collect_bellman(device, loader, supernet):
    from scripts.eval_bellman_refinement import BellmanEvalNet, load_router

    q0, q1, scale, c_max = load_router(
        Path("checkpoints/PEMS04/H12/budget_f2f/plan_b_bellman/seed1/router_best.pt"),
        device,
    )
    mdp = BudgetedRefinementMDP(12)
    net = BellmanEvalNet(supernet, q0, q1, mdp, c_max=mdp.costs.C_max).to(device)
    out = {}
    for eta in ETAS:
        sels = []
        net.set_eta(eta)
        for hist, _ in loader:
            o = net(history_data=hist.to(device), return_all=True)
            sels.extend(int(x) for x in o["executed_route_id"].cpu().view(-1).tolist())
        out[str(eta)] = np.asarray(sels, dtype=np.int64)
        print("Bellman", eta, "hist", dict(zip(*np.unique(out[str(eta)], return_counts=True))))
    return out


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ds = IndexedTimeSeriesForecastingDataset(DATA, INDEX, "valid")
    loader = DataLoader(ds, batch_size=16, shuffle=False, collate_fn=collate)
    print("n_valid", len(ds))
    plana = collect_plana(device, loader)
    supernet, _ = load_supernet_strict(STABLE, device)
    bv2 = collect_bv2(device, loader, supernet)
    bell = collect_bellman(device, loader, supernet)
    save = {}
    for e in map(str, ETAS):
        save[f"plana_{e}"] = plana[e]
        save[f"bv2_{e}"] = bv2[e]
        save[f"bellman_{e}"] = bell[e]
    Path("results").mkdir(exist_ok=True)
    np.savez("results/rootcause_valid_route_cache.npz", **save)
    print("wrote results/rootcause_valid_route_cache.npz")


if __name__ == "__main__":
    main()
