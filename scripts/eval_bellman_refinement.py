#!/usr/bin/env python3
"""Evaluate Budgeted Bellman router with sequential quarter-prefix reuse."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basicts.archs.arch_zoo.ChainForecasting_arch.bellman_refinement_qnet import Q0Net, Q1Net
from basicts.archs.arch_zoo.ChainForecasting_arch.budgeted_bellman_refinement import (
    BudgetedRefinementMDP,
    greedy_masked_argmax,
    route_name_from_actions,
    semantic_to_route,
)
from basicts.archs.arch_zoo.ChainForecasting_arch.plan_b_v2_state_cache import load_supernet_strict
from basicts.archs.arch_zoo.ChainForecasting_arch.sequential_f2f_environment import (
    SequentialF2FEnvironment,
)
from basicts.data.forecast_refinement_gain_dataset import ForecastRefinementGainDataset
from basicts.data.indexed_timeseries_dataset import IndexedTimeSeriesForecastingDataset

DEFAULT_SUPERNET = (
    "checkpoints/PEMS04/H12/budget_f2f/"
    "supernet_eta0p50_dynamic_fair_rawscale_loss_v2_60f53aa1c6/seed1/"
    "b5678fda5e8d94ed028c6c8bb073461d/BudgetConditionedAdaptiveF2FNet_best_val_MAE.pt"
)
DEFAULT_VALID_ORACLE = "results/pems04_budget_f2f_oracle_valid_rawscale.json"
DATA_FILE = "datasets/PEMS04/data_in12_out12.pkl"
INDEX_FILE = "datasets/PEMS04/index_in12_out12.pkl"
FORWARD_FEATURES = [0, 1, 2, 3]
TARGET_FEATURES = [0]


class BellmanEvalNet(nn.Module):
    def __init__(self, supernet, q0, q1, mdp: BudgetedRefinementMDP, *, eta: float = 1.0, c_max: float = 1.0):
        super().__init__()
        self.supernet = supernet
        self.q0 = q0
        self.q1 = q1
        self.mdp = mdp
        self.eta = float(eta)
        self.c_max = float(c_max)
        self.env = SequentialF2FEnvironment(supernet)
        self.candidate_routes = list(supernet.candidate_routes)
        self.route_costs = supernet.route_costs
        self.output_len = int(supernet.output_len)
        self.node_size = int(supernet.node_size)
        self.output_dim = int(supernet.output_dim)
        self.quarter_prefix_calls = 0
        self.budget_violations = 0
        self.route_hist = Counter()
        self._h4_instrumented = False

    def set_eta(self, eta: float) -> None:
        self.eta = float(eta)

    def instrument_h4(self) -> None:
        if self._h4_instrumented:
            return
        h4 = self.output_len // 4
        step = self.supernet.backbone.temporal_steps[self.supernet.res_to_index[h4]]
        orig = step.forward
        runner = self

        def wrapped(*args, **kwargs):
            runner.quarter_prefix_calls += 1
            return orig(*args, **kwargs)

        step.forward = wrapped
        self._h4_orig = orig
        self._h4_step = step
        self._h4_instrumented = True

    def restore_h4(self) -> None:
        if self._h4_instrumented:
            self._h4_step.forward = self._h4_orig
            self._h4_instrumented = False

    def forward(
        self,
        history_data=None,
        future_data=None,
        batch_seen=None,
        epoch=None,
        train=False,
        return_all=False,
        history=None,
        **kwargs,
    ):
        if history_data is None:
            history_data = history
        history = history_data
        self.q0.eval()
        self.q1.eval()
        B = history.shape[0]
        device = history.device
        budget = self.mdp.budget(self.eta)
        s0_mask_dict = self.mdp.s0_mask(budget)
        s0_mask = torch.tensor(
            [[s0_mask_dict["f"], s0_mask_dict["m"], s0_mask_dict["q"]]] * B,
            device=device,
            dtype=torch.bool,
        )
        bnorm = torch.full((B, 1), budget / self.c_max, device=device, dtype=history.dtype)
        with torch.no_grad():
            q0v = self.q0(history, bnorm, s0_mask)
            a0 = greedy_masked_argmax(q0v, s0_mask)  # 0=f,1=m,2=q

        preds = []
        routes_exec = []
        route_ids = []
        chain_res_list = []
        for i in range(B):
            ai = int(a0[i].item())
            if ai == 0:
                name = "D"
                route = semantic_to_route(name, self.output_len)
                if not self.mdp.terminal_route_feasible(name, budget):
                    self.budget_violations += 1
                out = self.supernet._execute_route(history[i : i + 1], route)
            elif ai == 1:
                name = "M"
                route = semantic_to_route(name, self.output_len)
                if not self.mdp.terminal_route_feasible(name, budget):
                    self.budget_violations += 1
                out = self.supernet._execute_route(history[i : i + 1], route)
            else:
                pref = self.env.execute_quarter_prefix(history[i : i + 1])
                z_q = pref["Z_q"]
                rem = budget - self.mdp.costs.c_q
                sq_mask_dict = self.mdp.sq_mask(rem)
                sq_mask = torch.tensor(
                    [[sq_mask_dict["f"], sq_mask_dict["m"]]],
                    device=device,
                    dtype=torch.bool,
                )
                b1 = torch.tensor([[rem / self.c_max]], device=device, dtype=history.dtype)
                q1v = self.q1(history[i : i + 1], z_q, b1, sq_mask)
                a1 = int(greedy_masked_argmax(q1v, sq_mask).item())
                if a1 == 0:
                    name = "Q"
                    out = self.env.resume_quarter_to_final(history[i : i + 1], pref["prev_forecast"])
                else:
                    name = "F"
                    out = self.env.resume_quarter_to_progressive(
                        history[i : i + 1], pref["prev_forecast"]
                    )
                if not self.mdp.terminal_route_feasible(name, budget):
                    self.budget_violations += 1
                route = semantic_to_route(name, self.output_len)
            preds.append(out["pred"])
            routes_exec.append(route)
            self.route_hist[str(route)] += 1
            # map to candidate index
            rid = next(
                (j for j, r in enumerate(self.candidate_routes) if list(r) == list(route)),
                0,
            )
            route_ids.append(rid)
            chain_res_list.append(list(route))

        pred = torch.cat(preds, dim=0)
        if not return_all:
            return pred
        costs = []
        for route in routes_exec:
            if route == semantic_to_route("D", self.output_len):
                costs.append(self.mdp.costs.route_costs["D"])
            elif route == semantic_to_route("M", self.output_len):
                costs.append(self.mdp.costs.route_costs["M"])
            elif route == semantic_to_route("Q", self.output_len):
                costs.append(self.mdp.costs.route_costs["Q"])
            else:
                costs.append(self.mdp.costs.route_costs["F"])
        rid_t = torch.tensor(route_ids, device=device, dtype=torch.long)
        cost_t = torch.tensor(costs, device=device, dtype=history.dtype)
        bud_t = torch.full((B,), budget, device=device, dtype=history.dtype)
        return {
            "prediction": pred,
            "pred": pred,
            "routes": routes_exec,
            "selected_route_id": rid_t,
            "executed_route_id": rid_t,
            "selected_cost": cost_t,
            "budget": bud_t,
            "chain_resolutions": chain_res_list[0] if chain_res_list else [self.output_len],
        }


def load_router(ckpt: Path, device):
    blob = torch.load(ckpt, map_location="cpu")
    q0 = Q0Net()
    q1 = Q1Net()
    if "q0" in blob:
        q0.load_state_dict(blob["q0"])
        q1.load_state_dict(blob["q1"])
    else:
        raise KeyError("router checkpoint missing q0/q1")
    q0.to(device).eval()
    q1.to(device).eval()
    scale = float(blob.get("scale", 1.0))
    c_max = float(blob.get("c_max", 1.0))
    return q0, q1, scale, c_max


def evaluate_split(
    *,
    split: str,
    supernet,
    q0,
    q1,
    mdp,
    c_max,
    etas,
    device,
    max_samples: int | None = None,
    batch_size: int = 8,
    compute_oracle_regret: bool = False,
    valid_oracle: str | None = None,
) -> dict[str, Any]:
    import pickle

    from basicts.archs.arch_zoo.ChainForecasting_arch.budget_route_utils import (
        default_candidate_routes,
    )
    from scripts.eval_budget_conditioned_f2f_intensity import evaluate_loader

    ds = IndexedTimeSeriesForecastingDataset(DATA_FILE, INDEX_FILE, split)
    n = len(ds) if max_samples is None else min(len(ds), max_samples)

    class Sub(torch.utils.data.Dataset):
        def __init__(self, base, n):
            self.base = base
            self.n = n

        def __len__(self):
            return self.n

        def __getitem__(self, i):
            fut, hist, _si = self.base[i]
            return fut, hist

    sub = Sub(ds, n)
    loader = DataLoader(sub, batch_size=batch_size, shuffle=False, num_workers=0)
    with open("datasets/PEMS04/scaler_in12_out12.pkl", "rb") as f:
        scaler = pickle.load(f)
    candidates = default_candidate_routes(12)

    oracle_map = None
    if compute_oracle_regret and valid_oracle:
        raw = json.loads(Path(valid_oracle).read_text())
        oracle_map = {}
        for r in raw["records"]:
            si = int(r["sample_index"])
            if si in oracle_map:
                continue
            losses = [float(x["final_mae"]) for x in r["route_final_losses"]]
            oracle_map[si] = torch.tensor(losses, dtype=torch.float32)

    out = {"split": split, "n_samples": n, "etas": {}}
    net = BellmanEvalNet(supernet, q0, q1, mdp, c_max=c_max)
    net.to(device)

    for eta in etas:
        net.set_eta(eta)
        net.route_hist = Counter()
        net.budget_violations = 0
        net.quarter_prefix_calls = 0
        net.instrument_h4()
        t0 = time.time()
        metrics = evaluate_loader(
            net,
            loader,
            device=device,
            forward_features=FORWARD_FEATURES,
            target_features=TARGET_FEATURES,
            scaler=scaler,
            null_val=0.0,
            candidates=candidates,
            max_batches=None,
        )
        net.restore_h4()
        wall = time.time() - t0
        hist = {k: int(v) for k, v in net.route_hist.items()}
        entry = {
            "MAE": metrics.get("mae"),
            "RMSE": metrics.get("rmse"),
            "MAPE": metrics.get("mape"),
            "route_histogram": hist,
            "avg_proxy_cost": metrics.get("average_selected_cost"),
            "avg_stages": metrics.get("average_stage_count"),
            "budget_violations": net.budget_violations,
            "quarter_prefix_calls": net.quarter_prefix_calls,
            "wall_time_sec": wall,
        }

        if compute_oracle_regret and oracle_map is not None:
            regs = []
            net.set_eta(eta)
            for i in range(n):
                fut, hist_x, si = ds[i]
                if int(si) not in oracle_map:
                    continue
                L = oracle_map[int(si)]
                x = hist_x.unsqueeze(0).to(device)
                budget = mdp.budget(eta)
                s0_mask_dict = mdp.s0_mask(budget)
                s0_mask = torch.tensor(
                    [[s0_mask_dict["f"], s0_mask_dict["m"], s0_mask_dict["q"]]],
                    device=device,
                    dtype=torch.bool,
                )
                bnorm = torch.tensor([[budget / c_max]], device=device)
                with torch.no_grad():
                    q0v = q0(x, bnorm, s0_mask)
                    a0 = int(greedy_masked_argmax(q0v, s0_mask).item())
                    if a0 == 0:
                        name = "D"
                    elif a0 == 1:
                        name = "M"
                    else:
                        pref = net.env.execute_quarter_prefix(x)
                        rem = budget - mdp.costs.c_q
                        sq = mdp.sq_mask(rem)
                        sq_mask = torch.tensor(
                            [[sq["f"], sq["m"]]], device=device, dtype=torch.bool
                        )
                        q1v = q1(
                            x,
                            pref["Z_q"],
                            torch.tensor([[rem / c_max]], device=device),
                            sq_mask,
                        )
                        a1 = int(greedy_masked_argmax(q1v, sq_mask).item())
                        name = "Q" if a1 == 0 else "F"
                idx = {"D": 0, "M": 1, "Q": 2, "F": 3}[name]
                feas = mdp.feasible_terminal_routes(budget)
                best = min(float(L[{"D": 0, "M": 1, "Q": 2, "F": 3}[nm]]) for nm in feas)
                regs.append(float(L[idx]) - best)
            entry["strict_oracle_regret"] = float(sum(regs) / max(len(regs), 1))
            entry["n_regret"] = len(regs)

        out["etas"][str(eta)] = entry
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--router", default="checkpoints/PEMS04/H12/budget_f2f/plan_b_bellman/router_best.pt")
    p.add_argument("--supernet", default=DEFAULT_SUPERNET)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--split", choices=["valid", "test", "both"], default="both")
    p.add_argument("--etas", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0])
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--valid-oracle", default=DEFAULT_VALID_ORACLE)
    p.add_argument("--out-valid", default="results/planB_bellman_valid_eval.json")
    p.add_argument("--out-test", default="results/planB_bellman_test_eval.json")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    supernet, _ = load_supernet_strict(args.supernet, device)
    q0, q1, scale, c_max = load_router(Path(args.router), device)
    mdp = BudgetedRefinementMDP(12)
    if abs(c_max - mdp.costs.C_max) > 1e-6:
        c_max = mdp.costs.C_max

    if args.split in ("valid", "both"):
        valid = evaluate_split(
            split="valid",
            supernet=supernet,
            q0=q0,
            q1=q1,
            mdp=mdp,
            c_max=c_max,
            etas=args.etas,
            device=device,
            max_samples=args.max_samples,
            batch_size=args.batch_size,
            compute_oracle_regret=True,
            valid_oracle=args.valid_oracle,
        )
        Path(args.out_valid).write_text(json.dumps(valid, indent=2))
        print("wrote", args.out_valid)

    if args.split in ("test", "both"):
        # NEVER load TEST oracle
        test = evaluate_split(
            split="test",
            supernet=supernet,
            q0=q0,
            q1=q1,
            mdp=mdp,
            c_max=c_max,
            etas=args.etas,
            device=device,
            max_samples=args.max_samples,
            batch_size=args.batch_size,
            compute_oracle_regret=False,
            valid_oracle=None,
        )
        Path(args.out_test).write_text(json.dumps(test, indent=2))
        print("wrote", args.out_test)


if __name__ == "__main__":
    main()
