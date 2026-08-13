#!/usr/bin/env python3
"""Evaluate Plan B-v2 policy with sequential quarter-prefix reuse (H/4 once)."""

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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basicts.archs.arch_zoo.ChainForecasting_arch.budget_route_utils import (
    budget_from_intensity,
    default_candidate_routes,
    load_route_costs,
)
from basicts.archs.arch_zoo.ChainForecasting_arch.exact_trajectory_policy_objective import (
    unique_nontrivial_feasibility_regimes,
)
from basicts.archs.arch_zoo.ChainForecasting_arch.forecast_refinement_routes import (
    build_refinement_route_index_map,
    standard_refinement_route_template,
)
from basicts.archs.arch_zoo.ChainForecasting_arch.group_relative_refinement_policy_v2 import (
    GroupRelativeRefinementPolicyV2,
)
from basicts.archs.arch_zoo.ChainForecasting_arch.plan_b_v2_state_cache import (
    load_supernet_strict,
)
from basicts.archs.arch_zoo.ChainForecasting_arch.route_quality_decision import (
    feasible_mask_from_budget,
)
from basicts.archs.arch_zoo.ChainForecasting_arch.sequential_f2f_environment import (
    A0_QUARTER,
    SequentialF2FEnvironment,
)
from basicts.data.forecast_refinement_gain_dataset import (
    ForecastRefinementGainDataset,
    collate_refinement_gains,
)
from basicts.data.indexed_timeseries_dataset import IndexedTimeSeriesForecastingDataset
from torch.utils.data import DataLoader

DEFAULT_SUPERNET = (
    "checkpoints/PEMS04/H12/budget_f2f/"
    "supernet_eta0p50_dynamic_fair_rawscale_loss_v2_60f53aa1c6/seed1/"
    "b5678fda5e8d94ed028c6c8bb073461d/BudgetConditionedAdaptiveF2FNet_best_val_MAE.pt"
)


class PlanBV2EvalNet(nn.Module):
    """Stable supernet + V2 policy with H/4 executed at most once per sample."""

    def __init__(self, supernet, policy, env, *, eta: float = 1.0):
        super().__init__()
        self.supernet = supernet
        self.policy = policy
        self.env = env
        self.eta = float(eta)
        self.candidate_routes = list(supernet.candidate_routes)
        self.route_costs = supernet.route_costs
        self.output_len = int(supernet.output_len)
        self.node_size = int(supernet.node_size)
        self.output_dim = int(supernet.output_dim)
        self.index_map = build_refinement_route_index_map(
            self.candidate_routes, self.output_len
        )
        self.h4_call_count = 0
        self._instrumented = False

    def set_eta(self, eta: float) -> None:
        self.eta = float(eta)

    def instrument_h4(self) -> None:
        if self._instrumented:
            return
        h4 = self.output_len // 4
        step = self.supernet.backbone.temporal_steps[self.supernet.res_to_index[h4]]
        orig = step.forward
        runner = self

        def wrapped(*args, **kwargs):
            runner.h4_call_count += 1
            return orig(*args, **kwargs)

        step.forward = wrapped
        self._h4_orig = orig
        self._h4_step = step
        self._instrumented = True

    def restore_h4(self) -> None:
        if self._instrumented:
            self._h4_step.forward = self._h4_orig
            self._instrumented = False

    @torch.no_grad()
    def forward(
        self,
        history_data: torch.Tensor,
        future_data=None,
        batch_seen=None,
        epoch=None,
        train: bool = False,
        return_all: bool = True,
        **kwargs,
    ) -> dict[str, Any]:
        del future_data, batch_seen, epoch, train, kwargs
        history = history_data
        b = history.shape[0]
        device = history.device
        costs = self.supernet.route_costs.detach().float()
        bval = budget_from_intensity(self.eta, costs.cpu().tolist())
        feas = feasible_mask_from_budget(costs.cpu(), torch.tensor([bval])).squeeze(0).to(device)

        # Pre-route context once
        h_shared = self.supernet.extract_pre_route_context(history, detach=True)
        s0 = self.policy.encode_state0(h_shared)
        hidden = s0["state0_hidden"]
        from basicts.archs.arch_zoo.ChainForecasting_arch.exact_trajectory_policy_objective import (
            action_masks_from_feasible,
        )

        masks = action_masks_from_feasible(feas, index_map=self.index_map)
        m0 = masks["mask0"].to(device).unsqueeze(0).expand(b, -1)
        m1 = masks["mask1"].to(device).unsqueeze(0).expand(b, -1)
        logits0 = self.policy.logits0(hidden)
        a0 = logits0.masked_fill(~m0, -1e9).argmax(dim=-1)

        route_ids = torch.empty(b, dtype=torch.long, device=device)
        preds = torch.zeros(
            b, self.output_len, self.node_size, self.output_dim, device=device, dtype=history.dtype
        )
        selected_cost = torch.zeros(b, device=device, dtype=costs.dtype)

        # Partition by a0
        for i in range(b):
            ai = int(a0[i].item())
            if ai != A0_QUARTER:
                route = self.env.route_from_actions(ai, None)
                key = {
                    tuple(self.env.template["direct"]): "direct",
                    tuple(self.env.template["half"]): "half",
                    tuple(self.env.template["quarter"]): "quarter",
                    tuple(self.env.template["progressive"]): "progressive",
                }[tuple(route)]
                rid = int(self.index_map[key])
                out = self.supernet._execute_route(history[i : i + 1], route)
                preds[i] = out["pred"][0]
                route_ids[i] = rid
                selected_cost[i] = costs[rid]
            else:
                # Execute H/4 ONCE, then policy1, then resume
                pref = self.env.execute_quarter_prefix(history[i : i + 1])
                zq = pref["Z_q"]
                zenc = self.policy.encode_zq(zq)
                logits1 = self.policy.logits1(hidden[i : i + 1], zenc["zq_hidden"])
                a1 = int(logits1.masked_fill(~m1[i : i + 1], -1e9).argmax(dim=-1).item())
                route = self.env.route_from_actions(A0_QUARTER, a1)
                key = {
                    tuple(self.env.template["quarter"]): "quarter",
                    tuple(self.env.template["progressive"]): "progressive",
                }[tuple(route)]
                rid = int(self.index_map[key])
                if a1 == 0:  # JUMP_FINAL
                    resume = self.env.resume_quarter_to_final(
                        history[i : i + 1], pref["prev_forecast"]
                    )
                else:
                    resume = self.env.resume_quarter_to_progressive(
                        history[i : i + 1], pref["prev_forecast"]
                    )
                preds[i] = resume["pred"][0]
                route_ids[i] = rid
                selected_cost[i] = costs[rid]

        out = {
            "pred": preds,
            "selected_route_id": route_ids,
            "executed_route_id": route_ids,
            "selected_cost": selected_cost,
            "budget": history.new_full((b,), bval, dtype=selected_cost.dtype),
            "chain_resolutions": [self.output_len],
            "batch_route_id": int(
                torch.bincount(route_ids.cpu(), minlength=len(self.candidate_routes)).argmax().item()
            ),
        }
        return out if return_all else out["pred"]


def load_policy_v2(ckpt_path: Path, h_dim: int, z_channels: int, device):
    blob = torch.load(ckpt_path, map_location="cpu")
    policy = GroupRelativeRefinementPolicyV2(h_dim=h_dim, z_channels=z_channels).to(device)
    policy.load_state_dict(blob["policy_state_dict"])
    policy.eval()
    return policy, blob


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy-checkpoint", required=True)
    p.add_argument("--supernet-checkpoint", default=DEFAULT_SUPERNET)
    p.add_argument("--cfg", default=None)
    p.add_argument("--valid-oracle", default="results/pems04_budget_f2f_oracle_valid_rawscale.json")
    p.add_argument("--scaler", default="datasets/PEMS04/scaler_in12_out12.pkl")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--etas", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0])
    p.add_argument("--split", default="valid", choices=["valid", "test", "both"])
    p.add_argument("--max-batches", type=int, default=None)
    p.add_argument("--out", default="results/planB_v2_policy_eval.json")
    args = p.parse_args()

    from basicts.utils import load_pkl
    from scripts.eval_budget_conditioned_f2f_intensity import evaluate_loader
    from scripts.eval_group_relative_refinement_policy import build_loader

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    supernet, load_meta = load_supernet_strict(args.supernet_checkpoint, device)
    env = SequentialF2FEnvironment(supernet)
    with torch.no_grad():
        probe = torch.zeros(1, 12, 307, 4, device=device)
        h = supernet.extract_pre_route_context(probe, detach=True)
        z = env.execute_quarter_prefix(probe)["Z_q"]
        h_dim = int(h.shape[-1])
        z_ch = int(z.shape[-1])
    policy, blob = load_policy_v2(Path(args.policy_checkpoint), h_dim, z_ch, device)
    runner = PlanBV2EvalNet(supernet, policy, env).to(device)
    runner.eval()

    routes = default_candidate_routes(12)
    costs = load_route_costs(None, routes, 12)
    data_file = "datasets/PEMS04/data_in12_out12.pkl"
    index_file = "datasets/PEMS04/index_in12_out12.pkl"
    scaler_path = Path(args.scaler)
    if not scaler_path.is_file():
        raise FileNotFoundError(f"scaler not found: {scaler_path}")
    scaler = load_pkl(str(scaler_path))

    results: dict[str, Any] = {
        "status": "ok",
        "method": "PlanB-v2 Exact Full-Information Trajectory Policy",
        "not_grpo_gspo": True,
        "selection_rule": "deterministic masked argmax + sequential H/4 resume",
        "policy_checkpoint": str(args.policy_checkpoint),
        "utility_scale": blob.get("utility_scale"),
        "supernet_load": {
            "n_missing": len(load_meta["missing"]),
            "allowed_missing_prefixes": load_meta["allowed_missing_prefixes"],
            "unexpected": load_meta["unexpected"],
            "sha1_16": load_meta["sha1_16"],
        },
        "candidate_routes": routes,
        "etas": list(args.etas),
        "splits": {},
        "notes": [
            "TEST oracle is forbidden",
            "VALID oracle regret is diagnostic only",
            "primary model selection uses nontrivial regimes only",
        ],
    }

    t0 = time.perf_counter()
    splits = ["valid", "test"] if args.split == "both" else [args.split]
    forward_features = [0, 1, 2, 3]
    target_features = [0]
    null_val = 0.0

    for split in splits:
        if split == "test":
            print(
                "[TEST] Final BasicTS forecasting metrics only. "
                "Do not load/build test route oracle for tuning."
            )
        loader, n = build_loader(split, int(args.batch_size))
        print(f"[{split}] n={n} batches≈{(n + args.batch_size - 1) // args.batch_size}")
        eta_rows: dict[str, Any] = {}
        for eta in args.etas:
            runner.set_eta(float(eta))
            print(f"  eta={eta}")
            row = evaluate_loader(
                runner,
                loader,
                device=device,
                forward_features=forward_features,
                target_features=target_features,
                scaler=scaler,
                null_val=null_val,
                candidates=routes,
                max_batches=args.max_batches,
            )
            row["eta"] = float(eta)
            eta_rows[str(eta)] = row
        results["splits"][split] = {"n_samples": int(n), "etas": eta_rows}

    # VALID oracle regret only (never TEST oracle)
    if args.split in {"valid", "both"} and Path(args.valid_oracle).is_file():
        print(f"[valid-oracle-regret] {args.valid_oracle}")
        ds = ForecastRefinementGainDataset(
            IndexedTimeSeriesForecastingDataset(data_file, index_file, "valid"),
            args.valid_oracle,
            expected_routes=routes,
            expected_costs=costs,
            expected_horizon=12,
            expected_dataset="PEMS04",
            require_len_match=False,
        )
        oloader = DataLoader(
            ds, batch_size=int(args.batch_size), shuffle=False, collate_fn=collate_refinement_gains
        )
        oracle_etas: dict[str, Any] = {}
        all_regrets: list[float] = []
        nt_regrets: list[float] = []
        for eta in args.etas:
            runner.set_eta(float(eta))
            regrets, hist, cost_sum, n = [], Counter(), 0.0, 0
            for history, _si, _g, losses in oloader:
                history = history.to(device)
                losses = losses.to(device)
                executed = runner.forward(history)
                rids = executed["selected_route_id"]
                bval = budget_from_intensity(float(eta), costs)
                feas = feasible_mask_from_budget(
                    torch.tensor(costs), torch.tensor([bval])
                ).squeeze(0).to(device)
                for i in range(history.shape[0]):
                    rid = int(rids[i].item())
                    best = losses[i][feas].min()
                    reg = float((losses[i, rid] - best).item())
                    regrets.append(reg)
                    all_regrets.append(reg)
                    if float(eta) >= 0.5 - 1e-9:
                        nt_regrets.append(reg)
                    hist[rid] += 1
                    cost_sum += float(costs[rid])
                    n += 1
            oracle_etas[str(eta)] = {
                "mean_regret": float(sum(regrets) / max(len(regrets), 1)),
                "mean_cost": cost_sum / max(n, 1),
                "route_histogram": {str(k): int(v) for k, v in hist.items()},
            }
        results["valid_oracle_regret"] = {
            "etas": oracle_etas,
            "mean_regret_all_eta": float(sum(all_regrets) / max(len(all_regrets), 1)),
            "mean_regret_nontrivial": float(sum(nt_regrets) / max(len(nt_regrets), 1)),
        }
        results["valid_oracle_path"] = str(args.valid_oracle)

    # Equivalence / H4-once check
    with torch.no_grad():
        hist0 = torch.randn(1, 12, 307, 4, device=device)
        eq = env.sequential_route_equivalence_check(hist0, atol=1e-6)
    results["prefix_resume_equivalence"] = eq
    results["wall_time_sec"] = time.perf_counter() - t0

    Path(args.out).write_text(json.dumps(results, indent=2, default=str))
    print(json.dumps({"wrote": args.out, "wall_time_sec": results["wall_time_sec"]}, indent=2))

    # Concise console summary
    for split in splits:
        print(f"\n=== {split.upper()} BasicTS ===")
        for eta, row in results["splits"][split]["etas"].items():
            hist = row.get("route_histogram") or row.get("selected_route_histogram") or {}
            print(
                f"eta={eta}: MAE={row.get('MAE', row.get('mae'))} "
                f"RMSE={row.get('RMSE', row.get('rmse'))} "
                f"MAPE={row.get('MAPE', row.get('mape'))} "
                f"cost={row.get('mean_selected_cost', row.get('avg_cost'))} "
                f"hist={hist}"
            )
    if "valid_oracle_regret" in results:
        vor = results["valid_oracle_regret"]
        print(
            f"\nVALID oracle regret: all_eta={vor['mean_regret_all_eta']:.6f} "
            f"nontrivial={vor['mean_regret_nontrivial']:.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
