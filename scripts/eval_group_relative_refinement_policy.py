#!/usr/bin/env python3
"""Evaluate Group-Relative Sequential Refinement Policy (Plan B).

Full BasicTS forecasting metrics (MAE/RMSE/MAPE) via the same inverse-transform
+ masked_* path as budget-conditioned F2F eval.

Deterministic masked argmax by default. Official VALID may optionally report
oracle route regret when --valid-oracle is provided. TEST never builds/loads a
test route oracle for tuning.
"""

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

from basicts.archs.arch_zoo.ChainForecasting_arch.adaptive_refinement_context import (
    pool_pre_route_context,
)
from basicts.archs.arch_zoo.ChainForecasting_arch.budget_route_utils import (
    default_candidate_routes,
)
from basicts.archs.arch_zoo.ChainForecasting_arch.group_relative_refinement_policy import (
    GroupRelativeRefinementPolicy,
)
from basicts.archs.arch_zoo.ChainForecasting_arch.sequential_f2f_environment import (
    A0_QUARTER,
    SequentialF2FEnvironment,
)
from basicts.utils import load_pkl
from scripts.eval_budget_conditioned_f2f_intensity import evaluate_loader
from scripts.train_forecast_refinement_controller import _build_model, _load_supernet
from scripts.train_group_relative_refinement_policy import _build_s0

DEFAULT_SUPERNET = (
    "checkpoints/PEMS04/H12/budget_f2f/"
    "supernet_eta0p50_dynamic_fair_rawscale_loss_v2_60f53aa1c6/seed1/"
    "b5678fda5e8d94ed028c6c8bb073461d/BudgetConditionedAdaptiveF2FNet_best_val_MAE.pt"
)
DEFAULT_CFG = (
    "checkpoints/PEMS04/H12/budget_f2f/"
    "supernet_eta0p50_dynamic_fair_rawscale_loss_v2_60f53aa1c6/seed1/"
    "b5678fda5e8d94ed028c6c8bb073461d/"
    "H12_supernet_eta0p50_dynamic_fair_rawscale_loss_v2_60f53aa1c6_seed1.py"
)
DEFAULT_VALID_ORACLE = "results/pems04_budget_f2f_oracle_valid_rawscale.json"


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (ROOT / p)


def _route_key(route: list[int]) -> str:
    return ",".join(str(int(x)) for x in route)


def _hist_from_ids(ids: list[int], candidates: list[list[int]]) -> dict[str, int]:
    counter = Counter(int(i) for i in ids)
    out: dict[str, int] = {}
    for rid, cnt in sorted(counter.items()):
        if 0 <= rid < len(candidates):
            key = _route_key(candidates[rid])
        else:
            key = f"id:{rid}"
        out[key] = int(cnt)
    return out


def build_loader(split: str, batch_size: int, data_dir: str = "datasets/PEMS04", horizon: int = 12):
    from basicts.data import TimeSeriesForecastingDataset

    mode = "valid" if split in {"val", "valid"} else split
    ds = TimeSeriesForecastingDataset(
        data_file_path=str(ROOT / data_dir / f"data_in12_out{horizon}.pkl"),
        index_file_path=str(ROOT / data_dir / f"index_in12_out{horizon}.pkl"),
        mode=mode,
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2), len(ds)


class PlanBPolicyEvalNet(nn.Module):
    """Wrap frozen supernet + Plan B policy so evaluate_loader can call forward()."""

    def __init__(
        self,
        supernet: nn.Module,
        policy: GroupRelativeRefinementPolicy,
        env: SequentialF2FEnvironment,
        *,
        eta: float = 1.0,
        stochastic: bool = False,
    ):
        super().__init__()
        self.supernet = supernet
        self.policy = policy
        self.env = env
        self.eta = float(eta)
        self.stochastic = bool(stochastic)
        # Compatibility fields used by some eval helpers
        self.candidate_routes = list(supernet.candidate_routes)
        self.route_costs = supernet.route_costs
        self.output_len = int(supernet.output_len)
        self.node_size = int(supernet.node_size)
        self.output_dim = int(supernet.output_dim)

    def set_eta(self, eta: float) -> None:
        self.eta = float(eta)

    @torch.no_grad()
    def select_route_ids(self, history: torch.Tensor) -> torch.Tensor:
        device = history.device
        b = history.shape[0]
        masks = self.env.action_masks(self.eta)
        m0 = masks["mask0"].to(device).unsqueeze(0).expand(b, -1)
        m1 = masks["mask1"].to(device).unsqueeze(0).expand(b, -1)

        s0 = self.policy.encode_s0(_build_s0(self.supernet, history))
        logits0 = self.policy.logits0(s0)
        if self.stochastic:
            log0 = self.policy.masked_log_softmax(logits0, m0)
            a0 = torch.multinomial(log0.exp(), 1).squeeze(1)
        else:
            a0 = logits0.masked_fill(~m0, -1e9).argmax(dim=-1)

        a1 = torch.zeros(b, dtype=torch.long, device=device)
        needs_q = a0 == A0_QUARTER
        if needs_q.any():
            # Explicit Z_q only for quarter trajectories (sequential semantics).
            idx = needs_q.nonzero(as_tuple=True)[0]
            pref = self.env.execute_quarter_prefix(history[idx])
            zqp = self.policy.pool_zq(pref["Z_q"].detach())
            logits1 = self.policy.logits1(s0[idx], zqp)
            m1_sub = m1[idx]
            if self.stochastic:
                log1 = self.policy.masked_log_softmax(logits1, m1_sub)
                a1_sub = torch.multinomial(log1.exp(), 1).squeeze(1)
            else:
                a1_sub = logits1.masked_fill(~m1_sub, -1e9).argmax(dim=-1)
            a1[idx] = a1_sub

        route_ids = torch.empty(b, dtype=torch.long, device=device)
        for i in range(b):
            route = self.env.route_from_actions(
                int(a0[i].item()),
                int(a1[i].item()) if int(a0[i].item()) == A0_QUARTER else None,
            )
            # Map route list -> semantic id via env index_map
            key = {
                tuple(self.env.template["direct"]): "direct",
                tuple(self.env.template["half"]): "half",
                tuple(self.env.template["quarter"]): "quarter",
                tuple(self.env.template["progressive"]): "progressive",
            }[tuple(route)]
            route_ids[i] = int(self.env.index_map[key])
        return route_ids

    @torch.no_grad()
    def forward(
        self,
        history_data: torch.Tensor,
        future_data: torch.Tensor | None = None,
        batch_seen: int | None = None,
        epoch: int | None = None,
        train: bool = False,
        return_all: bool = True,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del future_data, batch_seen, epoch, train, kwargs
        history = history_data
        route_ids = self.select_route_ids(history)
        executed = self.supernet._execute_routes_bucketed(history, route_ids)
        masks = self.env.action_masks(self.eta)
        budget = float(masks["budget"])
        b = history.shape[0]
        cost = executed["selected_cost"]
        out = {
            "pred": executed["pred"],
            "selected_route_id": route_ids,
            "executed_route_id": executed["executed_route_id"],
            "selected_cost": cost,
            "budget": history.new_full((b,), budget, dtype=cost.dtype),
            "chain_resolutions": executed.get("chain_resolutions", [self.output_len]),
            "batch_route_id": int(torch.bincount(route_ids.cpu(), minlength=len(self.candidate_routes)).argmax().item()),
        }
        if return_all:
            return out
        return out["pred"]


def load_policy(
    ckpt_path: Path,
    supernet: nn.Module,
    device: torch.device,
) -> tuple[GroupRelativeRefinementPolicy, dict[str, Any]]:
    blob = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(blob, dict) or "policy_state_dict" not in blob:
        raise RuntimeError(
            f"Plan B checkpoint must contain policy_state_dict: {ckpt_path}"
        )
    with torch.no_grad():
        probe = torch.zeros(1, 12, int(supernet.node_size), 4, device=device)
        ctx_dim = int(_build_s0(supernet, probe).shape[-1])
    # Infer hidden from saved weight if present
    sd = blob["policy_state_dict"]
    hidden = 256
    if "s0_proj.0.weight" in sd:
        hidden = int(sd["s0_proj.0.weight"].shape[0])
    policy = GroupRelativeRefinementPolicy(
        context_dim=ctx_dim, zq_dim=1, hidden=hidden
    ).to(device)
    missing, unexpected = policy.load_state_dict(sd, strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"policy load mismatch missing={missing} unexpected={unexpected}"
        )
    policy.eval()
    meta = {
        k: blob[k]
        for k in ("policy_params", "valid", "args", "method", "reward_source", "epoch")
        if k in blob
    }
    meta["context_dim"] = ctx_dim
    meta["hidden"] = hidden
    return policy, meta


@torch.no_grad()
def valid_oracle_regret(
    runner: PlanBPolicyEvalNet,
    *,
    valid_oracle: Path,
    device: torch.device,
    etas: list[float],
    batch_size: int,
    routes: list[list[int]],
) -> dict[str, Any]:
    """VALID-only: compare selected routes to official VALID oracle losses."""
    from basicts.archs.arch_zoo.ChainForecasting_arch.budget_route_utils import (
        load_route_costs,
    )
    from basicts.data.forecast_refinement_gain_dataset import (
        ForecastRefinementGainDataset,
        collate_refinement_gains,
    )
    from basicts.data.indexed_timeseries_dataset import IndexedTimeSeriesForecastingDataset

    costs = load_route_costs(None, routes, 12)
    ds = ForecastRefinementGainDataset(
        IndexedTimeSeriesForecastingDataset(
            str(ROOT / "datasets/PEMS04/data_in12_out12.pkl"),
            str(ROOT / "datasets/PEMS04/index_in12_out12.pkl"),
            "valid",
        ),
        str(valid_oracle),
        expected_routes=routes,
        expected_costs=costs,
        expected_horizon=12,
        expected_dataset="PEMS04",
        require_len_match=False,
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_refinement_gains,
    )
    out: dict[str, Any] = {}
    for eta in etas:
        runner.set_eta(float(eta))
        regrets: list[float] = []
        sel_costs: list[float] = []
        hist: Counter = Counter()
        for history, _si, _gains, losses in loader:
            history = history.to(device)
            losses = losses.to(device)
            rids = runner.select_route_ids(history)
            masks = runner.env.action_masks(float(eta))
            feas = masks["feasible_routes"].to(device)
            for i in range(history.shape[0]):
                rid = int(rids[i].item())
                best = float(losses[i][feas].min().item())
                regrets.append(float(losses[i, rid].item() - best))
                sel_costs.append(float(runner.supernet.route_costs[rid].item()))
                hist[rid] += 1
        t = torch.tensor(regrets, dtype=torch.float64)
        out[str(eta)] = {
            "mean_strict_regret": float(t.mean().item()) if t.numel() else 0.0,
            "median_strict_regret": float(t.median().item()) if t.numel() else 0.0,
            "p90_strict_regret": float(t.quantile(0.90).item()) if t.numel() else 0.0,
            "avg_selected_cost": float(sum(sel_costs) / max(len(sel_costs), 1)),
            "route_histogram": _hist_from_ids(list(hist.elements()), routes)
            if hist
            else {},
            "n_samples": int(t.numel()),
        }
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy-checkpoint", required=True)
    p.add_argument("--supernet-checkpoint", default=DEFAULT_SUPERNET)
    p.add_argument("--cfg", default=DEFAULT_CFG)
    p.add_argument("--valid-oracle", default=None, help="Official VALID oracle for regret (VALID only)")
    p.add_argument("--split", default="valid", choices=["valid", "test", "both"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--stochastic-eval", action="store_true")
    p.add_argument(
        "--etas",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.5, 0.75, 1.0],
    )
    p.add_argument("--scaler", default="datasets/PEMS04/scaler_in12_out12.pkl")
    p.add_argument("--out", default="results/planB_policy_eval.json")
    p.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optional cap for smoke/debug; omit for full split.",
    )
    args = p.parse_args()

    policy_path = _resolve(args.policy_checkpoint)
    supernet_path = _resolve(args.supernet_checkpoint)
    cfg_path = _resolve(args.cfg) if args.cfg else None
    scaler_path = _resolve(args.scaler)
    out_path = _resolve(args.out)

    if not policy_path.is_file():
        raise FileNotFoundError(f"policy checkpoint not found: {policy_path}")
    if not supernet_path.is_file():
        raise FileNotFoundError(f"supernet checkpoint not found: {supernet_path}")
    if cfg_path is not None and not cfg_path.is_file():
        raise FileNotFoundError(f"cfg not found: {cfg_path}")
    if not scaler_path.is_file():
        raise FileNotFoundError(f"scaler not found: {scaler_path}")

    device = torch.device(
        args.device if ("cpu" in args.device or torch.cuda.is_available()) else "cpu"
    )
    routes = default_candidate_routes(12)

    class _NS:
        horizon = 12
        controller_dim = 128
        pooling_queries = 4
        delta_abs = 0.05
        route_cost_file = None
        cfg = str(cfg_path) if cfg_path is not None else None

    print(f"[load] supernet={supernet_path}")
    supernet = _build_model(_NS(), routes, device)
    _load_supernet(supernet, supernet_path)
    supernet.eval()
    for param in supernet.parameters():
        param.requires_grad = False

    print(f"[load] policy={policy_path}")
    policy, policy_meta = load_policy(policy_path, supernet, device)
    env = SequentialF2FEnvironment(supernet)
    runner = PlanBPolicyEvalNet(
        supernet,
        policy,
        env,
        eta=1.0,
        stochastic=bool(args.stochastic_eval),
    ).to(device)
    runner.eval()

    scaler = load_pkl(str(scaler_path))
    if not isinstance(scaler, dict) or "func" not in scaler or "args" not in scaler:
        raise RuntimeError(f"invalid scaler: {scaler_path}")

    forward_features = [0, 1, 2, 3]
    target_features = [0]
    null_val = 0.0

    splits = ["valid", "test"] if args.split == "both" else [args.split]
    report: dict[str, Any] = {
        "status": "ok",
        "method": "Plan B GRPO-inspired group-relative trajectory policy",
        "selection_rule": "deterministic masked argmax"
        if not args.stochastic_eval
        else "stochastic masked sample",
        "metric_path": "basicts.metrics.masked_* + scaler inverse (evaluate_loader)",
        "policy_checkpoint": str(policy_path),
        "supernet_checkpoint": str(supernet_path),
        "cfg": str(cfg_path) if cfg_path else None,
        "scaler": str(scaler_path),
        "candidate_routes": routes,
        "etas": list(args.etas),
        "device": str(device),
        "policy_meta": {
            k: policy_meta[k]
            for k in policy_meta
            if k in {"policy_params", "method", "reward_source", "context_dim", "hidden", "epoch"}
            or k == "valid"
        },
        "splits": {},
        "notes": [
            "normalized_static_cost is a computation proxy, not measured latency",
            "TEST oracle is forbidden; VALID oracle regret is optional diagnostics only",
        ],
    }

    t_all = time.perf_counter()
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
        report["splits"][split] = {
            "n_samples": int(n),
            "etas": eta_rows,
        }

    # VALID oracle regret (optional; never on TEST)
    if args.valid_oracle or (
        args.split in {"valid", "both"} and _resolve(DEFAULT_VALID_ORACLE).is_file()
    ):
        voracle = _resolve(args.valid_oracle or DEFAULT_VALID_ORACLE)
        if args.split in {"valid", "both"} and voracle.is_file():
            print(f"[valid-oracle-regret] {voracle}")
            report["valid_oracle_regret"] = valid_oracle_regret(
                runner,
                valid_oracle=voracle,
                device=device,
                etas=list(args.etas),
                batch_size=int(args.batch_size),
                routes=routes,
            )
            report["valid_oracle_path"] = str(voracle)
        elif args.valid_oracle:
            raise FileNotFoundError(f"valid oracle not found: {voracle}")

    report["wall_time_sec"] = float(time.perf_counter() - t_all)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"wrote": str(out_path), "wall_time_sec": report["wall_time_sec"]}, indent=2))
    # Compact terminal summary
    for split, block in report["splits"].items():
        print(f"\n=== {split.upper()} BasicTS ===")
        for eta, row in block["etas"].items():
            print(
                f"eta={eta}: MAE={row['mae']:.4f} RMSE={row['rmse']:.4f} "
                f"MAPE={row['mape']:.4f} cost={row['average_selected_cost']:.4f} "
                f"stages={row['average_stage_count']:.3f} hist={row['route_histogram_sample']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
