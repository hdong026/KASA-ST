#!/usr/bin/env python3
"""Runner wrapper for budget-conditioned adaptive F2F (does not alter formal variants).

User executes training; this agent must not launch training.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_BASE = ROOT / "scripts" / "run_chain_forecasting_horizon.py"
_spec = importlib.util.spec_from_file_location("run_chain_forecasting_horizon", _BASE)
base = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(base)

from basicts.archs.arch_zoo.ChainForecasting_arch.budget_route_utils import (
    parse_candidate_routes,
    parse_route,
    validate_route,
)

VARIANT = "chain_budget_conditioned_adaptive_f2f_kasa_condition_adapter_token_loss"


def main() -> int:
    parser = argparse.ArgumentParser(description="Budget-conditioned adaptive F2F experiments.")
    parser.add_argument("--dataset", default="PEMS04", choices=list(base.DATASET_SPECS.keys()))
    parser.add_argument("--horizons", type=int, nargs="+", default=[12])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1])
    parser.add_argument("--gpus", nargs="+", default=["0"])
    parser.add_argument("--prepare_data", "--prepare-data", action="store_true")
    parser.add_argument("--out", default=None)
    parser.add_argument("--markdown", default=None)
    parser.add_argument(
        "--training-phase",
        default="supernet",
        choices=["supernet", "planner", "joint", "eval"],
    )
    parser.add_argument(
        "--candidate-routes",
        nargs="+",
        default=None,
        help="e.g. 12 6,12 3,12 3,6,12",
    )
    parser.add_argument("--forced-route", default=None)
    parser.add_argument(
        "--route-selection-mode",
        default="batch",
        choices=["batch", "sample", "forced"],
    )
    parser.add_argument("--route-granularity", default="batch", choices=["batch", "sample"])
    parser.add_argument("--inference-intensity", type=float, default=0.5)
    parser.add_argument("--route-cost-file", default=None)
    parser.add_argument("--route-cost-type", default="normalized_static_cost")
    parser.add_argument("--oracle-file", default=None)
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--freeze-forecasting-backbone", action="store_true")
    parser.add_argument(
        "--loss-mode",
        default="dynamic_fair",
        choices=["baseline_compatible", "dynamic_fair"],
    )
    parser.add_argument(
        "--route-sampling",
        default="sandwich",
        choices=["sandwich", "random", "none"],
    )
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--skip_completed", action="store_true", default=True)
    parser.add_argument("--no_skip_completed", action="store_false", dest="skip_completed")
    args = parser.parse_args()

    base.activate_dataset(args.dataset)
    if VARIANT not in base.VARIANT_SPECS:
        raise KeyError(f"Missing variant {VARIANT} in VARIANT_SPECS")
    spec = dict(base.VARIANT_SPECS[VARIANT])
    # Candidate routes depend on horizon; single-horizon runs are the expected path.
    horizon0 = int(args.horizons[0])
    if args.candidate_routes is not None:
        routes = parse_candidate_routes(args.candidate_routes, horizon0)
    else:
        routes = parse_candidate_routes(None, horizon0)
    for h in args.horizons:
        for r in routes:
            validate_route(r, horizon=h)
    spec["candidate_routes"] = routes
    spec["training_phase"] = args.training_phase
    spec["route_selection_mode"] = args.route_selection_mode
    spec["route_granularity"] = args.route_granularity
    spec["inference_intensity"] = float(args.inference_intensity)
    spec["route_cost_type"] = args.route_cost_type
    spec["loss_mode"] = args.loss_mode
    spec["chain_loss_mode"] = args.loss_mode
    if args.route_cost_file:
        spec["route_cost_file"] = args.route_cost_file
    if args.oracle_file:
        spec["oracle_file"] = args.oracle_file
    if args.init_checkpoint:
        spec["init_checkpoint"] = args.init_checkpoint
    spec["freeze_forecasting_backbone"] = bool(args.freeze_forecasting_backbone)
    spec["route_sampling"] = args.route_sampling
    if args.forced_route is not None:
        fr = parse_route(args.forced_route)
        for h in args.horizons:
            validate_route(fr, horizon=h)
        if fr not in routes:
            raise ValueError(f"forced-route {fr} not in candidate pool {routes}")
        spec["forced_route"] = fr
        spec["route_selection_mode"] = "forced"
        # Forced equivalence / ablation: disable sandwich
        spec["training_phase"] = "eval" if args.training_phase == "eval" else args.training_phase
        if args.training_phase == "supernet":
            # Still train forecasting, but execute the forced route only.
            spec["route_sampling"] = "none"
            spec["training_phase"] = "supernet"
            # Clear sandwich by marking phase and forced
        # Do not sandwich when forced
        if args.forced_route is not None:
            spec["route_sampling"] = "none"
    base.VARIANT_SPECS[VARIANT] = spec

    argv = [
        "run_chain_forecasting_horizon.py",
        "--dataset",
        args.dataset,
        "--variants",
        VARIANT,
        "--horizons",
        *[str(h) for h in args.horizons],
        "--seeds",
        *[str(s) for s in args.seeds],
        "--gpus",
        *args.gpus,
    ]
    if args.prepare_data:
        argv.append("--prepare_data")
    if args.out:
        argv.extend(["--out", args.out])
    if args.markdown:
        argv.extend(["--markdown", args.markdown])
    if args.dry_run:
        argv.append("--dry_run")
    if not args.skip_completed:
        argv.append("--no_skip_completed")
    sys.argv = argv
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
