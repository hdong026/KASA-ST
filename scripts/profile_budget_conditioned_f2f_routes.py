#!/usr/bin/env python3
"""Profile forced-route inference latency. Implement only — do not run in this task."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

# NOTE: This script is for the user to run later on GPU/CPU.
# The agent must not execute training or long profiling.


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True, help="easytorch cfg path with MODEL.PARAM")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--routes",
        nargs="+",
        default=["12", "6,12", "3,12", "3,6,12"],
    )
    args = parser.parse_args()

    # Lazy imports so py_compile of other files is unaffected
    import sys

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from easytorch.config import import_config
    from basicts.archs.arch_zoo.ChainForecasting_arch.budget_route_utils import (
        parse_route,
    )

    cfg = import_config(args.cfg)
    model = cfg.MODEL.ARCH(**cfg.MODEL.PARAM)
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu")
        # easytorch checkpoints may wrap weights
        sd = state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state
        model.load_state_dict(sd, strict=False)
    device = torch.device(args.device if torch.cuda.is_available() or "cpu" in args.device else "cpu")
    model = model.to(device).eval()
    p = int(cfg.MODEL.PARAM["input_len"])
    h = int(cfg.MODEL.PARAM["output_len"])
    n = int(cfg.MODEL.PARAM["node_size"])
    c = int(cfg.MODEL.PARAM.get("input_dim", 3))
    x = torch.randn(args.batch_size, p, n, c, device=device)

    results = {
        "device": str(device),
        "batch_size": args.batch_size,
        "horizon": h,
        "num_nodes": n,
        "param_count": sum(pp.numel() for pp in model.parameters()),
        "routes": [],
    }
    for rs in args.routes:
        route = parse_route(rs)
        model.set_forced_route(route)
        model.route_selection_mode = "forced"
        # warmup
        with torch.no_grad():
            for _ in range(args.warmup):
                _ = model(history_data=x, train=False, return_all=False)
                if device.type == "cuda":
                    torch.cuda.synchronize()
        times = []
        peak = 0
        with torch.no_grad():
            for _ in range(args.repeat):
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = model(history_data=x, train=False, return_all=False)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                    peak = max(peak, int(torch.cuda.max_memory_allocated(device)))
                times.append((time.perf_counter() - t0) * 1000.0)
        import statistics as stats

        entry = {
            "route": route,
            "measured_latency_ms": float(stats.mean(times)),
            "latency_std_ms": float(stats.pstdev(times)),
            "peak_memory_bytes": peak,
            "normalized_static_cost": None,
        }
        results["routes"].append(entry)
        print(route, entry["measured_latency_ms"], "ms")

    # Fill normalized costs from latency
    mx = max(r["measured_latency_ms"] for r in results["routes"]) or 1.0
    for r in results["routes"]:
        r["normalized_static_cost"] = r["measured_latency_ms"] / mx
        r["cost"] = r["normalized_static_cost"]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
