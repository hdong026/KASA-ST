"""Real CUDA-event latency profiling for history encoder, edges, and policy."""

from __future__ import annotations

import statistics
import time
from typing import Callable, Optional

import torch

from basicts.archs.arch_zoo.ForecastTrajectory_arch.forecast_trajectory_net import (
    ForecastTrajectoryNet,
)
from basicts.archs.arch_zoo.ForecastTrajectory_arch.online_trajectory_policy import (
    OnlineTrajectoryPolicy,
)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _percentile(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    ys = sorted(xs)
    k = min(len(ys) - 1, max(0, int(round((q / 100.0) * (len(ys) - 1)))))
    return float(ys[k])


def summarize_ms(samples: list[float]) -> dict:
    if not samples:
        return {"mean": None, "median": None, "p95": None, "std": None, "n": 0}
    return {
        "mean": float(statistics.fmean(samples)),
        "median": float(statistics.median(samples)),
        "p95": _percentile(samples, 95.0),
        "std": float(statistics.pstdev(samples)) if len(samples) > 1 else 0.0,
        "n": len(samples),
    }


def _time_cuda_ms(fn: Callable[[], None], device: torch.device) -> float:
    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize(device)
        return float(start.elapsed_time(end))
    t0 = time.perf_counter()
    fn()
    return (time.perf_counter() - t0) * 1000.0


@torch.no_grad()
def profile_callable(
    fn: Callable[[], None],
    device: torch.device,
    warmup: int = 50,
    iters: int = 200,
) -> dict:
    _sync(device)
    for _ in range(max(0, int(warmup))):
        fn()
    _sync(device)
    samples = []
    for _ in range(max(1, int(iters))):
        samples.append(_time_cuda_ms(fn, device))
    return summarize_ms(samples)


@torch.no_grad()
def profile_transition_latency(
    model: ForecastTrajectoryNet,
    x_one: torch.Tensor,
    device: torch.device,
    warmup: int = 50,
    iters: int = 200,
    policy: Optional[OnlineTrajectoryPolicy] = None,
    policy_h: Optional[torch.Tensor] = None,
) -> dict:
    """Profile history encoder and every legal edge at batch size 1."""
    model.eval()
    x_one = x_one.to(device)
    if x_one.ndim != 4 or x_one.shape[0] != 1:
        raise ValueError(f"x_one must be [1, L, N, C], got {tuple(x_one.shape)}")

    def hist_fn():
        model.prepare_history(x_one)

    hist_stats = profile_callable(hist_fn, device, warmup=warmup, iters=iters)
    history = model.prepare_history(x_one)

    # Cache a dummy Z for each state via a dense rollout (for s_prev>0 edges).
    dense = model.graph.dense_trajectory()
    z_states: dict[int, torch.Tensor] = {0: None}  # type: ignore
    z_map = model.rollout(x_one, dense, history=history)
    z_states.update(z_map)

    edge_table = {}
    for s_prev, s_next in model.graph.legal_edges():
        z_prev = None if int(s_prev) == 0 else z_states[int(s_prev)]

        def edge_fn(sp=s_prev, sn=s_next, zp=z_prev):
            model.transition(history, zp, sp, sn)

        edge_table[f"{s_prev}->{s_next}"] = profile_callable(
            edge_fn, device, warmup=warmup, iters=iters
        )

    policy_stats = None
    if policy is not None:
        policy.eval()
        h = policy_h if policy_h is not None else history.pooled
        b = h.shape[0]
        lam = torch.zeros(b, device=device)
        rem = torch.ones(b, device=device)
        nb = torch.ones(b, device=device)

        def pol_fn():
            policy(
                h_history=h,
                z_current=None,
                s_current=0,
                lam=lam,
                remaining_norm=rem,
                no_budget=nb,
                H=model.graph.H,
            )

        policy_stats = profile_callable(pol_fn, device, warmup=warmup, iters=iters)

    lookup = {
        "history_median_ms": hist_stats["median"],
        "edges_median_ms": {k: v["median"] for k, v in edge_table.items()},
        "policy_step_median_ms": None if policy_stats is None else policy_stats["median"],
    }
    return {
        "device": str(device),
        "warmup": int(warmup),
        "iters": int(iters),
        "batch_size": 1,
        "history_encoder": hist_stats,
        "edges": edge_table,
        "policy_step": policy_stats,
        "lookup": lookup,
        "source": "cuda_events" if device.type == "cuda" else "cpu_perf_counter",
        "eta_used": False,
        "proxy_costs_used": False,
    }


def trajectory_cost_ms(
    trajectory: list[int] | tuple[int, ...],
    graph,
    latency_lookup: dict,
    include_policy: bool = False,
) -> dict:
    edges = graph.edges_of_trajectory(trajectory)
    c_hist = float(latency_lookup["history_median_ms"] or 0.0)
    edge_ms = latency_lookup["edges_median_ms"]
    c_edges = 0.0
    for sp, sn in edges:
        key = f"{sp}->{sn}"
        c_edges += float(edge_ms[key])
    n_decisions = len(edges)
    c_pol = 0.0
    if include_policy:
        step = latency_lookup.get("policy_step_median_ms") or 0.0
        c_pol = float(step) * n_decisions
    c_trans = c_hist + c_edges
    c_ms = c_trans + c_pol
    return {
        "trajectory": list(trajectory),
        "c_history_ms": c_hist,
        "c_edges_ms": c_edges,
        "c_policy_ms": c_pol,
        "n_decisions": n_decisions,
        "c_transition_ms": c_trans,
        "c_ms": c_ms,
    }
