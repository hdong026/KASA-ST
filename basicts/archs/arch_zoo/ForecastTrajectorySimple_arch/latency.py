"""Measure actual trajectory execution latency on the deployment device."""

from __future__ import annotations

import statistics
import time
from collections.abc import Sequence

import torch


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def profile_trajectory_latency(
    model,
    history_data: torch.Tensor,
    trajectories: Sequence[Sequence[int]],
    *,
    warmup: int = 10,
    repeats: int = 50,
) -> dict[tuple[int, ...], dict[str, float]]:
    """Profile end-to-end batch latency for each real trajectory."""
    if warmup < 0 or repeats <= 0:
        raise ValueError("warmup must be >= 0 and repeats must be > 0.")
    device = history_data.device
    was_training = model.training
    model.eval()
    results = {}
    try:
        with torch.inference_mode():
            for values in trajectories:
                route = tuple(int(value) for value in values)
                for _ in range(warmup):
                    model.execute_trajectory(history_data, route)
                _synchronize(device)
                samples_ms = []
                for _ in range(repeats):
                    _synchronize(device)
                    start = time.perf_counter()
                    model.execute_trajectory(history_data, route)
                    _synchronize(device)
                    samples_ms.append((time.perf_counter() - start) * 1000.0)
                ordered = sorted(samples_ms)
                p90_index = min(len(ordered) - 1, int(0.9 * len(ordered)))
                results[route] = {
                    "median_ms": float(statistics.median(samples_ms)),
                    "mean_ms": float(statistics.fmean(samples_ms)),
                    "p90_ms": float(ordered[p90_index]),
                    "min_ms": float(ordered[0]),
                    "repeats": int(repeats),
                    "batch_size": int(history_data.shape[0]),
                }
    finally:
        model.train(was_training)
    return results
