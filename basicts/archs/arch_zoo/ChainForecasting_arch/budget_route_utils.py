"""Candidate forecast-to-forecast routes under a computation budget."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def default_candidate_routes(horizon: int) -> list[list[int]]:
    """Generate the controlled route pool for horizon H.

    Requires H divisible by 4:
      R0=[H], R1=[H/2,H], R2=[H/4,H], R3=[H/4,H/2,H]
    """
    h = int(horizon)
    if h < 1:
        raise ValueError(f"horizon must be >= 1, got {h}")
    if h % 4 != 0:
        raise ValueError(
            f"default_candidate_routes requires H divisible by 4, got H={h}"
        )
    h2 = h // 2
    h4 = h // 4
    return [
        [h],
        [h2, h],
        [h4, h],
        [h4, h2, h],
    ]


def parse_route(spec: str | list[int] | tuple[int, ...]) -> list[int]:
    if isinstance(spec, (list, tuple)):
        route = [int(x) for x in spec]
    else:
        text = str(spec).strip()
        if not text:
            raise ValueError("empty route")
        route = [int(x.strip()) for x in text.split(",") if x.strip()]
    validate_route(route, horizon=route[-1] if route else None)
    return route


def validate_route(route: list[int], horizon: int | None = None) -> None:
    if not route:
        raise ValueError("route must be non-empty")
    if any(int(x) < 1 for x in route):
        raise ValueError(f"route resolutions must be positive integers: {route}")
    for a, b in zip(route, route[1:]):
        if int(a) >= int(b):
            raise ValueError(f"route must be strictly increasing: {route}")
    if horizon is not None and int(route[-1]) != int(horizon):
        raise ValueError(
            f"route final resolution {route[-1]} must equal horizon {horizon}"
        )


def parse_candidate_routes(
    specs: list[str] | list[list[int]] | None,
    horizon: int,
) -> list[list[int]]:
    if specs is None:
        return default_candidate_routes(horizon)
    routes = []
    for s in specs:
        r = parse_route(s)
        validate_route(r, horizon=horizon)
        routes.append(r)
    if not routes:
        raise ValueError("candidate route pool is empty")
    return routes


def unique_resolutions(routes: list[list[int]]) -> list[int]:
    vals = sorted({int(x) for r in routes for x in r})
    return vals


def normalized_static_costs(routes: list[list[int]], horizon: int) -> list[float]:
    """Heuristic costs in [c_min, 1] with [H] cheapest and full chain cost=1."""
    h = float(horizon)
    raw = []
    for r in routes:
        # sum of stage lengths / H plus stage-launch overhead
        raw.append(sum(float(x) / h for x in r) + 0.05 * (len(r) - 1))
    full = max(raw) if raw else 1.0
    return [float(c / full) for c in raw]


def budget_from_intensity(eta: float, costs: list[float]) -> float:
    """Map eta in [0,1] so eta=0 allows min cost ([H]) and eta=1 allows max."""
    eta = float(eta)
    if eta < 0.0 or eta > 1.0:
        raise ValueError(f"inference intensity must be in [0,1], got {eta}")
    c_min = min(costs)
    c_max = max(costs)
    return float(c_min + eta * (c_max - c_min))


def budgets_from_intensity_tensor(
    etas: torch.Tensor,
    costs: list[float],
) -> torch.Tensor:
    """Vectorized budget mapping for per-sample intensities."""
    c_min = min(costs)
    c_max = max(costs)
    return c_min + etas * (c_max - c_min)


def load_route_costs(
    path: str | Path | None,
    routes: list[list[int]],
    horizon: int,
    cost_type: str = "normalized_static_cost",
) -> list[float]:
    """Load route costs.

    ``cost_type`` / route-cost-source aliases:
      static / normalized / normalized_static_cost -> heuristic or JSON normalized_cost
      measured_latency / latency -> JSON mean_latency_ms (or measured_latency_ms), then normalize
    """
    cost_type = str(cost_type).lower()
    static_aliases = {"normalized_static_cost", "static", "normalized"}
    latency_aliases = {
        "measured_latency",
        "latency",
        "measured_latency_ms",
        "mean_latency_ms",
    }
    if path is None:
        if cost_type not in static_aliases:
            raise ValueError(
                f"ROUTE_COST_FILE is required for cost_type={cost_type}"
            )
        return normalized_static_costs(routes, horizon)
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "routes" not in data:
        raise ValueError(f"route cost file missing 'routes': {path}")
    if cost_type in latency_aliases or "latency" in cost_type:
        preferred_keys = (
            "normalized_cost",
            "mean_latency_ms",
            "measured_latency_ms",
            "median_latency_ms",
            "cost",
        )
        use_normalize = True
    else:
        preferred_keys = (
            "normalized_cost",
            "normalized_static_cost",
            "cost",
            "mean_latency_ms",
        )
        use_normalize = False
    by_key = {}
    raw_latency = {}
    for entry in data["routes"]:
        r = tuple(int(x) for x in entry["route"])
        chosen = None
        for key in preferred_keys:
            if key in entry:
                chosen = float(entry[key])
                break
        if chosen is None:
            raise ValueError(f"route entry missing cost fields for {list(r)} in {path}")
        by_key[r] = chosen
        if "mean_latency_ms" in entry:
            raw_latency[r] = float(entry["mean_latency_ms"])
        elif "measured_latency_ms" in entry:
            raw_latency[r] = float(entry["measured_latency_ms"])
    out = []
    for r in routes:
        tr = tuple(int(x) for x in r)
        if tr not in by_key:
            raise ValueError(f"missing cost for route {list(tr)} in {path}")
        # Prefer explicit normalized_cost when present for budget comparisons.
        entry_cost = by_key[tr]
        out.append(entry_cost)
    if use_normalize:
        # If values look like milliseconds (>>1), normalize by max.
        if max(out) > 1.5:
            m = max(out) if out else 1.0
            out = [c / m for c in out]
    return out


def route_to_key(route: list[int]) -> str:
    return ",".join(str(int(x)) for x in route)


def forced_route_tag(route: list[int] | None) -> str | None:
    """Canonical id for a forced route, e.g. forced_12 / forced_3-6-12."""
    if route is None:
        return None
    return "forced_" + "-".join(str(int(x)) for x in route)


def candidate_routes_tag(routes: list[list[int]]) -> str:
    return "+".join(route_to_key(r).replace(",", "-") for r in routes)


def build_run_signature(
    *,
    dataset: str,
    horizon: int,
    seed: int,
    base_variant: str,
    route_selection_mode: str,
    forced_route: list[int] | None,
    training_phase: str,
    loss_mode: str,
    candidate_routes: list[list[int]],
    route_granularity: str,
    inference_intensity: float,
    route_cost_type: str,
    route_cost_file: str | None = None,
    run_tag: str | None = None,
    code_version: str | None = None,
    num_epochs: int | None = None,
    planner_lr: float | None = None,
    backbone_lr: float | None = None,
    lambda_mid: float | None = None,
    lambda_imitation: float | None = None,
    lambda_budget: float | None = None,
    planner_training_intensities: list[float] | None = None,
) -> dict[str, Any]:
    """Full experiment identity; used for paths, skip, and result rows."""
    import hashlib

    fr_tag = forced_route_tag(forced_route)
    eta_train = (
        "+".join(f"{float(x):.2f}" for x in planner_training_intensities)
        if planner_training_intensities
        else "none"
    )
    sig_parts = [
        f"ds={dataset}",
        f"H={int(horizon)}",
        f"seed={int(seed)}",
        f"variant={base_variant}",
        f"mode={route_selection_mode}",
        f"forced={route_to_key(forced_route) if forced_route else 'none'}",
        f"phase={training_phase}",
        f"loss={loss_mode}",
        f"cands={candidate_routes_tag(candidate_routes)}",
        f"gran={route_granularity}",
        f"eta={float(inference_intensity):.4f}",
        f"cost={route_cost_type}",
        f"cost_file={Path(route_cost_file).name if route_cost_file else 'none'}",
        f"code={code_version or 'unknown'}",
        f"epochs={num_epochs if num_epochs is not None else 'default'}",
        f"plr={planner_lr if planner_lr is not None else 'default'}",
        f"blr={backbone_lr if backbone_lr is not None else 'default'}",
        f"lmid={lambda_mid if lambda_mid is not None else 'default'}",
        f"lim={lambda_imitation if lambda_imitation is not None else 'default'}",
        f"lbud={lambda_budget if lambda_budget is not None else 'default'}",
        f"train_etas={eta_train}",
    ]
    if run_tag:
        sig_parts.append(f"tag={run_tag}")
    run_signature = "|".join(sig_parts)
    digest = hashlib.sha1(run_signature.encode("utf-8")).hexdigest()[:10]
    # EasyTorch imports cfg as a Python module path; dots in the filename break import.
    def _safe_tag(text: str) -> str:
        return (
            str(text)
            .replace(".", "p")
            .replace("/", "-")
            .replace("\\", "-")
            .replace(" ", "_")
        )

    if fr_tag is not None:
        # e.g. forced_3-6-12_supernet_baseline_compatible_<digest>
        experiment_tag = f"{fr_tag}_{training_phase}_{loss_mode}_{digest}"
    else:
        eta_tag = f"eta{float(inference_intensity):.2f}".replace(".", "p")
        base_tag = f"{training_phase}_{eta_tag}_{loss_mode}"
        if run_tag:
            base_tag = f"{base_tag}_{run_tag}"
        experiment_tag = f"{base_tag}_{digest}"
    experiment_tag = _safe_tag(experiment_tag)
    return {
        "run_signature": run_signature,
        "experiment_tag": experiment_tag,
        "forced_route_tag": fr_tag,
        "forced_route": list(forced_route) if forced_route else None,
        "candidate_routes": [list(r) for r in candidate_routes],
        "code_version": code_version or "unknown",
        "digest": digest,
    }


def sample_sandwich_routes(
    candidates: list[list[int]],
    rng: Any | None = None,
) -> list[list[int]]:
    """Return [min_route, max_route, random_intermediate] for supernet training."""
    import random

    if len(candidates) < 1:
        raise ValueError("empty candidate routes")
    by_len = sorted(candidates, key=lambda r: (len(r), sum(r)))
    min_r = by_len[0]
    max_r = by_len[-1]
    mids = [r for r in candidates if r != min_r and r != max_r]
    if not mids:
        mid = list(min_r) if len(candidates) == 1 else list(by_len[len(by_len) // 2])
    else:
        pick = rng.choice(mids) if rng is not None else random.choice(mids)
        mid = list(pick)
    return [list(min_r), list(max_r), mid]
