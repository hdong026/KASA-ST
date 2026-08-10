"""Standard forecast-refinement route templates and tuple lookups."""

from __future__ import annotations

from typing import Any


def standard_refinement_route_template(horizon: int) -> dict[str, list[int]]:
    """Semantic routes for horizon H (H divisible by 4)."""
    h = int(horizon)
    if h < 4 or h % 4 != 0:
        raise ValueError(f"standard refinement template requires H%4==0, got H={h}")
    return {
        "direct": [h],
        "half": [h // 2, h],
        "quarter": [h // 4, h],
        "progressive": [h // 4, h // 2, h],
    }


def route_key(route: list[int] | tuple[int, ...]) -> tuple[int, ...]:
    return tuple(int(x) for x in route)


def build_refinement_route_index_map(
    candidate_routes: list[list[int]],
    horizon: int,
) -> dict[str, int]:
    """Map semantic names -> indices in the configured candidate pool.

    Raises if the candidate pool does not contain the standard template.
    """
    template = standard_refinement_route_template(horizon)
    by_key = {route_key(r): i for i, r in enumerate(candidate_routes)}
    mapping: dict[str, int] = {}
    missing = []
    for name, route in template.items():
        k = route_key(route)
        if k not in by_key:
            missing.append(f"{name}={list(route)}")
        else:
            mapping[name] = by_key[k]
    if missing:
        raise ValueError(
            "candidate_routes missing standard refinement template entries: "
            f"{missing}; candidates={candidate_routes}; "
            f"expected template={ {k: list(v) for k, v in template.items()} }"
        )
    return mapping


def gains_from_route_losses(
    losses_by_name: dict[str, float],
) -> dict[str, float]:
    """Build G3/G6/G36 and check hierarchical identity."""
    l0 = float(losses_by_name["direct"])
    l1 = float(losses_by_name["half"])
    l2 = float(losses_by_name["quarter"])
    l3 = float(losses_by_name["progressive"])
    g3 = l0 - l2
    g6 = l0 - l1
    g36 = l2 - l3
    full = l0 - l3
    if abs((g3 + g36) - full) > 1e-5:
        raise RuntimeError(
            f"full-gain identity failed: (L0-L2)+(L2-L3)={g3 + g36} vs L0-L3={full}"
        )
    return {"g3": g3, "g6": g6, "g36": g36, "full": full}


def route_scores_from_gains(
    g3,
    g6,
    g36,
    *,
    index_map: dict[str, int],
    n_routes: int,
):
    """Assemble score vector in candidate-route order from semantic gains."""
    import torch

    if torch.is_tensor(g3):
        b = g3.shape[0]
        device = g3.device
        dtype = g3.dtype
        scores = torch.zeros(b, n_routes, device=device, dtype=dtype)
        scores[:, index_map["direct"]] = 0.0
        scores[:, index_map["quarter"]] = g3
        scores[:, index_map["half"]] = g6
        scores[:, index_map["progressive"]] = g3 + g36
        return scores
    scores = [0.0] * n_routes
    scores[index_map["direct"]] = 0.0
    scores[index_map["quarter"]] = float(g3)
    scores[index_map["half"]] = float(g6)
    scores[index_map["progressive"]] = float(g3) + float(g36)
    return scores
