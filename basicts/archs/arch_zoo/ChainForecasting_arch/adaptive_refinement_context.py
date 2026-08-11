"""Pre-route feature extraction utilities for Plan A/B adaptive routing.

IMPORTANT — cache / reuse policy
--------------------------------
``extract_pre_route_context`` taps the horizon-stage ``patch_encoder`` embedding
path (+ spatial codebook). This is **not** mathematically identical to the full
inputs of ``_execute_route`` / ``KASATemporalStep`` (which also run forecast
projections, progressive spatial refine, condition adapters, etc.).

Therefore we do **not** pretend this is a shared execution cache that can replace
route stages. Name the cost honestly:

    pre_route_feature_extraction

and count it as extra overhead in profiling. Never alter F2F numerical semantics
to "save" this cost.

What *is* fixed: controller/policy must not be forward'ed twice in one adaptive
``forward`` (plan extras are copied from ``_select_route_id``).
"""

from __future__ import annotations

from typing import Any

import torch


PRE_ROUTE_OVERHEAD_NAME = "pre_route_feature_extraction"


def pool_pre_route_context(h_shared: torch.Tensor) -> torch.Tensor:
    """Pool [B,M,N,D] (or [B,N,D]) pre-route features to [B,D] for policy heads."""
    if h_shared.ndim == 4:
        return h_shared.mean(dim=(1, 2))
    if h_shared.ndim == 3:
        return h_shared.mean(dim=1)
    if h_shared.ndim == 2:
        return h_shared
    raise ValueError(f"unexpected pre-route shape {tuple(h_shared.shape)}")


def pre_route_overhead_report(*, enabled: bool = True) -> dict[str, Any]:
    return {
        "name": PRE_ROUTE_OVERHEAD_NAME,
        "cache_reuse_into_execute_route": False,
        "reason": (
            "patch_encoder embed tap is not a safe drop-in for full KASATemporalStep "
            "stage execution; treating as explicit extra overhead"
        ),
        "enabled": bool(enabled),
    }
