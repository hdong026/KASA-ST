"""Shared adaptive refinement route templates (Plan A/B).

Thin re-export of ``forecast_refinement_routes`` so Plan A/B scripts share one
semantic entrypoint without duplicating route math.
"""

from __future__ import annotations

from basicts.archs.arch_zoo.ChainForecasting_arch.forecast_refinement_routes import (
    build_refinement_route_index_map,
    gains_from_route_losses,
    route_key,
    route_scores_from_gains,
    standard_refinement_route_template,
)

__all__ = [
    "standard_refinement_route_template",
    "route_key",
    "build_refinement_route_index_map",
    "gains_from_route_losses",
    "route_scores_from_gains",
]
