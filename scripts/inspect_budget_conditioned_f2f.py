#!/usr/bin/env python3
"""CPU architecture inspection for BudgetConditionedAdaptiveF2FNet (no training)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basicts.archs import ChainForecasting, BudgetConditionedAdaptiveF2FNet
from basicts.archs.arch_zoo.ChainForecasting_arch.budget_route_utils import (
    budget_from_intensity,
    default_candidate_routes,
)
from scripts.budget_f2f_synth_kwargs import synthetic_budget_f2f_kwargs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu"])
    args = parser.parse_args()
    assert args.device == "cpu"

    h, n, p, b = 12, 7, 12, 2
    routes = default_candidate_routes(h)
    print("candidate_routes:", routes)

    # Formal import check
    _ = ChainForecasting
    print("formal_ChainForecasting_import: OK")

    for route in routes:
        kw = synthetic_budget_f2f_kwargs(
            node_size=n,
            output_len=h,
            input_len=p,
            forced_route=route,
            route_selection_mode="forced",
            training_phase="eval",
        )
        model = BudgetConditionedAdaptiveF2FNet(**kw).eval()
        x = torch.randn(b, p, n, 4)
        with torch.no_grad():
            out = model(history_data=x, train=False, return_all=True)
        assert out["pred"].shape == (b, h, n, 1)
        assert out["pred"].shape[2] == n
        print(
            f"forced {route}: stages={out['chain_resolutions']} "
            f"shapes={[tuple(t.shape) for t in out['chain_preds']]} "
            f"params={sum(p.numel() for p in model.parameters())}"
        )

    # Intensity feasibility
    kw = synthetic_budget_f2f_kwargs(node_size=n, output_len=h, input_len=p)
    model = BudgetConditionedAdaptiveF2FNet(**kw)
    costs = model.route_costs.tolist()
    for eta in (0.0, 1.0):
        bud = budget_from_intensity(eta, costs)
        feasible = [r for r, c in zip(routes, costs) if c <= bud + 1e-8]
        print(f"eta={eta}: budget={bud:.4f} feasible={feasible}")

    full = BudgetConditionedAdaptiveF2FNet(
        **synthetic_budget_f2f_kwargs(
            node_size=n, output_len=h, forced_route=[3, 6, 12], route_selection_mode="forced"
        )
    )
    n_full = sum(p.numel() for p in full.parameters())
    print(f"forced_full_route_params: {n_full} ({n_full/1e6:.3f}M)")

    # Assertion: chain_resolutions must match forced_route
    bad = BudgetConditionedAdaptiveF2FNet(
        **synthetic_budget_f2f_kwargs(
            node_size=n, output_len=h, forced_route=[12], route_selection_mode="forced"
        )
    )
    orig_exec = bad._execute_route

    def _corrupt_exec(history_data, route):
        out = orig_exec(history_data, route)
        out = dict(out)
        out["chain_resolutions"] = [3, 6, 12]
        return out

    bad._execute_route = _corrupt_exec  # type: ignore[method-assign]
    raised = False
    try:
        with torch.no_grad():
            _ = bad(history_data=torch.randn(1, p, n, 4), train=False, return_all=True)
    except RuntimeError as exc:
        raised = True
        print(f"forced_assertion_ok: {exc}")
    if not raised:
        print("ERROR: expected forced-route assertion to fire")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
