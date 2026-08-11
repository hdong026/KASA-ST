"""Sequential F2F environment for Group-Relative Refinement Policy (Plan B)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from basicts.archs.arch_zoo.ChainForecasting_arch.forecast_refinement_routes import (
    build_refinement_route_index_map,
    route_key,
    standard_refinement_route_template,
)
from basicts.archs.arch_zoo.ChainForecasting_arch.budget_route_utils import (
    budget_from_intensity,
)
from basicts.archs.arch_zoo.ChainForecasting_arch.route_quality_decision import (
    feasible_mask_from_budget,
)


# Action ids
A0_DIRECT, A0_HALF, A0_QUARTER = 0, 1, 2
A1_JUMP_FINAL, A1_REFINE_HALF = 0, 1


class SequentialF2FEnvironment:
    """Enumerate / execute sequential refinement trajectories on a frozen supernet."""

    def __init__(self, model):
        self.model = model
        self.horizon = int(model.output_len)
        self.index_map = build_refinement_route_index_map(
            model.candidate_routes, self.horizon
        )
        self.template = standard_refinement_route_template(self.horizon)
        self.costs = model.route_costs

    def execute_full_route(self, history: torch.Tensor, route: list[int]) -> dict[str, Any]:
        return self.model._execute_route(history, route)

    def execute_quarter_prefix(self, history: torch.Tensor) -> dict[str, Any]:
        q = list(self.template["quarter"][:1])  # [H/4]
        out = self.model._execute_route(history, q, allow_prefix=True)
        return {
            "Z_q": out["pred"],  # explicit coarse forecast at H/4
            "prev_forecast": out["chain_preds"][-1],
            "chain_preds": out["chain_preds"],
            "chain_resolutions": out["chain_resolutions"],
        }

    def resume_quarter_to_final(
        self, history: torch.Tensor, prev_forecast: torch.Tensor
    ) -> dict[str, Any]:
        # suffix of quarter route after H/4: [H]
        suffix = [self.horizon]
        return self.model._execute_route(
            history, suffix, init_prev_forecast=prev_forecast
        )

    def resume_quarter_to_progressive(
        self, history: torch.Tensor, prev_forecast: torch.Tensor
    ) -> dict[str, Any]:
        # suffix of progressive after H/4: [H/2, H]
        suffix = [self.horizon // 2, self.horizon]
        return self.model._execute_route(
            history, suffix, init_prev_forecast=prev_forecast
        )

    def sequential_route_equivalence_check(
        self, history: torch.Tensor, atol: float = 1e-6
    ) -> dict[str, Any]:
        """Compare sequential resume vs original _execute_route."""
        report = {}
        # [H/4, H]
        full_q = self.execute_full_route(history, list(self.template["quarter"]))
        pref = self.execute_quarter_prefix(history)
        resume_q = self.resume_quarter_to_final(history, pref["prev_forecast"])
        # resume final pred is at H; full_q pred at H — compare
        d_q = float((full_q["pred"] - resume_q["pred"]).abs().max().item())
        report["quarter_max_abs_diff"] = d_q
        report["quarter_ok"] = d_q < atol

        full_p = self.execute_full_route(history, list(self.template["progressive"]))
        resume_p = self.resume_quarter_to_progressive(history, pref["prev_forecast"])
        d_p = float((full_p["pred"] - resume_p["pred"]).abs().max().item())
        report["progressive_max_abs_diff"] = d_p
        report["progressive_ok"] = d_p < atol
        return report

    def action_masks(self, eta: float) -> dict[str, torch.Tensor]:
        """Hard budget masks for policy0 (3) and policy1 (2)."""
        costs = self.costs.detach().float().cpu()
        bval = budget_from_intensity(float(eta), costs.tolist())
        feas = feasible_mask_from_budget(costs, torch.tensor([bval])).squeeze(0)
        # terminal route feasibility by semantic name
        f_direct = bool(feas[self.index_map["direct"]])
        f_half = bool(feas[self.index_map["half"]])
        f_quarter = bool(feas[self.index_map["quarter"]])
        f_prog = bool(feas[self.index_map["progressive"]])
        # a0: DIRECT / HALF / QUARTER — QUARTER allowed if quarter OR progressive feasible
        # (progressive needs quarter prefix)
        m0 = torch.tensor(
            [f_direct, f_half, f_quarter or f_prog], dtype=torch.bool
        )
        # a1 only if quarter opened: JUMP if quarter feasible; REFINE if progressive
        m1 = torch.tensor([f_quarter, f_prog], dtype=torch.bool)
        return {
            "mask0": m0,
            "mask1": m1,
            "feasible_routes": feas,
            "budget": bval,
        }

    def route_from_actions(self, a0: int, a1: int | None) -> list[int]:
        if a0 == A0_DIRECT:
            return list(self.template["direct"])
        if a0 == A0_HALF:
            return list(self.template["half"])
        if a0 == A0_QUARTER:
            if a1 is None:
                raise ValueError("quarter branch requires a1")
            if a1 == A1_JUMP_FINAL:
                return list(self.template["quarter"])
            if a1 == A1_REFINE_HALF:
                return list(self.template["progressive"])
        raise ValueError(f"unknown actions a0={a0} a1={a1}")

    def enumerate_feasible_trajectories(self, eta: float) -> list[dict[str, Any]]:
        masks = self.action_masks(eta)
        trajs = []
        if masks["mask0"][A0_DIRECT]:
            trajs.append({"a0": A0_DIRECT, "a1": None, "route": self.route_from_actions(A0_DIRECT, None),
                          "route_id": self.index_map["direct"]})
        if masks["mask0"][A0_HALF]:
            trajs.append({"a0": A0_HALF, "a1": None, "route": self.route_from_actions(A0_HALF, None),
                          "route_id": self.index_map["half"]})
        if masks["mask0"][A0_QUARTER]:
            if masks["mask1"][A1_JUMP_FINAL]:
                trajs.append({"a0": A0_QUARTER, "a1": A1_JUMP_FINAL,
                              "route": self.route_from_actions(A0_QUARTER, A1_JUMP_FINAL),
                              "route_id": self.index_map["quarter"]})
            if masks["mask1"][A1_REFINE_HALF]:
                trajs.append({"a0": A0_QUARTER, "a1": A1_REFINE_HALF,
                              "route": self.route_from_actions(A0_QUARTER, A1_REFINE_HALF),
                              "route_id": self.index_map["progressive"]})
        return trajs
