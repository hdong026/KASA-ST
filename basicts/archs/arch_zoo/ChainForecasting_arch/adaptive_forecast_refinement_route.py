"""Adaptive Forecast Refinement Route net: gain controller + frozen F2F supernet.

Forward semantics::

    history
      -> existing shared F2F representation (Priority-B tap)
      -> ForecastRefinementGainController -> [g3, g6, g36]
      -> route scores
      -> eta constrains feasible set
      -> tolerance cheapest-near-best
      -> existing _execute_route
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from basicts.archs.arch_zoo.ChainForecasting_arch.budget_conditioned_adaptive_f2f import (
    BudgetConditionedAdaptiveF2FNet,
)
from basicts.archs.arch_zoo.ChainForecasting_arch.forecast_refinement_decision import (
    select_batch_routes_from_scores,
    select_routes_from_scores,
)
from basicts.archs.arch_zoo.ChainForecasting_arch.forecast_refinement_gain_controller import (
    ForecastRefinementGainController,
)
from basicts.archs.arch_zoo.ChainForecasting_arch.forecast_refinement_routes import (
    build_refinement_route_index_map,
    route_scores_from_gains,
)


_CTRL_DROP = (
    "controller_dim",
    "pooling_queries",
    "delta_abs",
    "lambda_abs",
    "lambda_center",
    "lambda_corr",
    "lambda_rank",
    "lambda_full",
    "rank_ignore_margin",
    "rank_temperature",
    "valid_oracle_file",
    "train_oracle_file",
)


class AdaptiveForecastRefinementRouteNet(BudgetConditionedAdaptiveF2FNet):
    """Frozen F2F supernet + sample-aware refinement-gain controller."""

    def __init__(self, **model_args):
        parent_args = dict(model_args)
        for k in _CTRL_DROP:
            parent_args.pop(k, None)
        super().__init__(**parent_args)

        self.delta_abs = float(model_args.get("delta_abs", 0.05))
        self.index_map = build_refinement_route_index_map(
            self.candidate_routes, self.output_len
        )
        # Probe shared feature dim with a dummy (CPU) — real devices handled at runtime.
        with torch.no_grad():
            probe = torch.zeros(
                1, self.input_len, self.node_size, max(self.input_dim, 3)
            )
            # Temporarily move probe through modules on their current device
            try:
                # Use embedding path without requiring full GPU init
                h0 = self.extract_pre_route_context(probe, detach=True)
                in_dim = int(h0.shape[-1])
            except Exception:
                # Fallback for init-order edge cases: d_d (+ d_spa)
                d_d = int(getattr(self.backbone, "d_d", 32))
                d_spa = int(getattr(self.backbone, "d_spa", 32))
                in_dim = d_d + (d_spa if getattr(self.backbone, "if_spatial", False) else 0)

        self.gain_controller = ForecastRefinementGainController(
            input_dim=in_dim,
            controller_dim=int(model_args.get("controller_dim", 128)),
            pooling_queries=int(model_args.get("pooling_queries", 4)),
        )
        # Freeze legacy hard planner; unused in this variant.
        for p in self.planner.parameters():
            p.requires_grad = False
        phase = str(model_args.get("training_phase", "refinement_controller")).lower()
        self.set_training_phase(phase)
        if phase in {"refinement_controller", "route_quality", "eval", "quality_estimator"}:
            self.freeze_backbone(True)

    def set_training_phase(self, phase: str) -> None:
        phase = str(phase).lower()
        allowed = {
            "supernet",
            "planner",
            "joint",
            "eval",
            "refinement_controller",
            "route_quality",
            "quality_estimator",
        }
        if phase not in allowed:
            raise ValueError(f"Unknown training_phase: {phase}")
        self.training_phase = phase
        if phase in {"refinement_controller", "route_quality", "quality_estimator", "eval"}:
            self.freeze_backbone(True)

    def freeze_backbone(self, freeze: bool = True) -> None:
        super().freeze_backbone(freeze)
        if freeze:
            self.backbone.eval()

    def estimate_refinement_gains(self, history: torch.Tensor) -> dict[str, Any]:
        """Gain path: shared F2F representation only (no eta)."""
        h_shared = self.extract_pre_route_context(history, detach=True)
        out = self.gain_controller(h_shared)
        scores = route_scores_from_gains(
            out["g3_hat"],
            out["g6_hat"],
            out["g36_hat"],
            index_map=self.index_map,
            n_routes=len(self.candidate_routes),
        )
        out["route_scores"] = scores
        out["shared_feature_shape"] = tuple(h_shared.shape)
        return out

    def _select_route_id(
        self,
        history: torch.Tensor,
        train: bool,
        intensity_override: float | torch.Tensor | None = None,
    ) -> dict[str, Any]:
        del train
        if self.forced_route is not None or self.route_selection_mode == "forced":
            return BudgetConditionedAdaptiveF2FNet._select_route_id(
                self, history, train=False, intensity_override=intensity_override
            )

        quality = self.estimate_refinement_gains(history)
        scores = quality["route_scores"]
        eta = (
            self.inference_intensity
            if intensity_override is None
            else intensity_override
        )
        batch_mode = (
            self.route_granularity == "batch" or self.route_selection_mode == "batch"
        )
        if batch_mode:
            decision = select_batch_routes_from_scores(
                scores, self.route_costs, eta, delta_abs=self.delta_abs
            )
        else:
            decision = select_routes_from_scores(
                scores, self.route_costs, eta, delta_abs=self.delta_abs
            )
            decision["batch_route_id"] = None

        selected = decision["selected_route_id"]
        # Preference logits for diagnostics only (not CE training target).
        logits = scores
        feas = decision["feasible_mask"]
        masked = logits.masked_fill(~feas, -1e9)
        probs = F.softmax(masked, dim=-1)
        device = history.device
        dtype = history.dtype
        expected_cost = (probs * self.route_costs.to(device=device, dtype=dtype)).sum(-1)
        return {
            "route_logits": logits,
            "masked_route_logits": masked,
            "route_probs": probs,
            "feasible_mask": feas,
            "near_best_mask": decision["near_best_mask"],
            "selected_route_id": selected,
            "proposed_route_id": selected,
            "selected_cost": decision["selected_cost"],
            "expected_cost": expected_cost,
            "budget": decision["budget"],
            "predicted_gains": quality["predicted_gains"],
            "route_scores": scores,
            "batch_route_id": decision.get("batch_route_id"),
            "batch_route_logits": decision.get("batch_mean_scores"),
        }

    def forward(
        self,
        history_data: torch.Tensor,
        future_data: torch.Tensor = None,
        batch_seen: int = 0,
        epoch: int = 0,
        train: bool = False,
        return_all: bool = False,
        return_intermediates: bool = False,
        sandwich_routes: list[list[int]] | None = None,
        oracle_route_id: torch.Tensor | None = None,
        inference_intensity_override: float | torch.Tensor | None = None,
        sample_indices: torch.Tensor | None = None,
        true_gains: torch.Tensor | None = None,
        **kwargs,
    ):
        del kwargs
        if (
            train
            and self.training_phase == "refinement_controller"
            and self.forced_route is None
            and sandwich_routes is None
        ):
            quality = self.estimate_refinement_gains(history_data)
            b = history_data.shape[0]
            dummy = history_data.new_zeros(
                b, self.output_len, self.node_size, self.output_dim
            )
            eta = (
                self.inference_intensity
                if inference_intensity_override is None
                else inference_intensity_override
            )
            decision = select_routes_from_scores(
                quality["route_scores"],
                self.route_costs,
                eta,
                delta_abs=self.delta_abs,
            )
            result = {
                "pred": dummy,
                "prediction": dummy,
                "controller_only": True,
                "predicted_gains": quality["predicted_gains"],
                "route_scores": quality["route_scores"],
                "true_gains": true_gains,
                "feasible_mask": decision["feasible_mask"],
                "near_best_mask": decision["near_best_mask"],
                "selected_route_id": decision["selected_route_id"],
                "proposed_route_id": decision["selected_route_id"],
                "executed_route_id": decision["selected_route_id"],
                "selected_cost": decision["selected_cost"],
                "budget": decision["budget"],
                "route_logits": quality["route_scores"],
                "masked_route_logits": quality["route_scores"].masked_fill(
                    ~decision["feasible_mask"], -1e9
                ),
                "chain_preds": [dummy],
                "chain_resolutions": [self.output_len],
                "candidate_routes": self.candidate_routes,
                "route_costs": self.route_costs,
                "training_phase": self.training_phase,
                "sample_indices": sample_indices,
                "diagnostics": {"mode": "refinement_controller_only"},
            }
            if return_all or return_intermediates:
                return result
            return result["pred"]

        if self.freeze_forecasting_backbone:
            self.backbone.eval()

        out = BudgetConditionedAdaptiveF2FNet.forward(
            self,
            history_data=history_data,
            future_data=future_data,
            batch_seen=batch_seen,
            epoch=epoch,
            train=train,
            return_all=True,
            return_intermediates=True,
            sandwich_routes=sandwich_routes,
            oracle_route_id=oracle_route_id,
            inference_intensity_override=inference_intensity_override,
            sample_indices=sample_indices,
        )
        # Parent now forwards plan extras; do NOT recompute controller.
        if isinstance(out, dict) and "proposed_route_id" not in out:
            out["proposed_route_id"] = out.get(
                "selected_route_id", out.get("executed_route_id")
            )
        if return_all or return_intermediates:
            return out
        return out["pred"] if isinstance(out, dict) else out

    def trainable_parameter_report(self) -> dict[str, Any]:
        trainable = []
        for name, p in self.named_parameters():
            if p.requires_grad:
                trainable.append(name)
        bad = [n for n in trainable if not n.startswith("gain_controller.")]
        if bad:
            raise RuntimeError(
                f"Non-controller trainable parameters found: {bad[:10]}"
            )
        return {
            "trainable_names": trainable,
            "total_params": sum(p.numel() for p in self.parameters()),
            "backbone_params": sum(p.numel() for p in self.backbone.parameters()),
            "controller_params": self.gain_controller.count_parameters(),
            "trainable_params": sum(
                p.numel() for p in self.parameters() if p.requires_grad
            ),
        }
