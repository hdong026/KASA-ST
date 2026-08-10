"""Budget-conditioned F2F with Route Quality Estimator (frozen supernet reuse)."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from basicts.archs.arch_zoo.ChainForecasting_arch.budget_conditioned_adaptive_f2f import (
    BudgetConditionedAdaptiveF2FNet,
)
from basicts.archs.arch_zoo.ChainForecasting_arch.budget_route_quality_estimator import (
    RouteQualityEstimator,
)
from basicts.archs.arch_zoo.ChainForecasting_arch.route_quality_decision import (
    select_batch_route_from_quality,
    select_route_ids_from_quality,
)


_RQ_DROP_KEYS = (
    "rq_d_model",
    "rq_temporal_layers",
    "rq_spatial_query_count",
    "rq_sample_embedding_dim",
    "rq_route_embedding_dim",
    "delta_abs",
    "delta_rel",
    "lambda_abs",
    "lambda_center",
    "lambda_rank",
    "lambda_list",
    "rank_ignore_margin",
    "rank_temperature",
    "list_temperature",
    "route_cost_source",
    "valid_oracle_file",
    "train_oracle_file",
)


class BudgetConditionedRouteQualityF2FNet(BudgetConditionedAdaptiveF2FNet):
    """Reuse verified F2F supernet execution; replace hard CE planner with RQE."""

    def __init__(self, **model_args):
        # Strip RQE-only keys before parent (parent already strips its own extras).
        parent_args = dict(model_args)
        for k in _RQ_DROP_KEYS:
            parent_args.pop(k, None)
        # Keep legacy planner module for ablation weight compatibility, but unused.
        super().__init__(**parent_args)

        self.delta_abs = float(model_args.get("delta_abs", 0.05))
        self.delta_rel = float(model_args.get("delta_rel", 0.0))
        self.route_quality_estimator = RouteQualityEstimator(
            input_dim=int(model_args.get("input_dim", self.input_dim)),
            d_model=int(model_args.get("rq_d_model", 128)),
            temporal_layers=int(model_args.get("rq_temporal_layers", 2)),
            spatial_query_count=int(model_args.get("rq_spatial_query_count", 4)),
            sample_embedding_dim=int(model_args.get("rq_sample_embedding_dim", 256)),
            route_embedding_dim=int(model_args.get("rq_route_embedding_dim", 64)),
            max_len=max(int(self.input_len), 64),
        )
        # Default: freeze forecasting backbone for quality-estimator phase.
        phase = str(model_args.get("training_phase", "route_quality")).lower()
        if phase in {"route_quality", "quality_estimator", "eval"}:
            self.freeze_backbone(True)
            self.backbone.eval()
        # Freeze legacy hard planner so it cannot leak gradients.
        for p in self.planner.parameters():
            p.requires_grad = False
        self.set_training_phase(phase)

    def set_training_phase(self, phase: str) -> None:
        phase = str(phase).lower()
        if phase not in {
            "supernet",
            "planner",
            "joint",
            "eval",
            "route_quality",
            "quality_estimator",
        }:
            raise ValueError(f"Unknown training_phase: {phase}")
        self.training_phase = phase
        if phase in {"route_quality", "quality_estimator"}:
            self.freeze_backbone(True)
            self.backbone.eval()

    def freeze_backbone(self, freeze: bool = True) -> None:
        super().freeze_backbone(freeze)
        if freeze:
            self.backbone.eval()

    def estimate_route_quality(self, history: torch.Tensor) -> dict[str, Any]:
        """Quality path: history + static route descriptors only (no eta)."""
        return self.route_quality_estimator(
            history,
            routes=self.candidate_routes,
            route_costs=self.route_costs,
            horizon=self.output_len,
        )

    def _select_route_id(
        self,
        history: torch.Tensor,
        train: bool,
        intensity_override: float | torch.Tensor | None = None,
    ) -> dict[str, Any]:
        del train  # quality decision is deterministic given losses + budget
        b = history.shape[0]
        device = history.device
        dtype = history.dtype

        if self.forced_route is not None or self.route_selection_mode == "forced":
            return super()._select_route_id(
                history, train=False, intensity_override=intensity_override
            )

        quality = self.estimate_route_quality(history)
        pred_losses = quality["predicted_route_losses"]
        eta = (
            self.inference_intensity
            if intensity_override is None
            else intensity_override
        )
        batch_mode = (
            self.route_granularity == "batch" or self.route_selection_mode == "batch"
        )
        if batch_mode:
            decision = select_batch_route_from_quality(
                pred_losses,
                self.route_costs,
                eta,
                delta_abs=self.delta_abs,
                delta_rel=self.delta_rel,
            )
        else:
            decision = select_route_ids_from_quality(
                pred_losses,
                self.route_costs,
                eta,
                delta_abs=self.delta_abs,
                delta_rel=self.delta_rel,
            )
            decision["batch_route_id"] = None

        # Provide CE-compatible placeholders for shared runner diagnostics.
        selected = decision["selected_route_id"]
        logits = -pred_losses  # lower loss => higher preference
        feas = decision["feasible_mask"]
        masked = logits.masked_fill(~feas, -1e9)
        probs = F.softmax(masked, dim=-1)
        expected_cost = (probs * self.route_costs.to(device=device, dtype=dtype)).sum(
            dim=-1
        )
        out = {
            "route_logits": logits,
            "masked_route_logits": masked,
            "route_probs": probs,
            "feasible_mask": feas,
            "near_best_mask": decision["near_best_mask"],
            "selected_route_id": selected,
            "selected_cost": decision["selected_cost"],
            "expected_cost": expected_cost,
            "budget": decision["budget"],
            "predicted_route_losses": pred_losses,
            "sample_difficulty": quality["sample_difficulty"],
            "route_residuals": quality["route_residuals"],
            "proposed_route_id": selected,
            "batch_route_id": decision.get("batch_route_id"),
            "batch_route_logits": (
                (-decision["batch_mean_predicted_losses"])
                if decision.get("batch_mean_predicted_losses") is not None
                else None
            ),
        }
        return out

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
        true_route_losses: torch.Tensor | None = None,
        **kwargs,
    ):
        del kwargs
        # Route-quality training: estimate only; never execute F2F routes.
        if (
            train
            and self.training_phase in {"route_quality", "quality_estimator"}
            and self.forced_route is None
            and sandwich_routes is None
        ):
            quality = self.estimate_route_quality(history_data)
            b = history_data.shape[0]
            device = history_data.device
            dtype = history_data.dtype
            dummy = history_data.new_zeros(
                b, self.output_len, self.node_size, self.output_dim
            )
            eta = (
                self.inference_intensity
                if inference_intensity_override is None
                else inference_intensity_override
            )
            decision = select_route_ids_from_quality(
                quality["predicted_route_losses"],
                self.route_costs,
                eta,
                delta_abs=self.delta_abs,
                delta_rel=self.delta_rel,
            )
            result = {
                "pred": dummy,
                "prediction": dummy,
                "planner_only": True,
                "route_quality_only": True,
                "predicted_route_losses": quality["predicted_route_losses"],
                "sample_difficulty": quality["sample_difficulty"],
                "route_residuals": quality["route_residuals"],
                "true_route_losses": true_route_losses,
                "feasible_mask": decision["feasible_mask"],
                "near_best_mask": decision["near_best_mask"],
                "selected_route_id": decision["selected_route_id"],
                "proposed_route_id": decision["selected_route_id"],
                "executed_route_id": decision["selected_route_id"],
                "selected_cost": decision["selected_cost"],
                "budget": decision["budget"],
                "expected_cost": decision["selected_cost"],
                "route_logits": -quality["predicted_route_losses"],
                "masked_route_logits": (-quality["predicted_route_losses"]).masked_fill(
                    ~decision["feasible_mask"], -1e9
                ),
                "route_probs": F.softmax(
                    (-quality["predicted_route_losses"]).masked_fill(
                        ~decision["feasible_mask"], -1e9
                    ),
                    dim=-1,
                ),
                "chain_preds": [dummy],
                "chain_resolutions": [self.output_len],
                "candidate_routes": self.candidate_routes,
                "route_costs": self.route_costs,
                "inference_intensity": self.inference_intensity,
                "training_phase": self.training_phase,
                "sample_indices": sample_indices,
                "diagnostics": {"mode": "route_quality_only"},
            }
            if return_all or return_intermediates:
                return result
            return result["pred"]

        # Keep backbone in eval when frozen during adaptive inference training flags.
        if self.freeze_forecasting_backbone:
            self.backbone.eval()

        out = super().forward(
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
        # Attach quality predictions for diagnostics even in inference.
        if isinstance(out, dict) and "predicted_route_losses" not in out:
            # Recompute only if selection path already has them from _select_route_id;
            # parent forward stores route_logits but not predicted losses.
            if out.get("diagnostics", {}).get("mode") != "sandwich":
                q = self.estimate_route_quality(history_data)
                out["predicted_route_losses"] = q["predicted_route_losses"]
                out["proposed_route_id"] = out.get(
                    "selected_route_id", out.get("executed_route_id")
                )
        if return_all or return_intermediates:
            return out
        return out["pred"] if isinstance(out, dict) else out

    def trainable_parameter_report(self) -> dict[str, Any]:
        trainable = []
        frozen = []
        for name, p in self.named_parameters():
            if p.requires_grad:
                trainable.append(name)
            else:
                frozen.append(name)
        est = sum(p.numel() for p in self.route_quality_estimator.parameters())
        return {
            "trainable_names": trainable,
            "frozen_count": sum(
                p.numel() for n, p in self.named_parameters() if not p.requires_grad
            ),
            "trainable_count": sum(
                p.numel() for n, p in self.named_parameters() if p.requires_grad
            ),
            "estimator_param_count": est,
            "backbone_param_count": sum(p.numel() for p in self.backbone.parameters()),
        }
