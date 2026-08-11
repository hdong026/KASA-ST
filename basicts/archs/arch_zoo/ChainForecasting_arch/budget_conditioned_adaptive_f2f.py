"""Budget-conditioned adaptive F2FNet: planner selects routes; KASA stages execute them."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from basicts.archs.arch_zoo.ChainForecasting_arch.ChainForecasting_arch import (
    ChainForecasting,
)
from basicts.archs.arch_zoo.ChainForecasting_arch.budget_route_planner import (
    BudgetRoutePlanner,
    history_route_features,
)
from basicts.archs.arch_zoo.ChainForecasting_arch.budget_route_utils import (
    budget_from_intensity,
    budgets_from_intensity_tensor,
    default_candidate_routes,
    load_route_costs,
    parse_candidate_routes,
    parse_route,
    unique_resolutions,
    validate_route,
)
from basicts.archs.arch_zoo.ChainForecasting_arch.kasa_temporal_step import (
    interpolate_forecast,
)


class BudgetConditionedAdaptiveF2FNet(nn.Module):
    """One-shot route planner over a shared full-route ChainForecasting supernet.

    All candidate resolutions share the same ``KASATemporalStep`` / progressive
    spatial modules as a formal ``[H/4,H/2,H]`` ChainForecasting. Selected routes
    only choose which subset of stages to run.
    """

    def __init__(self, **model_args):
        super().__init__()
        self.output_len = int(model_args["output_len"])
        self.input_len = int(model_args["input_len"])
        self.node_size = int(model_args["node_size"])
        self.input_dim = int(model_args.get("input_dim", 3))
        self.output_dim = int(model_args.get("output_dim", 1))

        # Candidate routes
        raw_routes = model_args.get("candidate_routes")
        if raw_routes is None:
            self.candidate_routes = default_candidate_routes(self.output_len)
        else:
            self.candidate_routes = parse_candidate_routes(raw_routes, self.output_len)
        for r in self.candidate_routes:
            validate_route(r, horizon=self.output_len)

        full_route = unique_resolutions(self.candidate_routes)
        if full_route[-1] != self.output_len:
            raise ValueError("full supernet route must end at H")
        # Supernet always uses sorted unique resolutions as chain_lengths
        backbone_args = dict(model_args)
        backbone_args["chain_lengths"] = list(full_route)
        # Ensure formal F2F defaults for spatial / adapter
        backbone_args.setdefault("use_prev_condition", True)
        backbone_args.setdefault("spatial_placement", "interleaved_progressive")
        backbone_args.setdefault(
            "progressive_spatial_ratios", [0.25, 0.5, 1.0]
        )
        backbone_args.setdefault("progressive_spatial_topks", [8, 16, 32])
        backbone_args.setdefault("progressive_spatial_alphas", [0.03, 0.06, 0.10])
        backbone_args.setdefault("use_forecast_state_adapter", True)
        backbone_args.setdefault("forecast_state_adapter_mode", "condition_only")
        backbone_args.setdefault("forecast_state_adapter_hidden_dim", 16)
        backbone_args.setdefault("forecast_state_adapter_epsilon", 0.02)
        backbone_args.setdefault("post_spatial_mode", "adaptive_only")
        backbone_args.setdefault("use_adaptive_adj", True)
        backbone_args.setdefault("architecture_mode", "chain")

        # Strip budget-only keys so ChainForecasting init is not polluted.
        for drop in (
            "candidate_routes",
            "forced_route",
            "route_selection_mode",
            "route_granularity",
            "route_cost_type",
            "route_cost_file",
            "inference_intensity",
            "planner_hidden_dim",
            "planner_training_intensities",
            "training_phase",
            "loss_mode",
            "route_sampling",
            "freeze_forecasting_backbone",
            "run_signature",
            "experiment_tag",
            "oracle_file",
            "init_checkpoint",
            "backbone_lr",
            "planner_lr",
            "num_epochs",
            "lambda_mid",
            "lambda_imitation",
            "lambda_budget",
            "chain_loss_mode",
            "chain_loss_weights",
            "model_arch",
            "model_name",
            "is_chain",
        ):
            backbone_args.pop(drop, None)

        self._progressive_ratios = list(
            backbone_args.get("progressive_spatial_ratios", [0.25, 0.5, 1.0])
        )
        self._progressive_topks = list(
            backbone_args.get("progressive_spatial_topks", [8, 16, 32])
        )
        self._progressive_alphas = list(
            backbone_args.get("progressive_spatial_alphas", [0.03, 0.06, 0.10])
        )

        self.backbone = ChainForecasting(**backbone_args)
        self.full_resolutions = list(full_route)
        self.res_to_index = {int(k): i for i, k in enumerate(self.full_resolutions)}

        # Costs / intensity
        self.inference_intensity = float(model_args.get("inference_intensity", 0.5))
        self.route_selection_mode = str(
            model_args.get("route_selection_mode", "batch")
        ).lower()
        if self.route_selection_mode not in {"batch", "sample", "forced"}:
            raise ValueError(
                f"Unsupported route_selection_mode: {self.route_selection_mode}"
            )
        self.route_granularity = str(model_args.get("route_granularity", "batch")).lower()
        cost_type = str(model_args.get("route_cost_type", "normalized_static_cost"))
        costs = load_route_costs(
            model_args.get("route_cost_file"),
            self.candidate_routes,
            self.output_len,
            cost_type=cost_type,
        )
        self.register_buffer(
            "route_costs",
            torch.tensor(costs, dtype=torch.float32),
            persistent=False,
        )

        # Planner
        feat_dim = history_route_features(
            torch.zeros(1, self.input_len, self.node_size, max(self.input_dim, 3))
        ).shape[-1]
        self.planner = BudgetRoutePlanner(
            feat_dim=feat_dim,
            n_routes=len(self.candidate_routes),
            hidden_dim=int(model_args.get("planner_hidden_dim", 64)),
        )
        self.training_phase = str(model_args.get("training_phase", "supernet")).lower()
        self.forced_route = None
        if model_args.get("forced_route") is not None:
            self.forced_route = parse_route(model_args["forced_route"])
            validate_route(self.forced_route, horizon=self.output_len)
            if self.forced_route not in self.candidate_routes:
                raise ValueError(
                    f"forced_route {self.forced_route} not in candidate pool "
                    f"{self.candidate_routes}"
                )
        self.loss_mode = str(model_args.get("loss_mode", "dynamic_fair")).lower()
        if self.loss_mode not in {"baseline_compatible", "dynamic_fair"}:
            raise ValueError(f"Unsupported loss_mode: {self.loss_mode}")
        self.route_sampling = str(model_args.get("route_sampling", "sandwich")).lower()
        self.freeze_forecasting_backbone = bool(
            model_args.get("freeze_forecasting_backbone", False)
        )
        self._logged = False
        if self.freeze_forecasting_backbone:
            self.freeze_backbone(True)

    def freeze_backbone(self, freeze: bool = True) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = not freeze
        self.freeze_forecasting_backbone = bool(freeze)

    def set_forced_route(self, route: list[int] | None) -> None:
        if route is None:
            self.forced_route = None
            return
        r = parse_route(route)
        validate_route(r, horizon=self.output_len)
        if r not in self.candidate_routes:
            raise ValueError(f"forced_route {r} not in candidates")
        self.forced_route = r

    def set_training_phase(self, phase: str) -> None:
        phase = str(phase).lower()
        if phase not in {"supernet", "planner", "joint", "eval"}:
            raise ValueError(f"Unknown training_phase: {phase}")
        self.training_phase = phase

    def _budget(
        self,
        batch: int,
        device,
        dtype,
        intensity_override: float | torch.Tensor | None = None,
    ) -> torch.Tensor:
        costs = self.route_costs.tolist()
        if intensity_override is None:
            bval = budget_from_intensity(self.inference_intensity, costs)
            return torch.full((batch,), bval, device=device, dtype=dtype)
        if torch.is_tensor(intensity_override):
            etas = intensity_override.to(device=device, dtype=dtype)
            if etas.ndim == 0:
                etas = etas.reshape(1).expand(batch)
            elif etas.numel() == 1:
                etas = etas.reshape(1).expand(batch)
            elif etas.shape[0] != batch:
                raise ValueError(
                    f"intensity_override batch {etas.shape[0]} != history batch {batch}"
                )
            return budgets_from_intensity_tensor(etas.reshape(-1), costs).to(
                device=device, dtype=dtype
            )
        bval = budget_from_intensity(float(intensity_override), costs)
        return torch.full((batch,), bval, device=device, dtype=dtype)

    def _select_route_id(
        self,
        history: torch.Tensor,
        train: bool,
        intensity_override: float | torch.Tensor | None = None,
    ) -> dict[str, Any]:
        b = history.shape[0]
        device = history.device
        dtype = history.dtype
        budget = self._budget(b, device, dtype, intensity_override=intensity_override)

        if self.forced_route is not None or self.route_selection_mode == "forced":
            if self.forced_route is None:
                raise ValueError("forced mode requires forced_route")
            rid = self.candidate_routes.index(self.forced_route)
            selected = torch.full((b,), rid, device=device, dtype=torch.long)
            costs = self.route_costs.to(device=device, dtype=dtype)
            return {
                "route_logits": torch.zeros(
                    b, len(self.candidate_routes), device=device, dtype=dtype
                ),
                "masked_route_logits": torch.zeros(
                    b, len(self.candidate_routes), device=device, dtype=dtype
                ),
                "route_probs": F.one_hot(
                    selected, num_classes=len(self.candidate_routes)
                ).to(dtype),
                "feasible_mask": torch.ones(
                    b, len(self.candidate_routes), device=device, dtype=torch.bool
                ),
                "selected_route_id": selected,
                "selected_cost": costs[rid].expand(b),
                "expected_cost": costs[rid].expand(b),
                "budget": budget,
                "batch_route_id": rid,
                "batch_route_logits": torch.zeros(
                    len(self.candidate_routes), device=device, dtype=dtype
                ),
            }

        intensity = (
            self.inference_intensity
            if intensity_override is None
            else intensity_override
        )
        plan = self.planner(
            history,
            intensity=intensity,
            route_costs=self.route_costs,
            budget=budget,
            deterministic=not train or self.training_phase == "eval",
        )
        costs = self.route_costs.to(device=device, dtype=dtype)
        batch_mode = (
            self.route_granularity == "batch" or self.route_selection_mode == "batch"
        )
        if batch_mode:
            mean_masked_logits = plan["masked_route_logits"].mean(dim=0)
            mean_budget = budget.mean()
            feasible_batch = costs <= mean_budget
            cheapest = int(costs.argmin().item())
            feasible_batch = feasible_batch | F.one_hot(
                torch.tensor(cheapest, device=device),
                num_classes=len(self.candidate_routes),
            ).bool()
            batch_masked = mean_masked_logits.masked_fill(~feasible_batch, -1e9)
            batch_id = int(batch_masked.argmax().item())
            plan["batch_route_id"] = batch_id
            plan["batch_route_logits"] = mean_masked_logits
            plan["selected_route_id"] = torch.full(
                (b,), batch_id, device=device, dtype=torch.long
            )
            plan["selected_cost"] = costs[batch_id].expand(b)
        else:
            plan["batch_route_id"] = None
            plan["batch_route_logits"] = None
        return plan

    def _execute_routes_bucketed(
        self,
        history: torch.Tensor,
        route_ids: torch.Tensor,
    ) -> dict[str, Any]:
        """Execute per-sample routes by grouping indices with the same route id."""
        b = history.shape[0]
        device = history.device
        dtype = history.dtype
        executed_route_id = route_ids.long().view(-1)
        if executed_route_id.shape[0] != b:
            raise ValueError(
                f"route_ids batch {executed_route_id.shape[0]} != history batch {b}"
            )

        costs = self.route_costs.to(device=device, dtype=dtype)
        selected_cost = costs.gather(0, executed_route_id)
        unique_ids = executed_route_id.unique().tolist()

        if len(unique_ids) == 1:
            rid = int(unique_ids[0])
            route = list(self.candidate_routes[rid])
            out = self._execute_route(history, route)
            return {
                "pred": out["pred"],
                "chain_preds": out["chain_preds"],
                "chain_resolutions": out["chain_resolutions"],
                "executed_route_id": executed_route_id,
                "selected_cost": selected_cost,
                "executed_routes": [route],
            }

        pred = history.new_zeros(b, self.output_len, self.node_size, self.output_dim)
        executed_routes: list[list[int]] = []
        for rid in unique_ids:
            rid = int(rid)
            route = list(self.candidate_routes[rid])
            executed_routes.append(route)
            idx = (executed_route_id == rid).nonzero(as_tuple=True)[0]
            sub = self._execute_route(history[idx], route)
            pred[idx] = sub["pred"]

        return {
            "pred": pred,
            "chain_preds": [pred],
            "chain_resolutions": [self.output_len],
            "executed_route_id": executed_route_id,
            "selected_cost": selected_cost,
            "executed_routes": executed_routes,
        }

    def _spatial_index_for_resolution(self, res: int) -> int:
        """Map temporal resolution to progressive spatial module by capacity tier."""
        # Align with supernet stage index of this resolution in full_resolutions
        if res not in self.res_to_index:
            raise KeyError(f"resolution {res} not in supernet {self.full_resolutions}")
        return self.res_to_index[res]

    def extract_pre_route_context(
        self,
        history_data: torch.Tensor,
        *,
        detach: bool = True,
    ) -> torch.Tensor:
        """Read-only shared forecasting representation before route execution.

        Priority-B tap: reuses the horizon stage ``KASATemporalStep.patch_encoder``
        (shared backbone weights; no new encoder). Does not call ``_execute_route``
        and does not alter default forecasting numerics.

        Returns:
            H_shared with shape ``[B, M, N, D]`` where
            ``D = d_d (+ d_spa if spatial codebook enabled)``.
        """
        from math import ceil

        if int(self.output_len) not in self.res_to_index:
            raise RuntimeError(
                f"horizon H={self.output_len} missing from supernet resolutions "
                f"{self.full_resolutions}"
            )
        step = self.backbone.temporal_steps[self.res_to_index[int(self.output_len)]]
        pe = step.patch_encoder
        x_main = history_data[..., :3]
        in_len_add = ceil(1.0 * step.input_len / step.stride) * step.stride - step.input_len
        if in_len_add:
            step_input = torch.cat(
                (x_main[:, -1:, :, :].expand(-1, in_len_add, -1, -1), x_main),
                dim=1,
            )
        else:
            step_input = x_main
        patch_input = step_input.unfold(
            dimension=1, size=step.patch_len, step=step.patch_len
        ).permute(0, 1, 4, 2, 3)
        # Embedding path only (no forecast projection1).
        if pe.patch_embedding_mode == "serial_concat":
            data_emb = pe._embed_serial_concat(patch_input)
        elif pe.patch_embedding_mode == "time_feature_2d":
            data_emb = pe._embed_time_feature_2d(patch_input)
        else:
            raise ValueError(
                f"Unsupported patch_embedding_mode for context tap: {pe.patch_embedding_mode}"
            )
        data_emb = pe.data_encoder(data_emb.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        # [B, M, N, d_d]
        spatial_codebook = self.backbone._spatial_codebook()
        if spatial_codebook is not None and pe.if_spatial:
            b, m, n, _ = data_emb.shape
            spa = spatial_codebook.unsqueeze(0).unsqueeze(1).expand(b, m, n, -1)
            h_shared = torch.cat([data_emb, spa], dim=-1)
        else:
            h_shared = data_emb
        if detach:
            h_shared = h_shared.detach()
        return h_shared

    def _execute_route(
        self,
        history_data: torch.Tensor,
        route: list[int],
        return_trace: bool = False,
        init_prev_forecast: torch.Tensor | None = None,
        allow_prefix: bool = False,
    ) -> dict[str, Any]:
        """Run KASA stages for ``route`` using shared supernet modules.

        ``init_prev_forecast`` enables sequential resume: pass the last forecast
        state from a prefix execution so a suffix route continues conditioning
        without re-running earlier stages. Default ``None`` preserves original
        full-route semantics.

        ``allow_prefix=True`` permits routes whose last resolution is not H
        (Plan B quarter-prefix only). Default ``False`` keeps the original
        final-length == H assertion for all full terminal routes.
        """
        chain_preds: list[torch.Tensor] = []
        temporal_preds: list[torch.Tensor] = []
        conditions: list[torch.Tensor | None] = []
        spatial_cfgs: list[dict[str, Any]] = []
        prev_forecast = init_prev_forecast
        bb = self.backbone
        spatial_codebook = bb._spatial_codebook()

        for stage_i, res in enumerate(route):
            idx = self.res_to_index[int(res)]
            step = bb.temporal_steps[idx]
            is_last = stage_i == len(route) - 1
            previous_state = prev_forecast
            prev_up = None
            if (
                previous_state is not None
                and bb.use_prev_condition
                and bb.propagation_mode == "forecast_state"
            ):
                prev_up = interpolate_forecast(previous_state, int(res))

            stage_ratio = float(res) / float(self.output_len)
            router_kwargs = {}
            if bb.light_spectral_router is not None:
                router_kwargs = {
                    "branch_coefficients": bb.light_spectral_router(
                        history_data[..., 0], stage_ratio
                    )
                }
            elif bb.spectral_branch_router is not None:
                router_kwargs = {
                    "spectral_router": bb.spectral_branch_router,
                    "stage_ratio": stage_ratio,
                }

            t_k = step(
                history_data,
                prev_forecast=prev_up,
                spatial_codebook=spatial_codebook,
                **router_kwargs,
            )
            temporal_preds.append(t_k)
            # Progressive spatial by resolution tier (not route stage index)
            if bb.spatial_placement == "interleaved_progressive":
                z_raw = bb._apply_progressive_spatial_refine(
                    t_k, history_data, self._spatial_index_for_resolution(int(res))
                )
            else:
                z_raw = t_k

            # Match formal ChainForecasting condition_only path exactly:
            # adapter runs only at supernet step_idx == 1 (see ChainForecasting_arch.py
            # `_forward_chain`: `if step_idx == 1`). For default [3,6,12] this is res 6.
            supervised_state = z_raw
            next_prev = z_raw
            adapter_used = False
            if (
                bb.forecast_state_adapter is not None
                and previous_state is not None
                and bb.forecast_state_adapter_mode == "condition_only"
                and idx == 1
            ):
                next_prev = bb.forecast_state_adapter(
                    current_state=z_raw,
                    previous_state=previous_state,
                    stage_ratio=stage_ratio,
                )
                adapter_used = True
            chain_preds.append(supervised_state)
            conditions.append(next_prev if not is_last else None)
            prev_forecast = next_prev

            if return_trace and bb.spatial_placement == "interleaved_progressive":
                mod = bb.progressive_spatial_modules[idx]
                spatial_cfgs.append(
                    {
                        "resolution": int(res),
                        "supernet_index": idx,
                        "ratio": float(self._progressive_ratios[idx])
                        if idx < len(self._progressive_ratios)
                        else None,
                        "configured_topk": int(self._progressive_topks[idx])
                        if idx < len(self._progressive_topks)
                        else None,
                        "configured_alpha": float(self._progressive_alphas[idx])
                        if idx < len(self._progressive_alphas)
                        else None,
                        "module_class": type(mod).__name__,
                        "dyn_hidden_dim": getattr(mod, "dyn_hidden_dim", None),
                        "adp_hidden_dim": getattr(mod, "adp_hidden_dim", None),
                        "dyn_topk": getattr(mod, "dyn_topk", None),
                        "adp_topk": getattr(mod, "adp_topk", None),
                        "dyn_alpha": getattr(mod, "dyn_alpha", None),
                        "adp_alpha": getattr(mod, "adp_alpha", None),
                        "hybrid_alpha": getattr(mod, "hybrid_alpha", None),
                        "adapter_used": adapter_used,
                        "condition_present": prev_up is not None,
                        "step_input_channels": (
                            "cond_encoders"
                            if prev_up is not None
                            else "base_encoders_only"
                        ),
                    }
                )

        if not chain_preds:
            raise RuntimeError("empty route execution")
        final = chain_preds[-1]
        if (not allow_prefix) and final.shape[1] != self.output_len:
            raise RuntimeError(
                f"final stage length {final.shape[1]} != H={self.output_len}"
            )
        if final.shape[2] != self.node_size:
            raise RuntimeError("node dimension changed during route execution")
        out = {
            "pred": final,
            "chain_preds": chain_preds,
            "chain_resolutions": list(route),
            "temporal_preds": temporal_preds,
            "downstream_conditions": conditions,
        }
        if return_trace:
            out["spatial_cfgs"] = spatial_cfgs
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
        **kwargs,
    ):
        del batch_seen, epoch, kwargs, future_data, sample_indices
        history = history_data
        b = history.shape[0]
        device = history.device
        dtype = history.dtype

        forced_active = (
            self.forced_route is not None or self.route_selection_mode == "forced"
        )
        if forced_active and sandwich_routes is not None:
            raise RuntimeError(
                "forced-route mode received sandwich_routes; "
                "forced must override sandwich/random sampling"
            )

        if sandwich_routes is not None and not forced_active:
            outs = []
            for route in sandwich_routes:
                validate_route(route, horizon=self.output_len)
                outs.append(self._execute_route(history, route))
            max_route = max(sandwich_routes, key=lambda r: (len(r), sum(r)))
            primary = next(
                out
                for route, out in zip(sandwich_routes, outs)
                if list(route) == list(max_route)
            )
            max_rid = self.candidate_routes.index(list(max_route))
            result = {
                "pred": primary["pred"],
                "prediction": primary["pred"],
                "chain_preds": primary["chain_preds"],
                "chain_resolutions": primary["chain_resolutions"],
                "sandwich_outputs": outs,
                "selected_route": list(max_route),
                "selected_route_id": torch.full(
                    (b,), max_rid, device=device, dtype=torch.long
                ),
                "executed_route_id": torch.full(
                    (b,), max_rid, device=device, dtype=torch.long
                ),
                "route_logits": None,
                "masked_route_logits": None,
                "route_probs": None,
                "expected_cost": None,
                "selected_cost": self.route_costs[max_rid].to(
                    device=device, dtype=dtype
                ).expand(b),
                "budget": self._budget(
                    b, device, dtype, intensity_override=inference_intensity_override
                ),
                "diagnostics": {"mode": "sandwich"},
            }
            if return_all or return_intermediates:
                return result
            return result["pred"]

        if (
            train
            and self.training_phase == "joint"
            and self.route_granularity == "sample"
            and not forced_active
        ):
            raise RuntimeError(
                "joint training with sample route granularity is not supported "
                "for forecasting; use batch granularity or planner-only phase"
            )

        plan = self._select_route_id(
            history,
            train=train,
            intensity_override=inference_intensity_override,
        )

        if (
            self.training_phase == "planner"
            and oracle_route_id is not None
            and not forced_active
        ):
            oracle = oracle_route_id.long().view(-1)
            if oracle.numel() == 1 and b > 1:
                oracle = oracle.expand(b)
            elif oracle.numel() != b:
                raise ValueError(
                    f"oracle_route_id size {oracle.numel()} != batch {b}"
                )
            feas = plan["feasible_mask"]
            for bi in range(b):
                if not bool(feas[bi, oracle[bi]].item()):
                    raise RuntimeError(
                        f"oracle route {int(oracle[bi].item())} infeasible for "
                        f"sample {bi} under budget {float(plan['budget'][bi].item())}"
                    )
            dummy_pred = history.new_zeros(
                b, self.output_len, self.node_size, self.output_dim
            )
            result = {
                "pred": dummy_pred,
                "prediction": dummy_pred,
                "planner_only": True,
                "route_logits": plan["route_logits"],
                "masked_route_logits": plan["masked_route_logits"],
                "route_probs": plan["route_probs"],
                "feasible_mask": plan["feasible_mask"],
                "oracle_route_id": oracle.to(device=device),
                "expected_cost": plan["expected_cost"],
                "selected_cost": plan["selected_cost"],
                "budget": plan["budget"],
                "selected_route_id": plan["selected_route_id"],
                "executed_route_id": oracle.to(device=device),
                "chain_preds": [dummy_pred],
                "chain_resolutions": [self.output_len],
                "candidate_routes": self.candidate_routes,
                "route_costs": self.route_costs,
                "inference_intensity": self.inference_intensity,
                "training_phase": self.training_phase,
                "loss_mode": self.loss_mode,
                "diagnostics": {"mode": "planner_only"},
            }
            if plan.get("batch_route_logits") is not None:
                result["batch_route_logits"] = plan["batch_route_logits"]
            if return_all or return_intermediates:
                return result
            return result["pred"]

        use_oracle_exec = (
            oracle_route_id is not None
            and self.training_phase in {"planner", "joint"}
            and not forced_active
        )
        batch_mode = (
            self.route_granularity == "batch" or self.route_selection_mode == "batch"
        )

        if use_oracle_exec:
            exec_ids = oracle_route_id.long().view(-1)
            if exec_ids.numel() == 1 and b > 1:
                exec_ids = exec_ids.expand(b)
            elif exec_ids.numel() != b:
                raise ValueError(
                    f"oracle_route_id size {exec_ids.numel()} != batch {b}"
                )
            executed = self._execute_routes_bucketed(history, exec_ids)
            route_repr = list(self.candidate_routes[int(exec_ids[0].item())])
        elif batch_mode:
            rid = int(plan["batch_route_id"])
            route_repr = list(self.candidate_routes[rid])
            executed = self._execute_route(history, route_repr)
            exec_ids = torch.full((b,), rid, device=device, dtype=torch.long)
            executed["executed_route_id"] = exec_ids
            executed["selected_cost"] = self.route_costs.to(device=device, dtype=dtype)[
                rid
            ].expand(b)
            executed["executed_routes"] = [route_repr]
        else:
            exec_ids = plan["selected_route_id"]
            executed = self._execute_routes_bucketed(history, exec_ids)
            route_repr = list(
                self.candidate_routes[int(exec_ids[0].item())]
            )

        if forced_active:
            if self.forced_route is None:
                raise RuntimeError("forced mode requires forced_route")
            forced_rid = self.candidate_routes.index(self.forced_route)
            if not torch.all(exec_ids == forced_rid):
                raise RuntimeError(
                    f"executed_route_id {exec_ids.tolist()} != forced route id {forced_rid}"
                )
            route_repr = list(self.forced_route)
            if list(executed["chain_resolutions"]) != list(self.forced_route):
                if len(executed.get("executed_routes", [])) == 1:
                    if list(executed["executed_routes"][0]) != list(self.forced_route):
                        raise RuntimeError(
                            f"chain_resolutions {executed['chain_resolutions']} "
                            f"!= forced_route {self.forced_route}"
                        )
                elif list(executed["chain_resolutions"]) != [self.output_len]:
                    raise RuntimeError(
                        f"chain_resolutions {executed['chain_resolutions']} "
                        f"!= forced_route {self.forced_route}"
                    )
            if executed["pred"].shape[1] != self.output_len:
                raise RuntimeError(
                    f"final forecast length {executed['pred'].shape[1]} "
                    f"!= horizon {self.output_len}"
                )
            if not self._logged:
                print(
                    "[budget_f2f forced] "
                    f"executed_routes={[list(route_repr)]} "
                    f"chain_resolutions={list(executed['chain_resolutions'])} "
                    f"actual_stage_count={len(executed['chain_resolutions'])}"
                )
                self._logged = True

        result = {
            "pred": executed["pred"],
            "prediction": executed["pred"],
            "chain_preds": executed["chain_preds"],
            "chain_resolutions": executed["chain_resolutions"],
            "selected_route_id": plan["selected_route_id"],
            "executed_route_id": executed["executed_route_id"],
            "selected_route": route_repr,
            "actual_stage_count": len(executed["chain_resolutions"]),
            "executed_routes": executed.get(
                "executed_routes", [list(route_repr)]
            ),
            "route_logits": plan["route_logits"],
            "masked_route_logits": plan["masked_route_logits"],
            "route_probs": plan["route_probs"],
            "feasible_mask": plan["feasible_mask"],
            "selected_cost": executed["selected_cost"],
            "expected_cost": plan["expected_cost"],
            "budget": plan["budget"],
            "candidate_routes": self.candidate_routes,
            "route_costs": self.route_costs,
            "inference_intensity": self.inference_intensity,
            "training_phase": self.training_phase,
            "loss_mode": self.loss_mode,
            "diagnostics": {
                "stage_count": len(route_repr),
                "batch_route_id": plan.get("batch_route_id"),
                "node_size": self.node_size,
                "mode": "forced" if forced_active else "planned",
            },
        }
        # Pass through planner/controller extras once (avoid double forward in subclasses).
        for extra_key in (
            "predicted_gains",
            "route_scores",
            "near_best_mask",
            "proposed_route_id",
            "predicted_route_losses",
        ):
            if plan.get(extra_key) is not None:
                result[extra_key] = plan[extra_key]
        if plan.get("batch_route_logits") is not None:
            result["batch_route_logits"] = plan["batch_route_logits"]
        if oracle_route_id is not None:
            result["oracle_route_id"] = oracle_route_id.to(device=device)
        if return_all or return_intermediates:
            return result
        return result["pred"]
