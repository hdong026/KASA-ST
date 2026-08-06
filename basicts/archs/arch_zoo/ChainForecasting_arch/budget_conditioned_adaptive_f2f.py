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

    def _budget(self, batch: int, device, dtype) -> torch.Tensor:
        bval = budget_from_intensity(self.inference_intensity, self.route_costs.tolist())
        return torch.full((batch,), bval, device=device, dtype=dtype)

    def _select_route_id(
        self,
        history: torch.Tensor,
        train: bool,
    ) -> dict[str, Any]:
        b = history.shape[0]
        device = history.device
        dtype = history.dtype
        budget = self._budget(b, device, dtype)

        if self.forced_route is not None or self.route_selection_mode == "forced":
            if self.forced_route is None:
                raise ValueError("forced mode requires forced_route")
            rid = self.candidate_routes.index(self.forced_route)
            selected = torch.full((b,), rid, device=device, dtype=torch.long)
            costs = self.route_costs.to(device=device, dtype=dtype)
            return {
                "route_logits": torch.zeros(b, len(self.candidate_routes), device=device, dtype=dtype),
                "route_probs": F.one_hot(selected, num_classes=len(self.candidate_routes)).to(dtype),
                "feasible_mask": torch.ones(
                    b, len(self.candidate_routes), device=device, dtype=torch.bool
                ),
                "selected_route_id": selected,
                "selected_cost": costs[rid].expand(b),
                "budget": budget,
                "batch_route_id": rid,
            }

        plan = self.planner(
            history,
            intensity=self.inference_intensity,
            route_costs=self.route_costs,
            budget=budget,
            deterministic=not train or self.training_phase == "eval",
        )
        if self.route_granularity == "batch" or self.route_selection_mode == "batch":
            # Shared route for the whole batch: majority vote of sample decisions.
            # Note: converting the discrete route id to a Python int for ModuleList
            # indexing requires one host read; sandwich / forced / oracle paths
            # avoid this by supplying CPU-side route lists.
            ids = plan["selected_route_id"]
            counts = torch.bincount(ids, minlength=len(self.candidate_routes))
            batch_id = int(counts.argmax().item())
            plan["batch_route_id"] = batch_id
            plan["selected_route_id"] = torch.full(
                (b,), batch_id, device=device, dtype=torch.long
            )
            plan["selected_cost"] = self.route_costs.to(device=device, dtype=dtype)[
                batch_id
            ].expand(b)
        else:
            plan["batch_route_id"] = int(plan["selected_route_id"][0].item())
        return plan

    def _spatial_index_for_resolution(self, res: int) -> int:
        """Map temporal resolution to progressive spatial module by capacity tier."""
        # Align with supernet stage index of this resolution in full_resolutions
        if res not in self.res_to_index:
            raise KeyError(f"resolution {res} not in supernet {self.full_resolutions}")
        return self.res_to_index[res]

    def _execute_route(
        self,
        history_data: torch.Tensor,
        route: list[int],
        return_trace: bool = False,
    ) -> dict[str, Any]:
        """Run KASA stages for ``route`` using shared supernet modules."""
        chain_preds: list[torch.Tensor] = []
        temporal_preds: list[torch.Tensor] = []
        conditions: list[torch.Tensor | None] = []
        spatial_cfgs: list[dict[str, Any]] = []
        prev_forecast = None
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
        if final.shape[1] != self.output_len:
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
        **kwargs,
    ):
        del batch_seen, epoch, kwargs
        history = history_data
        b = history.shape[0]
        device = history.device
        dtype = history.dtype

        # Forced mode must dominate sandwich / random sampling.
        forced_active = (
            self.forced_route is not None or self.route_selection_mode == "forced"
        )
        if forced_active and sandwich_routes is not None:
            raise RuntimeError(
                "forced-route mode received sandwich_routes; "
                "forced must override sandwich/random sampling"
            )

        # Supernet sandwich: execute provided routes (caller accumulates loss)
        if sandwich_routes is not None and not forced_active:
            outs = []
            for route in sandwich_routes:
                validate_route(route, horizon=self.output_len)
                outs.append(self._execute_route(history, route))
            # Primary pred = last route's final (typically full chain)
            primary = outs[-1]
            result = {
                "pred": primary["pred"],
                "prediction": primary["pred"],
                "chain_preds": primary["chain_preds"],
                "chain_resolutions": primary["chain_resolutions"],
                "sandwich_outputs": outs,
                "selected_route": primary["chain_resolutions"],
                "selected_route_id": torch.tensor(
                    [self.candidate_routes.index(primary["chain_resolutions"])],
                    device=device,
                ).expand(b),
                "route_logits": None,
                "route_probs": None,
                "selected_cost": self.route_costs[
                    self.candidate_routes.index(primary["chain_resolutions"])
                ].to(device=device, dtype=dtype).expand(b),
                "budget": self._budget(b, device, dtype),
                "diagnostics": {"mode": "sandwich"},
            }
            if return_all or return_intermediates:
                return result
            return result["pred"]

        # Planner phase: always score routes, but execute oracle / forced labels
        # so the discrete route does not rely on STE for forecasting loss.
        plan = self._select_route_id(history, train=train)
        if (
            oracle_route_id is not None
            and self.training_phase in {"planner", "joint"}
            and self.forced_route is None
            and self.route_selection_mode != "forced"
        ):
            # Batch shared oracle: use mode of provided labels (CPU-side ints).
            if oracle_route_id.ndim == 0:
                rid = int(oracle_route_id.item())
            else:
                counts = torch.bincount(
                    oracle_route_id.long().view(-1),
                    minlength=len(self.candidate_routes),
                )
                rid = int(counts.argmax().item())
            route = list(self.candidate_routes[rid])
            plan["selected_route_id"] = torch.full(
                (b,), rid, device=device, dtype=torch.long
            )
            plan["selected_cost"] = self.route_costs.to(device=device, dtype=dtype)[
                rid
            ].expand(b)
            plan["batch_route_id"] = rid
        else:
            rid = int(plan["batch_route_id"])
            route = list(self.candidate_routes[rid])
        executed = self._execute_route(history, route)

        # Forced-route hard assertions (never silently diverge).
        if forced_active:
            if self.forced_route is None:
                raise RuntimeError("forced mode requires forced_route")
            if list(route) != list(self.forced_route):
                raise RuntimeError(
                    f"selected_route {route} != forced_route {self.forced_route}"
                )
            if list(executed["chain_resolutions"]) != list(self.forced_route):
                raise RuntimeError(
                    f"chain_resolutions {executed['chain_resolutions']} "
                    f"!= forced_route {self.forced_route}"
                )
            if len(executed["chain_resolutions"]) != len(self.forced_route):
                raise RuntimeError(
                    f"actual_stage_count {len(executed['chain_resolutions'])} "
                    f"!= len(forced_route) {len(self.forced_route)}"
                )
            if executed["pred"].shape[1] != self.output_len:
                raise RuntimeError(
                    f"final forecast length {executed['pred'].shape[1]} "
                    f"!= horizon {self.output_len}"
                )
            if not self._logged:
                print(
                    "[budget_f2f forced] "
                    f"executed_routes={[list(route)]} "
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
            "selected_route": route,
            "actual_stage_count": len(executed["chain_resolutions"]),
            "executed_routes": [list(route)],
            "route_logits": plan["route_logits"],
            "route_probs": plan["route_probs"],
            "feasible_mask": plan["feasible_mask"],
            "selected_cost": plan["selected_cost"],
            "budget": plan["budget"],
            "candidate_routes": self.candidate_routes,
            "route_costs": self.route_costs,
            "inference_intensity": self.inference_intensity,
            "training_phase": self.training_phase,
            "loss_mode": self.loss_mode,
            "diagnostics": {
                "stage_count": len(route),
                "batch_route_id": rid,
                "node_size": self.node_size,
                "mode": "forced" if forced_active else "planned",
            },
        }
        if oracle_route_id is not None:
            result["oracle_route_id"] = oracle_route_id.to(device=device)
        if return_all or return_intermediates:
            return result
        return result["pred"]
