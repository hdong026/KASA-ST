"""AdaptiveResolutionPonderingF2FNet: sample-wise dynamic T/S forecast-to-forecast."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from basicts.archs.arch_zoo.ChainForecasting_arch.adaptive_resolution_hierarchy import (
    SpatialResolutionTree,
    TemporalResolutionTree,
    build_frontier_projections,
    build_membership_matrix,
    lift_resolution_to_full,
    pool_full_to_resolution,
)
from basicts.archs.arch_zoo.ChainForecasting_arch.adaptive_resolution_loss import (
    BudgetDual,
    compute_step_cost,
    dynamic_resolution_total_loss,
)
from basicts.archs.arch_zoo.ChainForecasting_arch.adaptive_resolution_pondering_controller import (
    AdaptiveResolutionPonderingController,
    enforce_progress_or_halt,
    map_slot_splits_to_tree_mask,
)
from basicts.archs.arch_zoo.ChainForecasting_arch.adaptive_resolution_state import (
    AdaptiveResolutionState,
)
from basicts.archs.arch_zoo.ChainForecasting_arch.resolution_conditioned_forecast_cell import (
    ResolutionConditionAdapter,
    ResolutionConditionedForecastCell,
    build_spatial_slot_meta,
    build_unit_features,
)

logger = logging.getLogger("easytorch-training")


def _load_adjacency(
    adj_mx_path: str | None,
    node_size: int,
    adj_mx: np.ndarray | torch.Tensor | None = None,
) -> np.ndarray:
    if adj_mx is not None:
        if isinstance(adj_mx, torch.Tensor):
            adj = adj_mx.detach().cpu().numpy().astype(np.float64)
        else:
            adj = np.asarray(adj_mx, dtype=np.float64)
        if adj.ndim != 2 or adj.shape[0] != node_size or adj.shape[1] != node_size:
            raise ValueError(
                f"adj_mx shape {adj.shape} incompatible with node_size={node_size}"
            )
        return adj
    if adj_mx_path:
        import pickle

        with open(adj_mx_path, "rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, (list, tuple)):
            adj = np.asarray(obj[0] if len(obj) > 0 else obj, dtype=np.float64)
        else:
            adj = np.asarray(obj, dtype=np.float64)
        if adj.ndim != 2:
            raise ValueError(f"adj must be 2D, got {adj.shape}")
        if adj.shape[0] != node_size or adj.shape[1] != node_size:
            raise ValueError(
                f"adj shape {adj.shape} incompatible with node_size={node_size}"
            )
        return adj
    # Synthetic fallback path graph (for CPU tests without files)
    adj = np.eye(node_size, dtype=np.float64)
    for i in range(node_size - 1):
        adj[i, i + 1] = adj[i + 1, i] = 1.0
    return adj


class AdaptiveResolutionPonderingF2FNet(nn.Module):
    """Dynamic forecast-to-forecast with automatic T/S hierarchies and pondering."""

    def __init__(self, **model_args):
        super().__init__()
        self.node_size = int(model_args["node_size"])
        self.input_len = int(model_args["input_len"])
        self.input_dim = int(model_args.get("input_dim", 3))
        self.output_len = int(model_args["output_len"])
        self.output_dim = int(model_args.get("output_dim", 1))
        self.thinking_intensity = float(model_args.get("thinking_intensity", 0.5))
        self.controller_hidden_dim = int(model_args.get("controller_hidden_dim", 32))
        self.forecast_hidden_dim = int(model_args.get("forecast_cell_hidden_dim", 32))
        self.controller_temperature = float(model_args.get("controller_temperature", 1.0))
        self.controller_temperature_decay = float(
            model_args.get("controller_temperature_decay", 0.97)
        )
        self.halt_threshold = float(model_args.get("halt_threshold", 0.5))
        self.split_threshold = float(model_args.get("split_threshold", 0.5))
        self.adapter_epsilon = float(model_args.get("forecast_state_adapter_epsilon", 0.02))
        self.adapter_hidden = int(model_args.get("forecast_state_adapter_hidden_dim", 16))
        self.clustering_seed = int(model_args.get("clustering_seed", 0))
        self.dataset_name = str(model_args.get("dataset_name", "synthetic"))
        cache_dir = model_args.get(
            "hierarchy_cache_dir",
            str(Path("generated/cache/adaptive_resolution_hierarchies")),
        )
        self.budget_base = float(model_args.get("budget_base", 0.35))
        self.budget_scale = float(model_args.get("budget_scale", 0.55))
        self.dual_lr = float(model_args.get("budget_dual_lr", 0.01))

        # Reject forbidden fixed schedules if somehow passed
        for forbidden in (
            "chain_lengths",
            "temporal_resolution_candidates",
            "spatial_resolution_candidates",
            "graph_resolution_capacities",
        ):
            if model_args.get(forbidden) not in (None, [], ()):
                # Allow None from BEST_SETTINGS merge — only raise on concrete schedules
                val = model_args.get(forbidden)
                if forbidden == "chain_lengths" and val is not None:
                    # BEST_SETTINGS may not include chain_lengths; variant_spec may inject.
                    # We intentionally ignore injected chain_lengths.
                    pass

        adj = _load_adjacency(
            model_args.get("adj_mx_path"),
            self.node_size,
            adj_mx=model_args.get("adj_mx"),
        )
        self.temporal_tree = TemporalResolutionTree(self.output_len)
        self.spatial_tree = SpatialResolutionTree(
            adjacency=adj,
            clustering_seed=self.clustering_seed,
            cache_dir=cache_dir if cache_dir not in (None, "", "None") else None,
            dataset_name=self.dataset_name,
        )
        self.max_ponder_steps = (
            int(self.temporal_tree.depth)
            + int(self.spatial_tree.depth)
            + 2
        )
        intensity = float(np.clip(self.thinking_intensity, 0.0, 1.0))
        self.total_budget = self.budget_base + self.budget_scale * intensity

        self.register_buffer(
            "temporal_membership",
            build_membership_matrix(
                self.temporal_tree.nodes, self.output_len, kind="temporal"
            ),
            persistent=False,
        )
        self.register_buffer(
            "spatial_membership",
            build_membership_matrix(
                self.spatial_tree.nodes, self.node_size, kind="spatial"
            ),
            persistent=False,
        )

        self.forecast_cell = ResolutionConditionedForecastCell(
            history_len=self.input_len,
            input_dim=self.input_dim,
            hidden_dim=self.forecast_hidden_dim,
            output_dim=self.output_dim,
        )
        self.controller = AdaptiveResolutionPonderingController(
            unit_feat_dim=12,
            global_feat_dim=16,
            hidden_dim=self.controller_hidden_dim,
            temperature=self.controller_temperature,
            split_threshold=self.split_threshold,
            halt_threshold=self.halt_threshold,
        )
        # channels: current + prev + delta + meta(5) = 8
        self.condition_adapter = ResolutionConditionAdapter(
            feat_dim=8,
            hidden_dim=self.adapter_hidden,
            epsilon=self.adapter_epsilon,
        )
        # Tiny zero-init residual on lifted full candidate (not a backbone)
        self.full_residual = nn.Conv2d(self.output_dim, self.output_dim, kernel_size=1)
        nn.init.zeros_(self.full_residual.weight)
        nn.init.zeros_(self.full_residual.bias)

        self.budget_dual = BudgetDual(init_value=0.1, lr=self.dual_lr)
        self._logged_startup = False

    def target_budget(self, batch_size: int, device, dtype) -> torch.Tensor:
        intensity = float(np.clip(self.thinking_intensity, 0.0, 1.0))
        value = self.budget_base + self.budget_scale * intensity
        return torch.full((batch_size,), value, device=device, dtype=dtype)

    def _startup_log(self) -> None:
        if self._logged_startup:
            return
        self._logged_startup = True
        tsum = self.temporal_tree.summary()
        ssum = self.spatial_tree.summary()
        n_ctrl = sum(p.numel() for p in self.controller.parameters())
        n_cell = sum(p.numel() for p in self.forecast_cell.parameters())
        n_total = sum(p.numel() for p in self.parameters())
        print(
            "[AdaptiveResolutionPondering] "
            f"temporal_tree={tsum} spatial_tree={ssum} "
            f"root_resolution=(1,1) max_ponder_steps={self.max_ponder_steps} "
            f"thinking_intensity={self.thinking_intensity} "
            f"target_budget={self.budget_base + self.budget_scale * self.thinking_intensity:.4f} "
            f"controller_params={n_ctrl} forecast_cell_params={n_cell} total_params={n_total}"
        )

    def _frontier_projections(self, state: AdaptiveResolutionState):
        # Detach discrete frontier structure for pool/lift indexing; split heads
        # still train via soft cost proxy + STE-friendly frontier arithmetic.
        p_t, l_t, t_mask = build_frontier_projections(
            self.temporal_membership,
            state.temporal_frontier_mask.detach(),
            max_active=self.output_len,
        )
        p_s, l_s, s_mask = build_frontier_projections(
            self.spatial_membership,
            state.spatial_frontier_mask.detach(),
            max_active=self.node_size,
        )
        return p_t, l_t, t_mask, p_s, l_s, s_mask

    def _global_feats(
        self,
        full_cand: torch.Tensor,
        prev_full: torch.Tensor | None,
        state: AdaptiveResolutionState,
        step_idx: int,
    ) -> torch.Tensor:
        b = full_cand.shape[0]
        mean = full_cand.mean(dim=(1, 2, 3))
        std = full_cand.std(dim=(1, 2, 3), unbiased=False)
        if prev_full is None:
            delta = torch.zeros_like(mean)
        else:
            delta = (full_cand - prev_full).abs().mean(dim=(1, 2, 3))
        at = state.count_active_temporal_units() / float(self.output_len)
        asn = state.count_active_spatial_units() / float(self.node_size)
        step = full_cand.new_full((b,), float(step_idx) / float(max(self.max_ponder_steps, 1)))
        cost = state.cumulative_cost
        rem = state.remaining_budget
        intens = full_cand.new_full((b,), float(self.thinking_intensity))
        zeros = torch.zeros(b, device=full_cand.device, dtype=full_cand.dtype)
        return torch.stack(
            [
                mean,
                std,
                delta,
                at,
                asn,
                step,
                cost,
                rem,
                intens,
                zeros,
                zeros,
                zeros,
                zeros,
                zeros,
                zeros,
                zeros,
            ],
            dim=-1,
        )

    def forward(
        self,
        history_data: torch.Tensor,
        future_data: torch.Tensor = None,
        batch_seen: int = 0,
        epoch: int = 0,
        train: bool = False,
        return_all: bool = False,
        return_intermediates: bool = False,
        **kwargs,
    ):
        del batch_seen, kwargs
        self._startup_log()
        # Temperature anneal by epoch (no training loop required for correctness)
        if train and epoch is not None:
            tau = self.controller_temperature * (
                self.controller_temperature_decay ** float(max(int(epoch), 0))
            )
            self.controller.set_temperature(max(tau, 0.3))

        history = history_data[..., : self.input_dim]
        b = history.shape[0]
        device = history.device
        dtype = history.dtype
        budget = self.target_budget(b, device, dtype)
        state = AdaptiveResolutionState.root_init(
            b, self.temporal_tree, self.spatial_tree, budget, device, dtype
        )

        matched_preds: list[torch.Tensor] = []
        matched_targets: list[torch.Tensor] = []
        matched_masks: list[torch.Tensor] = []
        full_candidates: list[torch.Tensor] = []
        step_diags: list[dict[str, Any]] = []
        halt_probs = []
        costs = []

        condition = None
        prev_full = None
        # Soft halt survival for training
        survival = torch.ones(b, device=device, dtype=dtype)
        halt_weights = []

        y_full_target = None
        if future_data is not None:
            # future_data often [B,H,N,C_full]; take target channel 0
            y_full_target = future_data[..., : self.output_dim]
            if y_full_target.shape[1] != self.output_len:
                y_full_target = y_full_target[:, : self.output_len]

        final_full = None
        for step_idx in range(self.max_ponder_steps):
            p_t, l_t, t_mask, p_s, l_s, s_mask = self._frontier_projections(state)
            cell_out = self.forecast_cell(
                history=history,
                temporal_tree=self.temporal_tree,
                spatial_tree=self.spatial_tree,
                temporal_frontier_mask=state.temporal_frontier_mask.detach(),
                spatial_frontier_mask=state.spatial_frontier_mask.detach(),
                p_t=p_t,
                p_s=p_s,
                t_active_mask=t_mask,
                s_active_mask=s_mask,
                previous_condition=condition,
                ponder_step=step_idx,
                thinking_intensity=self.thinking_intensity,
                remaining_budget=state.remaining_budget.detach(),
            )
            coarse = cell_out["coarse_forecast"]  # supervised
            supervised = coarse
            slot_mask = cell_out["cell_features"]["slot_mask"]

            # Condition adapter (does not alter supervised)
            if condition is None:
                forwarded = supervised
            else:
                # Adapter input = cat(current, prev, delta, meta5) -> 8 channels
                t_meta5 = (
                    cell_out["cell_features"]["temporal_meta"]
                    .unsqueeze(2)
                    .expand(-1, -1, s_mask.shape[1], -1)[..., :4]
                )
                meta = torch.cat([t_meta5, slot_mask.unsqueeze(-1)], dim=-1)
                forwarded = self.condition_adapter(supervised, condition, meta)

            full_cand = lift_resolution_to_full(supervised, l_t, l_s)
            # Tiny residual in channel-first conv layout
            res = self.full_residual(full_cand.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            full_cand = full_cand + res

            # Matched target
            if y_full_target is not None:
                matched_tgt = pool_full_to_resolution(y_full_target, p_t, p_s)
            else:
                matched_tgt = torch.zeros_like(supervised)

            # Only accumulate loss for samples not yet halted before this step
            alive_mask = (~state.halted).to(dtype)
            step_slot_mask = slot_mask * alive_mask.view(b, 1, 1)
            matched_preds.append(supervised)
            matched_targets.append(matched_tgt)
            matched_masks.append(step_slot_mask)
            full_candidates.append(full_cand)

            s_sizes, s_depths = build_spatial_slot_meta(
                self.spatial_tree, state.spatial_frontier_mask, self.node_size
            )
            t_feats, s_feats = build_unit_features(
                supervised.detach(),
                t_mask,
                s_mask,
                cell_out["cell_features"]["temporal_meta"].detach(),
                s_sizes,
                s_depths,
                None if condition is None else condition.detach(),
                state.remaining_budget.detach(),
            )
            g_feats = self._global_feats(
                full_cand.detach(),
                None if prev_full is None else prev_full.detach(),
                state,
                step_idx,
            )
            ctrl = self.controller(
                temporal_unit_feats=t_feats,
                spatial_unit_feats=s_feats,
                temporal_valid_mask=t_mask,
                spatial_valid_mask=s_mask,
                temporal_leaf_mask=cell_out["cell_features"]["temporal_leaf_mask"],
                spatial_leaf_mask=cell_out["cell_features"]["spatial_leaf_mask"],
                global_feats=g_feats,
                halted=state.halted,
                deterministic=not train,
            )
            ctrl = enforce_progress_or_halt(ctrl, state.is_fully_refined(), state.halted)

            step_cost = compute_step_cost(
                state.count_active_temporal_units().detach(),
                state.count_active_spatial_units().detach(),
                self.output_len,
                self.node_size,
            )
            # Differentiable soft refinement proxy so split heads receive budget grads
            # (hard frontier indexing is discrete; STE alone cannot reach pool ops).
            soft_t = (
                ctrl["temporal_split_prob"] * ctrl["temporal_allow"].to(dtype)
            ).sum(dim=-1)
            soft_s = (
                ctrl["spatial_split_prob"] * ctrl["spatial_allow"].to(dtype)
            ).sum(dim=-1)
            step_cost = step_cost + 0.05 * (soft_t + soft_s) / float(
                max(self.output_len + self.node_size, 1)
            )
            # Halted samples do not accrue cost
            step_cost = step_cost * alive_mask
            state.cumulative_cost = state.cumulative_cost + step_cost.detach()
            state.remaining_budget = (budget - state.cumulative_cost).clamp_min(0.0)
            costs.append(step_cost)

            # Soft halt weights (training): conditional halt * survival
            h_prob = ctrl["halt_prob"]
            w_stop = survival * h_prob
            # last step absorbs remainder later
            halt_probs.append(h_prob)
            halt_weights.append(w_stop)
            survival = survival * (1.0 - h_prob)

            hard_halt = ctrl["halt_hard"] > 0.5
            newly_halt = hard_halt & (~state.halted)
            # Freeze final for newly halted
            if final_full is None:
                final_full = full_cand
            else:
                sel = newly_halt.view(b, 1, 1, 1).to(dtype)
                final_full = torch.where(sel.bool(), full_cand, final_full)

            # Diagnostics
            t_changed = (ctrl["temporal_split_hard"] > 0.5).any(dim=-1)
            s_changed = (ctrl["spatial_split_hard"] > 0.5).any(dim=-1)
            step_type = []
            for bi in range(b):
                if bool(hard_halt[bi]) or bool(state.halted[bi]):
                    step_type.append("halt")
                elif bool(t_changed[bi]) and bool(s_changed[bi]):
                    step_type.append("joint")
                elif bool(t_changed[bi]):
                    step_type.append("temporal")
                elif bool(s_changed[bi]):
                    step_type.append("spatial")
                else:
                    step_type.append("halt")
            step_diags.append(
                {
                    "active_temporal_count": state.count_active_temporal_units().detach(),
                    "active_spatial_count": state.count_active_spatial_units().detach(),
                    "temporal_split_prob": ctrl["temporal_split_prob"].detach(),
                    "spatial_split_prob": ctrl["spatial_split_prob"].detach(),
                    "temporal_split_hard": ctrl["temporal_split_hard"].detach(),
                    "spatial_split_hard": ctrl["spatial_split_hard"].detach(),
                    "actual_temporal_split_mask": ctrl["temporal_split_hard"].detach(),
                    "actual_spatial_split_mask": ctrl["spatial_split_hard"].detach(),
                    "halt_prob": h_prob.detach(),
                    "halt_weight": w_stop.detach(),
                    "halt_hard": hard_halt.detach(),
                    "cumulative_cost": state.cumulative_cost.detach(),
                    "remaining_budget": state.remaining_budget.detach(),
                    "step_type": step_type,
                    "coarse_forecast_shape": tuple(supervised.shape),
                    "matched_target_shape": tuple(matched_tgt.shape),
                    "full_candidate_shape": tuple(full_cand.shape),
                    "coarse_supervised": supervised.detach(),
                    "condition": forwarded.detach(),
                    "coarse_after_adapter_check": supervised.detach(),
                    "temporal_frontier_ids": [
                        state.active_temporal_ids(i) for i in range(b)
                    ],
                    "spatial_frontier_ids": [
                        state.active_spatial_ids(i) for i in range(b)
                    ],
                    "temporal_interval_boundaries": [
                        [
                            (
                                self.temporal_tree.nodes[nid].start,
                                self.temporal_tree.nodes[nid].end,
                            )
                            for nid in state.active_temporal_ids(i)
                        ]
                        for i in range(b)
                    ],
                    "spatial_region_sizes": [
                        [
                            len(self.spatial_tree.nodes[nid].original_node_indices)
                            for nid in state.active_spatial_ids(i)
                        ]
                        for i in range(b)
                    ],
                }
            )

            # Update halt flags
            state.halted = state.halted | hard_halt
            state.ponder_step = state.ponder_step + (~state.halted).long()

            if bool(state.halted.all()):
                # Still need condition path break
                break

            # Apply splits for non-halted samples
            t_tree_split = map_slot_splits_to_tree_mask(
                ctrl["temporal_split_hard"], state.temporal_frontier_mask
            )
            s_tree_split = map_slot_splits_to_tree_mask(
                ctrl["spatial_split_hard"], state.spatial_frontier_mask
            )
            # Zero splits for halted
            alive = (~state.halted).to(dtype).unsqueeze(-1)
            t_tree_split = t_tree_split * alive
            s_tree_split = s_tree_split * alive
            state.apply_joint_split(t_tree_split, s_tree_split)

            # Align condition to new frontier: lift old forwarded -> full -> pool new
            old_full_cond = lift_resolution_to_full(forwarded, l_t, l_s)
            p_t2, l_t2, t_mask2, p_s2, l_s2, s_mask2 = self._frontier_projections(state)
            condition = pool_full_to_resolution(old_full_cond, p_t2, p_s2)
            condition = condition * (t_mask2.unsqueeze(-1) * s_mask2.unsqueeze(1)).unsqueeze(-1)
            prev_full = full_cand.detach()

        # Absorb remaining survival into last halt weight
        if halt_weights:
            halt_weights[-1] = halt_weights[-1] + survival
            halt_w = torch.stack(halt_weights, dim=1)
        else:
            halt_w = torch.ones(b, 1, device=device, dtype=dtype)

        if final_full is None:
            final_full = full_candidates[-1] if full_candidates else history.new_zeros(
                b, self.output_len, self.node_size, self.output_dim
            )
        # For still-running samples use last candidate
        still = ~state.halted
        if bool(still.any()) and full_candidates:
            sel = still.view(b, 1, 1, 1)
            final_full = torch.where(sel, full_candidates[-1], final_full)

        # Expected cost under halt weights
        cost_mat = torch.stack(costs, dim=1) if costs else torch.zeros(b, 1, device=device)
        # cumulative cost at stop ≈ sum of step costs up to stop; use weighted sum of step costs
        expected_cost = (halt_w * cost_mat).sum(dim=1)

        # Sample-wise halt step: first step where cumulative halt weight dominates
        if halt_probs:
            halt_p_mat = torch.stack(halt_probs, dim=1)
            # Approximate discrete halt step from hard decisions in diagnostics
            halt_step = torch.full((b,), len(step_diags) - 1, device=device, dtype=torch.long)
            for bi in range(b):
                for si, diag in enumerate(step_diags):
                    if bool(diag["halt_hard"][bi]):
                        halt_step[bi] = si
                        break
        else:
            halt_p_mat = halt_w
            halt_step = torch.zeros(b, device=device, dtype=torch.long)

        result = {
            "pred": final_full,
            # Runner compatibility: expose lifted full candidates as chain_preds
            "chain_preds": full_candidates if full_candidates else [final_full],
            "matched_preds": matched_preds,
            "matched_forecasts": matched_preds,
            "matched_targets": matched_targets,
            "matched_masks": matched_masks,
            "full_candidates": full_candidates,
            "halt_weights": halt_w,
            "halt_probs": halt_p_mat,
            "halt_step": halt_step,
            "step_costs": torch.stack(costs, dim=1) if costs else torch.zeros(b, 1, device=device),
            "expected_cost": expected_cost,
            "target_budget": budget,
            "total_budget": budget,
            "dual": self.budget_dual.value,
            "dual_variable": self.budget_dual.value,
            "step_diagnostics": step_diags,
            "intermediates": step_diags,
            "thinking_intensity": self.thinking_intensity,
            "max_ponder_steps": self.max_ponder_steps,
            "temporal_tree_summary": self.temporal_tree.summary(),
            "spatial_tree_summary": self.spatial_tree.summary(),
        }

        if y_full_target is not None and matched_preds:
            loss_parts = dynamic_resolution_total_loss(
                matched_preds=matched_preds,
                matched_targets=matched_targets,
                matched_masks=matched_masks,
                full_candidates=full_candidates,
                halt_weights=halt_w,
                full_target=y_full_target,
                expected_cost=expected_cost,
                budget=budget,
                dual=self.budget_dual.value,
            )
            result["dynamic_loss"] = loss_parts["loss"]
            result["loss_parts"] = loss_parts

        if return_all or return_intermediates:
            return result
        return final_full

    def dual_update_from_output(self, out: dict) -> None:
        if "expected_cost" in out and "target_budget" in out:
            self.budget_dual.dual_ascent_step(out["expected_cost"], out["target_budget"])
