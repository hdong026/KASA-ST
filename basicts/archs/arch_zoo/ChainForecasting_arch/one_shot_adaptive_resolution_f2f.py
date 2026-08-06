"""OneShotAdaptiveResolutionF2FNet: one-shot planner + packed executor F2F."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from basicts.archs.arch_zoo.ChainForecasting_arch.dynamic_resolution_token_loss import (
    one_shot_resolution_total_loss,
)
from basicts.archs.arch_zoo.ChainForecasting_arch.one_shot_resolution_hierarchy import (
    SpatialResolutionTree,
    TemporalResolutionTree,
    build_leaf_cover_matrix,
    build_tree_child_tables,
)
from basicts.archs.arch_zoo.ChainForecasting_arch.one_shot_resolution_planner import (
    MAX_OPTIONAL_INTERMEDIATE_STEPS,
    OneShotResolutionProgramPlanner,
    SharedHistoryEncoder,
)
from basicts.archs.arch_zoo.ChainForecasting_arch.packed_resolution_executor import (
    PackedResolutionForecastExecutor,
)
from basicts.archs.arch_zoo.ChainForecasting_arch.resolution_program_compiler import (
    ResolutionProgramCompiler,
)
from basicts.archs.arch_zoo.ChainForecasting_arch.resolution_program_optimizer import (
    BudgetDualController,
    ResolutionProgramOptimizer,
)


def _load_adjacency(
    adj_mx_path: str | None,
    node_size: int,
    adj_mx=None,
) -> np.ndarray:
    if adj_mx is not None:
        if isinstance(adj_mx, torch.Tensor):
            adj = adj_mx.detach().cpu().numpy().astype(np.float64)
        else:
            adj = np.asarray(adj_mx, dtype=np.float64)
        return adj
    if adj_mx_path:
        import pickle

        with open(adj_mx_path, "rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, (list, tuple)):
            adj = np.asarray(obj[0], dtype=np.float64)
        else:
            adj = np.asarray(obj, dtype=np.float64)
        return adj
    adj = np.eye(node_size, dtype=np.float64)
    for i in range(node_size - 1):
        adj[i, i + 1] = adj[i + 1, i] = 1.0
    return adj


def _adj_to_edge_index(adj: np.ndarray) -> torch.Tensor:
    a = (np.asarray(adj) > 0).astype(np.int64)
    np.fill_diagonal(a, 0)
    src, dst = np.nonzero(a)
    if src.size == 0:
        # self-loop fallback for connectivity
        idx = np.arange(a.shape[0], dtype=np.int64)
        return torch.from_numpy(np.stack([idx, idx], axis=0))
    return torch.from_numpy(np.stack([src, dst], axis=0)).long()


def _build_node_meta(tree, kind: str) -> torch.Tensor:
    """[Ntree, 6] metadata embeddings inputs."""
    rows = []
    for n in tree.nodes:
        if kind == "temporal":
            h = float(tree.horizon)
            start = n.start / h
            end = n.end / h
            center = 0.5 * (start + end)
            width = end - start
            depth = n.depth / max(tree.depth, 1)
            leaf = 1.0 if n.is_leaf else 0.0
            rows.append([start, end, center, width, depth, leaf])
        else:
            nn_ = float(tree.n_nodes)
            size = len(n.original_node_indices) / nn_
            depth = n.depth / max(tree.depth, 1)
            leaf = 1.0 if n.is_leaf else 0.0
            rows.append([size, depth, leaf, size, depth, 1.0 - leaf])
    return torch.tensor(rows, dtype=torch.float32)


class OneShotAdaptiveResolutionF2FNet(nn.Module):
    """History encode once → plan once → 0..2 packed intermediates → one full final."""

    def __init__(self, **model_args):
        super().__init__()
        self.node_size = int(model_args["node_size"])
        self.input_len = int(model_args["input_len"])
        self.input_dim = int(model_args.get("input_dim", 3))
        self.output_len = int(model_args["output_len"])
        self.output_dim = int(model_args.get("output_dim", 1))
        self.thinking_intensity = float(model_args.get("thinking_intensity", 0.5))
        self.planner_hidden_dim = int(model_args.get("planner_hidden_dim", 32))
        self.executor_hidden_dim = int(
            model_args.get("executor_hidden_dim", model_args.get("forecast_cell_hidden_dim", 32))
        )
        self.planner_temperature = float(model_args.get("planner_temperature", 1.0))
        self.adapter_epsilon = float(model_args.get("forecast_state_adapter_epsilon", 0.02))
        self.adapter_hidden = int(model_args.get("forecast_state_adapter_hidden_dim", 16))
        self.clustering_seed = int(model_args.get("clustering_seed", 0))
        self.dataset_name = str(model_args.get("dataset_name", "synthetic"))
        cache_dir = model_args.get(
            "hierarchy_cache_dir",
            str(Path("generated/cache/adaptive_resolution_hierarchies")),
        )
        if cache_dir in (None, "", "None"):
            cache_dir = None
        self.budget_base = float(model_args.get("budget_base", 0.15))
        self.budget_scale = float(model_args.get("budget_scale", 0.45))
        self.k_steps = MAX_OPTIONAL_INTERMEDIATE_STEPS

        adj = _load_adjacency(
            model_args.get("adj_mx_path"), self.node_size, model_args.get("adj_mx")
        )
        self.temporal_tree = TemporalResolutionTree(self.output_len)
        self.spatial_tree = SpatialResolutionTree(
            adjacency=adj,
            clustering_seed=self.clustering_seed,
            cache_dir=cache_dir,
            dataset_name=self.dataset_name,
        )
        intensity = float(np.clip(self.thinking_intensity, 0.0, 1.0))
        self.total_optional_budget_value = self.budget_base + self.budget_scale * intensity

        left_t, right_t, leaf_t = build_tree_child_tables(self.temporal_tree)
        left_s, right_s, leaf_s = build_tree_child_tables(self.spatial_tree)
        self.register_buffer("left_t", left_t, persistent=False)
        self.register_buffer("right_t", right_t, persistent=False)
        self.register_buffer("leaf_t", leaf_t, persistent=False)
        self.register_buffer("left_s", left_s, persistent=False)
        self.register_buffer("right_s", right_s, persistent=False)
        self.register_buffer("leaf_s", leaf_s, persistent=False)
        self.register_buffer(
            "leaf_cover_t",
            build_leaf_cover_matrix(self.temporal_tree, self.output_len, "temporal"),
            persistent=False,
        )
        self.register_buffer(
            "leaf_cover_s",
            build_leaf_cover_matrix(self.spatial_tree, self.node_size, "spatial"),
            persistent=False,
        )
        self.register_buffer(
            "temporal_node_meta", _build_node_meta(self.temporal_tree, "temporal"), persistent=False
        )
        self.register_buffer(
            "spatial_node_meta", _build_node_meta(self.spatial_tree, "spatial"), persistent=False
        )
        self.register_buffer("edge_index", _adj_to_edge_index(adj), persistent=False)

        self.history_encoder = SharedHistoryEncoder(self.input_dim, self.planner_hidden_dim)
        # Project executor dim if different
        self.hist_to_exec = (
            nn.Identity()
            if self.planner_hidden_dim == self.executor_hidden_dim
            else nn.Linear(self.planner_hidden_dim, self.executor_hidden_dim)
        )
        self.planner = OneShotResolutionProgramPlanner(
            hidden_dim=self.planner_hidden_dim,
            n_temporal_nodes=len(self.temporal_tree.nodes),
            n_spatial_nodes=len(self.spatial_tree.nodes),
            k_steps=self.k_steps,
            temperature=self.planner_temperature,
        )
        self.compiler = ResolutionProgramCompiler(
            temporal_tree=self.temporal_tree,
            spatial_tree=self.spatial_tree,
            left_t=self.left_t,
            right_t=self.right_t,
            leaf_t=self.leaf_t,
            left_s=self.left_s,
            right_s=self.right_s,
            leaf_s=self.leaf_s,
            k_steps=self.k_steps,
        )
        self.executor = PackedResolutionForecastExecutor(
            hidden_dim=self.executor_hidden_dim,
            output_dim=self.output_dim,
            history_len=self.input_len,
            adapter_hidden=self.adapter_hidden,
            adapter_epsilon=self.adapter_epsilon,
            max_k=self.k_steps,
        )
        self.budget_dual = BudgetDualController(init_value=0.1, lr=float(model_args.get("budget_dual_lr", 0.01)))

        # Teacher optimizer is NOT a submodule — kept as plain attribute for future training only
        object.__setattr__(
            self,
            "_teacher_optimizer",
            ResolutionProgramOptimizer(
                self.temporal_tree, self.spatial_tree, backend="proxy_greedy", k_steps=self.k_steps
            ),
        )
        self._logged_startup = False

    @property
    def teacher_optimizer(self) -> ResolutionProgramOptimizer:
        return self._teacher_optimizer

    def optional_budget(self, batch: int, device, dtype) -> torch.Tensor:
        return torch.full(
            (batch,),
            float(self.total_optional_budget_value),
            device=device,
            dtype=dtype,
        )

    def _startup_log(self) -> None:
        if self._logged_startup:
            return
        self._logged_startup = True
        n_plan = sum(p.numel() for p in self.planner.parameters())
        n_exec = sum(p.numel() for p in self.executor.parameters())
        n_enc = sum(p.numel() for p in self.history_encoder.parameters())
        print(
            "[OneShotAdaptiveResolution] "
            f"temporal_tree={self.temporal_tree.summary()} "
            f"spatial_tree={self.spatial_tree.summary()} "
            f"MAX_OPTIONAL_INTERMEDIATE_STEPS={self.k_steps} "
            f"thinking_intensity={self.thinking_intensity} "
            f"optional_budget={self.total_optional_budget_value:.4f} "
            f"history_encoder_params={n_enc} planner_params={n_plan} "
            f"executor_params={n_exec} "
            f"final_stage=mandatory_full_HN"
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
        history = history_data[..., : self.input_dim]
        b = history.shape[0]
        device = history.device
        dtype = history.dtype

        self.history_encoder.reset_counter()
        self.planner.reset_counter()

        # 1) Encode history once
        encoded = self.history_encoder(history)
        encoded_exec = self.hist_to_exec(encoded)

        budget = self.optional_budget(b, device, dtype)
        # 2) Plan once
        planner_out = self.planner(
            encoded_history=encoded,
            temporal_node_meta=self.temporal_node_meta,
            spatial_node_meta=self.spatial_node_meta,
            thinking_intensity=self.thinking_intensity,
            total_optional_budget=budget,
            temporal_leaf_mask=self.leaf_t,
            spatial_leaf_mask=self.leaf_s,
        )
        # 3) Compile program
        program = self.compiler.compile(
            planner_out,
            optional_budget=budget,
            thinking_intensity=self.thinking_intensity,
            deterministic=not train,
        )

        y_full = None
        if future_data is not None:
            y_full = future_data[..., : self.output_dim]
            if y_full.shape[1] != self.output_len:
                y_full = y_full[:, : self.output_len]

        # 4) Execute optional intermediates (compact valid samples)
        condition_full = None
        inter_preds, inter_tgts, inter_masks = [], [], []
        stage_diags = []
        for k in range(self.k_steps):
            stage_out = self.executor.run_intermediate_stage(
                encoded_history=encoded_exec,
                t_frontier=program["temporal_frontiers"][:, k],
                s_frontier=program["spatial_frontiers"][:, k],
                stage_valid=program["stage_valid"][:, k],
                leaf_cover_t=self.leaf_cover_t,
                leaf_cover_s=self.leaf_cover_s,
                temporal_node_meta=self.temporal_node_meta,
                spatial_node_meta=self.spatial_node_meta,
                stage_idx=k,
                thinking_intensity=self.thinking_intensity,
                remaining_budget=program["remaining_budget"],
                edge_index=self.edge_index,
                previous_condition_full=condition_full,
                full_target=y_full,
                horizon=self.output_len,
                n_nodes=self.node_size,
            )
            if stage_out["has_work"]:
                # Matched supervision on coarse packed tensors
                inter_preds.append(stage_out["supervised_coarse"])
                inter_tgts.append(stage_out["matched_coarse"])
                inter_masks.append(stage_out["slot_mask"])
                condition_full = stage_out["forwarded_condition_full"]
                # Ensure adapter did not overwrite supervised: keep coarse separate
            if return_intermediates:
                stage_diags.append(
                    {
                        "active_temporal_count": stage_out["active_t"].detach(),
                        "active_spatial_count": stage_out["active_s"].detach(),
                        "stage_valid": program["stage_valid"][:, k].detach(),
                        "stage_type": program["stage_types"][:, k].detach(),
                        "stage_cost": program["stage_costs"][:, k].detach(),
                        "packed_token_count": stage_out["packed_token_count"].detach(),
                        "has_work": stage_out["has_work"],
                    }
                )

        # 5) Mandatory final full-resolution once
        final = self.executor.run_final_stage(
            encoded_history=encoded_exec,
            previous_condition_full=condition_full,
            thinking_intensity=self.thinking_intensity,
            edge_index=self.edge_index,
            leaf_cover_t=self.leaf_cover_t,
            leaf_cover_s=self.leaf_cover_s,
            temporal_node_meta=self.temporal_node_meta,
            spatial_node_meta=self.spatial_node_meta,
            t_final=program["final_temporal_frontier"],
            s_final=program["final_spatial_frontier"],
            horizon=self.output_len,
            n_nodes=self.node_size,
        )

        result = {
            "pred": final,
            "chain_preds": [final],
            "planner_out": planner_out,
            "program": program,
            "expected_cost": program["cumulative_optional_cost"],
            "expected_optional_cost": program["cumulative_optional_cost"],
            "target_budget": budget,
            "dual": self.budget_dual.value,
            "thinking_intensity": self.thinking_intensity,
            "history_encoder_call_count": self.history_encoder.call_counter.count,
            "planner_call_count": self.planner.call_counter.count,
            "intermediate_stage_count": program["intermediate_stage_count"],
            "max_optional_intermediate_steps": self.k_steps,
        }

        if y_full is not None:
            # Differentiable planner proxy so split/continue heads receive grads
            # (compiled frontiers are discrete; STE alone may not reach pool ops).
            cont = torch.sigmoid(planner_out["continue_logits"])
            soft_cost = (cont * planner_out["expected_cost"]).sum(dim=-1)
            soft_split = (
                torch.sigmoid(planner_out["temporal_split_logits"]).mean(dim=(1, 2))
                + torch.sigmoid(planner_out["spatial_split_logits"]).mean(dim=(1, 2))
            )
            expected_for_loss = program["cumulative_optional_cost"].detach() + soft_cost + 0.05 * soft_split
            loss_parts = one_shot_resolution_total_loss(
                intermediate_preds=inter_preds,
                intermediate_targets=inter_tgts,
                intermediate_masks=inter_masks,
                final_pred=final,
                final_target=y_full,
                expected_optional_cost=expected_for_loss,
                optional_budget=budget,
                dual=self.budget_dual.value,
                planner_out=planner_out,
                teacher=None,
                imitation_coef=0.0,
            )
            result["dynamic_loss"] = loss_parts["loss"]
            result["loss_parts"] = loss_parts
            result["matched_preds"] = inter_preds
            result["matched_targets"] = inter_tgts
            result["matched_masks"] = inter_masks

        if return_intermediates:
            result["intermediates"] = stage_diags
            result["continue_probs"] = program["continue_probs"].detach()
            result["planner_confidence"] = planner_out["planner_confidence"].detach()
        elif "intermediates" in result:
            pass

        if return_all or return_intermediates:
            return result
        return final

    def state_dict(self, *args, **kwargs):
        sd = super().state_dict(*args, **kwargs)
        # Guarantee teacher optimizer is not in inference state dict
        return {k: v for k, v in sd.items() if "teacher_optimizer" not in k}

    def dual_update_from_output(self, out: dict) -> None:
        """Runner hook interface; this task's tests must not require calling it."""
        if "expected_optional_cost" in out and "target_budget" in out:
            self.budget_dual.dual_ascent_step(out["expected_optional_cost"], out["target_budget"])
