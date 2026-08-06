"""Per-sample active temporal/spatial frontiers for adaptive resolution pondering."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from basicts.archs.arch_zoo.ChainForecasting_arch.adaptive_resolution_hierarchy import (
    SpatialResolutionTree,
    TemporalResolutionTree,
)


@dataclass
class AdaptiveResolutionState:
    """Batch of sample-wise frontiers over temporal/spatial trees."""

    temporal_frontier_mask: torch.Tensor  # [B, Tnodes]
    spatial_frontier_mask: torch.Tensor  # [B, Snodes]
    halted: torch.Tensor  # [B] bool
    ponder_step: torch.Tensor  # [B] long
    cumulative_cost: torch.Tensor  # [B]
    remaining_budget: torch.Tensor  # [B]
    temporal_tree: TemporalResolutionTree = field(repr=False)
    spatial_tree: SpatialResolutionTree = field(repr=False)

    @classmethod
    def root_init(
        cls,
        batch_size: int,
        temporal_tree: TemporalResolutionTree,
        spatial_tree: SpatialResolutionTree,
        total_budget: torch.Tensor | float,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> "AdaptiveResolutionState":
        b = int(batch_size)
        tmask = torch.zeros(b, len(temporal_tree.nodes), device=device, dtype=dtype)
        smask = torch.zeros(b, len(spatial_tree.nodes), device=device, dtype=dtype)
        tmask[:, temporal_tree.root_id] = 1.0
        smask[:, spatial_tree.root_id] = 1.0
        if not torch.is_tensor(total_budget):
            budget = torch.full((b,), float(total_budget), device=device, dtype=dtype)
        else:
            budget = total_budget.to(device=device, dtype=dtype)
            if budget.ndim == 0:
                budget = budget.expand(b)
        return cls(
            temporal_frontier_mask=tmask,
            spatial_frontier_mask=smask,
            halted=torch.zeros(b, device=device, dtype=torch.bool),
            ponder_step=torch.zeros(b, device=device, dtype=torch.long),
            cumulative_cost=torch.zeros(b, device=device, dtype=dtype),
            remaining_budget=budget.clone(),
            temporal_tree=temporal_tree,
            spatial_tree=spatial_tree,
        )

    def count_active_temporal_units(self) -> torch.Tensor:
        return self.temporal_frontier_mask.sum(dim=-1)

    def count_active_spatial_units(self) -> torch.Tensor:
        return self.spatial_frontier_mask.sum(dim=-1)

    def is_fully_refined(self) -> torch.Tensor:
        t_ok = self.count_active_temporal_units() >= float(self.temporal_tree.horizon)
        s_ok = self.count_active_spatial_units() >= float(self.spatial_tree.n_nodes)
        # All active nodes are leaves
        t_leaf = self._all_active_are_leaves(self.temporal_frontier_mask, self.temporal_tree)
        s_leaf = self._all_active_are_leaves(self.spatial_frontier_mask, self.spatial_tree)
        return t_ok & s_ok & t_leaf & s_leaf

    @staticmethod
    def _all_active_are_leaves(mask: torch.Tensor, tree) -> torch.Tensor:
        leaf_ids = [n.node_id for n in tree.nodes if n.is_leaf]
        leaf_mask = torch.zeros(mask.shape[-1], device=mask.device, dtype=mask.dtype)
        leaf_mask[leaf_ids] = 1.0
        active = mask > 0.5
        # For each sample: every active node is a leaf
        bad = active & (leaf_mask.view(1, -1) < 0.5)
        return ~bad.any(dim=-1)

    def validate_frontier(self) -> None:
        for bi in range(self.temporal_frontier_mask.shape[0]):
            self._validate_one(
                self.temporal_frontier_mask[bi],
                self.temporal_tree,
                kind="temporal",
            )
            self._validate_one(
                self.spatial_frontier_mask[bi],
                self.spatial_tree,
                kind="spatial",
            )

    @staticmethod
    def _validate_one(mask: torch.Tensor, tree, kind: str) -> None:
        ids = torch.nonzero(mask > 0.5, as_tuple=False).flatten().tolist()
        if not ids:
            raise RuntimeError(f"{kind} frontier empty")
        # Coverage of leaves
        covered: set[int] = set()
        for nid in ids:
            node = tree.nodes[nid]
            if kind == "temporal":
                members = set(range(node.start, node.end))
            else:
                members = set(node.original_node_indices)
            if covered & members:
                raise RuntimeError(f"{kind} frontier overlap at node {nid}")
            covered |= members
            # Parent of an active node must not also be active
            pid = node.parent_id
            while pid is not None:
                if pid in ids:
                    raise RuntimeError(f"{kind} parent/child both active ({pid}/{nid})")
                pid = tree.nodes[pid].parent_id
        expected = tree.horizon if kind == "temporal" else tree.n_nodes
        if len(covered) != expected:
            raise RuntimeError(
                f"{kind} frontier coverage {len(covered)} != {expected}"
            )

    def split_temporal_nodes(self, split_mask: torch.Tensor) -> torch.Tensor:
        """Apply temporal splits. ``split_mask``: [B, Tnodes] hard 0/1 on current frontier."""
        return self._apply_splits(
            self.temporal_frontier_mask, split_mask, self.temporal_tree, axis="temporal"
        )

    def split_spatial_nodes(self, split_mask: torch.Tensor) -> torch.Tensor:
        return self._apply_splits(
            self.spatial_frontier_mask, split_mask, self.spatial_tree, axis="spatial"
        )

    def _apply_splits(
        self,
        frontier: torch.Tensor,
        split_mask: torch.Tensor,
        tree,
        axis: str,
    ) -> torch.Tensor:
        """Apply splits with STE-friendly arithmetic (grad flows through split_mask)."""
        new_f = frontier.clone()
        b = frontier.shape[0]
        changed = torch.zeros(b, device=frontier.device, dtype=torch.bool)
        for bi in range(b):
            if bool(self.halted[bi]):
                continue
            ids = torch.nonzero(frontier[bi] > 0.5, as_tuple=False).flatten().tolist()
            for nid in ids:
                node = tree.nodes[nid]
                if node.is_leaf or node.left_child_id is None:
                    continue
                s = split_mask[bi, nid]
                # Parent stays if s≈0; children activate if s≈1 (STE 0/1 forward).
                new_f[bi, nid] = frontier[bi, nid] * (1.0 - s)
                new_f[bi, node.left_child_id] = frontier[bi, nid] * s
                new_f[bi, node.right_child_id] = frontier[bi, nid] * s
                if float(s.detach()) > 0.5:
                    changed[bi] = True
        if axis == "temporal":
            self.temporal_frontier_mask = new_f
        else:
            self.spatial_frontier_mask = new_f
        return changed

    def apply_joint_split(
        self,
        temporal_split_mask: torch.Tensor,
        spatial_split_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        t_changed = self.split_temporal_nodes(temporal_split_mask)
        s_changed = self.split_spatial_nodes(spatial_split_mask)
        return {
            "temporal_changed": t_changed,
            "spatial_changed": s_changed,
            "joint": t_changed & s_changed,
            "any": t_changed | s_changed,
        }

    def active_temporal_ids(self, batch_idx: int) -> list[int]:
        return torch.nonzero(
            self.temporal_frontier_mask[batch_idx] > 0.5, as_tuple=False
        ).flatten().tolist()

    def active_spatial_ids(self, batch_idx: int) -> list[int]:
        return torch.nonzero(
            self.spatial_frontier_mask[batch_idx] > 0.5, as_tuple=False
        ).flatten().tolist()

    def diagnostics(self) -> dict[str, Any]:
        return {
            "active_temporal_count": self.count_active_temporal_units().detach(),
            "active_spatial_count": self.count_active_spatial_units().detach(),
            "halted": self.halted.detach(),
            "ponder_step": self.ponder_step.detach(),
            "cumulative_cost": self.cumulative_cost.detach(),
            "remaining_budget": self.remaining_budget.detach(),
        }
