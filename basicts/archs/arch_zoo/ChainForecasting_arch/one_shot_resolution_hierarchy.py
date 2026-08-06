"""One-shot resolution hierarchy helpers.

Reuses TemporalResolutionTree / SpatialResolutionTree from
adaptive_resolution_hierarchy.py (no duplicate tree builders).

Adds scatter/gather assignment operators — no dense H×H / N×N projections.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from basicts.archs.arch_zoo.ChainForecasting_arch.adaptive_resolution_hierarchy import (
    SpatialResolutionTree,
    TemporalResolutionTree,
    TreeNode,
    build_membership_matrix,
)

# Re-export for callers that import from this module.
__all__ = [
    "TemporalResolutionTree",
    "SpatialResolutionTree",
    "TreeNode",
    "build_membership_matrix",
    "build_tree_child_tables",
    "build_leaf_cover_matrix",
    "frontier_active_counts",
    "frontier_to_leaf_assignment",
    "scatter_pool_temporal",
    "scatter_pool_spatial",
    "scatter_pool_full",
    "gather_lift_full",
    "build_sparse_region_edges",
    "full_leaf_frontier_mask",
    "root_frontier_mask",
]


def build_tree_child_tables(tree) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return left_child, right_child, is_leaf long/bool tables on CPU then movable."""
    n = len(tree.nodes)
    left = torch.full((n,), -1, dtype=torch.long)
    right = torch.full((n,), -1, dtype=torch.long)
    is_leaf = torch.zeros(n, dtype=torch.bool)
    for node in tree.nodes:
        is_leaf[node.node_id] = bool(node.is_leaf)
        if node.left_child_id is not None:
            left[node.node_id] = int(node.left_child_id)
        if node.right_child_id is not None:
            right[node.node_id] = int(node.right_child_id)
    return left, right, is_leaf


def build_leaf_cover_matrix(tree, num_leaves: int, kind: str) -> torch.Tensor:
    """Boolean cover [Ntree, Leaves]: node covers leaf positions."""
    n = len(tree.nodes)
    cover = torch.zeros(n, num_leaves, dtype=torch.bool)
    for node in tree.nodes:
        if kind == "temporal":
            members = range(node.start, node.end)
        else:
            members = node.original_node_indices
        for j in members:
            cover[node.node_id, j] = True
    return cover


def frontier_active_counts(frontier_mask: torch.Tensor) -> torch.Tensor:
    return frontier_mask.to(torch.float32).sum(dim=-1)


def root_frontier_mask(batch: int, tree, device, dtype=torch.float32) -> torch.Tensor:
    m = torch.zeros(batch, len(tree.nodes), device=device, dtype=dtype)
    m[:, tree.root_id] = 1.0
    return m


def full_leaf_frontier_mask(batch: int, tree, device, dtype=torch.float32) -> torch.Tensor:
    m = torch.zeros(batch, len(tree.nodes), device=device, dtype=dtype)
    for node in tree.nodes:
        if node.is_leaf:
            m[:, node.node_id] = 1.0
    return m


def frontier_to_leaf_assignment(
    frontier_mask: torch.Tensor,
    leaf_cover: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map each leaf to the active frontier node id and dense slot index.

    Args:
        frontier_mask: [B, Ntree] float/bool
        leaf_cover: [Ntree, Leaves] bool

    Returns:
        owner_node: [B, Leaves] long node ids
        owner_slot: [B, Leaves] long slot in [0, active_count) (padded invalid = -1 unused)
    """
    b, ntree = frontier_mask.shape
    leaves = leaf_cover.shape[1]
    device = frontier_mask.device
    # Which (sample, node, leaf) is active cover
    active = frontier_mask > 0.5  # [B, Ntree]
    # For each leaf, exactly one active node should cover it.
    # owner_node[b, leaf] = argmax over nodes of active*cover
    scores = active.to(torch.float32).unsqueeze(-1) * leaf_cover.to(torch.float32).unsqueeze(0)
    owner_node = scores.argmax(dim=1)  # [B, Leaves]

    # Slot rank among active nodes (stable by node id order)
    # active_rank[b, node] = number of active nodes with id < node
    active_f = active.to(torch.float32)
    csum = torch.cumsum(active_f, dim=-1)
    rank = csum - active_f  # rank starting at 0 for first active
    # Gather rank for owner
    owner_slot = rank.gather(1, owner_node)
    return owner_node, owner_slot.to(torch.long)


def scatter_pool_temporal(
    full_btnc: torch.Tensor,
    owner_slot_t: torch.Tensor,
    n_slots: int,
) -> torch.Tensor:
    """Pool [B,H,N,C] over temporal slots → [B,Tslots,N,C] via scatter mean."""
    b, h, n, c = full_btnc.shape
    device = full_btnc.device
    dtype = full_btnc.dtype
    # Flatten H into scatter index
    flat = full_btnc.permute(0, 2, 3, 1).reshape(b * n * c, h)
    idx = owner_slot_t.to(torch.long)  # [B, H]
    idx_exp = idx.unsqueeze(1).unsqueeze(1).expand(b, n, c, h).reshape(b * n * c, h)
    out = torch.zeros(b * n * c, n_slots, device=device, dtype=dtype)
    cnt = torch.zeros(b * n * c, n_slots, device=device, dtype=dtype)
    out.scatter_add_(1, idx_exp, flat)
    ones = torch.ones_like(flat)
    cnt.scatter_add_(1, idx_exp, ones)
    out = out / cnt.clamp_min(1.0)
    return out.reshape(b, n, c, n_slots).permute(0, 3, 1, 2).contiguous()


def scatter_pool_spatial(
    full_btsc: torch.Tensor,
    owner_slot_s: torch.Tensor,
    n_slots: int,
) -> torch.Tensor:
    """Pool [B,T,N,C] over spatial slots → [B,T,Sslots,C]."""
    b, t, n, c = full_btsc.shape
    device = full_btsc.device
    dtype = full_btsc.dtype
    flat = full_btsc.permute(0, 1, 3, 2).reshape(b * t * c, n)
    idx = owner_slot_s.to(torch.long)  # [B, N]
    idx_exp = idx.unsqueeze(1).unsqueeze(1).expand(b, t, c, n).reshape(b * t * c, n)
    out = torch.zeros(b * t * c, n_slots, device=device, dtype=dtype)
    cnt = torch.zeros(b * t * c, n_slots, device=device, dtype=dtype)
    out.scatter_add_(1, idx_exp, flat)
    ones = torch.ones_like(flat)
    cnt.scatter_add_(1, idx_exp, ones)
    out = out / cnt.clamp_min(1.0)
    return out.reshape(b, t, c, n_slots).permute(0, 1, 3, 2).contiguous()


def scatter_pool_full(
    full: torch.Tensor,
    owner_slot_t: torch.Tensor,
    owner_slot_s: torch.Tensor,
    t_slots: int,
    s_slots: int,
) -> torch.Tensor:
    """Pool [B,H,N,C] → [B,T,S,C] with scatter means (linear in H,N)."""
    mid = scatter_pool_temporal(full, owner_slot_t, t_slots)
    return scatter_pool_spatial(mid, owner_slot_s, s_slots)


def gather_lift_full(
    coarse: torch.Tensor,
    owner_slot_t: torch.Tensor,
    owner_slot_s: torch.Tensor,
    horizon: int,
    n_nodes: int,
) -> torch.Tensor:
    """Lift [B,T,S,C] → [B,H,N,C] by gathering owner slots."""
    b, t_slots, s_slots, c = coarse.shape
    # For each (h,n): value = coarse[b, slot_t[h], slot_s[n]]
    # Gather temporal: [B,H,S,C]
    slot_t = owner_slot_t.clamp(0, t_slots - 1)  # [B,H]
    idx_t = slot_t.view(b, horizon, 1, 1).expand(b, horizon, s_slots, c)
    tmp = torch.gather(coarse, 1, idx_t)  # [B,H,S,C]
    slot_s = owner_slot_s.clamp(0, s_slots - 1)  # [B,N]
    idx_s = slot_s.view(b, 1, n_nodes, 1).expand(b, horizon, n_nodes, c)
    return torch.gather(tmp, 2, idx_s)


def build_sparse_region_edges(
    edge_index: torch.Tensor,
    owner_slot_s: torch.Tensor,
    s_slots: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map sparse node edges → unique region-region edges.

    Args:
        edge_index: [2, E] long
        owner_slot_s: [B, N] long
        s_slots: int

    Returns:
        region_edges: [B, 2, E'] packed source/dest region slots (may contain duplicates)
        edge_mask: [B, E'] True if src_region != dst_region or self-loop kept
    """
    src = edge_index[0]  # [E]
    dst = edge_index[1]
    src_r = owner_slot_s[:, src]  # [B, E]
    dst_r = owner_slot_s[:, dst]
    region_edges = torch.stack([src_r, dst_r], dim=1)
    edge_mask = torch.ones(owner_slot_s.shape[0], src.shape[0], device=owner_slot_s.device, dtype=torch.bool)
    return region_edges, edge_mask


def apply_batched_splits(
    frontier: torch.Tensor,
    split_hard: torch.Tensor,
    left_child: torch.Tensor,
    right_child: torch.Tensor,
    is_leaf: torch.Tensor,
) -> torch.Tensor:
    """Apply hard splits on frontier without Python sample loops.

    frontier/split_hard: [B, Ntree]
    left/right_child: [Ntree] long (-1 if none)
    is_leaf: [Ntree] bool
    """
    allow = (frontier > 0.5) & (~is_leaf.unsqueeze(0)) & (split_hard > 0.5)
    allow_f = allow.to(frontier.dtype)
    # Deactivate split parents
    new_f = frontier * (1.0 - allow_f)
    # Activate children via scatter_add of allow onto child indices
    b, n = frontier.shape
    # Only nodes with valid children
    valid = left_child >= 0
    src_ids = torch.arange(n, device=frontier.device)
    # For each parent i with allow[b,i], add to left[i] and right[i]
    # Use index_add on flattened [B*N]
    left = left_child.clamp_min(0)
    right = right_child.clamp_min(0)
    # Build contribution [B, N]
    left_add = torch.zeros_like(frontier)
    right_add = torch.zeros_like(frontier)
    # Scatter allow_f along child dim: left_add[b, left[i]] += allow_f[b, i]
    # Use for loop over tree nodes is OK (static tree size, not batch) — but user
    # forbids loops over N in hot path. Tree N is static metadata size; K=2 stages.
    # Prefer vectorized: one-hot child selection.
    eye = torch.eye(n, device=frontier.device, dtype=frontier.dtype)
    left_oh = eye[left]  # [N, N] rows=parent -> left child onehot; invalid parents map 0
    right_oh = eye[right]
    left_oh = left_oh * valid.unsqueeze(-1).to(frontier.dtype)
    right_oh = right_oh * valid.unsqueeze(-1).to(frontier.dtype)
    # allow_f [B,N] @ left_oh [N,N] -> [B,N]
    left_add = allow_f @ left_oh
    right_add = allow_f @ right_oh
    new_f = new_f + left_add + right_add
    return new_f.clamp(0.0, 1.0)
