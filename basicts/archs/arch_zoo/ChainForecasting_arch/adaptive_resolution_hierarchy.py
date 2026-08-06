"""Automatic temporal/spatial resolution trees and pool/lift operators."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass
class TreeNode:
    node_id: int
    parent_id: int | None
    left_child_id: int | None
    right_child_id: int | None
    start: int | None = None  # temporal inclusive
    end: int | None = None  # temporal exclusive
    original_node_indices: list[int] = field(default_factory=list)
    depth: int = 0
    leaf_count: int = 1
    is_leaf: bool = True


class TemporalResolutionTree:
    """Binary interval tree over future horizon ``[0, H)`` auto-built from H."""

    def __init__(self, horizon: int):
        if int(horizon) < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        self.horizon = int(horizon)
        self.nodes: list[TreeNode] = []
        self.root_id = 0
        self._build(0, self.horizon, parent_id=None, depth=0)
        self.depth = max(n.depth for n in self.nodes)
        self.num_leaves = sum(1 for n in self.nodes if n.is_leaf)
        if self.num_leaves != self.horizon:
            raise RuntimeError(
                f"Temporal tree leaves {self.num_leaves} != horizon {self.horizon}"
            )

    def _build(self, start: int, end: int, parent_id: int | None, depth: int) -> int:
        node_id = len(self.nodes)
        self.nodes.append(
            TreeNode(
                node_id=node_id,
                parent_id=parent_id,
                left_child_id=None,
                right_child_id=None,
                start=start,
                end=end,
                original_node_indices=list(range(start, end)),
                depth=depth,
                leaf_count=end - start,
                is_leaf=(end - start == 1),
            )
        )
        if end - start == 1:
            return node_id
        mid = start + (end - start) // 2
        # Odd lengths: left/right differ by at most 1.
        left_id = self._build(start, mid, node_id, depth + 1)
        right_id = self._build(mid, end, node_id, depth + 1)
        self.nodes[node_id].left_child_id = left_id
        self.nodes[node_id].right_child_id = right_id
        self.nodes[node_id].is_leaf = False
        return node_id

    def summary(self) -> dict[str, Any]:
        return {
            "horizon": self.horizon,
            "num_nodes": len(self.nodes),
            "num_leaves": self.num_leaves,
            "depth": self.depth,
            "root_id": self.root_id,
        }


def _symmetrize_adj(adj: np.ndarray) -> np.ndarray:
    a = np.asarray(adj, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"adjacency must be square, got {a.shape}")
    a = 0.5 * (a + a.T)
    np.fill_diagonal(a, np.maximum(np.diag(a), 1.0))
    return a


def _fiedler_bisect(indices: list[int], adj: np.ndarray) -> tuple[list[int], list[int], str]:
    """Deterministic balanced bisection; returns (left, right, method)."""
    idx = list(indices)
    m = len(idx)
    if m < 2:
        raise ValueError("cannot bisect fewer than 2 nodes")
    if m == 2:
        return [idx[0]], [idx[1]], "pair"

    sub = adj[np.ix_(idx, idx)].copy()
    deg = sub.sum(axis=1)
    method = "fiedler"
    try:
        # Normalized Laplacian
        d_inv_sqrt = np.zeros_like(deg)
        pos = deg > 1e-12
        d_inv_sqrt[pos] = 1.0 / np.sqrt(deg[pos])
        dmat = np.diag(d_inv_sqrt)
        lap = np.eye(m) - dmat @ sub @ dmat
        # Symmetric eigh
        vals, vecs = np.linalg.eigh(0.5 * (lap + lap.T))
        # Fiedler: second smallest eigenvector
        order = np.argsort(vals)
        fiedler = vecs[:, order[1]]
        # Median split for balance
        med = np.median(fiedler)
        left_local = [i for i, v in enumerate(fiedler) if v <= med]
        right_local = [i for i, v in enumerate(fiedler) if v > med]
        if not left_local or not right_local:
            # Fallback: sorted by fiedler then half
            order_i = np.argsort(fiedler)
            half = m // 2
            left_local = order_i[:half].tolist()
            right_local = order_i[half:].tolist()
            method = "fiedler_half_fallback"
        left = [idx[i] for i in left_local]
        right = [idx[i] for i in right_local]
        # Ensure mutual exclusive cover
        left_set, right_set = set(left), set(right)
        if left_set & right_set or left_set | right_set != set(idx):
            raise RuntimeError("invalid fiedler partition")
        # Rebalance if |L-R| > 1 by moving boundary along sorted fiedler
        if abs(len(left) - len(right)) > 1:
            order_i = np.argsort(fiedler)
            half = m // 2
            left = [idx[i] for i in order_i[:half]]
            right = [idx[i] for i in order_i[half:]]
            method = "fiedler_rebalanced"
        return left, right, method
    except Exception:
        # Deterministic balanced index fallback (sorted original indices)
        ordered = sorted(idx)
        half = m // 2
        return ordered[:half], ordered[half:], "balanced_index_fallback"


class SpatialResolutionTree:
    """Full binary spatial hierarchy over N nodes from adjacency (seed=0)."""

    BUILDER_VERSION = "spatial_full_bisection_v1"

    def __init__(
        self,
        adjacency: np.ndarray | torch.Tensor,
        clustering_seed: int = 0,
        cache_dir: str | Path | None = None,
        dataset_name: str = "synthetic",
    ):
        if isinstance(adjacency, torch.Tensor):
            adj_np = adjacency.detach().cpu().numpy()
        else:
            adj_np = np.asarray(adjacency)
        self.adj = _symmetrize_adj(adj_np)
        self.n_nodes = int(self.adj.shape[0])
        self.clustering_seed = int(clustering_seed)
        self.dataset_name = str(dataset_name)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.nodes: list[TreeNode] = []
        self.root_id = 0
        self.fallback_split_count = 0
        self.split_methods: list[str] = []

        cached = self._try_load_cache()
        if cached is not None:
            self.nodes = cached["nodes"]
            self.fallback_split_count = int(cached["fallback_split_count"])
            self.split_methods = list(cached["split_methods"])
        else:
            rng_state = np.random.get_state()
            np.random.seed(self.clustering_seed)  # unused; keep deterministic env
            try:
                self._build(list(range(self.n_nodes)), parent_id=None, depth=0)
            finally:
                np.random.set_state(rng_state)
            self._save_cache()

        self.depth = max(n.depth for n in self.nodes) if self.nodes else 0
        self.num_leaves = sum(1 for n in self.nodes if n.is_leaf)
        self._validate()

    def _cache_key(self) -> str:
        adj_hash = hashlib.sha1(self.adj.tobytes()).hexdigest()[:16]
        raw = (
            f"{self.dataset_name}|N{self.n_nodes}|{self.BUILDER_VERSION}|"
            f"s{self.clustering_seed}|{adj_hash}"
        )
        return hashlib.sha1(raw.encode()).hexdigest()[:16]

    def _cache_path(self) -> Path | None:
        if self.cache_dir is None:
            return None
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        return self.cache_dir / (
            f"{self.dataset_name}_N{self.n_nodes}_{self.BUILDER_VERSION}_"
            f"s{self.clustering_seed}_{self._cache_key()}.json"
        )

    def _try_load_cache(self) -> dict | None:
        path = self._cache_path()
        if path is None or not path.is_file():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        nodes = [TreeNode(**nd) for nd in data["nodes"]]
        return {
            "nodes": nodes,
            "fallback_split_count": data.get("fallback_split_count", 0),
            "split_methods": data.get("split_methods", []),
        }

    def _save_cache(self) -> None:
        path = self._cache_path()
        if path is None:
            return
        payload = {
            "nodes": [nd.__dict__ for nd in self.nodes],
            "fallback_split_count": self.fallback_split_count,
            "split_methods": self.split_methods,
            "builder_version": self.BUILDER_VERSION,
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    def _build(self, indices: list[int], parent_id: int | None, depth: int) -> int:
        node_id = len(self.nodes)
        indices = sorted(indices)
        self.nodes.append(
            TreeNode(
                node_id=node_id,
                parent_id=parent_id,
                left_child_id=None,
                right_child_id=None,
                original_node_indices=list(indices),
                depth=depth,
                leaf_count=len(indices),
                is_leaf=(len(indices) == 1),
            )
        )
        if len(indices) == 1:
            return node_id
        left, right, method = _fiedler_bisect(indices, self.adj)
        self.split_methods.append(method)
        if "fallback" in method:
            self.fallback_split_count += 1
        left_id = self._build(left, node_id, depth + 1)
        right_id = self._build(right, node_id, depth + 1)
        self.nodes[node_id].left_child_id = left_id
        self.nodes[node_id].right_child_id = right_id
        self.nodes[node_id].is_leaf = False
        return node_id

    def _validate(self) -> None:
        if self.num_leaves != self.n_nodes:
            raise RuntimeError(
                f"Spatial tree leaves {self.num_leaves} != N {self.n_nodes}"
            )
        leaves = [n for n in self.nodes if n.is_leaf]
        covered = []
        for n in leaves:
            covered.extend(n.original_node_indices)
        if sorted(covered) != list(range(self.n_nodes)):
            raise RuntimeError("Spatial leaf coverage incomplete or overlapping")

    def balance_stats(self) -> dict[str, float]:
        ratios = []
        for n in self.nodes:
            if n.is_leaf:
                continue
            l = self.nodes[n.left_child_id].leaf_count
            r = self.nodes[n.right_child_id].leaf_count
            ratios.append(min(l, r) / max(l, r))
        if not ratios:
            return {"mean_balance": 1.0, "min_balance": 1.0}
        return {
            "mean_balance": float(np.mean(ratios)),
            "min_balance": float(np.min(ratios)),
        }

    def summary(self) -> dict[str, Any]:
        bal = self.balance_stats()
        return {
            "n_nodes": self.n_nodes,
            "num_tree_nodes": len(self.nodes),
            "num_leaves": self.num_leaves,
            "depth": self.depth,
            "fallback_split_count": self.fallback_split_count,
            "coverage_ok": True,
            "overlap_ok": True,
            **bal,
            "cache_key": self._cache_key(),
        }


def build_membership_matrix(tree_nodes: list[TreeNode], num_leaves: int, kind: str) -> torch.Tensor:
    """Return ``[num_tree_nodes, num_leaves]`` soft membership (uniform over members)."""
    m = torch.zeros(len(tree_nodes), num_leaves, dtype=torch.float32)
    for n in tree_nodes:
        if kind == "temporal":
            members = list(range(n.start, n.end))
        else:
            members = list(n.original_node_indices)
        if not members:
            continue
        w = 1.0 / len(members)
        for j in members:
            m[n.node_id, j] = w
    return m


def build_frontier_projections(
    membership: torch.Tensor,
    frontier_mask: torch.Tensor,
    max_active: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build padded P/L for a batch of frontiers.

    Args:
        membership: [Tnodes, L]
        frontier_mask: [B, Tnodes] bool/float
        max_active: pad width (typically L)

    Returns:
        P: [B, max_active, L]
        Lmat: [B, L, max_active]
        active_mask: [B, max_active]
    """
    b, n_tree = frontier_mask.shape
    leaf = membership.shape[1]
    device = membership.device
    dtype = membership.dtype
    p = torch.zeros(b, max_active, leaf, device=device, dtype=dtype)
    lift = torch.zeros(b, leaf, max_active, device=device, dtype=dtype)
    active_mask = torch.zeros(b, max_active, device=device, dtype=dtype)
    for bi in range(b):
        ids = torch.nonzero(frontier_mask[bi] > 0.5, as_tuple=False).flatten().tolist()
        if len(ids) > max_active:
            raise RuntimeError(
                f"active units {len(ids)} exceed max_active {max_active}"
            )
        for slot, nid in enumerate(ids):
            row = membership[nid]
            p[bi, slot] = row
            # Lifting: each leaf gets the value of its active region (copy)
            members = torch.nonzero(row > 0, as_tuple=False).flatten()
            lift[bi, members, slot] = 1.0
            active_mask[bi, slot] = 1.0
    return p, lift, active_mask


def pool_full_to_resolution(
    full_tensor: torch.Tensor,
    temporal_projection: torch.Tensor,
    spatial_projection: torch.Tensor,
) -> torch.Tensor:
    """Pool ``[B,H,N,C]`` to ``[B,Tpad,Spad,C]`` via P_t / P_s."""
    # full: B H N C; P_t: B T H; P_s: B S N
    x = torch.einsum("bth,bhnc->btnc", temporal_projection, full_tensor)
    y = torch.einsum("bsn,btnc->btsc", spatial_projection, x)
    return y


def lift_resolution_to_full(
    coarse_tensor: torch.Tensor,
    temporal_lifting: torch.Tensor,
    spatial_lifting: torch.Tensor,
) -> torch.Tensor:
    """Lift ``[B,Tpad,Spad,C]`` to ``[B,H,N,C]``."""
    # coarse B T S C; L_t: B H T; L_s: B N S
    x = torch.einsum("bht,btsc->bhsc", temporal_lifting, coarse_tensor)
    y = torch.einsum("bns,bhsc->bhnc", spatial_lifting, x)
    return y


def validate_projection_row_sums(p: torch.Tensor, active_mask: torch.Tensor, atol: float = 1e-5) -> bool:
    """Each active slot's projection weights should sum to 1."""
    sums = p.sum(dim=-1)  # [B, max]
    active = active_mask > 0.5
    if not active.any():
        return True
    return bool(torch.allclose(sums[active], torch.ones_like(sums[active]), atol=atol))
