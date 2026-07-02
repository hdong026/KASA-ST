"""Fixed graph cluster assignments for graph-resolution spatial residualization."""
from __future__ import annotations

import hashlib
import json
import os
import pickle
from pathlib import Path

import numpy as np
import torch

from basicts.archs.arch_zoo.ChainForecasting_arch.gcn import load_adj_from_pickle, normalize_adj


def resolve_graph_resolution_sizes(node_size: int, ratios: list[float]) -> list[int]:
    """Map ratios to cluster counts M_j with safeguards."""
    sizes: list[int] = []
    for ratio in ratios:
        m = int(round(float(ratio) * node_size))
        m = max(2, min(node_size, m))
        sizes.append(m)
    deduped: list[int] = []
    for m in sizes:
        if not deduped or deduped[-1] != m:
            deduped.append(m)
    if not deduped or deduped[-1] != node_size:
        if node_size in deduped[:-1]:
            deduped = [m for m in deduped if m != node_size]
        deduped.append(node_size)
    return deduped


def _normalized_laplacian(adj: np.ndarray) -> np.ndarray:
    adj = np.asarray(adj, dtype=np.float64)
    adj = adj + np.eye(adj.shape[0], dtype=np.float64)
    deg = adj.sum(axis=1)
    deg_inv_sqrt = np.power(deg, -0.5, where=deg > 0)
    deg_inv_sqrt[~np.isfinite(deg_inv_sqrt)] = 0.0
    d_mat = np.diag(deg_inv_sqrt)
    return d_mat @ adj @ d_mat


def _kmeans_numpy(features: np.ndarray, n_clusters: int, seed: int, max_iter: int = 100) -> np.ndarray:
    rng = np.random.RandomState(seed)
    n_samples = features.shape[0]
    if n_clusters >= n_samples:
        return np.arange(n_samples, dtype=np.int64) % n_clusters
    init_idx = rng.choice(n_samples, size=n_clusters, replace=False)
    centers = features[init_idx].copy()
    labels = np.zeros(n_samples, dtype=np.int64)
    for _ in range(max_iter):
        dists = ((features[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = dists.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for k in range(n_clusters):
            mask = labels == k
            if mask.any():
                centers[k] = features[mask].mean(axis=0)
            else:
                centers[k] = features[rng.randint(0, n_samples)]
    return labels


def _spectral_cluster_labels(adj: np.ndarray, n_clusters: int, seed: int) -> tuple[np.ndarray, str]:
    if n_clusters >= adj.shape[0]:
        labels = np.arange(adj.shape[0], dtype=np.int64) % max(n_clusters, 1)
        return labels, "trivial_modulo"

    try:
        from sklearn.cluster import SpectralClustering

        sc = SpectralClustering(
            n_clusters=n_clusters,
            affinity="precomputed",
            assign_labels="kmeans",
            random_state=seed,
            n_init=10,
        )
        adj_pos = np.maximum(adj, 0.0)
        labels = sc.fit_predict(adj_pos)
        return labels.astype(np.int64), "sklearn_spectral"
    except Exception:
        pass

    lap = _normalized_laplacian(adj)
    eigvals, eigvecs = np.linalg.eigh(lap)
    k = min(n_clusters, eigvecs.shape[1])
    features = eigvecs[:, :k]
    labels = _kmeans_numpy(features, n_clusters, seed=seed)
    return labels.astype(np.int64), "numpy_laplacian_kmeans"


def _balanced_fallback_labels(node_size: int, n_clusters: int) -> tuple[np.ndarray, str]:
    labels = np.arange(node_size, dtype=np.int64) % n_clusters
    return labels, "balanced_modulo_fallback"


def labels_to_assignment(labels: np.ndarray, node_size: int, num_clusters: int) -> np.ndarray:
    c = np.zeros((node_size, num_clusters), dtype=np.float32)
    c[np.arange(node_size), labels] = 1.0
    return c


def assignment_to_projection(c: np.ndarray) -> np.ndarray:
    """P = D^{-1} C^T with D = diag(C^T 1), shape [M, N]."""
    counts = c.sum(axis=0).clip(min=1.0)
    return (c.T / counts[:, None]).astype(np.float32)


def coarse_adjacency(adj: np.ndarray, c: np.ndarray) -> np.ndarray:
    a_coarse = c.T @ adj @ c
    np.fill_diagonal(a_coarse, a_coarse.diagonal() + 1e-6)
    return a_coarse.astype(np.float32)


def _cache_key(
    dataset_name: str,
    node_size: int,
    num_clusters: int,
    adj_mx_path: str | None,
    seed: int,
) -> str:
    adj_tag = "no_adj"
    if adj_mx_path and os.path.exists(adj_mx_path):
        stat = os.stat(adj_mx_path)
        adj_tag = f"{Path(adj_mx_path).name}_{stat.st_mtime_ns}_{stat.st_size}"
    raw = json.dumps(
        {
            "dataset": dataset_name,
            "node_size": node_size,
            "num_clusters": num_clusters,
            "adj": adj_tag,
            "seed": seed,
        },
        sort_keys=True,
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def default_cache_dir() -> Path:
    root = Path(__file__).resolve().parents[4]
    cache = root / "generated" / "cache" / "graph_clusters"
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def build_cluster_assignment(
    node_size: int,
    num_clusters: int,
    adj_mx_path: str | None,
    seed: int = 0,
    dataset_name: str = "unknown",
) -> dict:
    if num_clusters >= node_size:
        labels = np.arange(node_size, dtype=np.int64)
        method = "identity"
        c = labels_to_assignment(labels, node_size, node_size)
    else:
        adj_np = None
        if adj_mx_path and os.path.exists(adj_mx_path):
            adj_t = load_adj_from_pickle(adj_mx_path)
            if adj_t is not None and adj_t.shape[0] == node_size:
                adj_np = adj_t.detach().cpu().numpy()
        if adj_np is not None:
            labels, method = _spectral_cluster_labels(adj_np, num_clusters, seed=seed)
        else:
            labels, method = _balanced_fallback_labels(node_size, num_clusters)
        c = labels_to_assignment(labels, node_size, num_clusters)

    p = assignment_to_projection(c)
    out = {
        "node_size": node_size,
        "num_clusters": num_clusters,
        "clustering_method": method,
        "C": c,
        "P": p,
        "labels": labels if num_clusters < node_size else np.arange(node_size),
    }
    if adj_mx_path and os.path.exists(adj_mx_path) and num_clusters < node_size:
        adj_t = load_adj_from_pickle(adj_mx_path)
        if adj_t is not None:
            out["coarse_adj"] = coarse_adjacency(adj_t.numpy(), c)
    return out


def load_or_build_cluster_assignment(
    node_size: int,
    num_clusters: int,
    adj_mx_path: str | None,
    seed: int = 0,
    dataset_name: str = "unknown",
    cache_dir: str | Path | None = None,
) -> tuple[dict, Path]:
    cache_root = Path(cache_dir) if cache_dir else default_cache_dir()
    cache_root.mkdir(parents=True, exist_ok=True)
    key = _cache_key(dataset_name, node_size, num_clusters, adj_mx_path, seed)
    cache_path = cache_root / f"{dataset_name}_N{node_size}_M{num_clusters}_s{seed}_{key}.npz"
    if cache_path.is_file():
        data = np.load(cache_path, allow_pickle=True)
        meta = {
            "node_size": int(data["node_size"]),
            "num_clusters": int(data["num_clusters"]),
            "clustering_method": str(data["clustering_method"]),
            "C": data["C"].astype(np.float32),
            "P": data["P"].astype(np.float32),
            "labels": data["labels"],
        }
        if "coarse_adj" in data:
            meta["coarse_adj"] = data["coarse_adj"].astype(np.float32)
        return meta, cache_path

    meta = build_cluster_assignment(
        node_size=node_size,
        num_clusters=num_clusters,
        adj_mx_path=adj_mx_path,
        seed=seed,
        dataset_name=dataset_name,
    )
    save_obj = {
        "node_size": meta["node_size"],
        "num_clusters": meta["num_clusters"],
        "clustering_method": meta["clustering_method"],
        "C": meta["C"],
        "P": meta["P"],
        "labels": meta["labels"],
    }
    if "coarse_adj" in meta:
        save_obj["coarse_adj"] = meta["coarse_adj"]
    np.savez_compressed(cache_path, **save_obj)
    return meta, cache_path


def register_cluster_buffers(module: torch.nn.Module, prefix: str, meta: dict) -> None:
    module.register_buffer(f"{prefix}_C", torch.from_numpy(meta["C"]))
    module.register_buffer(f"{prefix}_P", torch.from_numpy(meta["P"]))
    if "coarse_adj" in meta:
        module.register_buffer(f"{prefix}_coarse_adj", torch.from_numpy(meta["coarse_adj"]))
