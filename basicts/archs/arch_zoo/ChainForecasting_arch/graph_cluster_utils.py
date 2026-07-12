"""Fixed graph cluster assignments for graph-resolution spatial residualization."""
from __future__ import annotations

import hashlib
import json
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch

from basicts.archs.arch_zoo.ChainForecasting_arch.gcn import load_adj_from_pickle, normalize_adj

GRAPH_CLUSTER_METHOD_CURRENT = "current"
GRAPH_CLUSTER_METHODS = {
    GRAPH_CLUSTER_METHOD_CURRENT,
    "pearson_balanced_pam",
    "xcorr_balanced_pam",
    "joint_pearson_spatial_balanced_pam",
    "pearson_standard_pam",
    "autocorr_feature_balanced_pam",
}

PAM_BALANCED_METHODS = {
    "pearson_balanced_pam",
    "xcorr_balanced_pam",
    "joint_pearson_spatial_balanced_pam",
    "autocorr_feature_balanced_pam",
}
PAM_STANDARD_METHODS = {"pearson_standard_pam"}


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


def default_cache_dir() -> Path:
    root = Path(__file__).resolve().parents[4]
    cache = root / "generated" / "cache" / "graph_clusters"
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def default_pam_cache_dir() -> Path:
    root = Path(__file__).resolve().parents[4]
    cache = root / "cluster_cache"
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _resolve_path(path: str | Path | None) -> Path | None:
    if path is None or str(path).strip() == "":
        return None
    p = Path(path)
    if not p.is_absolute():
        p = _repo_root() / p
    return p


def load_train_node_series(
    dataset_name: str = "PEMS04",
    train_series_path: str | Path | None = None,
    data_dir: str | Path | None = None,
    input_len: int = 12,
    output_len: int = 12,
) -> np.ndarray:
    """Load train-split node flow series as [T_train, N] (channel 0)."""
    custom = _resolve_path(train_series_path)
    if custom is not None:
        if not custom.is_file():
            raise FileNotFoundError(
                f"cluster_train_series_path not found: {custom}. "
                "Provide a .npy/.npz/.pkl with shape [T, N] or [T, N, C]."
            )
        return _load_series_file(custom)

    if data_dir is None:
        data_dir = _repo_root() / "datasets" / dataset_name
    data_dir = _resolve_path(data_dir)
    assert data_dir is not None

    train_npy = data_dir / "train_data.npy"
    if train_npy.is_file():
        arr = np.asarray(np.load(train_npy), dtype=np.float64)
        if arr.ndim == 2:
            return arr
        if arr.ndim == 3:
            return arr[..., 0]

    data_pkl = data_dir / f"data_in{input_len}_out{output_len}.pkl"
    index_pkl = data_dir / f"index_in{input_len}_out{output_len}.pkl"
    if data_pkl.is_file() and index_pkl.is_file():
        with open(data_pkl, "rb") as f:
            data_obj = pickle.load(f)
        with open(index_pkl, "rb") as f:
            index_obj = pickle.load(f)
        processed = np.asarray(data_obj["processed_data"], dtype=np.float64)
        if processed.ndim == 3:
            flow = processed[..., 0]
        elif processed.ndim == 2:
            flow = processed
        else:
            raise ValueError(f"Unexpected processed_data shape: {processed.shape}")
        train_rows = index_obj.get("train")
        if train_rows is None:
            raise KeyError(
                f"No 'train' split in {index_pkl}. Set cluster_train_series_path explicitly."
            )
        t_min = min(int(r[0]) for r in train_rows)
        t_max = max(int(r[2]) for r in train_rows)
        return flow[t_min:t_max]

    raise FileNotFoundError(
        f"Cannot load train node series for {dataset_name} from {data_dir}. "
        "Expected train_data.npy or data/index pkl pair. "
        "Set cluster_train_series_path to a [T, N] series file."
    )


def _load_series_file(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(path)
    elif suffix == ".npz":
        data = np.load(path)
        if "series" in data:
            arr = data["series"]
        elif "train_series" in data:
            arr = data["train_series"]
        else:
            key = list(data.keys())[0]
            arr = data[key]
    elif suffix == ".pkl":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, dict):
            for key in ("series", "train_series", "processed_data", "data"):
                if key in obj:
                    arr = obj[key]
                    break
            else:
                raise KeyError(f"No known series key in pickle {path}")
        else:
            arr = obj
    else:
        raise ValueError(f"Unsupported train series file: {path}")

    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"Train series must be [T, N], got {arr.shape} from {path}")
    return arr


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 2 or b.size < 2:
        return 0.0
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    if denom < 1e-12 or not np.isfinite(denom):
        return 0.0
    val = float((a * b).sum() / denom)
    if not np.isfinite(val):
        return 0.0
    return val


def pearson_distance_matrix(series: np.ndarray) -> np.ndarray:
    """Pearson distance D_ij = 1 - |corr(x_i, x_j)|."""
    series = np.asarray(series, dtype=np.float64)
    if series.ndim != 2:
        raise ValueError(f"series must be [T, N], got {series.shape}")
    x = series - series.mean(axis=0, keepdims=True)
    std = series.std(axis=0, keepdims=True)
    std[std < 1e-12] = 1.0
    x = x / std
    corr = (x.T @ x) / max(x.shape[0] - 1, 1)
    corr = np.clip(corr, -1.0, 1.0)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    dist = 1.0 - np.abs(corr)
    np.fill_diagonal(dist, 0.0)
    return dist.astype(np.float64)


def xcorr_distance_matrix(series: np.ndarray, max_lag: int = 12) -> np.ndarray:
    """Normalized cross-correlation distance D_ij = 1 - max_tau |corr(x_i, x_j shifted by tau)|."""
    series = np.asarray(series, dtype=np.float64)
    t_len, n = series.shape
    x = series - series.mean(axis=0, keepdims=True)
    std = series.std(axis=0, keepdims=True)
    std[std < 1e-12] = 1.0
    x = x / std

    best = np.zeros((n, n), dtype=np.float64)
    for tau in range(-max_lag, max_lag + 1):
        if tau >= 0:
            a = x[: t_len - tau] if tau else x
            b = x[tau:]
        else:
            a = x[-tau:]
            b = x[: t_len + tau]
        seg_len = a.shape[0]
        if seg_len < 2:
            continue
        corr = (a.T @ b) / float(seg_len)
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        corr = np.clip(np.abs(corr), 0.0, 1.0)
        best = np.maximum(best, corr)

    dist = 1.0 - best
    np.fill_diagonal(dist, 0.0)
    return dist


def autocorr_features(series: np.ndarray, acf_lag: int = 24) -> np.ndarray:
    series = np.asarray(series, dtype=np.float64)
    n = series.shape[1]
    feats = np.zeros((n, acf_lag), dtype=np.float64)
    for i in range(n):
        x = series[:, i]
        std = x.std()
        x = (x - x.mean()) / std if std > 1e-12 else x * 0.0
        for lag in range(1, acf_lag + 1):
            if lag >= x.shape[0]:
                feats[i, lag - 1] = 0.0
            else:
                feats[i, lag - 1] = _safe_corr(x[:-lag], x[lag:])
    return feats


def autocorr_feature_distance_matrix(series: np.ndarray, acf_lag: int = 24) -> np.ndarray:
    feats = autocorr_features(series, acf_lag=acf_lag)
    feats = feats - feats.mean(axis=1, keepdims=True)
    std = feats.std(axis=1, keepdims=True)
    std[std < 1e-12] = 1.0
    feats = feats / std
    corr = (feats @ feats.T) / max(feats.shape[1], 1)
    corr = np.clip(corr, -1.0, 1.0)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    dist = 1.0 - np.abs(corr)
    np.fill_diagonal(dist, 0.0)
    return dist.astype(np.float64)


def load_spatial_distance_matrix(
    spatial_path: str | Path,
    node_size: int,
) -> np.ndarray:
    """Load pairwise spatial distance and normalize to [0, 1]."""
    path = _resolve_path(spatial_path)
    if path is None or not path.is_file():
        raise FileNotFoundError(
            f"cluster_spatial_coord_path not found: {spatial_path}. "
            "Joint temporal-spatial clustering requires an N×N distance/coordinate file."
        )

    if path.suffix.lower() == ".pkl":
        obj = pickle.load(open(path, "rb"))
        if hasattr(obj, "detach"):
            mat = obj.detach().cpu().numpy()
        else:
            mat = np.asarray(obj)
    elif path.suffix.lower() == ".npy":
        mat = np.load(path)
    elif path.suffix.lower() == ".npz":
        data = np.load(path)
        key = "distance" if "distance" in data else list(data.keys())[0]
        mat = data[key]
    elif path.suffix.lower() == ".csv":
        import pandas as pd

        df = pd.read_csv(path)
        mat = np.full((node_size, node_size), np.nan, dtype=np.float64)
        cols = {c.lower(): c for c in df.columns}
        if not {"from", "to"}.issubset(cols):
            raise ValueError(f"CSV must have from/to columns: {path}")
        cost_col = cols.get("cost") or cols.get("distance") or list(df.columns)[-1]
        for _, row in df.iterrows():
            i = int(row[cols["from"]])
            j = int(row[cols["to"]])
            mat[i, j] = float(row[cost_col])
            mat[j, i] = float(row[cost_col])
    else:
        raise ValueError(f"Unsupported spatial distance file: {path}")

    mat = np.asarray(mat, dtype=np.float64)
    if mat.shape != (node_size, node_size):
        raise ValueError(f"Spatial matrix shape {mat.shape} != ({node_size}, {node_size})")

    mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
    max_val = float(mat.max())
    if max_val <= 1e-12:
        raise ValueError(f"Spatial distance matrix is all zeros: {path}")
    dist = mat / max_val
    np.fill_diagonal(dist, 0.0)
    return dist.astype(np.float64)


def joint_distance_matrix(
    temporal_dist: np.ndarray,
    spatial_dist: np.ndarray,
    lambda_s: float = 0.2,
) -> np.ndarray:
    lam = float(lambda_s)
    lam = min(max(lam, 0.0), 1.0)
    joint = (1.0 - lam) * temporal_dist + lam * spatial_dist
    np.fill_diagonal(joint, 0.0)
    return joint.astype(np.float64)


def cluster_capacities(node_size: int, num_clusters: int) -> np.ndarray:
    base = node_size // num_clusters
    rem = node_size % num_clusters
    caps = np.full(num_clusters, base, dtype=np.int64)
    if rem > 0:
        caps[:rem] = base + 1
    return caps


def _pam_assignment_cost(dist: np.ndarray, labels: np.ndarray, medoids: np.ndarray) -> float:
    med_map = {int(m): k for k, m in enumerate(medoids)}
    cost = 0.0
    for i, lab in enumerate(labels):
        cost += float(dist[i, medoids[int(lab)]])
    return cost


def _assign_nearest(dist: np.ndarray, medoids: np.ndarray) -> np.ndarray:
    sub = dist[:, medoids]
    return sub.argmin(axis=1).astype(np.int64)


def _pam_build(dist: np.ndarray, num_clusters: int, seed: int, candidate_limit: int = 80) -> np.ndarray:
    n = dist.shape[0]
    rng = np.random.RandomState(seed)
    medoids = [int(dist.sum(axis=1).argmin())]
    while len(medoids) < num_clusters:
        remaining = [i for i in range(n) if i not in medoids]
        if len(remaining) > candidate_limit:
            remaining = list(rng.choice(remaining, size=candidate_limit, replace=False))
        best_cand = None
        best_cost = float("inf")
        trial_medoids = np.asarray(medoids, dtype=np.int64)
        for cand in remaining:
            trial = np.append(trial_medoids, cand)
            labels = _assign_nearest(dist, trial)
            cost = _pam_assignment_cost(dist, labels, trial)
            if cost < best_cost:
                best_cost = cost
                best_cand = cand
        if best_cand is None:
            best_cand = int(rng.choice([i for i in range(n) if i not in medoids]))
        medoids.append(best_cand)
        trial_medoids = np.asarray(medoids, dtype=np.int64)
    return np.asarray(medoids, dtype=np.int64)


def _balanced_assign_greedy(
    dist: np.ndarray,
    medoids: np.ndarray,
    capacities: np.ndarray,
) -> np.ndarray:
    n = dist.shape[0]
    k = len(medoids)
    labels = np.full(n, -1, dtype=np.int64)
    remaining = capacities.astype(np.int64).copy()
    order = np.argsort(dist[:, medoids].min(axis=1))
    for node in order:
        cand_order = np.argsort(dist[node, medoids])
        assigned = False
        for c_idx in cand_order:
            if remaining[c_idx] > 0:
                labels[node] = c_idx
                remaining[c_idx] -= 1
                assigned = True
                break
        if not assigned:
            raise RuntimeError("Balanced PAM greedy assignment failed: no capacity left.")
    if (labels < 0).any():
        raise RuntimeError("Balanced PAM greedy assignment left unassigned nodes.")
    return labels


def _balanced_assign(
    dist: np.ndarray,
    medoids: np.ndarray,
    capacities: np.ndarray,
    use_optimal: bool = False,
) -> np.ndarray:
    n = dist.shape[0]
    slot_cluster: list[int] = []
    for c_idx, cap in enumerate(capacities):
        slot_cluster.extend([c_idx] * int(cap))
    total_slots = len(slot_cluster)
    if total_slots != n:
        raise ValueError(f"Capacity sum {total_slots} != node_size {n}")

    if not use_optimal:
        return _balanced_assign_greedy(dist, medoids, capacities)

    cost = np.zeros((n, total_slots), dtype=np.float64)
    for slot_idx, c_idx in enumerate(slot_cluster):
        cost[:, slot_idx] = dist[:, medoids[c_idx]]

    try:
        from scipy.optimize import linear_sum_assignment

        row_ind, col_ind = linear_sum_assignment(cost)
        labels = np.zeros(n, dtype=np.int64)
        for r, c in zip(row_ind, col_ind):
            labels[r] = slot_cluster[c]
        return labels
    except Exception:
        return _balanced_assign_greedy(dist, medoids, capacities)


def pam_standard(
    dist: np.ndarray,
    num_clusters: int,
    seed: int = 0,
    max_iter: int = 100,
    swap_candidates: int = 50,
) -> tuple[np.ndarray, np.ndarray, float]:
    dist = np.asarray(dist, dtype=np.float64)
    n = dist.shape[0]
    if num_clusters >= n:
        labels = np.arange(n, dtype=np.int64)
        return labels, np.arange(n, dtype=np.int64), 0.0

    try:
        from sklearn_extra.cluster import KMedoids

        km = KMedoids(
            n_clusters=num_clusters,
            metric="precomputed",
            method="pam",
            init="heuristic",
            max_iter=max_iter,
            random_state=seed,
        )
        labels = km.fit_predict(dist)
        medoids = km.medoid_indices_.astype(np.int64)
        cost = _pam_assignment_cost(dist, labels, medoids)
        return labels.astype(np.int64), medoids, cost
    except Exception:
        pass

    medoids = _pam_build(dist, num_clusters, seed=seed)
    labels = _assign_nearest(dist, medoids)
    best_cost = _pam_assignment_cost(dist, labels, medoids)

    non_medoids = [i for i in range(n) if i not in set(medoids.tolist())]
    for _ in range(max_iter):
        improved = False
        rng = np.random.RandomState(seed)
        cand_non = non_medoids
        if len(cand_non) > swap_candidates:
            cand_non = list(rng.choice(cand_non, size=swap_candidates, replace=False))
        best_swap = None
        best_swap_cost = best_cost
        for m_pos, m_node in enumerate(medoids):
            for h_node in cand_non:
                trial_medoids = medoids.copy()
                trial_medoids[m_pos] = h_node
                trial_labels = _assign_nearest(dist, trial_medoids)
                trial_cost = _pam_assignment_cost(dist, trial_labels, trial_medoids)
                if trial_cost + 1e-9 < best_swap_cost:
                    best_swap_cost = trial_cost
                    best_swap = (m_pos, h_node, trial_medoids.copy(), trial_labels.copy())
        if best_swap is None:
            break
        m_pos, h_node, medoids, labels = best_swap[0], best_swap[1], best_swap[2], best_swap[3]
        best_cost = best_swap_cost
        improved = True
        non_medoids = [i for i in range(n) if i not in set(medoids.tolist())]
        if not improved:
            break
    return labels, medoids, best_cost


def pam_balanced(
    dist: np.ndarray,
    num_clusters: int,
    seed: int = 0,
    max_iter: int = 15,
    swap_candidates: int = 16,
) -> tuple[np.ndarray, np.ndarray, float]:
    dist = np.asarray(dist, dtype=np.float64)
    n = dist.shape[0]
    if num_clusters >= n:
        labels = np.arange(n, dtype=np.int64)
        return labels, np.arange(n, dtype=np.int64), 0.0

    capacities = cluster_capacities(n, num_clusters)
    medoids = _pam_build(dist, num_clusters, seed=seed)
    labels = _balanced_assign(dist, medoids, capacities)
    best_cost = _pam_assignment_cost(dist, labels, medoids)

    non_medoids = [i for i in range(n) if i not in set(medoids.tolist())]
    for _ in range(max_iter):
        improved = False
        rng = np.random.RandomState(seed)
        cand_non = non_medoids
        if len(cand_non) > swap_candidates:
            cand_non = list(rng.choice(cand_non, size=swap_candidates, replace=False))
        for m_pos, _ in enumerate(medoids):
            for h_node in cand_non:
                trial_medoids = medoids.copy()
                trial_medoids[m_pos] = h_node
                try:
                    trial_labels = _balanced_assign(dist, trial_medoids, capacities)
                except Exception:
                    continue
                trial_cost = _pam_assignment_cost(dist, trial_labels, trial_medoids)
                if trial_cost + 1e-9 < best_cost:
                    medoids, labels = trial_medoids, trial_labels
                    best_cost = trial_cost
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
        non_medoids = [i for i in range(n) if i not in set(medoids.tolist())]

    sizes = np.bincount(labels, minlength=num_clusters)
    expected = cluster_capacities(n, num_clusters)
    if not np.array_equal(np.sort(sizes), np.sort(expected)):
        raise RuntimeError(
            f"Balanced PAM capacity violation: sizes={sizes.tolist()} expected={expected.tolist()}"
        )
    return labels, medoids, best_cost


def build_distance_matrix_for_method(
    method: str,
    train_series: np.ndarray,
    node_size: int,
    cluster_max_lag: int = 12,
    cluster_lambda_s: float = 0.2,
    cluster_acf_lag: int = 24,
    cluster_spatial_coord_path: str | Path | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    method = str(method).lower()
    summary: dict[str, Any] = {"method": method}

    if method in {"pearson_balanced_pam", "pearson_standard_pam"}:
        dist = pearson_distance_matrix(train_series)
        summary["distance_type"] = "pearson"
    elif method == "xcorr_balanced_pam":
        dist = xcorr_distance_matrix(train_series, max_lag=cluster_max_lag)
        summary["distance_type"] = "xcorr"
        summary["max_lag"] = cluster_max_lag
    elif method == "autocorr_feature_balanced_pam":
        dist = autocorr_feature_distance_matrix(train_series, acf_lag=cluster_acf_lag)
        summary["distance_type"] = "autocorr_feature"
        summary["acf_lag"] = cluster_acf_lag
    elif method == "joint_pearson_spatial_balanced_pam":
        if cluster_spatial_coord_path is None:
            raise ValueError(
                "joint_pearson_spatial_balanced_pam requires cluster_spatial_coord_path "
                "(N×N distance matrix or coordinate-derived distances)."
            )
        d_time = pearson_distance_matrix(train_series)
        d_spatial = load_spatial_distance_matrix(cluster_spatial_coord_path, node_size)
        dist = joint_distance_matrix(d_time, d_spatial, lambda_s=cluster_lambda_s)
        summary["distance_type"] = "joint_pearson_spatial"
        summary["lambda_s"] = cluster_lambda_s
        summary["spatial_path"] = str(cluster_spatial_coord_path)
    else:
        raise ValueError(f"Unsupported PAM distance method: {method}")

    if dist.shape != (node_size, node_size):
        raise ValueError(f"Distance matrix shape {dist.shape} != ({node_size}, {node_size})")
    summary["distance_mean"] = float(dist[np.triu_indices(node_size, k=1)].mean())
    summary["distance_std"] = float(dist[np.triu_indices(node_size, k=1)].std())
    return dist, summary


def cluster_labels_from_method(
    dist: np.ndarray,
    method: str,
    num_clusters: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, float, str]:
    method = str(method).lower()
    if method in PAM_STANDARD_METHODS:
        labels, medoids, cost = pam_standard(dist, num_clusters, seed=seed)
        return labels, medoids, cost, "standard_pam"
    if method in PAM_BALANCED_METHODS:
        labels, medoids, cost = pam_balanced(dist, num_clusters, seed=seed)
        return labels, medoids, cost, "balanced_pam"
    raise ValueError(f"Unknown PAM clustering method: {method}")


def validate_cluster_assignment(meta: dict) -> dict[str, Any]:
    c = np.asarray(meta["C"], dtype=np.float64)
    p = np.asarray(meta["P"], dtype=np.float64)
    labels = np.asarray(meta["labels"])
    n, m = c.shape
    row_sums = c.sum(axis=1)
    p_row_sums = p.sum(axis=1)
    sizes = np.bincount(labels.astype(np.int64), minlength=m)
    return {
        "C_shape": list(c.shape),
        "P_shape": list(p.shape),
        "row_one_hot_ok": bool(np.allclose(row_sums, 1.0)),
        "P_row_sum_ok": bool(np.allclose(p_row_sums, 1.0)),
        "cluster_sizes": sizes.tolist(),
        "min_cluster_size": int(sizes.min()) if sizes.size else 0,
        "max_cluster_size": int(sizes.max()) if sizes.size else 0,
        "mean_cluster_size": float(sizes.mean()) if sizes.size else 0.0,
        "std_cluster_size": float(sizes.std()) if sizes.size else 0.0,
    }


def _cache_key(
    dataset_name: str,
    node_size: int,
    num_clusters: int,
    adj_mx_path: str | None,
    seed: int,
    graph_cluster_method: str = GRAPH_CLUSTER_METHOD_CURRENT,
    extra: dict | None = None,
) -> str:
    adj_tag = "no_adj"
    if adj_mx_path and os.path.exists(adj_mx_path):
        stat = os.stat(adj_mx_path)
        adj_tag = f"{Path(adj_mx_path).name}_{stat.st_mtime_ns}_{stat.st_size}"
    payload = {
        "dataset": dataset_name,
        "node_size": node_size,
        "num_clusters": num_clusters,
        "adj": adj_tag,
        "seed": seed,
        "graph_cluster_method": graph_cluster_method,
    }
    if extra:
        payload.update(extra)
    raw = json.dumps(payload, sort_keys=True)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _pam_cache_path(
    cache_root: Path,
    dataset_name: str,
    method: str,
    num_clusters: int,
    seed: int,
    key: str,
    extra_tag: str = "",
) -> Path:
    method_dir = cache_root / dataset_name / method
    method_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{extra_tag}" if extra_tag else ""
    return method_dir / f"M{num_clusters}_seed{seed}{tag}_{key}.npz"


def build_cluster_assignment(
    node_size: int,
    num_clusters: int,
    adj_mx_path: str | None,
    seed: int = 0,
    dataset_name: str = "unknown",
    graph_cluster_method: str = GRAPH_CLUSTER_METHOD_CURRENT,
    cluster_train_series_path: str | Path | None = None,
    cluster_spatial_coord_path: str | Path | None = None,
    cluster_max_lag: int = 12,
    cluster_lambda_s: float = 0.2,
    cluster_acf_lag: int = 24,
    data_dir: str | Path | None = None,
) -> dict:
    method = str(graph_cluster_method).lower()
    if method not in GRAPH_CLUSTER_METHODS:
        raise ValueError(
            f"Unknown graph_cluster_method={graph_cluster_method}. "
            f"Choices: {sorted(GRAPH_CLUSTER_METHODS)}"
        )

    if num_clusters >= node_size:
        labels = np.arange(node_size, dtype=np.int64)
        clustering_method = "identity"
        c = labels_to_assignment(labels, node_size, node_size)
    elif method == GRAPH_CLUSTER_METHOD_CURRENT:
        adj_np = None
        if adj_mx_path and os.path.exists(adj_mx_path):
            adj_t = load_adj_from_pickle(adj_mx_path)
            if adj_t is not None and adj_t.shape[0] == node_size:
                adj_np = adj_t.detach().cpu().numpy()
        if adj_np is not None:
            labels, clustering_method = _spectral_cluster_labels(adj_np, num_clusters, seed=seed)
        else:
            labels, clustering_method = _balanced_fallback_labels(node_size, num_clusters)
        c = labels_to_assignment(labels, node_size, num_clusters)
        medoids = None
        pam_cost = None
        distance_summary = {}
    else:
        train_series = load_train_node_series(
            dataset_name=dataset_name,
            train_series_path=cluster_train_series_path,
            data_dir=data_dir,
        )
        if train_series.shape[1] != node_size:
            raise ValueError(
                f"Train series N={train_series.shape[1]} != node_size={node_size}"
            )
        dist, distance_summary = build_distance_matrix_for_method(
            method=method,
            train_series=train_series,
            node_size=node_size,
            cluster_max_lag=cluster_max_lag,
            cluster_lambda_s=cluster_lambda_s,
            cluster_acf_lag=cluster_acf_lag,
            cluster_spatial_coord_path=cluster_spatial_coord_path,
        )
        labels, medoids, pam_cost, pam_type = cluster_labels_from_method(
            dist, method, num_clusters, seed=seed
        )
        clustering_method = f"{method}_{pam_type}"
        c = labels_to_assignment(labels, node_size, num_clusters)

    p = assignment_to_projection(c)
    out: dict[str, Any] = {
        "node_size": node_size,
        "num_clusters": num_clusters,
        "clustering_method": clustering_method,
        "graph_cluster_method": method,
        "C": c,
        "P": p,
        "labels": labels if num_clusters < node_size else np.arange(node_size),
    }
    if method != GRAPH_CLUSTER_METHOD_CURRENT and num_clusters < node_size:
        out["medoids"] = medoids
        out["pam_cost"] = pam_cost
        out["distance_summary"] = distance_summary
        out["capacities"] = cluster_capacities(node_size, num_clusters)
    if adj_mx_path and os.path.exists(adj_mx_path) and num_clusters < node_size:
        adj_t = load_adj_from_pickle(adj_mx_path)
        if adj_t is not None:
            out["coarse_adj"] = coarse_adjacency(adj_t.numpy(), c)
    out["validation"] = validate_cluster_assignment(out)
    return out


def load_or_build_cluster_assignment(
    node_size: int,
    num_clusters: int,
    adj_mx_path: str | None,
    seed: int = 0,
    dataset_name: str = "unknown",
    cache_dir: str | Path | None = None,
    graph_cluster_method: str = GRAPH_CLUSTER_METHOD_CURRENT,
    cluster_train_series_path: str | Path | None = None,
    cluster_spatial_coord_path: str | Path | None = None,
    cluster_max_lag: int = 12,
    cluster_lambda_s: float = 0.2,
    cluster_acf_lag: int = 24,
    data_dir: str | Path | None = None,
) -> tuple[dict, Path]:
    method = str(graph_cluster_method).lower()
    extra = {}
    extra_tag = ""
    if method != GRAPH_CLUSTER_METHOD_CURRENT:
        if method == "xcorr_balanced_pam":
            extra["max_lag"] = cluster_max_lag
            extra_tag = f"lag{cluster_max_lag}"
        elif method == "joint_pearson_spatial_balanced_pam":
            extra["lambda_s"] = cluster_lambda_s
            sp = _resolve_path(cluster_spatial_coord_path)
            extra["spatial"] = str(sp) if sp else "none"
            extra_tag = f"ls{cluster_lambda_s}"
        elif method == "autocorr_feature_balanced_pam":
            extra["acf_lag"] = cluster_acf_lag
            extra_tag = f"acf{cluster_acf_lag}"

    if method == GRAPH_CLUSTER_METHOD_CURRENT:
        cache_root = Path(cache_dir) if cache_dir else default_cache_dir()
        cache_root.mkdir(parents=True, exist_ok=True)
        key = _cache_key(dataset_name, node_size, num_clusters, adj_mx_path, seed, method, extra)
        cache_path = cache_root / f"{dataset_name}_N{node_size}_M{num_clusters}_s{seed}_{key}.npz"
    else:
        cache_root = Path(cache_dir) if cache_dir else default_pam_cache_dir()
        key = _cache_key(
            dataset_name,
            node_size,
            num_clusters,
            adj_mx_path,
            seed,
            method,
            extra,
        )
        cache_path = _pam_cache_path(
            cache_root, dataset_name, method, num_clusters, seed, key, extra_tag=extra_tag
        )

    if cache_path.is_file():
        data = np.load(cache_path, allow_pickle=True)
        meta: dict[str, Any] = {
            "node_size": int(data["node_size"]),
            "num_clusters": int(data["num_clusters"]),
            "clustering_method": str(data["clustering_method"]),
            "graph_cluster_method": str(data.get("graph_cluster_method", method)),
            "C": data["C"].astype(np.float32),
            "P": data["P"].astype(np.float32),
            "labels": data["labels"],
        }
        for optional in ("coarse_adj", "medoids", "pam_cost", "distance_summary", "capacities", "validation"):
            if optional in data:
                val = data[optional]
                if isinstance(val, np.ndarray) and val.shape == ():
                    val = val.item()
                meta[optional] = val
        return meta, cache_path

    meta = build_cluster_assignment(
        node_size=node_size,
        num_clusters=num_clusters,
        adj_mx_path=adj_mx_path,
        seed=seed,
        dataset_name=dataset_name,
        graph_cluster_method=method,
        cluster_train_series_path=cluster_train_series_path,
        cluster_spatial_coord_path=cluster_spatial_coord_path,
        cluster_max_lag=cluster_max_lag,
        cluster_lambda_s=cluster_lambda_s,
        cluster_acf_lag=cluster_acf_lag,
        data_dir=data_dir,
    )
    save_obj: dict[str, Any] = {
        "node_size": meta["node_size"],
        "num_clusters": meta["num_clusters"],
        "clustering_method": meta["clustering_method"],
        "graph_cluster_method": meta.get("graph_cluster_method", method),
        "C": meta["C"],
        "P": meta["P"],
        "labels": meta["labels"],
    }
    for optional in ("coarse_adj", "medoids", "pam_cost", "distance_summary", "capacities", "validation"):
        if optional in meta:
            save_obj[optional] = meta[optional]
    np.savez_compressed(cache_path, **save_obj)
    return meta, cache_path


def register_cluster_buffers(module: torch.nn.Module, prefix: str, meta: dict) -> None:
    module.register_buffer(f"{prefix}_C", torch.from_numpy(meta["C"]))
    module.register_buffer(f"{prefix}_P", torch.from_numpy(meta["P"]))
    if "coarse_adj" in meta:
        module.register_buffer(f"{prefix}_coarse_adj", torch.from_numpy(meta["coarse_adj"]))


def summarize_cluster_quality(meta: dict, dist: np.ndarray | None = None) -> dict[str, Any]:
    labels = np.asarray(meta["labels"])
    val_raw = meta.get("validation")
    if isinstance(val_raw, np.ndarray):
        val_raw = val_raw.item()
    val = val_raw if isinstance(val_raw, dict) else validate_cluster_assignment(meta)
    out = {
        "num_clusters": int(meta["num_clusters"]),
        "clustering_method": meta.get("clustering_method", ""),
        "graph_cluster_method": meta.get("graph_cluster_method", ""),
        "min_cluster_size": val["min_cluster_size"],
        "max_cluster_size": val["max_cluster_size"],
        "mean_cluster_size": val["mean_cluster_size"],
        "std_cluster_size": val["std_cluster_size"],
        "medoid_count": int(len(meta["medoids"])) if "medoids" in meta else 0,
    }
    if dist is not None:
        intra = []
        corr_abs = []
        for k in range(int(meta["num_clusters"])):
            idx = np.where(labels == k)[0]
            if idx.size < 2:
                continue
            sub = dist[np.ix_(idx, idx)]
            tri = sub[np.triu_indices(sub.shape[0], k=1)]
            if tri.size:
                intra.append(float(tri.mean()))
            if meta.get("graph_cluster_method", "").startswith("pearson"):
                corr_abs.append(float((1.0 - tri).mean()))
        out["mean_intra_distance"] = float(np.mean(intra)) if intra else None
        out["mean_intra_abs_corr"] = float(np.mean(corr_abs)) if corr_abs else None
    return out
