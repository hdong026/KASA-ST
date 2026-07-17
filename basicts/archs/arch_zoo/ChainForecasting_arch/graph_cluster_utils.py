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
GRAPH_CLUSTER_METHOD_ROAD_SPECTRAL = "gr17_road_spectral"
GRAPH_CLUSTER_METHOD_CONSTRAINED_SPECTRAL = "gr18_constrained_spectral_cap_dist"
GRAPH_CLUSTER_METHOD_CAP_KMEANS_SPECTRAL = "gr19_spectral_constrained_kmeans_cap"
GRAPH_CLUSTER_METHOD_CAP_KMEANS_ADJ = "gr19a_cap_only_spectral"
GRAPH_CLUSTER_METHOD_CAP_KMEANS_ROAD = "gr19b_road_cap_spectral"
GRAPH_CLUSTER_METHOD_CAP_ONLY_SPECTRAL = "cap_only_spectral"
GRAPH_CLUSTER_METHOD_CAPDIST_SPECTRAL_PAIR = "capdist_spectral_pair"
GRAPH_CLUSTER_METHOD_CAPDIST_SPECTRAL = "capdist_spectral"
GRAPH_CLUSTER_METHOD_GRACLUS = "gr20_graclus_matching_4_2_1"
GRAPH_CLUSTER_METHOD_ROAD_GRACLUS = "gr21_road_graclus_matching_4_2_1"

GRAPH_CLUSTER_METHODS = {
    GRAPH_CLUSTER_METHOD_CURRENT,
    GRAPH_CLUSTER_METHOD_ROAD_SPECTRAL,
    GRAPH_CLUSTER_METHOD_CONSTRAINED_SPECTRAL,
    GRAPH_CLUSTER_METHOD_CAP_KMEANS_SPECTRAL,
    GRAPH_CLUSTER_METHOD_CAP_KMEANS_ADJ,
    GRAPH_CLUSTER_METHOD_CAP_KMEANS_ROAD,
    GRAPH_CLUSTER_METHOD_CAP_ONLY_SPECTRAL,
    GRAPH_CLUSTER_METHOD_CAPDIST_SPECTRAL_PAIR,
    GRAPH_CLUSTER_METHOD_CAPDIST_SPECTRAL,
    GRAPH_CLUSTER_METHOD_GRACLUS,
    GRAPH_CLUSTER_METHOD_ROAD_GRACLUS,
    "pearson_balanced_pam",
    "xcorr_balanced_pam",
    "joint_pearson_spatial_balanced_pam",
    "pearson_standard_pam",
    "autocorr_feature_balanced_pam",
}

ROAD_DISTANCE_METHODS = {
    GRAPH_CLUSTER_METHOD_ROAD_SPECTRAL,
    GRAPH_CLUSTER_METHOD_CONSTRAINED_SPECTRAL,
    GRAPH_CLUSTER_METHOD_ROAD_GRACLUS,
}

CAPACITY_MULTILEVEL_METHODS = {
    GRAPH_CLUSTER_METHOD_GRACLUS,
    GRAPH_CLUSTER_METHOD_ROAD_GRACLUS,
}

CAPACITY_SINGLE_STAGE_METHODS = {
    GRAPH_CLUSTER_METHOD_CAP_ONLY_SPECTRAL,
}

CAPACITY_CAPDIST_METHODS = {
    GRAPH_CLUSTER_METHOD_CAPDIST_SPECTRAL,
}

SPECTRAL_EMBEDDING_METHODS = {
    GRAPH_CLUSTER_METHOD_CONSTRAINED_SPECTRAL,
    GRAPH_CLUSTER_METHOD_CAP_KMEANS_SPECTRAL,
    GRAPH_CLUSTER_METHOD_CAP_KMEANS_ADJ,
    GRAPH_CLUSTER_METHOD_CAP_KMEANS_ROAD,
    GRAPH_CLUSTER_METHOD_CAP_ONLY_SPECTRAL,
}

DEFAULT_ROAD_DISTANCE_PATHS: dict[str, str] = {
    "PEMS04": "datasets/raw_data/PEMS04/adj_PEMS04_distance.pkl",
    "PEMS-BAY": "datasets/raw_data/PEMS-BAY/sensor_graph/distances_bay_2017.csv",
}

PAM_BALANCED_METHODS = {
    "pearson_balanced_pam",
    "xcorr_balanced_pam",
    "joint_pearson_spatial_balanced_pam",
    "autocorr_feature_balanced_pam",
}
PAM_STANDARD_METHODS = {"pearson_standard_pam"}


def resolve_graph_resolution_sizes(
    node_size: int,
    ratios: list[float],
    skip_final_identity: bool = False,
) -> list[int]:
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
    if skip_final_identity:
        return deduped
    if not deduped or deduped[-1] != node_size:
        if node_size in deduped[:-1]:
            deduped = [m for m in deduped if m != node_size]
        deduped.append(node_size)
    return deduped


def resolve_graph_resolution_capacities(
    capacities: list[int],
    skip_final_identity: bool = False,
) -> list[int]:
    """Return capacity schedule coarse-to-fine; append 1 unless skip_final_identity."""
    caps = [int(c) for c in capacities if int(c) > 1]
    deduped: list[int] = []
    for c in caps:
        if not deduped or deduped[-1] != c:
            deduped.append(c)
    if skip_final_identity:
        return deduped
    if not deduped or deduped[-1] != 1:
        deduped.append(1)
    return deduped


def num_clusters_for_capacity(node_size: int, max_capacity: int) -> int:
    """M = ceil(N / K) for capacity K."""
    k = max(1, int(max_capacity))
    return (int(node_size) + k - 1) // k


def cluster_graph_adjacency_normalized(
    adj_sym: np.ndarray,
    p: np.ndarray,
    c: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """Row-normalized cluster graph: A_cluster = Normalize(P @ A_sym @ C), [M, M]."""
    a = np.asarray(p, dtype=np.float64) @ np.asarray(adj_sym, dtype=np.float64) @ np.asarray(c, dtype=np.float64)
    row_sum = a.sum(axis=1, keepdims=True) + eps
    return (a / row_sum).astype(np.float32)


def row_normalize_square(adj: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    a = np.asarray(adj, dtype=np.float64)
    row_sum = a.sum(axis=1, keepdims=True) + eps
    return (a / row_sum).astype(np.float32)


def build_distance_aware_affinity_soft(
    adj_sym: np.ndarray,
    dist_norm: np.ndarray | None,
    sigma_d: float = 0.5,
) -> tuple[np.ndarray, str, bool]:
    """Soft road-distance-aware affinity without hard cannot-link cutoff."""
    if dist_norm is not None:
        sigma = max(float(sigma_d), 1e-6)
        w = adj_sym * np.exp(-(dist_norm ** 2) / (sigma ** 2))
        np.fill_diagonal(w, 0.0)
        return np.maximum(w, 0.0).astype(np.float64), "distance_aware_affinity", True
    return np.asarray(adj_sym, dtype=np.float64), "adj_sym", False


def _cluster_medoids(
    features: np.ndarray,
    labels: np.ndarray,
    num_clusters: int,
    dist_norm: np.ndarray | None,
) -> np.ndarray:
    medoids = np.zeros(num_clusters, dtype=np.int64)
    for k in range(num_clusters):
        members = np.where(labels == k)[0]
        if members.size == 0:
            medoids[k] = int(np.argmin(((features - features.mean(axis=0)) ** 2).sum(axis=1)))
            continue
        if dist_norm is not None and members.size > 1:
            sub = dist_norm[np.ix_(members, members)]
            medoids[k] = int(members[int(np.argmin(sub.sum(axis=1)))])
        elif members.size == 1:
            medoids[k] = int(members[0])
        else:
            center = features[members].mean(axis=0)
            medoids[k] = int(members[int(np.argmin(((features[members] - center) ** 2).sum(axis=1)))])
    return medoids


def _capdist_pair_assign_slots(
    features: np.ndarray,
    centers: np.ndarray,
    medoids: np.ndarray,
    max_capacity: int,
    dist_norm: np.ndarray | None,
    lambda_d: float,
) -> np.ndarray:
    n, m = features.shape[0], centers.shape[0]
    slots = m * int(max_capacity)
    emb_cost = ((features[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    cost = np.zeros((n, slots), dtype=np.float64)
    for cl in range(m):
        node_cost = emb_cost[:, cl]
        if dist_norm is not None and float(lambda_d) > 0:
            node_cost = node_cost + float(lambda_d) * dist_norm[:, medoids[cl]]
        for s in range(int(max_capacity)):
            cost[:, cl * int(max_capacity) + s] = node_cost
    try:
        from scipy.optimize import linear_sum_assignment

        _, col_ind = linear_sum_assignment(cost)
        return (col_ind // int(max_capacity)).astype(np.int64)
    except Exception:
        labels = np.full(n, -1, dtype=np.int64)
        remaining = np.full(m, int(max_capacity), dtype=np.int64)
        flat_cost = cost.min(axis=1)
        order = np.argsort(flat_cost)
        for node in order:
            slot_order = np.argsort(cost[node])
            for slot in slot_order:
                cl = int(slot // int(max_capacity))
                if remaining[cl] <= 0:
                    continue
                labels[node] = cl
                remaining[cl] -= 1
                break
            if labels[node] < 0:
                raise RuntimeError("CapDist greedy slot assignment failed.")
        return labels


def _capdist_pair_constrained_kmeans(
    features: np.ndarray,
    num_clusters: int,
    max_capacity: int,
    dist_norm: np.ndarray | None,
    lambda_d: float,
    seed: int,
    max_iter: int = 20,
) -> tuple[np.ndarray, str]:
    centers = _kmeans_centers(features, num_clusters, seed=seed)
    labels = np.zeros(features.shape[0], dtype=np.int64)
    medoids = _cluster_medoids(features, labels, num_clusters, dist_norm)
    tag = "capdist_slot_hungarian"
    for it in range(max_iter):
        new_labels = _capdist_pair_assign_slots(
            features, centers, medoids, max_capacity, dist_norm, lambda_d
        )
        if np.array_equal(new_labels, labels) and it > 0:
            break
        labels = new_labels
        for k in range(num_clusters):
            mask = labels == k
            if mask.any():
                centers[k] = features[mask].mean(axis=0)
        medoids = _cluster_medoids(features, labels, num_clusters, dist_norm)
    return labels, tag


def build_capdist_spectral_assignment(
    node_size: int,
    adj_mx_path: str | None,
    max_capacity: int,
    seed: int = 0,
    dataset_name: str = "PEMS04",
    cluster_road_distance_path: str | Path | None = None,
    cluster_sigma_d: float = 0.5,
    cluster_lambda_d: float = 0.1,
    use_road_distance: bool = True,
    graph_cluster_method: str = GRAPH_CLUSTER_METHOD_CAPDIST_SPECTRAL,
) -> dict[str, Any]:
    """Capacity + soft distance constrained spectral clustering (no hard cannot-link)."""
    adj_sym = symmetrize_adjacency(load_raw_adj_numpy(adj_mx_path, node_size))
    dist_norm = None
    road_path = ""
    w, affinity_source, road_used = build_distance_aware_affinity_soft(adj_sym, None, cluster_sigma_d)
    if use_road_distance:
        try:
            dist_raw, road_path = load_road_distance_matrix(
                node_size,
                dataset_name=dataset_name,
                cluster_road_distance_path=cluster_road_distance_path,
            )
            dist_norm = normalize_road_distance_matrix(dist_raw)
            w, affinity_source, road_used = build_distance_aware_affinity_soft(
                adj_sym, dist_norm, cluster_sigma_d
            )
            print(
                f"[CapDist] road_distance_used=True affinity_source={affinity_source} "
                f"path={road_path}"
            )
        except (FileNotFoundError, ValueError) as exc:
            w, affinity_source, road_used = build_distance_aware_affinity_soft(
                adj_sym, None, cluster_sigma_d
            )
            print(
                f"[CapDist] road_distance_used=False affinity_source=adj_sym "
                f"(fallback: {exc})"
            )
    else:
        print("[CapDist] road_distance_used=False affinity_source=adj_sym (disabled)")

    cap_k = max(1, int(max_capacity))
    num_clusters = num_clusters_for_capacity(node_size, cap_k)
    embed_dim = min(num_clusters, node_size - 1)
    z = _spectral_embedding(w, embed_dim)
    labels, sub_method = _capdist_pair_constrained_kmeans(
        z,
        num_clusters,
        cap_k,
        dist_norm=dist_norm,
        lambda_d=cluster_lambda_d,
        seed=seed,
    )
    c = labels_to_assignment(labels, node_size, num_clusters)
    p = assignment_to_projection(c)
    val = validate_cluster_assignment(
        {"C": c, "P": p, "labels": labels, "num_clusters": num_clusters}
    )
    if val["max_cluster_size"] > cap_k:
        raise RuntimeError(f"CapDist cluster size exceeds {cap_k}: {val}")
    if not val["row_one_hot_ok"]:
        raise RuntimeError("CapDist assignment rows are not one-hot.")
    method_tag = (
        "capdist_spectral_pair"
        if graph_cluster_method == GRAPH_CLUSTER_METHOD_CAPDIST_SPECTRAL_PAIR
        else "capdist_spectral"
    )
    return {
        "node_size": node_size,
        "num_clusters": num_clusters,
        "clustering_method": f"{method_tag}_{sub_method}",
        "graph_cluster_method": graph_cluster_method,
        "C": c,
        "P": p,
        "labels": labels,
        "affinity_W": w.astype(np.float32),
        "adj_sym": adj_sym.astype(np.float32),
        "road_distance_used": road_used,
        "affinity_source": affinity_source,
        "road_distance_path": road_path,
        "sigma_d": float(cluster_sigma_d),
        "lambda_d": float(cluster_lambda_d),
        "max_capacity": cap_k,
        "capacity": cap_k,
        "validation": val,
    }


def build_capdist_spectral_pair_assignment(
    node_size: int,
    adj_mx_path: str | None,
    seed: int = 0,
    dataset_name: str = "PEMS04",
    cluster_road_distance_path: str | Path | None = None,
    cluster_sigma_d: float = 0.5,
    cluster_lambda_d: float = 0.1,
    use_road_distance: bool = True,
) -> dict[str, Any]:
    """S1/2 clustering: capacity K=2, M=ceil(N/2), soft distance penalty."""
    return build_capdist_spectral_assignment(
        node_size=node_size,
        adj_mx_path=adj_mx_path,
        max_capacity=2,
        seed=seed,
        dataset_name=dataset_name,
        cluster_road_distance_path=cluster_road_distance_path,
        cluster_sigma_d=cluster_sigma_d,
        cluster_lambda_d=cluster_lambda_d,
        use_road_distance=use_road_distance,
        graph_cluster_method=GRAPH_CLUSTER_METHOD_CAPDIST_SPECTRAL_PAIR,
    )


def load_or_build_capdist_spectral_cluster(
    node_size: int,
    max_capacity: int,
    adj_mx_path: str | None,
    seed: int = 0,
    dataset_name: str = "PEMS04",
    cache_dir: str | Path | None = None,
    cluster_road_distance_path: str | Path | None = None,
    cluster_sigma_d: float = 0.5,
    cluster_lambda_d: float = 0.1,
    use_road_distance: bool = True,
) -> tuple[dict[str, Any], Path]:
    cap_k = max(1, int(max_capacity))
    num_clusters = num_clusters_for_capacity(node_size, cap_k)
    extra = {
        "sigma_d": cluster_sigma_d,
        "lambda_d": cluster_lambda_d,
        "max_capacity": cap_k,
        "use_road": bool(use_road_distance),
        "road": str(
            _resolve_path(cluster_road_distance_path)
            or DEFAULT_ROAD_DISTANCE_PATHS.get(dataset_name, "none")
        ),
    }
    cache_root = Path(cache_dir) if cache_dir else default_cache_dir()
    cache_root.mkdir(parents=True, exist_ok=True)
    key = _cache_key(
        dataset_name,
        node_size,
        num_clusters,
        adj_mx_path,
        seed,
        GRAPH_CLUSTER_METHOD_CAPDIST_SPECTRAL,
        extra,
    )
    cache_path = cache_root / (
        f"{dataset_name}_N{node_size}_M{num_clusters}_K{cap_k}_capdist_s{seed}_{key}.npz"
    )
    if cache_path.is_file():
        data = np.load(cache_path, allow_pickle=True)
        meta: dict[str, Any] = {
            "node_size": int(data["node_size"]),
            "num_clusters": int(data["num_clusters"]),
            "clustering_method": str(data["clustering_method"]),
            "graph_cluster_method": GRAPH_CLUSTER_METHOD_CAPDIST_SPECTRAL,
            "C": data["C"].astype(np.float32),
            "P": data["P"].astype(np.float32),
            "labels": data["labels"],
            "affinity_W": data["affinity_W"].astype(np.float32),
            "road_distance_used": bool(data.get("road_distance_used", False)),
            "affinity_source": str(data.get("affinity_source", "unknown")),
            "sigma_d": float(data.get("sigma_d", cluster_sigma_d)),
            "lambda_d": float(data.get("lambda_d", cluster_lambda_d)),
            "max_capacity": cap_k,
            "capacity": cap_k,
        }
        if "validation" in data:
            val = data["validation"]
            meta["validation"] = val.item() if isinstance(val, np.ndarray) and val.shape == () else val
        return meta, cache_path

    meta = build_capdist_spectral_assignment(
        node_size=node_size,
        adj_mx_path=adj_mx_path,
        max_capacity=cap_k,
        seed=seed,
        dataset_name=dataset_name,
        cluster_road_distance_path=cluster_road_distance_path,
        cluster_sigma_d=cluster_sigma_d,
        cluster_lambda_d=cluster_lambda_d,
        use_road_distance=use_road_distance,
        graph_cluster_method=GRAPH_CLUSTER_METHOD_CAPDIST_SPECTRAL,
    )
    meta["resolution_tag"] = capacity_stage_tag(cap_k)
    np.savez_compressed(
        cache_path,
        node_size=meta["node_size"],
        num_clusters=meta["num_clusters"],
        clustering_method=meta["clustering_method"],
        graph_cluster_method=meta["graph_cluster_method"],
        C=meta["C"],
        P=meta["P"],
        labels=meta["labels"],
        affinity_W=meta["affinity_W"],
        road_distance_used=meta["road_distance_used"],
        affinity_source=meta["affinity_source"],
        sigma_d=meta["sigma_d"],
        lambda_d=meta["lambda_d"],
        max_capacity=cap_k,
        validation=meta["validation"],
    )
    return meta, cache_path


def load_or_build_capdist_half_cluster(
    node_size: int,
    adj_mx_path: str | None,
    seed: int = 0,
    dataset_name: str = "PEMS04",
    cache_dir: str | Path | None = None,
    cluster_road_distance_path: str | Path | None = None,
    cluster_sigma_d: float = 0.5,
    cluster_lambda_d: float = 0.1,
    use_road_distance: bool = True,
) -> tuple[dict[str, Any], Path]:
    num_clusters = num_clusters_for_capacity(node_size, 2)
    extra = {
        "sigma_d": cluster_sigma_d,
        "lambda_d": cluster_lambda_d,
        "use_road": bool(use_road_distance),
        "road": str(
            _resolve_path(cluster_road_distance_path)
            or DEFAULT_ROAD_DISTANCE_PATHS.get(dataset_name, "none")
        ),
    }
    cache_root = Path(cache_dir) if cache_dir else default_cache_dir()
    cache_root.mkdir(parents=True, exist_ok=True)
    key = _cache_key(
        dataset_name,
        node_size,
        num_clusters,
        adj_mx_path,
        seed,
        GRAPH_CLUSTER_METHOD_CAPDIST_SPECTRAL_PAIR,
        extra,
    )
    cache_path = cache_root / (
        f"{dataset_name}_N{node_size}_M{num_clusters}_capdist_s{seed}_{key}.npz"
    )
    if cache_path.is_file():
        data = np.load(cache_path, allow_pickle=True)
        meta: dict[str, Any] = {
            "node_size": int(data["node_size"]),
            "num_clusters": int(data["num_clusters"]),
            "clustering_method": str(data["clustering_method"]),
            "graph_cluster_method": GRAPH_CLUSTER_METHOD_CAPDIST_SPECTRAL_PAIR,
            "C": data["C"].astype(np.float32),
            "P": data["P"].astype(np.float32),
            "labels": data["labels"],
            "affinity_W": data["affinity_W"].astype(np.float32),
            "road_distance_used": bool(data.get("road_distance_used", False)),
            "affinity_source": str(data.get("affinity_source", "unknown")),
            "sigma_d": float(data.get("sigma_d", cluster_sigma_d)),
            "lambda_d": float(data.get("lambda_d", cluster_lambda_d)),
            "max_capacity": 2,
        }
        if "validation" in data:
            val = data["validation"]
            meta["validation"] = val.item() if isinstance(val, np.ndarray) and val.shape == () else val
        return meta, cache_path

    meta = build_capdist_spectral_pair_assignment(
        node_size=node_size,
        adj_mx_path=adj_mx_path,
        seed=seed,
        dataset_name=dataset_name,
        cluster_road_distance_path=cluster_road_distance_path,
        cluster_sigma_d=cluster_sigma_d,
        cluster_lambda_d=cluster_lambda_d,
        use_road_distance=use_road_distance,
    )
    np.savez_compressed(
        cache_path,
        node_size=meta["node_size"],
        num_clusters=meta["num_clusters"],
        clustering_method=meta["clustering_method"],
        graph_cluster_method=meta["graph_cluster_method"],
        C=meta["C"],
        P=meta["P"],
        labels=meta["labels"],
        affinity_W=meta["affinity_W"],
        road_distance_used=meta["road_distance_used"],
        affinity_source=meta["affinity_source"],
        sigma_d=meta["sigma_d"],
        lambda_d=meta["lambda_d"],
        validation=meta["validation"],
    )
    return meta, cache_path


def capacity_stage_tag(capacity: int) -> str:
    if capacity >= 4:
        return "S4"
    if capacity >= 2:
        return "S2"
    return "S1"


def _load_pickle_obj(path: str | Path) -> Any:
    with open(path, "rb") as f:
        try:
            return pickle.load(f)
        except UnicodeDecodeError:
            f.seek(0)
            return pickle.load(f, encoding="latin1")


def load_raw_adj_numpy(adj_mx_path: str | Path, node_size: int) -> np.ndarray:
    path = _resolve_path(adj_mx_path)
    if path is None or not path.is_file():
        raise FileNotFoundError(f"adj_mx_path not found: {adj_mx_path}")
    obj = _load_pickle_obj(path)
    if isinstance(obj, (list, tuple)):
        for item in reversed(obj):
            if hasattr(item, "shape") and len(item.shape) == 2:
                adj = np.asarray(item, dtype=np.float64)
                break
        else:
            raise ValueError(f"No 2D adjacency matrix in {path}")
    else:
        adj = np.asarray(obj, dtype=np.float64)
    if adj.shape != (node_size, node_size):
        raise ValueError(f"Adjacency shape {adj.shape} != ({node_size}, {node_size})")
    return adj


def symmetrize_adjacency(adj: np.ndarray) -> np.ndarray:
    adj = np.asarray(adj, dtype=np.float64)
    return 0.5 * (adj + adj.T)


def normalize_road_distance_matrix(dist: np.ndarray) -> np.ndarray:
    dist = np.asarray(dist, dtype=np.float64)
    dist = np.nan_to_num(dist, nan=np.inf, posinf=np.inf, neginf=0.0)
    np.fill_diagonal(dist, 0.0)
    finite = dist[np.isfinite(dist) & (dist > 0)]
    if finite.size == 0:
        raise ValueError("Road distance matrix has no finite positive off-diagonal entries.")
    scale = float(np.quantile(finite, 0.95))
    if scale <= 1e-12:
        raise ValueError("Road distance quantile scale is zero.")
    d_norm = np.clip(dist / scale, 0.0, 1.0)
    np.fill_diagonal(d_norm, 0.0)
    return d_norm.astype(np.float64)


def load_road_distance_matrix(
    node_size: int,
    dataset_name: str = "unknown",
    cluster_road_distance_path: str | Path | None = None,
) -> tuple[np.ndarray, str]:
    """Load road distance and normalize with 95th-percentile scaling."""
    path = _resolve_path(cluster_road_distance_path)
    if path is None:
        default_rel = DEFAULT_ROAD_DISTANCE_PATHS.get(dataset_name)
        if default_rel:
            path = _resolve_path(default_rel)
    if path is None or not path.is_file():
        raise FileNotFoundError(
            f"Road distance required but not found for dataset={dataset_name}. "
            f"Provide cluster_road_distance_path. Tried default={DEFAULT_ROAD_DISTANCE_PATHS.get(dataset_name)}"
        )

    if path.suffix.lower() == ".pkl":
        obj = _load_pickle_obj(path)
        if isinstance(obj, (list, tuple)):
            for item in reversed(obj):
                if hasattr(item, "shape") and len(item.shape) == 2:
                    mat = np.asarray(item, dtype=np.float64)
                    break
            else:
                raise ValueError(f"No distance matrix in pickle: {path}")
        else:
            mat = np.asarray(obj, dtype=np.float64)
    elif path.suffix.lower() in {".npy", ".npz"}:
        if path.suffix.lower() == ".npz":
            data = np.load(path)
            key = "distance" if "distance" in data else list(data.keys())[0]
            mat = np.asarray(data[key], dtype=np.float64)
        else:
            mat = np.asarray(np.load(path), dtype=np.float64)
    elif path.suffix.lower() == ".csv":
        import pandas as pd

        df = pd.read_csv(path)
        mat = np.full((node_size, node_size), np.inf, dtype=np.float64)
        cols = {c.lower(): c for c in df.columns}
        if {"from", "to"}.issubset(cols):
            cost_col = cols.get("cost") or cols.get("distance") or list(df.columns)[-1]
            for _, row in df.iterrows():
                i = int(row[cols["from"]])
                j = int(row[cols["to"]])
                w = float(row[cost_col])
                mat[i, j] = min(mat[i, j], w)
                mat[j, i] = min(mat[j, i], w)
        else:
            # sensor_id, sensor_id, distance layout
            for row in df.itertuples(index=False):
                if len(row) < 3:
                    continue
                i, j, w = int(row[0]), int(row[1]), float(row[2])
                if 0 <= i < node_size and 0 <= j < node_size:
                    mat[i, j] = min(mat[i, j], w)
                    mat[j, i] = min(mat[j, i], w)
    else:
        raise ValueError(f"Unsupported road distance file: {path}")

    if mat.shape != (node_size, node_size):
        raise ValueError(f"Road distance shape {mat.shape} != ({node_size}, {node_size})")
    return normalize_road_distance_matrix(mat), str(path)


def build_road_distance_affinity(
    adj_sym: np.ndarray,
    dist_norm: np.ndarray,
    sigma_d: float = 0.5,
    delta: float | None = None,
) -> np.ndarray:
    sigma = max(float(sigma_d), 1e-6)
    w = adj_sym * np.exp(-(dist_norm ** 2) / (sigma ** 2))
    if delta is not None:
        w = w * (dist_norm <= float(delta))
    np.fill_diagonal(w, 0.0)
    return np.maximum(w, 0.0).astype(np.float64)


def _spectral_embedding(affinity: np.ndarray, n_components: int) -> np.ndarray:
    lap = _normalized_laplacian(affinity)
    eigvals, eigvecs = np.linalg.eigh(lap)
    k = min(max(1, n_components), eigvecs.shape[1])
    return eigvecs[:, :k].astype(np.float64)


def _kmeans_centers(features: np.ndarray, n_clusters: int, seed: int, max_iter: int = 50) -> np.ndarray:
    labels = _kmeans_numpy(features, n_clusters, seed=seed, max_iter=max_iter)
    centers = np.zeros((n_clusters, features.shape[1]), dtype=np.float64)
    for k in range(n_clusters):
        mask = labels == k
        if mask.any():
            centers[k] = features[mask].mean(axis=0)
        else:
            centers[k] = features[np.random.RandomState(seed + k).randint(0, features.shape[0])]
    return centers


def _capacitated_assign_embedding(
    features: np.ndarray,
    centers: np.ndarray,
    max_capacity: int,
    dist_norm: np.ndarray | None = None,
    delta: float | None = None,
) -> np.ndarray:
    n = features.shape[0]
    k = centers.shape[0]
    labels = np.full(n, -1, dtype=np.int64)
    remaining = np.full(k, int(max_capacity), dtype=np.int64)
    members: list[list[int]] = [[] for _ in range(k)]
    dists = ((features[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    order = np.argsort(dists.min(axis=1))
    for node in order:
        cand = np.argsort(dists[node])
        assigned = False
        for c_idx in cand:
            if remaining[c_idx] <= 0:
                continue
            if dist_norm is not None and delta is not None:
                ok = True
                for other in members[c_idx]:
                    if dist_norm[node, other] > float(delta):
                        ok = False
                        break
                if not ok:
                    continue
            labels[node] = c_idx
            remaining[c_idx] -= 1
            members[c_idx].append(int(node))
            assigned = True
            break
        if not assigned:
            raise RuntimeError(
                f"Constrained assignment failed for node={node} "
                f"(max_capacity={max_capacity}, delta={delta})."
            )
    return labels


def _capacitated_kmeans_on_embedding(
    features: np.ndarray,
    num_clusters: int,
    max_capacity: int,
    seed: int,
    dist_norm: np.ndarray | None = None,
    delta: float | None = None,
    max_iter: int = 30,
) -> tuple[np.ndarray, str]:
    centers = _kmeans_centers(features, num_clusters, seed=seed)
    labels = np.zeros(features.shape[0], dtype=np.int64)
    tag = "capacitated_kmeans"
    for it in range(max_iter):
        new_labels = _capacitated_assign_embedding(
            features, centers, max_capacity, dist_norm=dist_norm, delta=delta
        )
        if np.array_equal(new_labels, labels) and it > 0:
            break
        labels = new_labels
        for k in range(num_clusters):
            mask = labels == k
            if mask.any():
                centers[k] = features[mask].mean(axis=0)
        tag = "constrained_kmeans" if delta is not None else "capacitated_kmeans"
    return labels, tag


def _greedy_weighted_matching(
    w: np.ndarray,
    seed: int = 0,
    order_mode: str = "weighted_degree",
) -> tuple[np.ndarray, list[list[int]]]:
    n = w.shape[0]
    w = np.asarray(w, dtype=np.float64)
    if order_mode == "weighted_degree":
        order = np.argsort(-w.sum(axis=1), kind="stable")
    else:
        order = np.arange(n, dtype=np.int64)
    unmatched = set(range(n))
    clusters: list[list[int]] = []
    for i in order:
        if i not in unmatched:
            continue
        best_j = None
        best_w = -1.0
        for j in unmatched:
            if j == i:
                continue
            wij = w[i, j]
            if wij > best_w:
                best_w = wij
                best_j = j
        if best_j is not None and best_w > 0:
            clusters.append([int(i), int(best_j)])
            unmatched.remove(i)
            unmatched.remove(best_j)
        else:
            clusters.append([int(i)])
            unmatched.remove(i)
    labels = np.zeros(n, dtype=np.int64)
    for c_idx, members in enumerate(clusters):
        for node in members:
            labels[node] = c_idx
    return labels, clusters


def _supernode_affinity(w: np.ndarray, clusters: list[list[int]]) -> np.ndarray:
    k = len(clusters)
    w_super = np.zeros((k, k), dtype=np.float64)
    for a in range(k):
        for b in range(a + 1, k):
            s = 0.0
            for i in clusters[a]:
                for j in clusters[b]:
                    s += w[i, j]
            w_super[a, b] = s
            w_super[b, a] = s
    return w_super


def _build_refinement_matrix(labels_fine: np.ndarray, labels_coarse: np.ndarray) -> np.ndarray:
    m_fine = int(labels_fine.max()) + 1
    m_coarse = int(labels_coarse.max()) + 1
    h = np.zeros((m_fine, m_coarse), dtype=np.float32)
    for a in range(m_fine):
        nodes = np.where(labels_fine == a)[0]
        coarse_ids = np.unique(labels_coarse[nodes])
        if coarse_ids.size != 1:
            raise ValueError(
                f"Fine cluster {a} is not nested in a single coarse cluster: {coarse_ids}"
            )
        h[a, int(coarse_ids[0])] = 1.0
    return h


def check_nested_consistency(c_fine: np.ndarray, c_coarse: np.ndarray, h: np.ndarray) -> bool:
    recon = c_fine @ h
    return bool(np.allclose(recon, c_coarse))


def build_graclus_multilevel_assignment(
    node_size: int,
    adj_mx_path: str | None,
    seed: int = 0,
    dataset_name: str = "unknown",
    graph_cluster_method: str = GRAPH_CLUSTER_METHOD_GRACLUS,
    cluster_road_distance_path: str | Path | None = None,
    cluster_sigma_d: float = 0.5,
    cluster_road_delta: float | None = None,
) -> dict[str, Any]:
    method = str(graph_cluster_method).lower()
    if method not in CAPACITY_MULTILEVEL_METHODS:
        raise ValueError(f"Not a multilevel matching method: {method}")

    adj_sym = symmetrize_adjacency(load_raw_adj_numpy(adj_mx_path, node_size))
    road_used = False
    road_path = ""
    if method == GRAPH_CLUSTER_METHOD_ROAD_GRACLUS:
        dist_norm, road_path = load_road_distance_matrix(
            node_size, dataset_name=dataset_name, cluster_road_distance_path=cluster_road_distance_path
        )
        w = build_road_distance_affinity(
            adj_sym, dist_norm, sigma_d=cluster_sigma_d, delta=cluster_road_delta
        )
        road_used = True
    else:
        w = adj_sym.copy()
        dist_norm = None

    labels_2, clusters_2 = _greedy_weighted_matching(w, seed=seed)
    m2 = len(clusters_2)
    w_super = _supernode_affinity(w, clusters_2)
    labels_super, clusters_super = _greedy_weighted_matching(w_super, seed=seed)
    labels_4 = np.zeros(node_size, dtype=np.int64)
    for g_idx, group in enumerate(clusters_super):
        for s_idx in group:
            for node in clusters_2[s_idx]:
                labels_4[node] = g_idx
    m4 = int(labels_4.max()) + 1

    c2 = labels_to_assignment(labels_2, node_size, m2)
    c4 = labels_to_assignment(labels_4, node_size, m4)
    p2 = assignment_to_projection(c2)
    p4 = assignment_to_projection(c4)
    h_2_to_4 = _build_refinement_matrix(labels_2, labels_4)
    nested_ok = check_nested_consistency(c2, c4, h_2_to_4)

    sizes_4 = np.bincount(labels_4.astype(np.int64))
    sizes_2 = np.bincount(labels_2.astype(np.int64))
    out: dict[str, Any] = {
        "node_size": node_size,
        "graph_cluster_method": method,
        "road_distance_used": road_used,
        "road_distance_path": road_path,
        "affinity_source": (
            "road_distance_affinity" if road_used else "adj_sym"
        ),
        "sigma_d": float(cluster_sigma_d),
        "delta": cluster_road_delta,
        "M4": m4,
        "M2": m2,
        "max_cluster_size_S4": int(sizes_4.max()),
        "max_cluster_size_S2": int(sizes_2.max()),
        "singleton_ratio_S4": float((sizes_4 == 1).sum() / max(m4, 1)),
        "singleton_ratio_S2": float((sizes_2 == 1).sum() / max(m2, 1)),
        "nested_consistency": nested_ok,
        "H_2_to_4": h_2_to_4,
        "stages": [
            {
                "capacity": 4,
                "resolution_tag": "S4",
                "num_clusters": m4,
                "labels": labels_4,
                "C": c4,
                "P": p4,
                "clustering_method": f"{method}_level4_matching",
            },
            {
                "capacity": 2,
                "resolution_tag": "S2",
                "num_clusters": m2,
                "labels": labels_2,
                "C": c2,
                "P": p2,
                "clustering_method": f"{method}_level2_matching",
            },
            {
                "capacity": 1,
                "resolution_tag": "S1",
                "num_clusters": node_size,
                "labels": np.arange(node_size, dtype=np.int64),
                "C": np.eye(node_size, dtype=np.float32),
                "P": np.eye(node_size, dtype=np.float32),
                "clustering_method": "identity",
            },
        ],
    }
    if dist_norm is not None:
        out["dist_norm"] = dist_norm
    if not nested_ok:
        raise ValueError(f"{method}: nested consistency check failed (C4 != C2 @ H)")
    return out


def _stage_capacity_for_ratio(ratio: float) -> int:
    if abs(ratio - 0.25) < 0.02:
        return 4
    if abs(ratio - 0.50) < 0.02:
        return 2
    return 1


def _stage_delta_for_capacity(capacity: int, cluster_delta_4: float, cluster_delta_2: float) -> float | None:
    if capacity >= 4:
        return float(cluster_delta_4)
    if capacity >= 2:
        return float(cluster_delta_2)
    return None


def _build_spectral_variant_assignment(
    node_size: int,
    num_clusters: int,
    adj_mx_path: str | None,
    seed: int,
    dataset_name: str,
    graph_cluster_method: str,
    cluster_road_distance_path: str | Path | None = None,
    cluster_sigma_d: float = 0.5,
    cluster_road_delta: float | None = None,
    cluster_max_capacity: int = 4,
    cluster_delta: float | None = None,
    ratio: float | None = None,
) -> dict[str, Any]:
    method = str(graph_cluster_method).lower()
    adj_sym = symmetrize_adjacency(load_raw_adj_numpy(adj_mx_path, node_size))
    road_used = False
    road_path = ""
    dist_norm = None
    affinity_source = "adj_sym"

    if method in {GRAPH_CLUSTER_METHOD_CAP_KMEANS_ADJ, GRAPH_CLUSTER_METHOD_CAP_ONLY_SPECTRAL}:
        affinity = adj_sym
        affinity_source = "adj_sym"
    elif method == GRAPH_CLUSTER_METHOD_CAP_KMEANS_ROAD:
        dist_norm, road_path = load_road_distance_matrix(
            node_size, dataset_name=dataset_name, cluster_road_distance_path=cluster_road_distance_path
        )
        affinity = build_road_distance_affinity(adj_sym, dist_norm, sigma_d=cluster_sigma_d)
        road_used = True
        affinity_source = "road_distance_affinity"
    elif method in ROAD_DISTANCE_METHODS or method == GRAPH_CLUSTER_METHOD_ROAD_SPECTRAL:
        dist_norm, road_path = load_road_distance_matrix(
            node_size, dataset_name=dataset_name, cluster_road_distance_path=cluster_road_distance_path
        )
        w = build_road_distance_affinity(
            adj_sym, dist_norm, sigma_d=cluster_sigma_d, delta=cluster_road_delta
        )
        road_used = True
        affinity = w
        affinity_source = "road_distance_affinity"
    elif method == GRAPH_CLUSTER_METHOD_CAP_KMEANS_SPECTRAL:
        affinity = adj_sym
        try:
            dist_norm, road_path = load_road_distance_matrix(
                node_size, dataset_name=dataset_name, cluster_road_distance_path=cluster_road_distance_path
            )
            w = build_road_distance_affinity(adj_sym, dist_norm, sigma_d=cluster_sigma_d)
            affinity = w
            road_used = True
            affinity_source = "road_distance_affinity"
        except FileNotFoundError:
            affinity = adj_sym
            affinity_source = "adj_sym_fallback"
    else:
        affinity = adj_sym
        affinity_source = "adj_sym"

    if method == GRAPH_CLUSTER_METHOD_ROAD_SPECTRAL:
        labels, sub_method = _spectral_cluster_labels(affinity, num_clusters, seed=seed)
        clustering_method = f"road_spectral_{sub_method}"
    elif method in SPECTRAL_EMBEDDING_METHODS:
        embed_dim = min(num_clusters, node_size - 1)
        z = _spectral_embedding(affinity, embed_dim)
        cap = int(cluster_max_capacity)
        delta = cluster_delta
        labels, sub_method = _capacitated_kmeans_on_embedding(
            z,
            num_clusters,
            max_capacity=cap,
            seed=seed,
            dist_norm=dist_norm if method == GRAPH_CLUSTER_METHOD_CONSTRAINED_SPECTRAL else None,
            delta=delta if method == GRAPH_CLUSTER_METHOD_CONSTRAINED_SPECTRAL else None,
        )
        clustering_method = f"{method}_{sub_method}"
    else:
        raise ValueError(f"Unhandled spectral variant: {method}")

    c = labels_to_assignment(labels, node_size, num_clusters)
    p = assignment_to_projection(c)
    meta: dict[str, Any] = {
        "node_size": node_size,
        "num_clusters": num_clusters,
        "clustering_method": clustering_method,
        "graph_cluster_method": method,
        "C": c,
        "P": p,
        "labels": labels,
        "road_distance_used": road_used,
        "road_distance_path": road_path,
        "affinity_source": affinity_source,
        "sigma_d": float(cluster_sigma_d),
        "delta": cluster_delta if method == GRAPH_CLUSTER_METHOD_CONSTRAINED_SPECTRAL else cluster_road_delta,
        "max_capacity": int(cluster_max_capacity),
        "capacities": cluster_capacities(node_size, num_clusters),
    }
    if ratio is not None:
        meta["ratio"] = float(ratio)
    return meta


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
    cluster_road_distance_path: str | Path | None = None,
    cluster_sigma_d: float = 0.5,
    cluster_road_delta: float | None = None,
    cluster_max_capacity: int = 4,
    cluster_delta: float | None = None,
    cluster_delta_4: float = 0.8,
    cluster_delta_2: float = 0.5,
    ratio: float | None = None,
) -> dict:
    method = str(graph_cluster_method).lower()
    if method not in GRAPH_CLUSTER_METHODS:
        raise ValueError(
            f"Unknown graph_cluster_method={graph_cluster_method}. "
            f"Choices: {sorted(GRAPH_CLUSTER_METHODS)}"
        )

    if method in CAPACITY_MULTILEVEL_METHODS:
        raise ValueError(
            f"{method} requires build_graclus_multilevel_assignment / "
            "load_or_build_multilevel_cluster_assignments."
        )

    if num_clusters >= node_size:
        labels = np.arange(node_size, dtype=np.int64)
        clustering_method = "identity"
        c = labels_to_assignment(labels, node_size, node_size)
    elif method in {
        GRAPH_CLUSTER_METHOD_ROAD_SPECTRAL,
        GRAPH_CLUSTER_METHOD_CONSTRAINED_SPECTRAL,
        GRAPH_CLUSTER_METHOD_CAP_KMEANS_SPECTRAL,
        GRAPH_CLUSTER_METHOD_CAP_KMEANS_ADJ,
        GRAPH_CLUSTER_METHOD_CAP_KMEANS_ROAD,
        GRAPH_CLUSTER_METHOD_CAP_ONLY_SPECTRAL,
    }:
        cap = int(cluster_max_capacity)
        delta = cluster_delta
        if ratio is not None:
            cap = _stage_capacity_for_ratio(ratio)
            delta = _stage_delta_for_capacity(cap, cluster_delta_4, cluster_delta_2)
        meta = _build_spectral_variant_assignment(
            node_size=node_size,
            num_clusters=num_clusters,
            adj_mx_path=adj_mx_path,
            seed=seed,
            dataset_name=dataset_name,
            graph_cluster_method=method,
            cluster_road_distance_path=cluster_road_distance_path,
            cluster_sigma_d=cluster_sigma_d,
            cluster_road_delta=cluster_road_delta,
            cluster_max_capacity=cap,
            cluster_delta=delta,
            ratio=ratio,
        )
        labels = meta["labels"]
        clustering_method = meta["clustering_method"]
        c = meta["C"]
        medoids = None
        pam_cost = None
        distance_summary = {
            "road_distance_used": meta.get("road_distance_used"),
            "road_distance_path": meta.get("road_distance_path"),
            "affinity_source": meta.get("affinity_source"),
            "sigma_d": meta.get("sigma_d"),
            "delta": meta.get("delta"),
            "max_capacity": meta.get("max_capacity"),
        }
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
        distance_summary = {"affinity_source": "adj_sym", "road_distance_used": False}
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
    if method in {
        GRAPH_CLUSTER_METHOD_ROAD_SPECTRAL,
        GRAPH_CLUSTER_METHOD_CONSTRAINED_SPECTRAL,
        GRAPH_CLUSTER_METHOD_CAP_KMEANS_SPECTRAL,
        GRAPH_CLUSTER_METHOD_CAP_KMEANS_ADJ,
        GRAPH_CLUSTER_METHOD_CAP_KMEANS_ROAD,
        GRAPH_CLUSTER_METHOD_CAP_ONLY_SPECTRAL,
    } and num_clusters < node_size:
        out["distance_summary"] = distance_summary
        out["capacities"] = meta.get("capacities", cluster_capacities(node_size, num_clusters))
    elif method != GRAPH_CLUSTER_METHOD_CURRENT and num_clusters < node_size:
        out["medoids"] = medoids
        out["pam_cost"] = pam_cost
        out["distance_summary"] = distance_summary
        out["capacities"] = cluster_capacities(node_size, num_clusters)
    if adj_mx_path and os.path.exists(adj_mx_path) and num_clusters < node_size:
        adj_t = load_adj_from_pickle(adj_mx_path)
        if adj_t is not None:
            out["coarse_adj"] = coarse_adjacency(adj_t.numpy(), c)
    out["validation"] = validate_cluster_assignment(out)
    if "distance_summary" in out and isinstance(out["distance_summary"], dict):
        for k, v in out["distance_summary"].items():
            out.setdefault(k, v)
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
    cluster_road_distance_path: str | Path | None = None,
    cluster_sigma_d: float = 0.5,
    cluster_road_delta: float | None = None,
    cluster_max_capacity: int = 4,
    cluster_delta: float | None = None,
    cluster_delta_4: float = 0.8,
    cluster_delta_2: float = 0.5,
    ratio: float | None = None,
) -> tuple[dict, Path]:
    method = str(graph_cluster_method).lower()
    extra: dict[str, Any] = {}
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
        elif method in {
            GRAPH_CLUSTER_METHOD_ROAD_SPECTRAL,
            GRAPH_CLUSTER_METHOD_CONSTRAINED_SPECTRAL,
            GRAPH_CLUSTER_METHOD_CAP_KMEANS_SPECTRAL,
            GRAPH_CLUSTER_METHOD_CAP_KMEANS_ADJ,
            GRAPH_CLUSTER_METHOD_CAP_KMEANS_ROAD,
            GRAPH_CLUSTER_METHOD_CAP_ONLY_SPECTRAL,
        }:
            rd = _resolve_path(cluster_road_distance_path)
            extra["sigma_d"] = cluster_sigma_d
            extra["road_delta"] = cluster_road_delta
            extra["max_capacity"] = cluster_max_capacity
            extra["delta"] = cluster_delta
            extra["delta_4"] = cluster_delta_4
            extra["delta_2"] = cluster_delta_2
            extra["road"] = str(rd) if rd else DEFAULT_ROAD_DISTANCE_PATHS.get(dataset_name, "none")
            extra_tag = f"sd{cluster_sigma_d}"
            if ratio is not None:
                extra["ratio"] = ratio

    literature_methods = {
        GRAPH_CLUSTER_METHOD_ROAD_SPECTRAL,
        GRAPH_CLUSTER_METHOD_CONSTRAINED_SPECTRAL,
        GRAPH_CLUSTER_METHOD_CAP_KMEANS_SPECTRAL,
        GRAPH_CLUSTER_METHOD_CAP_KMEANS_ADJ,
        GRAPH_CLUSTER_METHOD_CAP_KMEANS_ROAD,
        GRAPH_CLUSTER_METHOD_CAP_ONLY_SPECTRAL,
    }
    if method in literature_methods or method == GRAPH_CLUSTER_METHOD_CURRENT:
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
        if "distance_summary" in meta and isinstance(meta["distance_summary"], dict):
            for k, v in meta["distance_summary"].items():
                if k not in meta:
                    meta[k] = v
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
        cluster_road_distance_path=cluster_road_distance_path,
        cluster_sigma_d=cluster_sigma_d,
        cluster_road_delta=cluster_road_delta,
        cluster_max_capacity=cluster_max_capacity,
        cluster_delta=cluster_delta,
        cluster_delta_4=cluster_delta_4,
        cluster_delta_2=cluster_delta_2,
        ratio=ratio,
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


def load_or_build_multilevel_cluster_assignments(
    node_size: int,
    capacities: list[int],
    adj_mx_path: str | None,
    seed: int = 0,
    dataset_name: str = "unknown",
    cache_dir: str | Path | None = None,
    graph_cluster_method: str = GRAPH_CLUSTER_METHOD_GRACLUS,
    cluster_road_distance_path: str | Path | None = None,
    cluster_sigma_d: float = 0.5,
    cluster_road_delta: float | None = None,
) -> tuple[list[dict[str, Any]], Path]:
    method = str(graph_cluster_method).lower()
    if method not in CAPACITY_MULTILEVEL_METHODS:
        raise ValueError(f"Not a multilevel method: {method}")

    caps = resolve_graph_resolution_capacities(capacities)
    extra = {
        "capacities": caps,
        "sigma_d": cluster_sigma_d,
        "road_delta": cluster_road_delta,
        "road": str(_resolve_path(cluster_road_distance_path) or DEFAULT_ROAD_DISTANCE_PATHS.get(dataset_name, "none")),
    }
    cache_root = Path(cache_dir) if cache_dir else default_cache_dir()
    cache_root.mkdir(parents=True, exist_ok=True)
    key = _cache_key(dataset_name, node_size, node_size, adj_mx_path, seed, method, extra)
    cache_path = cache_root / f"{dataset_name}_N{node_size}_multilevel_{method}_s{seed}_{key}.npz"

    if cache_path.is_file():
        data = np.load(cache_path, allow_pickle=True)
        stages_raw = data["stages"]
        if isinstance(stages_raw, np.ndarray) and stages_raw.shape == ():
            stages_raw = stages_raw.item()
        nested = bool(data["nested_consistency"]) if "nested_consistency" in data else None
        road_used = bool(data["road_distance_used"]) if "road_distance_used" in data else False
        road_path = str(data["road_distance_path"]) if "road_distance_path" in data else ""
        stage_metas: list[dict[str, Any]] = []
        for st in stages_raw:
            sm = dict(st)
            sm["C"] = np.asarray(sm["C"], dtype=np.float32)
            sm["P"] = np.asarray(sm["P"], dtype=np.float32)
            sm["labels"] = np.asarray(sm["labels"])
            sm["nested_consistency"] = nested
            sm["road_distance_used"] = road_used
            sm["road_distance_path"] = road_path
            sm["validation"] = validate_cluster_assignment(sm)
            stage_metas.append(sm)
        return stage_metas, cache_path

    bundle = build_graclus_multilevel_assignment(
        node_size=node_size,
        adj_mx_path=adj_mx_path,
        seed=seed,
        dataset_name=dataset_name,
        graph_cluster_method=method,
        cluster_road_distance_path=cluster_road_distance_path,
        cluster_sigma_d=cluster_sigma_d,
        cluster_road_delta=cluster_road_delta,
    )
    stage_metas = []
    for st in bundle["stages"]:
        sm = dict(st)
        sm["node_size"] = node_size
        sm["graph_cluster_method"] = method
        sm["road_distance_used"] = bundle.get("road_distance_used", False)
        sm["road_distance_path"] = bundle.get("road_distance_path", "")
        sm["sigma_d"] = bundle.get("sigma_d")
        sm["nested_consistency"] = bundle.get("nested_consistency")
        sm["validation"] = validate_cluster_assignment(sm)
        stage_metas.append(sm)

    save_stages = []
    for sm in stage_metas:
        save_stages.append(
            {
                "capacity": sm.get("capacity"),
                "resolution_tag": sm.get("resolution_tag"),
                "num_clusters": sm["num_clusters"],
                "labels": sm["labels"],
                "C": sm["C"],
                "P": sm["P"],
                "clustering_method": sm["clustering_method"],
            }
        )
    np.savez_compressed(
        cache_path,
        node_size=node_size,
        graph_cluster_method=method,
        M4=bundle["M4"],
        M2=bundle["M2"],
        H_2_to_4=bundle["H_2_to_4"],
        nested_consistency=bundle["nested_consistency"],
        road_distance_used=bundle["road_distance_used"],
        road_distance_path=bundle.get("road_distance_path", ""),
        sigma_d=bundle.get("sigma_d"),
        stages=np.array(save_stages, dtype=object),
    )
    return stage_metas, cache_path


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
