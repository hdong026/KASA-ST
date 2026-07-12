"""Utilities for Graph Resolution cluster matrix visualization."""
from __future__ import annotations

import csv
import pickle
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.manifold import MDS
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from basicts.archs.arch_zoo.ChainForecasting_arch.graph_cluster_utils import (
    GRAPH_CLUSTER_METHODS,
    build_distance_matrix_for_method,
    load_or_build_cluster_assignment,
    load_train_node_series,
    pearson_distance_matrix,
    resolve_graph_resolution_sizes,
    validate_cluster_assignment,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

METHOD_ALIASES: dict[str, str] = {
    "joint_pearson_roadcost_pam": "joint_pearson_spatial_balanced_pam",
}

METHOD_DISPLAY: dict[str, str] = {
    "current": "current",
    "pearson_balanced_pam": "pearson_balanced_pam",
    "xcorr_balanced_pam": "xcorr_balanced_pam",
    "joint_pearson_spatial_balanced_pam": "joint_pearson_roadcost_pam",
    "pearson_standard_pam": "pearson_standard_pam",
    "autocorr_feature_balanced_pam": "autocorr_feature_balanced_pam",
}

METHOD_DEFAULT_KWARGS: dict[str, dict[str, Any]] = {
    "current": {},
    "pearson_balanced_pam": {},
    "xcorr_balanced_pam": {"cluster_max_lag": 12},
    "joint_pearson_spatial_balanced_pam": {
        "cluster_lambda_s": 0.2,
        "cluster_spatial_coord_path": "datasets/raw_data/PEMS04/adj_PEMS04_distance.pkl",
    },
    "pearson_standard_pam": {},
    "autocorr_feature_balanced_pam": {"cluster_acf_lag": 24},
}

DATASET_PATHS: dict[str, dict[str, str]] = {
    "PEMS04": {
        "adj_mx_path": "datasets/PEMS04/adj_mx.pkl",
        "edge_csv": "datasets/raw_data/PEMS04/PEMS04.csv",
        "distance_pkl": "datasets/raw_data/PEMS04/adj_PEMS04_distance.pkl",
        "coord_csv": "",
        "node_size": "307",
    },
    "PEMS-BAY": {
        "adj_mx_path": "datasets/PEMS-BAY/adj_mx.pkl",
        "edge_csv": "",
        "distance_pkl": "",
        "coord_csv": "datasets/raw_data/PEMS-BAY/sensor_graph/graph_sensor_locations_bay.csv",
        "node_size": "325",
    },
}


def resolve_method(name: str) -> str:
    name = str(name).lower()
    return METHOD_ALIASES.get(name, name)


def method_file_tag(method: str) -> str:
    internal = resolve_method(method)
    return METHOD_DISPLAY.get(internal, internal)


def ratio_to_tag(ratio: float, m: int) -> str:
    if abs(ratio - 0.25) < 0.02:
        return "S14"
    if abs(ratio - 0.50) < 0.02:
        return "S12"
    if abs(ratio - 1.0) < 0.02 or m >= 300:
        return "S1"
    return f"M{m}"


def dataset_paths(dataset: str) -> dict[str, Any]:
    if dataset not in DATASET_PATHS:
        raise ValueError(f"Unsupported dataset: {dataset}. Choices: {list(DATASET_PATHS)}")
    cfg = DATASET_PATHS[dataset].copy()
    cfg["node_size"] = int(cfg["node_size"])
    for k in ("adj_mx_path", "edge_csv", "distance_pkl", "coord_csv"):
        val = cfg.get(k, "")
        if not val:
            cfg[k] = ""
            continue
        p = REPO_ROOT / val
        if k != "edge_csv" and not p.is_file():
            raise FileNotFoundError(f"Missing {dataset} file: {p}")
        cfg[k] = str(p)
    return cfg


def _load_adj_matrix(adj_mx_path: str) -> np.ndarray:
    with open(adj_mx_path, "rb") as f:
        try:
            obj = pickle.load(f)
        except UnicodeDecodeError:
            f.seek(0)
            obj = pickle.load(f, encoding="latin1")
    if isinstance(obj, (list, tuple)):
        for item in reversed(obj):
            if hasattr(item, "shape") and len(item.shape) == 2:
                return np.asarray(item, dtype=np.float64)
    return np.asarray(obj, dtype=np.float64)


def load_sensor_coordinates(coord_csv: str, node_size: int) -> tuple[np.ndarray, np.ndarray] | None:
    if not coord_csv:
        return None
    path = Path(coord_csv)
    if not path.is_file():
        return None
    df = pd.read_csv(path, header=None, names=["sensor_id", "lat", "lon"])
    if len(df) != node_size:
        raise ValueError(f"Coordinate file {path} has {len(df)} rows, expected {node_size}")
    return df["lon"].to_numpy(dtype=np.float64), df["lat"].to_numpy(dtype=np.float64)


def haversine_distance_matrix(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    lat_r = np.radians(lat)
    lon_r = np.radians(lon)
    dlat = lat_r[:, None] - lat_r[None, :]
    dlon = lon_r[:, None] - lon_r[None, :]
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat_r[:, None]) * np.cos(lat_r[None, :]) * np.sin(dlon / 2.0) ** 2
    dist = 2.0 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    np.fill_diagonal(dist, 0.0)
    return dist.astype(np.float64)


def build_edges_from_adj(adj: np.ndarray) -> list[tuple[int, int]]:
    edges: list[tuple[int, int]] = []
    n = adj.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] > 0:
                edges.append((i, j))
    return edges


def build_shortest_paths_from_adj_weights(adj: np.ndarray, weights: np.ndarray) -> np.ndarray:
    g = nx.Graph()
    n = adj.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] > 0:
                w = float(weights[i, j]) if weights is not None else 1.0
                g.add_edge(i, j, weight=max(w, 1e-6))
    dist = np.full((n, n), np.inf, dtype=np.float64)
    np.fill_diagonal(dist, 0.0)
    for src, lengths in nx.all_pairs_dijkstra_path_length(g, weight="weight"):
        for dst, d in lengths.items():
            dist[src, dst] = d
    finite = dist[np.isfinite(dist)]
    fill = float(finite.max()) * 1.05 if finite.size else 1.0
    dist[~np.isfinite(dist)] = fill
    return dist


def ensure_spatial_distance_file(dataset: str, cfg: dict[str, Any], out_dir: Path) -> str:
    if dataset == "PEMS04":
        return cfg["distance_pkl"]
    coord = load_sensor_coordinates(cfg.get("coord_csv", ""), cfg["node_size"])
    if coord is None:
        raise FileNotFoundError(f"{dataset} has no coordinate file for joint spatial clustering.")
    lon, lat = coord
    dist = haversine_distance_matrix(lat, lon)
    maxv = dist.max()
    if maxv > 0:
        dist = dist / maxv
    out = out_dir / f"{dataset.lower()}_gps_distance_matrix.npy"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, dist.astype(np.float32))
    return str(out)


def method_kwargs_for_dataset(method: str, dataset: str, out_dir: Path) -> dict[str, Any]:
    internal = resolve_method(method)
    kwargs = dict(METHOD_DEFAULT_KWARGS.get(internal, {}))
    if internal == "joint_pearson_spatial_balanced_pam":
        cfg = dataset_paths(dataset)
        kwargs["cluster_spatial_coord_path"] = ensure_spatial_distance_file(dataset, cfg, out_dir)
    return kwargs


def prepare_dataset_assets(
    dataset: str,
    out_dir: Path,
    force_rebuild_layout: bool = False,
    seed: int = 0,
) -> dict[str, Any]:
    cfg = dataset_paths(dataset)
    node_size = cfg["node_size"]
    adj = _load_adj_matrix(cfg["adj_mx_path"])

    if cfg.get("edge_csv"):
        edges = load_edges(cfg["edge_csv"])
        road_dist = build_road_shortest_paths(cfg["edge_csv"], node_size)
    else:
        edges = build_edges_from_adj(adj)
        coords = load_sensor_coordinates(cfg.get("coord_csv", ""), node_size)
        if coords is not None:
            lon, lat = coords
            haversine = haversine_distance_matrix(lat, lon)
            road_dist = build_shortest_paths_from_adj_weights(adj, haversine)
        else:
            road_dist = build_shortest_paths_from_adj_weights(adj, adj)

    layout_cache = out_dir / f"{dataset.lower()}_layout_mds.npz"
    xy_mds, layout_method = compute_layout(road_dist, layout_cache, force_rebuild=force_rebuild_layout, seed=seed)

    xy_geo = None
    coords = load_sensor_coordinates(cfg.get("coord_csv", ""), node_size)
    if coords is not None:
        lon, lat = coords
        xy_geo = np.stack([lon, lat], axis=1)

    adj_bin = (adj > 0).astype(np.float64)
    return {
        "cfg": cfg,
        "node_size": node_size,
        "adj": adj,
        "adj_bin": adj_bin,
        "edges": edges,
        "road_dist": road_dist,
        "xy_mds": xy_mds,
        "xy_geo": xy_geo,
        "layout_method": layout_method,
        "layout_cache": layout_cache,
    }


def build_road_shortest_paths(edge_csv: str, node_size: int) -> np.ndarray:
    df = pd.read_csv(edge_csv)
    g = nx.Graph()
    for row in df.itertuples(index=False):
        i, j, w = int(row[0]), int(row[1]), float(row[2])
        if g.has_edge(i, j):
            if w < g[i][j]["weight"]:
                g[i][j]["weight"] = w
        else:
            g.add_edge(i, j, weight=w)

    dist = np.full((node_size, node_size), np.inf, dtype=np.float64)
    np.fill_diagonal(dist, 0.0)
    for src, lengths in nx.all_pairs_dijkstra_path_length(g, weight="weight"):
        for dst, d in lengths.items():
            dist[src, dst] = d

    finite = dist[np.isfinite(dist)]
    if finite.size == 0:
        raise RuntimeError("No finite shortest-path distances computed from edge file.")
    fill = float(finite.max()) * 1.05
    dist[~np.isfinite(dist)] = fill
    return dist


def compute_layout(
    road_dist: np.ndarray,
    cache_path: Path,
    force_rebuild: bool = False,
    seed: int = 0,
) -> tuple[np.ndarray, str]:
    if cache_path.is_file() and not force_rebuild:
        data = np.load(cache_path)
        return data["xy"].astype(np.float64), "cached_mds"

    try:
        mds = MDS(
            n_components=2,
            dissimilarity="precomputed",
            random_state=seed,
            normalized_stress="auto",
            max_iter=400,
            n_init=4,
        )
        xy = mds.fit_transform(road_dist)
        method = "mds"
    except Exception as exc:
        g = nx.Graph()
        n = road_dist.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                if road_dist[i, j] < np.inf and road_dist[i, j] > 0:
                    g.add_edge(i, j, weight=float(road_dist[i, j]))
        pos = nx.spring_layout(g, seed=seed, weight="weight", dim=2)
        xy = np.array([pos[i] for i in range(n)], dtype=np.float64)
        method = f"spring_layout_fallback({exc.__class__.__name__})"

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, xy=xy, layout_method=method)
    return xy, method


def load_edges(edge_csv: str) -> list[tuple[int, int]]:
    df = pd.read_csv(edge_csv)
    return [(int(a), int(b)) for a, b in zip(df["from"], df["to"])]


def load_cluster_meta(
    method: str,
    m: int,
    dataset: str,
    seed: int,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    internal = resolve_method(method)
    if internal not in GRAPH_CLUSTER_METHODS:
        raise ValueError(f"Unknown method {method}. Choices: {sorted(GRAPH_CLUSTER_METHODS)}")

    cfg = dataset_paths(dataset)
    kwargs = {
        "node_size": cfg["node_size"],
        "num_clusters": m,
        "adj_mx_path": cfg["adj_mx_path"],
        "seed": seed,
        "dataset_name": dataset,
        "graph_cluster_method": internal,
    }
    extra = method_kwargs_for_dataset(method, dataset, out_dir or REPO_ROOT / "results" / "cluster_viz")
    kwargs.update(extra)
    meta, cache_path = load_or_build_cluster_assignment(**kwargs)
    meta["cache_path"] = str(cache_path)
    meta["method"] = method_file_tag(internal)
    meta["internal_method"] = internal
    return meta


def plot_geo_clusters(
    lon: np.ndarray,
    lat: np.ndarray,
    labels: np.ndarray,
    edges: list[tuple[int, int]],
    title: str,
    out_path: Path,
) -> None:
    """Plot clusters on real lat/lon axes (San Francisco Bay Area)."""
    n_clusters = int(labels.max()) + 1
    cmap = plt.get_cmap("tab20" if n_clusters <= 20 else "nipy_spectral")
    colors = cmap(np.linspace(0, 1, n_clusters))

    fig, ax = plt.subplots(figsize=(10, 9), dpi=200)
    for i, j in edges:
        ax.plot([lon[i], lon[j]], [lat[i], lat[j]], color="#cccccc", lw=0.2, alpha=0.35, zorder=1)
    for k in range(n_clusters):
        mask = labels == k
        ax.scatter(lon[mask], lat[mask], s=22, c=[colors[k]], edgecolors="white", linewidths=0.2, zorder=2)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_folium_map(
    lon: np.ndarray,
    lat: np.ndarray,
    labels: np.ndarray,
    title: str,
    out_path: Path,
) -> bool:
    try:
        import folium
    except ImportError:
        return False

    center = [float(lat.mean()), float(lon.mean())]
    fmap = folium.Map(location=center, zoom_start=10, tiles="OpenStreetMap")
    n_clusters = int(labels.max()) + 1
    cmap = plt.get_cmap("tab20" if n_clusters <= 20 else "nipy_spectral")
    for i in range(len(lat)):
        c = cmap(int(labels[i]) / max(n_clusters - 1, 1))
        color = f"#{int(c[0]*255):02x}{int(c[1]*255):02x}{int(c[2]*255):02x}"
        folium.CircleMarker(
            location=[float(lat[i]), float(lon[i])],
            radius=4,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            popup=f"node={i}, cluster={int(labels[i])}",
        ).add_to(fmap)
    folium.map.LayerControl().add_to(fmap)
    fmap.get_root().html.add_child(folium.Element(f"<title>{title}</title>"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(out_path))
    return True


def labels_from_meta(meta: dict) -> np.ndarray:
    return np.asarray(meta["labels"], dtype=np.int64)


def draw_base_edges(ax, xy: np.ndarray, edges: list[tuple[int, int]]) -> None:
    for i, j in edges:
        ax.plot([xy[i, 0], xy[j, 0]], [xy[i, 1], xy[j, 1]], color="#d0d0d0", lw=0.25, alpha=0.45, zorder=1)


def plot_layout_base(xy: np.ndarray, edges: list[tuple[int, int]], out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 8), dpi=200)
    draw_base_edges(ax, xy, edges)
    ax.scatter(xy[:, 0], xy[:, 1], s=12, c="#1f77b4", alpha=0.8, zorder=2)
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_node_clusters(
    xy: np.ndarray,
    labels: np.ndarray,
    edges: list[tuple[int, int]],
    title: str,
    out_path: Path,
    medoids: np.ndarray | None = None,
) -> None:
    n_clusters = int(labels.max()) + 1
    cmap = plt.get_cmap("tab20" if n_clusters <= 20 else "nipy_spectral")
    colors = cmap(np.linspace(0, 1, n_clusters))

    fig, ax = plt.subplots(figsize=(10, 8), dpi=200)
    draw_base_edges(ax, xy, edges)
    for k in range(n_clusters):
        mask = labels == k
        ax.scatter(xy[mask, 0], xy[mask, 1], s=26, c=[colors[k]], edgecolors="white", linewidths=0.25, zorder=2)
    if medoids is not None and len(medoids):
        medoids = np.asarray(medoids, dtype=np.int64)
        ax.scatter(
            xy[medoids, 0],
            xy[medoids, 1],
            s=70,
            facecolors="none",
            edgecolors="black",
            linewidths=1.0,
            zorder=3,
        )
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def build_adj_binary(edge_csv: str, node_size: int) -> np.ndarray:
    adj = np.zeros((node_size, node_size), dtype=np.float64)
    for i, j in load_edges(edge_csv):
        adj[i, j] = 1.0
        adj[j, i] = 1.0
    return adj


def cluster_graph_edges(
    labels: np.ndarray,
    adj: np.ndarray,
    road_dist: np.ndarray,
    topk_per_cluster: int = 3,
) -> list[tuple[int, int, float]]:
    n_clusters = int(labels.max()) + 1
    edge_w: dict[tuple[int, int], float] = {}
    n = labels.shape[0]
    for i in range(n):
        ci = int(labels[i])
        for j in range(i + 1, n):
            if adj[i, j] <= 0:
                continue
            cj = int(labels[j])
            if ci == cj:
                continue
            a, b = (ci, cj) if ci < cj else (cj, ci)
            edge_w[(a, b)] = edge_w.get((a, b), 0.0) + 1.0

    selected: list[tuple[int, int, float]] = []
    for c in range(n_clusters):
        cand = [(w, a, b) for (a, b), w in edge_w.items() if a == c or b == c]
        cand.sort(reverse=True)
        for w, a, b in cand[:topk_per_cluster]:
            selected.append((a, b, w))

    seen = set()
    uniq = []
    for a, b, w in sorted(selected, key=lambda x: -x[2]):
        key = (a, b) if a < b else (b, a)
        if key in seen:
            continue
        seen.add(key)
        mean_d = 0.0
        idx_a = np.where(labels == a)[0]
        idx_b = np.where(labels == b)[0]
        if idx_a.size and idx_b.size:
            sub = road_dist[np.ix_(idx_a, idx_b)]
            mean_d = float(sub.min())
        uniq.append((a, b, mean_d if mean_d > 0 else w))
    return uniq


def plot_cluster_graph(
    xy: np.ndarray,
    labels: np.ndarray,
    edges_cc: list[tuple[int, int, float]],
    title: str,
    out_path: Path,
    medoids: np.ndarray | None = None,
) -> None:
    n_clusters = int(labels.max()) + 1
    centers = np.zeros((n_clusters, 2), dtype=np.float64)
    sizes = np.zeros(n_clusters, dtype=np.int64)
    for k in range(n_clusters):
        idx = np.where(labels == k)[0]
        sizes[k] = len(idx)
        if medoids is not None and k < len(medoids):
            centers[k] = xy[int(medoids[k])]
        else:
            centers[k] = xy[idx].mean(axis=0)

    cmap = plt.get_cmap("tab20" if n_clusters <= 20 else "nipy_spectral")
    colors = cmap(np.linspace(0, 1, n_clusters))
    area = 40 + 25 * np.sqrt(np.maximum(sizes, 1))

    fig, ax = plt.subplots(figsize=(10, 8), dpi=200)
    for a, b, _ in edges_cc:
        ax.plot([centers[a, 0], centers[b, 0]], [centers[a, 1], centers[b, 1]], color="#888888", lw=0.8, alpha=0.6, zorder=1)
    for k in range(n_clusters):
        ax.scatter(centers[k, 0], centers[k, 1], s=area[k], c=[colors[k]], edgecolors="black", linewidths=0.4, zorder=2)
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def overlap_matrix(labels_coarse: np.ndarray, labels_fine: np.ndarray) -> np.ndarray:
    c_max = int(labels_coarse.max()) + 1
    f_max = int(labels_fine.max()) + 1
    mat = np.zeros((c_max, f_max), dtype=np.int64)
    for i in range(labels_coarse.shape[0]):
        mat[int(labels_coarse[i]), int(labels_fine[i])] += 1
    return mat


def sort_overlap_rows_cols(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    row_order = np.argsort(-mat.sum(axis=1))
    col_order = np.argsort(-mat.sum(axis=0))
    return mat[row_order][:, col_order], row_order, col_order


def plot_overlap_heatmap(mat: np.ndarray, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 7), dpi=200)
    im = ax.imshow(mat, aspect="auto", cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("S1/2 clusters")
    ax.set_ylabel("S1/4 clusters")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="node overlap count")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_cluster_size_hist(
    sizes_by_tag: dict[str, np.ndarray],
    method: str,
    out_path: Path,
) -> None:
    n = len(sizes_by_tag)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), dpi=200)
    if n == 1:
        axes = [axes]
    for ax, (tag, sizes) in zip(axes, sizes_by_tag.items()):
        ax.hist(sizes, bins=max(10, min(30, int(sizes.max() - sizes.min() + 1))))
        ax.set_title(f"{method} | {tag}")
        ax.set_xlabel("cluster size")
        ax.set_ylabel("count")
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_side_by_side(
    xy: np.ndarray,
    labels_a: np.ndarray,
    labels_b: np.ndarray,
    title_a: str,
    title_b: str,
    edges: list[tuple[int, int]],
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=200)
    for ax, labels, title in zip(axes, [labels_a, labels_b], [title_a, title_b]):
        n_clusters = int(labels.max()) + 1
        cmap = plt.get_cmap("tab20" if n_clusters <= 20 else "nipy_spectral")
        colors = cmap(np.linspace(0, 1, n_clusters))
        draw_base_edges(ax, xy, edges)
        for k in range(n_clusters):
            mask = labels == k
            ax.scatter(xy[mask, 0], xy[mask, 1], s=20, c=[colors[k]], edgecolors="white", linewidths=0.2, zorder=2)
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def compute_label_similarity(method_labels: dict[str, np.ndarray]) -> pd.DataFrame:
    methods = sorted(method_labels.keys())
    rows = []
    for i, ma in enumerate(methods):
        for mb in methods[i:]:
            la, lb = method_labels[ma], method_labels[mb]
            rows.append(
                {
                    "method_a": ma,
                    "method_b": mb,
                    "ari": adjusted_rand_score(la, lb),
                    "nmi": normalized_mutual_info_score(la, lb),
                }
            )
    return pd.DataFrame(rows)


def plot_similarity_heatmap(df: pd.DataFrame, metric: str, out_path: Path, title: str) -> None:
    methods = sorted(set(df["method_a"]).union(df["method_b"]))
    idx = {m: i for i, m in enumerate(methods)}
    mat = np.eye(len(methods))
    for row in df.itertuples(index=False):
        mat[idx[row.method_a], idx[row.method_b]] = getattr(row, metric)
        mat[idx[row.method_b], idx[row.method_a]] = getattr(row, metric)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
    im = ax.imshow(mat, vmin=0, vmax=1, cmap="magma")
    ax.set_xticks(range(len(methods)), labels=methods, rotation=45, ha="right")
    ax.set_yticks(range(len(methods)), labels=methods)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def cluster_size_stats(labels: np.ndarray) -> dict[str, float | int]:
    sizes = np.bincount(labels.astype(np.int64))
    return {
        "num_clusters": int(len(sizes)),
        "min_cluster_size": int(sizes.min()),
        "max_cluster_size": int(sizes.max()),
        "mean_cluster_size": float(sizes.mean()),
        "std_cluster_size": float(sizes.std()),
    }


def compute_extra_metrics(
    internal_method: str,
    labels: np.ndarray,
    road_dist: np.ndarray,
    dataset: str,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "mean_intra_abs_corr": None,
        "mean_inter_abs_corr": None,
        "mean_intra_road_distance": None,
    }
    n = labels.shape[0]
    intra_d = []
    for k in range(int(labels.max()) + 1):
        idx = np.where(labels == k)[0]
        if idx.size < 2:
            continue
        sub = road_dist[np.ix_(idx, idx)]
        tri = sub[np.triu_indices(sub.shape[0], k=1)]
        intra_d.append(float(tri.mean()))
    if intra_d:
        out["mean_intra_road_distance"] = float(np.mean(intra_d))

    if internal_method in {"pearson_balanced_pam", "pearson_standard_pam", "joint_pearson_spatial_balanced_pam"}:
        try:
            series = load_train_node_series(dataset_name=dataset)
            corr = 1.0 - pearson_distance_matrix(series)
            intra_c, inter_c = [], []
            clusters = int(labels.max()) + 1
            for a in range(clusters):
                idx_a = np.where(labels == a)[0]
                if idx_a.size < 2:
                    continue
                sub = corr[np.ix_(idx_a, idx_a)]
                tri = sub[np.triu_indices(sub.shape[0], k=1)]
                intra_c.append(float(np.abs(tri).mean()))
                for b in range(a + 1, clusters):
                    idx_b = np.where(labels == b)[0]
                    sub2 = corr[np.ix_(idx_a, idx_b)]
                    inter_c.append(float(np.abs(sub2).mean()))
            if intra_c:
                out["mean_intra_abs_corr"] = float(np.mean(intra_c))
            if inter_c:
                out["mean_inter_abs_corr"] = float(np.mean(inter_c))
        except Exception:
            pass
    return out


def write_summary_table(rows: list[dict[str, Any]], out_csv: Path, out_md: Path) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    md = ["# Graph Resolution Cluster Summary\n\n", "| " + " | ".join(fields) + " |\n", "|" + "|".join(["---"] * len(fields)) + "|\n"]
    for r in rows:
        md.append("| " + " | ".join(str(r.get(k, "")) for k in fields) + " |\n")
    out_md.write_text("".join(md), encoding="utf-8")
