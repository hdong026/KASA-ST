#!/usr/bin/env python3
"""Visualize PEMS-BAY Graph Resolution cluster assignments on real GPS coordinates."""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basicts.archs.arch_zoo.ChainForecasting_arch.graph_cluster_utils import (
    CAPACITY_MULTILEVEL_METHODS,
    GRAPH_CLUSTER_METHOD_CURRENT,
    check_nested_consistency,
    load_or_build_cluster_assignment,
    load_or_build_multilevel_cluster_assignments,
    resolve_graph_resolution_capacities,
    resolve_graph_resolution_sizes,
    validate_cluster_assignment,
)
from scripts.utils.cluster_visualization_utils import (
    build_edges_from_adj,
    build_shortest_paths_from_adj_weights,
    cluster_graph_edges,
    cluster_size_stats,
    haversine_distance_matrix,
    overlap_matrix,
    plot_cluster_graph,
    plot_cluster_size_hist,
    plot_overlap_heatmap,
    ratio_to_tag,
    sort_overlap_rows_cols,
    write_summary_table,
)

EXPECTED_N = 325

COORD_CANDIDATES = [
    "data/sensor_graph/graph_sensor_locations_bay.csv",
    "data/PEMS-BAY/graph_sensor_locations_bay.csv",
    "datasets/PEMS-BAY/graph_sensor_locations_bay.csv",
    "data/PEMSBAY/graph_sensor_locations_bay.csv",
    "datasets/raw_data/PEMS-BAY/sensor_graph/graph_sensor_locations_bay.csv",
]

ADJ_CANDIDATES = [
    "data/sensor_graph/adj_mx_bay.pkl",
    "data/PEMS-BAY/adj_mx_bay.pkl",
    "datasets/PEMS-BAY/adj_mx_bay.pkl",
    "datasets/PEMS-BAY/adj_mx.pkl",
    "datasets/raw_data/PEMS-BAY/adj_PEMS-BAY.pkl",
]

DIST_CANDIDATES = [
    "data/sensor_graph/distances_bay_2017.csv",
    "data/PEMS-BAY/distances_bay_2017.csv",
    "datasets/PEMS-BAY/distances_bay_2017.csv",
]

VARIANT_CONFIGS: dict[str, dict[str, Any]] = {
    "GR7_sparse_topk": {
        "graph_cluster_method": "current",
        "graph_resolution_ratios": [0.25, 0.50, 1.00],
        "graph_resolution_topks": [4, 8, 16],
        "graph_resolution_alphas": [0.03, 0.06, 0.10],
    },
    "GR14_two_level_sparse": {
        "graph_cluster_method": "current",
        "graph_resolution_ratios": [0.50, 1.00],
    },
    "GR20_graclus_matching_4_2_1": {
        "graph_cluster_method": "gr20_graclus_matching_4_2_1",
        "graph_resolution_capacities": [4, 2, 1],
    },
    "GR21_road_graclus_matching_4_2_1": {
        "graph_cluster_method": "gr21_road_graclus_matching_4_2_1",
        "graph_resolution_capacities": [4, 2, 1],
        "requires_road_distance": True,
    },
}

VARIANT_TO_METHOD = {
    "GR7_sparse_topk": "current",
    "GR14_two_level_sparse": "current",
    "GR17_road_spectral": "gr17_road_spectral",
    "GR20_graclus_matching_4_2_1": "gr20_graclus_matching_4_2_1",
    "GR21_road_graclus_matching_4_2_1": "gr21_road_graclus_matching_4_2_1",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PEMS-BAY Graph Resolution cluster visualization.")
    p.add_argument("--variants", nargs="+", default=["GR7_sparse_topk", "GR14_two_level_sparse"])
    p.add_argument("--out_dir", default="results/pems_bay_cluster_viz")
    p.add_argument("--cache_dir", default="results/pems_bay_cluster_viz/cache")
    p.add_argument("--dataset", default="PEMS-BAY")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force_rebuild_clusters", action="store_true")
    p.add_argument("--draw_edges", action="store_true")
    p.add_argument("--max_cluster_edges", type=int, default=3)
    return p.parse_args()


def _resolve_first(candidates: list[str]) -> Path | None:
    for rel in candidates:
        p = ROOT / rel
        if p.is_file():
            return p
    return None


def load_adj_pickle(adj_path: Path) -> tuple[list[str], dict[str, int], np.ndarray]:
    with open(adj_path, "rb") as f:
        try:
            obj = pickle.load(f)
        except UnicodeDecodeError:
            f.seek(0)
            obj = pickle.load(f, encoding="latin1")

    if not isinstance(obj, (list, tuple)) or len(obj) < 3:
        raise ValueError(f"Unexpected adj pickle structure at {adj_path}")

    sensor_ids = [str(s) for s in obj[0]]
    sensor_id_to_ind = {str(k): int(v) for k, v in obj[1].items()}
    adj = np.asarray(obj[2], dtype=np.float64)

    misaligned = [(i, sid, sensor_id_to_ind.get(sid)) for i, sid in enumerate(sensor_ids) if sensor_id_to_ind.get(sid) != i]
    if misaligned:
        raise ValueError(
            f"sensor_id_to_ind does not match sensor_ids order ({len(misaligned)} mismatches). "
            f"First: {misaligned[:3]}"
        )
    if adj.shape[0] != adj.shape[1]:
        raise ValueError(f"Adjacency matrix is not square: {adj.shape}")
    return sensor_ids, sensor_id_to_ind, adj


def load_coordinate_table(coord_path: Path) -> pd.DataFrame:
    df = pd.read_csv(coord_path, header=None, names=["sensor_id", "lat", "lon"])
    df["sensor_id"] = df["sensor_id"].astype(str)
    if df["sensor_id"].duplicated().any():
        dup = df["sensor_id"][df["sensor_id"].duplicated()].unique()[:5]
        raise ValueError(f"Duplicate sensor_id in {coord_path}: {dup}")
    return df


def align_coordinates(sensor_ids: list[str], coord_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    coord_map = {row.sensor_id: (float(row.lat), float(row.lon)) for row in coord_df.itertuples(index=False)}
    lons: list[float] = []
    lats: list[float] = []
    missing: list[tuple[int, str]] = []
    for i, sid in enumerate(sensor_ids):
        if sid not in coord_map:
            missing.append((i, sid))
            continue
        lat, lon = coord_map[sid]
        lats.append(lat)
        lons.append(lon)
    if missing:
        raise ValueError(
            f"Failed to align {len(missing)}/{len(sensor_ids)} sensors to coordinates. "
            f"Examples: {missing[:5]}. No silent fallback."
        )
    return np.asarray(lons, dtype=np.float64), np.asarray(lats, dtype=np.float64)


def load_road_distance_from_csv(dist_path: Path, sensor_ids: list[str], sensor_id_to_ind: dict[str, int]) -> np.ndarray | None:
    df = pd.read_csv(dist_path)
    cols = list(df.columns)
    if len(cols) < 3:
        return None
    n = len(sensor_ids)
    dist = np.full((n, n), np.inf, dtype=np.float64)
    np.fill_diagonal(dist, 0.0)
    for row in df.itertuples(index=False):
        sid_a, sid_b = str(row[0]), str(row[1])
        w = float(row[2])
        if sid_a not in sensor_id_to_ind or sid_b not in sensor_id_to_ind:
            continue
        i, j = sensor_id_to_ind[sid_a], sensor_id_to_ind[sid_b]
        dist[i, j] = min(dist[i, j], w)
        dist[j, i] = min(dist[j, i], w)
    finite = dist[np.isfinite(dist) & (dist > 0)]
    if finite.size == 0:
        return None
    fill = float(finite.max()) * 1.05
    dist[~np.isfinite(dist)] = fill
    return dist


def extended_cluster_stats(labels: np.ndarray) -> dict[str, float | int]:
    stats = cluster_size_stats(labels)
    sizes = np.bincount(labels.astype(np.int64))
    stats["median_cluster_size"] = float(np.median(sizes))
    stats["num_singleton_clusters"] = int((sizes == 1).sum())
    return stats


def compute_road_metrics(labels: np.ndarray, road_dist: np.ndarray, geo_km: np.ndarray | None) -> dict[str, float | None]:
    out: dict[str, float | None] = {
        "mean_intra_road_distance": None,
        "mean_inter_road_distance": None,
        "mean_cluster_spatial_diameter_km": None,
    }
    n_clusters = int(labels.max()) + 1
    intra_vals: list[float] = []
    inter_vals: list[float] = []
    diameters: list[float] = []

    for a in range(n_clusters):
        idx_a = np.where(labels == a)[0]
        if idx_a.size >= 2:
            sub = road_dist[np.ix_(idx_a, idx_a)]
            tri = sub[np.triu_indices(sub.shape[0], k=1)]
            intra_vals.append(float(tri.mean()))
            if geo_km is not None:
                gsub = geo_km[np.ix_(idx_a, idx_a)]
                diameters.append(float(gsub.max()))
        for b in range(a + 1, n_clusters):
            idx_b = np.where(labels == b)[0]
            if idx_a.size and idx_b.size:
                sub = road_dist[np.ix_(idx_a, idx_b)]
                inter_vals.append(float(sub.mean()))

    if intra_vals:
        out["mean_intra_road_distance"] = float(np.mean(intra_vals))
    if inter_vals:
        out["mean_inter_road_distance"] = float(np.mean(inter_vals))
    if diameters:
        out["mean_cluster_spatial_diameter_km"] = float(np.mean(diameters))
    return out


def validate_resolution(meta: dict[str, Any], variant: str, resolution: str, expected_m: int, node_size: int) -> dict[str, Any]:
    if node_size != EXPECTED_N:
        raise ValueError(f"Expected N={EXPECTED_N}, got N={node_size}")

    report = validate_cluster_assignment(meta)
    c = np.asarray(meta["C"])
    m = c.shape[1]

    if resolution != "S1" and abs(m - expected_m) > max(3, int(0.05 * expected_m)):
        raise ValueError(f"{variant} {resolution}: C has M={m}, expected about {expected_m}")

    if not report["row_one_hot_ok"]:
        raise ValueError(f"{variant} {resolution}: C rows are not one-hot")
    if not report["P_row_sum_ok"]:
        raise ValueError(f"{variant} {resolution}: P row sums are not 1")
    if report["min_cluster_size"] <= 0:
        raise ValueError(f"{variant} {resolution}: found empty cluster")

    print(
        f"  [check] {variant} {resolution}: C{c.shape} P{report['P_shape']} "
        f"min/max/mean size={report['min_cluster_size']}/{report['max_cluster_size']}/{report['mean_cluster_size']:.2f}"
    )
    return report


def save_extended_cache(
    cache_path: Path,
    meta: dict[str, Any],
    *,
    variant: str,
    ratio: float,
    sensor_ids: list[str],
    lon: np.ndarray,
    lat: np.ndarray,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        labels=meta["labels"],
        C=meta["C"],
        P=meta["P"],
        sensor_ids=np.asarray(sensor_ids, dtype=object),
        lon=lon,
        lat=lat,
        method=np.array("current_spectral"),
        ratio=np.array(ratio),
        num_clusters=np.array(meta["num_clusters"]),
        variant=np.array(variant),
    )
    sidecar = cache_path.with_suffix(".json")
    sidecar.write_text(
        json.dumps(
            {
                "variant": variant,
                "ratio": ratio,
                "num_clusters": int(meta["num_clusters"]),
                "method": "current_spectral",
                "sensor_ids": sensor_ids,
                "cache_npz": str(cache_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def plot_raw_sensors(
    lon: np.ndarray,
    lat: np.ndarray,
    edges: list[tuple[int, int]],
    out_path: Path,
    *,
    draw_edges: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 9), dpi=200)
    if draw_edges:
        for i, j in edges:
            ax.plot([lon[i], lon[j]], [lat[i], lat[j]], color="#dddddd", lw=0.2, alpha=0.35, zorder=1)
    ax.scatter(lon, lat, s=14, c="#1f77b4", edgecolors="white", linewidths=0.2, zorder=2)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("PEMS-BAY sensor locations")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_geo_node_clusters(
    lon: np.ndarray,
    lat: np.ndarray,
    labels: np.ndarray,
    edges: list[tuple[int, int]],
    title: str,
    out_path: Path,
    *,
    draw_edges: bool,
    show_centroids: bool = True,
) -> None:
    n_clusters = int(labels.max()) + 1
    cmap = plt.get_cmap("tab20" if n_clusters <= 20 else "nipy_spectral")
    colors = cmap(np.linspace(0, 1, n_clusters))

    fig, ax = plt.subplots(figsize=(10, 9), dpi=200)
    if draw_edges:
        for i, j in edges:
            ax.plot([lon[i], lon[j]], [lat[i], lat[j]], color="#cccccc", lw=0.2, alpha=0.35, zorder=1)
    for k in range(n_clusters):
        mask = labels == k
        ax.scatter(lon[mask], lat[mask], s=22, c=[colors[k]], edgecolors="white", linewidths=0.2, zorder=2)
    if show_centroids:
        for k in range(n_clusters):
            mask = labels == k
            if not mask.any():
                continue
            ax.scatter(
                lon[mask].mean(),
                lat[mask].mean(),
                s=55,
                facecolors="none",
                edgecolors="black",
                linewidths=0.8,
                zorder=3,
            )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def variant_prefix(variant: str) -> str:
    mapping = {
        "GR7_sparse_topk": "GR7",
        "GR14_two_level_sparse": "GR14",
        "GR20_graclus_matching_4_2_1": "GR20",
        "GR21_road_graclus_matching_4_2_1": "GR21",
    }
    return mapping.get(variant, variant.replace("_", ""))


def compute_cluster_edge_metrics(
    labels: np.ndarray,
    adj: np.ndarray,
    geo_km: np.ndarray,
) -> dict[str, float | int]:
    n = labels.shape[0]
    same_edges = 0
    total_edges = 0
    weighted_same = 0.0
    weighted_total = 0.0
    diameters: list[float] = []
    n_clusters = int(labels.max()) + 1
    fragmented = 0
    for k in range(n_clusters):
        idx = np.where(labels == k)[0]
        if idx.size <= 1:
            fragmented += 1
        if idx.size >= 2:
            sub = geo_km[np.ix_(idx, idx)]
            diameters.append(float(sub.max()))
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] <= 0:
                continue
            total_edges += 1
            w = float(adj[i, j])
            weighted_total += w
            if labels[i] == labels[j]:
                same_edges += 1
                weighted_same += w
    return {
        "edge_same_ratio": float(same_edges / total_edges) if total_edges else 0.0,
        "weighted_edge_same_ratio": float(weighted_same / weighted_total) if weighted_total else 0.0,
        "num_fragmented_clusters": int(fragmented),
        "mean_cluster_diameter_km": float(np.mean(diameters)) if diameters else 0.0,
        "max_cluster_diameter_km": float(np.max(diameters)) if diameters else 0.0,
    }


def plot_same_cluster_edges(
    lon: np.ndarray,
    lat: np.ndarray,
    labels: np.ndarray,
    adj: np.ndarray,
    title: str,
    out_path: Path,
) -> None:
    n_clusters = int(labels.max()) + 1
    cmap = plt.get_cmap("tab20" if n_clusters <= 20 else "nipy_spectral")
    colors = cmap(np.linspace(0, 1, n_clusters))
    fig, ax = plt.subplots(figsize=(10, 9), dpi=200)
    for i in range(len(lon)):
        for j in range(i + 1, len(lon)):
            if adj[i, j] <= 0 or labels[i] != labels[j]:
                continue
            c = colors[int(labels[i])]
            ax.plot([lon[i], lon[j]], [lat[i], lat[j]], color=c, lw=0.35, alpha=0.55, zorder=1)
    ax.scatter(lon, lat, s=12, c="#333333", zorder=2)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def build_or_load_cluster(
    *,
    variant: str,
    m: int,
    ratio: float,
    res_tag: str,
    node_size: int,
    adj_mx_path: str,
    cache_dir: Path,
    seed: int,
    dataset: str,
    sensor_ids: list[str],
    lon: np.ndarray,
    lat: np.ndarray,
    force_rebuild: bool,
    graph_cluster_method: str = GRAPH_CLUSTER_METHOD_CURRENT,
    cluster_road_distance_path: str | None = None,
) -> dict[str, Any]:
    ext_cache = cache_dir / f"{variant}_{res_tag}_M{m}.npz"
    if force_rebuild:
        for p in (ext_cache, ext_cache.with_suffix(".json")):
            if p.is_file():
                p.unlink()

    meta, base_cache = load_or_build_cluster_assignment(
        node_size=node_size,
        num_clusters=m,
        adj_mx_path=adj_mx_path,
        seed=seed,
        dataset_name=dataset,
        cache_dir=cache_dir,
        graph_cluster_method=graph_cluster_method,
        cluster_road_distance_path=cluster_road_distance_path,
        ratio=ratio,
    )
    meta["cache_path"] = str(base_cache)
    meta["variant"] = variant
    meta["ratio"] = ratio
    save_extended_cache(ext_cache, meta, variant=variant, ratio=ratio, sensor_ids=sensor_ids, lon=lon, lat=lat)
    meta["extended_cache_path"] = str(ext_cache)
    return meta


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    cache_dir = Path(args.cache_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    if not cache_dir.is_absolute():
        cache_dir = ROOT / cache_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    coord_path = _resolve_first(COORD_CANDIDATES)
    adj_path = _resolve_first(ADJ_CANDIDATES)
    dist_path = _resolve_first(DIST_CANDIDATES)
    if coord_path is None:
        raise FileNotFoundError(f"No PEMS-BAY coordinate file found. Tried: {COORD_CANDIDATES}")
    if adj_path is None:
        raise FileNotFoundError(f"No PEMS-BAY adjacency pickle found. Tried: {ADJ_CANDIDATES}")

    print(f"coord_csv: {coord_path}")
    print(f"adj_mx:    {adj_path}")
    print(f"dist_csv:  {dist_path if dist_path else '(not found, using adj+haversine proxy)'}")

    sensor_ids, sensor_id_to_ind, adj = load_adj_pickle(adj_path)
    node_size = adj.shape[0]
    if node_size != EXPECTED_N:
        raise ValueError(f"Expected {EXPECTED_N} nodes, got {node_size}")

    coord_df = load_coordinate_table(coord_path)
    lon, lat = align_coordinates(sensor_ids, coord_df)
    if lon.shape[0] != node_size:
        raise ValueError(f"Coordinate count {lon.shape[0]} != node_size {node_size}")

    print(f"Aligned {node_size} sensors via adj_mx sensor_ids (no file-order fallback).")
    print(f"  node_index=0 -> sensor_id={sensor_ids[0]} -> lon/lat=({lon[0]:.5f}, {lat[0]:.5f})")

    edges = build_edges_from_adj(adj)
    adj_bin = (adj > 0).astype(np.float64)
    geo_km = haversine_distance_matrix(lat, lon)

    road_dist: np.ndarray | None = None
    road_dist_source = ""
    if dist_path is not None:
        road_dist = load_road_distance_from_csv(dist_path, sensor_ids, sensor_id_to_ind)
        if road_dist is not None:
            road_dist_source = str(dist_path)
    if road_dist is None:
        road_dist = build_shortest_paths_from_adj_weights(adj, geo_km)
        road_dist_source = "adj_shortest_path_on_haversine_edge_weights"

    plot_raw_sensors(lon, lat, edges, out_dir / "pems_bay_sensors_raw.png", draw_edges=args.draw_edges)
    print("Saved pems_bay_sensors_raw.png")

    summary_rows: list[dict[str, Any]] = []
    labels_by_variant: dict[str, dict[str, np.ndarray]] = {}
    size_hist_by_variant: dict[str, dict[str, np.ndarray]] = {}

    for variant in args.variants:
        if variant not in VARIANT_CONFIGS:
            raise ValueError(f"Unknown variant {variant}. Choices: {list(VARIANT_CONFIGS)}")
        cfg = VARIANT_CONFIGS[variant]
        graph_method = cfg.get("graph_cluster_method", VARIANT_TO_METHOD.get(variant, "current"))
        prefix = variant_prefix(variant)
        labels_by_res: dict[str, np.ndarray] = {}
        size_hist: dict[str, np.ndarray] = {}
        road_path_arg = str(dist_path) if dist_path else None

        if cfg.get("requires_road_distance") and dist_path is None:
            print(f"\n=== {variant}: skipped (road distance file required but not found) ===")
            continue

        if "graph_resolution_capacities" in cfg:
            caps = resolve_graph_resolution_capacities(cfg["graph_resolution_capacities"])
            print(f"\n=== {variant} | capacities={caps} method={graph_method} ===")
            stages, ml_cache = load_or_build_multilevel_cluster_assignments(
                node_size=node_size,
                capacities=caps,
                adj_mx_path=str(adj_path),
                seed=args.seed,
                dataset_name=args.dataset,
                cache_dir=cache_dir,
                graph_cluster_method=graph_method,
                cluster_road_distance_path=road_path_arg,
            )
            nested_summary: dict[str, Any] = {"variant": variant, "nested_consistency": stages[0].get("nested_consistency")}
            for st in stages:
                res_tag = str(st.get("resolution_tag", ""))
                if res_tag == "S1":
                    continue
                m = int(st["num_clusters"])
                labels = np.asarray(st["labels"], dtype=np.int64)
                meta = dict(st)
                meta["cache_path"] = str(ml_cache)
                validate_resolution(meta, variant, res_tag, m, node_size)
                labels_by_res[res_tag] = labels
                stats = extended_cluster_stats(labels)
                edge_metrics = compute_cluster_edge_metrics(labels, adj_bin, geo_km)
                road_metrics = compute_road_metrics(labels, road_dist, geo_km)

                node_fig = out_dir / f"{prefix}_{res_tag}_node_clusters.png"
                same_fig = out_dir / f"{prefix}_{res_tag}_same_cluster_edges.png"
                cg_fig = out_dir / f"{prefix}_{res_tag}_cluster_graph.png"
                title_node = f"{variant}, {res_tag}, M={m}, max_size={stats['max_cluster_size']}"
                plot_geo_node_clusters(lon, lat, labels, edges, title_node, node_fig, draw_edges=args.draw_edges)
                plot_same_cluster_edges(lon, lat, labels, adj_bin, f"{variant} {res_tag} same-cluster edges", same_fig)
                xy = np.stack([lon, lat], axis=1)
                cc_edges = cluster_graph_edges(labels, adj_bin, road_dist, topk_per_cluster=args.max_cluster_edges)
                plot_cluster_graph(xy, labels, cc_edges, f"{prefix} {res_tag} cluster graph", cg_fig)
                size_hist[res_tag] = np.bincount(labels.astype(np.int64))
                print(
                    f"  {res_tag}: M={m} max={stats['max_cluster_size']} "
                    f"edge_same={edge_metrics['edge_same_ratio']:.3f} "
                    f"fragmented={edge_metrics['num_fragmented_clusters']}"
                )
                summary_rows.append(
                    {
                        "variant": variant,
                        "resolution": res_tag,
                        "cluster_method": graph_method,
                        "num_nodes": node_size,
                        "num_clusters": stats["num_clusters"],
                        "max_cluster_size": stats["max_cluster_size"],
                        "num_singleton_clusters": stats["num_singleton_clusters"],
                        "edge_same_ratio": edge_metrics["edge_same_ratio"],
                        "weighted_edge_same_ratio": edge_metrics["weighted_edge_same_ratio"],
                        "num_fragmented_clusters": edge_metrics["num_fragmented_clusters"],
                        "mean_cluster_diameter_km": edge_metrics["mean_cluster_diameter_km"],
                        "max_cluster_diameter_km": edge_metrics["max_cluster_diameter_km"],
                        "nested_consistency": st.get("nested_consistency"),
                        "figure_node_cluster": str(node_fig),
                        "figure_same_cluster_edges": str(same_fig),
                        "figure_cluster_graph": str(cg_fig),
                        "cache_path": str(ml_cache),
                        "road_distance_source": road_dist_source,
                    }
                )
            if "S4" in labels_by_res and "S2" in labels_by_res:
                mat = overlap_matrix(labels_by_res["S4"], labels_by_res["S2"])
                mat_sorted, _, _ = sort_overlap_rows_cols(mat)
                overlap_path = out_dir / f"{prefix}_overlap_S4_to_S2_heatmap.png"
                plot_overlap_heatmap(mat_sorted, f"{variant} overlap S<=4 -> S<=2", overlap_path)
            nested_path = out_dir / f"{prefix}_nested_consistency.json"
            nested_path.write_text(json.dumps(nested_summary, indent=2), encoding="utf-8")
        else:
            ratios = cfg["graph_resolution_ratios"]
            sizes = resolve_graph_resolution_sizes(node_size, ratios)
            ratio_by_m = {m: r for r, m in zip(ratios, sizes)}
            print(f"\n=== {variant} | ratios={ratios} -> M={sizes} ===")

            for m in sizes:
                ratio = ratio_by_m[m]
                res_tag = ratio_to_tag(ratio, m)
                if res_tag == "S1":
                    print(f"  {res_tag}: identity (M={m}), using raw sensor map only")
                    continue

                meta = build_or_load_cluster(
                    variant=variant,
                    m=m,
                    ratio=ratio,
                    res_tag=res_tag,
                    node_size=node_size,
                    adj_mx_path=str(adj_path),
                    cache_dir=cache_dir,
                    seed=args.seed,
                    dataset=args.dataset,
                    sensor_ids=sensor_ids,
                    lon=lon,
                    lat=lat,
                    force_rebuild=args.force_rebuild_clusters,
                    graph_cluster_method=graph_method,
                    cluster_road_distance_path=road_path_arg,
                )
                validate_resolution(meta, variant, res_tag, m, node_size)

                labels = np.asarray(meta["labels"], dtype=np.int64)
                labels_by_res[res_tag] = labels
                stats = extended_cluster_stats(labels)
                edge_metrics = compute_cluster_edge_metrics(labels, adj_bin, geo_km)
                road_metrics = compute_road_metrics(labels, road_dist, geo_km)

                node_fig = out_dir / f"{prefix}_{res_tag}_node_clusters.png"
                same_fig = out_dir / f"{prefix}_{res_tag}_same_cluster_edges.png"
                cg_fig = out_dir / f"{prefix}_{res_tag}_cluster_graph.png"
                title_node = (
                    f"{variant}, {res_tag.replace('S14', 'S1/4').replace('S12', 'S1/2')}, "
                    f"M≈{m}, num_clusters={stats['num_clusters']}"
                )
                plot_geo_node_clusters(lon, lat, labels, edges, title_node, node_fig, draw_edges=args.draw_edges)
                plot_same_cluster_edges(
                    lon, lat, labels, adj_bin,
                    f"{variant} {res_tag.replace('S14', 'S1/4').replace('S12', 'S1/2')} same-cluster edges",
                    same_fig,
                )
                xy = np.stack([lon, lat], axis=1)
                cc_edges = cluster_graph_edges(labels, adj_bin, road_dist, topk_per_cluster=args.max_cluster_edges)
                plot_cluster_graph(xy, labels, cc_edges, f"{prefix} {res_tag} cluster graph", cg_fig)

                size_hist[res_tag] = np.bincount(labels.astype(np.int64))
                print(
                    f"  {res_tag}: clusters={stats['num_clusters']} "
                    f"max={stats['max_cluster_size']} singletons={stats['num_singleton_clusters']} "
                    f"edge_same={edge_metrics['edge_same_ratio']:.3f}"
                )
                summary_rows.append(
                    {
                        "variant": variant,
                        "resolution": res_tag,
                        "cluster_method": graph_method,
                        "num_nodes": node_size,
                        "num_clusters": stats["num_clusters"],
                        "max_cluster_size": stats["max_cluster_size"],
                        "num_singleton_clusters": stats["num_singleton_clusters"],
                        "edge_same_ratio": edge_metrics["edge_same_ratio"],
                        "weighted_edge_same_ratio": edge_metrics["weighted_edge_same_ratio"],
                        "num_fragmented_clusters": edge_metrics["num_fragmented_clusters"],
                        "mean_cluster_diameter_km": edge_metrics["mean_cluster_diameter_km"],
                        "max_cluster_diameter_km": edge_metrics["max_cluster_diameter_km"],
                        "figure_node_cluster": str(node_fig),
                        "figure_same_cluster_edges": str(same_fig),
                        "figure_cluster_graph": str(cg_fig),
                        "cache_path": meta.get("extended_cache_path", meta.get("cache_path", "")),
                        "mean_intra_road_distance": road_metrics["mean_intra_road_distance"],
                        "mean_inter_road_distance": road_metrics["mean_inter_road_distance"],
                        "road_distance_source": road_dist_source,
                    }
                )

            if "S14" in labels_by_res and "S12" in labels_by_res:
                mat = overlap_matrix(labels_by_res["S14"], labels_by_res["S12"])
                mat_sorted, _, _ = sort_overlap_rows_cols(mat)
                overlap_path = out_dir / f"{prefix}_overlap_S14_to_S12_heatmap.png"
                plot_overlap_heatmap(mat_sorted, f"{variant} overlap S1/4 -> S1/2", overlap_path)

        labels_by_variant[variant] = labels_by_res
        size_hist_by_variant[variant] = size_hist
        if size_hist:
            plot_cluster_size_hist(size_hist, prefix, out_dir / f"{prefix}_cluster_size_hist.png")

    if "GR7_sparse_topk" in labels_by_variant:
        gr7 = labels_by_variant["GR7_sparse_topk"]
        if "S14" in gr7 and "S12" in gr7 and not (out_dir / "GR7_overlap_S14_to_S12_heatmap.png").is_file():
            mat = overlap_matrix(gr7["S14"], gr7["S12"])
            mat_sorted, _, _ = sort_overlap_rows_cols(mat)
            overlap_path = out_dir / "GR7_overlap_S14_to_S12_heatmap.png"
            plot_overlap_heatmap(
                mat_sorted,
                "GR7_sparse_topk overlap heatmap S1/4 -> S1/2",
                overlap_path,
            )
            for row in summary_rows:
                if row["variant"] == "GR7_sparse_topk":
                    row["figure_overlap"] = str(overlap_path)

    write_summary_table(summary_rows, out_dir / "cluster_summary.csv", out_dir / "cluster_summary.md")

    print("\n=== Generated figures ===")
    for p in sorted(out_dir.glob("*.png")):
        print(f"  {p}")

    print("\n=== Cluster size summary ===")
    for variant in args.variants:
        if variant not in labels_by_variant:
            continue
        print(f"\n{variant}:")
        for res_tag, labels in sorted(labels_by_variant[variant].items()):
            stats = extended_cluster_stats(labels)
            print(
                f"  {res_tag}: num_clusters={stats['num_clusters']} "
                f"min={stats['min_cluster_size']} max={stats['max_cluster_size']} "
                f"mean={stats['mean_cluster_size']:.2f} median={stats['median_cluster_size']:.1f} "
                f"singletons={stats['num_singleton_clusters']}"
            )

    if dist_path is None:
        print("\nNote: distances_bay_2017.csv not found; road metrics use adj shortest-path on haversine edge weights.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
