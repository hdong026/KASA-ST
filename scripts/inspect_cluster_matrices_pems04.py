#!/usr/bin/env python3
"""Inspect and summarize PAM / distance-based graph cluster matrices for PeMS04."""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basicts.archs.arch_zoo.ChainForecasting_arch.graph_cluster_utils import (
    build_distance_matrix_for_method,
    cluster_capacities,
    load_or_build_cluster_assignment,
    load_spatial_distance_matrix,
    load_train_node_series,
    resolve_graph_resolution_sizes,
    summarize_cluster_quality,
    validate_cluster_assignment,
)

ADJ = ROOT / "datasets" / "PEMS04" / "adj_mx.pkl"
SPATIAL = ROOT / "datasets" / "raw_data" / "PEMS04" / "adj_PEMS04_distance.pkl"
NODE_SIZE = 307
RATIOS = [0.25, 0.50, 1.00]

METHOD_CONFIGS = [
    ("current", {}),
    ("pearson_balanced_pam", {}),
    ("xcorr_balanced_pam", {"cluster_max_lag": 12}),
    ("joint_pearson_spatial_balanced_pam", {
        "cluster_lambda_s": 0.2,
        "cluster_spatial_coord_path": str(SPATIAL.relative_to(ROOT)),
    }),
    ("pearson_standard_pam", {}),
    ("autocorr_feature_balanced_pam", {"cluster_acf_lag": 24}),
]


def mean_spatial_diameter(labels: np.ndarray, spatial_dist: np.ndarray) -> float | None:
    diameters = []
    for k in range(int(labels.max()) + 1):
        idx = np.where(labels == k)[0]
        if idx.size < 2:
            continue
        sub = spatial_dist[np.ix_(idx, idx)]
        diameters.append(float(sub.max()))
    return float(np.mean(diameters)) if diameters else None


def inspect_one(method: str, m: int, seed: int, extra: dict) -> dict:
    kwargs = {
        "node_size": NODE_SIZE,
        "num_clusters": m,
        "adj_mx_path": str(ADJ),
        "seed": seed,
        "dataset_name": "PEMS04",
        "graph_cluster_method": method,
        "cluster_max_lag": extra.get("cluster_max_lag", 12),
        "cluster_lambda_s": extra.get("cluster_lambda_s", 0.2),
        "cluster_acf_lag": extra.get("cluster_acf_lag", 24),
        "cluster_spatial_coord_path": extra.get("cluster_spatial_coord_path"),
    }
    meta, cache_path = load_or_build_cluster_assignment(**kwargs)
    val = validate_cluster_assignment(meta)
    dist = None
    if method != "current":
        series = load_train_node_series(dataset_name="PEMS04")
        dist, _ = build_distance_matrix_for_method(
            method=method,
            train_series=series,
            node_size=NODE_SIZE,
            cluster_max_lag=kwargs["cluster_max_lag"],
            cluster_lambda_s=kwargs["cluster_lambda_s"],
            cluster_acf_lag=kwargs["cluster_acf_lag"],
            cluster_spatial_coord_path=kwargs["cluster_spatial_coord_path"],
        )
    quality = summarize_cluster_quality(meta, dist=dist)
    spatial_diam = None
    if SPATIAL.is_file():
        try:
            sdist = load_spatial_distance_matrix(SPATIAL, NODE_SIZE)
            spatial_diam = mean_spatial_diameter(np.asarray(meta["labels"]), sdist)
        except Exception:
            spatial_diam = None
    caps = cluster_capacities(NODE_SIZE, m)
    return {
        "method": method,
        "num_clusters": m,
        "seed": seed,
        "min_cluster_size": val["min_cluster_size"],
        "max_cluster_size": val["max_cluster_size"],
        "mean_cluster_size": val["mean_cluster_size"],
        "std_cluster_size": val["std_cluster_size"],
        "expected_capacities": str(sorted(caps.tolist(), reverse=True)),
        "mean_intra_distance": quality.get("mean_intra_distance"),
        "mean_intra_abs_corr": quality.get("mean_intra_abs_corr"),
        "mean_spatial_diameter": spatial_diam,
        "medoid_count": quality.get("medoid_count", 0),
        "clustering_method": meta.get("clustering_method", ""),
        "cache_path": str(cache_path),
    }


def write_outputs(rows: list[dict], out_csv: Path, out_md: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    lines = [
        "# PeMS04 Graph Cluster Matrix Summary\n\n",
        "Train-only series used for distance-based methods. "
        "Spectral `current` uses adjacency only.\n\n",
        "| method | M | min | max | mean size | std size | intra dist | intra |corr| | spatial diam | medoids | cache |\n",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n",
    ]
    for r in rows:
        lines.append(
            f"| {r['method']} | {r['num_clusters']} | {r['min_cluster_size']} | {r['max_cluster_size']} | "
            f"{r['mean_cluster_size']:.2f} | {r['std_cluster_size']:.2f} | "
            f"{r.get('mean_intra_distance', '')} | {r.get('mean_intra_abs_corr', '')} | "
            f"{r.get('mean_spatial_diameter', '')} | {r.get('medoid_count', '')} | "
            f"`{Path(r['cache_path']).name}` |\n"
        )
    out_md.write_text("".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-csv", default="results/pems04_cluster_matrix_summary.csv")
    parser.add_argument("--out-md", default="results/pems04_cluster_matrix_summary.md")
    args = parser.parse_args()

    sizes = resolve_graph_resolution_sizes(NODE_SIZE, RATIOS)
    coarse_sizes = [m for m in sizes if m < NODE_SIZE]
    rows = []
    for method, extra in METHOD_CONFIGS:
        if method == "joint_pearson_spatial_balanced_pam" and not SPATIAL.is_file():
            print(f"[skip] {method}: spatial distance file missing")
            continue
        for m in coarse_sizes:
            print(f"Inspect {method} M={m} ...")
            rows.append(inspect_one(method, m, args.seed, extra))

    out_csv = Path(args.out_csv)
    out_md = Path(args.out_md)
    if not out_csv.is_absolute():
        out_csv = ROOT / out_csv
    if not out_md.is_absolute():
        out_md = ROOT / out_md
    write_outputs(rows, out_csv, out_md)
    print(f"Wrote {out_csv}, {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
