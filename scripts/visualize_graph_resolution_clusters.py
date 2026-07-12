#!/usr/bin/env python3
"""Visualize Graph Resolution cluster matrices and compare clustering methods."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basicts.archs.arch_zoo.ChainForecasting_arch.graph_cluster_utils import resolve_graph_resolution_sizes
from scripts.utils.cluster_visualization_utils import (
    cluster_graph_edges,
    cluster_size_stats,
    compute_extra_metrics,
    compute_label_similarity,
    dataset_paths,
    labels_from_meta,
    load_cluster_meta,
    load_sensor_coordinates,
    method_file_tag,
    overlap_matrix,
    plot_cluster_graph,
    plot_cluster_size_hist,
    plot_folium_map,
    plot_geo_clusters,
    plot_layout_base,
    plot_node_clusters,
    plot_overlap_heatmap,
    plot_side_by_side,
    plot_similarity_heatmap,
    prepare_dataset_assets,
    ratio_to_tag,
    resolve_method,
    sort_overlap_rows_cols,
    write_summary_table,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize Graph Resolution cluster matrices.")
    p.add_argument("--dataset", default="PEMS04")
    p.add_argument("--methods", nargs="+", default=["current", "pearson_balanced_pam", "joint_pearson_roadcost_pam"])
    p.add_argument("--ratios", type=float, nargs="+", default=[0.25, 0.50, 1.0])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", default="")
    p.add_argument("--force_rebuild_layout", action="store_true")
    p.add_argument("--compare_methods", nargs="+", default=None)
    p.add_argument("--skip_s1", action="store_true", help="Skip full-resolution S1=identity plots.")
    p.add_argument("--layout", choices=["mds", "geo", "both"], default="both")
    p.add_argument("--folium", action="store_true", help="Also export interactive folium HTML maps when GPS exists.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    dataset = args.dataset
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "results" / f"cluster_viz_{dataset.lower()}"
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = dataset_paths(dataset)
    node_size = cfg["node_size"]
    sizes = resolve_graph_resolution_sizes(node_size, args.ratios)
    ratio_by_m = {m: r for m, r in zip(sizes, args.ratios)}
    if args.skip_s1:
        sizes = [m for m in sizes if m < node_size]

    print(f"Dataset={dataset}, N={node_size}, resolutions={sizes}")

    assets = prepare_dataset_assets(
        dataset,
        out_dir,
        force_rebuild_layout=args.force_rebuild_layout,
        seed=args.seed,
    )
    xy_mds = assets["xy_mds"]
    xy_geo = assets["xy_geo"]
    edges = assets["edges"]
    road_dist = assets["road_dist"]
    adj_bin = assets["adj_bin"]
    layout_method = assets["layout_method"]

    plot_layout_base(
        xy_mds,
        edges,
        out_dir / f"{dataset.lower()}_layout_mds.png",
        f"{dataset} node layout ({layout_method})",
    )
    if xy_geo is not None:
        plot_layout_base(
            xy_geo,
            edges,
            out_dir / f"{dataset.lower()}_layout_geo.png",
            f"{dataset} GPS layout (lat/lon)",
        )
    print(f"Layout saved ({layout_method})")

    lon = lat = None
    if xy_geo is not None:
        lon, lat = xy_geo[:, 0], xy_geo[:, 1]

    summary_rows: list[dict] = []
    labels_store: dict[str, dict[str, np.ndarray]] = {}

    for method in args.methods:
        internal = resolve_method(method)
        tag = method_file_tag(internal)
        print(f"\n=== Method: {tag} (internal={internal}) ===")
        labels_by_res: dict[str, np.ndarray] = {}
        size_hist: dict[str, np.ndarray] = {}

        for m in sorted([x for x in sizes if ratio_to_tag(ratio_by_m[x], x) != "S1"], key=lambda x: -x):
            ratio = ratio_by_m[m]
            res_tag = ratio_to_tag(ratio, m)

            meta = load_cluster_meta(method, m, dataset, args.seed, out_dir=out_dir)
            labels = labels_from_meta(meta)
            labels_by_res[res_tag] = labels
            stats = cluster_size_stats(labels)
            extra = compute_extra_metrics(internal, labels, road_dist, dataset)
            medoids = meta.get("medoids")
            if medoids is not None:
                medoids = np.asarray(medoids)

            node_path = out_dir / f"{tag}_{res_tag}_node_clusters.png"
            cg_path = out_dir / f"{tag}_{res_tag}_cluster_graph.png"
            geo_path = out_dir / f"{tag}_{res_tag}_geo_map.png"
            folium_path = out_dir / f"{tag}_{res_tag}_geo_map.html"
            title = (
                f"{tag} | {res_tag} (M={m}) | mean_size={stats['mean_cluster_size']:.2f} "
                f"min/max={stats['min_cluster_size']}/{stats['max_cluster_size']}"
            )

            if args.layout in ("mds", "both"):
                plot_node_clusters(xy_mds, labels, edges, title + " [MDS]", node_path, medoids=medoids)
                cc_edges = cluster_graph_edges(labels, adj_bin, road_dist, topk_per_cluster=3)
                plot_cluster_graph(
                    xy_mds,
                    labels,
                    cc_edges,
                    f"{tag} | {res_tag} coarse cluster graph [MDS]",
                    cg_path,
                    medoids=medoids,
                )

            if args.layout in ("geo", "both") and lon is not None and lat is not None:
                plot_geo_clusters(lon, lat, labels, edges, title + " [GPS]", geo_path)
                if args.folium:
                    plot_folium_map(lon, lat, labels, title, folium_path)

            size_hist[res_tag] = np.bincount(labels.astype(np.int64))
            print(
                f"  {res_tag}: clusters={stats['num_clusters']} "
                f"sizes min/max/mean={stats['min_cluster_size']}/{stats['max_cluster_size']}/{stats['mean_cluster_size']:.2f}"
            )

            summary_rows.append(
                {
                    "dataset": dataset,
                    "method": tag,
                    "resolution": res_tag,
                    "ratio": ratio,
                    "num_nodes": node_size,
                    "num_clusters": stats["num_clusters"],
                    "min_cluster_size": stats["min_cluster_size"],
                    "max_cluster_size": stats["max_cluster_size"],
                    "mean_cluster_size": round(stats["mean_cluster_size"], 4),
                    "std_cluster_size": round(stats["std_cluster_size"], 4),
                    "medoid_available": bool(medoids is not None and len(medoids) > 0),
                    "mean_intra_abs_corr": extra.get("mean_intra_abs_corr"),
                    "mean_inter_abs_corr": extra.get("mean_inter_abs_corr"),
                    "mean_intra_road_distance": extra.get("mean_intra_road_distance"),
                    "cache_path": meta.get("cache_path", ""),
                    "figure_path_node": str(node_path),
                    "figure_path_cluster_graph": str(cg_path),
                    "figure_path_geo": str(geo_path) if lon is not None else "",
                    "figure_path_overlap": "",
                }
            )

        labels_store[tag] = labels_by_res
        if size_hist:
            plot_cluster_size_hist(size_hist, tag, out_dir / f"{tag}_cluster_size_hist.png")

        if "S14" in labels_by_res and "S12" in labels_by_res:
            mat = overlap_matrix(labels_by_res["S14"], labels_by_res["S12"])
            mat_sorted, _, _ = sort_overlap_rows_cols(mat)
            overlap_path = out_dir / f"{tag}_overlap_S14_to_S12_heatmap.png"
            plot_overlap_heatmap(mat_sorted, f"{tag} overlap heatmap S1/4 -> S1/2", overlap_path)
            for row in summary_rows:
                if row["method"] == tag:
                    row["figure_path_overlap"] = str(overlap_path)

    compare_pairs = args.compare_methods or []
    if len(compare_pairs) >= 2:
        a, b = compare_pairs[0], compare_pairs[1]
        ta, tb = method_file_tag(resolve_method(a)), method_file_tag(resolve_method(b))
        if "S14" in labels_store.get(ta, {}) and "S14" in labels_store.get(tb, {}):
            xy_cmp = xy_geo if xy_geo is not None and args.layout in ("geo", "both") else xy_mds
            plot_side_by_side(
                xy_cmp,
                labels_store[ta]["S14"],
                labels_store[tb]["S14"],
                f"{ta} S1/4",
                f"{tb} S1/4",
                edges,
                out_dir / f"compare_{ta}_vs_{tb}_S14.png",
            )

    if "current" in labels_store and "pearson_balanced_pam" in labels_store:
        if "S14" in labels_store["current"] and "S14" in labels_store["pearson_balanced_pam"]:
            xy_cmp = xy_geo if xy_geo is not None and args.layout in ("geo", "both") else xy_mds
            plot_side_by_side(
                xy_cmp,
                labels_store["current"]["S14"],
                labels_store["pearson_balanced_pam"]["S14"],
                "current S1/4",
                "pearson_balanced_pam S1/4",
                edges,
                out_dir / "compare_current_vs_pearson_balanced_pam_S14.png",
            )

    ds_tag = dataset.lower().replace("-", "")
    for res_tag in ("S14", "S12"):
        subset = {m: lab[res_tag] for m, lab in labels_store.items() if res_tag in lab}
        if len(subset) >= 2:
            sim = compute_label_similarity(subset)
            sim.to_csv(out_dir / f"{ds_tag}_cluster_method_similarity_{res_tag}.csv", index=False)
            if res_tag == "S14":
                plot_similarity_heatmap(
                    sim,
                    "ari",
                    out_dir / f"{ds_tag}_cluster_method_similarity_heatmap.png",
                    f"{dataset} cluster label similarity ({res_tag}, ARI)",
                )

    sim_all = []
    for res_tag in ("S14", "S12"):
        p = out_dir / f"{ds_tag}_cluster_method_similarity_{res_tag}.csv"
        if p.is_file():
            df = pd.read_csv(p)
            df["resolution"] = res_tag
            sim_all.append(df)
    if sim_all:
        pd.concat(sim_all, ignore_index=True).to_csv(out_dir / f"{ds_tag}_cluster_method_similarity.csv", index=False)

    write_summary_table(summary_rows, out_dir / "cluster_summary.csv", out_dir / "cluster_summary.md")

    print("\n=== Generated figures ===")
    for p in sorted(out_dir.glob("*")):
        if p.suffix in {".png", ".html"}:
            print(f"  {p}")
    print(f"\nSummary: {out_dir / 'cluster_summary.csv'}")
    if lon is None:
        print("Note: no GPS coordinates available; only MDS layout was produced.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
