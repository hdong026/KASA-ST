#!/usr/bin/env python3
"""Visualize PeMS04 graph-resolution cluster assignments.

PeMS04 in this repo has NO official sensor lat/lon. Layout uses 2D MDS on the
road-network distance matrix (adj_PEMS04_distance.pkl) as a geographic proxy.
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from sklearn.manifold import MDS

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basicts.archs.arch_zoo.ChainForecasting_arch.graph_cluster_utils import (
    load_or_build_cluster_assignment,
    resolve_graph_resolution_sizes,
)

NODE_SIZE = 307
RATIOS = [0.25, 0.50, 1.00]
ADJ = ROOT / "datasets" / "PEMS04" / "adj_mx.pkl"
DIST = ROOT / "datasets" / "raw_data" / "PEMS04" / "adj_PEMS04_distance.pkl"
ADJ_EDGE = ROOT / "datasets" / "raw_data" / "PEMS04" / "PEMS04.csv"

METHODS = [
    ("current", "GR7 spectral (current)", {}),
    ("pearson_balanced_pam", "GR9 Pearson + balanced PAM", {}),
    ("xcorr_balanced_pam", "GR10 xcorr + balanced PAM", {"cluster_max_lag": 12}),
    (
        "joint_pearson_spatial_balanced_pam",
        "GR11 joint Pearson+spatial PAM",
        {
            "cluster_lambda_s": 0.2,
            "cluster_spatial_coord_path": str(DIST.relative_to(ROOT)),
        },
    ),
    ("pearson_standard_pam", "GR12 Pearson + standard PAM", {}),
    ("autocorr_feature_balanced_pam", "GR13 autocorr feature PAM", {"cluster_acf_lag": 24}),
]


def load_distance_matrix() -> np.ndarray:
    obj = pickle.load(open(DIST, "rb"))
    if hasattr(obj, "numpy"):
        mat = obj.numpy()
    else:
        mat = np.asarray(obj, dtype=np.float64)
    mat = np.nan_to_num(mat, nan=0.0)
    np.fill_diagonal(mat, 0.0)
    maxv = mat.max()
    if maxv > 0:
        mat = mat / maxv
    return mat


def mds_layout(dist: np.ndarray, seed: int = 0) -> np.ndarray:
    """2D embedding from pairwise road distances."""
    mds = MDS(
        n_components=2,
        dissimilarity="precomputed",
        random_state=seed,
        normalized_stress="auto",
        max_iter=300,
        n_init=4,
    )
    return mds.fit_transform(dist)


def load_labels(method: str, m: int, extra: dict) -> tuple[np.ndarray, str]:
    kwargs = {
        "node_size": NODE_SIZE,
        "num_clusters": m,
        "adj_mx_path": str(ADJ),
        "seed": 0,
        "dataset_name": "PEMS04",
        "graph_cluster_method": method,
        "cluster_max_lag": extra.get("cluster_max_lag", 12),
        "cluster_lambda_s": extra.get("cluster_lambda_s", 0.2),
        "cluster_acf_lag": extra.get("cluster_acf_lag", 24),
        "cluster_spatial_coord_path": extra.get("cluster_spatial_coord_path"),
    }
    meta, cache_path = load_or_build_cluster_assignment(**kwargs)
    return np.asarray(meta["labels"]), str(cache_path)


def plot_clusters(
    xy: np.ndarray,
    labels: np.ndarray,
    title: str,
    out_path: Path,
    edges: bool = True,
) -> None:
    n_clusters = int(labels.max()) + 1
    cmap = plt.get_cmap("tab20" if n_clusters <= 20 else "nipy_spectral")
    colors = cmap(np.linspace(0, 1, n_clusters))

    fig, ax = plt.subplots(figsize=(11, 9), dpi=150)
    if edges and ADJ_EDGE.is_file():
        import pandas as pd

        df = pd.read_csv(ADJ_EDGE)
        for _, row in df.iterrows():
            i, j = int(row["from"]), int(row["to"])
            ax.plot(
                [xy[i, 0], xy[j, 0]],
                [xy[i, 1], xy[j, 1]],
                color="#cccccc",
                linewidth=0.25,
                alpha=0.35,
                zorder=1,
            )

    for k in range(n_clusters):
        mask = labels == k
        ax.scatter(
            xy[mask, 0],
            xy[mask, 1],
            s=28,
            c=[colors[k]],
            edgecolors="white",
            linewidths=0.3,
            alpha=0.92,
            zorder=2,
        )

    sizes = np.bincount(labels, minlength=n_clusters)
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=colors[k],
            markersize=6,
            label=f"C{k} (n={sizes[k]})",
        )
        for k in range(min(n_clusters, 12))
    ]
    if n_clusters > 12:
        legend_handles.append(
            Line2D([0], [0], marker="", color="w", label=f"... +{n_clusters - 12} clusters")
        )
    ax.legend(handles=legend_handles, loc="upper left", fontsize=7, framealpha=0.9)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("MDS dim-1 (from road distance)")
    ax.set_ylabel("MDS dim-2 (from road distance)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_overview(xy: np.ndarray, m: int, out_path: Path) -> None:
    n_methods = len(METHODS)
    cols = 3
    rows = int(np.ceil(n_methods / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows), dpi=130)
    axes = np.atleast_1d(axes).ravel()

    for ax, (method, title, extra) in zip(axes, METHODS):
        labels, _ = load_labels(method, m, extra)
        n_clusters = int(labels.max()) + 1
        cmap = plt.get_cmap("nipy_spectral")
        colors = cmap(np.linspace(0, 1, n_clusters))
        for k in range(n_clusters):
            mask = labels == k
            ax.scatter(xy[mask, 0], xy[mask, 1], s=10, c=[colors[k]], alpha=0.85)
        ax.set_title(f"{title}\nM={m}", fontsize=9)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.15)

    for j in range(len(METHODS), len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        "PeMS04 cluster partitions (MDS layout from road distances; NOT real GPS)",
        fontsize=12,
        y=1.01,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def write_readme(out_dir: Path) -> None:
    text = """# PeMS04 聚类可视化说明

## 是否有真实 GPS 坐标？

**本仓库没有 PeMS04 传感器的经纬度（lat/lon）文件。**

现有空间信息仅包括：
- `datasets/raw_data/PEMS04/PEMS04.csv`：传感器之间的**道路距离**边
- `adj_PEMS04_distance.pkl`：307×307 距离矩阵

因此**无法**直接调用 Google Maps / OpenStreetMap 等 API 把节点落到真实地图上
（除非额外从 Caltrans PeMS 或第三方数据集如 LargeST 的 `ca_meta.csv` 匹配 station ID）。

## 当前可视化方法

使用 **MDS（多维缩放）** 将道路距离矩阵嵌入 2D 平面：
- 距离近的传感器在图上更近
- 叠加 `PEMS04.csv` 道路边（灰色细线）
- 颜色 = 聚类 label（M=77 粗分辨率）

这是交通预测论文中常用的**伪地理布局**，可对比不同 C^(r) 构造，但不是真实地图。

## 文件

- `overview_M77.png`：6 种方法总览
- `{method}_M77.png` / `{method}_M154.png`：单方法详图

## 如何获得真实地图

若需要真实地图可视化，需要：
1. 获取 PeMS station ID ↔ (lat, lon) 映射（如 LargeST / 手动从 pems.dot.ca.gov 查询）
2. 确认 307 个节点 ID 与 PeMS04 数据集节点顺序一致
3. 再用 folium / mapbox 绑点

"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="results/pems04_cluster_visualizations")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir

    print("Loading distance matrix and computing MDS layout...")
    dist = load_distance_matrix()
    xy = mds_layout(dist)
    sizes = resolve_graph_resolution_sizes(NODE_SIZE, RATIOS)
    coarse = [m for m in sizes if m < NODE_SIZE]
    print(f"Coarse resolutions: {coarse}")

    for m in coarse:
        print(f"Plotting overview M={m}...")
        plot_overview(xy, m, out_dir / f"overview_M{m}.png")
        for method, title, extra in METHODS:
            labels, cache = load_labels(method, m, extra)
            safe = method.replace("/", "_")
            fname = out_dir / f"{safe}_M{m}.png"
            plot_clusters(
                xy,
                labels,
                f"{title} | M={m} | N=307\n(cache: {Path(cache).name})",
                fname,
            )
            print(f"  wrote {fname.name}")

    write_readme(out_dir)
    print(f"\nDone. See {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
