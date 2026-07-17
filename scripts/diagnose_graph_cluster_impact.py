#!/usr/bin/env python3
"""Diagnose why different graph_cluster_method variants yield similar test MAE."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basicts.archs import ChainForecasting
from basicts.archs.arch_zoo.ChainForecasting_arch.graph_cluster_utils import (
    CAPACITY_MULTILEVEL_METHODS,
    load_or_build_cluster_assignment,
    load_or_build_multilevel_cluster_assignments,
    load_raw_adj_numpy,
    resolve_graph_resolution_capacities,
    resolve_graph_resolution_sizes,
    validate_cluster_assignment,
)
from basicts.data import TimeSeriesForecastingDataset
from basicts.metrics import masked_mae
from basicts.runners.base_tsf_runner import BaseTimeSeriesForecastingRunner
from basicts.utils import load_pkl
from scripts.run_pems04_16_32_64_unified_aux import (
    HORIZON_CONFIGS,
    generate_temp_config,
    variant_spec,
)

NODE_SIZE = 307
ADJ = ROOT / "datasets" / "PEMS04" / "adj_mx.pkl"
ROAD = ROOT / "datasets" / "raw_data" / "PEMS04" / "adj_PEMS04_distance.pkl"
CKPT_ROOT = ROOT / "checkpoints" / "pems04_unified_aux"
WORK_DIR = ROOT / "experiments" / "pems04_unified_aux"

DEFAULT_VARIANTS = [
    "GR7_sparse_topk",
    "GR17_road_spectral",
    "GR18_constrained_spectral_cap_dist",
    "GR19_spectral_constrained_kmeans_cap",
    "GR19a_cap_only_spectral",
    "GR19b_road_cap_spectral",
    "GR20_graclus_matching_4_2_1",
    "GR21_road_graclus_matching_4_2_1",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Graph cluster method impact diagnostics.")
    p.add_argument("--horizon", type=int, default=32)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS)
    p.add_argument("--out_dir", default="results/graph_cluster_diagnosis")
    p.add_argument("--max_val_batches", type=int, default=0, help="0 = all val batches")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def ratio_to_tag(ratio: float, m: int) -> str:
    if abs(ratio - 0.25) < 0.02:
        return "S14"
    if abs(ratio - 0.50) < 0.02:
        return "S12"
    if abs(ratio - 1.0) < 0.02 or m >= NODE_SIZE:
        return "S1"
    return f"M{m}"


def edge_same_ratio(labels: np.ndarray, adj_bin: np.ndarray) -> float:
    same, total = 0, 0
    n = labels.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if adj_bin[i, j] <= 0:
                continue
            total += 1
            if labels[i] == labels[j]:
                same += 1
    return float(same / total) if total else 0.0


def cluster_seed_for_variant(variant: str) -> int:
    return int(variant_spec(variant).get("clustering_seed", 0))


def infer_affinity_meta(variant: str, stage: dict[str, Any]) -> tuple[str, bool]:
    aff = stage.get("affinity_source")
    road = stage.get("road_distance_used")
    if aff and aff != "unknown":
        return str(aff), bool(road)
    method = str(stage.get("graph_cluster_method", variant_spec(variant).get("graph_cluster_method", ""))).lower()
    if method in {"gr19a_cap_only_spectral"}:
        return "adj_sym", False
    if method in {
        "gr19b_road_cap_spectral",
        "gr17_road_spectral",
        "gr21_road_graclus_matching_4_2_1",
    }:
        return "road_distance_affinity", True
    if method == "gr19_spectral_constrained_kmeans_cap":
        road_exists = ROAD.exists()
        return ("road_distance_affinity" if road_exists else "adj_sym_fallback"), road_exists
    if method in {"gr18_constrained_spectral_cap_dist"}:
        return "adj_sym", False
    if method in {"gr20_graclus_matching_4_2_1"}:
        return "adj_sym", False
    if method in {"current", "sparse_topk"}:
        return "adj_sym", False
    return "unknown", False


def load_variant_stages(variant: str, cluster_seed: int) -> list[dict[str, Any]]:
    spec = variant_spec(variant)
    method = spec.get("graph_cluster_method", "current")
    adj_path = str(ADJ)
    road_path = spec.get("cluster_road_distance_path", str(ROAD))
    stages: list[dict[str, Any]] = []

    if method in CAPACITY_MULTILEVEL_METHODS or spec.get("graph_resolution_capacities"):
        caps = resolve_graph_resolution_capacities(
            spec.get("graph_resolution_capacities") or [4, 2, 1]
        )
        metas, cache = load_or_build_multilevel_cluster_assignments(
            node_size=NODE_SIZE,
            capacities=caps,
            adj_mx_path=adj_path,
            seed=cluster_seed,
            dataset_name="PEMS04",
            graph_cluster_method=method,
            cluster_road_distance_path=road_path,
            cluster_sigma_d=spec.get("cluster_sigma_d", 0.5),
        )
        for st in metas:
            if st.get("resolution_tag") == "S1":
                continue
            stages.append({**st, "cache_path": str(cache), "variant": variant})
        return stages

    ratios = list(spec.get("graph_resolution_ratios") or [0.25, 0.50, 1.00])
    skip_final = bool(spec.get("graph_resolution_skip_final_identity", False))
    sizes = resolve_graph_resolution_sizes(NODE_SIZE, ratios, skip_final_identity=skip_final)
    ratio_by_m = {}
    for r, m in zip(ratios, sizes):
        ratio_by_m[m] = r
    for m in sizes:
        if m >= NODE_SIZE:
            continue
        ratio = ratio_by_m.get(m, 1.0)
        meta, cache = load_or_build_cluster_assignment(
            node_size=NODE_SIZE,
            num_clusters=m,
            adj_mx_path=adj_path,
            seed=cluster_seed,
            dataset_name="PEMS04",
            graph_cluster_method=method,
            cluster_road_distance_path=road_path,
            cluster_sigma_d=spec.get("cluster_sigma_d", 0.5),
            cluster_delta_4=spec.get("cluster_delta_4", 0.8),
            cluster_delta_2=spec.get("cluster_delta_2", 0.5),
            ratio=ratio,
        )
        tag = ratio_to_tag(ratio, m)
        stages.append(
            {
                **meta,
                "resolution_tag": tag,
                "ratio": ratio,
                "cache_path": str(cache),
                "variant": variant,
            }
        )
    return stages


def cluster_meta_row(variant: str, stage: dict[str, Any]) -> dict[str, Any]:
    val = validate_cluster_assignment(stage)
    labels = np.asarray(stage["labels"])
    sizes = np.bincount(labels.astype(np.int64))
    spec = variant_spec(variant)
    aff, road = infer_affinity_meta(variant, stage)
    return {
        "variant": variant,
        "resolution": stage.get("resolution_tag", ""),
        "graph_cluster_method": stage.get("graph_cluster_method", spec.get("graph_cluster_method")),
        "affinity_source": aff,
        "road_distance_used": road,
        "capacity_used": stage.get("max_capacity", stage.get("capacity", "")),
        "delta_used": stage.get("delta", ""),
        "sigma_d": stage.get("sigma_d", spec.get("cluster_sigma_d", "")),
        "num_clusters": int(stage["num_clusters"]),
        "max_cluster_size": val["max_cluster_size"],
        "min_cluster_size": val["min_cluster_size"],
        "mean_cluster_size": round(val["mean_cluster_size"], 3),
        "num_singleton_clusters": int((sizes == 1).sum()),
        "cluster_size_hist": ",".join(str(int(x)) for x in sizes),
        "cache_path": stage.get("cache_path", ""),
        "clustering_method": stage.get("clustering_method", ""),
        "clustering_seed": cluster_seed_for_variant(variant),
    }


def find_checkpoint(horizon: int, variant: str, seed: int) -> Path | None:
    base = CKPT_ROOT / f"h{horizon}" / variant / f"seed{seed}"
    if not base.is_dir():
        return None
    hits = sorted(base.glob("*/ChainForecasting_best_val_MAE.pt"))
    return hits[-1] if hits else None


def load_cfg_from_variant(horizon: int, variant: str, seed: int):
    cfg_path = generate_temp_config(horizon, variant, seed, WORK_DIR, CKPT_ROOT)
    spec = importlib.util.spec_from_file_location("diag_cfg", cfg_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod.CFG, cfg_path


def rescale_mae(pred: torch.Tensor, target: torch.Tensor, scaler: dict) -> float:
    from basicts.data import SCALER_REGISTRY

    pred_r = SCALER_REGISTRY.get(scaler["func"])(pred, **scaler["args"])
    tgt_r = SCALER_REGISTRY.get(scaler["func"])(target, **scaler["args"])
    return float(masked_mae(pred_r, tgt_r, null_val=0.0).item())


@torch.no_grad()
def eval_stage_mae(
    variant: str,
    horizon: int,
    seed: int,
    device: str,
    max_batches: int,
) -> dict[str, Any] | None:
    ckpt = find_checkpoint(horizon, variant, seed)
    if ckpt is None:
        return None
    cfg, _ = load_cfg_from_variant(horizon, variant, seed)
    model = ChainForecasting(**cfg.MODEL.PARAM).to(device)
    state = torch.load(ckpt, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()

    val_ds = BaseTimeSeriesForecastingRunner.build_val_dataset(cfg)
    loader = torch.utils.data.DataLoader(val_ds, batch_size=cfg.VAL.DATA.BATCH_SIZE, shuffle=False)
    scaler = load_pkl(
        f"{cfg.TRAIN.DATA.DIR}/scaler_in{cfg.DATASET_INPUT_LEN}_out{cfg.DATASET_OUTPUT_LEN}.pkl"
    )

    stage_maes: dict[str, list[float]] = {}
    residual_cluster: list[float] = []
    residual_lifted: list[float] = []
    n_batches = 0

    for future, history in loader:
        if max_batches and n_batches >= max_batches:
            break
        history = history.to(device)
        future = future.to(device)
        target = future[..., :1]
        out = model(history, return_all=True)
        diag = out.get("graph_resolution_diagnostics") or {}

        before = diag.get("temporal_input")
        if before is not None:
            stage_maes.setdefault("mae_before_graph", []).append(rescale_mae(before, target, scaler))

        node_preds = diag.get("node_stage_preds") or []
        ratios = diag.get("graph_ratios") or variant_spec(variant).get("graph_resolution_ratios", [])
        sizes = diag.get("graph_resolution_sizes") or []
        for idx, pred in enumerate(node_preds):
            if idx < len(sizes):
                m = sizes[idx]
                if m >= NODE_SIZE:
                    tag = "S1"
                elif idx < len(ratios):
                    tag = ratio_to_tag(float(ratios[idx]), m)
                else:
                    tag = f"stage{idx}"
            else:
                tag = f"stage{idx}"
            stage_maes.setdefault(f"mae_after_{tag}", []).append(rescale_mae(pred, target, scaler))

        residual_cluster.extend(diag.get("residual_energy_cluster") or [])
        residual_lifted.extend(diag.get("residual_energy_lifted") or [])
        n_batches += 1

    row = {
        "variant": variant,
        "horizon": horizon,
        "checkpoint": str(ckpt),
        "val_batches": n_batches,
    }
    for k, vals in stage_maes.items():
        row[k] = round(float(np.mean(vals)), 4)
    if residual_cluster:
        row["residual_energy_cluster_mean"] = round(float(np.mean(residual_cluster)), 6)
    if residual_lifted:
        row["residual_energy_lifted_mean"] = round(float(np.mean(residual_lifted)), 6)
    if "mae_before_graph" in row and "mae_after_S1" in row:
        row["mae_delta_graph_total"] = round(row["mae_after_S1"] - row["mae_before_graph"], 4)
    if "mae_after_S1" in row:
        row["mae_after_final"] = row["mae_after_S1"]
    return row


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    adj_bin = (load_raw_adj_numpy(ADJ, NODE_SIZE) > 0).astype(np.float64)

    # 1) per-variant cluster metadata
    meta_rows: list[dict[str, Any]] = []
    label_store: dict[str, dict[str, np.ndarray]] = {}
    for variant in args.variants:
        cluster_seed = cluster_seed_for_variant(variant)
        try:
            stages = load_variant_stages(variant, cluster_seed)
        except Exception as exc:
            print(f"[skip] {variant}: {exc}")
            continue
        label_store[variant] = {}
        for st in stages:
            tag = st.get("resolution_tag", "")
            meta_rows.append(cluster_meta_row(variant, st))
            labels = np.asarray(st["labels"])
            label_store[variant][tag] = labels
            val = validate_cluster_assignment(st)
            esr = edge_same_ratio(labels, adj_bin)
            print(
                f"{variant:40s} {tag:4s} method={meta_rows[-1]['graph_cluster_method']:35s} "
                f"aff={meta_rows[-1]['affinity_source']:22s} road={meta_rows[-1]['road_distance_used']} "
                f"M={meta_rows[-1]['num_clusters']:3d} max={val['max_cluster_size']:2d} "
                f"edge_same={esr:.3f} cache={Path(meta_rows[-1]['cache_path']).name}"
            )
    write_csv(meta_rows, out_dir / "cluster_metadata.csv")

    # 2) label similarity + edge_same
    compare_rows: list[dict[str, Any]] = []
    variants_ok = [v for v in args.variants if v in label_store]
    for res in ("S14", "S12", "S4", "S2"):
        present = [v for v in variants_ok if res in label_store[v]]
        for i, va in enumerate(present):
            for vb in present[i + 1 :]:
                la, lb = label_store[va][res], label_store[vb][res]
                if la.shape != lb.shape:
                    continue
                compare_rows.append(
                    {
                        "resolution": res,
                        "variant_a": va,
                        "variant_b": vb,
                        "ARI": round(adjusted_rand_score(la, lb), 4),
                        "NMI": round(normalized_mutual_info_score(la, lb), 4),
                        "edge_same_ratio_a": round(edge_same_ratio(la, adj_bin), 4),
                        "edge_same_ratio_b": round(edge_same_ratio(lb, adj_bin), 4),
                        "num_clusters_a": int(la.max()) + 1,
                        "num_clusters_b": int(lb.max()) + 1,
                    }
                )
    write_csv(compare_rows, out_dir / "label_similarity.csv")

    # overlap S14->S12 or S4->S2
    overlap_rows: list[dict[str, Any]] = []
    for variant in variants_ok:
        coarse, fine = None, None
        if "S14" in label_store[variant] and "S12" in label_store[variant]:
            coarse, fine = "S14", "S12"
        elif "S4" in label_store[variant] and "S2" in label_store[variant]:
            coarse, fine = "S4", "S2"
        if coarse is None:
            continue
        lc = label_store[variant][coarse]
        lf = label_store[variant][fine]
        c_max, f_max = int(lc.max()) + 1, int(lf.max()) + 1
        mat = np.zeros((c_max, f_max), dtype=np.int64)
        for i in range(lc.shape[0]):
            mat[int(lc[i]), int(lf[i])] += 1
        overlap_rows.append(
            {
                "variant": variant,
                "coarse": coarse,
                "fine": fine,
                "mean_overlap_per_coarse": round(float(mat.max(axis=1).mean()), 3),
                "max_overlap": int(mat.max()),
                "pure_coarse_blocks": int((mat.max(axis=1) == np.bincount(lc.astype(int))).sum()),
            }
        )
    write_csv(overlap_rows, out_dir / "coarse_fine_overlap.csv")

    # 3) validation stage MAE from checkpoints
    val_rows: list[dict[str, Any]] = []
    for variant in args.variants:
        try:
            row = eval_stage_mae(
                variant, args.horizon, args.seed, args.device, args.max_val_batches
            )
        except Exception as exc:
            print(f"[val] {variant}: eval failed: {exc}")
            continue
        if row:
            val_rows.append(row)
            print(f"[val] {variant}: " + ", ".join(
                f"{k}={row[k]}" for k in sorted(row) if k.startswith("mae_") or k.startswith("residual")
            ))
        else:
            print(f"[val] {variant}: no checkpoint")
    write_csv(val_rows, out_dir / "validation_stage_mae.csv")

    # markdown summary
    md = ["# Graph Cluster Method Diagnosis\n\n"]
    md.append(f"Horizon={args.horizon}, train_seed={args.seed}, clustering_seed=0 (from variant spec)\n\n")
    md.append("## Cluster metadata\n\n")
    if meta_rows:
        cols = list(meta_rows[0].keys())
        md.append("| " + " | ".join(cols) + " |\n")
        md.append("|" + "|".join(["---"] * len(cols)) + "|\n")
        for r in meta_rows:
            md.append("| " + " | ".join(str(r[c]) for c in cols) + " |\n")
    md.append("\n## Label similarity (low ARI/NMI => similar performance expected)\n\n")
    if compare_rows:
        cols = list(compare_rows[0].keys())
        md.append("| " + " | ".join(cols) + " |\n")
        md.append("|" + "|".join(["---"] * len(cols)) + "|\n")
        for r in sorted(compare_rows, key=lambda x: (x["resolution"], -x["ARI"])):
            md.append("| " + " | ".join(str(r[c]) for c in cols) + " |\n")
    md.append("\n## Validation stage MAE\n\n")
    if val_rows:
        cols = list(val_rows[0].keys())
        md.append("| " + " | ".join(cols) + " |\n")
        md.append("|" + "|".join(["---"] * len(cols)) + "|\n")
        for r in val_rows:
            md.append("| " + " | ".join(str(r[c]) for c in cols) + " |\n")
    md.append("\n## Interpretation checklist\n\n")
    md.append("- If ARI/NMI between GR7 and GR17–GR21 are high at S14/S12, C differences are weak.\n")
    md.append("- If mae_before_graph ≈ mae_after_S1 but test MAE differs little, S1 adaptive stage dominates.\n")
    md.append("- If residual_energy_lifted is tiny vs temporal error scale, pooling/lifting impact is weak.\n")
    md.append("- Compare GR19 vs GR19a/GR19b to see if GR19 silently used road distance.\n")
    (out_dir / "diagnosis.md").write_text("".join(md), encoding="utf-8")

    print(f"\nWrote {out_dir}/cluster_metadata.csv")
    print(f"Wrote {out_dir}/label_similarity.csv")
    print(f"Wrote {out_dir}/validation_stage_mae.csv")
    print(f"Wrote {out_dir}/diagnosis.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
