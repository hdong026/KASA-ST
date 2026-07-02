#!/usr/bin/env python3
"""MTSR Experiment Group A on PeMS04: temporal/spatial organization ablation."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHAIN_BASE_CFG = ROOT / "examples" / "ChainForecasting" / "ChainForecasting_PEMS04.py"
RUN_PY = ROOT / "examples" / "run.py"
TEMP_CFG_DIR = ROOT / "tmp_configs" / "mtsr_experiment_a_pems04"
LOG_DIR = ROOT / "results" / "mtsr_experiment_a_pems04_logs"
CKPT_ROOT = ROOT / "checkpoints" / "mtsr_experiment_a_pems04"

INPUT_LEN = 12
DEFAULT_HORIZONS = [12, 48]
DEFAULT_VARIANTS = [
    "mtsr_temporal_first_final_spatial",
    "mtsr_temporal_first_node_preserving_multiscale_spatial",
    "mtsr_temporal_first_graph_resolution_spatial",
]
DEFAULT_SEEDS = [1]
DEFAULT_GPUS = ["0", "1"]

HORIZON_CHAIN_LENGTHS: dict[int, list[int]] = {
    12: [3, 6, 12],
    24: [6, 12, 24],
    48: [12, 24, 48],
}
CHAIN_LOSS_WEIGHTS = [0.2, 0.3, 1.0]

IST_FSC_BASE: dict = {
    "is_chain": True,
    "chain_lengths": None,
    "chain_loss_weights": CHAIN_LOSS_WEIGHTS,
    "use_prev_condition": True,
    "propagation_mode": "forecast_state",
    "spatial_placement": "interleaved_progressive",
    "progressive_spatial_ratios": [0.25, 0.5, 1.0],
    "progressive_spatial_topks": [8, 16, 32],
    "progressive_spatial_alphas": [0.03, 0.06, 0.10],
    "post_spatial_mode": "adaptive_only",
    "use_adaptive_adj": True,
    "chain_supervision_source": "temporal_chain",
    "spatial_stage_loss_weights": [0.0, 0.0, 0.0],
}

BEST_SETTINGS = {
    "use_patch_branch": True,
    "use_downsample_branch": True,
    "use_linear_residual_branch": True,
    "use_pre_temporal_spatial_enhancement": False,
    "keep_output_prior_residual": False,
    "use_input_prior_enhancement": False,
    "use_graph_spectral_calibration": False,
    "use_extra_prior_input": False,
    "main_input_dim": 3,
    "patch_embedding_mode": "serial_concat",
    "patch_data_input_mode": "all",
}

POST_CHAIN_SPATIAL = {
    "post_chain_spatial_ratios": [0.25, 0.5, 1.0],
    "post_chain_spatial_topks": [8, 16, 32],
    "post_chain_spatial_alphas": [0.03, 0.06, 0.10],
}

GRAPH_RESOLUTION = {
    "graph_resolution_ratios": [0.25, 0.50, 1.00],
    "graph_resolution_alphas": [0.03, 0.06, 0.10],
    "graph_resolution_topks": [8, 16, 32],
    "graph_resolution_rhos": [0.25, 0.50, 1.00],
    "graph_resolution_betas": [1.0, 1.0, 1.0],
    "spatial_graph_loss_weights": [0.0, 0.0, 0.0],
    "dataset_name": "PEMS04",
    "clustering_seed": 0,
}

SPATIAL_ORG_MAP = {
    "none": "none",
    "interleaved_progressive": "interleaved",
    "final": "final_only",
    "each_level": "each_level",
    "temporal_first_multiscale": "temporal_first_multiscale",
    "temporal_first_graph_resolution": "graph_resolution",
}

VARIANT_SPECS: dict[str, dict] = {
    "chain_interleaved_progressive_spatial": {
        **IST_FSC_BASE,
        "spatial_placement": "interleaved_progressive",
        "chain_supervision_source": "spatial_chain",
    },
    "mtsr_temporal_only": {
        **IST_FSC_BASE,
        "spatial_placement": "none",
    },
    "mtsr_temporal_first_final_spatial": {
        **IST_FSC_BASE,
        "spatial_placement": "final",
    },
    "mtsr_temporal_first_node_preserving_multiscale_spatial": {
        **IST_FSC_BASE,
        "spatial_placement": "temporal_first_multiscale",
        **POST_CHAIN_SPATIAL,
    },
    "mtsr_temporal_first_multiscale_spatial": {
        **IST_FSC_BASE,
        "spatial_placement": "temporal_first_multiscale",
        **POST_CHAIN_SPATIAL,
    },
    "mtsr_temporal_first_graph_resolution_spatial": {
        **IST_FSC_BASE,
        "spatial_placement": "temporal_first_graph_resolution",
        **GRAPH_RESOLUTION,
    },
}

METRIC_PATTERNS = {
    "mae": re.compile(r"test_MAE:\s*([0-9.eE+-]+)", re.I),
    "rmse": re.compile(r"test_RMSE:\s*([0-9.eE+-]+)", re.I),
    "mape": re.compile(r"test_MAPE:\s*([0-9.eE+-]+)", re.I),
}
RESULT_TEST_BLOCK = re.compile(
    r"Result\s*(?:<[^>]+>)?\s*:\s*\[(.*?)\]",
    re.I | re.S,
)
T_CRIT = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}


def load_cfg(cfg_path: Path):
    spec = importlib.util.spec_from_file_location("mtsr_exp_a_cfg", cfg_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.CFG


def chain_lengths_for(horizon: int) -> list[int]:
    if horizon not in HORIZON_CHAIN_LENGTHS:
        raise ValueError(f"Unsupported horizon: {horizon}")
    return HORIZON_CHAIN_LENGTHS[horizon]


def resolve_chain_loss_weights(spec: dict, num_stages: int) -> list[float]:
    weights = spec.get("chain_loss_weights", CHAIN_LOSS_WEIGHTS)
    if weights == "final_only":
        return [0.0] * max(num_stages - 1, 0) + [1.0]
    return list(weights)


def variant_spec(variant: str, horizon: int) -> dict:
    if variant not in VARIANT_SPECS:
        raise ValueError(f"Unknown variant: {variant}")
    spec = {**BEST_SETTINGS, **VARIANT_SPECS[variant]}
    spec = dict(spec)
    if spec.get("is_chain", True):
        spec["chain_lengths"] = chain_lengths_for(horizon)
    spec["chain_loss_weights"] = resolve_chain_loss_weights(spec, len(spec["chain_lengths"]))
    return spec


def spatial_budget_str(spec: dict) -> str:
    placement = spec.get("spatial_placement", "")
    if placement == "interleaved_progressive":
        return (
            f"ratios={spec.get('progressive_spatial_ratios')};"
            f"topks={spec.get('progressive_spatial_topks')};"
            f"alphas={spec.get('progressive_spatial_alphas')}"
        )
    if placement == "temporal_first_multiscale":
        return (
            f"ratios={spec.get('post_chain_spatial_ratios')};"
            f"topks={spec.get('post_chain_spatial_topks')};"
            f"alphas={spec.get('post_chain_spatial_alphas')}"
        )
    if placement == "temporal_first_graph_resolution":
        return (
            f"ratios={spec.get('graph_resolution_ratios')};"
            f"topks={spec.get('graph_resolution_topks')};"
            f"alphas={spec.get('graph_resolution_alphas')};"
            f"rhos={spec.get('graph_resolution_rhos')}"
        )
    if placement == "final":
        return "single_final_spatial"
    return "none"


def graph_resolution_log_fields(cfg_path: Path) -> dict:
    try:
        from basicts.archs.arch_zoo.ChainForecasting_arch.graph_cluster_utils import (
            resolve_graph_resolution_sizes,
        )

        cfg = load_cfg(cfg_path)
        param = cfg.MODEL.PARAM
        if param.get("spatial_placement") != "temporal_first_graph_resolution":
            return {}
        node_size = int(param["node_size"])
        ratios = list(param.get("graph_resolution_ratios", [0.25, 0.50, 1.00]))
        sizes = resolve_graph_resolution_sizes(node_size, ratios)
        model = cfg.MODEL.ARCH(**param)
        meta = {}
        if hasattr(model, "graph_resolution_stack") and model.graph_resolution_stack is not None:
            meta = model.graph_resolution_stack.metadata()
        return {
            "graph_resolution_ratios": str(ratios),
            "graph_resolution_sizes": str(sizes),
            "clustering_methods": str(meta.get("clustering_methods", [])),
            "cluster_cache_paths": str(meta.get("cluster_cache_paths", [])),
            "spatial_operator_type": meta.get("spatial_operator_type", param.get("post_spatial_mode")),
        }
    except Exception:
        return {}


def ckpt_dir_for(horizon: int, variant: str, seed: int) -> Path:
    return CKPT_ROOT / f"h{horizon}" / f"{variant}_seed{seed}"


def temp_cfg_path(horizon: int, variant: str, seed: int) -> Path:
    TEMP_CFG_DIR.mkdir(parents=True, exist_ok=True)
    return TEMP_CFG_DIR / f"h{horizon}_{variant}_seed{seed}.py"


def strip_hardcoded_cuda_devices(content: str) -> str:
    kept: list[str] = []
    for line in content.splitlines():
        if "CUDA_VISIBLE_DEVICES" in line and "os.environ" in line:
            continue
        kept.append(line)
    return "\n".join(kept) + "\n"


def _py_literal(v) -> str:
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, str):
        return repr(v)
    if isinstance(v, list):
        return repr(v)
    return str(v)


def generate_temp_config(horizon: int, variant: str, seed: int) -> Path:
    spec = variant_spec(variant, horizon)
    content = strip_hardcoded_cuda_devices(CHAIN_BASE_CFG.read_text(encoding="utf-8"))
    ckpt_rel = os.path.join("checkpoints", "mtsr_experiment_a_pems04", f"h{horizon}", f"{variant}_seed{seed}")
    lines = [
        "",
        "# ===== mtsr_experiment_a_pems04 overrides (auto-generated) =====",
        f"CFG.ENV.SEED = {seed}",
        "if hasattr(CFG, 'SEED'):",
        f"    CFG.SEED = {seed}",
        f"CFG.DATASET_INPUT_LEN = {INPUT_LEN}",
        f"CFG.DATASET_OUTPUT_LEN = {horizon}",
        f'CFG.TRAIN.CKPT_SAVE_DIR = os.path.join("{ckpt_rel}")',
        "CFG.MODEL.FORWARD_FEATURES = [0, 1, 2, 3]",
        "CFG.MODEL.TARGET_FEATURES = [0]",
        f'CFG.MODEL.PARAM["input_len"] = {INPUT_LEN}',
        f'CFG.MODEL.PARAM["output_len"] = {horizon}',
        f"CFG.TEST.EVALUATION_HORIZONS = list(range(1, {horizon + 1}))",
    ]
    skip = {"is_chain", "chain_loss_weights"}
    for key, val in spec.items():
        if key in skip or val is None:
            continue
        lines.append(f'CFG.MODEL.PARAM["{key}"] = {_py_literal(val)}')
    lines.append(f'CFG.MODEL.PARAM["chain_loss_weights"] = {_py_literal(spec["chain_loss_weights"])}')
    out = temp_cfg_path(horizon, variant, seed)
    out.write_text(content + "\n".join(lines) + "\n", encoding="utf-8")
    return out


def cfg_for_easytorch(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def count_params(cfg_path: Path) -> int | None:
    try:
        cfg = load_cfg(cfg_path)
        model = cfg.MODEL.ARCH(**cfg.MODEL.PARAM)
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    except Exception:
        return None


def _parse_block(block: str) -> dict[str, float | None]:
    out: dict[str, float | None] = {k: None for k in METRIC_PATTERNS}
    for k, pat in METRIC_PATTERNS.items():
        m = pat.search(block)
        if m:
            out[k] = float(m.group(1))
    return out


def parse_metrics(log_text: str) -> dict[str, float | None]:
    best: dict[str, float | None] | None = None
    best_mae: float | None = None
    for m in RESULT_TEST_BLOCK.finditer(log_text):
        parsed = _parse_block(m.group(1))
        if parsed["mae"] is None:
            continue
        if best_mae is None or parsed["mae"] < best_mae:
            best_mae = parsed["mae"]
            best = parsed
    if best is not None:
        return best
    fallback = _parse_block(log_text)
    if fallback["mae"] is not None:
        return fallback
    return {k: None for k in METRIC_PATTERNS}


def collect_log_text(horizon: int, variant: str, seed: int, wrapper_log: Path | None) -> str:
    parts: list[str] = []
    if wrapper_log and wrapper_log.is_file():
        parts.append(wrapper_log.read_text(errors="replace"))
    ckpt_base = ckpt_dir_for(horizon, variant, seed)
    if ckpt_base.is_dir():
        for tlog in sorted(ckpt_base.glob("*/training_log_*.log"), key=lambda p: p.stat().st_mtime, reverse=True):
            parts.append(tlog.read_text(errors="replace"))
    return "\n".join(parts)


def wrapper_log_path(horizon: int, variant: str, seed: int) -> Path | None:
    matches = sorted(LOG_DIR.glob(f"h{horizon}_{variant}_seed{seed}_gpu*.log"))
    return matches[-1] if matches else None


class GPUQueue:
    def __init__(self, gpus: list[str]):
        self.gpus = gpus
        self.lock = threading.Lock()
        self.available = list(gpus)

    def acquire(self) -> str:
        while True:
            with self.lock:
                if self.available:
                    return self.available.pop(0)
            time.sleep(2)

    def release(self, gpu: str):
        with self.lock:
            self.available.append(gpu)


def base_row(horizon: int, variant: str, seed: int, cfg_path: Path) -> dict:
    spec = variant_spec(variant, horizon)
    placement = spec.get("spatial_placement", "")
    return {
        "horizon": horizon,
        "variant": variant,
        "seed": seed,
        "input_len": INPUT_LEN,
        "output_len": horizon,
        "chain_lengths": str(spec["chain_lengths"]),
        "chain_loss_weights": str(spec["chain_loss_weights"]),
        "chain_supervision_source": spec.get("chain_supervision_source"),
        "spatial_placement": placement,
        "spatial_organization_type": SPATIAL_ORG_MAP.get(placement, placement),
        "spatial_budgets": spatial_budget_str(spec),
        "spatial_stage_loss_weights": str(spec.get("spatial_stage_loss_weights")),
        "spatial_graph_loss_weights": str(spec.get("spatial_graph_loss_weights", [0.0, 0.0, 0.0])),
        "dataset": spec.get("dataset_name", "PEMS04"),
        "graph_resolution_ratios": str(spec.get("graph_resolution_ratios", "")),
        "graph_resolution_sizes": "",
        "clustering_methods": "",
        "cluster_cache_paths": "",
        "spatial_operator_type": spec.get("post_spatial_mode"),
        "mae": None,
        "rmse": None,
        "mape": None,
        "params": count_params(cfg_path),
        "failed": 1,
        "status": "pending",
        "ckpt_dir": str(ckpt_dir_for(horizon, variant, seed)),
        "cfg_path": str(cfg_path),
        "log_file": "",
        "result_csv": "",
    }
    row.update(graph_resolution_log_fields(cfg_path))
    return row


def summarize_row(horizon: int, variant: str, seed: int, cfg_path: Path) -> dict:
    row = base_row(horizon, variant, seed, cfg_path)
    wrapper_log = wrapper_log_path(horizon, variant, seed)
    metrics = parse_metrics(collect_log_text(horizon, variant, seed, wrapper_log))
    row.update(metrics)
    if metrics["mae"] is not None:
        row["status"] = "ok"
        row["failed"] = 0
    else:
        row["status"] = "failed_no_metrics"
        row["failed"] = 1
    row["log_file"] = str(wrapper_log) if wrapper_log else ""
    return row


def is_completed(horizon: int, variant: str, seed: int, cfg_path: Path) -> bool:
    row = summarize_row(horizon, variant, seed, cfg_path)
    return row.get("status") == "ok" and row.get("mae") is not None


def run_one(horizon: int, variant: str, cfg_path: Path, gpu: str, seed: int, result_csv: Path) -> dict:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"h{horizon}_{variant}_seed{seed}_gpu{gpu}.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    cmd = [sys.executable, str(RUN_PY), "--cfg", cfg_for_easytorch(cfg_path), "--gpus", "0"]
    row = base_row(horizon, variant, seed, cfg_path)
    row["log_file"] = str(log_file)
    row["result_csv"] = str(result_csv)
    row["status"] = "running"
    try:
        with open(log_file, "w", encoding="utf-8") as lf:
            proc = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=lf, stderr=subprocess.STDOUT)
        metrics = parse_metrics(collect_log_text(horizon, variant, seed, log_file))
        row.update(metrics)
        if proc.returncode != 0:
            row["status"] = f"exit_{proc.returncode}"
            row["failed"] = 1
        elif metrics["mae"] is None:
            row["status"] = "failed_no_metrics"
            row["failed"] = 1
        else:
            row["status"] = "ok"
            row["failed"] = 0
    except Exception as e:
        row["status"] = f"error:{e}"
        row["failed"] = 1
    return row


def mean_std_ci(values: list[float]) -> tuple[float, float, float]:
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    mean = sum(values) / n
    if n == 1:
        return mean, 0.0, float("nan")
    var = sum((x - mean) ** 2 for x in values) / (n - 1)
    std = math.sqrt(var)
    t_val = T_CRIT.get(n, 1.96)
    return mean, std, t_val * std / math.sqrt(n)


def write_outputs(rows: list[dict], out_csv: Path, out_md: Path) -> None:
    rows = sorted(rows, key=lambda r: (int(r["horizon"]), r["variant"], int(r["seed"])))
    fields = [
        "horizon", "variant", "seed", "input_len", "output_len", "chain_lengths",
        "chain_loss_weights", "chain_supervision_source", "spatial_placement",
        "spatial_organization_type", "spatial_budgets", "spatial_stage_loss_weights",
        "spatial_graph_loss_weights", "dataset", "graph_resolution_ratios",
        "graph_resolution_sizes", "clustering_methods", "cluster_cache_paths",
        "spatial_operator_type", "mae", "rmse", "mape", "params", "failed",
        "status", "ckpt_dir", "cfg_path", "log_file", "result_csv",
    ]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    summary_csv = out_csv.with_name(out_csv.stem + "_summary.csv")
    groups: dict[tuple[int, str], list[dict]] = {}
    for r in rows:
        groups.setdefault((int(r["horizon"]), r["variant"]), []).append(r)
    summary = []
    for (horizon, variant), runs in sorted(groups.items()):
        ok = [r for r in runs if r.get("failed", 1) == 0 and r.get("mae") is not None]
        for metric in ("mae", "rmse", "mape"):
            vals = [float(r[metric]) for r in ok if r.get(metric) is not None]
            m, s, c = mean_std_ci(vals)
            summary.append({
                "horizon": horizon, "variant": variant, "metric": metric,
                "n": len(vals), "mean": m, "std": s, "ci95": c,
                "failed": len(runs) - len(ok),
            })
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["horizon", "variant", "metric", "n", "mean", "std", "ci95", "failed"])
        w.writeheader()
        w.writerows(summary)

    md = ["# MTSR Experiment Group A on PeMS04\n\n"]
    md.append("| horizon | variant | seed | chain | spatial_org | MAE | RMSE | MAPE | params | failed |\n")
    md.append("|---:|---|---:|---|---|---:|---:|---:|---:|---:|\n")
    for r in rows:
        md.append(
            f"| {r['horizon']} | {r['variant']} | {r['seed']} | {r['chain_lengths']} | "
            f"{r.get('spatial_organization_type','')} | {r.get('mae','')} | {r.get('rmse','')} | "
            f"{r.get('mape','')} | {r.get('params','')} | {r.get('failed',1)} |\n"
        )
    out_md.write_text("".join(md), encoding="utf-8")


def dry_run_info(horizon: int, variant: str, seed: int, cfg_path: Path) -> None:
    spec = variant_spec(variant, horizon)
    print(f"  [h={horizon} {variant} seed={seed}]")
    print(f"    chain_lengths: {spec['chain_lengths']}")
    print(f"    chain_loss_weights: {spec['chain_loss_weights']}")
    print(f"    chain_supervision_source: {spec.get('chain_supervision_source')}")
    print(f"    spatial_placement: {spec.get('spatial_placement')}")
    print(f"    spatial_organization_type: {SPATIAL_ORG_MAP.get(spec.get('spatial_placement',''))}")
    print(f"    spatial_budgets: {spatial_budget_str(spec)}")
    print(f"    params: {count_params(cfg_path)}")
    print(f"    ckpt: {ckpt_dir_for(horizon, variant, seed)}")
    print(f"    cfg: {cfg_path}")
    cfg = load_cfg(cfg_path)
    assert cfg.MODEL.PARAM["output_len"] == horizon


def main() -> int:
    parser = argparse.ArgumentParser(description="MTSR Experiment Group A on PeMS04.")
    parser.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS, choices=list(VARIANT_SPECS.keys()))
    parser.add_argument("--horizons", type=int, nargs="+", default=DEFAULT_HORIZONS)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--gpus", nargs="+", default=DEFAULT_GPUS)
    parser.add_argument("--out", default="results/pems04_mtsr_experiment_a_v1.csv")
    parser.add_argument("--markdown", default="results/pems04_mtsr_experiment_a_v1.md")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--skip_completed", action="store_true", default=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out_csv = ROOT / args.out
    out_md = ROOT / args.markdown
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    jobs = [(h, v, generate_temp_config(h, v, s), s) for h in args.horizons for v in args.variants for s in args.seeds]

    if args.dry_run:
        print("MTSR Experiment A dry run:\n")
        for h, v, p, s in jobs:
            dry_run_info(h, v, s, p)
        return 0

    queue = GPUQueue(args.gpus)
    rows: list[dict] = []
    lock = threading.Lock()
    skip = args.skip_completed and not args.force

    def worker(h, v, p, s):
        if skip and is_completed(h, v, s, p):
            row = summarize_row(h, v, s, p)
            row["result_csv"] = str(out_csv)
            with lock:
                rows.append(row)
            print(f"[skip] h={h} {v} seed={s} mae={row['mae']}")
            return
        gpu = queue.acquire()
        try:
            print(f"[start] h={h} {v} seed={s} gpu={gpu}")
            row = run_one(h, v, p, gpu, s, out_csv)
            with lock:
                rows.append(row)
            print(f"[done] h={h} {v} seed={s} status={row['status']} mae={row['mae']}")
        finally:
            queue.release(gpu)

    threads = [threading.Thread(target=worker, args=j) for j in jobs]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    write_outputs(rows, out_csv, out_md)
    print(f"Wrote {out_csv}, {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
