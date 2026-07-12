#!/usr/bin/env python3
"""Graph module / Graph Resolution ablation on PeMS04 12→12."""
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
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_BASE_CFG = ROOT / "examples" / "ChainForecasting" / "ChainForecasting_PEMS04.py"
RUN_PY = ROOT / "examples" / "run.py"
DEFAULT_WORK_DIR = ROOT / "experiments" / "graph_resolution_pems04"
DEFAULT_LOG_ROOT = ROOT / "logs" / "graph_resolution_pems04"
DEFAULT_CKPT_ROOT = ROOT / "checkpoints" / "graph_resolution_pems04"

DEFAULT_VARIANTS = [
    "G0_no_spatial",
    "G1_final_adaptive",
    "G2_each_level_adaptive",
    "GR0_default",
    "GR1_two_level",
    "GR2_coarse_full",
    "GR3_small_alpha",
    "GR4_large_alpha",
    "GR5_dense_topk",
]
DEFAULT_SEEDS = [1, 2, 3, 4, 5]
DEFAULT_GPUS = ["0", "1"]

GRAPH_RES_KEYS = (
    "graph_resolution_ratios",
    "graph_resolution_alphas",
    "graph_resolution_topks",
    "graph_resolution_betas",
    "graph_resolution_rhos",
)

GR7_SPARSE_BASE: dict[str, Any] = {
    "spatial_placement": "temporal_first_graph_resolution",
    "post_spatial_mode": "adaptive_only",
    "graph_resolution_ratios": [0.25, 0.50, 1.00],
    "graph_resolution_alphas": [0.03, 0.06, 0.10],
    "graph_resolution_topks": [4, 8, 16],
    "graph_resolution_betas": [1.0, 1.0, 1.0],
    "graph_resolution_rhos": [0.25, 0.50, 1.00],
    "clustering_seed": 0,
    "dataset_name": "PEMS04",
    "graph_cluster_method": "current",
}


def graph_stage_loss_weights(num_stages: int) -> list[float]:
    """Per-stage node MAE weights for graph resolution intermediate outputs.

    The runner skips the final stage (already covered by chain_loss_weights[-1]).
    Coarse-to-fine weights increase linearly: 0.10, 0.20, 0.30, ...
    """
    if num_stages <= 1:
        return [0.15]
    return [round(0.10 * (i + 1), 2) for i in range(num_stages - 1)] + [0.0]

PEMS04_SPATIAL_DISTANCE = os.path.join(
    "datasets", "raw_data", "PEMS04", "adj_PEMS04_distance.pkl"
)

COMMON_FIXED: dict[str, Any] = {
    "chain_lengths": [3, 6, 12],
    "chain_loss_weights": [0.2, 0.3, 1.0],
    "use_prev_condition": True,
    "use_patch_branch": True,
    "use_downsample_branch": True,
    "use_linear_residual_branch": True,
    "patch_embedding_mode": "serial_concat",
    "patch_data_input_mode": "all",
    "use_pre_temporal_spatial_enhancement": False,
    "keep_output_prior_residual": False,
    "use_input_prior_enhancement": False,
    "use_graph_spectral_calibration": False,
    "use_extra_prior_input": False,
    "main_input_dim": 3,
}

VARIANT_SPECS: dict[str, dict[str, Any]] = {
    "G0_no_spatial": {
        "spatial_placement": "none",
        "post_spatial_mode": "none",
    },
    "G1_final_adaptive": {
        "spatial_placement": "final",
        "post_spatial_mode": "adaptive_only",
    },
    "G2_each_level_adaptive": {
        "spatial_placement": "each_level",
        "post_spatial_mode": "adaptive_only",
    },
    "GR0_default": {
        "spatial_placement": "temporal_first_graph_resolution",
        "post_spatial_mode": "adaptive_only",
        "graph_resolution_ratios": [0.25, 0.50, 1.00],
        "graph_resolution_alphas": [0.03, 0.06, 0.10],
        "graph_resolution_topks": [8, 16, 32],
        "graph_resolution_betas": [1.0, 1.0, 1.0],
        "graph_resolution_rhos": [0.25, 0.50, 1.00],
        "clustering_seed": 0,
    },
    "GR1_two_level": {
        "spatial_placement": "temporal_first_graph_resolution",
        "post_spatial_mode": "adaptive_only",
        "graph_resolution_ratios": [0.50, 1.00],
        "graph_resolution_alphas": [0.06, 0.10],
        "graph_resolution_topks": [16, 32],
        "graph_resolution_betas": [1.0, 1.0],
        "graph_resolution_rhos": [0.50, 1.00],
        "clustering_seed": 0,
    },
    "GR2_coarse_full": {
        "spatial_placement": "temporal_first_graph_resolution",
        "post_spatial_mode": "adaptive_only",
        "graph_resolution_ratios": [0.25, 1.00],
        "graph_resolution_alphas": [0.04, 0.10],
        "graph_resolution_topks": [8, 32],
        "graph_resolution_betas": [1.0, 1.0],
        "graph_resolution_rhos": [0.25, 1.00],
        "clustering_seed": 0,
    },
    "GR3_small_alpha": {
        "spatial_placement": "temporal_first_graph_resolution",
        "post_spatial_mode": "adaptive_only",
        "graph_resolution_ratios": [0.25, 0.50, 1.00],
        "graph_resolution_alphas": [0.01, 0.03, 0.05],
        "graph_resolution_topks": [8, 16, 32],
        "graph_resolution_betas": [1.0, 1.0, 1.0],
        "graph_resolution_rhos": [0.25, 0.50, 1.00],
        "clustering_seed": 0,
    },
    "GR4_large_alpha": {
        "spatial_placement": "temporal_first_graph_resolution",
        "post_spatial_mode": "adaptive_only",
        "graph_resolution_ratios": [0.25, 0.50, 1.00],
        "graph_resolution_alphas": [0.05, 0.10, 0.15],
        "graph_resolution_topks": [8, 16, 32],
        "graph_resolution_betas": [1.0, 1.0, 1.0],
        "graph_resolution_rhos": [0.25, 0.50, 1.00],
        "clustering_seed": 0,
    },
    "GR5_dense_topk": {
        "spatial_placement": "temporal_first_graph_resolution",
        "post_spatial_mode": "adaptive_only",
        "graph_resolution_ratios": [0.25, 0.50, 1.00],
        "graph_resolution_alphas": [0.03, 0.06, 0.10],
        "graph_resolution_topks": [16, 32, 64],
        "graph_resolution_betas": [1.0, 1.0, 1.0],
        "graph_resolution_rhos": [0.25, 0.50, 1.00],
        "clustering_seed": 0,
    },
    "GR7_sparse_topk": {
        **GR7_SPARSE_BASE,
    },
    "GR8_ultra_sparse_topk": {
        **GR7_SPARSE_BASE,
        "graph_resolution_topks": [2, 4, 8],
    },
    "GR14_two_level_sparse": {
        **GR7_SPARSE_BASE,
        "graph_resolution_ratios": [0.50, 1.00],
        "graph_resolution_alphas": [0.06, 0.10],
        "graph_resolution_topks": [8, 16],
        "graph_resolution_betas": [1.0, 1.0],
        "graph_resolution_rhos": [0.50, 1.00],
    },
    "GR15_four_level_sparse": {
        **GR7_SPARSE_BASE,
        "graph_resolution_ratios": [0.125, 0.25, 0.50, 1.00],
        "graph_resolution_alphas": [0.02, 0.04, 0.06, 0.10],
        "graph_resolution_topks": [2, 4, 8, 16],
        "graph_resolution_betas": [1.0, 1.0, 1.0, 1.0],
        "graph_resolution_rhos": [0.125, 0.25, 0.50, 1.00],
    },
    "GR16_one_level_sparse": {
        **GR7_SPARSE_BASE,
        "graph_resolution_ratios": [1.00],
        "graph_resolution_alphas": [0.10],
        "graph_resolution_topks": [16],
        "graph_resolution_betas": [1.0],
        "graph_resolution_rhos": [1.00],
    },
    "GR7_sparse_topk_stageloss": {
        **GR7_SPARSE_BASE,
        "spatial_graph_loss_weights": graph_stage_loss_weights(3),
    },
    "GR14_two_level_sparse_stageloss": {
        **GR7_SPARSE_BASE,
        "graph_resolution_ratios": [0.50, 1.00],
        "graph_resolution_alphas": [0.06, 0.10],
        "graph_resolution_topks": [8, 16],
        "graph_resolution_betas": [1.0, 1.0],
        "graph_resolution_rhos": [0.50, 1.00],
        "spatial_graph_loss_weights": graph_stage_loss_weights(2),
    },
    "GR15_four_level_sparse_stageloss": {
        **GR7_SPARSE_BASE,
        "graph_resolution_ratios": [0.125, 0.25, 0.50, 1.00],
        "graph_resolution_alphas": [0.02, 0.04, 0.06, 0.10],
        "graph_resolution_topks": [2, 4, 8, 16],
        "graph_resolution_betas": [1.0, 1.0, 1.0, 1.0],
        "graph_resolution_rhos": [0.125, 0.25, 0.50, 1.00],
        "spatial_graph_loss_weights": graph_stage_loss_weights(4),
    },
    "GR16_one_level_sparse_stageloss": {
        **GR7_SPARSE_BASE,
        "graph_resolution_ratios": [1.00],
        "graph_resolution_alphas": [0.10],
        "graph_resolution_topks": [16],
        "graph_resolution_betas": [1.0],
        "graph_resolution_rhos": [1.00],
        "spatial_graph_loss_weights": graph_stage_loss_weights(1),
    },
    "GR9_pearson_balanced_pam": {
        **GR7_SPARSE_BASE,
        "graph_cluster_method": "pearson_balanced_pam",
    },
    "GR10_xcorr_balanced_pam": {
        **GR7_SPARSE_BASE,
        "graph_cluster_method": "xcorr_balanced_pam",
        "cluster_max_lag": 12,
    },
    "GR11_joint_pearson_spatial_pam": {
        **GR7_SPARSE_BASE,
        "graph_cluster_method": "joint_pearson_spatial_balanced_pam",
        "cluster_lambda_s": 0.2,
        "cluster_spatial_coord_path": PEMS04_SPATIAL_DISTANCE,
    },
    "GR12_pearson_standard_pam": {
        **GR7_SPARSE_BASE,
        "graph_cluster_method": "pearson_standard_pam",
    },
    "GR13_autocorr_feature_pam": {
        **GR7_SPARSE_BASE,
        "graph_cluster_method": "autocorr_feature_balanced_pam",
        "cluster_acf_lag": 24,
    },
    "G3_final_adaptive_ms": {
        "spatial_placement": "final",
        "post_spatial_mode": "adaptive_multiscale_only",
        "adaptive_ms_topks": [8, 16, 32],
        "adaptive_ms_alpha": 0.10,
        "adaptive_ms_fusion": "softmax",
        "adaptive_ms_share_logits": True,
        "adaptive_ms_init": "favor_largest",
    },
    "G4_final_adaptive_ms_uniform": {
        "spatial_placement": "final",
        "post_spatial_mode": "adaptive_multiscale_only",
        "adaptive_ms_topks": [8, 16, 32],
        "adaptive_ms_alpha": 0.10,
        "adaptive_ms_fusion": "softmax",
        "adaptive_ms_share_logits": True,
        "adaptive_ms_init": "uniform",
    },
    "G5_final_adaptive_ms_small": {
        "spatial_placement": "final",
        "post_spatial_mode": "adaptive_multiscale_only",
        "adaptive_ms_topks": [4, 8, 16],
        "adaptive_ms_alpha": 0.08,
        "adaptive_ms_fusion": "softmax",
        "adaptive_ms_share_logits": True,
        "adaptive_ms_init": "uniform",
    },
    "GR6_default_adaptive_ms": {
        "spatial_placement": "temporal_first_graph_resolution",
        "post_spatial_mode": "adaptive_multiscale_only",
        "graph_resolution_ratios": [0.25, 0.50, 1.00],
        "graph_resolution_alphas": [0.03, 0.06, 0.10],
        "graph_resolution_topks": [8, 16, 32],
        "graph_resolution_betas": [1.0, 1.0, 1.0],
        "graph_resolution_rhos": [0.25, 0.50, 1.00],
        "clustering_seed": 0,
        "adaptive_ms_topks": [8, 16, 32],
        "adaptive_ms_alpha": 0.10,
        "adaptive_ms_fusion": "softmax",
        "adaptive_ms_share_logits": True,
        "adaptive_ms_init": "favor_largest",
    },
}

EPOCH_LINE = re.compile(r"Epoch\s+(\d+)\s*/", re.I)
VAL_LINE = re.compile(r"Result\s*<val>.*?val_MAE:\s*([0-9.eE+-]+)", re.I | re.S)
TRAIN_LINE = re.compile(r"Result\s*<train>.*?train_MAE:\s*([0-9.eE+-]+)", re.I | re.S)
TEST_BLOCK = re.compile(
    r"Result\s*<test>.*?test_MAE:\s*([0-9.eE+-]+).*?"
    r"test_RMSE:\s*([0-9.eE+-]+).*?"
    r"test_MAPE:\s*([0-9.eE+-]+)",
    re.I | re.S,
)
BEST_CKPT = re.compile(r"best_val_MAE\.pt saved", re.I)
TRAIN_TIME = re.compile(r"train_time:\s*([0-9.]+)", re.I)
VAL_TIME = re.compile(r"val_time:\s*([0-9.]+)", re.I)
TEST_TIME = re.compile(r"test_time:\s*([0-9.]+)", re.I)
T_CRIT = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}
ERROR_TAIL_LINES = 120
SHOW_ERROR_LINES = 80


def variant_spec(variant: str) -> dict[str, Any]:
    if variant not in VARIANT_SPECS:
        raise ValueError(f"Unknown variant: {variant}")
    return {**COMMON_FIXED, **VARIANT_SPECS[variant]}


def cfg_dir(work_dir: Path) -> Path:
    return work_dir / "configs"


def ckpt_dir_for(variant: str, seed: int, ckpt_root: Path) -> Path:
    return ckpt_root / variant / f"seed{seed}"


def log_dir_for(variant: str, seed: int, log_root: Path) -> Path:
    return log_root / variant / f"seed{seed}"


def temp_cfg_path(variant: str, seed: int, work_dir: Path) -> Path:
    out = cfg_dir(work_dir) / f"{variant}_seed{seed}.py"
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def strip_hardcoded_cuda_devices(content: str) -> str:
    kept: list[str] = []
    for line in content.splitlines():
        if "CUDA_VISIBLE_DEVICES" in line and "os.environ" in line:
            continue
        kept.append(line)
    return "\n".join(kept) + "\n"


def _py_literal(v: Any) -> str:
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, str):
        return repr(v)
    if isinstance(v, list):
        return repr(v)
    return str(v)


def generate_temp_config(
    variant: str,
    seed: int,
    base_cfg: Path,
    work_dir: Path,
    ckpt_root: Path,
) -> Path:
    spec = variant_spec(variant)
    content = strip_hardcoded_cuda_devices(base_cfg.read_text(encoding="utf-8"))
    ckpt_rel = os.path.join("checkpoints", "graph_resolution_pems04", variant, f"seed{seed}")
    lines = [
        "",
        "# ===== graph_resolution_pems04 overrides (auto-generated) =====",
        f"CFG.ENV.SEED = {seed}",
        "if hasattr(CFG, 'SEED'):",
        f"    CFG.SEED = {seed}",
        "if hasattr(CFG, 'TRAIN') and hasattr(CFG.TRAIN, 'SEED'):",
        f"    CFG.TRAIN.SEED = {seed}",
        f'CFG.TRAIN.CKPT_SAVE_DIR = os.path.join("{ckpt_rel}")',
        "CFG.MODEL.FORWARD_FEATURES = [0, 1, 2, 3]",
        "CFG.MODEL.TARGET_FEATURES = [0]",
        f'CFG.DESCRIPTION = "Graph Resolution ablation: {variant} seed{seed}"',
    ]
    for key, val in spec.items():
        if val is None:
            continue
        lines.append(f'CFG.MODEL.PARAM["{key}"] = {_py_literal(val)}')
    out = temp_cfg_path(variant, seed, work_dir)
    out.write_text(content + "\n".join(lines) + "\n", encoding="utf-8")
    return out


def load_cfg(cfg_path: Path):
    spec = importlib.util.spec_from_file_location("graph_res_cfg", cfg_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.CFG


def validate_generated_config(cfg_path: Path, variant: str) -> None:
    spec = variant_spec(variant)
    cfg = load_cfg(cfg_path)
    param = cfg.MODEL.PARAM
    placement = str(param.get("spatial_placement", "")).lower()
    post_mode = str(param.get("post_spatial_mode", "")).lower()
    if placement != str(spec["spatial_placement"]).lower():
        raise ValueError(
            f"{cfg_path}: spatial_placement expected {spec['spatial_placement']}, got {placement}"
        )
    if post_mode != str(spec["post_spatial_mode"]).lower():
        raise ValueError(
            f"{cfg_path}: post_spatial_mode expected {spec['post_spatial_mode']}, got {post_mode}"
        )
    if placement == "temporal_first_graph_resolution":
        lengths = []
        for key in GRAPH_RES_KEYS:
            if key not in param:
                raise ValueError(f"{cfg_path}: missing {key} for graph resolution variant")
            val = list(param[key])
            lengths.append(len(val))
        if len(set(lengths)) != 1:
            raise ValueError(f"{cfg_path}: graph_resolution_* lengths mismatch: {dict(zip(GRAPH_RES_KEYS, lengths))}")
        if "clustering_seed" not in param:
            raise ValueError(f"{cfg_path}: missing clustering_seed")
    if post_mode == "adaptive_multiscale_only":
        for key in (
            "adaptive_ms_topks",
            "adaptive_ms_alpha",
            "adaptive_ms_fusion",
            "adaptive_ms_share_logits",
            "adaptive_ms_init",
        ):
            if key not in param:
                raise ValueError(f"{cfg_path}: missing {key} for adaptive_multiscale_only")
        if len(param["adaptive_ms_topks"]) == 0:
            raise ValueError(f"{cfg_path}: adaptive_ms_topks must be non-empty")
    if int(cfg.ENV.SEED) != int(cfg_path.stem.split("seed")[-1]):
        raise ValueError(f"{cfg_path}: seed mismatch in filename vs CFG.ENV.SEED")


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


def pick_canonical_training_log(ckpt_base: Path) -> Path | None:
    """Pick the largest training_log in the newest run subdir (complete run)."""
    if not ckpt_base.is_dir():
        return None
    run_dirs = sorted(
        [d for d in ckpt_base.iterdir() if d.is_dir()],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    candidates: list[Path] = []
    for run_dir in run_dirs:
        candidates.extend(run_dir.glob("training_log_*.log"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_size)


def collect_log_text(variant: str, seed: int, log_root: Path, ckpt_root: Path) -> str:
    parts: list[str] = []
    log_dir = log_dir_for(variant, seed, log_root)
    wrapper = log_dir / "train.log"
    if wrapper.is_file():
        parts.append(wrapper.read_text(errors="replace"))
    ckpt_base = ckpt_dir_for(variant, seed, ckpt_root)
    tlog = pick_canonical_training_log(ckpt_base)
    if tlog is not None:
        parts.append(tlog.read_text(errors="replace"))
    return "\n".join(parts)


def collect_log_sources(variant: str, seed: int, log_root: Path, ckpt_root: Path) -> dict[str, str]:
    log_dir = log_dir_for(variant, seed, log_root)
    ckpt_base = ckpt_dir_for(variant, seed, ckpt_root)
    tlog = pick_canonical_training_log(ckpt_base)
    return {
        "parsed_wrapper_log": str(log_dir / "train.log") if (log_dir / "train.log").is_file() else "",
        "parsed_training_log": str(tlog) if tlog is not None else "",
        "parsed_checkpoint_dir": str(ckpt_base),
    }


def parse_training_log(log_text: str, source: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {
        "best_val_mae": None,
        "test_mae_at_best_val": None,
        "test_rmse_at_best_val": None,
        "test_mape_at_best_val": None,
        "final_test_mae": None,
        "final_test_rmse": None,
        "final_test_mape": None,
        "best_epoch": None,
        "train_mae_at_best_val": None,
        "epoch_time": None,
        "total_time": None,
        "parsed_val_source": source,
        "parsed_test_source": source,
    }
    if not log_text.strip():
        return out

    current_epoch: int | None = None
    last_train_mae: float | None = None
    last_val_mae: float | None = None
    last_train_time: float | None = None
    last_val_time: float | None = None
    total_time = 0.0

    best_val_so_far = float("inf")

    lines = log_text.splitlines()
    for i, line in enumerate(lines):
        m = EPOCH_LINE.search(line)
        if m:
            current_epoch = int(m.group(1))

        m = TRAIN_LINE.search(line)
        if m:
            last_train_mae = float(m.group(1))
            tm = TRAIN_TIME.search(line)
            if tm:
                last_train_time = float(tm.group(1))
                total_time += last_train_time

        m = VAL_LINE.search(line)
        if m:
            last_val_mae = float(m.group(1))
            vm = VAL_TIME.search(line)
            if vm:
                last_val_time = float(vm.group(1))
                total_time += last_val_time

        if BEST_CKPT.search(line):
            if last_val_mae is None:
                continue
            if last_val_mae <= best_val_so_far:
                best_val_so_far = last_val_mae
                out["best_val_mae"] = last_val_mae
                out["best_epoch"] = current_epoch
                out["train_mae_at_best_val"] = last_train_mae
                epoch_parts = [t for t in (last_train_time, last_val_time) if t is not None]
                for j in range(i + 1, min(i + 40, len(lines))):
                    tm = TEST_BLOCK.search(lines[j])
                    if tm:
                        out["test_mae_at_best_val"] = float(tm.group(1))
                        out["test_rmse_at_best_val"] = float(tm.group(2))
                        out["test_mape_at_best_val"] = float(tm.group(3))
                        out["parsed_test_source"] = source
                        ttm = TEST_TIME.search(lines[j])
                        if ttm:
                            epoch_parts.append(float(ttm.group(1)))
                        break
                if epoch_parts:
                    out["epoch_time"] = sum(epoch_parts)

        m = TEST_BLOCK.search(line)
        if m:
            out["final_test_mae"] = float(m.group(1))
            out["final_test_rmse"] = float(m.group(2))
            out["final_test_mape"] = float(m.group(3))

    if total_time > 0:
        out["total_time"] = total_time
    return out


def fmt_list(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, list):
        return repr(v)
    return str(v)


def base_row(variant: str, seed: int, cfg_path: Path, ckpt_root: Path) -> dict[str, Any]:
    spec = variant_spec(variant)
    return {
        "variant": variant,
        "seed": seed,
        "spatial_placement": spec.get("spatial_placement"),
        "post_spatial_mode": spec.get("post_spatial_mode"),
        "graph_resolution_ratios": fmt_list(spec.get("graph_resolution_ratios")),
        "graph_resolution_alphas": fmt_list(spec.get("graph_resolution_alphas")),
        "graph_resolution_topks": fmt_list(spec.get("graph_resolution_topks")),
        "graph_resolution_betas": fmt_list(spec.get("graph_resolution_betas")),
        "graph_resolution_rhos": fmt_list(spec.get("graph_resolution_rhos")),
        "clustering_seed": spec.get("clustering_seed", ""),
        "graph_cluster_method": spec.get("graph_cluster_method", "current"),
        "cluster_max_lag": spec.get("cluster_max_lag", ""),
        "cluster_lambda_s": spec.get("cluster_lambda_s", ""),
        "cluster_acf_lag": spec.get("cluster_acf_lag", ""),
        "best_epoch": None,
        "best_val_mae": None,
        "test_mae_at_best_val": None,
        "test_rmse_at_best_val": None,
        "test_mape_at_best_val": None,
        "final_test_mae": None,
        "train_mae_at_best_val": None,
        "params": count_params(cfg_path),
        "epoch_time": None,
        "total_time": None,
        "status": "pending",
        "error_message": "",
        "config_path": str(cfg_path),
        "checkpoint_dir": str(ckpt_dir_for(variant, seed, ckpt_root)),
        "log_file": "",
    }


def summarize_row(
    variant: str,
    seed: int,
    cfg_path: Path,
    log_root: Path,
    ckpt_root: Path,
) -> dict[str, Any]:
    row = base_row(variant, seed, cfg_path, ckpt_root)
    log_dir = log_dir_for(variant, seed, log_root)
    train_log = log_dir / "train.log"
    row["log_file"] = str(train_log) if train_log.is_file() else ""
    sources = collect_log_sources(variant, seed, log_root, ckpt_root)
    tlog = sources.get("parsed_training_log") or ""
    parsed = parse_training_log(collect_log_text(variant, seed, log_root, ckpt_root), source=tlog)
    row.update(parsed)
    row.update(sources)
    if parsed.get("test_mae_at_best_val") is not None:
        row["status"] = "ok"
    elif parsed.get("final_test_mae") is not None:
        row["status"] = "ok_final_only"
    else:
        row["status"] = "failed_no_metrics"
    return row


def is_completed(
    variant: str,
    seed: int,
    cfg_path: Path,
    log_root: Path,
    ckpt_root: Path,
) -> bool:
    row = summarize_row(variant, seed, cfg_path, log_root, ckpt_root)
    return row.get("test_mae_at_best_val") is not None


class GPUScheduler:
    def __init__(self, gpus: list[str], max_workers_per_gpu: int = 1):
        self.gpus = gpus
        self.max_workers = max(1, max_workers_per_gpu)
        self.lock = threading.Lock()
        self.in_use = {g: 0 for g in gpus}

    def acquire(self) -> str:
        while True:
            with self.lock:
                for gpu in self.gpus:
                    if self.in_use[gpu] < self.max_workers:
                        self.in_use[gpu] += 1
                        return gpu
            time.sleep(2)

    def release(self, gpu: str) -> None:
        with self.lock:
            self.in_use[gpu] = max(0, self.in_use[gpu] - 1)


def save_error_tail(log_file: Path, variant: str, seed: int, log_root: Path) -> str:
    log_dir = log_dir_for(variant, seed, log_root)
    tail_path = log_dir / "error_tail.txt"
    log_dir.mkdir(parents=True, exist_ok=True)
    if log_file.is_file():
        lines = log_file.read_text(errors="replace").splitlines()
        tail_path.write_text("\n".join(lines[-ERROR_TAIL_LINES:]) + "\n", encoding="utf-8")
    else:
        tail_path.write_text("", encoding="utf-8")
    return str(tail_path)


def print_log_tail(log_file: Path, variant: str, seed: int, lines: int = SHOW_ERROR_LINES) -> None:
    if not log_file.is_file():
        return
    text_lines = log_file.read_text(errors="replace").splitlines()
    tail = text_lines[-lines:]
    print(f"\n--- error tail: {variant} seed={seed} (last {len(tail)} lines) ---")
    for line in tail:
        print(line)
    print("--- end error tail ---\n")


def show_failed_errors(rows: list[dict], lines: int = SHOW_ERROR_LINES) -> None:
    failed = [r for r in rows if not str(r.get("status", "")).startswith("ok")]
    if not failed:
        return
    print("\nFailed run diagnostics:\n")
    for row in sorted(failed, key=lambda r: (r["variant"], int(r["seed"]))):
        log_path = Path(row.get("log_file", ""))
        print(f"  {row['variant']} seed={row['seed']} status={row['status']}")
        print_log_tail(log_path, row["variant"], int(row["seed"]), lines=lines)


def run_one(
    variant: str,
    cfg_path: Path,
    gpu: str,
    seed: int,
    log_root: Path,
    ckpt_root: Path,
    show_errors: bool,
) -> dict[str, Any]:
    log_dir = log_dir_for(variant, seed, log_root)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "train.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    cmd = [sys.executable, str(RUN_PY), "--cfg", cfg_for_easytorch(cfg_path), "--gpus", "0"]
    row = base_row(variant, seed, cfg_path, ckpt_root)
    row["log_file"] = str(log_file)
    row["status"] = "running"
    try:
        with open(log_file, "w", encoding="utf-8") as lf:
            proc = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=lf, stderr=subprocess.STDOUT)
        sources = collect_log_sources(variant, seed, log_root, ckpt_root)
        tlog = sources.get("parsed_training_log") or ""
        parsed = parse_training_log(
            collect_log_text(variant, seed, log_root, ckpt_root), source=tlog
        )
        row.update(parsed)
        row.update(sources)
        if proc.returncode != 0:
            row["status"] = f"exit_{proc.returncode}"
            row["error_message"] = save_error_tail(log_file, variant, seed, log_root)
            if show_errors:
                print_log_tail(log_file, variant, seed)
        elif parsed.get("test_mae_at_best_val") is None and parsed.get("final_test_mae") is None:
            row["status"] = "failed_no_metrics"
            row["error_message"] = save_error_tail(log_file, variant, seed, log_root)
        elif parsed.get("test_mae_at_best_val") is not None:
            row["status"] = "ok"
        else:
            row["status"] = "ok_final_only"
    except Exception as e:
        row["status"] = f"error:{e}"
        row["error_message"] = str(e)
    return row


def build_jobs(
    variants: list[str],
    seeds: list[int],
    schedule_order: str,
) -> list[tuple[str, int]]:
    jobs: list[tuple[str, int]] = []
    if schedule_order == "variant_first":
        for variant in variants:
            for seed in seeds:
                jobs.append((variant, seed))
    else:
        for seed in seeds:
            for variant in variants:
                jobs.append((variant, seed))
    return jobs


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    var = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return mean, math.sqrt(var)


def fmt_val(v: Any, digits: int = 4) -> str:
    if v is None or v == "":
        return ""
    if isinstance(v, float) and math.isnan(v):
        return ""
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def build_variant_summary(rows: list[dict]) -> list[dict[str, Any]]:
    by_variant: dict[str, list[dict]] = {}
    for row in rows:
        by_variant.setdefault(row["variant"], []).append(row)
    summary = []
    for variant, runs in by_variant.items():
        ok = [
            r for r in runs
            if r.get("test_mae_at_best_val") is not None
        ]
        mae_vals = [float(r["test_mae_at_best_val"]) for r in ok]
        rmse_vals = [
            float(r["test_rmse_at_best_val"]) for r in ok if r.get("test_rmse_at_best_val") is not None
        ]
        mape_vals = [
            float(r["test_mape_at_best_val"]) for r in ok if r.get("test_mape_at_best_val") is not None
        ]
        m_mae, s_mae = mean_std(mae_vals)
        m_rmse, s_rmse = mean_std(rmse_vals)
        m_mape, s_mape = mean_std(mape_vals)
        summary.append({
            "variant": variant,
            "n": len(ok),
            "failed": len(runs) - len(ok),
            "mae_mean": m_mae,
            "mae_std": s_mae,
            "rmse_mean": m_rmse,
            "rmse_std": s_rmse,
            "mape_mean": m_mape,
            "mape_std": s_mape,
        })
    summary.sort(key=lambda s: (math.inf if math.isnan(s["mae_mean"]) else s["mae_mean"]))
    return summary


def auto_conclusions(rows: list[dict], summary: list[dict]) -> list[str]:
    lines: list[str] = []
    ok_summary = [s for s in summary if s["n"] > 0 and not math.isnan(s["mae_mean"])]
    if not ok_summary:
        lines.append("- 暂无可汇总结果。")
        return lines

    best = ok_summary[0]
    lines.append(
        f"- 当前最优 variant 是 **{best['variant']}** "
        f"(mean test MAE@best-val = {fmt_val(best['mae_mean'])}, std = {fmt_val(best['mae_std'])})。"
    )

    def mean_for(name: str) -> float | None:
        for s in ok_summary:
            if s["variant"] == name:
                return s["mae_mean"]
        return None

    g0 = mean_for("G0_no_spatial")
    g1 = mean_for("G1_final_adaptive")
    g3 = mean_for("G3_final_adaptive_ms")
    g4 = mean_for("G4_final_adaptive_ms_uniform")
    g5 = mean_for("G5_final_adaptive_ms_small")
    gr0 = mean_for("GR0_default")
    gr6 = mean_for("GR6_default_adaptive_ms")
    gr7 = mean_for("GR7_sparse_topk")
    gr_variants = [s for s in ok_summary if s["variant"].startswith("GR")]
    if g3 is not None and g1 is not None:
        diff = g3 - g1
        lines.append(
            f"- adaptive_multiscale_only (G3) vs adaptive_only (G1): "
            f"{'更好' if diff < 0 else '更差' if diff > 0 else '持平'} "
            f"(ΔMAE = {fmt_val(diff)})."
        )
    if gr6 is not None and gr0 is not None:
        diff = gr6 - gr0
        lines.append(
            f"- Graph Resolution + adaptive_ms (GR6) vs GR0_default: "
            f"{'更好' if diff < 0 else '更差' if diff > 0 else '持平'} "
            f"(ΔMAE = {fmt_val(diff)})."
        )
    init_cmp = {
        "G4_final_adaptive_ms_uniform": g4,
        "G3_final_adaptive_ms": g3,
    }
    init_valid = {k: v for k, v in init_cmp.items() if v is not None}
    if len(init_valid) >= 2:
        best_init = min(init_valid, key=init_valid.get)
        lines.append(
            f"- adaptive_ms 初始化对比: **{best_init}** 更好 "
            f"(uniform={fmt_val(g4)}, favor_largest={fmt_val(g3)})."
        )
    if g5 is not None and g3 is not None:
        diff = g5 - g3
        lines.append(
            f"- small topk [4,8,16] (G5) vs default [8,16,32] (G3): "
            f"{'更稳/更好' if diff <= 0 else '更差'} (ΔMAE = {fmt_val(diff)})."
        )
    if gr0 is not None and g1 is not None:
        diff = gr0 - g1
        lines.append(
            f"- Graph Resolution (GR0_default) vs final adaptive (G1): "
            f"{'更好' if diff < 0 else '更差' if diff > 0 else '持平'} "
            f"(ΔMAE = {fmt_val(diff)})."
        )
    if gr7 is not None and gr0 is not None:
        diff = gr7 - gr0
        lines.append(
            f"- sparse topk [4,8,16] (GR7_sparse_topk) vs default [8,16,32] (GR0_default): "
            f"{'更好' if diff < 0 else '更差' if diff > 0 else '持平'} "
            f"(ΔMAE = {fmt_val(diff)})."
        )
    gr9 = mean_for("GR9_pearson_balanced_pam")
    if gr9 is not None and gr7 is not None:
        diff = gr9 - gr7
        lines.append(
            f"- pearson balanced PAM (GR9) vs spectral baseline (GR7): "
            f"{'更好' if diff < 0 else '更差' if diff > 0 else '持平'} "
            f"(ΔMAE = {fmt_val(diff)})."
        )
    if gr0 is not None and g0 is not None:
        diff = gr0 - g0
        lines.append(
            f"- Graph Resolution (GR0_default) vs no spatial (G0): "
            f"{'更好' if diff < 0 else '更差' if diff > 0 else '持平'} "
            f"(ΔMAE = {fmt_val(diff)})."
        )

    alpha_cmp = {
        "GR3_small_alpha": mean_for("GR3_small_alpha"),
        "GR0_default": mean_for("GR0_default"),
        "GR4_large_alpha": mean_for("GR4_large_alpha"),
        "GR5_dense_topk": mean_for("GR5_dense_topk"),
        "GR7_sparse_topk": mean_for("GR7_sparse_topk"),
    }
    alpha_valid = {k: v for k, v in alpha_cmp.items() if v is not None}
    if alpha_valid:
        best_alpha = min(alpha_valid, key=alpha_valid.get)
        lines.append(
            f"- alpha/topk 对比中最好的是 **{best_alpha}** "
            f"(MAE = {fmt_val(alpha_valid[best_alpha])})。"
        )

    unstable: list[str] = []
    by_variant: dict[str, list[dict]] = {}
    for row in rows:
        if row.get("test_mae_at_best_val") is not None:
            by_variant.setdefault(row["variant"], []).append(row)
    for variant, runs in by_variant.items():
        vals = [float(r["test_mae_at_best_val"]) for r in runs]
        if len(vals) >= 2:
            m, s = mean_std(vals)
            if m > 0 and s / m > 0.03:
                unstable.append(f"{variant} (std/mean={fmt_val(s / m)})")
    if unstable:
        lines.append(f"- 存在较不稳定 seed 的 variants: {', '.join(unstable)}。")
    else:
        lines.append("- 已完成 variants 的 seed 波动整体较平稳。")

    if gr_variants:
        best_gr = min(gr_variants, key=lambda s: s["mae_mean"])
        lines.append(
            f"- Graph Resolution 系列内部最优: **{best_gr['variant']}** "
            f"(MAE = {fmt_val(best_gr['mae_mean'])})。"
        )
    return lines


def write_outputs(rows: list[dict], out_csv: Path, out_md: Path) -> None:
    rows = sorted(rows, key=lambda r: (r["variant"], int(r["seed"])))
    fields = [
        "variant", "seed", "spatial_placement", "post_spatial_mode",
        "graph_resolution_ratios", "graph_resolution_alphas", "graph_resolution_topks",
        "graph_resolution_betas", "graph_resolution_rhos", "clustering_seed",
        "graph_cluster_method", "cluster_max_lag", "cluster_lambda_s", "cluster_acf_lag",
        "best_epoch", "best_val_mae", "test_mae_at_best_val", "test_rmse_at_best_val",
        "test_mape_at_best_val", "final_test_mae", "train_mae_at_best_val",
        "params", "epoch_time", "total_time", "status", "error_message",
        "config_path", "checkpoint_dir", "log_file",
        "parsed_training_log", "parsed_wrapper_log", "parsed_checkpoint_dir",
        "parsed_val_source", "parsed_test_source",
    ]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    summary = build_variant_summary(rows)
    summary_csv = out_csv.with_name(out_csv.stem + "_summary.csv")
    sum_fields = [
        "variant", "n", "failed", "mae_mean", "mae_std",
        "rmse_mean", "rmse_std", "mape_mean", "mape_std",
    ]
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=sum_fields)
        w.writeheader()
        w.writerows(summary)

    md = [
        "# PeMS04 Graph Module / Graph Resolution Ablation\n\n",
        "Protocol: PeMS04 12→12, chain_lengths=[3,6,12], use_prev_condition=True, "
        "no prior / no PTSE / no MTSR-P.\n\n",
        "## Per-run results\n\n",
        "| variant | seed | placement | test MAE@best | test RMSE@best | test MAPE@best | "
        "best val MAE | best epoch | status |\n",
        "|---|---:|---|---:|---:|---:|---:|---:|---|\n",
    ]
    for r in rows:
        md.append(
            f"| {r['variant']} | {r['seed']} | {r.get('spatial_placement','')} | "
            f"{fmt_val(r.get('test_mae_at_best_val'))} | {fmt_val(r.get('test_rmse_at_best_val'))} | "
            f"{fmt_val(r.get('test_mape_at_best_val'))} | {fmt_val(r.get('best_val_mae'))} | "
            f"{r.get('best_epoch','')} | {r.get('status','')} |\n"
        )

    md.append("\n## Per-variant mean/std\n\n")
    md.append("| variant | n | MAE mean | MAE std | RMSE mean | RMSE std | MAPE mean | MAPE std | failed |\n")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for s in summary:
        md.append(
            f"| {s['variant']} | {s['n']} | {fmt_val(s['mae_mean'])} | {fmt_val(s['mae_std'])} | "
            f"{fmt_val(s['rmse_mean'])} | {fmt_val(s['rmse_std'])} | {fmt_val(s['mape_mean'])} | "
            f"{fmt_val(s['mape_std'])} | {s['failed']} |\n"
        )

    md.append("\n## Leaderboard (by mean test MAE@best-val)\n\n")
    md.append("| rank | variant | MAE mean | MAE std |\n")
    md.append("|---:|---|---:|---:|\n")
    for i, s in enumerate(summary, 1):
        md.append(f"| {i} | {s['variant']} | {fmt_val(s['mae_mean'])} | {fmt_val(s['mae_std'])} |\n")

    md.append("\n## Auto conclusions\n\n")
    for line in auto_conclusions(rows, summary):
        md.append(line + "\n")

    out_md.write_text("".join(md), encoding="utf-8")


def variant_skip_reason(variant: str) -> str | None:
    spec = variant_spec(variant)
    method = str(spec.get("graph_cluster_method", "current")).lower()
    if method == "joint_pearson_spatial_balanced_pam":
        sp = spec.get("cluster_spatial_coord_path")
        if not sp:
            return "joint_pearson_spatial_balanced_pam requires cluster_spatial_coord_path"
        p = Path(sp)
        if not p.is_absolute():
            p = ROOT / p
        if not p.is_file():
            return f"cluster_spatial_coord_path not found: {p}"
    return None


def skipped_row(variant: str, seed: int, cfg_path: Path, ckpt_root: Path, reason: str) -> dict[str, Any]:
    row = base_row(variant, seed, cfg_path, ckpt_root)
    row["status"] = "skipped"
    row["error_message"] = reason
    return row


def dry_run_info(variant: str, seed: int, cfg_path: Path, ckpt_root: Path) -> None:
    spec = variant_spec(variant)
    cmd = f"{sys.executable} {RUN_PY} --cfg {cfg_for_easytorch(cfg_path)} --gpus <GPU>"
    print(f"  [{variant} seed={seed}]")
    print(f"    config_path: {cfg_path}")
    print(f"    checkpoint_dir: {ckpt_dir_for(variant, seed, ckpt_root)}")
    print(f"    spatial_placement: {spec.get('spatial_placement')}")
    print(f"    post_spatial_mode: {spec.get('post_spatial_mode')}")
    if spec.get("spatial_placement") == "temporal_first_graph_resolution":
        for key in GRAPH_RES_KEYS:
            print(f"    {key}: {spec.get(key)}")
        print(f"    clustering_seed: {spec.get('clustering_seed')}")
        print(f"    graph_cluster_method: {spec.get('graph_cluster_method', 'current')}")
        if spec.get("cluster_max_lag") not in (None, ""):
            print(f"    cluster_max_lag: {spec.get('cluster_max_lag')}")
        if spec.get("cluster_lambda_s") not in (None, ""):
            print(f"    cluster_lambda_s: {spec.get('cluster_lambda_s')}")
        if spec.get("cluster_acf_lag") not in (None, ""):
            print(f"    cluster_acf_lag: {spec.get('cluster_acf_lag')}")
        if spec.get("cluster_spatial_coord_path"):
            print(f"    cluster_spatial_coord_path: {spec.get('cluster_spatial_coord_path')}")
        if spec.get("spatial_graph_loss_weights") not in (None, ""):
            print(f"    spatial_graph_loss_weights: {spec.get('spatial_graph_loss_weights')}")
    if spec.get("post_spatial_mode") == "adaptive_multiscale_only":
        print(f"    adaptive_ms_topks: {spec.get('adaptive_ms_topks')}")
        print(f"    adaptive_ms_alpha: {spec.get('adaptive_ms_alpha')}")
        print(f"    adaptive_ms_init: {spec.get('adaptive_ms_init')}")
        print(f"    adaptive_ms_share_logits: {spec.get('adaptive_ms_share_logits')}")
    print(f"    params: {count_params(cfg_path)}")
    print(f"    cmd: {cmd}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Graph Resolution ablation on PeMS04 12→12.")
    parser.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS, choices=list(VARIANT_SPECS.keys()))
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--gpus", nargs="+", default=DEFAULT_GPUS)
    parser.add_argument("--base_cfg", default=str(DEFAULT_BASE_CFG.relative_to(ROOT)))
    parser.add_argument("--work_dir", default=str(DEFAULT_WORK_DIR.relative_to(ROOT)))
    parser.add_argument("--out", default="results/pems04_graph_resolution_ablation.csv")
    parser.add_argument("--markdown", default="results/pems04_graph_resolution_ablation.md")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--summary-only", "--summary_only", action="store_true", dest="summary_only")
    parser.add_argument("--show_errors", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--max_workers_per_gpu", type=int, default=1)
    parser.add_argument(
        "--schedule_order",
        choices=["seed_first", "variant_first"],
        default="seed_first",
    )
    args = parser.parse_args()

    base_cfg = Path(args.base_cfg)
    if not base_cfg.is_absolute():
        base_cfg = ROOT / base_cfg
    if not base_cfg.is_file():
        print(f"Missing base config: {base_cfg}")
        return 1
    if not RUN_PY.is_file():
        print(f"Missing run entry: {RUN_PY}")
        return 1

    work_dir = Path(args.work_dir)
    if not work_dir.is_absolute():
        work_dir = ROOT / work_dir
    log_root = DEFAULT_LOG_ROOT
    ckpt_root = DEFAULT_CKPT_ROOT
    out_csv = Path(args.out)
    if not out_csv.is_absolute():
        out_csv = ROOT / out_csv
    out_md = Path(args.markdown)
    if not out_md.is_absolute():
        out_md = ROOT / out_md

    job_keys = build_jobs(args.variants, args.seeds, args.schedule_order)
    configs: list[tuple[str, int, Path]] = []
    skipped: list[dict[str, Any]] = []
    for variant, seed in job_keys:
        skip_reason = variant_skip_reason(variant)
        cfg_path = generate_temp_config(variant, seed, base_cfg, work_dir, ckpt_root)
        if skip_reason:
            skipped.append(skipped_row(variant, seed, cfg_path, ckpt_root, skip_reason))
            print(f"[skip-setup] {variant} seed={seed}: {skip_reason}")
            continue
        try:
            validate_generated_config(cfg_path, variant)
        except Exception as e:
            print(f"Config validation failed for {variant} seed={seed}: {e}")
            return 1
        configs.append((variant, seed, cfg_path))

    if args.dry_run:
        print("Dry run — variants / seeds / GPU queue:\n")
        print(f"  variants: {args.variants}")
        print(f"  seeds: {args.seeds}")
        print(f"  gpus: {args.gpus}")
        print(f"  schedule_order: {args.schedule_order}")
        print(f"  max_workers_per_gpu: {args.max_workers_per_gpu}")
        print(f"  work_dir: {work_dir}")
        print()
        for variant, seed, cfg_path in configs:
            dry_run_info(variant, seed, cfg_path, ckpt_root)
        print(f"\n{len(configs)} jobs queued.")
        return 0

    if args.summary_only:
        rows = skipped + [
            summarize_row(v, s, p, log_root, ckpt_root)
            for v, s, p in configs
        ]
        write_outputs(rows, out_csv, out_md)
        print(f"Wrote {out_csv}, {out_md}, {out_csv.with_name(out_csv.stem + '_summary.csv')}")
        if args.show_errors:
            show_failed_errors(rows)
        return 0

    scheduler = GPUScheduler(args.gpus, max_workers_per_gpu=args.max_workers_per_gpu)
    rows: list[dict] = []
    lock = threading.Lock()

    def worker(variant: str, seed: int, cfg_path: Path):
        if args.skip_existing and is_completed(variant, seed, cfg_path, log_root, ckpt_root):
            row = summarize_row(variant, seed, cfg_path, log_root, ckpt_root)
            with lock:
                rows.append(row)
            print(f"[skip] {variant} seed={seed} mae={row.get('test_mae_at_best_val')}")
            return
        gpu = scheduler.acquire()
        try:
            print(f"[start] {variant} seed={seed} gpu={gpu}")
            row = run_one(
                variant, cfg_path, gpu, seed, log_root, ckpt_root, args.show_errors
            )
            with lock:
                rows.append(row)
            print(
                f"[done]  {variant} seed={seed} status={row['status']} "
                f"mae={row.get('test_mae_at_best_val')}"
            )
        finally:
            scheduler.release(gpu)

    threads = [
        threading.Thread(target=worker, args=(variant, seed, cfg_path))
        for variant, seed, cfg_path in configs
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    write_outputs(skipped + rows, out_csv, out_md)
    print(f"Wrote {out_csv}, {out_md}, {out_csv.with_name(out_csv.stem + '_summary.csv')}")
    if args.show_errors:
        show_failed_errors(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
