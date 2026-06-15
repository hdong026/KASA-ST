#!/usr/bin/env python3
"""Evaluate chain_3_6_12_no_spatial checkpoints as a Forecast-State Chain (no training)."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import os
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basicts.archs.arch_zoo.ChainForecasting_arch import ChainForecasting
from basicts.archs.arch_zoo.ChainForecasting_arch.kasa_temporal_step import interpolate_forecast
from basicts.data import SCALER_REGISTRY
from basicts.metrics import masked_mae, masked_mape, masked_rmse
from basicts.utils import load_pkl

VARIANT = "chain_3_6_12_no_spatial"
CKPT_ROOT = ROOT / "checkpoints" / "chain_forecasting_pems04"
TEMP_CFG_DIR = ROOT / "tmp_configs" / "chain_forecasting_pems04"
ABLATION_CSV = ROOT / "results" / "pems04_chain_forecasting_ablation.csv"
SEED1_REF_MAE = 18.1579
SEED1_MAE_TOL = 0.10

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
METRIC_FUNCS = {"MAE": masked_mae, "RMSE": masked_rmse, "MAPE": masked_mape}


def load_cfg(cfg_path: Path):
    spec = importlib.util.spec_from_file_location("chain_cfg", cfg_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.CFG


def find_state_dict(ckpt_obj) -> dict | None:
    if isinstance(ckpt_obj, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            if key in ckpt_obj and isinstance(ckpt_obj[key], dict):
                return ckpt_obj[key]
        if all(isinstance(k, str) for k in ckpt_obj.keys()):
            return ckpt_obj
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


def t_crit(n: int) -> float:
    if n in T_CRIT:
        return T_CRIT[n]
    try:
        from scipy import stats

        return float(stats.t.ppf(0.975, n - 1))
    except Exception:
        return 1.96


def mean_std_ci(values: list[float]) -> tuple[float, float, float]:
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    mean = sum(values) / n
    if n == 1:
        return mean, 0.0, float("nan")
    var = sum((x - mean) ** 2 for x in values) / (n - 1)
    std = math.sqrt(var)
    return mean, std, t_crit(n) * std / math.sqrt(n)


def temp_cfg_path(seed: int) -> Path:
    return TEMP_CFG_DIR / f"{VARIANT}_seed{seed}.py"


def ckpt_dir_for(seed: int) -> Path:
    return CKPT_ROOT / f"{VARIANT}_seed{seed}"


def resolve_run_assets(seed: int) -> tuple[Path, Path, Path, float | None]:
    """Return (cfg_path, ckpt_path, run_dir, logged_test_mae)."""
    ckpt_base = ckpt_dir_for(seed)
    if not ckpt_base.is_dir():
        raise FileNotFoundError(f"Checkpoint root missing for seed {seed}: {ckpt_base}")

    candidates: list[tuple[float, Path, Path, Path]] = []
    for run_dir in sorted(ckpt_base.iterdir()):
        if not run_dir.is_dir():
            continue
        ckpt_path = run_dir / "ChainForecasting_best_val_MAE.pt"
        if not ckpt_path.is_file():
            continue
        cfg_in_run = run_dir / f"{VARIANT}_seed{seed}.py"
        cfg_path = cfg_in_run if cfg_in_run.is_file() else temp_cfg_path(seed)
        if not cfg_path.is_file():
            raise FileNotFoundError(
                f"No config for seed {seed}: expected {cfg_in_run} or {temp_cfg_path(seed)}"
            )
        log_text = "\n".join(
            p.read_text(errors="replace") for p in sorted(run_dir.glob("training_log_*.log"))
        )
        logged = parse_metrics(log_text).get("mae")
        sort_key = logged if logged is not None else float("inf")
        candidates.append((sort_key, cfg_path, ckpt_path, run_dir))

    if not candidates:
        raise FileNotFoundError(
            f"No ChainForecasting_best_val_MAE.pt found under {ckpt_base}"
        )

    candidates.sort(key=lambda x: x[0])
    _, cfg_path, ckpt_path, run_dir = candidates[0]
    logged_mae = candidates[0][0] if candidates[0][0] != float("inf") else None
    return cfg_path, ckpt_path, run_dir, logged_mae


def load_model(cfg, ckpt_path: Path, device: str):
    model = cfg.MODEL.ARCH(**cfg.MODEL.PARAM).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = find_state_dict(ckpt)
    if state is None:
        raise ValueError(f"Could not find model state dict in {ckpt_path}")
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def build_dataloader(cfg, split: str) -> DataLoader:
    split_key = split.upper()
    if split_key == "VALID":
        split_key = "VAL"
    data_cfg = getattr(cfg, split_key)
    mode = "valid" if split == "val" else split
    data_file = (
        f"{data_cfg.DATA.DIR}/data_in{cfg.DATASET_INPUT_LEN}_out{cfg.DATASET_OUTPUT_LEN}.pkl"
    )
    index_file = (
        f"{data_cfg.DATA.DIR}/index_in{cfg.DATASET_INPUT_LEN}_out{cfg.DATASET_OUTPUT_LEN}.pkl"
    )
    dataset = cfg.DATASET_CLS(
        data_file_path=data_file,
        index_file_path=index_file,
        mode=mode,
    )
    return DataLoader(
        dataset,
        batch_size=data_cfg.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )


def select_features(data: torch.Tensor, features: list[int] | None) -> torch.Tensor:
    if features is not None:
        return data[:, :, :, features]
    return data


def load_scaler(cfg):
    path = (
        f"{cfg.TRAIN.DATA.DIR}/scaler_in{cfg.DATASET_INPUT_LEN}_out{cfg.DATASET_OUTPUT_LEN}.pkl"
    )
    return load_pkl(path)


def rescale_tensor(scaler: dict, data: torch.Tensor) -> torch.Tensor:
    return SCALER_REGISTRY.get(scaler["func"])(data, **scaler["args"])


def metric_value(metric_func, pred: torch.Tensor, target: torch.Tensor, null_val: float) -> float:
    return float(metric_func(pred, target, null_val=null_val).item())


def compute_all_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    scaler: dict,
    null_val: float,
) -> dict[str, float]:
    pred_r = rescale_tensor(scaler, pred)
    target_r = rescale_tensor(scaler, target)
    return {
        name: metric_value(func, pred_r, target_r, null_val)
        for name, func in METRIC_FUNCS.items()
    }


def per_sample_mae(pred: torch.Tensor, target: torch.Tensor, scaler: dict, null_val: float) -> torch.Tensor:
    pred_r = rescale_tensor(scaler, pred)
    target_r = rescale_tensor(scaler, target)
    eps = 5e-5
    mask = ~torch.isclose(
        target_r,
        torch.tensor(null_val, device=target_r.device, dtype=target_r.dtype),
        atol=eps,
        rtol=0.0,
    )
    abs_err = torch.abs(pred_r - target_r) * mask.float()
    counts = mask.sum(dim=(1, 2, 3)).clamp(min=1)
    return abs_err.sum(dim=(1, 2, 3)) / counts


def horizon_mae_series(
    pred: torch.Tensor,
    target: torch.Tensor,
    scaler: dict,
    null_val: float,
    horizons: int = 12,
) -> list[float]:
    pred_r = rescale_tensor(scaler, pred)
    target_r = rescale_tensor(scaler, target)
    return [
        metric_value(masked_mae, pred_r[:, h], target_r[:, h], null_val)
        for h in range(horizons)
    ]


@torch.no_grad()
def evaluate_seed(
    seed: int,
    split: str,
    device: str,
) -> dict:
    cfg_path, ckpt_path, run_dir, logged_mae = resolve_run_assets(seed)
    cfg = load_cfg(cfg_path)
    model = load_model(cfg, ckpt_path, device)
    loader = build_dataloader(cfg, split)
    scaler = load_scaler(cfg)
    null_val = cfg.TRAIN.NULL_VAL
    forward_features = cfg.MODEL.get("FORWARD_FEATURES", None)
    target_features = cfg.MODEL.get("TARGET_FEATURES", None)

    z3_parts, z6_parts, z12_parts, pred_parts, y_parts = [], [], [], [], []
    pred_z12_max_diff = 0.0

    for future_data, history_data in loader:
        history_data = history_data.to(device)
        future_data = future_data.to(device)
        history_in = select_features(history_data, forward_features)
        future_in = select_features(future_data, forward_features)

        out = model(
            history_data=history_in,
            future_data=future_in,
            train=False,
            return_all=True,
        )
        z3 = select_features(out["chain_preds"][0], target_features)
        z6 = select_features(out["chain_preds"][1], target_features)
        z12 = select_features(out["chain_preds"][2], target_features)
        pred = select_features(out["pred"], target_features)
        y = select_features(future_data, target_features)

        diff = torch.max(torch.abs(pred - z12)).item()
        pred_z12_max_diff = max(pred_z12_max_diff, diff)

        z3_parts.append(z3.cpu())
        z6_parts.append(z6.cpu())
        z12_parts.append(z12.cpu())
        pred_parts.append(pred.cpu())
        y_parts.append(y.cpu())

    z3_all = torch.cat(z3_parts, dim=0)
    z6_all = torch.cat(z6_parts, dim=0)
    z12_all = torch.cat(z12_parts, dim=0)
    pred_all = torch.cat(pred_parts, dim=0)
    y_all = torch.cat(y_parts, dim=0)

    pred_equals_z12 = bool(torch.allclose(pred_all, z12_all))
    pi3 = ChainForecasting.pool_target(y_all, 3)
    pi6 = ChainForecasting.pool_target(y_all, 6)
    u3 = interpolate_forecast(z3_all, 12)
    u6 = interpolate_forecast(z6_all, 12)
    u3_oracle = interpolate_forecast(pi3, 12)
    u6_oracle = interpolate_forecast(pi6, 12)

    final_metrics = compute_all_metrics(pred_all, y_all, scaler, null_val)
    native_z3 = compute_all_metrics(z3_all, pi3, scaler, null_val)
    native_z6 = compute_all_metrics(z6_all, pi6, scaler, null_val)
    native_z12 = compute_all_metrics(z12_all, y_all, scaler, null_val)
    recon_u3 = compute_all_metrics(u3, y_all, scaler, null_val)
    recon_u6 = compute_all_metrics(u6, y_all, scaler, null_val)
    recon_z12 = compute_all_metrics(z12_all, y_all, scaler, null_val)
    oracle_u3 = compute_all_metrics(u3_oracle, y_all, scaler, null_val)
    oracle_u6 = compute_all_metrics(u6_oracle, y_all, scaler, null_val)

    e3 = per_sample_mae(u3, y_all, scaler, null_val)
    e6 = per_sample_mae(u6, y_all, scaler, null_val)
    e12 = per_sample_mae(z12_all, y_all, scaler, null_val)

    mono = {
        "P_e6_lt_e3": float((e6 < e3).float().mean().item()),
        "P_e12_lt_e6": float((e12 < e6).float().mean().item()),
        "P_e3_gt_e6_gt_e12": float(((e3 > e6) & (e6 > e12)).float().mean().item()),
    }

    hz_u3 = horizon_mae_series(u3, y_all, scaler, null_val)
    hz_u6 = horizon_mae_series(u6, y_all, scaler, null_val)
    hz_z12 = horizon_mae_series(z12_all, y_all, scaler, null_val)

    return {
        "seed": seed,
        "cfg_path": str(cfg_path),
        "ckpt_path": str(ckpt_path),
        "run_dir": str(run_dir),
        "logged_test_mae": logged_mae,
        "pred_equals_z12": pred_equals_z12,
        "pred_z12_max_diff": pred_z12_max_diff,
        "final_mae": final_metrics["MAE"],
        "final_rmse": final_metrics["RMSE"],
        "final_mape": final_metrics["MAPE"],
        "native_z3_mae": native_z3["MAE"],
        "native_z3_rmse": native_z3["RMSE"],
        "native_z3_mape": native_z3["MAPE"],
        "native_z6_mae": native_z6["MAE"],
        "native_z6_rmse": native_z6["RMSE"],
        "native_z6_mape": native_z6["MAPE"],
        "native_z12_mae": native_z12["MAE"],
        "native_z12_rmse": native_z12["RMSE"],
        "native_z12_mape": native_z12["MAPE"],
        "recon_u3_mae": recon_u3["MAE"],
        "recon_u3_rmse": recon_u3["RMSE"],
        "recon_u3_mape": recon_u3["MAPE"],
        "recon_u6_mae": recon_u6["MAE"],
        "recon_u6_rmse": recon_u6["RMSE"],
        "recon_u6_mape": recon_u6["MAPE"],
        "recon_z12_mae": recon_z12["MAE"],
        "recon_z12_rmse": recon_z12["RMSE"],
        "recon_z12_mape": recon_z12["MAPE"],
        "oracle_u3_mae": oracle_u3["MAE"],
        "oracle_u3_rmse": oracle_u3["RMSE"],
        "oracle_u3_mape": oracle_u3["MAPE"],
        "oracle_u6_mae": oracle_u6["MAE"],
        "oracle_u6_rmse": oracle_u6["RMSE"],
        "oracle_u6_mape": oracle_u6["MAPE"],
        "E3": float(e3.mean().item()),
        "E6": float(e6.mean().item()),
        "E12": float(e12.mean().item()),
        **mono,
        "horizon_u3": hz_u3,
        "horizon_u6": hz_u6,
        "horizon_z12": hz_z12,
        "per_sample_e3": e3.numpy(),
        "per_sample_e6": e6.numpy(),
        "per_sample_e12": e12.numpy(),
    }


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize_metric(rows: list[dict], key: str) -> dict:
    vals = [float(r[key]) for r in rows]
    mean, std, ci = mean_std_ci(vals)
    return {"metric": key, "mean": mean, "std": std, "ci95": ci, "n": len(vals)}


def build_summary_rows(rows: list[dict]) -> list[dict]:
    keys = [
        "final_mae", "final_rmse", "final_mape",
        "native_z3_mae", "native_z6_mae", "native_z12_mae",
        "recon_u3_mae", "recon_u6_mae", "recon_z12_mae",
        "oracle_u3_mae", "oracle_u6_mae",
        "E3", "E6", "E12",
        "P_e6_lt_e3", "P_e12_lt_e6", "P_e3_gt_e6_gt_e12",
    ]
    return [summarize_metric(rows, k) for k in keys]


def build_paired_diff(rows: list[dict], left: str, right: str) -> dict:
    diffs = [float(r[left]) - float(r[right]) for r in rows]
    mean, std, ci = mean_std_ci(diffs)
    return {
        "pair": f"{left}-{right}",
        "left": left,
        "right": right,
        "mean_diff": mean,
        "std_diff": std,
        "ci95": ci,
        "n": len(diffs),
    }


def setup_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 100,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def save_figure(fig, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    plt.close(fig)


def plot_progressive_refinement(summary_rows: list[dict], out_dir: Path) -> None:
    setup_plot_style()
    labels = ["U$_{3\\to12}$(z3)", "U$_{6\\to12}$(z6)", "z12"]
    keys = ["recon_u3_mae", "recon_u6_mae", "recon_z12_mae"]
    means, cis = [], []
    for key in keys:
        row = next(r for r in summary_rows if r["metric"] == key)
        means.append(row["mean"])
        cis.append(row["ci95"] if not math.isnan(row["ci95"]) else 0.0)

    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=cis, capsize=3, color=["#4C72B0", "#55A868", "#C44E52"], width=0.62)
    ax.set_xticks(x, labels)
    ax.set_ylabel("MAE")
    ax.set_title("Progressive Refinement (full-resolution)")
    save_figure(fig, out_dir / "figures" / "progressive_refinement")


def plot_monotonicity(summary_rows: list[dict], out_dir: Path) -> None:
    setup_plot_style()
    labels = ["P(e$_6$<e$_3$)", "P(e$_{12}$<e$_6$)", "P(e$_3$>e$_6$>e$_{12}$)"]
    keys = ["P_e6_lt_e3", "P_e12_lt_e6", "P_e3_gt_e6_gt_e12"]
    means, cis = [], []
    for key in keys:
        row = next(r for r in summary_rows if r["metric"] == key)
        means.append(row["mean"])
        cis.append(row["ci95"] if not math.isnan(row["ci95"]) else 0.0)

    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=cis, capsize=3, color="#8172B3", width=0.62)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Proportion")
    ax.set_ylim(0, 1.0)
    ax.set_title("Sample-level Monotonic Improvement")
    save_figure(fig, out_dir / "figures" / "monotonicity")


def plot_horizon_wise(per_seed_rows: list[dict], out_dir: Path) -> None:
    setup_plot_style()
    horizons = np.arange(1, 13)
    series = {
        "U$_{3\\to12}$(z3)": np.array([r["horizon_u3"] for r in per_seed_rows]).mean(axis=0),
        "U$_{6\\to12}$(z6)": np.array([r["horizon_u6"] for r in per_seed_rows]).mean(axis=0),
        "z12": np.array([r["horizon_z12"] for r in per_seed_rows]).mean(axis=0),
    }
    colors = {"U$_{3\\to12}$(z3)": "#4C72B0", "U$_{6\\to12}$(z6)": "#55A868", "z12": "#C44E52"}

    fig, ax = plt.subplots(figsize=(4.0, 2.8))
    for name, values in series.items():
        ax.plot(horizons, values, marker="o", markersize=3.5, linewidth=1.6, label=name, color=colors[name])
    ax.set_xlabel("Horizon")
    ax.set_ylabel("MAE")
    ax.set_xticks(horizons)
    ax.set_title("Horizon-wise MAE")
    ax.legend(frameon=False)
    save_figure(fig, out_dir / "figures" / "horizon_wise")


def write_report(
    out_dir: Path,
    per_seed: list[dict],
    summary_rows: list[dict],
    mono_rows: list[dict],
    horizon_rows: list[dict],
    oracle_rows: list[dict],
    paired_rows: list[dict],
    split: str,
    sanity: dict,
) -> None:
    lines = [
        "# No-Spatial Forecast-State Chain Analysis (PeMS04)",
        "",
        f"Variant: `{VARIANT}` | Split: `{split}` | Seeds: {[r['seed'] for r in per_seed]}",
        "",
        "## Sanity Check",
        "",
        f"- Seed 1 final MAE: **{sanity['seed1_mae']:.4f}** (reference {SEED1_REF_MAE:.4f})",
        f"- |Δ| = {sanity['seed1_abs_diff']:.4f} (tolerance {SEED1_MAE_TOL:.2f})",
        f"- pred ≡ z12 (all seeds): {sanity['all_pred_eq_z12']}",
        f"- max |pred − z12|: {sanity['max_pred_z12_diff']:.2e}",
        "",
        "## Checkpoints",
        "",
        "| seed | config | checkpoint | logged test MAE |",
        "|---:|---|---|---:|",
    ]
    for r in per_seed:
        lines.append(
            f"| {r['seed']} | `{r['cfg_path']}` | `{r['ckpt_path']}` | "
            f"{r['logged_test_mae'] if r['logged_test_mae'] is not None else 'n/a'} |"
        )

    lines.extend(["", "## Summary (mean ± std, 95% CI)", ""])
    for row in summary_rows:
        ci = row["ci95"]
        ci_str = f"{ci:.4f}" if not math.isnan(ci) else "n/a"
        lines.append(f"- **{row['metric']}**: {row['mean']:.4f} ± {row['std']:.4f} (CI {ci_str})")

    lines.extend(["", "## Paired Differences (E3−E6, E6−E12)", ""])
    for row in paired_rows:
        ci_str = f"{row['ci95']:.4f}" if not math.isnan(row["ci95"]) else "n/a"
        lines.append(
            f"- **{row['pair']}**: mean {row['mean_diff']:.4f}, std {row['std_diff']:.4f}, CI {ci_str}"
        )

    lines.extend(["", "## Files", ""])
    for name in [
        "metrics_per_seed.csv",
        "metrics_summary.csv",
        "monotonicity.csv",
        "horizon_wise.csv",
        "oracle_floor.csv",
        "figures/progressive_refinement.{pdf,svg,png}",
        "figures/monotonicity.{pdf,svg,png}",
        "figures/horizon_wise.{pdf,svg,png}",
    ]:
        lines.append(f"- `{name}`")

    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze chain_3_6_12_no_spatial as Forecast-State Chain (eval only)."
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--gpus", type=str, default="0", help="CUDA device id(s); uses first id.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument(
        "--output-dir",
        default="results/no_spatial_forecast_chain_analysis",
        help="Directory for CSV/NPZ/report/figures.",
    )
    args = parser.parse_args()

    os.chdir(ROOT)
    gpu = args.gpus.split(",")[0].strip()
    device = f"cuda:{gpu}" if torch.cuda.is_available() and gpu.lower() != "cpu" else "cpu"
    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    per_seed: list[dict] = []
    for seed in args.seeds:
        print(f"Evaluating seed {seed} on {args.split} ({device}) ...")
        result = evaluate_seed(seed, args.split, device)
        per_seed.append(result)

        np.savez(
            out_dir / f"per_sample_seed{seed}.npz",
            e3=result["per_sample_e3"],
            e6=result["per_sample_e6"],
            e12=result["per_sample_e12"],
        )

    seed1 = next(r for r in per_seed if r["seed"] == 1)
    seed1_diff = abs(seed1["final_mae"] - SEED1_REF_MAE)
    if seed1_diff > SEED1_MAE_TOL:
        msg = (
            f"Seed 1 MAE mismatch: got {seed1['final_mae']:.4f}, "
            f"expected ~{SEED1_REF_MAE:.4f} (|Δ|={seed1_diff:.4f} > {SEED1_MAE_TOL:.2f}). "
            f"Check scaler ({load_scaler(load_cfg(Path(seed1['cfg_path'])))}), "
            f"checkpoint ({seed1['ckpt_path']}), or metric pipeline."
        )
        print(f"ERROR: {msg}", file=sys.stderr)
        return 2

    per_seed_csv_fields = [
        "seed", "cfg_path", "ckpt_path", "run_dir", "logged_test_mae",
        "pred_equals_z12", "pred_z12_max_diff",
        "final_mae", "final_rmse", "final_mape",
        "native_z3_mae", "native_z3_rmse", "native_z3_mape",
        "native_z6_mae", "native_z6_rmse", "native_z6_mape",
        "native_z12_mae", "native_z12_rmse", "native_z12_mape",
        "recon_u3_mae", "recon_u3_rmse", "recon_u3_mape",
        "recon_u6_mae", "recon_u6_rmse", "recon_u6_mape",
        "recon_z12_mae", "recon_z12_rmse", "recon_z12_mape",
        "oracle_u3_mae", "oracle_u3_rmse", "oracle_u3_mape",
        "oracle_u6_mae", "oracle_u6_rmse", "oracle_u6_mape",
        "E3", "E6", "E12",
        "P_e6_lt_e3", "P_e12_lt_e6", "P_e3_gt_e6_gt_e12",
    ]
    write_csv(out_dir / "metrics_per_seed.csv", per_seed, per_seed_csv_fields)

    summary_rows = build_summary_rows(per_seed)
    write_csv(
        out_dir / "metrics_summary.csv",
        summary_rows,
        ["metric", "mean", "std", "ci95", "n"],
    )

    mono_rows = [
        {
            "seed": r["seed"],
            "P_e6_lt_e3": r["P_e6_lt_e3"],
            "P_e12_lt_e6": r["P_e12_lt_e6"],
            "P_e3_gt_e6_gt_e12": r["P_e3_gt_e6_gt_e12"],
            "E3": r["E3"],
            "E6": r["E6"],
            "E12": r["E12"],
        }
        for r in per_seed
    ]
    write_csv(
        out_dir / "monotonicity.csv",
        mono_rows,
        ["seed", "P_e6_lt_e3", "P_e12_lt_e6", "P_e3_gt_e6_gt_e12", "E3", "E6", "E12"],
    )

    horizon_rows: list[dict] = []
    for r in per_seed:
        for h in range(12):
            horizon_rows.append(
                {
                    "seed": r["seed"],
                    "horizon": h + 1,
                    "u3_mae": r["horizon_u3"][h],
                    "u6_mae": r["horizon_u6"][h],
                    "z12_mae": r["horizon_z12"][h],
                }
            )
    write_csv(
        out_dir / "horizon_wise.csv",
        horizon_rows,
        ["seed", "horizon", "u3_mae", "u6_mae", "z12_mae"],
    )

    oracle_rows = [
        {
            "seed": r["seed"],
            "oracle_u3_mae": r["oracle_u3_mae"],
            "oracle_u3_rmse": r["oracle_u3_rmse"],
            "oracle_u3_mape": r["oracle_u3_mape"],
            "oracle_u6_mae": r["oracle_u6_mae"],
            "oracle_u6_rmse": r["oracle_u6_rmse"],
            "oracle_u6_mape": r["oracle_u6_mape"],
        }
        for r in per_seed
    ]
    write_csv(
        out_dir / "oracle_floor.csv",
        oracle_rows,
        [
            "seed",
            "oracle_u3_mae", "oracle_u3_rmse", "oracle_u3_mape",
            "oracle_u6_mae", "oracle_u6_rmse", "oracle_u6_mape",
        ],
    )

    paired_rows = [
        build_paired_diff(per_seed, "E3", "E6"),
        build_paired_diff(per_seed, "E6", "E12"),
    ]
    write_csv(
        out_dir / "paired_differences.csv",
        paired_rows,
        ["pair", "left", "right", "mean_diff", "std_diff", "ci95", "n"],
    )

    plot_progressive_refinement(summary_rows, out_dir)
    plot_monotonicity(summary_rows, out_dir)
    plot_horizon_wise(per_seed, out_dir)

    sanity = {
        "seed1_mae": seed1["final_mae"],
        "seed1_abs_diff": seed1_diff,
        "all_pred_eq_z12": all(r["pred_equals_z12"] for r in per_seed),
        "max_pred_z12_diff": max(r["pred_z12_max_diff"] for r in per_seed),
    }
    write_report(
        out_dir, per_seed, summary_rows, mono_rows, horizon_rows, oracle_rows, paired_rows, args.split, sanity
    )

    print(f"Done. Results written to {out_dir}")
    print(f"Seed 1 final MAE: {seed1['final_mae']:.4f} (ref {SEED1_REF_MAE:.4f})")
    print(f"pred == z12: {sanity['all_pred_eq_z12']}, max diff {sanity['max_pred_z12_diff']:.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
