#!/usr/bin/env python3
"""Evaluation-only qualitative export for the final F2FNet variant.

This script never launches training, never modifies checkpoints or training
logs, and never alters the model's numerical forward path.

Examples:

```bash
# Inspect paths without loading checkpoints
python scripts/export_f2fnet_qualitative.py \\
  --datasets PEMS04 PEMS07 PEMS08 KnowAir \\
  --horizon 12 \\
  --seed 1 \\
  --dry-run

# Export all four datasets
python scripts/export_f2fnet_qualitative.py \\
  --datasets PEMS04 PEMS07 PEMS08 KnowAir \\
  --horizon 12 \\
  --seed 1 \\
  --gpu 0

# Export one dataset only
python scripts/export_f2fnet_qualitative.py \\
  --datasets KnowAir \\
  --horizon 12 \\
  --seed 1 \\
  --gpu 0
```
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import pickle
import sys
import traceback
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_DATASETS = ["PEMS04", "PEMS07", "PEMS08", "KnowAir"]
DEFAULT_VARIANT = (
    "chain_interleaved_progressive_spatial_state_adapter_fixed_token_loss"
)
DEFAULT_HORIZON = 12
DEFAULT_SEED = 1
DEFAULT_INPUT_LEN = 12
EXPECTED_CHAIN = [3, 6, 12]
CKPT_SUFFIXES = (".pt", ".pth", ".ckpt", ".bin")
STATE_DICT_KEYS = ("state_dict", "model_state_dict", "model", "net")
EVAL_BANNER = (
    "EVALUATION-ONLY MODE: no training, checkpoint modification,\n"
    "or training-log modification will be performed."
)


def load_horizon_module():
    path = ROOT / "scripts" / "run_chain_forecasting_horizon.py"
    spec = importlib.util.spec_from_file_location("horizon_runner_qual", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


HZ = load_horizon_module()


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def results_ckpt_dir_for(slug: str, horizon: int, variant: str, seed: int) -> Path:
    """Alternate EasyTorch layout used by some PEMS04/PEMS08 runs.

    Same relative naming as ``HZ.ckpt_dir_for``, but under
    ``results/fixed_input_horizon_<slug>/`` instead of ``checkpoints/...``.
    """
    return (
        ROOT
        / "results"
        / f"fixed_input_horizon_{slug}"
        / f"h{horizon}"
        / f"{variant}_seed{seed}"
    )


def resolve_checkpoint_dir(
    primary: Path, slug: str, horizon: int, variant: str, seed: int
) -> tuple[Path, list[str]]:
    """Resolve the experiment checkpoint directory without guessing among peers.

    Preference order:
    1. Horizon-runner path under ``checkpoints/fixed_input_horizon_<slug>/``
    2. Same relative path under ``results/fixed_input_horizon_<slug>/``
    """
    warnings: list[str] = []
    alt = results_ckpt_dir_for(slug, horizon, variant, seed)
    if primary.is_dir():
        return primary, warnings
    if alt.is_dir():
        warnings.append(
            f"Primary checkpoint directory missing ({primary}); "
            f"using alternate EasyTorch results layout ({alt})"
        )
        return alt, warnings
    raise FileNotFoundError(
        "Checkpoint directory missing in both expected locations:\n"
        f"  primary: {primary}\n"
        f"  alternate: {alt}"
    )


def resolve_experiment(dataset: str, horizon: int, variant: str, seed: int) -> dict:
    if dataset not in HZ.DATASET_SPECS:
        raise ValueError(
            f"Unsupported dataset {dataset!r}. "
            f"Expected one of {sorted(HZ.DATASET_SPECS)}"
        )
    HZ.activate_dataset(dataset)
    cfg_path = HZ.generate_temp_config(horizon, variant, seed)
    primary_ckpt_dir = Path(HZ.ckpt_dir_for(horizon, variant, seed))
    expected_cfg = HZ.temp_cfg_path(horizon, variant, seed)
    if cfg_path.resolve() != expected_cfg.resolve():
        raise RuntimeError(
            f"Config path mismatch: generated {cfg_path} vs expected {expected_cfg}"
        )
    slug = HZ.DATASET_SLUG
    expected_chain = HZ.chain_lengths_for(horizon)
    ckpt_dir, ckpt_dir_warnings = resolve_checkpoint_dir(
        primary_ckpt_dir, slug, horizon, variant, seed
    )
    return {
        "dataset": dataset,
        "slug": slug,
        "horizon": int(horizon),
        "seed": int(seed),
        "variant": variant,
        "cfg_path": Path(cfg_path),
        "ckpt_dir": Path(ckpt_dir),
        "primary_ckpt_dir": primary_ckpt_dir,
        "alternate_ckpt_dir": results_ckpt_dir_for(slug, horizon, variant, seed),
        "ckpt_dir_warnings": ckpt_dir_warnings,
        "expected_chain_lengths": list(expected_chain),
        "expected_input_len": int(HZ.INPUT_LEN),
    }


def find_checkpoint_candidates(ckpt_dir: Path) -> list[Path]:
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory missing: {ckpt_dir}")
    files = sorted(
        p for p in ckpt_dir.rglob("*") if p.is_file() and p.suffix in CKPT_SUFFIXES
    )
    if not files:
        raise FileNotFoundError(
            f"No checkpoint files with suffixes {CKPT_SUFFIXES} under {ckpt_dir}"
        )
    return files


def select_checkpoint(ckpt_dir: Path) -> tuple[Path, list[Path], list[str]]:
    """Prefer an unambiguous best-val-MAE checkpoint; never silent mtime picks."""
    warnings: list[str] = []
    files = find_checkpoint_candidates(ckpt_dir)
    best = [
        p
        for p in files
        if "best_val_mae" in p.name.lower() or "best_val_MAE" in p.name
    ]
    # Also accept common BestTS naming that embeds best_val_MAE literally.
    best = sorted({p.resolve() for p in best})
    if len(best) == 1:
        return best[0], files, warnings
    if len(best) > 1:
        listing = "\n".join(f"  - {p}" for p in best)
        raise RuntimeError(
            "Multiple best-validation-MAE checkpoints found; refusing to guess:\n"
            f"{listing}"
        )

    # No explicit best_val_MAE: if exactly one ckpt overall, use it with warning.
    if len(files) == 1:
        warnings.append(
            f"No best_val_MAE checkpoint found; using sole candidate {files[0]}"
        )
        return files[0], files, warnings

    listing = "\n".join(f"  - {p}" for p in files)
    raise RuntimeError(
        "Could not uniquely identify a best-validation-MAE checkpoint. "
        "Candidates:\n"
        f"{listing}"
    )


def extract_state_dict(obj: Any) -> tuple[dict[str, torch.Tensor], str]:
    if isinstance(obj, dict):
        # Direct state-dict heuristics: tensor values and model-like keys.
        if obj and all(isinstance(v, torch.Tensor) for v in obj.values()):
            sample = next(iter(obj))
            if any(tok in str(sample) for tok in (".", "weight", "bias", "codebook")):
                return obj, "<root>"
        for key in STATE_DICT_KEYS:
            if key in obj and isinstance(obj[key], dict):
                nested = obj[key]
                if nested and all(isinstance(v, torch.Tensor) for v in nested.values()):
                    return nested, key
                # nested under 'model'/'net' may itself wrap a state_dict
                if isinstance(nested, dict):
                    for inner in STATE_DICT_KEYS:
                        if inner in nested and isinstance(nested[inner], dict):
                            cand = nested[inner]
                            if cand and all(
                                isinstance(v, torch.Tensor) for v in cand.values()
                            ):
                                return cand, f"{key}.{inner}"
        tensor_dicts = [
            (k, v)
            for k, v in obj.items()
            if isinstance(v, dict)
            and v
            and all(isinstance(t, torch.Tensor) for t in v.values())
        ]
        if len(tensor_dicts) == 1:
            return tensor_dicts[0][1], str(tensor_dicts[0][0])
        keys = list(obj.keys())
        raise RuntimeError(
            "Unable to locate a unique model state dictionary in checkpoint. "
            f"Top-level keys: {keys}"
        )
    raise RuntimeError(f"Unsupported checkpoint object type: {type(obj)}")


def maybe_strip_module_prefix(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    keys = list(state.keys())
    if keys and all(k.startswith("module.") for k in keys):
        return {k[len("module.") :]: v for k, v in state.items()}
    return state


def load_model_strict(model: torch.nn.Module, ckpt_path: Path) -> str:
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state, key = extract_state_dict(obj)
    state = maybe_strip_module_prefix(state)
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        # strict=True should already raise; keep explicit guard.
        raise RuntimeError(
            f"Strict load failed for {ckpt_path}: "
            f"missing={missing}, unexpected={unexpected}"
        )
    return key


def load_cfg(cfg_path: Path):
    return HZ.load_cfg(cfg_path)


def verify_cfg(cfg, *, horizon: int, variant: str, expected_chain: list[int]) -> list[str]:
    warnings: list[str] = []
    param = dict(cfg.MODEL.PARAM)
    in_len = int(cfg.DATASET_INPUT_LEN)
    out_len = int(cfg.DATASET_OUTPUT_LEN)
    model_in = int(param["input_len"])
    model_out = int(param["output_len"])
    chain = list(param.get("chain_lengths", []))
    if in_len != DEFAULT_INPUT_LEN or model_in != DEFAULT_INPUT_LEN:
        raise RuntimeError(
            f"Expected input length {DEFAULT_INPUT_LEN}, got dataset={in_len}, model={model_in}"
        )
    if out_len != horizon or model_out != horizon:
        raise RuntimeError(
            f"Expected horizon {horizon}, got dataset={out_len}, model={model_out}"
        )
    if chain != list(expected_chain):
        raise RuntimeError(
            f"Expected chain lengths {expected_chain}, got {chain}"
        )
    if not bool(param.get("use_forecast_state_adapter", False)):
        raise RuntimeError("Requested variant requires use_forecast_state_adapter=True")
    if str(param.get("forecast_state_adapter_mode", "")) != "condition_only":
        raise RuntimeError(
            "Requested final variant requires forecast_state_adapter_mode='condition_only'"
        )
    if str(param.get("spatial_placement", "")) != "interleaved_progressive":
        raise RuntimeError(
            "Requested variant requires spatial_placement='interleaved_progressive'"
        )
    if str(param.get("chain_loss_mode", "")) != "token_normalized":
        warnings.append(
            f"Unexpected chain_loss_mode={param.get('chain_loss_mode')!r} "
            "(expected token_normalized for the final variant)"
        )
    shuffle = bool(getattr(cfg.TEST.DATA, "SHUFFLE", False))
    if shuffle:
        raise RuntimeError("TEST.DATA.SHUFFLE must be False for qualitative export")
    data_dir = Path(cfg.TEST.DATA.DIR)
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    in_l = int(cfg.DATASET_INPUT_LEN)
    out_l = int(cfg.DATASET_OUTPUT_LEN)
    data_file = data_dir / f"data_in{in_l}_out{out_l}.pkl"
    index_file = data_dir / f"index_in{in_l}_out{out_l}.pkl"
    if not data_file.is_file() or not index_file.is_file():
        raise FileNotFoundError(
            f"Processed data split files missing under {data_dir}: "
            f"{data_file.name}, {index_file.name}"
        )
    # Soft check that variant name is reflected in MODEL.NAME when present.
    model_name = getattr(cfg.MODEL, "NAME", None)
    if model_name and "StateAdapterFixed" not in str(model_name):
        warnings.append(
            f"MODEL.NAME={model_name!r} does not contain StateAdapterFixed "
            f"(variant={variant})"
        )
    return warnings


def build_test_loader(cfg) -> tuple[DataLoader, Any, Path, Path]:
    from basicts.data import TimeSeriesForecastingDataset

    data_dir = Path(cfg.TEST.DATA.DIR)
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    in_l = int(cfg.DATASET_INPUT_LEN)
    out_l = int(cfg.DATASET_OUTPUT_LEN)
    data_file = data_dir / f"data_in{in_l}_out{out_l}.pkl"
    index_file = data_dir / f"index_in{in_l}_out{out_l}.pkl"
    dataset = TimeSeriesForecastingDataset(
        data_file_path=str(data_file),
        index_file_path=str(index_file),
        mode="test",
    )
    batch_size = int(cfg.TEST.DATA.BATCH_SIZE)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    return loader, dataset, data_file, index_file


def load_scaler(cfg) -> dict:
    data_dir = Path(cfg.TRAIN.DATA.DIR)
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    in_l = int(cfg.DATASET_INPUT_LEN)
    out_l = int(cfg.DATASET_OUTPUT_LEN)
    path = data_dir / f"scaler_in{in_l}_out{out_l}.pkl"
    if not path.is_file():
        raise FileNotFoundError(f"Scaler file missing: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def inverse_transform(tensor: torch.Tensor, scaler: dict) -> torch.Tensor:
    from basicts.runners.base_tsf_runner import SCALER_REGISTRY

    return SCALER_REGISTRY.get(scaler["func"])(tensor, **scaler["args"])


def select_features(data: torch.Tensor, features) -> torch.Tensor:
    if features is None:
        return data
    return data[:, :, :, list(features)]


def valid_mask(target: torch.Tensor, null_val: float) -> torch.Tensor:
    if isinstance(null_val, float) and math.isnan(null_val):
        return ~torch.isnan(target)
    eps = 5e-5
    ref = torch.as_tensor(null_val, device=target.device, dtype=target.dtype)
    return ~torch.isclose(target, ref.expand_as(target), atol=eps, rtol=0.0)


def masked_mae_per_sample(pred: torch.Tensor, target: torch.Tensor, null_val: float) -> np.ndarray:
    """Per-sample MAE with the same null-value mask convention as repository metrics."""
    # pred/target: [B, T, N, C]
    mask = valid_mask(target, null_val).to(dtype=pred.dtype)
    err = (pred - target).abs() * mask
    denom = mask.flatten(1).sum(dim=1).clamp_min(1.0)
    mae = err.flatten(1).sum(dim=1) / denom
    return mae.detach().cpu().numpy()


def masked_mae_per_node(pred: torch.Tensor, target: torch.Tensor, null_val: float) -> np.ndarray:
    """Mean absolute error per node averaged over samples and time. Shape [N]."""
    mask = valid_mask(target, null_val).to(dtype=pred.dtype)
    err = (pred - target).abs() * mask
    # sum over B,T,C
    num = err.sum(dim=(0, 1, 3))
    den = mask.sum(dim=(0, 1, 3)).clamp_min(1.0)
    return (num / den).detach().cpu().numpy()


class AdapterCapture:
    """Capture condition-only adapter outputs without modifying the forward path."""

    def __init__(self, module: torch.nn.Module):
        if module is None:
            raise RuntimeError(
                "model.forecast_state_adapter is missing; cannot capture adapter condition"
            )
        self.module = module
        self.captures: list[torch.Tensor] = []
        self._handle = None

    def _hook(self, _module, _inputs, output):
        # Do not modify inputs/outputs; store a detached CPU copy.
        if not torch.is_tensor(output):
            raise RuntimeError(
                f"Adapter hook expected a tensor output, got {type(output)}"
            )
        self.captures.append(output.detach().cpu().clone())
        return None

    def __enter__(self):
        self.captures.clear()
        self._handle = self.module.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        return False

    def pop_batch(self) -> torch.Tensor:
        if len(self.captures) != 1:
            raise RuntimeError(
                f"Expected exactly one adapter capture per inference batch, "
                f"got {len(self.captures)}"
            )
        return self.captures.pop()


def stack_or_fail(chunks: list[torch.Tensor], name: str) -> torch.Tensor:
    if not chunks:
        raise RuntimeError(f"No tensors collected for {name}")
    ranks = {t.ndim for t in chunks}
    if len(ranks) != 1:
        raise RuntimeError(f"{name}: inconsistent ranks {ranks}")
    shapes_tail = {tuple(t.shape[1:]) for t in chunks}
    if len(shapes_tail) != 1:
        raise RuntimeError(f"{name}: inconsistent non-batch shapes {shapes_tail}")
    return torch.cat(chunks, dim=0)


def to_numpy_btnc(t: torch.Tensor) -> np.ndarray:
    arr = t.detach().cpu().numpy()
    if arr.ndim != 4:
        raise RuntimeError(f"Expected rank-4 tensor [B,T,N,C], got shape {arr.shape}")
    return arr


def reconstruct_adjacencies(model) -> tuple[list[np.ndarray], list[dict]]:
    modules = list(getattr(model, "progressive_spatial_modules", []))
    if len(modules) != 3:
        raise RuntimeError(
            f"Expected 3 progressive_spatial_modules, got {len(modules)}"
        )
    adjs: list[np.ndarray] = []
    stats: list[dict] = []
    n_nodes = int(model.node_size)
    for stage_idx, module in enumerate(modules):
        if not hasattr(module, "get_adaptive_adj"):
            # Fallback: call the private builder used by adaptive_only.
            if hasattr(module, "_build_adaptive_adj"):
                adj_t = module._build_adaptive_adj().detach()
            else:
                raise RuntimeError(
                    f"Stage {stage_idx + 1} spatial module lacks adjacency construction"
                )
        else:
            adj_t = module.get_adaptive_adj()
            if adj_t is None:
                adj_t = module._build_adaptive_adj().detach()
        adj = adj_t.detach().cpu().numpy().astype(np.float64)
        if adj.shape != (n_nodes, n_nodes):
            raise RuntimeError(
                f"Stage {stage_idx + 1} adjacency shape {adj.shape} != {(n_nodes, n_nodes)}"
            )
        if not np.isfinite(adj).all():
            raise RuntimeError(f"Stage {stage_idx + 1} adjacency contains non-finite values")
        row_sums = adj.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-4):
            raise RuntimeError(
                f"Stage {stage_idx + 1} adjacency rows do not sum to ~1 "
                f"(mean={row_sums.mean():.6f}, "
                f"max_abs_err={float(np.max(np.abs(row_sums - 1.0))):.6e})"
            )
        topk = int(getattr(module, "adp_topk", -1))
        nnz = int((adj > 0).sum())
        expected_nnz = n_nodes * topk if topk > 0 else None
        # mask_topk keeps topk including possible self; nnz should be N*topk when topk < N
        ok_topk = expected_nnz is None or nnz == expected_nnz or topk >= n_nodes
        if not ok_topk and topk > 0:
            # allow tiny tolerance only when topk >= N (full dense after softmax)
            raise RuntimeError(
                f"Stage {stage_idx + 1} nnz={nnz} inconsistent with topk={topk} "
                f"(expected ~{expected_nnz})"
            )
        # row entropy
        with np.errstate(divide="ignore", invalid="ignore"):
            logp = np.where(adj > 0, np.log(adj + 1e-12), 0.0)
            ent = -(adj * logp).sum(axis=1)
        stats.append(
            {
                "stage": stage_idx + 1,
                "shape": list(adj.shape),
                "topk": topk,
                "nnz": nnz,
                "expected_nnz": expected_nnz,
                "row_sum_mean": float(row_sums.mean()),
                "row_sum_max_abs_err": float(np.max(np.abs(row_sums - 1.0))),
                "mean_row_entropy": float(ent.mean()),
                "finite": True,
            }
        )
        adjs.append(adj)
    return adjs, stats


def choose_sample_cases(final_mae: np.ndarray) -> dict:
    n = int(final_mae.shape[0])
    if n == 0:
        raise RuntimeError("Empty test set; cannot select cases")
    order = np.arange(n)
    median = float(np.median(final_mae))
    p90 = float(np.quantile(final_mae, 0.90))

    def closest(target: float) -> int:
        dist = np.abs(final_mae - target)
        # deterministic tie-break: lowest sample index
        best = float(dist.min())
        candidates = order[np.isclose(dist, best) | (dist == best)]
        # use exact min distance set
        candidates = order[dist <= best + 1e-12]
        return int(candidates.min())

    typical = closest(median)
    difficult = closest(p90)

    thresh = float(np.quantile(final_mae, 0.90))
    # highest 10%: mae >= p90
    high_mask = final_mae >= thresh - 1e-12
    high_idx = order[high_mask]
    if high_idx.size == 0:
        raise RuntimeError("No samples in the highest 10% final MAE band")
    return {
        "typical_index": typical,
        "difficult_index": difficult,
        "high_band_indices": high_idx.tolist(),
        "median_final_mae": median,
        "p90_final_mae": p90,
        "n_samples": n,
        "n_high_band": int(high_idx.size),
    }


def choose_failure_case(
    final_mae: np.ndarray,
    stage2_aligned_mae: np.ndarray,
    high_band_indices: list[int],
) -> int:
    high = np.asarray(high_band_indices, dtype=np.int64)
    delta = final_mae[high] - stage2_aligned_mae[high]
    best = float(delta.max())
    winners = high[delta >= best - 1e-12]
    return int(winners.min())


def choose_nodes(node_mae: np.ndarray) -> dict:
    n = int(node_mae.shape[0])
    order = np.arange(n)
    worst_val = float(node_mae.max())
    best_val = float(node_mae.min())
    median_target = float(np.median(node_mae))

    def pick(mask_or_dist, mode: str) -> int:
        if mode == "max":
            candidates = order[np.isclose(node_mae, worst_val) | (node_mae == worst_val)]
            return int(candidates.min())
        if mode == "min":
            candidates = order[np.isclose(node_mae, best_val) | (node_mae == best_val)]
            return int(candidates.min())
        dist = np.abs(node_mae - median_target)
        best_d = float(dist.min())
        candidates = order[dist <= best_d + 1e-12]
        return int(candidates.min())

    primary = pick(None, "max")
    median_node = pick(None, "median")
    best_node = pick(None, "min")
    return {
        "primary_node": primary,
        "median_node": median_node,
        "best_node": best_node,
        "primary_node_mae": float(node_mae[primary]),
        "median_node_mae": float(node_mae[median_node]),
        "best_node_mae": float(node_mae[best_node]),
        "rule": (
            "primary=largest mean abs final-stage error over full test set; "
            "also record median-node and smallest-error node; ties -> lowest node index"
        ),
    }


def save_figure(fig: plt.Figure, out_stem: Path) -> list[str]:
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    png = out_stem.with_suffix(".png")
    pdf = out_stem.with_suffix(".pdf")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [png.name, pdf.name]


def plot_forecast_evolution(
    out_dir: Path,
    dataset: str,
    history: np.ndarray,
    full_target: np.ndarray,
    stage1_al: np.ndarray,
    stage2_al: np.ndarray,
    adapter_al: np.ndarray,
    final_pred: np.ndarray,
    cases: dict,
    node: int,
) -> list[str]:
    # arrays: [T] after node select
    labels = [
        ("typical", cases["typical_index"]),
        ("difficult", cases["difficult_index"]),
        ("failure", cases["failure_index"]),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 9.5), sharex=False)
    in_len = history.shape[1]
    horizon = full_target.shape[1]
    hist_t = np.arange(-in_len + 1, 1)
    fut_t = np.arange(1, horizon + 1)
    for ax, (name, idx) in zip(axes, labels):
        h = history[idx, :, node, 0]
        y = full_target[idx, :, node, 0]
        s1 = stage1_al[idx, :, node, 0]
        s2 = stage2_al[idx, :, node, 0]
        ad = adapter_al[idx, :, node, 0]
        fp = final_pred[idx, :, node, 0]
        ax.plot(hist_t, h, color="#1f4e79", lw=2.0, label="Observed history")
        ax.axvline(0.0, color="#666666", ls="--", lw=1.0)
        ax.plot(fut_t, y, color="#222222", lw=2.0, label="Ground truth")
        ax.plot(fut_t, s1, color="#7a5195", lw=1.6, ls="-.", label="Stage-1 aligned (H=3→12)")
        ax.plot(fut_t, s2, color="#ef5675", lw=1.6, ls="--", label="Stage-2 aligned (H=6→12)")
        ax.plot(fut_t, ad, color="#ffa600", lw=1.6, ls=":", label="Adapter condition aligned")
        ax.plot(fut_t, fp, color="#003f5c", lw=2.0, label="Final forecast")
        # Do not connect history endpoint to future predictions.
        ax.set_title(
            f"{dataset} | {name} sample={idx} | node={node}",
            fontsize=11,
        )
        ax.set_ylabel("Target")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper left", fontsize=8, ncol=2)
    axes[-1].set_xlabel("Relative time (0 = last observed step)")
    fig.suptitle(
        f"{dataset}: forecast evolution (primary node {node})",
        fontsize=13,
        y=0.995,
    )
    fig.tight_layout()
    return save_figure(fig, out_dir / "forecast_evolution")


def plot_stage_error_maps(
    out_dir: Path,
    dataset: str,
    stage_preds: list[np.ndarray],
    stage_targets: list[np.ndarray],
) -> list[str]:
    # mean abs error over samples -> [T, N]
    maps = []
    for pred, tgt in zip(stage_preds, stage_targets):
        maps.append(np.mean(np.abs(pred[..., 0] - tgt[..., 0]), axis=0))
    vmax = max(float(m.max()) for m in maps)
    vmin = 0.0
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))
    for i, (ax, m) in enumerate(zip(axes, maps)):
        im = ax.imshow(
            m.T,
            aspect="auto",
            origin="lower",
            cmap="magma",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        ax.set_title(f"Stage {i + 1} ({m.shape[0]} steps × {m.shape[1]} nodes)")
        ax.set_xlabel("Forecast step")
        ax.set_ylabel("Node index")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(
        f"{dataset}: stage-wise mean |error| (shared color scale; "
        "do not compare panel area across stages)",
        fontsize=12,
    )
    fig.tight_layout()
    return save_figure(fig, out_dir / "stage_error_maps")


def plot_spatial_relations(
    out_dir: Path,
    dataset: str,
    adjs: list[np.ndarray],
    adj_stats: list[dict],
    nodes: dict,
) -> tuple[list[str], dict]:
    n = adjs[0].shape[0]
    # For large graphs, show a deterministic submatrix that includes representatives.
    max_show = 64
    reps = [nodes["primary_node"], nodes["median_node"], nodes["best_node"]]
    if n > max_show:
        chosen = []
        for r in reps:
            if r not in chosen:
                chosen.append(int(r))
        for i in range(n):
            if len(chosen) >= max_show:
                break
            if i not in chosen:
                chosen.append(i)
        show_idx = np.array(sorted(chosen), dtype=np.int64)
        submatrix_note = (
            f"Publication heatmap shows a deterministic {len(show_idx)}×{len(show_idx)} "
            f"submatrix including representative nodes {reps}, filled by lowest node "
            f"indices; full NxN matrices are saved as .npy (N={n})."
        )
    else:
        show_idx = np.arange(n, dtype=np.int64)
        submatrix_note = f"Heatmaps show the full {n}×{n} adaptive adjacency."

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4))
    for i, (ax, adj, st) in enumerate(zip(axes, adjs, adj_stats)):
        sub = adj[np.ix_(show_idx, show_idx)]
        im = ax.imshow(sub, aspect="equal", origin="upper", cmap="viridis")
        ax.set_title(
            f"Stage {i + 1}\nH_row={st['mean_row_entropy']:.3f}, topk={st['topk']}",
            fontsize=10,
        )
        ax.set_xlabel("Destination node (subset idx)")
        ax.set_ylabel("Source node (subset idx)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(
        f"{dataset}: learned adaptive adjacency (no geographic claim)\n{submatrix_note}",
        fontsize=11,
    )
    fig.tight_layout()
    files = save_figure(fig, out_dir / "spatial_relations")

    # Neighbor lists + stage overlap for representative nodes
    neighbor_payload = {
        "submatrix_node_indices": show_idx.tolist(),
        "submatrix_note": submatrix_note,
        "nodes": {},
        "stage_overlap": {},
    }
    for label, node in [
        ("primary", nodes["primary_node"]),
        ("median", nodes["median_node"]),
        ("best", nodes["best_node"]),
    ]:
        stage_neighbors = []
        for adj, st in zip(adjs, adj_stats):
            row = adj[node]
            # exclude self for neighbor list display; keep top by weight
            idxs = np.argsort(-row)
            neigh = []
            for j in idxs:
                j = int(j)
                if j == node:
                    continue
                if row[j] <= 0:
                    break
                neigh.append({"node": j, "weight": float(row[j])})
                if len(neigh) >= max(1, int(st["topk"]) - 1):
                    break
            stage_neighbors.append(neigh)
        neighbor_payload["nodes"][label] = {
            "node_id": int(node),
            "neighbors_by_stage": stage_neighbors,
        }
        # overlap of retained neighbor sets across consecutive stages
        sets = [set(x["node"] for x in neigh) for neigh in stage_neighbors]
        neighbor_payload["stage_overlap"][label] = {
            "stage1_stage2": len(sets[0] & sets[1]),
            "stage2_stage3": len(sets[1] & sets[2]),
            "stage1_stage3": len(sets[0] & sets[2]),
            "stage1_size": len(sets[0]),
            "stage2_size": len(sets[1]),
            "stage3_size": len(sets[2]),
        }

    # CSV stats
    csv_path = out_dir / "spatial_relation_statistics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "stage",
                "topk",
                "nnz",
                "expected_nnz",
                "row_sum_mean",
                "row_sum_max_abs_err",
                "mean_row_entropy",
            ],
        )
        writer.writeheader()
        for st in adj_stats:
            writer.writerow({k: st[k] for k in writer.fieldnames})
    write_json(out_dir / "representative_neighbors.json", neighbor_payload)
    files.extend(
        [
            "spatial_relation_statistics.csv",
            "representative_neighbors.json",
            "adjacency_stage1.npy",
            "adjacency_stage2.npy",
            "adjacency_stage3.npy",
        ]
    )
    return files, neighbor_payload


def plot_failure_case(
    out_dir: Path,
    dataset: str,
    stage1_err: np.ndarray,
    stage2_err: np.ndarray,
    adapter_err: np.ndarray,
    final_err: np.ndarray,
    failure_idx: int,
    nodes: dict,
) -> list[str]:
    # Line plots for representative nodes; shared y-scale
    node_ids = [nodes["primary_node"], nodes["median_node"], nodes["best_node"]]
    series = [
        ("Stage-1 aligned |err|", stage1_err[failure_idx, :, :, 0]),
        ("Stage-2 aligned |err|", stage2_err[failure_idx, :, :, 0]),
        ("Adapter condition aligned |err|", adapter_err[failure_idx, :, :, 0]),
        ("Final |err|", final_err[failure_idx, :, :, 0]),
    ]
    vmax = max(float(arr[:, node_ids].max()) for _, arr in series)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True, sharey=True)
    fut_t = np.arange(1, series[0][1].shape[0] + 1)
    colors = ["#003f5c", "#7a5195", "#ef5675"]
    for ax, (title, err) in zip(axes.ravel(), series):
        for c, nid in zip(colors, node_ids):
            ax.plot(fut_t, err[:, nid], color=c, lw=1.8, label=f"node {nid}")
        ax.set_ylim(0, vmax * 1.05 if vmax > 0 else 1.0)
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    axes[1, 0].set_xlabel("Forecast step")
    axes[1, 1].set_xlabel("Forecast step")
    axes[0, 0].set_ylabel("Absolute error")
    axes[1, 0].set_ylabel("Absolute error")
    fig.suptitle(
        f"{dataset}: failure sample {failure_idx} absolute errors "
        f"(shared scale; nodes={node_ids})",
        fontsize=12,
    )
    fig.tight_layout()
    return save_figure(fig, out_dir / "failure_case")


def run_inference(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    dataset,
    cfg,
    scaler: dict,
    device: torch.device,
) -> dict:
    from basicts.archs.arch_zoo.ChainForecasting_arch import ChainForecasting
    from basicts.archs.arch_zoo.ChainForecasting_arch.kasa_temporal_step import (
        interpolate_forecast,
    )

    forward_features = list(cfg.MODEL.FORWARD_FEATURES)
    target_features = list(cfg.MODEL.TARGET_FEATURES)
    null_val = float(getattr(cfg.TRAIN, "NULL_VAL", getattr(cfg, "NULL_VAL", 0.0)))
    chain_lengths = list(cfg.MODEL.PARAM["chain_lengths"])
    horizon = int(cfg.DATASET_OUTPUT_LEN)

    model.eval()
    if model.forecast_state_adapter is None:
        raise RuntimeError("model.forecast_state_adapter is None")

    buckets: dict[str, list[torch.Tensor]] = {
        "history_target": [],
        "full_target": [],
        "stage1_target": [],
        "stage2_target": [],
        "stage3_target": [],
        "stage1_pred": [],
        "stage2_pred": [],
        "stage3_pred": [],
        "final_pred": [],
        "stage1_aligned_to_horizon": [],
        "stage2_aligned_to_horizon": [],
        "stage2_adapter_condition": [],
        "stage2_adapter_condition_aligned_to_horizon": [],
        "temporal_s1": [],
        "temporal_s2": [],
        "temporal_s3": [],
        "spatial_s1": [],
        "spatial_s2": [],
        "spatial_s3": [],
    }
    sample_indices: list[int] = []
    global_indices: list[int] = []
    n_batches = 0
    adapter_captures = 0

    with AdapterCapture(model.forecast_state_adapter) as hook:
        with torch.no_grad():
            sample_cursor = 0
            for future_data, history_data in loader:
                n_batches += 1
                bsz = int(future_data.shape[0])
                history_data = history_data.to(device)
                future_data = future_data.to(device)
                hist_in = select_features(history_data, forward_features)
                fut_tgt = select_features(future_data, target_features)
                hist_tgt = select_features(history_data, target_features)

                out = model(
                    history_data=hist_in,
                    future_data=None,
                    train=False,
                    return_intermediates=True,
                )
                if not isinstance(out, dict):
                    raise RuntimeError("Expected dict from return_intermediates=True")

                chain = list(out["chain_preds"])
                temporal = list(out["temporal_preds"])
                spatial = list(out["spatial_preds"])
                if len(chain) != 3 or len(temporal) != 3 or len(spatial) != 3:
                    raise RuntimeError(
                        f"Expected 3 stage tensors, got chain={len(chain)}, "
                        f"temporal={len(temporal)}, spatial={len(spatial)}"
                    )
                for i, (c, L) in enumerate(zip(chain, chain_lengths)):
                    if int(c.shape[1]) != int(L):
                        raise RuntimeError(
                            f"Stage {i + 1} chain length {c.shape[1]} != {L}"
                        )

                adapter_cond = hook.pop_batch().to(device)
                adapter_captures += 1
                if int(adapter_cond.shape[1]) != int(chain_lengths[1]):
                    raise RuntimeError(
                        f"Adapter condition length {adapter_cond.shape[1]} "
                        f"!= stage-2 length {chain_lengths[1]}"
                    )

                y_final = out["pred"]
                # Stage targets via repository pooling
                t1 = ChainForecasting.pool_target(fut_tgt, chain_lengths[0])
                t2 = ChainForecasting.pool_target(fut_tgt, chain_lengths[1])
                t3 = ChainForecasting.pool_target(fut_tgt, chain_lengths[2])

                # Align stage-1/2 and adapter condition to horizon
                s1_al = interpolate_forecast(chain[0], horizon)
                s2_al = interpolate_forecast(chain[1], horizon)
                ad_al = interpolate_forecast(adapter_cond, horizon)

                # Inverse-transform target-related tensors only (channel already target)
                def inv(x: torch.Tensor) -> torch.Tensor:
                    return inverse_transform(x, scaler)

                buckets["history_target"].append(inv(hist_tgt).cpu())
                buckets["full_target"].append(inv(fut_tgt).cpu())
                buckets["stage1_target"].append(inv(t1).cpu())
                buckets["stage2_target"].append(inv(t2).cpu())
                buckets["stage3_target"].append(inv(t3).cpu())
                buckets["stage1_pred"].append(inv(chain[0]).cpu())
                buckets["stage2_pred"].append(inv(chain[1]).cpu())
                buckets["stage3_pred"].append(inv(chain[2]).cpu())
                buckets["final_pred"].append(inv(y_final).cpu())
                buckets["stage1_aligned_to_horizon"].append(inv(s1_al).cpu())
                buckets["stage2_aligned_to_horizon"].append(inv(s2_al).cpu())
                buckets["stage2_adapter_condition"].append(inv(adapter_cond).cpu())
                buckets["stage2_adapter_condition_aligned_to_horizon"].append(
                    inv(ad_al).cpu()
                )
                buckets["temporal_s1"].append(inv(temporal[0]).cpu())
                buckets["temporal_s2"].append(inv(temporal[1]).cpu())
                buckets["temporal_s3"].append(inv(temporal[2]).cpu())
                buckets["spatial_s1"].append(inv(spatial[0]).cpu())
                buckets["spatial_s2"].append(inv(spatial[1]).cpu())
                buckets["spatial_s3"].append(inv(spatial[2]).cpu())

                for i in range(bsz):
                    local = sample_cursor + i
                    sample_indices.append(local)
                    # Original/global index from dataset index table when available.
                    try:
                        idx_tuple = dataset.index[local]
                        if isinstance(idx_tuple, (list, tuple)) and len(idx_tuple) >= 2:
                            # Prefer future-start / current-time index.
                            g = idx_tuple[1]
                            if isinstance(g, (int, np.integer)):
                                global_indices.append(int(g))
                            else:
                                global_indices.append(local)
                        else:
                            global_indices.append(local)
                    except Exception:
                        global_indices.append(local)
                sample_cursor += bsz

    if adapter_captures != n_batches:
        raise RuntimeError(
            f"Adapter hook capture count {adapter_captures} != batch count {n_batches}"
        )

    stacked = {k: stack_or_fail(v, k) for k, v in buckets.items()}
    n = stacked["final_pred"].shape[0]
    if n != len(sample_indices):
        raise RuntimeError("Sample index count mismatch after stacking")

    # Metrics on physical scale
    final = stacked["final_pred"]
    full = stacked["full_target"]
    s1a = stacked["stage1_aligned_to_horizon"]
    s2a = stacked["stage2_aligned_to_horizon"]
    ada = stacked["stage2_adapter_condition_aligned_to_horizon"]

    final_mae = masked_mae_per_sample(final, full, null_val)
    s1_mae = masked_mae_per_sample(s1a, full, null_val)
    s2_mae = masked_mae_per_sample(s2a, full, null_val)
    ad_mae = masked_mae_per_sample(ada, full, null_val)
    final_minus_s2 = masked_mae_per_sample(final, s2a, null_val)
    # Spec: final_minus_stage2_mae means final_stage_mae - aligned_stage2_mae (delta of MAEs)
    # and also listed as a metric name. Provide both the MAE-of-difference series name as
    # abs-error MAE between final and stage2, AND the selection uses mae delta.
    # Keeping named fields as MAE of each series vs GT, plus explicit delta columns.
    final_minus_s2_mae_delta = final_mae - s2_mae
    final_minus_ad_mae_delta = final_mae - ad_mae
    # Also compute MAE between final and adapter predictions for completeness.
    final_vs_ad_mae = masked_mae_per_sample(final, ada, null_val)

    node_mae = masked_mae_per_node(final, full, null_val)
    case_meta = choose_sample_cases(final_mae)
    failure_idx = choose_failure_case(
        final_mae, s2_mae, case_meta["high_band_indices"]
    )
    case_meta["failure_index"] = failure_idx
    nodes = choose_nodes(node_mae)

    return {
        "stacked": stacked,
        "sample_index": np.asarray(sample_indices, dtype=np.int64),
        "global_sample_index": np.asarray(global_indices, dtype=np.int64),
        "metrics": {
            "stage1_aligned_mae": s1_mae,
            "stage2_aligned_mae": s2_mae,
            "adapter_condition_aligned_mae": ad_mae,
            "final_mae": final_mae,
            "final_minus_stage2_mae": final_minus_s2_mae_delta,
            "final_minus_adapter_condition_mae": final_minus_ad_mae_delta,
            "final_vs_stage2_pred_mae": final_minus_s2,
            "final_vs_adapter_pred_mae": final_vs_ad_mae,
        },
        "node_mae": node_mae,
        "cases": case_meta,
        "nodes": nodes,
        "n_batches": n_batches,
        "adapter_captures": adapter_captures,
        "null_val": null_val,
    }


def export_dataset(
    *,
    dataset: str,
    horizon: int,
    seed: int,
    variant: str,
    gpu: int,
    output_root: Path,
    dry_run: bool,
) -> dict:
    warnings: list[str] = []
    meta = resolve_experiment(dataset, horizon, variant, seed)
    warnings.extend(list(meta.get("ckpt_dir_warnings") or []))
    out_dir = output_root / f"{meta['slug']}_h{horizon}_seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    record: dict[str, Any] = {
        "dataset": dataset,
        "slug": meta["slug"],
        "horizon": horizon,
        "seed": seed,
        "variant": variant,
        "config_path": str(meta["cfg_path"]),
        "checkpoint_directory": str(meta["ckpt_dir"]),
        "primary_checkpoint_directory": str(meta["primary_ckpt_dir"]),
        "alternate_checkpoint_directory": str(meta["alternate_ckpt_dir"]),
        "output_directory": str(out_dir),
        "expected_chain_lengths": meta["expected_chain_lengths"],
        "status": "pending",
        "generated_files": [],
        "warnings": [],
    }

    ckpt_path, candidates, ckpt_warnings = select_checkpoint(meta["ckpt_dir"])
    warnings.extend(ckpt_warnings)
    record["selected_checkpoint"] = str(ckpt_path)
    record["checkpoint_candidates"] = [str(p) for p in candidates]

    cfg = load_cfg(meta["cfg_path"])
    cfg_warnings = verify_cfg(
        cfg,
        horizon=horizon,
        variant=variant,
        expected_chain=meta["expected_chain_lengths"],
    )
    warnings.extend(cfg_warnings)
    record["observed_chain_lengths"] = list(cfg.MODEL.PARAM["chain_lengths"])
    record["input_length"] = int(cfg.DATASET_INPUT_LEN)

    if dry_run:
        record["status"] = "dry_run"
        record["warnings"] = warnings
        record["test_dataset_size"] = None
        record["note"] = (
            "Dry-run: configs/checkpoints resolved; no inference, no figures."
        )
        write_json(out_dir / "manifest.json", {**record, "dry_run": True})
        record["generated_files"] = ["manifest.json"]
        return record

    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        warnings.append("CUDA unavailable; running on CPU")

    model = cfg.MODEL.ARCH(**dict(cfg.MODEL.PARAM))
    state_key = load_model_strict(model, ckpt_path)
    model.to(device)
    model.eval()

    loader, ds, data_file, index_file = build_test_loader(cfg)
    scaler = load_scaler(cfg)

    print(EVAL_BANNER)
    print(
        f"[{dataset}] EVAL start | cfg={meta['cfg_path']} | ckpt={ckpt_path} | "
        f"test_n={len(ds)} | device={device}"
    )

    result = run_inference(
        model=model,
        loader=loader,
        dataset=ds,
        cfg=cfg,
        scaler=scaler,
        device=device,
    )
    stacked = result["stacked"]
    metrics = result["metrics"]
    cases = result["cases"]
    nodes = result["nodes"]

    # forecasts.npz
    temporal_predictions = np.empty(3, dtype=object)
    spatial_predictions = np.empty(3, dtype=object)
    temporal_predictions[0] = to_numpy_btnc(stacked["temporal_s1"])
    temporal_predictions[1] = to_numpy_btnc(stacked["temporal_s2"])
    temporal_predictions[2] = to_numpy_btnc(stacked["temporal_s3"])
    spatial_predictions[0] = to_numpy_btnc(stacked["spatial_s1"])
    spatial_predictions[1] = to_numpy_btnc(stacked["spatial_s2"])
    spatial_predictions[2] = to_numpy_btnc(stacked["spatial_s3"])

    npz_path = out_dir / "forecasts.npz"
    np.savez_compressed(
        npz_path,
        sample_index=result["sample_index"],
        global_sample_index=result["global_sample_index"],
        history_target=to_numpy_btnc(stacked["history_target"]),
        full_target=to_numpy_btnc(stacked["full_target"]),
        stage1_target=to_numpy_btnc(stacked["stage1_target"]),
        stage2_target=to_numpy_btnc(stacked["stage2_target"]),
        stage3_target=to_numpy_btnc(stacked["stage3_target"]),
        stage1_pred=to_numpy_btnc(stacked["stage1_pred"]),
        stage2_pred=to_numpy_btnc(stacked["stage2_pred"]),
        stage3_pred=to_numpy_btnc(stacked["stage3_pred"]),
        final_pred=to_numpy_btnc(stacked["final_pred"]),
        stage1_aligned_to_horizon=to_numpy_btnc(stacked["stage1_aligned_to_horizon"]),
        stage2_aligned_to_horizon=to_numpy_btnc(stacked["stage2_aligned_to_horizon"]),
        stage2_adapter_condition=to_numpy_btnc(stacked["stage2_adapter_condition"]),
        stage2_adapter_condition_aligned_to_horizon=to_numpy_btnc(
            stacked["stage2_adapter_condition_aligned_to_horizon"]
        ),
        temporal_predictions=temporal_predictions,
        spatial_predictions=spatial_predictions,
    )
    meta_json = {
        "dataset": dataset,
        "horizon": horizon,
        "seed": seed,
        "variant": variant,
        "checkpoint_path": str(ckpt_path),
        "config_path": str(meta["cfg_path"]),
        "data_file": str(data_file),
        "index_file": str(index_file),
        "tensor_shapes": {
            k: list(to_numpy_btnc(stacked[k]).shape)
            for k in [
                "history_target",
                "full_target",
                "stage1_pred",
                "stage2_pred",
                "stage3_pred",
                "final_pred",
                "stage2_adapter_condition",
            ]
        },
    }
    write_json(out_dir / "forecasts_meta.json", meta_json)

    # sample_metrics.csv
    metrics_path = out_dir / "sample_metrics.csv"
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "sample_index",
            "global_sample_index",
            "stage1_aligned_mae",
            "stage2_aligned_mae",
            "adapter_condition_aligned_mae",
            "final_mae",
            "final_minus_stage2_mae",
            "final_minus_adapter_condition_mae",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(result["sample_index"])):
            writer.writerow(
                {
                    "sample_index": int(result["sample_index"][i]),
                    "global_sample_index": int(result["global_sample_index"][i]),
                    "stage1_aligned_mae": float(metrics["stage1_aligned_mae"][i]),
                    "stage2_aligned_mae": float(metrics["stage2_aligned_mae"][i]),
                    "adapter_condition_aligned_mae": float(
                        metrics["adapter_condition_aligned_mae"][i]
                    ),
                    "final_mae": float(metrics["final_mae"][i]),
                    "final_minus_stage2_mae": float(
                        metrics["final_minus_stage2_mae"][i]
                    ),
                    "final_minus_adapter_condition_mae": float(
                        metrics["final_minus_adapter_condition_mae"][i]
                    ),
                }
            )

    selected = {
        "dataset": dataset,
        "horizon": horizon,
        "seed": seed,
        "variant": variant,
        "checkpoint_path": str(ckpt_path),
        "config_path": str(meta["cfg_path"]),
        "ranking_rules": {
            "typical": "final MAE closest to median final MAE",
            "difficult": "final MAE closest to 90th percentile",
            "failure": (
                "among highest 10% final MAE, maximize "
                "(final_stage_mae - aligned_stage2_mae)"
            ),
            "tie_break": "lowest dataset-relative sample index",
        },
        "target_quantiles": {
            "median_final_mae": cases["median_final_mae"],
            "p90_final_mae": cases["p90_final_mae"],
        },
        "n_candidate_samples": cases["n_samples"],
        "n_failure_band_candidates": cases["n_high_band"],
        "selected_sample_indices": {
            "typical": cases["typical_index"],
            "difficult": cases["difficult_index"],
            "failure": cases["failure_index"],
        },
        "selected_errors": {
            "typical_final_mae": float(metrics["final_mae"][cases["typical_index"]]),
            "difficult_final_mae": float(
                metrics["final_mae"][cases["difficult_index"]]
            ),
            "failure_final_mae": float(metrics["final_mae"][cases["failure_index"]]),
            "failure_stage2_aligned_mae": float(
                metrics["stage2_aligned_mae"][cases["failure_index"]]
            ),
            "failure_final_minus_stage2_mae": float(
                metrics["final_minus_stage2_mae"][cases["failure_index"]]
            ),
        },
        "selected_nodes": nodes,
        "node_selection_rule": nodes["rule"],
    }
    write_json(out_dir / "selected_cases.json", selected)

    # Adjacency
    adjs, adj_stats = reconstruct_adjacencies(model)
    for i, adj in enumerate(adjs, start=1):
        np.save(out_dir / f"adjacency_stage{i}.npy", adj)

    hist = to_numpy_btnc(stacked["history_target"])
    full = to_numpy_btnc(stacked["full_target"])
    s1a = to_numpy_btnc(stacked["stage1_aligned_to_horizon"])
    s2a = to_numpy_btnc(stacked["stage2_aligned_to_horizon"])
    ada = to_numpy_btnc(stacked["stage2_adapter_condition_aligned_to_horizon"])
    final = to_numpy_btnc(stacked["final_pred"])

    fig_files: list[str] = []
    fig_files += plot_forecast_evolution(
        out_dir,
        dataset,
        hist,
        full,
        s1a,
        s2a,
        ada,
        final,
        {
            "typical_index": cases["typical_index"],
            "difficult_index": cases["difficult_index"],
            "failure_index": cases["failure_index"],
        },
        nodes["primary_node"],
    )
    fig_files += plot_stage_error_maps(
        out_dir,
        dataset,
        [
            to_numpy_btnc(stacked["stage1_pred"]),
            to_numpy_btnc(stacked["stage2_pred"]),
            to_numpy_btnc(stacked["stage3_pred"]),
        ],
        [
            to_numpy_btnc(stacked["stage1_target"]),
            to_numpy_btnc(stacked["stage2_target"]),
            to_numpy_btnc(stacked["stage3_target"]),
        ],
    )
    spatial_files, _ = plot_spatial_relations(
        out_dir, dataset, adjs, adj_stats, nodes
    )
    fig_files += [f for f in spatial_files if f.endswith((".png", ".pdf"))]
    fig_files += plot_failure_case(
        out_dir,
        dataset,
        np.abs(s1a - full),
        np.abs(s2a - full),
        np.abs(ada - full),
        np.abs(final - full),
        cases["failure_index"],
        nodes,
    )

    generated = [
        "forecasts.npz",
        "forecasts_meta.json",
        "sample_metrics.csv",
        "selected_cases.json",
        "spatial_relation_statistics.csv",
        "representative_neighbors.json",
        "adjacency_stage1.npy",
        "adjacency_stage2.npy",
        "adjacency_stage3.npy",
        "manifest.json",
        *fig_files,
    ]
    # dedupe preserving order
    seen = set()
    generated_unique = []
    for g in generated:
        if g not in seen:
            seen.add(g)
            generated_unique.append(g)

    tensor_shapes = {
        k: list(to_numpy_btnc(stacked[k]).shape)
        for k in [
            "history_target",
            "full_target",
            "final_pred",
            "stage1_pred",
            "stage2_pred",
            "stage3_pred",
            "stage2_adapter_condition",
            "stage1_aligned_to_horizon",
            "stage2_aligned_to_horizon",
            "stage2_adapter_condition_aligned_to_horizon",
        ]
    }
    manifest = {
        "dataset": dataset,
        "horizon": horizon,
        "input_length": int(cfg.DATASET_INPUT_LEN),
        "seed": seed,
        "variant": variant,
        "expected_chain_lengths": meta["expected_chain_lengths"],
        "observed_chain_lengths": list(cfg.MODEL.PARAM["chain_lengths"]),
        "config_path": str(meta["cfg_path"]),
        "checkpoint_directory": str(meta["ckpt_dir"]),
        "selected_checkpoint": str(ckpt_path),
        "checkpoint_state_dict_key": state_key,
        "test_dataset_size": int(len(ds)),
        "number_of_inference_batches": int(result["n_batches"]),
        "device": str(device),
        "tensor_shapes": tensor_shapes,
        "selected_sample_indices": selected["selected_sample_indices"],
        "selected_node_indices": {
            "primary": nodes["primary_node"],
            "median": nodes["median_node"],
            "best": nodes["best_node"],
        },
        "output_files": generated_unique,
        "adjacency_validation_statistics": adj_stats,
        "adapter_hook_one_capture_per_batch": bool(
            result["adapter_captures"] == result["n_batches"]
        ),
        "adapter_hook_capture_count": int(result["adapter_captures"]),
        "adapter_hook_expected_batches": int(result["n_batches"]),
        "warnings": warnings,
        "unsupported_optional_fields": [],
        "data_file": str(data_file),
        "index_file": str(index_file),
        "null_val": result["null_val"],
    }
    write_json(out_dir / "manifest.json", manifest)

    record.update(
        {
            "status": "success",
            "selected_checkpoint": str(ckpt_path),
            "checkpoint_state_dict_key": state_key,
            "test_sample_count": int(len(ds)),
            "number_of_inference_batches": int(result["n_batches"]),
            "tensor_shapes": tensor_shapes,
            "selected_cases": selected["selected_sample_indices"],
            "selected_nodes": {
                "primary": nodes["primary_node"],
                "median": nodes["median_node"],
                "best": nodes["best_node"],
            },
            "generated_files": generated_unique,
            "adapter_hook_capture_count": int(result["adapter_captures"]),
            "adapter_hook_expected_batches": int(result["n_batches"]),
            "warnings": warnings,
            "device": str(device),
        }
    )
    return record


def print_console_manifest(records: list[dict], summary_path: Path) -> None:
    print("\n" + "=" * 72)
    print("QUALITATIVE EXPORT MANIFEST")
    print("=" * 72)
    for rec in records:
        print(f"\nDataset: {rec.get('dataset')}  [{rec.get('status')}]")
        print(f"  config: {rec.get('config_path')}")
        print(f"  checkpoint_dir: {rec.get('checkpoint_directory')}")
        print(f"  selected_checkpoint: {rec.get('selected_checkpoint')}")
        print(f"  test_split_size: {rec.get('test_sample_count')}")
        print(f"  n_batches: {rec.get('number_of_inference_batches')}")
        print(f"  tensor_shapes: {rec.get('tensor_shapes')}")
        print(f"  selected_samples: {rec.get('selected_cases')}")
        print(f"  selected_nodes: {rec.get('selected_nodes')}")
        print(f"  output_dir: {rec.get('output_directory')}")
        print(f"  generated_files: {rec.get('generated_files')}")
        print(
            "  adapter_hook: "
            f"{rec.get('adapter_hook_capture_count')}/"
            f"{rec.get('adapter_hook_expected_batches')}"
        )
        if rec.get("error"):
            print(f"  error: {rec['error']}")
        if rec.get("warnings"):
            print(f"  warnings: {rec['warnings']}")
    print(f"\nGlobal summary: {summary_path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluation-only F2FNet qualitative export"
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        choices=sorted(HZ.DATASET_SPECS.keys()),
        help="Datasets to export (processed independently)",
    )
    p.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--variant", type=str, default=DEFAULT_VARIANT)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "results" / "qualitative",
    )
    p.add_argument(
        "--fail-fast",
        action="store_true",
        default=False,
        help="Stop immediately on the first dataset failure",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Resolve paths and checkpoints without inference or figures",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_root = args.output_root
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    print(EVAL_BANNER)
    print(
        f"datasets={args.datasets} horizon={args.horizon} seed={args.seed} "
        f"variant={args.variant} gpu={args.gpu} dry_run={args.dry_run}"
    )

    if args.horizon != 12:
        print(
            f"WARNING: script defaults and expected chain [3,6,12] assume horizon=12; "
            f"got horizon={args.horizon}"
        )

    records: list[dict] = []
    any_fail = False
    for dataset in args.datasets:
        try:
            rec = export_dataset(
                dataset=dataset,
                horizon=int(args.horizon),
                seed=int(args.seed),
                variant=str(args.variant),
                gpu=int(args.gpu),
                output_root=output_root,
                dry_run=bool(args.dry_run),
            )
            records.append(rec)
            if rec.get("status") not in {"success", "dry_run"}:
                any_fail = True
        except Exception as exc:  # noqa: BLE001 - dataset-aware isolation
            any_fail = True
            err = f"{type(exc).__name__}: {exc}"
            tb = traceback.format_exc()
            # Best-effort path resolution for the summary even on failure.
            try:
                meta = resolve_experiment(
                    dataset, int(args.horizon), str(args.variant), int(args.seed)
                )
                out_dir = output_root / f"{meta['slug']}_h{args.horizon}_seed{args.seed}"
                out_dir.mkdir(parents=True, exist_ok=True)
                fail_rec = {
                    "dataset": dataset,
                    "slug": meta["slug"],
                    "status": "failure",
                    "config_path": str(meta["cfg_path"]),
                    "checkpoint_directory": str(meta["ckpt_dir"]),
                    "output_directory": str(out_dir),
                    "error": err,
                    "traceback": tb,
                    "generated_files": [],
                }
                write_json(out_dir / "manifest.json", fail_rec)
                fail_rec["generated_files"] = ["manifest.json"]
            except Exception as nested:  # noqa: BLE001
                fail_rec = {
                    "dataset": dataset,
                    "status": "failure",
                    "error": err,
                    "secondary_error": f"{type(nested).__name__}: {nested}",
                    "traceback": tb,
                    "generated_files": [],
                }
            records.append(fail_rec)
            print(f"[{dataset}] FAILED: {err}", file=sys.stderr)
            if args.fail_fast:
                break

    summary = {
        "variant": args.variant,
        "horizon": args.horizon,
        "seed": args.seed,
        "dry_run": bool(args.dry_run),
        "datasets_requested": list(args.datasets),
        "records": records,
        "all_succeeded": (not any_fail)
        and all(r.get("status") in {"success", "dry_run"} for r in records),
    }
    summary_path = output_root / "qualitative_export_summary.json"
    write_json(summary_path, summary)
    print_console_manifest(records, summary_path)
    return 1 if any_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
