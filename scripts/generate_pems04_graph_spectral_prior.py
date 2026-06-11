#!/usr/bin/env python3
"""Generate offline graph spectral prior channels for PeMS04.

Aligned with SPECTRA/scripts/data_preparation/PEMS04/generate_gft_data.py:
- read processed PeMS04 12→12 data (flow / ToD / DoW)
- build graph Laplacian from adj_mx.pkl
- append train-normalized graph low / optional high prior channels
- write standalone npz plus optional full dataset dir (pkl/index/adj)
"""
from __future__ import annotations

import argparse
import os
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = ROOT / "datasets" / "PEMS04"
DEFAULT_RAW_NPZ = ROOT / "datasets" / "raw_data" / "PEMS04" / "PEMS04.npz"
DEFAULT_PROCESSED_NPZ = DEFAULT_DATASET_DIR / "data_in12_out12.npz"
DEFAULT_ADJ = DEFAULT_DATASET_DIR / "adj_mx.pkl"
DEFAULT_INDEX = DEFAULT_DATASET_DIR / "index_in12_out12.pkl"
HISTORY_SEQ_LEN = 12
FUTURE_SEQ_LEN = 12


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    return p


def load_processed_data(input_path: Path) -> np.ndarray:
    """Load [T, N, C] processed array from npz/pkl."""
    if input_path.suffix == ".pkl":
        with open(input_path, "rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, dict):
            if "processed_data" in obj:
                return np.array(obj["processed_data"], dtype=np.float32)
            if "data" in obj:
                return np.array(obj["data"], dtype=np.float32)
        return np.array(obj, dtype=np.float32)

    archive = np.load(input_path)
    for key in ("processed_data", "data"):
        if key in archive:
            return np.array(archive[key], dtype=np.float32)
    raise KeyError(f"{input_path} must contain 'processed_data' or 'data'")


def load_train_end(index_path: Path, t_len: int, train_ratio: float = 0.6) -> int:
    if index_path.is_file():
        with open(index_path, "rb") as f:
            index = pickle.load(f)
        train_index = index.get("train")
        if train_index:
            return int(train_index[-1][1])
    return int(t_len * train_ratio)


def load_adjacency(adj_path: Path, num_nodes: int) -> np.ndarray:
    with open(adj_path, "rb") as f:
        try:
            adj_obj = pickle.load(f)
        except UnicodeDecodeError:
            f.seek(0)
            adj_obj = pickle.load(f, encoding="latin1")

    if isinstance(adj_obj, (list, tuple)):
        for item in reversed(adj_obj):
            if hasattr(item, "shape") and len(item.shape) == 2:
                adj_obj = item
                break

    if hasattr(adj_obj, "toarray"):
        adj_obj = adj_obj.toarray()

    adj = np.asarray(adj_obj, dtype=np.float32)
    if adj.ndim != 2:
        raise ValueError(f"Adjacency must be 2D, got shape {adj.shape}")
    if adj.shape[0] != num_nodes or adj.shape[1] != num_nodes:
        raise ValueError(
            f"Adjacency shape {adj.shape} does not match num_nodes={num_nodes}"
        )

    adj = np.nan_to_num(adj, nan=0.0, posinf=0.0, neginf=0.0)
    adj = np.maximum(adj, 0.0)
    return adj


def build_normalized_laplacian(
    adj: np.ndarray,
    self_loop_weight: float = 1e-3,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a_sym = 0.5 * (adj + adj.T)
    n = a_sym.shape[0]
    a_sym = a_sym + self_loop_weight * np.eye(n, dtype=np.float32)

    deg = np.diag(a_sym.sum(axis=1))
    d_inv_sqrt = np.power(np.diag(deg) + eps, -0.5)
    d_inv_sqrt = np.diag(d_inv_sqrt)
    lap = np.eye(n, dtype=np.float32) - d_inv_sqrt @ a_sym @ d_inv_sqrt

    eigvals, eigvecs = np.linalg.eigh(lap)
    order = np.argsort(eigvals)
    eigvals = eigvals[order].astype(np.float32)
    eigvecs = eigvecs[:, order].astype(np.float32)
    return lap, eigvals, eigvecs


def normalize_prior(
    x: np.ndarray,
    train_end: int,
    eps: float = 1e-8,
) -> tuple[np.ndarray, float, float]:
    train_slice = x[:train_end]
    mean = float(train_slice.mean())
    std = float(train_slice.std() + eps)
    return ((x - mean) / std).astype(np.float32), mean, std


def dump_training_package(
    processed_data: np.ndarray,
    output_dir: Path,
    index_path: Path,
    adj_path: Path,
    scaler_path: Path | None = None,
    history_seq_len: int = HISTORY_SEQ_LEN,
    future_seq_len: int = FUTURE_SEQ_LEN,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    data_pkl = output_dir / f"data_in{history_seq_len}_out{future_seq_len}.pkl"
    index_out = output_dir / f"index_in{history_seq_len}_out{future_seq_len}.pkl"
    adj_out = output_dir / "adj_mx.pkl"
    scaler_src = scaler_path or (
        DEFAULT_DATASET_DIR / f"scaler_in{history_seq_len}_out{future_seq_len}.pkl"
    )
    scaler_out = output_dir / scaler_src.name

    with open(data_pkl, "wb") as f:
        pickle.dump({"processed_data": processed_data.astype(np.float32)}, f)
    shutil.copy2(index_path, index_out)
    if adj_path.is_file():
        shutil.copy2(adj_path, adj_out)
    if scaler_src.is_file():
        shutil.copy2(scaler_src, scaler_out)
    else:
        raise FileNotFoundError(
            f"Missing scaler file: {scaler_src}. "
            "Run PeMS04 data preparation before generating graph spectral priors."
        )


def generate_prior(
    input_npz: Path,
    adj_path: Path,
    output_npz: Path,
    k: int,
    include_high: bool = False,
    self_loop_weight: float = 1e-3,
    index_path: Path | None = None,
    output_dir: Path | None = None,
) -> None:
    data = load_processed_data(input_npz)
    if data.ndim != 3:
        raise ValueError(f"Expected data shape [T, N, C], got {data.shape}")

    t_len, num_nodes, num_channels = data.shape
    if num_channels < 3:
        raise ValueError(f"Expected at least 3 channels (flow, ToD, DoW), got {num_channels}")

    index_path = index_path or DEFAULT_INDEX
    train_end = load_train_end(index_path, t_len)
    val_len = int(t_len * 0.2)
    test_len = t_len - train_end - val_len

    adj = load_adjacency(adj_path, num_nodes)
    _, eigvals, eigvecs = build_normalized_laplacian(adj, self_loop_weight=self_loop_weight)
    k_eff = min(k, num_nodes)
    u_low = eigvecs[:, :k_eff]
    p_low = u_low @ u_low.T

    flow = data[..., 0]
    x_low = flow @ p_low.T
    x_high = flow - x_low

    x_low_norm, low_mean, low_std = normalize_prior(x_low, train_end)
    out_channels = [
        data[..., 0:1],
        data[..., 1:2],
        data[..., 2:3],
        x_low_norm[..., np.newaxis],
    ]
    high_mean = np.float32(0.0)
    high_std = np.float32(1.0)
    if include_high:
        x_high_norm, high_mean, high_std = normalize_prior(x_high, train_end)
        out_channels.append(x_high_norm[..., np.newaxis])

    out_data = np.concatenate(out_channels, axis=-1).astype(np.float32)

    save_kwargs = {
        "data": out_data,
        "processed_data": out_data,
        "graph_spectral_k": np.int32(k_eff),
        "graph_low_mean": np.float32(low_mean),
        "graph_low_std": np.float32(low_std),
        "graph_spectral_eigvals": eigvals,
    }
    if include_high:
        save_kwargs["graph_high_mean"] = np.float32(high_mean)
        save_kwargs["graph_high_std"] = np.float32(high_std)

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_npz, **save_kwargs)

    if output_dir is not None:
        dump_training_package(
            processed_data=out_data,
            output_dir=output_dir,
            index_path=index_path,
            adj_path=adj_path,
        )

    print(f"input path: {input_npz}")
    print(f"input shape: {data.shape}")
    print(f"output shape: {out_data.shape}")
    print(f"k: {k_eff}")
    print(f"train_end (official index): {train_end}")
    print(f"first 10 eigenvalues: {eigvals[:10].tolist()}")
    print(
        f"train low prior mean/std: "
        f"{x_low_norm[:train_end].mean():.6f} / {x_low_norm[:train_end].std():.6f}"
    )
    if include_high:
        print(
            f"train high prior mean/std: "
            f"{x_high_norm[:train_end].mean():.6f} / {x_high_norm[:train_end].std():.6f}"
        )
    print(f"approx split lens train/val/test: {train_end}/{val_len}/{test_len}")
    print(f"output npz: {output_npz}")
    if output_dir is not None:
        print(f"output dataset dir: {output_dir}")


def auto_input_path(user_input: str | None) -> Path:
    if user_input:
        p = resolve_path(user_input)
        if not p.is_file():
            raise FileNotFoundError(f"Input file not found: {p}")
        return p

    candidates = [
        DEFAULT_PROCESSED_NPZ,
        DEFAULT_DATASET_DIR / "data_in12_out12.pkl",
        DEFAULT_DATASET_DIR / "data.npz",
        DEFAULT_RAW_NPZ,
    ]
    for cand in candidates:
        if cand.is_file():
            return cand

    msg = (
        "No PeMS04 input found. Provide --input_npz, or prepare one of:\n"
        f"  - {DEFAULT_PROCESSED_NPZ}\n"
        f"  - {DEFAULT_DATASET_DIR / 'data_in12_out12.pkl'}\n"
        f"  - {DEFAULT_RAW_NPZ}"
    )
    raise FileNotFoundError(msg)


def auto_output_dir(output_npz: Path, k: int, include_high: bool) -> Path | None:
    name = output_npz.stem
    if name.startswith("data_graph_spectral_"):
        suffix = name.replace("data_graph_spectral_", "")
        return ROOT / "datasets" / f"PEMS04_graph_spectral_{suffix}"
    if include_high:
        return ROOT / "datasets" / f"PEMS04_graph_spectral_k{k}_lowhigh"
    return ROOT / "datasets" / f"PEMS04_graph_spectral_k{k}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate offline graph spectral prior channels for PeMS04."
    )
    parser.add_argument(
        "--input_npz",
        type=str,
        default="",
        help="Processed PeMS04 npz/pkl. Default: datasets/PEMS04/data_in12_out12.npz",
    )
    parser.add_argument(
        "--adj_path",
        type=str,
        default=str(DEFAULT_ADJ.relative_to(ROOT)),
        help="Adjacency pickle path.",
    )
    parser.add_argument("--output_npz", type=str, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--include_high", action="store_true")
    parser.add_argument("--self_loop_weight", type=float, default=1e-3)
    parser.add_argument(
        "--index_path",
        type=str,
        default=str(DEFAULT_INDEX.relative_to(ROOT)),
        help="Official index pkl for train-only normalization.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Optional dataset dir for data/index/adj pkl package.",
    )
    parser.add_argument(
        "--write_dataset_dir",
        action="store_true",
        help="Also write datasets/PEMS04_graph_spectral_k*/ package.",
    )
    args = parser.parse_args()

    input_npz = auto_input_path(args.input_npz or None)
    adj_path = resolve_path(args.adj_path)
    output_npz = resolve_path(args.output_npz)
    index_path = resolve_path(args.index_path)

    output_dir = resolve_path(args.output_dir) if args.output_dir else None
    if args.write_dataset_dir and output_dir is None:
        output_dir = auto_output_dir(output_npz, args.k, args.include_high)

    if not adj_path.is_file():
        raise FileNotFoundError(f"Adjacency file not found: {adj_path}")
    if not index_path.is_file():
        raise FileNotFoundError(
            f"Index file not found: {index_path}. "
            "Run scripts/data_preparation/PEMS04/generate_holost_data.py first."
        )

    generate_prior(
        input_npz=input_npz,
        adj_path=adj_path,
        output_npz=output_npz,
        k=args.k,
        include_high=args.include_high,
        self_loop_weight=args.self_loop_weight,
        index_path=index_path,
        output_dir=output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
