"""Prepare HyperD official data files (desc.json / data.dat) without duplicating repo data."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np

from baselines.HyperD.hyperd_settings import (
    DATASET_META,
    INPUT_LEN,
    OUTPUT_LEN,
    TRAIN_VAL_TEST_RATIO,
)

ROOT = Path(__file__).resolve().parents[2]
STEPS_PER_DAY = 288


def repo_path(*parts: str) -> Path:
    return ROOT.joinpath(*parts)


def desc_json_path(dataset_name: str) -> Path:
    return repo_path("datasets", dataset_name, "desc.json")


def data_dat_path(dataset_name: str) -> Path:
    return repo_path("datasets", dataset_name, "data.dat")


def init_npy_paths(dataset_name: str) -> tuple[Path, Path]:
    base = repo_path("datasets", dataset_name)
    return base / "daily_init.npy", base / "weekly_init.npy"


def build_desc_dict(dataset_name: str) -> dict:
    meta = DATASET_META[dataset_name]
    return {
        "name": dataset_name,
        "num_nodes": meta["num_nodes"],
        "num_time_steps": meta["num_time_steps"],
        "shape": [meta["num_time_steps"], meta["num_nodes"], 3],
        "regular_settings": {
            "INPUT_LEN": INPUT_LEN,
            "OUTPUT_LEN": OUTPUT_LEN,
            "TRAIN_VAL_TEST_RATIO": TRAIN_VAL_TEST_RATIO,
            "NORM_EACH_CHANNEL": False,
            "RESCALE": True,
            "NULL_VAL": 0.0,
        },
    }


def write_desc_json(dataset_name: str) -> Path:
    path = desc_json_path(dataset_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    desc = build_desc_dict(dataset_name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(desc, f, indent=2)
        f.write("\n")
    return path


def _resolve_raw_npz(dataset_name: str) -> Path:
    meta = DATASET_META[dataset_name]
    candidates = [
        repo_path(meta["raw_npz"]),
        repo_path("datasets", dataset_name, f"{dataset_name}.npz"),
        repo_path("datasets", "raw_data", dataset_name, f"{dataset_name}.npz"),
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"Cannot find raw npz for {dataset_name}. Expected one of: "
        + ", ".join(str(p) for p in candidates)
    )


def _build_three_channel_series(raw: np.ndarray) -> np.ndarray:
    """Match BasicTS/HyperD 3-channel layout: [flow, time-of-day, day-of-week]."""
    if raw.ndim != 3:
        raise ValueError(f"Expected raw shape (T, N, C), got {raw.shape}")
    flow = raw[..., 0].astype(np.float64)
    t_steps, num_nodes, _ = raw.shape

    total_len = t_steps
    valid_len = int(total_len * TRAIN_VAL_TEST_RATIO[1])
    test_len = int(total_len * TRAIN_VAL_TEST_RATIO[2])
    train_len = total_len - valid_len - test_len

    train_flow = flow[:train_len]
    mean = train_flow.mean()
    std = train_flow.std()
    if std < 1e-6:
        std = 1.0
    flow_norm = (flow - mean) / std

    tod = np.array([i % STEPS_PER_DAY / STEPS_PER_DAY for i in range(t_steps)], dtype=np.float32)
    tod_tiled = np.tile(tod, (num_nodes, 1)).T[..., None]

    dow = np.array([(i // STEPS_PER_DAY) % 7 for i in range(t_steps)], dtype=np.float32)
    dow_tiled = np.tile(dow, (num_nodes, 1)).T[..., None]

    processed = np.concatenate(
        [flow_norm[..., None].astype(np.float32), tod_tiled, dow_tiled],
        axis=-1,
    )
    return processed


def write_data_dat(dataset_name: str, force: bool = False) -> Path:
    out_path = data_dat_path(dataset_name)
    if out_path.is_file() and not force:
        return out_path

    raw_path = _resolve_raw_npz(dataset_name)
    raw = np.load(raw_path)["data"]
    processed = _build_three_channel_series(raw)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    mmap = np.memmap(out_path, dtype="float32", mode="w+", shape=processed.shape)
    mmap[:] = processed
    mmap.flush()
    del mmap
    return out_path


def ensure_hyperd_data(dataset_name: str, force_dat: bool = False) -> dict[str, Path]:
    """Ensure desc.json and data.dat exist for HyperD official loader."""
    if dataset_name not in DATASET_META:
        raise KeyError(f"Unsupported HyperD dataset: {dataset_name}")

    desc_path = write_desc_json(dataset_name)
    dat_path = write_data_dat(dataset_name, force=force_dat)
    return {"desc_json": desc_path, "data_dat": dat_path}


def missing_requirements(dataset_name: str) -> list[str]:
    missing: list[str] = []
    meta = DATASET_META.get(dataset_name)
    if meta is None:
        return [f"unknown dataset {dataset_name}"]

    try:
        _resolve_raw_npz(dataset_name)
    except FileNotFoundError as exc:
        missing.append(str(exc))

    adj = repo_path(meta["adj_path"])
    if not adj.is_file():
        missing.append(f"missing adjacency matrix: {adj}")
    return missing


def all_dataset_names(names: Iterable[str] | None = None) -> list[str]:
    if names is None:
        return list(DATASET_META.keys())
    return [n.upper() for n in names]
