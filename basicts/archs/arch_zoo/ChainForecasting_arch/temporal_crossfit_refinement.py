"""Temporal purged holdout / rolling-origin cross-fit utilities for Plan A.

Index semantics (PEMS04 in12_out12)::

    each sample = (history_start, forecast_start, end)
    history raw interval  = [history_start, forecast_start)
    forecast raw interval = [forecast_start, end)
    full span             = [history_start, end)
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from basicts.utils import load_pkl


def load_split_index(index_path: str | Path, split: str = "train") -> list[tuple[int, int, int]]:
    data = load_pkl(str(index_path))
    if split not in data:
        raise KeyError(f"split {split} missing in {index_path}; keys={list(data.keys())}")
    rows = data[split]
    out = []
    for row in rows:
        a, b, c = int(row[0]), int(row[1]), int(row[2])
        out.append((a, b, c))
    return out


def sample_raw_span(triple: tuple[int, int, int]) -> tuple[int, int]:
    """Inclusive-exclusive raw time span covering history+future."""
    return int(triple[0]), int(triple[2])


def spans_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return max(a[0], b[0]) < min(a[1], b[1])


def compute_min_purge_samples(index: list[tuple[int, int, int]]) -> int:
    """Minimal chronological gap so later sample's span doesn't overlap earlier span.

    For contiguous +1 sampling with span length S = end-history_start,
    samples need gap >= S indices. Measured from data, not hard-coded.
    """
    if len(index) < 2:
        return 0
    # span length is constant for fixed H,P
    s0 = sample_raw_span(index[0])
    span_len = s0[1] - s0[0]
    # starts increase by 1 typically
    start_step = index[1][0] - index[0][0]
    if start_step <= 0:
        raise RuntimeError(f"non-increasing sample starts: {index[0]} -> {index[1]}")
    # number of samples to skip so later.start >= earlier.end
    # later.history_start >= earlier.end  => gap_in_starts >= span_len
    purge = int((span_len + start_step - 1) // start_step)
    return max(purge, 0)


def build_temporal_holdout_manifest(
    index: list[tuple[int, int, int]],
    *,
    train_fraction: float = 0.8,
    dataset: str = "PEMS04",
    horizon: int = 12,
    purge_mode: str = "auto",
) -> dict[str, Any]:
    n = len(index)
    if n < 10:
        raise ValueError(f"too few samples for holdout: {n}")
    cut = int(n * float(train_fraction))
    if cut < 1 or cut >= n:
        raise ValueError(f"invalid cut {cut} for n={n}")
    purge = compute_min_purge_samples(index) if purge_mode == "auto" else int(purge_mode)
    # supernet train: [0, cut)
    # purge zone: [cut, cut+purge)
    # oracle holdout: [cut+purge, n)
    oracle_start = cut + purge
    if oracle_start >= n:
        raise RuntimeError(
            f"purge={purge} leaves empty holdout (cut={cut}, n={n}). "
            "Reduce train_fraction or purge."
        )
    supernet_idx = list(range(0, cut))
    purged_idx = list(range(cut, oracle_start))
    oracle_idx = list(range(oracle_start, n))

    # Overlap audit between supernet and oracle
    overlaps = []
    # Only need to check near boundary (efficient)
    check_left = supernet_idx[-min(len(supernet_idx), purge + 5) :]
    check_right = oracle_idx[: min(len(oracle_idx), purge + 5)]
    for i in check_left:
        si = sample_raw_span(index[i])
        for j in check_right:
            sj = sample_raw_span(index[j])
            if spans_overlap(si, sj):
                overlaps.append({"supernet_i": i, "oracle_j": j, "si": si, "sj": sj})
    if overlaps:
        raise RuntimeError(
            f"temporal holdout still has {len(overlaps)} raw overlaps; "
            f"first={overlaps[0]}; increase purge"
        )

    def _range(idxs: list[int]) -> dict[str, Any]:
        if not idxs:
            return {"empty": True}
        spans = [sample_raw_span(index[i]) for i in idxs]
        return {
            "n": len(idxs),
            "sample_index_first": idxs[0],
            "sample_index_last": idxs[-1],
            "raw_start_min": min(s[0] for s in spans),
            "raw_end_max": max(s[1] for s in spans),
        }

    manifest = {
        "dataset": dataset,
        "horizon": int(horizon),
        "split": "train",
        "train_fraction": float(train_fraction),
        "purge_mode": purge_mode,
        "purge_samples": purge,
        "original_train_samples": n,
        "supernet_train_samples": supernet_idx,
        "purged_samples": purged_idx,
        "oracle_holdout_samples": oracle_idx,
        "ranges": {
            "supernet": _range(supernet_idx),
            "purged": _range(purged_idx),
            "oracle_holdout": _range(oracle_idx),
        },
        "overlap_audit": {"n_overlaps": 0, "status": "PASS"},
        "index_semantics": {
            "tuple": "(history_start, forecast_start, end)",
            "history_interval": "[history_start, forecast_start)",
            "forecast_interval": "[forecast_start, end)",
        },
    }
    blob = json.dumps(
        {
            "supernet": supernet_idx[:3] + supernet_idx[-3:],
            "oracle": oracle_idx[:3] + oracle_idx[-3:],
            "purge": purge,
            "n": n,
        },
        sort_keys=True,
    )
    manifest["manifest_hash"] = hashlib.sha1(blob.encode()).hexdigest()[:16]
    return manifest


def build_rolling_crossfit_manifest(
    index: list[tuple[int, int, int]],
    *,
    num_blocks: int = 5,
    dataset: str = "PEMS04",
    horizon: int = 12,
) -> dict[str, Any]:
    """Causal rolling-origin: train on strict past blocks, oracle on next block."""
    n = len(index)
    if num_blocks < 2:
        raise ValueError("num_blocks must be >= 2")
    block_size = n // num_blocks
    if block_size < 1:
        raise ValueError("block_size empty")
    blocks = []
    for b in range(num_blocks):
        start = b * block_size
        end = n if b == num_blocks - 1 else (b + 1) * block_size
        blocks.append(list(range(start, end)))
    purge = compute_min_purge_samples(index)
    folds = []
    for k in range(1, num_blocks):
        # teacher train = blocks[0..k-1], with purge before oracle block
        train_idx = [i for b in range(k) for i in blocks[b]]
        oracle_block = blocks[k]
        # purge last `purge` samples from train that would overlap oracle
        oracle_start_raw = sample_raw_span(index[oracle_block[0]])[0]
        teacher_train = [
            i for i in train_idx if sample_raw_span(index[i])[1] <= oracle_start_raw
        ]
        purged = [i for i in train_idx if i not in teacher_train]
        # verify no overlap
        overlaps = 0
        for i in teacher_train[-min(20, len(teacher_train)) :]:
            for j in oracle_block[: min(20, len(oracle_block))]:
                if spans_overlap(sample_raw_span(index[i]), sample_raw_span(index[j])):
                    overlaps += 1
        if overlaps:
            raise RuntimeError(f"fold {k} still overlaps after purge")
        folds.append(
            {
                "fold": k,
                "teacher_train_indices": teacher_train,
                "purge_indices": purged,
                "oracle_indices": oracle_block,
                "n_teacher": len(teacher_train),
                "n_oracle": len(oracle_block),
                "n_purged": len(purged),
            }
        )
    return {
        "dataset": dataset,
        "horizon": int(horizon),
        "num_blocks": num_blocks,
        "purge_samples": purge,
        "blocks": [{"block": i, "n": len(b), "first": b[0], "last": b[-1]} for i, b in enumerate(blocks)],
        "folds": folds,
        "scheme": "rolling_origin_temporal_crossfit",
        "causality": "teacher uses only earlier blocks than oracle block",
    }


def save_manifest(manifest: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def load_manifest(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
