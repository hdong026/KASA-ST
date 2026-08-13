"""Plan B-v2 dual-view OOF state cache (teacher reward-aligned + stable deployment).

Stores sharded fp16 tensors on disk. Does NOT recompute route losses.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from basicts.archs.arch_zoo.ChainForecasting_arch.budget_route_utils import (
    default_candidate_routes,
    load_route_costs,
)
from basicts.archs.arch_zoo.ChainForecasting_arch.sequential_f2f_environment import (
    SequentialF2FEnvironment,
)
from basicts.data.indexed_timeseries_dataset import IndexedTimeSeriesForecastingDataset


ALLOWED_MISSING_PREFIXES = ("gain_controller.",)


def sha1_file(path: Path, n: int | None = 16) -> str:
    h = hashlib.sha1(Path(path).read_bytes()).hexdigest()
    return h if n is None else h[:n]


def sha1_bytes(data: bytes, n: int | None = 16) -> str:
    h = hashlib.sha1(data).hexdigest()
    return h if n is None else h[:n]


def load_supernet_strict(
    ckpt: Path | str,
    device: torch.device,
    *,
    cfg: str | None = None,
    allowed_missing_prefixes: tuple[str, ...] = ALLOWED_MISSING_PREFIXES,
):
    """Load forecasting supernet; only allow known Plan-A controller missing keys."""
    from scripts.train_forecast_refinement_controller import _build_model

    ckpt = Path(ckpt)
    if cfg is None:
        sib = list(ckpt.parent.glob("H12_*.py"))
        if not sib:
            raise FileNotFoundError(f"no H12_*.py next to {ckpt}")
        cfg = str(sib[0])

    class _Args:
        horizon = 12
        controller_dim = 128
        pooling_queries = 4
        delta_abs = 0.05
        route_cost_file = None

    _Args.cfg = cfg
    routes = default_candidate_routes(12)
    model = _build_model(_Args(), routes, device)
    raw = torch.load(ckpt, map_location="cpu")
    if isinstance(raw, dict) and "model_state_dict" in raw:
        state = raw["model_state_dict"]
    elif isinstance(raw, dict) and "state_dict" in raw:
        state = raw["state_dict"]
    else:
        state = raw
    missing, unexpected = model.load_state_dict(state, strict=False)
    bad_missing = [
        k for k in missing if not any(k.startswith(p) for p in allowed_missing_prefixes)
    ]
    if bad_missing:
        raise RuntimeError(f"unexpected missing keys: {bad_missing[:20]}")
    if unexpected:
        raise RuntimeError(f"unexpected keys in checkpoint: {unexpected[:20]}")
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    if hasattr(model, "freeze_backbone"):
        model.freeze_backbone(True)
    return model, {
        "missing": list(missing),
        "unexpected": list(unexpected),
        "allowed_missing_prefixes": list(allowed_missing_prefixes),
        "cfg": cfg,
        "checkpoint": str(ckpt),
        "sha1_16": sha1_file(ckpt, 16),
    }


def discover_teacher_checkpoints() -> dict[int, Path]:
    out: dict[int, Path] = {}
    root = Path("checkpoints/PEMS04/H12/budget_f2f")
    for fold in [1, 2, 3, 4]:
        matches = list(
            root.glob(
                f"supernet_eta0p50_dynamic_fair_temporal_cf_fold{fold}_teacher_*/seed1/*/BudgetConditionedAdaptiveF2FNet_best_val_MAE.pt"
            )
        )
        if matches:
            out[fold] = matches[0]
    return out


def estimate_cache_bytes(
    n_samples: int,
    *,
    h_shape: tuple[int, ...] = (4, 307, 64),
    z_shape: tuple[int, ...] = (3, 307, 1),
    dtype_bytes: int = 2,
) -> dict[str, float]:
    """Estimate dual-view cache size (teacher+stable H and Zq)."""
    h_el = int(np.prod(h_shape))
    z_el = int(np.prod(z_shape))
    per = 2 * (h_el + z_el) * dtype_bytes  # teacher+stable
    meta = 256  # rough per-sample metadata
    total = n_samples * (per + meta)
    return {
        "n_samples": n_samples,
        "bytes_per_sample": per + meta,
        "total_bytes": total,
        "total_gib": total / (1024**3),
        "dtype": "fp16" if dtype_bytes == 2 else f"{dtype_bytes}B",
        "h_shape": list(h_shape),
        "z_shape": list(z_shape),
    }


@torch.no_grad()
def extract_hz(
    model,
    history: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return H_shared and Z_q for a batch of histories."""
    env = SequentialF2FEnvironment(model)
    h = model.extract_pre_route_context(history, detach=True)
    z = env.execute_quarter_prefix(history)["Z_q"].detach()
    return h, z


class PlanBV2StateCacheWriter:
    """Write sharded dual-view state cache to disk."""

    def __init__(
        self,
        out_dir: Path | str,
        *,
        shard_size: int = 256,
        use_fp16: bool = True,
    ):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.shard_size = int(shard_size)
        self.use_fp16 = bool(use_fp16)
        self.records: list[dict[str, Any]] = []
        self._buf: list[dict[str, Any]] = []
        self._shard_id = 0

    def _flush(self) -> None:
        if not self._buf:
            return
        path = self.out_dir / f"shard_{self._shard_id:05d}.pt"
        payload = {
            "sample_index": [r["sample_index"] for r in self._buf],
            "fold_id": [r["fold_id"] for r in self._buf],
            "teacher_checkpoint_hash": [r["teacher_checkpoint_hash"] for r in self._buf],
            "stable_checkpoint_hash": [r["stable_checkpoint_hash"] for r in self._buf],
            "oracle_hash": [r["oracle_hash"] for r in self._buf],
            "H_teacher": torch.stack([r["H_teacher"] for r in self._buf]),
            "Zq_teacher": torch.stack([r["Zq_teacher"] for r in self._buf]),
            "H_stable": torch.stack([r["H_stable"] for r in self._buf]),
            "Zq_stable": torch.stack([r["Zq_stable"] for r in self._buf]),
            "route_losses": torch.stack([r["route_losses"] for r in self._buf]),
        }
        torch.save(payload, path)
        for i, r in enumerate(self._buf):
            self.records.append(
                {
                    "sample_index": r["sample_index"],
                    "fold_id": r["fold_id"],
                    "shard": path.name,
                    "offset": i,
                    "teacher_checkpoint_hash": r["teacher_checkpoint_hash"],
                    "stable_checkpoint_hash": r["stable_checkpoint_hash"],
                }
            )
        self._buf = []
        self._shard_id += 1

    def add(self, record: dict[str, Any]) -> None:
        # cast to fp16 if requested
        if self.use_fp16:
            for k in ("H_teacher", "Zq_teacher", "H_stable", "Zq_stable"):
                record[k] = record[k].half().cpu()
            record["route_losses"] = record["route_losses"].float().cpu()
        else:
            for k in ("H_teacher", "Zq_teacher", "H_stable", "Zq_stable", "route_losses"):
                record[k] = record[k].cpu()
        self._buf.append(record)
        if len(self._buf) >= self.shard_size:
            self._flush()

    def finalize(self, meta: dict[str, Any]) -> Path:
        self._flush()
        # integrity
        sis = [r["sample_index"] for r in self.records]
        if len(sis) != len(set(sis)):
            raise RuntimeError("duplicate sample_index in cache")
        meta_path = self.out_dir / "manifest.json"
        meta = dict(meta)
        meta["n_samples"] = len(self.records)
        meta["records"] = self.records
        meta["use_fp16"] = self.use_fp16
        meta_path.write_text(json.dumps(meta, indent=2))
        return meta_path


class PlanBV2StateCache:
    """Read dual-view states by sample_index (mmap-friendly shard loads)."""

    def __init__(self, cache_dir: Path | str):
        self.cache_dir = Path(cache_dir)
        self.manifest = json.loads((self.cache_dir / "manifest.json").read_text())
        self.index = {
            int(r["sample_index"]): r for r in self.manifest["records"]
        }
        self._shard_cache: dict[str, dict] = {}

    def __len__(self) -> int:
        return len(self.index)

    def sample_indices(self) -> list[int]:
        return sorted(self.index.keys())

    def _load_shard(self, name: str) -> dict:
        if name not in self._shard_cache:
            self._shard_cache[name] = torch.load(
                self.cache_dir / name, map_location="cpu"
            )
            # keep only a few shards
            if len(self._shard_cache) > 4:
                oldest = next(iter(self._shard_cache))
                if oldest != name:
                    del self._shard_cache[oldest]
        return self._shard_cache[name]

    def get(self, sample_index: int) -> dict[str, Any]:
        rec = self.index[int(sample_index)]
        shard = self._load_shard(rec["shard"])
        off = int(rec["offset"])
        return {
            "sample_index": int(sample_index),
            "fold_id": int(shard["fold_id"][off]),
            "teacher_checkpoint_hash": shard["teacher_checkpoint_hash"][off],
            "stable_checkpoint_hash": shard["stable_checkpoint_hash"][off],
            "H_teacher": shard["H_teacher"][off],
            "Zq_teacher": shard["Zq_teacher"][off],
            "H_stable": shard["H_stable"][off],
            "Zq_stable": shard["Zq_stable"][off],
            "route_losses": shard["route_losses"][off],
        }


@torch.no_grad()
def build_dual_view_cache(
    *,
    oracle_path: str | Path,
    stable_ckpt: str | Path,
    out_dir: str | Path,
    device: torch.device,
    data_file: str = "datasets/PEMS04/data_in12_out12.pkl",
    index_file: str = "datasets/PEMS04/index_in12_out12.pkl",
    max_per_fold: int | None = None,
    shard_size: int = 256,
    use_fp16: bool = True,
    time_budget_sec: float | None = 120.0,
) -> dict[str, Any]:
    """Build teacher/stable dual-view OOF cache. Caps runtime if requested."""
    t0 = time.time()
    oracle_path = Path(oracle_path)
    oracle = json.loads(oracle_path.read_text())
    oracle_hash = sha1_file(oracle_path, 16)
    records = oracle["records"]
    teachers = discover_teacher_checkpoints()
    # verify hashes
    hash_ok = {}
    for fold, path in teachers.items():
        h = sha1_file(path, 16)
        expected = next(
            (r["teacher_checkpoint_hash"] for r in records if int(r["teacher_fold"]) == fold),
            None,
        )
        hash_ok[fold] = {"path": str(path), "sha1_16": h, "oracle_hash": expected, "match": h == expected}
        if h != expected:
            raise RuntimeError(f"teacher fold {fold} hash mismatch: {h} vs {expected}")

    stable, stable_meta = load_supernet_strict(stable_ckpt, device)
    stable_hash = stable_meta["sha1_16"]

    # group samples by fold
    by_fold: dict[int, list[dict]] = {1: [], 2: [], 3: [], 4: []}
    for r in records:
        f = int(r["teacher_fold"])
        by_fold[f].append(r)
    if max_per_fold is not None:
        for f in by_fold:
            by_fold[f] = by_fold[f][: int(max_per_fold)]

    n_total = sum(len(v) for v in by_fold.values())
    # probe shapes
    base = IndexedTimeSeriesForecastingDataset(data_file, index_file, "train")
    probe_si = int(by_fold[1][0]["sample_index"])
    item = base[probe_si]
    hist = item[1] if len(item) == 3 else item[0]
    if not torch.is_tensor(hist):
        raise RuntimeError("unexpected dataset item")
    hist = hist.unsqueeze(0).to(device)
    Hp, Zp = extract_hz(stable, hist)
    est = estimate_cache_bytes(
        n_total, h_shape=tuple(Hp.shape[1:]), z_shape=tuple(Zp.shape[1:]), dtype_bytes=2 if use_fp16 else 4
    )
    print(f"[cache] estimated_size_GiB={est['total_gib']:.4f} n={n_total}")

    writer = PlanBV2StateCacheWriter(out_dir, shard_size=shard_size, use_fp16=use_fp16)
    routes = default_candidate_routes(12)
    # process fold by fold to avoid loading all teachers at once
    processed = 0
    timed_out = False
    for fold in sorted(by_fold.keys()):
        if time_budget_sec is not None and (time.time() - t0) > time_budget_sec:
            timed_out = True
            break
        tpath = teachers[fold]
        teacher, tmeta = load_supernet_strict(tpath, device)
        for rec in by_fold[fold]:
            if time_budget_sec is not None and (time.time() - t0) > time_budget_sec:
                timed_out = True
                break
            si = int(rec["sample_index"])
            item = base[si]
            history = item[1] if len(item) == 3 else item[0]
            history = history.unsqueeze(0).to(device)
            Ht, Zt = extract_hz(teacher, history)
            Hs, Zs = extract_hz(stable, history)
            losses = torch.tensor(rec["true_route_losses"], dtype=torch.float32)
            writer.add(
                {
                    "sample_index": si,
                    "fold_id": fold,
                    "teacher_checkpoint_hash": tmeta["sha1_16"],
                    "stable_checkpoint_hash": stable_hash,
                    "oracle_hash": oracle_hash,
                    "H_teacher": Ht.squeeze(0),
                    "Zq_teacher": Zt.squeeze(0),
                    "H_stable": Hs.squeeze(0),
                    "Zq_stable": Zs.squeeze(0),
                    "route_losses": losses,
                }
            )
            processed += 1
        del teacher
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if timed_out:
            break

    meta = {
        "oracle_path": str(oracle_path),
        "oracle_hash": oracle_hash,
        "stable": stable_meta,
        "teachers": {str(k): v for k, v in hash_ok.items()},
        "estimate": est,
        "processed": processed,
        "timed_out": timed_out,
        "max_per_fold": max_per_fold,
        "elapsed_sec": time.time() - t0,
        "routes": routes,
        "costs": load_route_costs(None, routes, 12),
    }
    manifest = writer.finalize(meta)
    return {
        "manifest": str(manifest),
        "out_dir": str(out_dir),
        **meta,
        "n_written": processed,
    }
