"""OOF dataset / cache for Budgeted Bellman Forecast Refinement.

Stores teacher-aligned Z_q + route losses/returns. Raw history X is loaded
on the fly (teacher-independent). Does NOT use stable H_shared as Q-state.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from basicts.archs.arch_zoo.ChainForecasting_arch.budgeted_bellman_refinement import (
    BudgetedRefinementMDP,
    centered_terminal_returns,
    derive_additive_stage_costs,
    exact_q0_targets,
    exact_q1_targets,
    global_return_scale_from_gains,
)
from basicts.archs.arch_zoo.ChainForecasting_arch.plan_b_v2_state_cache import (
    PlanBV2StateCache,
    discover_teacher_checkpoints,
    extract_hz,
    load_supernet_strict,
    sha1_file,
)
from basicts.archs.arch_zoo.ChainForecasting_arch.sequential_f2f_environment import (
    SequentialF2FEnvironment,
)
from basicts.data.indexed_timeseries_dataset import IndexedTimeSeriesForecastingDataset


DEFAULT_ORACLE = "results/pems04_temporal_crossfit_refinement_oracle.json"
DEFAULT_DATA = "datasets/PEMS04/data_in12_out12.pkl"
DEFAULT_INDEX = "datasets/PEMS04/index_in12_out12.pkl"
DEFAULT_V2_CACHE = "results/planB_v2_oof_state_cache"


def _sha1_path(p: Path, n: int | None = 16) -> str:
    return sha1_file(p, n)


class BellmanOOFCache:
    """Sharded cache: Z_q (teacher) + losses/returns + meta. X loaded live."""

    def __init__(self, cache_dir: Path | str):
        self.cache_dir = Path(cache_dir)
        self.manifest = json.loads((self.cache_dir / "manifest.json").read_text())
        self.index = {int(r["sample_index"]): r for r in self.manifest["records"]}
        self._shard: dict[str, dict] = {}

    def __len__(self) -> int:
        return len(self.index)

    def sample_indices(self) -> list[int]:
        return sorted(self.index.keys())

    def fold_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for r in self.manifest["records"]:
            f = str(r["fold_id"])
            counts[f] = counts.get(f, 0) + 1
        return counts

    def _load(self, name: str) -> dict:
        if name not in self._shard:
            self._shard[name] = torch.load(self.cache_dir / name, map_location="cpu")
            if len(self._shard) > 4:
                oldest = next(iter(self._shard))
                if oldest != name:
                    del self._shard[oldest]
        return self._shard[name]

    def get(self, sample_index: int) -> dict[str, Any]:
        rec = self.index[int(sample_index)]
        shard = self._load(rec["shard"])
        off = int(rec["offset"])
        losses = shard["losses_DMQF"][off].float()
        gains = shard["gains_DMQF"][off].float()
        return {
            "sample_index": int(sample_index),
            "fold_id": int(shard["fold_id"][off]),
            "Z_q": shard["Z_q"][off].float(),
            "L_D": float(losses[0]),
            "L_M": float(losses[1]),
            "L_Q": float(losses[2]),
            "L_F": float(losses[3]),
            "g_D": float(gains[0]),
            "g_M": float(gains[1]),
            "g_Q": float(gains[2]),
            "g_F": float(gains[3]),
            "losses_DMQF": losses,
            "gains_DMQF": gains,
            "teacher_checkpoint_hash": shard["teacher_checkpoint_hash"][off],
        }


class BellmanOOFDataset(Dataset):
    def __init__(
        self,
        cache: BellmanOOFCache,
        data_file: str = DEFAULT_DATA,
        index_file: str = DEFAULT_INDEX,
        sample_indices: list[int] | None = None,
        *,
        scale: float = 1.0,
        mdp: BudgetedRefinementMDP | None = None,
    ):
        self.cache = cache
        self.base = IndexedTimeSeriesForecastingDataset(data_file, index_file, "train")
        self.indices = sample_indices or cache.sample_indices()
        self.scale = float(scale)
        self.mdp = mdp or BudgetedRefinementMDP(12)
        self.regimes = self.mdp.unique_nontrivial_budget_regimes()

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> dict[str, Any]:
        si = int(self.indices[i])
        row = self.cache.get(si)
        future, history, _ = self.base[si]
        # history: [P,N,C]
        x = history.float()
        g = row["gains_DMQF"] / self.scale
        q1_t = exact_q1_targets(g)
        # pack all regimes' Q0 targets
        q0_targets = []
        q0_valids = []
        budgets = []
        s0_masks = []
        sq_masks = []
        for reg in self.regimes:
            t, v = exact_q0_targets(g, reg["s0_mask"], reg["sq_mask_after_q"])
            q0_targets.append(t)
            q0_valids.append(v)
            budgets.append(reg["budget"])
            s0_masks.append(
                torch.tensor(
                    [reg["s0_mask"]["f"], reg["s0_mask"]["m"], reg["s0_mask"]["q"]],
                    dtype=torch.bool,
                )
            )
            sq_masks.append(
                torch.tensor(
                    [reg["sq_mask_after_q"]["f"], reg["sq_mask_after_q"]["m"]],
                    dtype=torch.bool,
                )
            )
        return {
            "sample_index": si,
            "fold_id": row["fold_id"],
            "X": x,
            "Z_q": row["Z_q"],
            "gains": g,
            "losses": row["losses_DMQF"],
            "q1_target": q1_t,
            "q0_targets": torch.stack(q0_targets, dim=0),  # [R,3]
            "q0_valids": torch.stack(q0_valids, dim=0),
            "budgets": torch.tensor(budgets, dtype=torch.float32),
            "s0_masks": torch.stack(s0_masks, dim=0),
            "sq_masks": torch.stack(sq_masks, dim=0),
        }


def collate_bellman(batch: list[dict]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k in batch[0]:
        vals = [b[k] for b in batch]
        if isinstance(vals[0], torch.Tensor):
            out[k] = torch.stack(vals, dim=0)
        else:
            out[k] = vals
    return out


class BellmanCacheWriter:
    def __init__(self, out_dir: Path | str, shard_size: int = 256, use_fp16: bool = True):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.shard_size = shard_size
        self.use_fp16 = use_fp16
        self._buf: list[dict] = []
        self.records: list[dict] = []
        self._shard_id = 0

    def _flush(self) -> None:
        if not self._buf:
            return
        name = f"shard_{self._shard_id:05d}.pt"
        payload = {
            "sample_index": [r["sample_index"] for r in self._buf],
            "fold_id": [r["fold_id"] for r in self._buf],
            "teacher_checkpoint_hash": [r["teacher_checkpoint_hash"] for r in self._buf],
            "Z_q": torch.stack([r["Z_q"] for r in self._buf], dim=0),
            "losses_DMQF": torch.stack([r["losses_DMQF"] for r in self._buf], dim=0),
            "gains_DMQF": torch.stack([r["gains_DMQF"] for r in self._buf], dim=0),
        }
        torch.save(payload, self.out_dir / name)
        for off, r in enumerate(self._buf):
            self.records.append(
                {
                    "sample_index": r["sample_index"],
                    "fold_id": r["fold_id"],
                    "shard": name,
                    "offset": off,
                    "teacher_checkpoint_hash": r["teacher_checkpoint_hash"],
                }
            )
        self._buf = []
        self._shard_id += 1

    def add(self, record: dict) -> None:
        z = record["Z_q"]
        if self.use_fp16:
            z = z.half()
        self._buf.append(
            {
                "sample_index": int(record["sample_index"]),
                "fold_id": int(record["fold_id"]),
                "teacher_checkpoint_hash": str(record["teacher_checkpoint_hash"]),
                "Z_q": z.cpu(),
                "losses_DMQF": record["losses_DMQF"].float().cpu(),
                "gains_DMQF": record["gains_DMQF"].float().cpu(),
            }
        )
        if len(self._buf) >= self.shard_size:
            self._flush()

    def finalize(self, meta: dict) -> Path:
        self._flush()
        meta = dict(meta)
        meta["n_samples"] = len(self.records)
        meta["records"] = self.records
        path = self.out_dir / "manifest.json"
        path.write_text(json.dumps(meta, indent=2))
        return path


@torch.no_grad()
def build_bellman_oof_cache(
    *,
    out_dir: str | Path,
    oracle_path: str | Path = DEFAULT_ORACLE,
    device: torch.device,
    max_per_fold: int | None = None,
    max_total: int | None = None,
    reuse_v2_zq_cache: str | Path | None = DEFAULT_V2_CACHE,
    data_file: str = DEFAULT_DATA,
    index_file: str = DEFAULT_INDEX,
    shard_size: int = 256,
) -> dict[str, Any]:
    """Build Bellman OOF cache.

    Prefer reusing teacher Z_q from an existing Plan-B-v2 dual-view cache
    (same fold teachers / same losses) to avoid re-running teachers.
    Does NOT store H_shared; only Z_q + losses/returns.
    """
    t0 = time.time()
    out_dir = Path(out_dir)
    oracle = json.loads(Path(oracle_path).read_text())
    oracle_hash = _sha1_path(Path(oracle_path), 16)
    records = oracle["records"]

    # ordering / losses
    by_fold: dict[int, list] = {1: [], 2: [], 3: [], 4: []}
    for r in records:
        by_fold[int(r["teacher_fold"])].append(r)
    if max_per_fold is not None:
        for f in by_fold:
            by_fold[f] = by_fold[f][: int(max_per_fold)]

    selected: list[dict] = []
    for f in (1, 2, 3, 4):
        selected.extend(by_fold[f])
    if max_total is not None:
        selected = selected[: int(max_total)]

    writer = BellmanCacheWriter(out_dir, shard_size=shard_size)
    v2 = None
    if reuse_v2_zq_cache and Path(reuse_v2_zq_cache).is_dir():
        try:
            v2 = PlanBV2StateCache(reuse_v2_zq_cache)
        except Exception:
            v2 = None

    teachers = discover_teacher_checkpoints()
    teacher_models: dict[int, Any] = {}
    base = IndexedTimeSeriesForecastingDataset(data_file, index_file, "train")

    n_from_v2 = 0
    n_from_teacher = 0
    for r in selected:
        si = int(r["sample_index"])
        fold = int(r["teacher_fold"])
        losses = torch.tensor(r["true_route_losses"], dtype=torch.float32)  # D,M,Q,F
        gains = centered_terminal_returns(losses.unsqueeze(0)).squeeze(0)
        thash = str(r["teacher_checkpoint_hash"])
        z_q = None
        if v2 is not None and si in v2.index:
            row = v2.get(si)
            z_q = row["Zq_teacher"].float()
            # verify fold / hash when present
            if int(row["fold_id"]) != fold:
                z_q = None
            n_from_v2 += 1 if z_q is not None else 0
        if z_q is None:
            if fold not in teacher_models:
                if fold not in teachers:
                    raise FileNotFoundError(f"missing teacher fold {fold}")
                model, meta = load_supernet_strict(teachers[fold], device)
                teacher_models[fold] = model
            model = teacher_models[fold]
            fut, hist, _ = base[si]
            hist_b = hist.unsqueeze(0).to(device)
            # only need Z_q
            env = SequentialF2FEnvironment(model)
            z_q = env.execute_quarter_prefix(hist_b)["Z_q"].squeeze(0).detach().cpu().float()
            n_from_teacher += 1
        writer.add(
            {
                "sample_index": si,
                "fold_id": fold,
                "teacher_checkpoint_hash": thash,
                "Z_q": z_q,
                "losses_DMQF": losses,
                "gains_DMQF": gains,
            }
        )

    # global scale from all selected gains M,Q,F
    all_g = []
    # re-read from writer records via rebuilding gains from oracle selected
    for r in selected:
        L = torch.tensor(r["true_route_losses"], dtype=torch.float32)
        g = centered_terminal_returns(L.unsqueeze(0)).squeeze(0)
        all_g.append(g[1:])  # M,Q,F
    gains_cat = torch.cat(all_g, dim=0)
    scale = global_return_scale_from_gains(gains_cat)

    costs = derive_additive_stage_costs(12)
    meta = {
        "oracle_path": str(oracle_path),
        "oracle_hash": oracle_hash,
        "n_from_v2_zq": n_from_v2,
        "n_from_teacher_forward": n_from_teacher,
        "fold_counts": {str(f): len([r for r in selected if int(r["teacher_fold"]) == f]) for f in (1, 2, 3, 4)},
        "global_return_scale": scale,
        "scale_method": "IQR/1.349",
        "stage_costs": {
            "c_f": costs.c_f,
            "c_m": costs.c_m,
            "c_q": costs.c_q,
            "route_costs": costs.route_costs,
        },
        "elapsed_sec": time.time() - t0,
        "note": "X not stored; loaded live from IndexedTimeSeriesForecastingDataset. No H_shared.",
    }
    writer.finalize(meta)
    return meta


def audit_dataset_ordering(cache: BellmanOOFCache, max_check: int = 512) -> dict[str, Any]:
    ok = 0
    bad = 0
    for si in cache.sample_indices()[:max_check]:
        row = cache.get(si)
        L = row["losses_DMQF"]
        G = row["gains_DMQF"]
        if int(torch.argmin(L)) == int(torch.argmax(G)):
            # also verify g_D==0 and g=L_D-L
            if abs(float(G[0])) < 1e-8 and torch.allclose(G[1:], L[0] - L[1:], atol=1e-5):
                ok += 1
            else:
                bad += 1
        else:
            bad += 1
    return {"checked": ok + bad, "ok": ok, "bad": bad, "pass": bad == 0}
