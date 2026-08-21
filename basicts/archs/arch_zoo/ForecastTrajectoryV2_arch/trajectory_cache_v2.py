"""fp16 sharded prefix-DAG cache: H/Z of unique prefixes computed once."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import torch

from basicts.archs.arch_zoo.ForecastTrajectory_arch.trajectory_graph import (
    ForecastTrajectoryGraph,
)


def prefix_key(prefix: tuple[int, ...]) -> str:
    return "start" if not prefix else "-".join(str(int(s)) for s in prefix)


def parse_prefix_key(key: str) -> tuple[int, ...]:
    if key in ("start", "", "()"):
        return ()
    return tuple(int(x) for x in str(key).split("-") if x)


class TrajectoryCacheV2Writer:
    def __init__(self, out_dir: str | Path, graph: ForecastTrajectoryGraph, shard_size: int = 128):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.graph = graph
        self.shard_size = int(shard_size)
        self._buf: list[dict[str, Any]] = []
        self._shard_id = 0
        self._count = 0
        self.shard_files: list[str] = []
        self.transition_counts: list[int] = []

    def add(
        self,
        sample_index: int,
        history_summary: torch.Tensor,
        prefix_h: dict[tuple[int, ...], Optional[torch.Tensor]],
        prefix_z: dict[tuple[int, ...], Optional[torch.Tensor]],
        traj_metrics: dict[str, dict],
        n_transitions: int,
    ) -> None:
        rec = {
            "sample_index": int(sample_index),
            "history_summary": history_summary.detach().cpu().to(torch.float16),
            "prefix_h": {},
            "prefix_z": {},
            "traj_metrics": traj_metrics,
            "n_transitions": int(n_transitions),
        }
        for pref, h in prefix_h.items():
            rec["prefix_h"][prefix_key(pref)] = None if h is None else h.detach().cpu().to(torch.float16)
        for pref, z in prefix_z.items():
            rec["prefix_z"][prefix_key(pref)] = None if z is None else z.detach().cpu().to(torch.float16)
        self._buf.append(rec)
        self._count += 1
        self.transition_counts.append(int(n_transitions))
        if len(self._buf) >= self.shard_size:
            self._flush()

    def _flush(self) -> None:
        if not self._buf:
            return
        path = self.out_dir / f"shard_{self._shard_id:05d}.pt"
        torch.save(self._buf, path)
        self.shard_files.append(str(path))
        self._shard_id += 1
        self._buf = []

    def close(self, extra_manifest: Optional[dict] = None) -> dict:
        self._flush()
        man = {
            "n_samples": self._count,
            "shard_size": self.shard_size,
            "shard_files": self.shard_files,
            "states": list(self.graph.states),
            "H": self.graph.H,
            "n_trajectories": len(self.graph.terminal_trajectories()),
            "n_edges": len(self.graph.legal_edges()),
            "storage": "fp16_pt_shards",
            "json_tensors": False,
            "mean_transitions": (
                float(sum(self.transition_counts) / max(len(self.transition_counts), 1))
            ),
            "v2_prefix_dag": True,
        }
        if extra_manifest:
            man.update(extra_manifest)
        (self.out_dir / "manifest.json").write_text(json.dumps(man, indent=2))
        return man


class TrajectoryCacheV2:
    """Lazy shard cache. Does not keep every shard in RAM."""

    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)
        self.manifest = json.loads((self.cache_dir / "manifest.json").read_text())
        self._index: dict[int, tuple[int, int]] = {}
        self._sample_ids: list[int] = []
        self._shards: dict[int, list] = {}
        self._shard_files = list(self.manifest["shard_files"])
        for si, f in enumerate(self._shard_files):
            recs = torch.load(f, map_location="cpu", weights_only=False)
            self._shards[si] = recs
            for pi, rec in enumerate(recs):
                sid = int(rec["sample_index"])
                self._index[sid] = (si, pi)
                self._sample_ids.append(sid)

    def __len__(self):
        return len(self._sample_ids)

    def sample_indices(self) -> list[int]:
        return list(self._sample_ids)

    def get(self, sample_index: int) -> dict:
        si, pi = self._index[int(sample_index)]
        return self._shards[si][pi]
