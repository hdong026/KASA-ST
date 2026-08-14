"""fp16 sharded trajectory + prefix-state cache (not JSON tensors)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import torch

from basicts.archs.arch_zoo.ForecastTrajectory_arch.trajectory_graph import (
    ForecastTrajectoryGraph,
)


def _prefix_key(prefix: tuple[int, ...]) -> str:
    return "start" if not prefix else "-".join(str(s) for s in prefix)


class TrajectoryCacheWriter:
    def __init__(
        self,
        out_dir: str | Path,
        graph: ForecastTrajectoryGraph,
        shard_size: int = 256,
    ):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.graph = graph
        self.shard_size = int(shard_size)
        self._buf: list[dict[str, Any]] = []
        self._shard_id = 0
        self._count = 0
        self.shard_files: list[str] = []

    def add(
        self,
        sample_index: int,
        history_summary: torch.Tensor,
        prefix_z: dict[tuple[int, ...], Optional[torch.Tensor]],
        traj_metrics: dict[str, dict],
    ) -> None:
        rec = {
            "sample_index": int(sample_index),
            "history_summary": history_summary.detach().cpu().to(torch.float16),
            "prefix_z": {},
            "traj_metrics": traj_metrics,
        }
        for pref, z in prefix_z.items():
            key = _prefix_key(pref)
            rec["prefix_z"][key] = None if z is None else z.detach().cpu().to(torch.float16)
        self._buf.append(rec)
        self._count += 1
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
        }
        if extra_manifest:
            man.update(extra_manifest)
        (self.out_dir / "manifest.json").write_text(json.dumps(man, indent=2))
        return man


class TrajectoryCache:
    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)
        self.manifest = json.loads((self.cache_dir / "manifest.json").read_text())
        self._index: dict[int, tuple[int, int]] = {}  # sample -> (shard, pos)
        self._shards: dict[int, list] = {}
        self._sample_ids: list[int] = []
        for si, f in enumerate(self.manifest["shard_files"]):
            recs = torch.load(f, map_location="cpu", weights_only=False)
            self._shards[si] = recs
            for pi, rec in enumerate(recs):
                sid = int(rec["sample_index"])
                self._index[sid] = (si, pi)
                self._sample_ids.append(sid)

    def __len__(self) -> int:
        return len(self._sample_ids)

    def sample_indices(self) -> list[int]:
        return list(self._sample_ids)

    def get(self, sample_index: int) -> dict:
        si, pi = self._index[int(sample_index)]
        return self._shards[si][pi]
