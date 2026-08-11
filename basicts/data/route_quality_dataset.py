"""Deduplicated route-quality supervision from forecasting oracles."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from basicts.data.indexed_timeseries_dataset import IndexedTimeSeriesForecastingDataset


def load_oracle_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dedupe_route_loss_records(
    oracle: dict[str, Any],
    *,
    expected_routes: list[list[int]] | None = None,
    expected_costs: list[float] | None = None,
    expected_horizon: int | None = None,
    expected_dataset: str | None = None,
    expected_checkpoint_hash: str | None = None,
    atol: float = 1e-5,
) -> dict[str, Any]:
    """Collapse intensity-duplicated oracle records into one loss vector per sample."""
    meta = oracle.get("metadata") or {}
    records = oracle.get("records") or []
    if not records:
        raise ValueError("oracle has no records")

    routes = meta.get("candidate_routes")
    if routes is None:
        raise ValueError("oracle metadata missing candidate_routes")
    routes = [[int(x) for x in r] for r in routes]
    costs = [float(x) for x in (meta.get("route_costs") or [])]
    # Defensive fallback for older/partial merges that omitted route_costs.
    if not costs and records and isinstance(records[0].get("route_final_losses"), list):
        try:
            costs = [float(e["cost"]) for e in records[0]["route_final_losses"]]
            meta = dict(meta)
            meta["route_costs"] = costs
        except Exception:
            costs = []
    if expected_routes is not None and routes != [[int(x) for x in r] for r in expected_routes]:
        raise RuntimeError(
            f"oracle route order mismatch: got {routes}, expected {expected_routes}"
        )
    if expected_costs is not None:
        if len(costs) != len(expected_costs):
            raise RuntimeError(
                f"oracle route_costs length mismatch: "
                f"oracle_len={len(costs)} expected_len={len(expected_costs)} "
                f"oracle_costs={costs}"
            )
        for a, b in zip(costs, expected_costs):
            if abs(float(a) - float(b)) > atol:
                raise RuntimeError(
                    f"oracle route_costs mismatch: {costs} vs {expected_costs}"
                )
    if expected_horizon is not None and int(meta.get("horizon")) != int(expected_horizon):
        raise RuntimeError(
            f"oracle horizon mismatch: {meta.get('horizon')} vs {expected_horizon}"
        )
    if expected_dataset is not None and str(meta.get("dataset")) != str(expected_dataset):
        raise RuntimeError(
            f"oracle dataset mismatch: {meta.get('dataset')} vs {expected_dataset}"
        )
    if (
        expected_checkpoint_hash is not None
        and str(meta.get("checkpoint_hash")) != str(expected_checkpoint_hash)
    ):
        raise RuntimeError(
            f"oracle checkpoint_hash mismatch: {meta.get('checkpoint_hash')} "
            f"vs {expected_checkpoint_hash}"
        )

    by_sample: dict[int, list[list[float]]] = {}
    for rec in records:
        si = int(rec["sample_index"])
        losses = [float(x["final_mae"]) for x in rec["route_final_losses"]]
        if len(losses) != len(routes):
            raise RuntimeError(
                f"sample {si}: route_final_losses length {len(losses)} != R={len(routes)}"
            )
        # Validate route order inside record
        for i, entry in enumerate(rec["route_final_losses"]):
            if [int(x) for x in entry["route"]] != routes[i]:
                raise RuntimeError(
                    f"sample {si}: route_final_losses[{i}] route mismatch "
                    f"{entry['route']} vs {routes[i]}"
                )
        by_sample.setdefault(si, []).append(losses)

    unique_indices = sorted(by_sample.keys())
    loss_table: dict[int, list[float]] = {}
    for si in unique_indices:
        vecs = by_sample[si]
        ref = vecs[0]
        for v in vecs[1:]:
            if len(v) != len(ref) or any(abs(a - b) > atol for a, b in zip(v, ref)):
                raise RuntimeError(
                    f"sample {si}: route_final_losses inconsistent across intensities: "
                    f"{ref} vs {v}"
                )
        loss_table[si] = ref

    n_meta = meta.get("n_samples")
    if n_meta is not None and int(n_meta) != len(loss_table):
        raise RuntimeError(
            f"n_unique_samples={len(loss_table)} != metadata.n_samples={n_meta}"
        )

    return {
        "metadata": meta,
        "sample_indices": unique_indices,
        "route_losses": loss_table,
        "candidate_routes": routes,
        "route_costs": costs,
        "n_samples": len(loss_table),
    }


class RouteQualityDataset(Dataset):
    """One history + one route-loss vector per sample_index (eta-deduplicated)."""

    def __init__(
        self,
        forecasting_dataset: IndexedTimeSeriesForecastingDataset | Dataset,
        oracle_path: str | Path,
        *,
        expected_routes: list[list[int]] | None = None,
        expected_costs: list[float] | None = None,
        expected_horizon: int | None = None,
        expected_dataset: str | None = None,
        expected_checkpoint_hash: str | None = None,
        require_len_match: bool = True,
    ):
        self.base = forecasting_dataset
        oracle = load_oracle_json(oracle_path)
        packed = dedupe_route_loss_records(
            oracle,
            expected_routes=expected_routes,
            expected_costs=expected_costs,
            expected_horizon=expected_horizon,
            expected_dataset=expected_dataset,
            expected_checkpoint_hash=expected_checkpoint_hash,
        )
        self.metadata = packed["metadata"]
        self.candidate_routes = packed["candidate_routes"]
        self.route_costs = packed["route_costs"]
        self.sample_indices = packed["sample_indices"]
        self.route_losses = packed["route_losses"]

        if require_len_match and len(self.base) != len(self.sample_indices):
            raise RuntimeError(
                f"forecasting dataset length {len(self.base)} != "
                f"unique oracle samples {len(self.sample_indices)}"
            )
        # Ensure contiguous 0..N-1 indexing for train/valid splits
        expected = list(range(len(self.sample_indices)))
        if self.sample_indices != expected:
            # allow non-contiguous only if every index is in base
            missing = [i for i in self.sample_indices if i < 0 or i >= len(self.base)]
            if missing:
                raise RuntimeError(
                    f"oracle sample_index out of range for dataset len={len(self.base)}: "
                    f"{missing[:5]}"
                )

    def __len__(self) -> int:
        return len(self.sample_indices)

    def __getitem__(self, index: int):
        sample_index = int(self.sample_indices[index])
        item = self.base[sample_index]
        if isinstance(item, (list, tuple)) and len(item) == 3:
            _future, history, si = item
            if int(si) != sample_index:
                raise RuntimeError(
                    f"history/sample_index misaligned: dataset returned {si}, "
                    f"expected {sample_index}"
                )
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            _future, history = item
        else:
            raise ValueError(f"unsupported base dataset item arity: {type(item)}")
        losses = torch.tensor(self.route_losses[sample_index], dtype=torch.float32)
        return history, int(sample_index), losses


def collate_route_quality(batch):
    histories, indices, losses = zip(*batch)
    history = torch.stack(list(histories), dim=0)
    sample_index = torch.tensor(list(indices), dtype=torch.long)
    route_losses = torch.stack(list(losses), dim=0)
    return history, sample_index, route_losses
