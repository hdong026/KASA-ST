"""Deduplicated forecast-refinement gain supervision from route-loss oracles."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from basicts.archs.arch_zoo.ChainForecasting_arch.forecast_refinement_routes import (
    build_refinement_route_index_map,
    gains_from_route_losses,
    route_key,
)
from basicts.data.indexed_timeseries_dataset import IndexedTimeSeriesForecastingDataset
from basicts.data.route_quality_dataset import dedupe_route_loss_records, load_oracle_json


class ForecastRefinementGainDataset(Dataset):
    """One history + gain targets per sample_index (eta-deduplicated).

    target_scale = raw_physical_mae_gain
    """

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
        horizon = int(self.metadata.get("horizon") or expected_horizon or self.candidate_routes[0][-1])
        self.horizon = horizon
        self.index_map = build_refinement_route_index_map(self.candidate_routes, horizon)
        self.sample_indices = packed["sample_indices"]
        self.route_losses = packed["route_losses"]
        self.target_scale = "raw_physical_mae_gain"

        self.gains: dict[int, list[float]] = {}
        for si, losses in self.route_losses.items():
            by_name = {
                "direct": losses[self.index_map["direct"]],
                "half": losses[self.index_map["half"]],
                "quarter": losses[self.index_map["quarter"]],
                "progressive": losses[self.index_map["progressive"]],
            }
            g = gains_from_route_losses(by_name)
            self.gains[si] = [g["g3"], g["g6"], g["g36"]]

        if require_len_match and len(self.base) != len(self.sample_indices):
            raise RuntimeError(
                f"forecasting dataset length {len(self.base)} != "
                f"unique oracle samples {len(self.sample_indices)}"
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
                    f"history/sample_index misaligned: got {si}, expected {sample_index}"
                )
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            _future, history = item
        else:
            raise ValueError(f"unsupported base item: {type(item)}")
        gains = torch.tensor(self.gains[sample_index], dtype=torch.float32)
        losses = torch.tensor(self.route_losses[sample_index], dtype=torch.float32)
        return history, int(sample_index), gains, losses


def collate_refinement_gains(batch):
    histories, indices, gains, losses = zip(*batch)
    return (
        torch.stack(list(histories), dim=0),
        torch.tensor(list(indices), dtype=torch.long),
        torch.stack(list(gains), dim=0),
        torch.stack(list(losses), dim=0),
    )


def lookup_route_losses_by_tuple(
    route_final_losses: list[dict[str, Any]],
    candidate_routes: list[list[int]],
) -> list[float]:
    """Tuple-safe loss vector in candidate order (never assume JSON index)."""
    by_key = {
        route_key(entry["route"]): float(entry["final_mae"]) for entry in route_final_losses
    }
    out = []
    for r in candidate_routes:
        k = route_key(r)
        if k not in by_key:
            raise KeyError(f"missing route {list(k)} in route_final_losses")
        out.append(by_key[k])
    return out
