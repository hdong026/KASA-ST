import torch

from .dataset import TimeSeriesForecastingDataset


class IndexedTimeSeriesForecastingDataset(TimeSeriesForecastingDataset):
    """Time series forecasting dataset that also returns the stable split index."""

    def __getitem__(self, index: int) -> tuple:
        future, history = super().__getitem__(index)
        # sample_index is stable local split index (same as Dataset index)
        return future, history, int(index)
