import torch

from ...utils.temporal_features import adapt_tod_dow_for_scaled_embedding
from .simple_tsf_runner import SimpleTimeSeriesForecastingRunner


class STAEformerRunner(SimpleTimeSeriesForecastingRunner):
    """STAEformer runner with PeMS04 DoW channel adaptation."""

    def forward(self, data: tuple, epoch: int = None, iter_num: int = None, train: bool = True, **kwargs) -> tuple:
        future_data, history_data = data
        history_data = adapt_tod_dow_for_scaled_embedding(history_data)
        future_data = adapt_tod_dow_for_scaled_embedding(future_data)
        return super().forward((future_data, history_data), epoch=epoch, iter_num=iter_num, train=train, **kwargs)
