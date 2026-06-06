import torch
from torch import nn

from ...utils.temporal_features import adapt_tod_dow_for_scaled_embedding
from ..base_tsf_runner import BaseTimeSeriesForecastingRunner


class STDNRunner(BaseTimeSeriesForecastingRunner):
    """Runner for STDN (ported from BasicTS eb65f4b to v0.2 tuple forward API)."""

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.forward_features = cfg["MODEL"].get("FORWARD_FEATURES", None)
        self.target_features = cfg["MODEL"].get("TARGET_FEATURES", None)
        self.lpls = self.to_running_device(cfg["MODEL"]["LPLS"])
        self.freq = cfg["MODEL"]["PARAM"]["args"]["Data"]["time_slice_size"]

    @staticmethod
    def define_model(cfg: dict) -> nn.Module:
        model = cfg["MODEL"]["ARCH"](**cfg["MODEL"]["PARAM"])
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return model

    def select_input_features(self, data: torch.Tensor) -> torch.Tensor:
        if self.forward_features is not None:
            data = data[:, :, :, self.forward_features]
        return data

    def select_target_features(self, data: torch.Tensor) -> torch.Tensor:
        data = data[:, :, :, self.target_features]
        return data

    def forward(self, data: tuple, epoch: int = None, iter_num: int = None, train: bool = True, **kwargs) -> tuple:
        future_data, history_data = data
        history_data = self.to_running_device(history_data)
        future_data = self.to_running_device(future_data)
        batch_size, length, num_nodes, _ = future_data.shape

        history_data = adapt_tod_dow_for_scaled_embedding(self.select_input_features(history_data))
        future_data_4_dec = adapt_tod_dow_for_scaled_embedding(self.select_input_features(future_data))

        if not train:
            future_data_4_dec[..., 0] = torch.empty_like(future_data_4_dec[..., 0])

        mode = "train" if train else "test"
        times_all_day = 24 * 60 / self.freq

        x = history_data[..., [0]]
        te_h = history_data[..., [2, 1]]
        te_f = future_data_4_dec[..., [2, 1]]
        te_h = te_h * torch.tensor([7, times_all_day], device=te_h.device).view(1, 1, 2)
        te_f = te_f * torch.tensor([7, times_all_day], device=te_f.device).view(1, 1, 2)
        te = torch.cat([te_h, te_f], dim=1)
        te = te[:, :, 0, :].squeeze(2)

        x = self.to_running_device(x)
        te = self.to_running_device(te)

        prediction_data = self.model(x, te, self.lpls, mode)
        assert list(prediction_data.shape)[:3] == [batch_size, length, num_nodes]

        prediction = self.select_target_features(prediction_data)
        real_value = self.select_target_features(future_data)
        return prediction, real_value
