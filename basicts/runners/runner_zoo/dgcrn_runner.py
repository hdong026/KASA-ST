import torch

from .simple_tsf_runner import SimpleTimeSeriesForecastingRunner


class DGCRNRunner(SimpleTimeSeriesForecastingRunner):
    """Runner for DGCRN (official BasicTS v0.2 style)."""

    def forward(self, data: tuple, epoch: int = None, iter_num: int = None, train: bool = True, **kwargs) -> tuple:
        future_data, history_data = data
        history_data = self.to_running_device(history_data)
        future_data = self.to_running_device(future_data)
        batch_size, length, num_nodes, _ = future_data.shape

        history_data = self.select_input_features(history_data)
        if train:
            future_data_4_dec = self.select_input_features(future_data)
        else:
            future_data_4_dec = self.select_input_features(future_data)
            future_data_4_dec[:, 0, ...] = torch.empty_like(future_data_4_dec[:, 0, ...])

        task_level = self.curriculum_learning(epoch)
        prediction_data = self.model(
            history_data=history_data,
            future_data=future_data_4_dec,
            batch_seen=iter_num,
            epoch=epoch,
            train=train,
            task_level=task_level,
        )
        assert list(prediction_data.shape)[:3] == [batch_size, length, num_nodes], \
            "error shape of the output, edit the forward function to reshape it to [B, L, N, C]"
        prediction = self.select_target_features(prediction_data)
        real_value = self.select_target_features(future_data)
        return prediction, real_value
