"""TF-STGN losses including optional frequency-domain alignment."""

import torch
import torch.nn.functional as F

from basicts.metrics import masked_mae


def frequency_alignment_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """FFT amplitude MSE along the forecast horizon."""
    pred_fft = torch.fft.rfft(prediction, dim=1)
    target_fft = torch.fft.rfft(target, dim=1)
    return F.mse_loss(pred_fft.abs(), target_fft.abs())


def masked_mae_with_frequency_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    null_val: float = 0.0,
    lambda_freq: float = 0.0,
) -> torch.Tensor:
    main_loss = masked_mae(prediction, target, null_val)
    if lambda_freq <= 0:
        return main_loss
    freq_loss = frequency_alignment_loss(prediction, target)
    return main_loss + lambda_freq * freq_loss
