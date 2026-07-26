from __future__ import annotations

from typing import Callable, Iterable, Optional

import numpy as np
import torch


def _valid_mask(labels: torch.Tensor, null_val: float) -> torch.Tensor:
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        eps = 5e-5
        mask = ~torch.isclose(
            labels,
            torch.tensor(null_val, device=labels.device, dtype=labels.dtype).expand_as(labels),
            atol=eps,
            rtol=0.0,
        )
    return mask.float()


def forecast_state_token_mae(
    stage_preds: Iterable[torch.Tensor],
    stage_targets: Iterable[torch.Tensor],
    null_val: float = 0.0,
    rescale_pair: Optional[Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]] = None,
) -> torch.Tensor:
    """Token-normalized MAE over forecast-state stages (no artificial stage weights).

    L = sum_s sum_valid |pred_s - target_s| / sum_s valid_token_count_s
    """
    numerator = None
    denominator = None
    for pred, target in zip(stage_preds, stage_targets):
        if rescale_pair is not None:
            pred, target = rescale_pair(pred, target)
        mask = _valid_mask(target, null_val)
        abs_err = torch.abs(pred - target) * mask
        stage_num = abs_err.sum()
        stage_den = mask.sum()
        numerator = stage_num if numerator is None else numerator + stage_num
        denominator = stage_den if denominator is None else denominator + stage_den

    if numerator is None or denominator is None:
        raise ValueError("forecast_state_token_mae requires at least one stage.")
    return numerator / torch.clamp(denominator, min=1.0)
