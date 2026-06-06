import numpy as np
import torch

from ..metrics import masked_mae


def stwave_masked_mae(prediction: torch.Tensor, target: torch.Tensor, null_val: float = np.nan) -> torch.Tensor:
    """STWave composite loss (from GestaltCogTeam/BasicTS @ eb65f4b baselines/STWave/loss.py)."""
    lloss = masked_mae(prediction[..., 1:2], prediction[..., 2:], null_val)
    loss = masked_mae(prediction[..., :1], target, null_val)
    return loss + lloss
