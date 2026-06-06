import numpy as np
import torch
from torch.nn import HuberLoss


def masked_huber(prediction: torch.Tensor, target: torch.Tensor, reduction='mean', delta=1.0, null_val: float = np.nan) -> torch.Tensor:
    """Masked Huber loss (from GestaltCogTeam/BasicTS @ eb65f4b)."""
    if np.isnan(null_val):
        mask = ~torch.isnan(target)
    else:
        eps = 5e-5
        mask = ~torch.isclose(target, torch.tensor(null_val).to(target.device), atol=eps)

    mask = mask.float()
    prediction, target = prediction * mask, target * mask

    prediction = torch.nan_to_num(prediction)
    target = torch.nan_to_num(target)

    loss = HuberLoss(reduction, delta)(prediction, target)

    return loss
