"""Adapt PeMS04 temporal channels for embedding-based ST baselines.

KASA-ST PeMS04 data (4-channel):
  ch0 = flow
  ch1 = ToD normalized in [0, 1)
  ch2 = DoW integer in [0, 6]  (NOT normalized; STID uses direct index)

BasicTS STAEformer/STWave/STDN runners assume ch2 is normalized [0, 1] and scale by 7.
"""
import torch


def adapt_tod_dow_for_scaled_embedding(data: torch.Tensor) -> torch.Tensor:
    """Normalize DoW to [0, 1] for models/runners that multiply ch2 by 7."""
    out = data.clone()
    if out.shape[-1] > 2:
        if out[..., 2].max() > 1.0:
            out[..., 2] = out[..., 2] / 7.0
        out[..., 2] = out[..., 2].clamp(0.0, 1.0 - 1e-6)
    if out.shape[-1] > 1:
        out[..., 1] = out[..., 1].clamp(0.0, 1.0 - 1e-6)
    return out


def adapt_dow_as_class_index(data: torch.Tensor) -> torch.Tensor:
    """Ensure DoW is integer class index in [0, 6] for direct nn.Embedding(7)."""
    out = data.clone()
    if out.shape[-1] > 2:
        if out[..., 2].max() <= 1.0:
            out[..., 2] = torch.round(out[..., 2] * 7)
        out[..., 2] = out[..., 2].round().clamp(0, 6)
    if out.shape[-1] > 1:
        out[..., 1] = out[..., 1].clamp(0.0, 1.0 - 1e-6)
    return out
