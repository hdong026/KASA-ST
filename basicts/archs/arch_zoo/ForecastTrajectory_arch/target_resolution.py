"""Resolution targets Y^(s) from a full H-step forecast target."""

from __future__ import annotations

import math
from typing import Optional

import torch


def resolution_bin_bounds(H: int, s: int, k: int) -> tuple[int, int]:
    """Half-open interval [start, end) for bin ``k`` of destination resolution ``s``."""
    if not (0 <= k < s):
        raise ValueError(f"bin k={k} out of range for s={s}")
    start = int(math.floor(k * H / s))
    end = int(math.floor((k + 1) * H / s))
    if end <= start:
        # Degenerate bin: keep a single token at start (clipped).
        end = min(start + 1, H)
        start = min(start, H - 1)
    return start, end


def build_resolution_target(
    Y: torch.Tensor,
    s: int,
    H: Optional[int] = None,
) -> torch.Tensor:
    """Average-pool full-horizon target ``Y`` into ``s`` resolution bins.

    Args:
        Y: ``[B, H, N, Cy]`` (or any trailing dims after time).
        s: destination resolution (number of bins).
        H: optional explicit horizon; defaults to ``Y.shape[1]``.

    Returns:
        ``[B, s, N, Cy]`` (same trailing dims as ``Y``).
    """
    if Y.ndim < 2:
        raise ValueError(f"Y must have at least 2 dims [B,H,...], got {tuple(Y.shape)}")
    s = int(s)
    if s <= 0:
        raise ValueError(f"s must be positive, got {s}")
    horizon = int(Y.shape[1] if H is None else H)
    if int(Y.shape[1]) != horizon:
        raise ValueError(f"Y time dim {Y.shape[1]} != H={horizon}")
    if s == horizon:
        return Y.clone()

    batch = Y.shape[0]
    tail = Y.shape[2:]
    bins = []
    for k in range(s):
        start, end = resolution_bin_bounds(horizon, s, k)
        chunk = Y[:, start:end]
        bins.append(chunk.mean(dim=1))
    out = torch.stack(bins, dim=1)
    expected = (batch, s) + tuple(tail)
    if tuple(out.shape) != expected:
        raise RuntimeError(f"resolution target shape {tuple(out.shape)} != {expected}")
    return out


def assert_resolution_target_shapes(Y: torch.Tensor, states: list[int]) -> None:
    H = int(Y.shape[1])
    for s in states:
        y_s = build_resolution_target(Y, s, H=H)
        if y_s.shape[1] != int(s):
            raise AssertionError(f"s={s}: expected time={s}, got {tuple(y_s.shape)}")
        if y_s.shape[0] != Y.shape[0] or y_s.shape[2:] != Y.shape[2:]:
            raise AssertionError(f"s={s}: shape mismatch {tuple(y_s.shape)} vs {tuple(Y.shape)}")
        if int(s) == H:
            if not torch.allclose(y_s, Y, atol=1e-6, rtol=1e-6):
                max_abs = float((y_s - Y).abs().max().item())
                raise AssertionError(f"Y^(H) must equal Y, max_abs={max_abs}")
