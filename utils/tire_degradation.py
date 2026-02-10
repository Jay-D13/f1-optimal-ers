"""Utilities for scalar tire degradation scheduling."""

import numpy as np


def build_lap_grip_scales(
    n_laps: int,
    wear_rate: float,
    min_scale: float,
) -> np.ndarray:
    """
    Build per-lap scalar grip multipliers.
    """
    if n_laps < 1:
        raise ValueError("n_laps must be >= 1")
    if wear_rate < 0.0:
        raise ValueError("wear_rate must be >= 0")
    if not (0.0 < min_scale <= 1.0):
        raise ValueError("min_scale must be in (0, 1]")
    scales = np.ones(n_laps, dtype=float)
    for lap_idx in range(n_laps):
        scales[lap_idx] = max(min_scale, 1.0 - wear_rate * lap_idx)

    return scales
