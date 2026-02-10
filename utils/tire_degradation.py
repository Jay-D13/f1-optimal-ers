"""Utilities for scalar tire degradation scheduling."""

import numpy as np


def build_lap_grip_scales(
    n_laps: int,
    wear_rate: float,
    min_scale: float,
    pit_lap_end: int | None = None,
) -> np.ndarray:
    """
    Build per-lap scalar grip multipliers.

    Convention:
    - `pit_lap_end = L` means a pit stop at the end of lap L.
    - Lap L+1 starts on fresh tires (age reset to 0).
    """
    if n_laps < 1:
        raise ValueError("n_laps must be >= 1")
    if wear_rate < 0.0:
        raise ValueError("wear_rate must be >= 0")
    if not (0.0 < min_scale <= 1.0):
        raise ValueError("min_scale must be in (0, 1]")
    if pit_lap_end is not None and not (1 <= pit_lap_end <= n_laps):
        raise ValueError("pit_lap_end must be within [1, n_laps]")

    scales = np.ones(n_laps, dtype=float)
    tire_age = 0

    for lap in range(1, n_laps + 1):
        scales[lap - 1] = max(min_scale, 1.0 - wear_rate * tire_age)
        if pit_lap_end is not None and lap == pit_lap_end:
            tire_age = 0
        else:
            tire_age += 1

    return scales
