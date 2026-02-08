"""Shared plotting defaults for visualization modules."""

from __future__ import annotations

from typing import Dict, Tuple

import matplotlib.pyplot as plt

DEFAULT_COLORS: Dict[str, str] = {
    "velocity_primary": "#1f77b4",
    "velocity_reference": "#d62728",
    "deploy": "#2ca02c",
    "harvest": "#d62728",
    "soc": "#2ca02c",
    "constraint": "#7f7f7f",
    "curvature": "#9467bd",
    "track": "#111111",
    "car": "#e31a1c",
}


def apply_plot_style() -> None:
    """Apply consistent matplotlib defaults used by project plots."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "lines.linewidth": 1.8,
            "legend.frameon": True,
            "legend.framealpha": 0.9,
            "figure.dpi": 120,
        }
    )


def get_soc_bounds(ers_config=None) -> Tuple[float, float]:
    """Return SOC bounds as percentages."""
    if ers_config is None:
        return 20.0, 90.0
    return float(ers_config.min_soc * 100.0), float(ers_config.max_soc * 100.0)
