from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from simulation.lap import MultiLapResult
from visualization.plot_config import DEFAULT_COLORS, apply_plot_style, get_soc_bounds


def _get_track_length_from_result(result: MultiLapResult) -> float:
    if result.lap_results:
        return float(np.max(result.lap_results[0].positions))
    if len(result.positions) > 0:
        return float(np.max(result.positions))
    return 0.0


def plot_multilap_overview(
    result: MultiLapResult,
    track_name: str = "Track",
    ers_config=None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Overview figure: lap time deltas, SOC start/end, deploy/harvest/net."""
    apply_plot_style()
    soc_min, soc_max = get_soc_bounds(ers_config)

    laps = np.array([s["lap"] for s in result.lap_summaries], dtype=int)
    lap_times = np.array([s["lap_time"] for s in result.lap_summaries], dtype=float)
    soc_start = np.array([s["soc_start"] * 100.0 for s in result.lap_summaries], dtype=float)
    soc_end = np.array([s["soc_end"] * 100.0 for s in result.lap_summaries], dtype=float)
    e_dep = np.array([s["energy_deployed_MJ"] for s in result.lap_summaries], dtype=float)
    e_rec = np.array([s["energy_recovered_MJ"] for s in result.lap_summaries], dtype=float)
    e_net = np.array([s["net_energy_MJ"] for s in result.lap_summaries], dtype=float)

    best = float(np.min(lap_times)) if len(lap_times) else 0.0
    lap_delta = lap_times - best

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Lap time deltas.
    axes[0].bar(laps, lap_delta, color=DEFAULT_COLORS["velocity_primary"], alpha=0.8)
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_title("Lap Time Delta vs Best")
    axes[0].set_xlabel("Lap")
    axes[0].set_ylabel("Delta Time (s)")
    for lap, dt in zip(laps, lap_delta):
        axes[0].text(lap, dt + 0.01, f"+{dt:.3f}", ha="center", va="bottom", fontsize=8)

    # SOC start/end dumbbell.
    for lap, start, end in zip(laps, soc_start, soc_end):
        axes[1].plot([lap, lap], [start, end], color=DEFAULT_COLORS["constraint"], linewidth=3, alpha=0.8)
        axes[1].scatter([lap], [start], color="#1f9ccf", label="Start SOC" if lap == laps[0] else None, zorder=3)
        axes[1].scatter([lap], [end], color=DEFAULT_COLORS["soc"], label="End SOC" if lap == laps[0] else None, zorder=3)
    axes[1].axhline(soc_min, color=DEFAULT_COLORS["constraint"], linestyle="--", alpha=0.6)
    axes[1].axhline(soc_max, color=DEFAULT_COLORS["constraint"], linestyle="--", alpha=0.6)
    axes[1].set_ylim(0.0, 100.0)
    axes[1].set_title("SOC Carry-Over (Start vs End)")
    axes[1].set_xlabel("Lap")
    axes[1].set_ylabel("SOC (%)")
    axes[1].legend(loc="upper right")

    # Energy bars.
    width = 0.25
    axes[2].bar(laps - width, e_dep, width=width, color=DEFAULT_COLORS["deploy"], label="Deploy")
    axes[2].bar(laps, e_rec, width=width, color="#1f9ccf", label="Recover")
    axes[2].bar(laps + width, e_net, width=width, color="black", alpha=0.7, label="Net")
    axes[2].set_title("Energy by Lap")
    axes[2].set_xlabel("Lap")
    axes[2].set_ylabel("Energy (MJ)")
    axes[2].legend(loc="upper right")

    fig.suptitle(f"{track_name} - Multi-Lap Overview", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_multilap_distance_heatmap(
    result: MultiLapResult,
    track_name: str = "Track",
    track_length_m: Optional[float] = None,
    ds_m: float = 5.0,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Heatmap of ERS power by lap (rows) vs distance (columns)."""
    apply_plot_style()

    if not result.lap_results:
        raise ValueError("No lap results provided for heatmap")

    if track_length_m is None:
        track_length_m = _get_track_length_from_result(result)
    if track_length_m <= 0:
        raise ValueError("Invalid track length for heatmap")

    s_grid = np.arange(0.0, track_length_m, max(ds_m, 1.0))
    n_laps = len(result.lap_results)
    heatmap = np.zeros((n_laps, len(s_grid)))

    for i, lap in enumerate(result.lap_results):
        n_ctrl = len(lap.P_ers_history)
        if n_ctrl == 0:
            continue
        s_ctrl = np.mod(lap.positions[:n_ctrl], track_length_m)
        order = np.argsort(s_ctrl)
        s_sorted = s_ctrl[order]
        p_sorted = lap.P_ers_history[order] / 1000.0
        s_unique, idx = np.unique(s_sorted, return_index=True)
        p_unique = p_sorted[idx]
        heatmap[i] = np.interp(s_grid, s_unique, p_unique, left=0.0, right=0.0)

    fig, ax = plt.subplots(figsize=(14, 4.8))
    vmax = max(np.max(np.abs(heatmap)), 1.0)
    im = ax.imshow(
        heatmap,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        origin="lower",
        extent=[0.0, track_length_m / 1000.0, 1, n_laps],
    )
    ax.set_title("ERS Power Heatmap (+Deploy / -Harvest)")
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Lap")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("ERS Power (kW)")

    fig.suptitle(f"{track_name} - Multi-Lap ERS by Distance", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_multilap_speed_overlay(
    result: MultiLapResult,
    track_name: str = "Track",
    track_length_m: Optional[float] = None,
    ds_m: float = 5.0,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Overlay speed profiles across laps plus median trace."""
    apply_plot_style()

    if not result.lap_results:
        raise ValueError("No lap results provided for speed overlay")

    if track_length_m is None:
        track_length_m = _get_track_length_from_result(result)
    if track_length_m <= 0:
        raise ValueError("Invalid track length for speed overlay")

    s_grid = np.arange(0.0, track_length_m, max(ds_m, 1.0))
    speed_matrix = []

    fig, ax = plt.subplots(figsize=(14, 6))
    for lap in result.lap_results:
        s = np.mod(lap.positions, track_length_m)
        v = lap.velocities * 3.6
        order = np.argsort(s)
        s_sorted = s[order]
        v_sorted = v[order]
        s_unique, idx = np.unique(s_sorted, return_index=True)
        v_unique = v_sorted[idx]
        v_interp = np.interp(s_grid, s_unique, v_unique, left=np.nan, right=np.nan)
        speed_matrix.append(v_interp)
        ax.plot(s_grid / 1000.0, v_interp, color=DEFAULT_COLORS["velocity_primary"], alpha=0.25, linewidth=1.3)

    speed_matrix_np = np.array(speed_matrix)
    median_speed = np.nanmedian(speed_matrix_np, axis=0)
    ax.plot(s_grid / 1000.0, median_speed, color="black", linewidth=2.4, label="Median Lap Speed")

    ax.set_title("Multi-Lap Speed Overlay")
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Speed (km/h)")
    ax.legend(loc="upper right")
    fig.suptitle(f"{track_name} - Multi-Lap Speed Consistency", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
