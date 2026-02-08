from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from models import F1TrackModel
from solvers import OptimalTrajectory
from visualization.plot_config import DEFAULT_COLORS, apply_plot_style, get_soc_bounds


def _align_to_reference(x_ref: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Align 1D array length to reference axis length using interpolation."""
    if len(y) == len(x_ref):
        return y
    src = np.linspace(x_ref[0], x_ref[-1], len(y))
    return np.interp(x_ref, src, y)


def _compute_cumulative_ers_energy_mj(trajectory: OptimalTrajectory) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute cumulative deployed/recovered/net ERS energy using true time steps."""
    t = np.asarray(trajectory.t_opt, dtype=float)
    p_ers = np.asarray(trajectory.P_ers_opt, dtype=float)
    n = min(len(p_ers), max(len(t) - 1, 0))

    if n == 0:
        return np.array([]), np.array([]), np.array([])

    dt = np.diff(t[: n + 1])
    deployed = np.maximum(p_ers[:n], 0.0) * dt / 1e6
    recovered = np.maximum(-p_ers[:n], 0.0) * dt / 1e6

    cumulative_deployed = np.cumsum(deployed)
    cumulative_recovered = np.cumsum(recovered)
    cumulative_net = cumulative_deployed - cumulative_recovered
    return cumulative_deployed, cumulative_recovered, cumulative_net


def plot_offline_solution(
    trajectory: OptimalTrajectory,
    title: str = "Offline Optimal Solution",
    save_path: Optional[str] = None,
    ers_config=None,
) -> plt.Figure:
    apply_plot_style()
    soc_min, soc_max = get_soc_bounds(ers_config)

    fig, axes = plt.subplots(3, 2, figsize=(14, 11))
    s_km = trajectory.s / 1000.0

    # Velocity
    axes[0, 0].plot(s_km, trajectory.v_opt * 3.6, color=DEFAULT_COLORS["velocity_primary"])
    axes[0, 0].set_ylabel("Velocity (km/h)")
    axes[0, 0].set_title("Optimal Velocity Profile")

    # SOC
    axes[0, 1].plot(s_km, trajectory.soc_opt * 100.0, color=DEFAULT_COLORS["soc"])
    axes[0, 1].axhline(y=soc_min, color=DEFAULT_COLORS["constraint"], linestyle="--", alpha=0.6)
    axes[0, 1].axhline(y=soc_max, color=DEFAULT_COLORS["constraint"], linestyle="--", alpha=0.6)
    axes[0, 1].set_ylabel("State of Charge (%)")
    axes[0, 1].set_ylim([0, 100])
    axes[0, 1].set_title("SOC Trajectory")

    # ERS power
    p_ers_kw = trajectory.P_ers_opt / 1000.0
    s_ers = s_km[: len(p_ers_kw)]
    axes[1, 0].fill_between(s_ers, 0, p_ers_kw, where=p_ers_kw >= 0, color=DEFAULT_COLORS["deploy"], alpha=0.6, label="+Deploy")
    axes[1, 0].fill_between(s_ers, 0, p_ers_kw, where=p_ers_kw < 0, color=DEFAULT_COLORS["harvest"], alpha=0.6, label="-Harvest")
    axes[1, 0].axhline(y=0, color="black", linewidth=0.8, alpha=0.7)
    max_power_kw = 120.0 if ers_config is None else float(ers_config.max_deployment_power / 1000.0)
    axes[1, 0].axhline(y=max_power_kw, color=DEFAULT_COLORS["constraint"], linestyle=":")
    axes[1, 0].axhline(y=-max_power_kw, color=DEFAULT_COLORS["constraint"], linestyle=":")
    axes[1, 0].set_ylabel("ERS Power (kW)")
    axes[1, 0].set_title("ERS Power Strategy")
    axes[1, 0].legend(loc="upper right")

    # Throttle / brake
    s_controls = s_km[: len(trajectory.throttle_opt)]
    axes[1, 1].fill_between(s_controls, 0, trajectory.throttle_opt * 100.0, color=DEFAULT_COLORS["deploy"], alpha=0.5, label="Throttle")
    axes[1, 1].fill_between(s_controls, 0, -trajectory.brake_opt * 100.0, color=DEFAULT_COLORS["harvest"], alpha=0.5, label="Brake")
    axes[1, 1].set_ylabel("Pedal (%)")
    axes[1, 1].set_title("Throttle and Brake")
    axes[1, 1].legend(loc="upper right")

    # Lap time accumulation
    axes[2, 0].plot(s_km, trajectory.t_opt, color=DEFAULT_COLORS["velocity_primary"])
    axes[2, 0].set_ylabel("Time (s)")
    axes[2, 0].set_xlabel("Distance (km)")
    axes[2, 0].set_title("Cumulative Lap Time")

    # Summary panel (fills previously empty slot)
    energy = trajectory.compute_energy_stats()
    axes[2, 1].axis("off")
    summary = (
        f"Lap Time: {trajectory.lap_time:.3f} s\n"
        f"Solve Time: {trajectory.solve_time:.2f} s\n"
        f"Status: {trajectory.solver_status}\n\n"
        f"Deploy: {energy['total_deployed_MJ']:.3f} MJ\n"
        f"Recover: {energy['total_recovered_MJ']:.3f} MJ\n"
        f"Net: {energy['net_energy_MJ']:.3f} MJ\n\n"
        f"SOC Start/End: {trajectory.soc_opt[0]*100:.1f}% / {trajectory.soc_opt[-1]*100:.1f}%"
    )
    axes[2, 1].text(
        0.02,
        0.98,
        summary,
        transform=axes[2, 1].transAxes,
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="#f3f3f3", alpha=0.95),
    )

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_simple_results(
    trajectory: OptimalTrajectory,
    velocity_profile,
    track,
    track_name: str,
    ers_config=None,
) -> plt.Figure:
    apply_plot_style()
    soc_min, soc_max = get_soc_bounds(ers_config)
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    s = trajectory.s / 1000.0
    v_ref = _align_to_reference(trajectory.s, velocity_profile.v) * 3.6

    axes[0].plot(s, trajectory.v_opt * 3.6, color=DEFAULT_COLORS["velocity_primary"], label="Optimal")
    axes[0].plot(s, v_ref, color=DEFAULT_COLORS["velocity_reference"], linestyle="--", alpha=0.8, label="Grip Limit")
    axes[0].set_ylabel("Velocity (km/h)")
    axes[0].set_title(f"{track_name} - Diagnostics")
    axes[0].legend(loc="upper right")

    axes[1].plot(s, trajectory.soc_opt * 100.0, color=DEFAULT_COLORS["soc"])
    axes[1].axhline(y=soc_min, color=DEFAULT_COLORS["constraint"], linestyle="--", alpha=0.6, label="SOC bounds")
    axes[1].axhline(y=soc_max, color=DEFAULT_COLORS["constraint"], linestyle="--", alpha=0.6)
    axes[1].set_ylim(0.0, 100.0)
    axes[1].set_ylabel("SOC (%)")
    axes[1].legend(loc="upper right")

    p_ers_kw = trajectory.P_ers_opt / 1000.0
    s_ers = s[: len(p_ers_kw)]
    axes[2].fill_between(s_ers, 0, p_ers_kw, where=p_ers_kw >= 0, color=DEFAULT_COLORS["deploy"], alpha=0.6, label="+Deploy")
    axes[2].fill_between(s_ers, 0, p_ers_kw, where=p_ers_kw < 0, color=DEFAULT_COLORS["harvest"], alpha=0.6, label="-Harvest")
    axes[2].axhline(y=0, color="black", linewidth=0.8, alpha=0.8)
    axes[2].set_ylabel("ERS Power (kW)")
    axes[2].legend(loc="upper right")

    segment_distances = np.array([seg.distance for seg in track.segments], dtype=float) / 1000.0
    segment_radii = np.array([seg.radius for seg in track.segments], dtype=float)
    curvature = 1000.0 / np.maximum(segment_radii, 50.0)
    axes[3].fill_between(segment_distances, curvature, 0, alpha=0.35, color=DEFAULT_COLORS["curvature"])
    axes[3].set_ylabel("Curvature (1/km)")
    axes[3].set_xlabel("Distance (km)")

    plt.tight_layout()
    return fig


def create_comparison_plot(
    track: F1TrackModel,
    velocity_no_ers: np.ndarray,
    velocity_with_ers: np.ndarray,
    optimal_trajectory: OptimalTrajectory,
    track_name: str,
    ers_config=None,
) -> plt.Figure:
    """Comprehensive diagnostics-first comparison plot."""
    apply_plot_style()
    soc_min, soc_max = get_soc_bounds(ers_config)

    s_km = optimal_trajectory.s / 1000.0
    vel_no_ers = _align_to_reference(optimal_trajectory.s, velocity_no_ers) * 3.6
    vel_with_ers = _align_to_reference(optimal_trajectory.s, velocity_with_ers) * 3.6
    vel_opt = optimal_trajectory.v_opt * 3.6

    fig = plt.figure(figsize=(16, 11))

    # 1) Velocity stack
    ax1 = plt.subplot(3, 2, 1)
    ax1.plot(s_km, vel_no_ers, color=DEFAULT_COLORS["velocity_reference"], linestyle="--", label="No ERS Limit")
    ax1.plot(s_km, vel_with_ers, color="#ff7f0e", linestyle="-", alpha=0.9, label="With ERS Limit")
    ax1.plot(s_km, vel_opt, color=DEFAULT_COLORS["velocity_primary"], linewidth=2.0, label="Optimal")
    ax1.set_title("Velocity Profiles")
    ax1.set_ylabel("Velocity (km/h)")
    ax1.set_xlabel("Distance (km)")
    ax1.legend(loc="upper right")

    # 2) Velocity delta
    ax2 = plt.subplot(3, 2, 2)
    v_delta = vel_with_ers - vel_no_ers
    ax2.fill_between(s_km, 0, v_delta, where=v_delta >= 0, color=DEFAULT_COLORS["deploy"], alpha=0.45, label="ERS advantage")
    ax2.fill_between(s_km, 0, v_delta, where=v_delta < 0, color=DEFAULT_COLORS["harvest"], alpha=0.45, label="ERS disadvantage")
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_title("Velocity Delta (With ERS - No ERS)")
    ax2.set_ylabel("Delta Speed (km/h)")
    ax2.set_xlabel("Distance (km)")
    ax2.legend(loc="upper right")

    # 3) SOC
    ax3 = plt.subplot(3, 2, 3)
    ax3.plot(s_km, optimal_trajectory.soc_opt * 100.0, color=DEFAULT_COLORS["soc"], linewidth=2)
    ax3.axhline(soc_min, color=DEFAULT_COLORS["constraint"], linestyle="--", alpha=0.6, label="SOC bounds")
    ax3.axhline(soc_max, color=DEFAULT_COLORS["constraint"], linestyle="--", alpha=0.6)
    ax3.set_ylim(0, 100)
    ax3.set_title("State of Charge")
    ax3.set_ylabel("SOC (%)")
    ax3.set_xlabel("Distance (km)")
    ax3.legend(loc="upper right")

    # 4) ERS power
    ax4 = plt.subplot(3, 2, 4)
    p_ers_kw = optimal_trajectory.P_ers_opt / 1000.0
    s_ers = s_km[: len(p_ers_kw)]
    ax4.fill_between(s_ers, 0, p_ers_kw, where=p_ers_kw >= 0, color=DEFAULT_COLORS["deploy"], alpha=0.6, label="+Deploy")
    ax4.fill_between(s_ers, 0, p_ers_kw, where=p_ers_kw < 0, color=DEFAULT_COLORS["harvest"], alpha=0.6, label="-Harvest")
    max_power_kw = 120.0 if ers_config is None else float(ers_config.max_deployment_power / 1000.0)
    ax4.axhline(max_power_kw, color=DEFAULT_COLORS["constraint"], linestyle=":")
    ax4.axhline(-max_power_kw, color=DEFAULT_COLORS["constraint"], linestyle=":")
    ax4.axhline(0, color="black", linewidth=0.8)
    ax4.set_title("ERS Power")
    ax4.set_ylabel("Power (kW)")
    ax4.set_xlabel("Distance (km)")
    ax4.legend(loc="upper right")

    # 5) Cumulative ERS energy
    ax5 = plt.subplot(3, 2, 5)
    cumulative_deployed, cumulative_recovered, cumulative_net = _compute_cumulative_ers_energy_mj(optimal_trajectory)
    s_energy = s_km[: len(cumulative_deployed)]
    ax5.plot(s_energy, cumulative_deployed, color=DEFAULT_COLORS["deploy"], label="Deploy")
    ax5.plot(s_energy, cumulative_recovered, color="#1f9ccf", label="Recover")
    ax5.plot(s_energy, cumulative_net, color="black", linestyle="--", label="Net")
    ax5.set_title("Cumulative ERS Energy")
    ax5.set_ylabel("Energy (MJ)")
    ax5.set_xlabel("Distance (km)")
    ax5.legend(loc="upper left")

    # 6) Lap time accumulation + curvature context
    ax6 = plt.subplot(3, 2, 6)
    ax6.plot(s_km, optimal_trajectory.t_opt, color=DEFAULT_COLORS["velocity_primary"], label="Cumulative Lap Time")
    seg_s = np.array([seg.distance for seg in track.segments], dtype=float) / 1000.0
    seg_r = np.array([seg.radius for seg in track.segments], dtype=float)
    curvature = 1000.0 / np.maximum(seg_r, 50.0)
    curvature_scaled = curvature / max(np.max(curvature), 1e-6) * max(np.max(optimal_trajectory.t_opt) * 0.2, 1.0)
    ax6.fill_between(seg_s, 0, curvature_scaled, color=DEFAULT_COLORS["curvature"], alpha=0.25, label="Curvature (scaled)")
    ax6.set_title("Lap Time Accumulation + Curvature Context")
    ax6.set_ylabel("Time (s)")
    ax6.set_xlabel("Distance (km)")
    ax6.legend(loc="upper left")

    plt.suptitle(f"{track_name} - ERS Diagnostics", fontsize=15, fontweight="bold")
    plt.tight_layout()
    return fig
