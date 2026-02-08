from __future__ import annotations

from typing import Dict, Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from models import F1TrackModel
from visualization.plot_config import DEFAULT_COLORS, apply_plot_style, get_soc_bounds


def _extract_track_xy(track_model: F1TrackModel) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Get track coordinates from telemetry first, then segment fallback."""
    if track_model.telemetry_data is not None:
        telemetry = track_model.telemetry_data
        x_track = telemetry["X"].values
        y_track = telemetry["Y"].values
        s_track = telemetry["Distance"].values
        return x_track, y_track, s_track

    if track_model.segments:
        x_track = np.array([seg.x for seg in track_model.segments], dtype=float)
        y_track = np.array([seg.y for seg in track_model.segments], dtype=float)
        s_track = np.array([seg.distance for seg in track_model.segments], dtype=float)
        return x_track, y_track, s_track

    return None, None, None


def _compute_frame_times(
    times: np.ndarray,
    timing_mode: Literal["physical", "smooth"],
    fps: int,
    playback_rate: float,
) -> np.ndarray:
    """Build frame timestamps used by the animation engine."""
    duration = float(times[-1] - times[0])
    if duration <= 0:
        return np.array([times[0]])

    fps = max(int(fps), 1)
    playback_rate = max(float(playback_rate), 0.01)

    if timing_mode == "smooth":
        frame_count = max(int(np.ceil(duration * fps * playback_rate)), 2)
        return np.linspace(times[0], times[-1], frame_count)

    # Physical mode: frame time step equals simulated time step / playback rate.
    dt_sim = playback_rate / fps
    frame_count = max(int(np.floor(duration / dt_sim)) + 1, 2)
    frame_times = times[0] + np.arange(frame_count) * dt_sim
    frame_times[-1] = times[-1]
    return frame_times


def _trail_window_mask(frame_s: np.ndarray, frame_idx: int, trail_length_m: float) -> np.ndarray:
    """Boolean mask selecting positions within trail window up to frame index."""
    current_s = float(frame_s[frame_idx])
    return frame_s[: frame_idx + 1] >= (current_s - float(trail_length_m))


def visualize_lap_animated(
    track_model: F1TrackModel,
    results: Dict,
    strategy_name: str = "MPC",
    save_path: Optional[str] = "figures/lap_animation.gif",
    timing_mode: Literal["physical", "smooth"] = "physical",
    fps: int = 20,
    playback_rate: float = 1.0,
    trail_length_m: float = 120.0,
    ers_config=None,
):
    """Create animated lap visualization with physically meaningful timing."""
    x_track, y_track, s_track = _extract_track_xy(track_model)
    if x_track is None or y_track is None or s_track is None:
        print("No track coordinate data for animation")
        return None, None

    states = np.asarray(results["states"])
    times = np.asarray(results["times"], dtype=float)
    if states.shape[0] != times.shape[0]:
        raise ValueError("Animation input mismatch: results['states'] and results['times'] lengths differ")

    if states.shape[1] < 3:
        raise ValueError("Animation input requires state columns: [distance, speed, soc]")

    # Ensure interpolation domain is sorted and unique.
    order = np.argsort(s_track)
    s_track = s_track[order]
    x_track = x_track[order]
    y_track = y_track[order]
    s_track, unique_idx = np.unique(s_track, return_index=True)
    x_track = x_track[unique_idx]
    y_track = y_track[unique_idx]

    car_distances = states[:, 0]
    car_speeds = states[:, 1]
    car_soc = states[:, 2]

    frame_times = _compute_frame_times(times, timing_mode=timing_mode, fps=fps, playback_rate=playback_rate)
    frame_s = np.interp(frame_times, times, car_distances)
    frame_v = np.interp(frame_times, times, car_speeds)
    frame_soc = np.interp(frame_times, times, car_soc)

    max_track_s = float(s_track[-1])
    if np.max(frame_s) > max_track_s + 1e-6:
        s_interp = np.mod(frame_s, max_track_s)
    else:
        s_interp = np.clip(frame_s, s_track[0], max_track_s)

    frame_x = np.interp(s_interp, s_track, x_track)
    frame_y = np.interp(s_interp, s_track, y_track)

    apply_plot_style()
    fig = plt.figure(figsize=(16, 9))
    ax_track = plt.subplot2grid((2, 3), (0, 0), colspan=2, rowspan=2)
    ax_speed = plt.subplot2grid((2, 3), (0, 2))
    ax_soc = plt.subplot2grid((2, 3), (1, 2))

    ax_track.plot(x_track, y_track, color=DEFAULT_COLORS["track"], linewidth=2, alpha=0.35, label="Track")
    ax_track.set_xlabel("X (m)")
    ax_track.set_ylabel("Y (m)")
    ax_track.set_title(f"{strategy_name} Strategy - Animated Lap ({timing_mode})")
    ax_track.set_aspect("equal")

    trail_line, = ax_track.plot([], [], color=DEFAULT_COLORS["car"], linewidth=2.6, alpha=0.55)
    car_point = ax_track.scatter([frame_x[0]], [frame_y[0]], c=DEFAULT_COLORS["car"], s=80, zorder=10)

    ax_speed.set_xlim(float(frame_times[0]), float(frame_times[-1]))
    ax_speed.set_ylim(0.0, float(np.max(frame_v) * 3.6 * 1.12))
    ax_speed.set_xlabel("Time (s)")
    ax_speed.set_ylabel("Speed (km/h)")
    ax_speed.set_title("Speed")
    speed_line, = ax_speed.plot([], [], color=DEFAULT_COLORS["velocity_primary"], linewidth=2)
    speed_point = ax_speed.scatter([], [], c=DEFAULT_COLORS["velocity_primary"], s=60, zorder=10)

    ax_soc.set_xlim(float(frame_times[0]), float(frame_times[-1]))
    ax_soc.set_ylim(0.0, 100.0)
    ax_soc.set_xlabel("Time (s)")
    ax_soc.set_ylabel("SOC (%)")
    ax_soc.set_title("State of Charge")
    ax_soc.axhline(y=soc_min, color=DEFAULT_COLORS["constraint"], linestyle="--", alpha=0.5)
    ax_soc.axhline(y=soc_max, color=DEFAULT_COLORS["constraint"], linestyle="--", alpha=0.5)
    soc_line, = ax_soc.plot([], [], color=DEFAULT_COLORS["soc"], linewidth=2)
    soc_point = ax_soc.scatter([], [], c=DEFAULT_COLORS["soc"], s=60, zorder=10)

    info_text = ax_track.text(
        0.02,
        0.98,
        "",
        transform=ax_track.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.88),
    )

    def init():
        trail_line.set_data([], [])
        speed_line.set_data([], [])
        soc_line.set_data([], [])
        return trail_line, speed_line, soc_line, car_point, speed_point, soc_point, info_text

    def animate(frame_idx: int):
        current_s = frame_s[frame_idx]
        current_v = frame_v[frame_idx]
        current_soc = frame_soc[frame_idx]
        current_t = frame_times[frame_idx]

        # Distance-window trail instead of fixed number of points.
        trail_mask = _trail_window_mask(frame_s, frame_idx, trail_length_m)
        trail_line.set_data(frame_x[: frame_idx + 1][trail_mask], frame_y[: frame_idx + 1][trail_mask])
        car_point.set_offsets([[frame_x[frame_idx], frame_y[frame_idx]]])

        speed_line.set_data(frame_times[: frame_idx + 1], frame_v[: frame_idx + 1] * 3.6)
        speed_point.set_offsets([[current_t, current_v * 3.6]])

        soc_line.set_data(frame_times[: frame_idx + 1], frame_soc[: frame_idx + 1] * 100.0)
        soc_point.set_offsets([[current_t, current_soc * 100.0]])

        progress = (current_s % max(track_model.total_length, 1.0)) / max(track_model.total_length, 1.0) * 100.0
        info_text.set_text(
            f"Time: {current_t:.2f}s\n"
            f"Distance: {current_s:.1f}m ({progress:.1f}%)\n"
            f"Speed: {current_v * 3.6:.1f} km/h\n"
            f"SOC: {current_soc * 100.0:.1f}%"
        )

        return trail_line, speed_line, soc_line, car_point, speed_point, soc_point, info_text

    anim = FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames=len(frame_times),
        interval=1000.0 / max(int(fps), 1),
        blit=True,
        repeat=True,
    )

    plt.tight_layout()

    if save_path:
        print("Saving animation (this may take a while)...")
        anim.save(save_path, writer="pillow", fps=max(int(fps), 1), dpi=110)
        print(f"Animation saved to {save_path}")

    return fig, anim
    soc_min, soc_max = get_soc_bounds(ers_config)
