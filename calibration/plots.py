"""Per-round diagnostics for a parameter set: model speed against the reference lap."""
from concurrent.futures import ProcessPoolExecutor
from typing import Mapping, Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .dataset import reference_lap  # noqa: E402
from .model import model_lap  # noqa: E402


def _trace(args):
    params, round_number = args
    reference = reference_lap(round_number)
    trajectory = model_lap(params, reference)
    return round_number, reference, trajectory.s, trajectory.v_opt, trajectory.lap_time


def speed_overlays(params: Mapping[str, float], rounds: Sequence[int], path, title: str = "", workers: int = 8) -> None:
    """One panel per round: measured and model speed against lap distance, with the lap times."""
    with ProcessPoolExecutor(workers) as pool:
        traces = list(pool.map(_trace, [(dict(params), r) for r in rounds]))
    columns = 2
    rows = int(np.ceil(len(traces) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(16, 3.2 * rows), squeeze=False)
    for ax, (round_number, reference, s, v, lap_time) in zip(axes.ravel(), traces):
        ax.plot(reference.s, reference.speed * 3.6, ".", ms=2.5, color="black", label=f"{reference.driver} {reference.lap_time:.3f} s")
        ax.plot(s, v * 3.6, color="tab:red", lw=1.2, label=f"model {lap_time:.3f} s ({lap_time - reference.lap_time:+.2f})")
        ax.set_title(f"R{round_number} {reference.name}", fontsize=10)
        ax.set_ylabel("km/h")
        ax.set_xlim(0, reference.track.total_length)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(alpha=0.3)
    for ax in axes.ravel()[len(traces):]:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


STRETCH_COLOURS = {"deploy": "tab:green", "clip": "tab:red", "terminal": "tab:blue"}


def straight_panels(prepared, path, models, model_speed, highlight=(), implied=None, title: str = "") -> None:
    """
    One panel per full-throttle run of a lap (calibration/straights.py): measured speed (dots), the 1 s smoothed
    speed (grey), the (a) deploy, (b) clip and (c) terminal stretches shaded (pale where left out of the fit),
    and the model's speed on each stretch for each (label, car, line style) in models, integrated from the
    measured speed at the stretch's start by model_speed(prepared, stretch, car). Stretches whose id is in
    highlight get a star; implied(prepared, stretch) gives the number written at each stretch (default: its
    implied_kw).
    """
    lap, runs = prepared.lap, prepared.runs
    columns = 4
    rows = max(1, int(np.ceil(len(runs) / columns)))
    fig, axes = plt.subplots(rows, columns, figsize=(4.2 * columns, 3.0 * rows), squeeze=False)
    for ax, run in zip(axes.ravel(), runs):
        idx = np.arange(run.i0, run.i1 + 1)
        t0 = lap.t[run.i0]
        ax.plot(lap.t[idx] - t0, lap.v[idx] * 3.6, ".", ms=4, color="black")
        ax.plot(lap.t[idx] - t0, prepared.v_s[idx] * 3.6, "-", lw=0.8, color="0.6")
        for st in run.stretches:
            colour = STRETCH_COLOURS[st.kind]
            ts = lap.t[st.idx] - t0
            ax.axvspan(ts[0], ts[-1], color=colour, alpha=0.10 if st.excluded else 0.25, lw=0)
            for label, car, style in models:
                ax.plot(ts, model_speed(prepared, st, car) * 3.6, color=colour, **style)
            number = implied(prepared, st) if implied else st.implied_kw
            if np.isfinite(number):
                star = "*" if id(st) in highlight else ""
                ax.annotate(f"{number:+.0f}{star}", (ts[0], prepared.v_s[st.idx[0]] * 3.6), fontsize=7, color=colour,
                            xytext=(2, -9), textcoords="offset points")
        zone = lap.w[idx].mean()
        gap = f", gap ahead ≥ {run.min_gap_s:.1f} s" if np.isfinite(run.min_gap_s) else ""
        s0, s1 = lap.s[run.i0] % lap.length, lap.s[run.i1] % lap.length
        ax.set_title(f"run {run.number}: {s0:.0f}–{s1:.0f} m, Straight Mode {zone:.0%}{gap}", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
    for ax in axes.ravel()[len(runs):]:
        ax.axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel("km/h", fontsize=8)
    for ax in axes[-1, :]:
        ax.set_xlabel("s from full throttle", fontsize=8)
    handles = [plt.Line2D([], [], color="0.3", **style) for _, _, style in models]
    fig.legend(handles, [label for label, _, _ in models], loc="lower right", fontsize=8, ncol=len(models))
    fig.suptitle(title or lap.label, fontsize=9, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    fig.savefig(path, dpi=100)
    plt.close(fig)


def implied_power_scatter(points, path, title: str = "") -> None:
    """Implied MGU-K power at the wheels against speed, per sample: (kind, v km/h, kW, highlighted)."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for kind, colour in STRETCH_COLOURS.items():
        for big in (False, True):
            sel = [(v, p) for k, v, p, h in points if k == kind and h == big]
            if sel:
                v, p = np.array(sel).T
                ax.scatter(v, p, s=22 if big else 6, color=colour, alpha=0.9 if big else 0.35,
                           edgecolor="black" if big else "none", lw=0.4,
                           label=f"{kind}{' (best exit of each lap)' if big else ''}")
    ax.axhline(0.95 * 350, color="tab:green", lw=0.8, ls=":")
    ax.axhline(-350 / 0.95, color="tab:red", lw=0.8, ls=":")
    ax.axhline(-250 / 0.95, color="tab:red", lw=0.8, ls=":")
    ax.set_xlabel("speed (km/h)")
    ax.set_ylabel("implied MGU-K power at the wheels (kW)")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
