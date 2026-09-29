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
