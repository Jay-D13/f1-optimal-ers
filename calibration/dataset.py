"""
Reference laps for calibration: the pole lap of each 2026 round that has a current raceline, on the model's path.

Each car-data sample of the lap (speed, throttle, brake) gets a lap distance on the placed raceline by projecting
its interpolated position. Frozen car-data blocks are dropped; a pole lap with many of them (Suzuka: 99 of 333
samples) is replaced by the fastest lap that has almost none.

FastF1's lap start times are not consistent: at Spielberg some laps start at the timing line and others about
80 m before it, while the lap times are right. So the samples keep their own lap distances, taken modulo the lap.
"""
import contextlib
import io
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from scipy.spatial import cKDTree

from config.events import EVENTS_2026
from models import F1TrackModel, find_tumftm_raceline
from models.telemetry import load_session

CACHE = Path("data/cache/calibration")


@dataclass
class ReferenceLap:
    round: int
    name: str
    driver: str
    lap_time: float          # (s) Official lap time of this lap
    pole_time: float         # (s) The session's pole, which this lap may not be (see frozen blocks)
    s: np.ndarray            # Lap distance of each sample on the model's path (m), in [0, lap length), sorted
    speed: np.ndarray        # (m/s)
    throttle: np.ndarray     # (%)
    brake: np.ndarray        # (0 or 1)
    air_density: Optional[float]
    track: F1TrackModel      # The model's path for this round (placed raceline, zones, height)
    rules: str = "qualifying"  # The session's 2026 energy rules (config.get_ers_config): qualifying or practice

    def speed_at(self, s) -> np.ndarray:
        """Measured speed (m/s) at lap distances s, interpolated around the lap."""
        return np.interp(s, self.s, self.speed, period=self.track.total_length)

    @property
    def top_speed(self) -> float:
        return float(self.speed.max())


def good_rounds() -> Dict[int, str]:
    """2026 rounds with a current raceline: round → raceline name."""
    return {r: e.raceline for r, e in EVENTS_2026.items() if e.raceline}


def lap_distances(xy: np.ndarray, speed: np.ndarray, seconds: np.ndarray, track: F1TrackModel) -> np.ndarray:
    """
    Lap distance on the track's path of each sample of one lap, from about 0 at the start to about the lap length.

    Positions are projected in the plan view. Where that jumps far from the distance expected by integrating
    the speed (a crossover, a stray sample), the integrated distance is used instead.
    """
    td, length = track.track_data, track.total_length
    _, nearest = cKDTree(np.column_stack([td.x, td.y])).query(xy)
    s = td.s[nearest].astype(float)
    along = np.concatenate([[0.0], np.cumsum(0.5 * (speed[1:] + speed[:-1]) * np.diff(seconds))])
    wrap = lambda d: (d + 0.5 * length) % length - 0.5 * length
    offset = np.median(wrap(s - along))
    stray = np.abs(wrap(s - along - offset)) > 100.0
    s[stray] = (along[stray] + offset) % length
    # Continuous over the lap: samples before the line come out near the end, samples after the lap near 0
    return along + offset + wrap(s - along - offset)


def lap_samples(car, track: F1TrackModel):
    """
    Lap distance on the track's path (in [0, lap length), sorted, unique), speed (m/s), throttle (%) and brake
    (0 or 1) of one lap's car-data samples, which must carry positions and no frozen blocks.
    """
    seconds = car["SessionTime"].dt.total_seconds().to_numpy()
    speed = car["Speed"].to_numpy(dtype=float) / 3.6
    s = lap_distances(car[["X", "Y"]].to_numpy(dtype=float) / 10.0, speed, seconds - seconds[0], track) % track.total_length
    order = np.argsort(s, kind="stable")
    order = order[np.concatenate([[True], np.diff(s[order]) > 0.0])]
    return (s[order], speed[order], car["Throttle"].to_numpy(dtype=float)[order],
            car["Brake"].astype(float).to_numpy()[order])


MAX_FROZEN = 0.05   # Largest share of frozen car-data samples in a reference lap


def reference_lap(round_number: int, ds: float = 5.0, refresh: bool = False) -> ReferenceLap:
    """
    The pole lap of a round, or the fastest lap with at most MAX_FROZEN frozen samples, with the model's path.
    Cached in data/cache/calibration.
    """
    path = CACHE / f"R{round_number:02d}_ds{ds:g}.pkl"
    if path.exists() and not refresh:
        with open(path, "rb") as f:
            return pickle.load(f)

    raceline = EVENTS_2026[round_number].raceline
    if raceline is None:
        raise ValueError(f"Round {round_number} has no current raceline (config/events.py)")
    logging.getLogger("fastf1").setLevel(logging.ERROR)
    track = F1TrackModel(2026, round_number, ds=ds)
    with contextlib.redirect_stdout(io.StringIO()):
        track.load_from_fastf1(raceline=str(find_tumftm_raceline(raceline)))
    track.telemetry_data = None          # Keep the pickle small

    session, _, _ = load_session(2026, round_number)
    laps = session.laps[session.laps["LapTime"].notna()].sort_values("LapTime")
    pole_time = float(laps.iloc[0]["LapTime"].total_seconds())
    for _, lap in laps.head(10).iterrows():
        telemetry = lap.get_telemetry()
        car = telemetry[telemetry["Source"] == "car"]
        frozen = ((car["Throttle"] >= 104) & car["Brake"].astype(bool)).to_numpy()
        if frozen.mean() <= MAX_FROZEN:
            break
    else:
        raise ValueError(f"No lap in the top 10 of round {round_number} has fewer than {MAX_FROZEN:.0%} frozen samples")
    s, speed, throttle, brake = lap_samples(car[~frozen], track)

    reference = ReferenceLap(
        round=round_number, name=EVENTS_2026[round_number].name, driver=str(lap["Driver"]),
        lap_time=float(lap["LapTime"].total_seconds()), pole_time=pole_time,
        s=s, speed=speed, throttle=throttle, brake=brake, air_density=track.air_density, track=track,
    )
    CACHE.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(reference, f)
    return reference
