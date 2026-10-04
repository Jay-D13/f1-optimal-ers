"""
Reference laps for calibration: one qualifying lap of each 2026 round that has a current raceline, on the model's
path: the pole lap, or the fastest lap of the top 10 that is usable and clean.

Each car-data sample of the lap (speed, throttle, brake) gets a lap distance on the placed raceline by projecting
its interpolated position. Frozen car-data blocks are dropped; a lap with many of them (Suzuka's pole: 99 of 333
samples) is skipped.

Towed laps are skipped too (BUGS CAL-4): more than TOW_SECONDS at full throttle with a car less than TOW_DISTANCE
ahead. If every usable lap of the top 10 is towed, the fastest is kept and flagged (ReferenceLap.towed). The gaps
come from every car's own positions, projected on the lap's own line. FastF1's DistanceToDriverAhead isn't used:
in qualifying it misses cars, because it picks which laps of another driver to integrate by comparing lap numbers,
which only works in a race (at Monza it never sees Hamilton ahead of Gasly's pole lap), and it counts cars in the
pit lane.

FastF1's lap start times are not consistent: at Spielberg some laps start at the timing line and others about
80 m before it, while the lap times are right. So the samples keep their own lap distances, taken modulo the lap.
"""
import contextlib
import io
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, NamedTuple, Optional

import numpy as np
from scipy.spatial import cKDTree

from config.events import EVENTS_2026
from models import F1TrackModel, find_tumftm_raceline
from models.telemetry import load_session

CACHE = Path("data/cache/calibration")
CACHE_VERSION = 2   # 2: tow exposure and frozen share on the lap, towed laps skipped. Version 1 pickles still load

MAX_FROZEN = 0.05     # Largest share of frozen car-data samples in a reference lap
FULL_THROTTLE = 98.0  # (%) Throttle from which a sample counts as full throttle
TOW_DISTANCE = 80.0   # (m) A car closer than this ahead gives a tow
TOW_SECONDS = 3.0     # (s) More time than this at full throttle within TOW_DISTANCE makes a lap towed
PIT_OFFSET = 5.0      # (m) A car farther than this from the lap's line is in the pit lane or garage
MAX_HOLE = 2.0        # (s) Longest hole in another car's positions that is interpolated across


@dataclass
class ReferenceLap:
    round: int
    name: str
    driver: str
    lap_time: float          # (s) Official lap time of this lap
    pole_time: float         # (s) The session's pole, which this lap may not be (frozen blocks, tows)
    s: np.ndarray            # Lap distance of each sample on the model's path (m), in [0, lap length), sorted
    speed: np.ndarray        # (m/s)
    throttle: np.ndarray     # (%)
    brake: np.ndarray        # (0 or 1)
    air_density: Optional[float]
    track: F1TrackModel      # The model's path for this round (placed raceline, zones, height)
    rules: str = "qualifying"  # The session's 2026 energy rules (config.get_ers_config): qualifying or practice
    # Tow exposure (CAL-4); None where not measured (version 1 pickles, practice laps)
    tow_seconds: Optional[float] = None   # (s) Time at full throttle with a car less than TOW_DISTANCE ahead
    driver_ahead: str = ""                # The car ahead for most of that time; without a tow, the nearest one
    ahead_gap: Optional[float] = None     # (m) Smallest distance to a car ahead at full throttle
    frozen_share: Optional[float] = None  # Share of the lap's car-data samples in frozen blocks (dropped)

    def speed_at(self, s) -> np.ndarray:
        """Measured speed (m/s) at lap distances s, interpolated around the lap."""
        return np.interp(s, self.s, self.speed, period=self.track.total_length)

    @property
    def top_speed(self) -> float:
        return float(self.speed.max())

    @property
    def towed(self) -> bool:
        """True when the lap has a tow: no usable lap of the top 10 was clean."""
        return self.tow_seconds is not None and self.tow_seconds > TOW_SECONDS


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


def frozen_samples(car) -> np.ndarray:
    """Which car-data samples are in frozen blocks (Throttle ≥ 104 with the brake on)."""
    return ((car["Throttle"] >= 104) & car["Brake"].astype(bool)).to_numpy()


class TowExposure(NamedTuple):
    seconds: float         # (s) Time at full throttle with a car less than TOW_DISTANCE ahead
    driver: str            # The car ahead for most of that time; without a tow, the nearest car ahead at full throttle
    gap: Optional[float]   # (m) Smallest distance to a car ahead at full throttle; None if there was none


def tow_exposure(seconds: np.ndarray, full: np.ndarray, gaps: Mapping[str, np.ndarray],
                 distance: float = TOW_DISTANCE) -> TowExposure:
    """
    Tow exposure of one lap from its samples' times (s), which samples are at full throttle, and each other car's
    distance ahead (m) along the lap at those samples (inf or NaN where it isn't on track). Only the nearest car
    ahead counts at each sample. Each sample stands for half the time to each neighbour, at most 1 s.
    """
    seconds = np.asarray(seconds, dtype=float)
    full = np.asarray(full, dtype=bool)
    names = list(gaps)
    if not names or len(seconds) < 2:
        return TowExposure(0.0, "", None)
    dt = np.minimum(np.gradient(seconds), 1.0)
    table = np.column_stack([np.asarray(gaps[n], dtype=float) for n in names])
    table = np.where(np.isfinite(table) & (table >= 0.0), table, np.inf)
    nearest = table.argmin(axis=1)
    gap = table[np.arange(len(seconds)), nearest]

    near = full & (gap < distance)
    full_gap = np.where(full, gap, np.inf)
    if not np.isfinite(full_gap).any():
        return TowExposure(0.0, "", None)
    closest = int(np.argmin(full_gap))
    if near.any():
        driver = names[int(np.bincount(nearest[near], weights=dt[near], minlength=len(names)).argmax())]
    else:
        driver = names[nearest[closest]]
    return TowExposure(float(dt[near].sum()), driver, float(full_gap[closest]))


def distance_ahead(line: np.ndarray, own: np.ndarray, other: np.ndarray, max_offset: float = PIT_OFFSET) -> np.ndarray:
    """
    Distance (m) along a closed line (n × 2 positions over one lap, in order) from each own position to the other
    car's position at the same instant, in [0, line length); inf where the other car is farther than max_offset
    from the line.
    """
    along = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(line, axis=0), axis=1))])
    length = along[-1] + np.linalg.norm(line[0] - line[-1])
    dense = np.arange(0.0, along[-1], 1.0)
    tree = cKDTree(np.column_stack([np.interp(dense, along, line[:, 0]), np.interp(dense, along, line[:, 1])]))
    _, i_own = tree.query(own)
    offset, i_other = tree.query(other)
    return np.where(offset <= max_offset, (dense[i_other] - dense[i_own]) % length, np.inf)


def in_pits(laps, seconds: np.ndarray) -> np.ndarray:
    """Whether a driver is in the pit lane or garage at these session times (s), from its laps' pit times."""
    out = laps["PitOutTime"].dropna().dt.total_seconds().to_numpy()
    into = laps["PitInTime"].dropna().dt.total_seconds().to_numpy()
    times = np.concatenate([out, into])
    if len(times) == 0:
        return np.zeros(len(seconds), dtype=bool)
    leaving = np.concatenate([np.ones(len(out), dtype=bool), np.zeros(len(into), dtype=bool)])
    order = np.argsort(times, kind="stable")
    times, leaving = times[order], leaving[order]
    last = np.searchsorted(times, seconds, side="right") - 1
    # Before the first pit event: in the garage if that event is a pit exit
    return np.where(last < 0, leaving[0], ~leaving[np.maximum(last, 0)])


def gaps_ahead(session, lap, car) -> Dict[str, np.ndarray]:
    """
    Distance (m) along the lap from the lap's car to every other car, at the lap's car-data samples (car, with
    positions): driver abbreviation → array, inf where that car is in the pits, off the line or without positions.
    """
    line = lap.get_pos_data()[["X", "Y"]].to_numpy(dtype=float) / 10.0
    seconds = car["SessionTime"].dt.total_seconds().to_numpy()
    own = car[["X", "Y"]].to_numpy(dtype=float) / 10.0
    gaps = {}
    for number in session.drivers:
        if number == str(lap["DriverNumber"]) or number not in session.pos_data:
            continue
        pos = session.pos_data[number]
        t = pos["SessionTime"].dt.total_seconds().to_numpy()
        inside = (t > seconds[0] - 10.0) & (t < seconds[-1] + 10.0)
        if inside.sum() < 2:
            continue
        t, xy = t[inside], pos[["X", "Y"]].to_numpy(dtype=float)[inside] / 10.0
        other = np.column_stack([np.interp(seconds, t, xy[:, 0]), np.interp(seconds, t, xy[:, 1])])
        after = np.clip(np.searchsorted(t, seconds), 1, len(t) - 1)
        known = (seconds >= t[0]) & (seconds <= t[-1]) & (t[after] - t[after - 1] <= MAX_HOLE)
        gap = distance_ahead(line, own, other)
        gap[~known | in_pits(session.laps.pick_drivers(number), seconds)] = np.inf
        gaps[str(session.get_driver(number)["Abbreviation"])] = gap
    return gaps


def lap_tow(session, lap, car) -> TowExposure:
    """
    Tow exposure of a FastF1 lap from its car-data samples with positions (car, frozen blocks included: they are
    not counted as full throttle).
    """
    car = car.dropna(subset=["X", "Y"])
    full = (car["Throttle"].to_numpy(dtype=float) >= FULL_THROTTLE) & ~frozen_samples(car)
    return tow_exposure(car["SessionTime"].dt.total_seconds().to_numpy(), full, gaps_ahead(session, lap, car))


def reference_lap(round_number: int, ds: float = 5.0, refresh: bool = False) -> ReferenceLap:
    """
    The fastest lap of a round's qualifying top 10 with at most MAX_FROZEN frozen samples and no tow, with the
    model's path; if all of those are towed, the fastest of them, flagged (ReferenceLap.towed). Deleted laps don't
    count. Cached in data/cache/calibration.
    """
    path = CACHE / f"R{round_number:02d}_v{CACHE_VERSION}_ds{ds:g}.pkl"
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
    laps = session.laps[session.laps["LapTime"].notna()]
    if "Deleted" in laps:
        laps = laps[laps["Deleted"] != True]   # noqa: E712 (NaN means not deleted)
    laps = laps.sort_values("LapTime")
    pole_time = float(laps.iloc[0]["LapTime"].total_seconds())
    chosen = None
    for _, lap in laps.head(10).iterrows():
        telemetry = lap.get_telemetry()
        car = telemetry[telemetry["Source"] == "car"]
        frozen = frozen_samples(car)
        if frozen.mean() > MAX_FROZEN:
            continue
        tow = lap_tow(session, lap, car)
        if chosen is None or tow.seconds <= TOW_SECONDS:
            chosen = (lap, car, frozen, tow)   # The fastest usable lap stands in until a clean one turns up
        if tow.seconds <= TOW_SECONDS:
            break
    if chosen is None:
        raise ValueError(f"No lap in the top 10 of round {round_number} has fewer than {MAX_FROZEN:.0%} frozen samples")
    lap, car, frozen, tow = chosen
    s, speed, throttle, brake = lap_samples(car[~frozen], track)

    reference = ReferenceLap(
        round=round_number, name=EVENTS_2026[round_number].name, driver=str(lap["Driver"]),
        lap_time=float(lap["LapTime"].total_seconds()), pole_time=pole_time,
        s=s, speed=speed, throttle=throttle, brake=brake, air_density=track.air_density, track=track,
        tow_seconds=tow.seconds, driver_ahead=tow.driver, ahead_gap=tow.gap, frozen_share=float(frozen.mean()),
    )
    CACHE.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(reference, f)
    return reference
