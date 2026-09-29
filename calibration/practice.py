"""
Speeds from an event's sessions before qualifying (practice, and sprint qualifying on sprint weekends), and a
bound on the path curvature from them.

The placed TUM racelines are tighter than the driven line at a few fast corners (TRK-11): the pole speeds would
need 6–7 g there. The bound caps the curvature so that the speeds seen in those earlier sessions need at most
G_MAX. It never uses the qualifying session, so a held-out round stays a prediction.
"""
import logging
import pickle
from pathlib import Path

import numpy as np

from models.telemetry import clean_laps, load_session

from .dataset import CACHE, lap_distances

SESSIONS = ("FP1", "FP2", "FP3", "SQ")   # Whichever the event has
BIN = 10.0                               # (m) Width of the speed bins along the lap
PERCENTILE = 90.0                        # Speed kept per bin, over the pooled samples
MIN_SAMPLES = 3                          # Bins with fewer samples are interpolated from their neighbours
G_MAX = 5.0                              # (g) Largest lateral acceleration the bound allows at those speeds


def practice_speed(round_number: int, track, refresh: bool = False) -> np.ndarray:
    """
    High-percentile speed (m/s) at each point of the track's grid, from the clean laps of the sessions before
    qualifying. Cached in data/cache/calibration.
    """
    path = CACHE / f"R{round_number:02d}_practice_speed.pkl"
    if path.exists() and not refresh:
        with open(path, "rb") as f:
            cached = pickle.load(f)
        if len(cached) == len(track.track_data.s):
            return cached

    logging.getLogger("fastf1").setLevel(logging.ERROR)
    length = track.total_length
    distances, speeds = [], []
    for name in SESSIONS:
        try:
            session, _, _ = load_session(2026, round_number, name)
        except Exception:
            continue                      # The event doesn't have this session
        for _, lap in clean_laps(session).iterrows():
            try:
                # Car data with positions interpolated onto it (get_telemetry's driver-ahead step fails on
                # some practice laps and isn't needed)
                car = lap.get_car_data().merge_channels(lap.get_pos_data(), frequency="original")
            except Exception:
                continue
            car = car[car["Source"] == "car"]
            car = car[~((car["Throttle"] >= 104) & car["Brake"].astype(bool))].dropna(subset=["X", "Y", "Speed"])
            if len(car) < 50:
                continue
            seconds = (car["SessionTime"] - car["SessionTime"].iloc[0]).dt.total_seconds().to_numpy()
            speed = car["Speed"].to_numpy(dtype=float) / 3.6
            distances.append(lap_distances(car[["X", "Y"]].to_numpy(dtype=float) / 10.0, speed, seconds, track) % length)
            speeds.append(speed)
    if not distances:
        raise ValueError(f"Round {round_number}: no laps before qualifying")

    s, v = np.concatenate(distances), np.concatenate(speeds)
    n_bins = int(np.ceil(length / BIN))
    bins = np.minimum((s / BIN).astype(int), n_bins - 1)
    centres = (np.arange(n_bins) + 0.5) * BIN
    value = np.full(n_bins, np.nan)
    for b in range(n_bins):
        inside = v[bins == b]
        if len(inside) >= MIN_SAMPLES:
            value[b] = np.percentile(inside, PERCENTILE)
    known = np.isfinite(value)
    result = np.interp(track.track_data.s, centres[known], value[known], period=length)

    CACHE.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(result, f)
    return result


def bound_curvature(track, speed: np.ndarray, g_max: float = G_MAX) -> np.ndarray:
    """
    Cap the track's curvature at g_max·g / speed² in place (radius too). Returns the fraction of the original
    curvature kept at each point (1 where the bound doesn't bite).
    """
    td = track.track_data
    limit = g_max * 9.81 / np.maximum(speed, 1.0) ** 2
    kept = np.minimum(1.0, limit / np.maximum(np.abs(td.curvature), 1e-9))
    td.curvature = td.curvature * kept
    td.radius = np.clip(1.0 / (np.abs(td.curvature) + 1e-6), 10, 10000)
    return kept
