"""
FastF1 access for track geometry (ROADMAP Phase 3): exact event lookup, clean laps, and a cached fitted geometry.
"""
import logging
import unicodedata
from pathlib import Path
from typing import List, Optional, Tuple

import fastf1 as ff1
import numpy as np
from scipy.spatial import cKDTree

from .geometry import TrackGeometry, fit_track

CACHE_DIR = Path("data/cache")

# Track names (TUM raceline files, presets, CLI) whose FastF1 location has another name
_LOCATION_ALIASES = {
    "catalunya": "barcelona",
    "sepang": "kualalumpur",
    "spa": "spafrancorchamps",
    "hungaroring": "budapest",
    "albertpark": "melbourne",
    "monaco": "montecarlo",
    "miami": "miamigardens",
    "madring": "madrid",
    "yasmarina": "yasisland",
}


def _normalise(name: str) -> str:
    """Lower case, no accents, letters and digits only: 'Montréal' → 'montreal'."""
    ascii_name = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    return "".join(ch for ch in ascii_name.lower() if ch.isalnum())


def find_round(year: int, name) -> Tuple[int, str]:
    """
    Round number and location of an event, by round number, location, country or event name (exact, after
    normalising). FastF1's own lookup is fuzzy (TRK-6): in 2026 'Bahrain' finds Kuala Lumpur and 'Spanish' finds
    Madrid. Raises ValueError when nothing or more than one event matches.
    """
    schedule = ff1.get_event_schedule(year, include_testing=False)
    if isinstance(name, int) or str(name).isdigit():
        row = schedule[schedule["RoundNumber"] == int(name)]
        if len(row) != 1:
            raise ValueError(f"No round {name} in {year}")
        return int(name), str(row.iloc[0]["Location"])

    key = _normalise(name)
    for field in ("Location", "EventName", "Country"):
        values = schedule[field].map(_normalise)
        keys = {key, _LOCATION_ALIASES.get(key, key)} if field == "Location" else {key}
        # Aliases can be one year's name for a location that is spelled differently in another (Yas Marina)
        if field == "Location":
            keys |= {k for k, v in _LOCATION_ALIASES.items() if v == key}
        matches = schedule[values.isin(keys)]
        if len(matches) == 1:
            return int(matches.iloc[0]["RoundNumber"]), str(matches.iloc[0]["Location"])
        if len(matches) > 1:
            options = ", ".join(f"{r.RoundNumber} {r.Location}" for r in matches.itertuples())
            raise ValueError(f"'{name}' matches several {year} events ({options}); use the round number")
    options = ", ".join(f"{r.RoundNumber} {r.Location}" for r in schedule.itertuples())
    raise ValueError(f"No {year} event matches '{name}'. Events: {options}")


def load_session(year: int, event, session: str = "Q"):
    """A loaded FastF1 session with telemetry, found with find_round."""
    ff1.Cache.enable_cache(str(CACHE_DIR))
    round_number, location = find_round(year, event)
    loaded = ff1.get_session(year, round_number, session)
    loaded.load(laps=True, telemetry=True, weather=True, messages=False)
    return loaded, round_number, location


def clean_laps(session):
    """
    Laps whose line is representative: within 107 % of the session's fastest, not in or out of the pits,
    not deleted, and not under yellow or worse track status (TRK-8).
    """
    laps = session.laps.pick_quicklaps(1.07).pick_wo_box()
    if "Deleted" in laps:
        laps = laps[laps["Deleted"] != True]   # noqa: E712 (NaN means not deleted)
    return laps.pick_track_status("1", how="equals")


def lap_positions(lap, frozen_filter: bool = True):
    """
    A FastF1 lap's position samples as an (n × 3) array in metres and their speeds (m/s), or None if unusable.

    The live-timing positions are snapped to the timing provider's own map of the circuit: every lap, whoever
    drives it, lies within about 2 cm of the same line, and the height is piecewise constant. So the pooled
    laps give that map line, not the line the cars drive (TRK-10). FastF1 positions are in 1/10 m. With frozen_filter, samples near frozen car-data blocks
    (Throttle ≥ 104 with Brake on, TRK-7) are dropped.
    """
    pos = lap.get_pos_data()
    if pos is None or len(pos) < 50:
        return None
    xyz = pos[["X", "Y", "Z"]].to_numpy(dtype=float) / 10.0
    keep = np.all(np.isfinite(xyz), axis=1)
    car = lap.get_car_data()
    if car is None or len(car) < 50:
        return None
    # Speed at the position samples, interpolated in session time
    pos_seconds = pos["SessionTime"].dt.total_seconds().to_numpy()
    car_seconds = car["SessionTime"].dt.total_seconds().to_numpy()
    speed = np.interp(pos_seconds, car_seconds, car["Speed"].to_numpy(dtype=float)) / 3.6
    if frozen_filter:
        frozen = ((car["Throttle"] >= 104) & car["Brake"].astype(bool)).to_numpy()
        if frozen.any():
            # Drop position samples within 0.3 s of a frozen car-data sample
            frozen_seconds = car_seconds[frozen]
            nearest = np.clip(np.searchsorted(frozen_seconds, pos_seconds), 1, len(frozen_seconds)) - 1
            gap = np.minimum(
                np.abs(pos_seconds - frozen_seconds[nearest]),
                np.abs(pos_seconds - frozen_seconds[np.minimum(nearest + 1, len(frozen_seconds) - 1)]),
            )
            keep &= gap > 0.3
    return (xyz[keep], speed[keep]) if keep.sum() >= 50 else None


def laps_geometry(laps, source: str = "", **fit_options) -> TrackGeometry:
    """
    Fit a geometry to the position samples of the given FastF1 laps (see fit_track for the options), with the
    lap starting at the timing line.
    """
    positions: List[np.ndarray] = []
    speeds: List[np.ndarray] = []
    starts = []   # Per lap: first position sample, seconds after the line, speed there
    for _, lap in laps.iterrows():
        data = lap_positions(lap)
        if data is None:
            continue
        positions.append(data[0])
        speeds.append(data[1])
        pos = lap.get_pos_data()
        after_line = (pos["SessionTime"].iloc[0] - lap["LapStartTime"]).total_seconds()
        if 0.0 <= after_line < 1.0:
            starts.append((pos[["X", "Y", "Z"]].iloc[0].to_numpy(dtype=float) / 10.0, after_line, data[1][0]))
    if len(positions) < 5:
        raise ValueError(f"Only {len(positions)} usable laps for the geometry")
    geometry = fit_track(positions, speeds=speeds, source=f"{source}, {len(positions)} laps", **fit_options)
    if starts:
        geometry = geometry.shifted(_line_distance(geometry, starts))
    return geometry


def _line_distance(geometry: TrackGeometry, starts) -> float:
    """Lap distance of the timing line: each lap's first sample, moved back by the distance covered since the line."""
    points = np.array([point for point, _, _ in starts])
    back = np.array([after_line * speed for _, after_line, speed in starts])
    line = geometry.project(points) - back
    # Median on the circle, around the first estimate
    offsets = (line - line[0] + 0.5 * geometry.length) % geometry.length - 0.5 * geometry.length
    return float((line[0] + np.median(offsets)) % geometry.length)


def session_geometry(year: int, event, session: str = "Q", refresh: bool = False,
                     loaded_session=None) -> TrackGeometry:
    """
    The fitted geometry of a session's clean laps, cached in data/cache/geometry.

    Pass loaded_session to reuse a session that is already loaded.
    """
    if loaded_session is None:
        loaded_session, round_number, location = load_session(year, event, session)
    else:
        round_number, location = int(loaded_session.event["RoundNumber"]), str(loaded_session.event["Location"])
    source = f"FastF1 {year} R{round_number} {location} {session}"
    path = CACHE_DIR / "geometry" / f"{year}_R{round_number:02d}_{session}.csv"
    if path.exists() and not refresh:
        cached = TrackGeometry.from_csv(path)
        if cached.source.startswith(source):
            return cached
    geometry = laps_geometry(clean_laps(loaded_session), source=source)
    path.parent.mkdir(parents=True, exist_ok=True)
    geometry.to_csv(path)
    logging.getLogger(__name__).info("Saved geometry to %s", path)
    return geometry


def corner_distances(session, geometry: TrackGeometry) -> Optional[dict]:
    """
    Lap distance (m) of each corner marker in FastF1's circuit info, keyed like '1' or '1A', or None when the
    session has no circuit info. The markers have no height, so they are projected in the plan view.

    FastF1's own marker distances integrate the speed of one lap, which the 2026 frozen car-data blocks corrupt
    (at Monza they are 115 m off after T4), so the geometry's distances are used instead.
    """
    try:
        corners = session.get_circuit_info().corners
    except Exception:
        return None
    if corners is None or len(corners) == 0:
        return None
    markers = corners[["X", "Y"]].to_numpy(dtype=float) / 10.0
    _, nearest = cKDTree(np.column_stack([geometry.x, geometry.y])).query(markers)
    labels = [f"{number}{letter or ''}" for number, letter in zip(corners["Number"], corners["Letter"])]
    return dict(zip(labels, geometry.s[nearest]))


def zone_intervals(activation_points, corners: dict, length: float) -> List[Tuple[float, float]]:
    """
    Straight Mode zones as (start, end) lap distances (m); end < start when a zone crosses the line.

    Each zone starts at its activation point, (corner label, metres after the marker), and ends at the next
    corner marker, where the wings close for braking anyway.
    """
    marker_s = np.sort(np.array(list(corners.values())))
    zones = []
    for corner, offset in activation_points:
        if corner not in corners:
            raise ValueError(f"No corner {corner} in the circuit info ({', '.join(corners)})")
        start = (corners[corner] + offset) % length
        ahead = (marker_s - start) % length
        end = marker_s[np.argmin(np.where(ahead > 1.0, ahead, np.inf))]
        zones.append((float(start), float(end)))
    return zones
