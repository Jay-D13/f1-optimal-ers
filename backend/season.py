"""
The 2026 season for the UI: each event's energy rules (config/events.py), its real pole lap, and the latest
qualifying run of the model on it.

Pole laps are from Jolpica, cross-checked with F1 timing.
"""
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from config.events import EVENTS_2026, Event2026

RESULTS_DIR = Path(__file__).parent.parent / "results"


@dataclass(frozen=True)
class Pole:
    driver: str
    team: str
    time: float   # (s)


POLES_2026 = {
    1: Pole("Russell", "Mercedes", 78.518),
    2: Pole("Antonelli", "Mercedes", 92.064),
    3: Pole("Antonelli", "Mercedes", 88.778),
    4: Pole("Antonelli", "Mercedes", 87.798),
    5: Pole("Russell", "Mercedes", 72.578),
    6: Pole("Antonelli", "Mercedes", 72.051),
    7: Pole("Russell", "Mercedes", 74.679),
    8: Pole("Russell", "Mercedes", 66.113),
    9: Pole("Antonelli", "Mercedes", 88.111),
    10: Pole("Antonelli", "Mercedes", 104.361),
    11: Pole("Norris", "McLaren", 77.207),
    12: Pole("Norris", "McLaren", 71.163),
    13: Pole("Gasly", "Alpine", 81.786),
    14: Pole("Norris", "McLaren", 91.824),
    15: Pole("Russell", "Mercedes", 102.526),
}


def track_key(event: Event2026) -> str:
    """The --track name the model runs this event under: its raceline if it has one, else its first track name."""
    return event.raceline or event.tracks[0]


def run_summaries(track: str):
    """Every saved run of a track, newest first, as (run_id, results_summary dict)."""
    track_dir = RESULTS_DIR / track.lower()
    if not track_dir.is_dir():
        return []
    runs = []
    for run_dir in sorted(track_dir.iterdir(), reverse=True):
        summary = run_dir / "data" / "results_summary.json"
        if summary.is_file():
            try:
                runs.append((run_dir.name, json.loads(summary.read_text())))
            except (OSError, json.JSONDecodeError):
                continue
    return runs


def run_info(run_id: str, summary: dict, round_number: Optional[int]) -> dict:
    meta = summary.get("metadata", {})
    perf = summary.get("performance", {})
    info = {
        "run_id": run_id,
        "track": meta.get("track"),
        "timestamp": meta.get("timestamp"),
        "regulations": meta.get("regulations"),
        "session": meta.get("session"),
        "n_laps": meta.get("n_laps"),
        "lap_time": perf.get("lap_time"),
        "solver_status": perf.get("solver_status"),
        "solve_time": perf.get("solve_time"),
        "recovered_mj": summary.get("energy", {}).get("total_recovered_MJ"),
        "delta_to_pole": None,
    }
    pole = POLES_2026.get(round_number) if round_number else None
    if pole and info["lap_time"] is not None and info["n_laps"] == 1 and info["session"] == "qualifying":
        info["delta_to_pole"] = info["lap_time"] - pole.time
    return info


def round_runs(round_number: int) -> list:
    """This round's runs made with the event's energy rules (2026 runs of the round), newest first."""
    event = EVENTS_2026[round_number]
    out = []
    for run_id, summary in run_summaries(track_key(event)):
        meta = summary.get("metadata", {})
        if meta.get("year") != 2026:
            continue
        out.append(run_info(run_id, summary, round_number))
    return out


def season() -> list:
    rounds = []
    for number, event in sorted(EVENTS_2026.items()):
        pole = POLES_2026.get(number)
        latest = next((r for r in round_runs(number)
                       if r["regulations"] == "2026" and r["session"] == "qualifying" and r["n_laps"] == 1), None)
        rounds.append({
            "round": number,
            "name": event.name,
            "circuit": event.tracks[0].title(),
            "track": track_key(event),
            "has_raceline": event.raceline is not None,
            "quali_cap_mj": event.quali_recharge_mj,
            "race_cap_mj": event.race_recharge_mj,
            "superclip_kw": event.superclip_kw,
            "ramp_kw_s": event.ramp_rate_kw_s,
            "power_limited_m": event.power_limited_distance_m,
            "verified": event.verified,
            "straight_mode_zones": [{"corner": c, "offset_m": m} for c, m in event.straight_mode_zones],
            "pole": asdict(pole) if pole else None,
            "latest_run": latest,
        })
    return rounds
