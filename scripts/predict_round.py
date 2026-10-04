#!/usr/bin/env python
"""
Blind pole prediction for a 2026 round, made from the sessions before qualifying only.

    # 1. Place the raceline and build the practice speed envelope (FastF1 only, no solver, no lock needed)
    .venv/bin/python scripts/predict_round.py --round 16 --sessions FP1,FP2 --prepare-only
    # 2. Solve under the event's qualifying rules with frozen parameters and record the prediction
    .venv/bin/python scripts/solver_lock.py --label WP6 -- \
        .venv/bin/python scripts/predict_round.py --round 16 --sessions FP1,FP2 --params start
    # 3. After qualifying: fetch the pole and append prediction, real time and error to the ledger
    .venv/bin/python scripts/predict_round.py --round 16 --settle

The prediction: the event's TUM raceline placed on the last practice session (height, timing line, Straight Mode
zones), its curvature bounded with the practice speeds (calibration/practice.py, TRK-11), the car from a frozen
parameter set (calibration/model.py car_for) with that session's air density, solved under the event's qualifying
rules. Written to data/predictions/R<round>_<params hash>[_<label>]_<UTC timestamp>.json with the lap time, speed
trace, energy map, clip seconds, parameters and git commit. Nothing from the qualifying session is read before
--settle.

--params takes a built-in set (start: calibration/model.py start values; fit2: Fit 2 of 2026-09-29 as listed in
the roadmap) or a JSON file of {name: value}. A set may carry "g_max", the curvature bound it was fitted with
(0 = none); --g-max overrides it.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import datetime as dt
import hashlib
import io
import json
import logging
import os
import pickle
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PREDICTIONS = Path("data/predictions")
LEDGER = PREDICTIONS / "ledger.csv"
DEFAULT_SESSIONS = ("FP1", "FP2", "FP3", "SQ")   # In order; the last one used places the raceline

# Fit 2 (2026-09-29, 6 parameters trained on rounds 2, 5, 9, 10, 13), at the precision the roadmap gives. It was
# fitted on the unbounded TUM path (before the practice curvature bound, bc6aa81), hence g_max 0.
FIT2 = {
    "c_w_a": 1.18,
    "c_z_a": 5.3,
    "front_downforce_share": 0.48,
    "straight_mode_drag_factor": 0.70,
    "mu_scale": 0.93,
    "pow_max_ice": 330e3,
    "g_max": 0.0,
}

# Corner markers (lap distance in m on the placed TUM line, and the line's length) for rounds whose circuit has no
# FastF1 circuit info, so that the FIA Straight Mode activation points can be placed. Sepang has none (the
# MultiViewer API has no circuit 12): the markers are the curvature peaks of the TUM line placed on 2026 FP2, at
# the exit apex of multi-apex corners, and agree with the PUI's corner windows (REFERENCE §4: T1–T2 600–750 m,
# exit T5 1950–2100, exit T6 2150–2350, T9–T11 3100–3400, T12–T13 3750–4000, exit T15 5050–5300, on the 5543 m
# centreline). Scaled to the placed line's length.
FALLBACK_CORNERS = {
    16: (5441.2, {"1": 660, "2": 745, "3": 1045, "4": 1535, "5": 1890, "6": 2105, "7": 2490, "8": 2600,
                  "9": 3080, "10": 3180, "11": 3430, "12": 3775, "13": 3910, "14": 4125, "15": 5050}),
}

LEDGER_FIELDS = (
    "settled_utc", "round", "event", "file", "label", "params", "params_hash", "created_utc", "sessions",
    "g_max", "predicted_s", "pole_s", "pole_driver", "pole_source", "error_s", "error_pct",
)


# ---------------------------------------------------------------------------------------------------------------
# Parameters and records (no FastF1, no solver)

def parameter_set(spec: str) -> tuple[str, Dict[str, float]]:
    """(name, values) for a built-in set name or a JSON file path."""
    if spec == "start":
        from calibration.model import start_values
        return "start", start_values()
    if spec == "fit2":
        return "fit2", dict(FIT2)
    path = Path(spec)
    values = json.loads(path.read_text())
    if not isinstance(values, dict) or not all(isinstance(v, (int, float)) for v in values.values()):
        raise ValueError(f"{spec}: expected a JSON object of parameter name → number")
    return path.stem, {k: float(v) for k, v in values.items()}


def params_hash(params: Mapping[str, float]) -> str:
    """Short, order-independent hash of a parameter set."""
    canonical = json.dumps({k: float(params[k]) for k in sorted(params)}, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:8]


def prediction_name(round_number: int, phash: str, created: dt.datetime, label: str = "") -> str:
    stamp = created.astimezone(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    middle = f"_{label}" if label else ""
    return f"R{round_number:02d}_{phash}{middle}_{stamp}.json"


def git_commit() -> Dict[str, object]:
    def run(*cmd):
        return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = [line for line in run("git", "status", "--porcelain", "--untracked-files=no").splitlines() if line]
    return {"commit": run("git", "rev-parse", "HEAD"), "branch": run("git", "rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(dirty)}


def runs_seconds(dt_s: np.ndarray, flag: np.ndarray) -> float:
    return float(np.sum(dt_s[flag]))


def lap_structure(traj, ers) -> Dict[str, float]:
    """
    Seconds per lap at full throttle, braking, super-clipping (full throttle, no brake, speed falling) and
    deploying, and the DC-side harvest split by phase: the CAL-1 diagnostic's definitions.
    """
    t = np.asarray(traj.t_opt, dtype=float)
    v = np.asarray(traj.v_opt, dtype=float)
    s = np.asarray(traj.s, dtype=float)
    n = min(len(t) - 1, len(traj.throttle_opt), len(traj.brake_opt))
    dt_s = np.diff(t)[:n]
    thr = np.asarray(traj.throttle_opt, dtype=float)[:n]
    brk = np.asarray(traj.brake_opt, dtype=float)[:n]
    dv_ds = (np.diff(v * 3.6) / np.diff(s))[:n]
    full, braking = thr >= 0.98, brk > 0.02
    out = {
        "full_throttle_s": runs_seconds(dt_s, full),
        "braking_s": runs_seconds(dt_s, braking),
        "clip_s": runs_seconds(dt_s, full & ~braking & (dv_ds < -0.05)),
    }
    if traj.P_harvest_opt is not None and traj.P_deploy_opt is not None:
        P_h = np.asarray(traj.P_harvest_opt, dtype=float)[:n]
        P_d = np.asarray(traj.P_deploy_opt, dtype=float)[:n]
        e_h = P_h * ers.mgu_k_efficiency * dt_s / 1e6
        out.update({
            "harvest_full_throttle_s": runs_seconds(dt_s, full & ~braking & (P_h > 50e3)),
            "deploy_over_300kW_s": runs_seconds(dt_s, P_d > 300e3),
            "harvest_braking_MJ": float(e_h[braking].sum()),
            "harvest_full_throttle_MJ": float(e_h[full & ~braking].sum()),
            "harvest_lift_MJ": float(e_h[~full & ~braking].sum()),
        })
    return out


def _rounded(values, digits: int) -> List[float]:
    return [round(float(x), digits) for x in np.asarray(values, dtype=float)]


def prediction_record(*, round_number: int, event_name: str, params_name: str, params: Mapping[str, float],
                      label: str, created: dt.datetime, sessions: Sequence[str], placement: Mapping,
                      g_max: float, kept: Optional[np.ndarray], ers, traj, git: Mapping) -> Dict:
    """The JSON record of one prediction."""
    n = len(traj.v_opt)
    structure = lap_structure(traj, ers)
    record = {
        "kind": "blind pole prediction",
        "round": round_number,
        "event": event_name,
        "label": label,
        "created_utc": created.astimezone(dt.timezone.utc).isoformat(timespec="seconds"),
        "created_local": created.astimezone().isoformat(timespec="seconds"),
        "git": dict(git),
        "params_name": params_name,
        "params_hash": params_hash(params),
        "params": {k: float(v) for k, v in params.items()},
        "data": {
            "sessions": list(sessions),
            "placement_session": placement["session"],
            "raceline": placement["raceline"],
            "raceline_gap_m": placement["gap_m"],
            "air_density": placement["air_density"],
            "air_density_overridden": bool(placement.get("air_density_overridden", False)),
            "lap_length_m": placement["length_m"],
            "straight_mode_zones_m": placement["zones"],
            "straight_mode_zones_source": placement.get("zones_source"),
        },
        "path": {
            "g_max": g_max,
            "points_bounded": int(np.sum(kept < 0.999)) if kept is not None else 0,
            "min_curvature_kept": float(np.min(kept)) if kept is not None else 1.0,
        },
        "rules": {
            "session": "qualifying",
            "recharge_cap_MJ": ers.recovery_limit_per_lap / 1e6,
            "superclip_kW": (ers.superclip_power or 0.0) / 1e3,
            "ramp_rate": ers.ramp_rate,
        },
        "result": {
            "lap_time_s": float(traj.lap_time),
            "solver_status": traj.solver_status,
            "solve_time_s": float(traj.solve_time),
            "top_speed_kmh": float(np.max(traj.v_opt) * 3.6),
            "min_speed_kmh": float(np.min(traj.v_opt) * 3.6),
            "deployed_MJ": float(traj.energy_deployed / 1e6),
            "recovered_MJ": float(traj.energy_recovered / 1e6),
            **structure,
            "run_up": {k: float(v) for k, v in (traj.run_up or {}).items() if np.isscalar(v)},
        },
        "trace": {
            "s_m": _rounded(traj.s[:n], 2),
            "t_s": _rounded(traj.t_opt[:n], 4),
            "speed_kmh": _rounded(np.asarray(traj.v_opt) * 3.6, 2),
            "soc": _rounded(traj.soc_opt[:n], 5),
        },
        "energy_map": {
            "s_m": _rounded(traj.s[: len(traj.P_ers_opt)], 2),
            "P_ers_kW": _rounded(np.asarray(traj.P_ers_opt) / 1e3, 2),
            "P_deploy_kW": _rounded(np.asarray(traj.P_deploy_opt) / 1e3, 2) if traj.P_deploy_opt is not None else None,
            "P_harvest_kW": _rounded(np.asarray(traj.P_harvest_opt) / 1e3, 2) if traj.P_harvest_opt is not None else None,
            "throttle": _rounded(traj.throttle_opt, 4),
            "brake": _rounded(traj.brake_opt, 4),
        },
    }
    return record


def summary_line(record: Mapping, path: Path) -> str:
    r = record["result"]
    return (f"R{record['round']:02d} {record['event']}: {r['lap_time_s']:.3f} s "
            f"[{record['params_name']} {record['params_hash']}, {'+'.join(record['data']['sessions'])}, "
            f"g_max {record['path']['g_max']:g}, top {r['top_speed_kmh']:.0f} km/h, clip {r['clip_s']:.1f} s, "
            f"{r['solver_status']}] -> {path}")


def ledger_rows(round_number: int, directory: Path, pole_s: float, pole_driver: str, pole_source: str,
                settled: dt.datetime, already: set) -> List[Dict]:
    """Ledger rows for this round's predictions not yet in the ledger (by file name)."""
    rows = []
    for path in sorted(directory.glob(f"R{round_number:02d}_*.json")):
        if path.name in already:
            continue
        record = json.loads(path.read_text())
        if record.get("kind") != "blind pole prediction":
            continue
        predicted = float(record["result"]["lap_time_s"])
        rows.append({
            "settled_utc": settled.astimezone(dt.timezone.utc).isoformat(timespec="seconds"),
            "round": round_number, "event": record["event"], "file": path.name, "label": record.get("label", ""),
            "params": record["params_name"], "params_hash": record["params_hash"],
            "created_utc": record["created_utc"], "sessions": "+".join(record["data"]["sessions"]),
            "g_max": record["path"]["g_max"], "predicted_s": f"{predicted:.3f}", "pole_s": f"{pole_s:.3f}",
            "pole_driver": pole_driver, "pole_source": pole_source, "error_s": f"{predicted - pole_s:+.3f}",
            "error_pct": f"{100.0 * (predicted - pole_s) / pole_s:+.2f}",
        })
    return rows


def append_ledger(ledger: Path, rows: Sequence[Mapping]) -> None:
    new = not ledger.exists()
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with open(ledger, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LEDGER_FIELDS)
        if new:
            writer.writeheader()
        writer.writerows(rows)


def ledger_files(ledger: Path) -> set:
    if not ledger.exists():
        return set()
    with open(ledger, newline="") as f:
        return {row["file"] for row in csv.DictReader(f)}


# ---------------------------------------------------------------------------------------------------------------
# FastF1 and the solver

@contextlib.contextmanager
def _practice_envelope_config(sessions: Sequence[str], cache: Path):
    """calibration.practice reads its sessions and cache directory from module globals: point them at ours."""
    from calibration import practice
    old = practice.SESSIONS, practice.CACHE
    practice.SESSIONS, practice.CACHE = tuple(sessions), cache
    try:
        yield practice
    finally:
        practice.SESSIONS, practice.CACHE = old


def available_sessions(round_number: int, wanted: Sequence[str]) -> List[str]:
    """The wanted sessions that load and have timed laps (never the qualifying session)."""
    from models.telemetry import load_session
    found = []
    for name in wanted:
        if name.upper() in ("Q", "QUALIFYING", "R", "RACE", "S", "SPRINT"):
            raise ValueError(f"{name}: only sessions before qualifying may be used")
        try:
            session, _, _ = load_session(2026, round_number, name)
            if session.laps is not None and session.laps["LapTime"].notna().any():
                found.append(name)
        except Exception as e:                  # Not run yet, or not on this weekend
            print(f"   {name}: not available ({type(e).__name__})")
    return found


def prepare(round_number: int, sessions: Sequence[str], ds: float, refresh: bool = False) -> Dict:
    """
    The raceline placed on the last of the sessions, its practice speed envelope and the session's air density,
    cached in data/predictions/cache. No solver.
    """
    from config.events import EVENTS_2026
    from models import F1TrackModel, find_tumftm_raceline

    event = EVENTS_2026[round_number]
    if not event.raceline:
        raise ValueError(f"Round {round_number} has no current raceline (config/events.py)")
    used = available_sessions(round_number, sessions)
    if not used:
        raise ValueError(f"Round {round_number}: none of {', '.join(sessions)} is available")
    key = "-".join(used)
    cache = PREDICTIONS / "cache" / f"R{round_number:02d}_{key}_ds{ds:g}"
    prepared_path = cache / "prepared.pkl"
    if prepared_path.exists() and not refresh:
        with open(prepared_path, "rb") as f:
            return pickle.load(f)

    logging.getLogger("fastf1").setLevel(logging.ERROR)
    placement_session = used[-1]
    raceline = find_tumftm_raceline(event.raceline)
    track = F1TrackModel(2026, round_number, session=placement_session, ds=ds)
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        track.load_from_fastf1(raceline=str(raceline))
    print(log.getvalue(), end="")
    gap = re.search(r"largest gap ([0-9.]+) m", log.getvalue())
    track.telemetry_data = None
    zones_source = "FastF1 corner markers"
    if event.straight_mode_zones and track.straight_mode_zones is None:
        if round_number not in FALLBACK_CORNERS:
            raise ValueError(f"Round {round_number}: no corner markers for the Straight Mode zones")
        from models.telemetry import zone_intervals
        length, corners = FALLBACK_CORNERS[round_number]
        scale = track.total_length / length
        track.straight_mode_zones = zone_intervals(
            event.straight_mode_zones, {k: v * scale for k, v in corners.items()}, track.total_length)
        zones_source = "curvature-peak corner markers (predict_round.FALLBACK_CORNERS)"
        print("   Straight Mode zones (m): " + ", ".join(f"{a:.0f}–{b:.0f}" for a, b in track.straight_mode_zones)
              + " from " + zones_source)
    with _practice_envelope_config(used, cache) as practice:
        speed = practice.practice_speed(round_number, track, refresh=True)
    prepared = {
        "round": round_number, "event": event.name, "sessions": used, "track": track, "practice_speed": speed,
        "placement": {
            "session": placement_session, "raceline": raceline.name, "gap_m": float(gap.group(1)) if gap else None,
            "air_density": track.air_density, "length_m": float(track.total_length),
            "zones": [list(map(float, z)) for z in (track.straight_mode_zones or [])],
            "zones_source": zones_source,
        },
    }
    cache.mkdir(parents=True, exist_ok=True)
    with open(prepared_path, "wb") as f:
        pickle.dump(prepared, f)
    return prepared


def solve(prepared: Mapping, params: Mapping[str, float], g_max: float, ds: float):
    """The optimal qualifying lap on the prepared path. Raises SolverError unless Ipopt reports optimal."""
    from calibration.model import car_for
    from calibration.practice import bound_curvature
    from config import get_ers_config
    from models import VehicleDynamicsModel
    from solvers import SolverError, SpatialNLPSolver

    round_number = prepared["round"]
    vehicle, tires = car_for(params, prepared["placement"]["air_density"])
    ers = get_ers_config("2026", session="qualifying", event=round_number)
    track = copy.deepcopy(prepared["track"])
    kept = bound_curvature(track, prepared["practice_speed"], g_max) if g_max else None
    solver = SpatialNLPSolver(VehicleDynamicsModel(vehicle, ers, tires), track, ers, ds=ds)
    solver.verbose = False
    with contextlib.redirect_stdout(io.StringIO()):
        traj = solver.solve()
    if traj.solver_status != "optimal":
        raise SolverError(f"Round {round_number}: solver status {traj.solver_status}, not recorded")
    return traj, ers, kept


def settle(round_number: int, pole_s: Optional[float], pole_driver: str) -> List[Dict]:
    """Fetch the pole (FastF1 qualifying, or --pole) and append this round's unsettled predictions to the ledger."""
    from config.events import EVENTS_2026
    source = "manual"
    if pole_s is None:
        import fastf1 as ff1
        from models.telemetry import CACHE_DIR
        ff1.Cache.enable_cache(str(CACHE_DIR))
        session = ff1.get_session(2026, round_number, "Q")
        session.load(laps=True, telemetry=False, weather=False, messages=False)
        laps = session.laps[session.laps["LapTime"].notna()]
        if "Deleted" in laps:
            laps = laps[laps["Deleted"] != True]   # noqa: E712
        if len(laps) == 0:
            raise ValueError(f"Round {round_number}: no qualifying laps yet; try later or pass --pole")
        best = laps.sort_values("LapTime").iloc[0]
        pole_s, pole_driver, source = float(best["LapTime"].total_seconds()), str(best["Driver"]), "FastF1 Q"
    rows = ledger_rows(round_number, PREDICTIONS, pole_s, pole_driver, source,
                       dt.datetime.now(dt.timezone.utc), ledger_files(LEDGER))
    append_ledger(LEDGER, rows)
    for row in rows:
        print(f"R{round_number:02d} {EVENTS_2026[round_number].name}: {row['file']} predicted {row['predicted_s']} s, "
              f"pole {row['pole_s']} s ({row['pole_driver']}), error {row['error_s']} s ({row['error_pct']} %)")
    if not rows:
        print(f"R{round_number:02d}: nothing new to settle")
    return rows


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--round", type=int, required=True, help="2026 round number")
    parser.add_argument("--params", default="start", help="start, fit2, or a JSON file of parameter values")
    parser.add_argument("--sessions", default=",".join(DEFAULT_SESSIONS),
                        help="Sessions before qualifying to use, in order; the last available one places the raceline")
    parser.add_argument("--g-max", type=float, default=None,
                        help="Curvature bound (g) from the practice speeds; 0 = none (default: the set's, else 5)")
    parser.add_argument("--ds", type=float, default=5.0, help="Grid step (m)")
    parser.add_argument("--air-density", type=float, default=None,
                        help="Override the placement session's air density (kg/m³), e.g. for a reproduction check")
    parser.add_argument("--label", default="", help="Tag in the file name and the ledger (e.g. FP3)")
    parser.add_argument("--prepare-only", action="store_true", help="Place the raceline and build the envelope, no solve")
    parser.add_argument("--refresh", action="store_true", help="Rebuild the prepared path")
    parser.add_argument("--settle", action="store_true", help="After qualifying: append the pole and errors to the ledger")
    parser.add_argument("--pole", type=float, default=None, help="With --settle: the pole time (s), instead of FastF1")
    parser.add_argument("--pole-driver", default="", help="With --settle --pole: the pole sitter")
    args = parser.parse_args(argv)

    os.chdir(ROOT)                      # Cache and raceline paths are relative to the repository
    if args.settle:
        settle(args.round, args.pole, args.pole_driver)
        return 0

    from calibration.practice import G_MAX
    params_name, params = parameter_set(args.params)
    sessions = [s.strip() for s in args.sessions.split(",") if s.strip()]
    prepared = prepare(args.round, sessions, args.ds, refresh=args.refresh)
    print(f"   R{args.round:02d}: raceline on {prepared['placement']['session']} "
          f"({prepared['placement']['length_m']:.0f} m), sessions {'+'.join(prepared['sessions'])}, "
          f"air density {prepared['placement']['air_density'] or float('nan'):.4f} kg/m³")
    if args.prepare_only:
        return 0

    if args.air_density:
        placement = {**prepared["placement"], "air_density": args.air_density, "air_density_overridden": True}
        prepared = {**prepared, "placement": placement}
    g_max = args.g_max if args.g_max is not None else params.get("g_max", G_MAX)
    git = git_commit()
    traj, ers, kept = solve(prepared, params, g_max, args.ds)
    created = dt.datetime.now(dt.timezone.utc)
    record = prediction_record(
        round_number=args.round, event_name=prepared["event"], params_name=params_name, params=params,
        label=args.label, created=created, sessions=prepared["sessions"], placement=prepared["placement"],
        g_max=g_max, kept=kept, ers=ers, traj=traj, git=git)
    path = PREDICTIONS / prediction_name(args.round, record["params_hash"], created, args.label)
    PREDICTIONS.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1))
    print(summary_line(record, path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
