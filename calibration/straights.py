"""
ICE power and Corner Mode drag area from the straights alone (BUGS CAL-1, CAL-3).

On a 2026 qualifying lap the energy strategy between two corners is the team's choice, but two kinds of stretch
on a full-throttle run should be fixed by the rules, whatever that strategy is:

- (a) deploy: the first seconds after a corner exit, above 150 km/h, where the MGU-K deploys on the Overtake
  curve (350 kW DC up to 337.5 km/h), so 0.95 × 350 kW reaches the wheels;
- (b) clip: full throttle with the speed falling (the REFERENCE §7 super-clip detector), where the MGU-K harvests
  at the event's full-throttle limit (350 kW DC from Miami on, 250 kW before), so that /0.95 leaves the wheels.

A third, (c) terminal, the last seconds before braking that aren't clipping, is marked and modelled with the
deploy on the curve (its taper above 337.5 km/h); it is a check, not used in the fit, since below the taper the
deploy there is a strategy choice.

In (a) and (b) the measured acceleration, from the speed trace smoothed over about 1 s, is

    m·a = (P_ice + k·P_k)/v − ½ρv²·CdA·(1 + (r − 1)·w) − rolling − slope

with P_k the rule's MGU-K power at the wheels (k = 1), the session's air density, m = 772 + 4 kg (fuel), the
gradient and vertical curvature of the placed path, Straight Mode w = 1 inside the FIA zones after the activation
point and 0 elsewhere, and the Straight Mode drag ratio r fixed (CFD: 0.79–0.82, REFERENCE §2), since the fit
can't separate it from CdA (CAL-3). That is linear in P_ice and CdA, so least squares over every (a) and (b)
sample identifies both: (a) gives P_ice + 332 kW − drag at 150–280 km/h, (b) gives P_ice − 368 kW − drag at
250–330 km/h. A stretch whose implied MGU-K power, with the fitted car, is more than OUTLIER_KW from the rule's
breaks its assumption (no deploy after the corner, Baku style, or a partial harvest) and is left out with the
reason; the fit is repeated until the set stops changing.

As a diagnostic the MGU-K scale k can be fitted too (it is linear as well): k < 1 says the stretches see less
MGU-K power, both ways, than the rules allow.

Run: .venv/bin/python -m calibration.straights   (no NLP solves, about 1 min; figures and a JSON report in
data/cache/calibration/straights/)
"""
import argparse
import json
import logging
import pickle
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from config import get_ers_config, get_vehicle_config
from models.car import CarModel

from .dataset import lap_distances, reference_lap
from .practice import practice_lap

OUT = Path("data/cache/calibration/straights")
ROUNDS = (2, 3, 5, 8, 9, 10, 11, 13)        # The rounds with a qualifying session and a current raceline

FUEL = 4.0                  # (kg) Qualifying fuel on board (REFERENCE §2 estimate)
ETA_K = 0.95                # MGU-K and inverter efficiency, wheel ↔ DC bus (config/ers.py)
FULL_THROTTLE = 98.0        # (%) Full throttle
MIN_RUN = 2.0               # (s) Shortest full-throttle run
MAX_GAP = 1.5               # (s) A longer gap between samples (a dropped frozen block) splits a run
SMOOTH = 1.0                # (s) Window of the local linear fit for speed and acceleration
EXIT_SPEED = 150.0 / 3.6    # (m/s) Deploy stretches start once the car is this fast...
EXIT_START = 250.0 / 3.6    # (m/s) ... on runs that start below this (a corner exit, not a flat kink)
DEPLOY_TIME = 2.0           # (s) Length of a deploy stretch (1.5–2.5 s)
CLIP_SLOPE = -0.05          # ((km/h)/m) Super-clip detector: dv/ds below this at full throttle...
CLIP_SAMPLES = 5            # ... over at least this many samples (REFERENCE §7, j5t3313)
TERMINAL_TIME = 2.0         # (s) Terminal stretch before braking
OUTLIER_KW = 120.0          # (kW, wheel side) Stretches implying an MGU-K power further than this from the rule
RATIOS = (0.80, 0.79, 0.82)  # Straight Mode drag ratios: the fixed value and the sensitivity cases
PRIOR = (400e3, 0.95)       # Start of the fit: ICE power (W, REFERENCE §1) and Corner Mode CdA (m², §2)


# ---------------------------------------------------------------------------------------------- the laps
@dataclass
class TimedLap:
    """One reference lap's car-data samples in time order, with what the physics needs at each."""
    round: int
    kind: str                # "pole" (qualifying reference lap) or "practice" (fastest lap before qualifying)
    label: str               # e.g. "R13 Italy FP3 RUS 82.219"
    t: np.ndarray            # (s) From the first sample
    s: np.ndarray            # (m) Lap distance, continuous (may run slightly below 0 or past the lap length)
    v: np.ndarray            # (m/s)
    throttle: np.ndarray     # (%)
    brake: np.ndarray        # (0 or 1)
    gap: np.ndarray          # (m) Distance to the car ahead (NaN where unknown)
    w: np.ndarray            # Straight Mode allowed (1) or not (0): inside an FIA zone after its activation point
    gradient: np.ndarray     # (rad) Of the placed path
    kappa_v: np.ndarray      # (1/m) Vertical curvature of the placed path
    length: float            # (m) Lap length
    rho: float               # (kg/m³) Session air density
    superclip: float         # (W, DC) Harvest allowed at full throttle
    rules: str               # Energy rules of the session


def _session_name(reference) -> str:
    """FastF1 session of a reference lap: Q for the pole reference, else the suffix of a practice lap's name."""
    last = reference.name.split()[-1]
    return last if last in ("FP1", "FP2", "FP3", "SQ") else "Q"


def _car_samples(lap):
    """Car-data samples of a FastF1 lap with interpolated positions and, where FastF1 can, the gap ahead."""
    try:
        car = lap.get_telemetry()
    except Exception:
        car = lap.get_car_data().merge_channels(lap.get_pos_data(), frequency="original")
    car = car[car["Source"] == "car"].dropna(subset=["X", "Y", "Speed"])
    return car[~((car["Throttle"] >= 104) & car["Brake"].astype(bool))]       # Frozen blocks (TRK-7)


def timed_lap(round_number: int, kind: str = "pole", refresh: bool = False) -> TimedLap:
    """
    The pole reference lap (calibration/dataset.py) or the fastest practice lap (calibration/practice.py) of a
    round, reloaded from FastF1 in time order: their cached samples are sorted by distance and carry no time.
    Cached in data/cache/calibration/straights.
    """
    path = OUT / f"R{round_number:02d}_{kind}_timed.pkl"
    if path.exists() and not refresh:
        with open(path, "rb") as f:
            return pickle.load(f)
    from models.telemetry import load_session

    reference = reference_lap(round_number) if kind == "pole" else practice_lap(round_number)
    track = reference.track
    logging.getLogger("fastf1").setLevel(logging.ERROR)
    session, _, _ = load_session(2026, round_number, _session_name(reference))
    laps = session.laps[(session.laps["Driver"] == reference.driver) & session.laps["LapTime"].notna()]
    lap = laps.iloc[int(np.argmin(np.abs(laps["LapTime"].dt.total_seconds().to_numpy() - reference.lap_time)))]
    car = _car_samples(lap)

    t = car["SessionTime"].dt.total_seconds().to_numpy()
    t = t - t[0]
    v = car["Speed"].to_numpy(dtype=float) / 3.6
    s = lap_distances(car[["X", "Y"]].to_numpy(dtype=float) / 10.0, v, t, track)
    gap = car["DistanceToDriverAhead"].to_numpy(dtype=float) if "DistanceToDriverAhead" in car else np.full(len(t), np.nan)
    td, length = track.track_data, track.total_length
    on_path = lambda values: np.interp(np.mod(s, length), td.s, values, period=length)
    kappa_v = td.vertical_curvature if td.vertical_curvature is not None else np.zeros_like(td.s)
    mask = track.straight_mode_mask(s)
    ers = get_ers_config("2026", session=reference.rules, event=round_number)

    timed = TimedLap(
        round=round_number, kind=kind,
        label=f"R{round_number} {reference.name}{' Q' if kind == 'pole' else ''} {reference.driver} {reference.lap_time:.3f}",
        t=t, s=s, v=v, throttle=car["Throttle"].to_numpy(dtype=float), brake=car["Brake"].astype(float).to_numpy(),
        gap=gap, w=mask if mask is not None else np.zeros_like(s), gradient=on_path(td.gradient),
        kappa_v=on_path(kappa_v), length=length, rho=reference.air_density or 1.18,
        superclip=ers.superclip_power, rules=reference.rules,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(timed, f)
    return timed


# ----------------------------------------------------------------------------------------- smoothing
def smooth(t: np.ndarray, v: np.ndarray, window: float = SMOOTH, bounds: Optional[np.ndarray] = None):
    """
    Speed and acceleration from a local linear fit over ±window/2 around each sample (NaN acceleration with
    fewer than 3 samples). bounds, a [start, stop) index pair per sample, keeps each fit inside its own run, so
    the part-throttle samples around a run don't leak into it.
    """
    n = len(t)
    speed, accel = np.array(v, dtype=float), np.full(n, np.nan)
    for i in range(n):
        lo, hi = (0, n) if bounds is None else bounds[i]
        near = np.arange(lo, hi)
        near = near[np.abs(t[near] - t[i]) <= 0.5 * window + 1e-9]
        if len(near) < 3:
            continue
        slope, intercept = np.polyfit(t[near] - t[i], v[near], 1)
        speed[i], accel[i] = intercept, slope
    return speed, accel


# ------------------------------------------------------------------------------------------ segmenting
@dataclass
class Stretch:
    kind: str                # "deploy" (a), "clip" (b) or "terminal" (c)
    lap: str                 # TimedLap label
    round: int
    run: int                 # Index of the full-throttle run on the lap
    idx: np.ndarray          # Sample indices on the lap
    excluded: str = ""       # Why the stretch is left out of the fit ("" when it is used)
    implied_kw: float = float("nan")   # MGU-K power at the wheels the fitted car implies (+ deploy, − harvest)


@dataclass
class Run:
    lap: str
    round: int
    number: int
    i0: int                  # First and last sample (inclusive)
    i1: int
    starts_at_line: bool     # The run was already going when the lap started (no corner exit in this lap)
    ends_in_braking: bool    # False when the lap ends during the run
    min_gap_s: float         # Smallest time gap to the car ahead during the run (s; NaN if unknown)
    stretches: List[Stretch] = field(default_factory=list)


def full_throttle_runs(lap: TimedLap) -> List[Tuple[int, int]]:
    """(first, last) sample of each run at full throttle with the brake off, at least MIN_RUN long."""
    full = (lap.throttle >= FULL_THROTTLE) & (lap.brake < 0.5)
    runs, i, n = [], 0, len(full)
    while i < n:
        if not full[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and full[j + 1] and lap.t[j + 1] - lap.t[j] <= MAX_GAP:
            j += 1
        if lap.t[j] - lap.t[i] >= MIN_RUN:
            runs.append((i, j))
        i = j + 1
    return runs


def _flag_runs(flag: np.ndarray, min_len: int) -> List[Tuple[int, int]]:
    """(first, last) index of each run of True at least min_len long."""
    out, i, n = [], 0, len(flag)
    while i < n:
        if flag[i]:
            j = i
            while j + 1 < n and flag[j + 1]:
                j += 1
            if j - i + 1 >= min_len:
                out.append((i, j))
            i = j + 1
        else:
            i += 1
    return out


def segment(lap: TimedLap):
    """
    The lap's full-throttle runs with their (a) deploy, (b) clip and (c) terminal stretches, and the smoothed
    speed and acceleration (acceleration NaN outside the runs).
    """
    n = len(lap.t)
    bounds = np.zeros((n, 2), dtype=int)
    pairs = full_throttle_runs(lap)
    inside = np.zeros(n, dtype=bool)
    for i0, i1 in pairs:
        bounds[i0:i1 + 1] = (i0, i1 + 1)
        inside[i0:i1 + 1] = True
    v_s, a_s = smooth(lap.t, lap.v, SMOOTH, bounds)
    a_s[~inside] = np.nan
    slope = a_s * 3.6 / np.maximum(lap.v, 1.0)          # (km/h)/m

    runs = []
    for number, (i0, i1) in enumerate(pairs):
        idx = np.arange(i0, i1 + 1)
        gaps = lap.gap[idx] / np.maximum(lap.v[idx], 1.0)
        run = Run(lap.label, lap.round, number, i0, i1, starts_at_line=i0 == 0, ends_in_braking=i1 < n - 1,
                  min_gap_s=float(np.nanmin(gaps)) if np.isfinite(gaps).any() else float("nan"))

        # (b) Clipping: the REFERENCE §7 detector inside the run
        clip = np.zeros(n, dtype=bool)
        flag = np.zeros(n, dtype=bool)
        flag[idx] = slope[idx] < CLIP_SLOPE
        for c0, c1 in _flag_runs(flag, CLIP_SAMPLES):
            clip[c0:c1 + 1] = True
            run.stretches.append(Stretch("clip", lap.label, lap.round, number, np.arange(c0, c1 + 1)))
            # Inside the stretch only, so the deploy before it doesn't leak into the edges
            inner = np.tile([c0, c1 + 1], (c1 + 1 - c0, 1))
            v_c, a_c = smooth(lap.t[c0:c1 + 1], lap.v[c0:c1 + 1], SMOOTH, inner - c0)
            v_s[c0:c1 + 1], a_s[c0:c1 + 1] = v_c, a_c

        # (a) Deploy: the first DEPLOY_TIME after the run passes EXIT_SPEED, on a run that starts at a corner exit
        if not run.starts_at_line and lap.v[i0] < EXIT_START:
            fast = idx[lap.v[idx] >= EXIT_SPEED]
            if len(fast):
                t0 = lap.t[fast[0]]
                chosen = idx[(lap.t[idx] >= t0) & (lap.t[idx] <= t0 + DEPLOY_TIME) & ~clip[idx]]
                if len(chosen) >= 3:
                    run.stretches.append(Stretch("deploy", lap.label, lap.round, number, chosen))

        # (c) Terminal: the last TERMINAL_TIME before braking that isn't clipping
        if run.ends_in_braking:
            chosen = idx[(lap.t[idx] >= lap.t[i1] - TERMINAL_TIME) & ~clip[idx]]
            if len(chosen) >= 3:
                run.stretches.append(Stretch("terminal", lap.label, lap.round, number, chosen))
        runs.append(run)
    return runs, v_s, a_s


# ---------------------------------------------------------------------------------------------- physics
@dataclass(frozen=True)
class Car:
    p_ice: float             # (W) At the wheels, as in models/car.py
    cda: float               # (m²) Corner Mode
    ratio: float = 0.80      # Straight Mode drag / Corner Mode drag
    mgu_k_scale: float = 1.0  # MGU-K power / the rule's (1: deploy on the curve, harvest at the full-throttle limit)


def car_model(rho: float) -> CarModel:
    """The 2026 car of the model with 4 kg of fuel, this air density and no drag (drag is handled here)."""
    vehicle = replace(get_vehicle_config("2026"), fuel_mass=FUEL, rho_air=rho, c_w_a=0.0)
    return CarModel(vehicle)


def overtake_curve(v):
    """The Overtake deploy limit (W, DC) at speed v (m/s), C5.2.8: min(350, 7100 − 20·v_kph) kW, at least 0."""
    return np.clip((7100.0 - 20.0 * np.asarray(v, dtype=float) * 3.6) * 1e3, 0.0, 350e3)


def mgu_k_wheel_power(kind: str, v, superclip: float):
    """MGU-K power at the wheels (W) the rules give a stretch: + deploy on the Overtake curve, − full harvest."""
    v = np.asarray(v, dtype=float)
    if kind == "clip":
        return -superclip / ETA_K * np.ones_like(v)
    return ETA_K * overtake_curve(v)


def other_forces(lap: TimedLap, idx, v):
    """Rolling resistance and slope (N) at samples idx and speeds v, in each sample's aero mode."""
    return car_model(lap.rho).resistance(v, 0.0, lap.gradient[idx], lap.w[idx], lap.kappa_v[idx])


def drag_per_cda(lap: TimedLap, idx, v, ratio: float):
    """Drag (N) per m² of Corner Mode CdA, in each sample's aero mode."""
    return 0.5 * lap.rho * v**2 * (1.0 + (ratio - 1.0) * lap.w[idx])


def acceleration(lap: TimedLap, idx, v, car: Car, kind: str):
    """Model acceleration (m/s²) at samples idx and speeds v in a stretch of this kind."""
    p_k = car.mgu_k_scale * mgu_k_wheel_power(kind, v, lap.superclip)
    mass = car_model(lap.rho).mass
    return ((car.p_ice + p_k) / v - car.cda * drag_per_cda(lap, idx, v, car.ratio) - other_forces(lap, idx, v)) / mass


def simulate(lap: TimedLap, stretch: Stretch, v0: float, car: Car) -> np.ndarray:
    """Model speed (m/s) at the stretch's samples, integrated (RK4 on the sample times) from v0 at its first."""
    idx = stretch.idx
    out = np.empty(len(idx))
    out[0] = v0
    for k in range(len(idx) - 1):
        i = idx[k:k + 1]
        f = lambda v: float(acceleration(lap, i, np.array([v]), car, stretch.kind)[0])
        h, v = lap.t[idx[k + 1]] - lap.t[idx[k]], out[k]
        k1 = f(v)
        k2 = f(v + 0.5 * h * k1)
        k3 = f(v + 0.5 * h * k2)
        k4 = f(v + h * k3)
        out[k + 1] = v + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return out


# -------------------------------------------------------------------------------------------------- fit
@dataclass
class Prepared:
    """A lap with its runs, stretches and smoothed speed and acceleration."""
    lap: TimedLap
    runs: List[Run]
    v_s: np.ndarray
    a_s: np.ndarray

    @property
    def stretches(self) -> List[Stretch]:
        return [st for run in self.runs for st in run.stretches]


def prepare(rounds: Sequence[int] = ROUNDS, kinds: Sequence[str] = ("pole", "practice")) -> List[Prepared]:
    out = []
    for r in rounds:
        for kind in kinds:
            lap = timed_lap(r, kind)
            runs, v_s, a_s = segment(lap)
            out.append(Prepared(lap, runs, v_s, a_s))
    return out


def rows(p: Prepared, st: Stretch, ratio: float, free_scale: bool = False):
    """
    Least-squares rows of one stretch, in acceleration units: y = X·θ with θ = [P_ice (W), CdA (m²)] and, with
    free_scale, the MGU-K scale k. y = a + rolling and slope/m (− P_k/(m·v) when k is fixed at 1),
    X = [1/(m·v), −drag per CdA/m (, P_k/(m·v))]. Also returns the sample indices.
    """
    lap, idx = p.lap, st.idx
    idx = idx[np.isfinite(p.a_s[idx])]
    v, a = p.v_s[idx], p.a_s[idx]
    m = car_model(lap.rho).mass
    p_k = mgu_k_wheel_power(st.kind, v, lap.superclip)
    y = a + other_forces(lap, idx, v) / m
    columns = [1.0 / (m * v), -drag_per_cda(lap, idx, v, ratio) / m]
    if free_scale:
        columns.append(p_k / (m * v))
    else:
        y = y - p_k / (m * v)
    return y, np.column_stack(columns), idx


def implied_power(p: Prepared, st: Stretch, car: Car) -> float:
    """Mean MGU-K power at the wheels (W) this car needs to match the stretch's measured acceleration."""
    idx = st.idx[np.isfinite(p.a_s[st.idx])]
    v, a = p.v_s[idx], p.a_s[idx]
    m = car_model(p.lap.rho).mass
    needed = v * (m * a + car.cda * drag_per_cda(p.lap, idx, v, car.ratio) + other_forces(p.lap, idx, v)) - car.p_ice
    return float(np.mean(needed))


@dataclass
class FitResult:
    ratio: float
    p_ice: float             # (W)
    cda: float               # (m²) Corner Mode
    mgu_k_scale: float       # 1 unless fitted
    se_p_ice: float          # Standard errors, clustered by stretch
    se_cda: float
    se_mgu_k_scale: float    # NaN unless fitted
    corr: float              # Correlation of the P_ice and CdA estimates
    n_samples: int
    n_deploy: int            # Stretches used
    n_clip: int
    rms_accel: float         # (m/s²) Residual RMS of the fitted samples
    iterations: int

    @property
    def car(self) -> Car:
        return Car(self.p_ice, self.cda, self.ratio, self.mgu_k_scale)


def least_squares(stretches: Sequence[Tuple[Prepared, Stretch]], ratio: float, free_scale: bool = False):
    """Ordinary least squares over the stretches' samples: θ, its covariance (clustered by stretch), RMS, N."""
    blocks = [rows(p, st, ratio, free_scale) for p, st in stretches]
    Y = np.concatenate([b[0] for b in blocks])
    X = np.vstack([b[1] for b in blocks])
    scale = np.array([1e-5, 1.0, 1.0])[:X.shape[1]]      # P_ice in units of 100 kW, for conditioning
    Xs = X / scale
    theta_s, *_ = np.linalg.lstsq(Xs, Y, rcond=None)
    resid = Y - Xs @ theta_s
    bread = np.linalg.pinv(Xs.T @ Xs)
    meat = np.zeros((X.shape[1], X.shape[1]))
    start = 0
    for b in blocks:
        n = len(b[0])
        g = Xs[start:start + n].T @ resid[start:start + n]
        meat += np.outer(g, g)
        start += n
    G, N, K = len(blocks), len(Y), X.shape[1]
    cov_s = bread @ meat @ bread * (G / max(G - 1, 1)) * ((N - 1) / max(N - K, 1))
    cov = cov_s / np.outer(scale, scale)
    return theta_s / scale, cov, float(np.sqrt(np.mean(resid**2))), N


def fit(prepared: Sequence[Prepared], ratio: float = 0.80, free_scale: bool = False, start=PRIOR,
        outlier_kw: float = OUTLIER_KW, max_iterations: int = 30) -> Tuple[FitResult, List[Tuple[Prepared, Stretch]]]:
    """
    Fit P_ice and Corner Mode CdA (and with free_scale the MGU-K scale) over the (a) deploy and (b) clip
    stretches. Each stretch whose implied MGU-K power with the current car is more than outlier_kw from the
    rule's (times the scale) is left out, with the reason in its `excluded`, and the fit repeated from the prior
    until the set of stretches stops changing. Every candidate stretch gets its implied power in `implied_kw`.
    """
    candidates = [(p, st) for p in prepared for st in p.stretches if st.kind in ("deploy", "clip")]
    if len(candidates) < 3:
        raise ValueError(f"{len(candidates)} deploy and clip stretches: too few to fit")
    car = Car(start[0], start[1], ratio)
    fitted: Optional[List] = None
    for iteration in range(1, max_iterations + 1):
        used = []
        for p, st in candidates:
            implied = implied_power(p, st, car)
            rule = car.mgu_k_scale * float(np.mean(mgu_k_wheel_power(st.kind, p.v_s[st.idx], p.lap.superclip)))
            if abs(implied - rule) <= outlier_kw * 1e3:
                st.excluded = ""
                used.append((p, st))
            else:
                what = "deploy" if st.kind == "deploy" else "harvest"
                sign = 1.0 if st.kind == "deploy" else -1.0
                st.excluded = f"implies {sign * implied / 1e3:.0f} kW of {what} at the wheels, not {sign * rule / 1e3:.0f}"
        if len(used) < 3:
            if fitted is not None:
                break                                # Keep the last fit
            used = list(candidates)                  # Nothing near the rule at the prior: start from everything
        if fitted is not None and [id(st) for _, st in used] == [id(st) for _, st in fitted]:
            break
        fitted = used
        theta, cov, rms, n = least_squares(fitted, ratio, free_scale)
        car = Car(theta[0], theta[1], ratio, theta[2] if free_scale else 1.0)
    used = fitted
    for p, st in candidates:
        st.implied_kw = implied_power(p, st, car) / 1e3
        if any(st is s for _, s in used):
            st.excluded = ""
    se = np.sqrt(np.diag(cov))
    result = FitResult(
        ratio=ratio, p_ice=car.p_ice, cda=car.cda, mgu_k_scale=car.mgu_k_scale, se_p_ice=float(se[0]),
        se_cda=float(se[1]), se_mgu_k_scale=float(se[2]) if free_scale else float("nan"),
        corr=float(cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])), n_samples=n,
        n_deploy=sum(st.kind == "deploy" for _, st in used), n_clip=sum(st.kind == "clip" for _, st in used),
        rms_accel=rms, iterations=iteration,
    )
    return result, used


def ceiling_stretches(prepared: Sequence[Prepared], top: int = 1, car: Car = Car(*PRIOR)) -> List[Tuple[Prepared, Stretch]]:
    """
    The `top` (a) deploy stretches of each lap with the largest implied deploy (ranked with `car`; the ranking
    hardly depends on it): the corner exits most likely at the full 350 kW, if any exit of the lap is.
    """
    out = []
    for p in prepared:
        deploy = sorted((st for st in p.stretches if st.kind == "deploy"), key=lambda st: -implied_power(p, st, car))
        out += [(p, st) for st in deploy[:top]]
    return out


def fit_ceiling(prepared: Sequence[Prepared], ratio: float = 0.80, top: int = 1, cda: Optional[float] = None) -> FitResult:
    """
    P_ice and CdA (or P_ice alone at a fixed cda) from each lap's best corner exits, taken at the full deploy:
    the reading that survives when most exits aren't at full deploy and most clipping isn't at full harvest.
    """
    chosen = ceiling_stretches(prepared, top)
    if cda is None:
        theta, cov, rms, n = least_squares(chosen, ratio)
        se = np.sqrt(np.diag(cov))
        p_ice, cda, se_p, se_c, corr = theta[0], theta[1], se[0], se[1], cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    else:
        blocks = [rows(p, st, ratio) for p, st in chosen]
        Y = np.concatenate([b[0] for b in blocks]) - cda * np.concatenate([b[1][:, 1] for b in blocks])
        x = np.concatenate([b[1][:, 0] for b in blocks])
        p_ice = float(np.sum(x * Y) / np.sum(x * x))
        resid = Y - x * p_ice
        # Clustered by stretch
        start, meat = 0, 0.0
        for b in blocks:
            k = len(b[0])
            meat += float(np.sum(x[start:start + k] * resid[start:start + k])) ** 2
            start += k
        se_p, se_c, corr, n = float(np.sqrt(meat) / np.sum(x * x)), float("nan"), float("nan"), len(Y)
        rms = float(np.sqrt(np.mean(resid**2)))
    return FitResult(ratio=ratio, p_ice=float(p_ice), cda=float(cda), mgu_k_scale=1.0, se_p_ice=float(se_p),
                     se_cda=float(se_c), se_mgu_k_scale=float("nan"), corr=float(corr), n_samples=n,
                     n_deploy=len(chosen), n_clip=0, rms_accel=rms, iterations=1)


def speed_residuals(prepared: Sequence[Prepared], car: Car, used=None) -> Dict[str, Dict[str, float]]:
    """
    Speed residual RMS (km/h) per stretch kind, the model integrated over each stretch from the measured
    (smoothed) speed at its start: over the stretches in `used` (default: those not excluded from the main fit;
    (c) stretches are never excluded) and over all of them.
    """
    collected = {kind: {"used": [], "all": []} for kind in ("deploy", "clip", "terminal")}
    chosen = None if used is None else {id(st) for _, st in used}
    for p in prepared:
        for st in p.stretches:
            v_model = simulate(p.lap, st, p.v_s[st.idx[0]], car)
            err = list((v_model - p.lap.v[st.idx]) * 3.6)
            collected[st.kind]["all"] += err
            if (not st.excluded) if chosen is None else (id(st) in chosen):
                collected[st.kind]["used"] += err
    rms = lambda e: float(np.sqrt(np.mean(np.square(e)))) if e else float("nan")
    return {kind: {"rms_used": rms(c["used"]), "rms_all": rms(c["all"]), "n_used": len(c["used"]), "n_all": len(c["all"])}
            for kind, c in collected.items()}


def frontier(prepared: Sequence[Prepared], car: Car) -> List[Dict]:
    """
    Per lap: the largest implied MGU-K deploy at the wheels over its (a) stretches and the largest implied
    harvest over its (b) stretches, with this car. A full deploy somewhere on the lap puts the first at the
    rule's 332 kW; a full harvest puts the second at 368 kW (263 kW before Miami).
    """
    out = []
    for p in prepared:
        deploy = [implied_power(p, st, car) / 1e3 for st in p.stretches if st.kind == "deploy"]
        harvest = [-implied_power(p, st, car) / 1e3 for st in p.stretches if st.kind == "clip"]
        out.append(dict(lap=p.lap.label, round=p.lap.round, kind=p.lap.kind,
                        max_deploy_kw=max(deploy) if deploy else float("nan"),
                        max_harvest_kw=max(harvest) if harvest else float("nan"),
                        rule_deploy_kw=ETA_K * 350.0, rule_harvest_kw=p.lap.superclip / 1e3 / ETA_K))
    return out


# ----------------------------------------------------------------------------------------------- report
def _stretch_rows(prepared: Sequence[Prepared]) -> List[Dict]:
    out = []
    for p in prepared:
        lap = p.lap
        for run in p.runs:
            for st in run.stretches:
                v = lap.v[st.idx] * 3.6
                out.append(dict(
                    lap=lap.label, round=lap.round, run=run.number, kind=st.kind,
                    s_from=float(lap.s[st.idx[0]] % lap.length), s_to=float(lap.s[st.idx[-1]] % lap.length),
                    v_from=float(v[0]), v_to=float(v[-1]), seconds=float(lap.t[st.idx[-1]] - lap.t[st.idx[0]]),
                    straight_mode=float(lap.w[st.idx].mean()), min_gap_s=run.min_gap_s,
                    implied_kw=st.implied_kw, excluded=st.excluded,
                ))
    return out


def run_all(rounds: Sequence[int] = ROUNDS, figures: bool = True, out: Path = OUT) -> Dict:
    """The fits, sensitivities, per-round checks, residuals and figures; returns the report (also as JSON)."""
    from .plots import implied_power_scatter, straight_panels

    prepared = prepare(rounds)
    prior = Car(PRIOR[0], PRIOR[1], 0.80)
    report: Dict = {"settings": dict(rounds=list(rounds), fuel=FUEL, eta_k=ETA_K, smooth_s=SMOOTH,
                                     deploy_time_s=DEPLOY_TIME, exit_kph=EXIT_SPEED * 3.6, outlier_kw=OUTLIER_KW,
                                     prior=list(PRIOR))}

    # Without exclusion: what every candidate stretch says
    every = [(p, st) for p in prepared for st in p.stretches if st.kind in ("deploy", "clip")]
    report["no_exclusion"] = {}
    for name, sel in (("all", every), ("deploy only", [c for c in every if c[1].kind == "deploy"]),
                      ("clip only", [c for c in every if c[1].kind == "clip"])):
        theta, cov, rms, n = least_squares(sel, 0.80)
        report["no_exclusion"][name] = dict(
            p_ice_kw=theta[0] / 1e3, cda=theta[1], se_p_ice_kw=np.sqrt(cov[0, 0]) / 1e3, se_cda=np.sqrt(cov[1, 1]),
            corr=cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1]), n_stretches=len(sel), rms_accel=rms)

    # With CdA fixed at the prior: the ICE power each kind of stretch implies
    report["p_ice_kw_at_prior_cda"] = {}
    for kind in ("deploy", "clip"):
        sel = [c for c in every if c[1].kind == kind]
        Y = np.concatenate([rows(p, st, 0.80)[0] for p, st in sel])
        X = np.vstack([rows(p, st, 0.80)[1] for p, st in sel])
        report["p_ice_kw_at_prior_cda"][kind] = float(np.sum(X[:, 0] * (Y - X[:, 1] * PRIOR[1])) / np.sum(X[:, 0] ** 2)) / 1e3

    # The brief's fit at the three ratios, with the MGU-K scale free, and per round (both laps of a round)
    report["fits"] = {f"{ratio:.2f}": asdict(fit(prepared, ratio)[0]) for ratio in RATIOS}
    report["fit_mgu_k_scale_free"] = asdict(fit(prepared, 0.80, free_scale=True)[0])
    report["per_round"] = {}
    for r in rounds:
        try:
            report["per_round"][r] = asdict(fit([p for p in prepared if p.lap.round == r], 0.80)[0])
        except ValueError as error:
            report["per_round"][r] = {"error": str(error)}

    # The ceiling reading: each lap's best corner exit at the full deploy
    report["ceiling"] = {f"top{top} {ratio:.2f}": asdict(fit_ceiling(prepared, ratio, top)) for top in (1, 2) for ratio in RATIOS}
    report["ceiling_at_prior_cda"] = asdict(fit_ceiling(prepared, 0.80, 1, cda=PRIOR[1]))
    report["ceiling_per_round_at_prior_cda"] = {
        r: asdict(fit_ceiling([p for p in prepared if p.lap.round == r], 0.80, 1, cda=PRIOR[1])) for r in rounds}
    report["ceiling_per_lap_at_prior_cda"] = {
        p.lap.label: fit_ceiling([p], 0.80, 1, cda=PRIOR[1]).p_ice / 1e3 for p in prepared}
    ceiling = fit_ceiling(prepared, 0.80, 1, cda=PRIOR[1])

    # The brief's fit again, started from the ceiling car instead of the prior: the other fixed point of the
    # exclusion loop. It is the main fit: it sets the exclusions and implied powers kept in the report and figures.
    start = (ceiling.p_ice, PRIOR[1])
    report["fits_from_ceiling"] = {f"{ratio:.2f}": asdict(fit(prepared, ratio, start=start)[0]) for ratio in RATIOS}
    main, _ = fit(prepared, 0.80, start=start)
    report["main"] = asdict(main)
    report["excluded"] = [dict(lap=p.lap.label, kind=st.kind, s_from=float(p.lap.s[st.idx[0]] % p.lap.length),
                               s_to=float(p.lap.s[st.idx[-1]] % p.lap.length), reason=st.excluded)
                          for p in prepared for st in p.stretches if st.excluded]
    report["speed_rms_kph"] = {
        "main fit": speed_residuals(prepared, main.car),
        "prior car": speed_residuals(prepared, prior),
        "ceiling car, best exits": speed_residuals(prepared, ceiling.car, ceiling_stretches(prepared, 1)),
    }
    report["frontier_at_prior"] = frontier(prepared, prior)
    report["stretches"] = _stretch_rows(prepared)

    out.mkdir(parents=True, exist_ok=True)
    if figures:
        best = {id(st) for _, st in ceiling_stretches(prepared, 1)}
        models = [
            (f"prior {PRIOR[0] / 1e3:.0f} kW, {PRIOR[1]:.2f} m²", prior, dict(lw=1.0, ls=":")),
            (f"ceiling {ceiling.p_ice / 1e3:.0f} kW, {ceiling.cda:.2f} m²", ceiling.car, dict(lw=1.6, ls="-")),
        ]
        for p in prepared:
            straight_panels(p, out / f"R{p.lap.round:02d}_{p.lap.kind}.png", models,
                            lambda pp, st, car: simulate(pp.lap, st, pp.v_s[st.idx[0]], car), best,
                            implied=lambda pp, st: implied_power(pp, st, prior) / 1e3,
                            title=f"{p.lap.label}: model speed on each stretch from its measured start (Straight Mode ×0.80)\n"
                                  f"shaded: green (a) deploy, red (b) clip, blue (c) terminal, pale = left out of the "
                                  f"main fit; numbers: implied MGU-K kW at the wheels for the prior car; * the lap's best exit")
        points = []
        for p in prepared:
            m = car_model(p.lap.rho).mass
            for st in p.stretches:
                idx = st.idx[np.isfinite(p.a_s[st.idx])]
                v, a = p.v_s[idx], p.a_s[idx]
                need = v * (m * a + prior.cda * drag_per_cda(p.lap, idx, v, 0.80) + other_forces(p.lap, idx, v)) - prior.p_ice
                points += [(st.kind, vi * 3.6, ni / 1e3, id(st) in best) for vi, ni in zip(v, need)]
        implied_power_scatter(points, out / "implied_power.png",
                              f"Implied MGU-K power at the wheels, P_ice {PRIOR[0] / 1e3:.0f} kW, CdA {PRIOR[1]:.2f} m², "
                              f"Straight Mode ×0.80\ndotted: the rules' +332, −263 and −368 kW; large: each lap's best exit")
    with open(out / "straights_fit.json", "w") as f:
        json.dump(report, f, indent=1, default=float)
    return report


def print_report(report: Dict) -> None:
    kw = lambda x: f"{x / 1e3:.0f}"

    def line(label, r):
        corr = f", corr {r['corr']:+.2f}" if np.isfinite(r["corr"]) else ""
        se_c = f" ± {r['se_cda']:.3f}" if np.isfinite(r["se_cda"]) else " (fixed)"
        k = f", k {r['mgu_k_scale']:.2f} ± {r['se_mgu_k_scale']:.2f}" if np.isfinite(r["se_mgu_k_scale"]) else ""
        print(f"  {label:22s} P_ice {kw(r['p_ice']):>4} ± {kw(r['se_p_ice']):>3} kW, CdA {r['cda']:6.3f}{se_c}{k}{corr}, "
              f"{r['n_deploy']} deploy + {r['n_clip']} clip stretches, RMS {r['rms_accel']:.2f} m/s²")

    print("No exclusion, Straight Mode ×0.80 (every (a) and (b) stretch at the rule's MGU-K power):")
    for name, r in report["no_exclusion"].items():
        print(f"  {name:22s} P_ice {r['p_ice_kw']:4.0f} ± {r['se_p_ice_kw']:3.0f} kW, CdA {r['cda']:6.3f} ± {r['se_cda']:.3f}, "
              f"corr {r['corr']:+.2f}, {r['n_stretches']} stretches")
    f = report["p_ice_kw_at_prior_cda"]
    print(f"  P_ice at CdA {PRIOR[1]}: {f['deploy']:.0f} kW from the deploy stretches, {f['clip']:.0f} kW from the clip stretches")
    print(f"Brief's fit (iterated exclusion at ±{OUTLIER_KW:.0f} kW from the prior):")
    for label, r in report["fits"].items():
        line(f"ratio {label}", r)
    line("ratio 0.80, k free", report["fit_mgu_k_scale_free"])
    for r, x in report["per_round"].items():
        if "error" in x:
            print(f"  R{r}: {x['error']}")
        else:
            line(f"R{r}", x)
    print("Ceiling: each lap's best exit at the full deploy:")
    for label, r in report["ceiling"].items():
        line(label, r)
    line("top1 0.80, CdA fixed", report["ceiling_at_prior_cda"])
    for r, x in report["ceiling_per_round_at_prior_cda"].items():
        line(f"R{r}, CdA fixed", x)
    print("Brief's fit started from the ceiling car (the main fit):")
    for label, r in report["fits_from_ceiling"].items():
        line(f"ratio {label}", r)
    print(f"Left out of the main fit: {len(report['excluded'])} stretches (see straights_fit.json)")
    print("Speed residual RMS (km/h) per stretch kind: used | all")
    for car, kinds in report["speed_rms_kph"].items():
        print(f"  {car:24s} " + "   ".join(f"{kind} {x['rms_used']:5.1f} (n {x['n_used']:3d}) | {x['rms_all']:5.1f} (n {x['n_all']:3d})"
                                          for kind, x in kinds.items()))
    print("Per lap, largest implied deploy and harvest at the wheels with the prior car (kW):")
    for x in report["frontier_at_prior"]:
        print(f"  {x['lap']:36s} deploy {x['max_deploy_kw']:5.0f} (rule {x['rule_deploy_kw']:.0f}), "
              f"harvest {x['max_harvest_kw']:5.0f} (rule {x['rule_harvest_kw']:.0f})")


OLDER_POLES = ((2024, 16), (2025, 19), (2025, 20), (2025, 21))   # Pole laps of the 2024–25 cars in the shared cache


def exit_power(year: int, round_number: int, cdas=(1.0, 1.2), fuel: float = 5.0) -> List[Dict]:
    """
    Total power at the wheels (kW) on each corner exit of a qualifying session's fastest lap, from the same
    (a) stretches and the model's resistance with the given CdA: v·(m·a + drag + rolling). For a 2014–25 car
    (MGU-K 120 kW, deployed at every exit) it shows how the measured ceiling compares with the rated power.
    """
    from models.telemetry import load_session
    from .practice import session_air_density

    logging.getLogger("fastf1").setLevel(logging.ERROR)
    session, _, _ = load_session(year, round_number, "Q")
    rho = session_air_density(session) or 1.18
    car = session.laps.pick_fastest().get_car_data()
    car = car[~((car["Throttle"] >= 104) & car["Brake"].astype(bool))]
    t = car["SessionTime"].dt.total_seconds().to_numpy()
    t, v, n = t - t[0], car["Speed"].to_numpy(dtype=float) / 3.6, len(car)
    zeros = np.zeros(n)
    lap = TimedLap(0, "older", f"{year} R{round_number}", t, zeros, v, car["Throttle"].to_numpy(dtype=float),
                   car["Brake"].astype(float).to_numpy(), zeros, zeros, zeros, zeros, 1e9, rho, 0.0, "qualifying")
    runs, v_s, a_s = segment(lap)
    vehicle = replace(get_vehicle_config(str(year) if year >= 2026 else "2025"), fuel_mass=fuel, rho_air=rho, c_w_a=0.0)
    model = CarModel(vehicle)
    out = []
    for run in runs:
        for st in run.stretches:
            if st.kind != "deploy":
                continue
            idx = st.idx[np.isfinite(a_s[st.idx])]
            vv, aa = v_s[idx], a_s[idx]
            rolling = model.resistance(vv, 0.0, 0.0, 0.0)
            out.append(dict(v_from=float(vv[0] * 3.6), v_to=float(vv[-1] * 3.6), **{
                f"total_kw_cda_{cda:g}": float(np.mean(vv * (model.mass * aa + rolling + 0.5 * rho * vv**2 * cda))) / 1e3
                for cda in cdas}))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--rounds", type=int, nargs="*", default=list(ROUNDS))
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--older", action="store_true", help="also the corner-exit power of 2024–25 pole laps")
    args = parser.parse_args(argv)
    report = run_all(args.rounds, figures=not args.no_figures)
    print_report(report)
    if args.older:
        print("Corner-exit total power at the wheels, 2024–25 pole laps (rated about 575 + 120 kW at the crank):")
        older = {}
        for year, round_number in OLDER_POLES:
            exits = exit_power(year, round_number)
            older[f"{year} R{round_number}"] = exits
            print(f"  {year} R{round_number}: " + ", ".join(
                f"{x['v_from']:.0f}–{x['v_to']:.0f} km/h {x['total_kw_cda_1']:.0f}/{x['total_kw_cda_1.2']:.0f}" for x in exits)
                  + "  (kW at CdA 1.0/1.2)")
        report["older_exit_power"] = older
        with open(OUT / "straights_fit.json", "w") as f:
            json.dump(report, f, indent=1, default=float)


if __name__ == "__main__":
    main()
