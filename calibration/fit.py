"""
Fit the shared car parameters to reference laps: least squares on a residual vector per round, the speed trace
at the measured samples and the lap time (and optionally the lap's time structure, calibration/structure.py), with
finite-difference Jacobians whose solves run in parallel.
"""
import json
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Mapping, Sequence

import numpy as np
from scipy.optimize import least_squares

from .dataset import reference_lap
from .model import PARAMETERS, model_lap, model_speed_at
from .structure import DURATIONS, reference_gaps, reference_structure, trajectory_structure

SPEED_SIGMA = 8.0 / 3.6      # (m/s) Expected speed-trace error of a good model (ROADMAP Phase 4 target)
LAP_SIGMA = 0.2              # (s)
SPEED_SAMPLES = 100          # A round's speed trace weighs like this many independent samples
STRUCTURE_SIGMA = 1.5        # (s) Per time-structure duration: a 3 s miss weighs like a 0.4 s lap-time error


def structure_residuals(trajectory, reference) -> np.ndarray:
    """
    Seconds at full throttle, at part throttle, braking and clipping (calibration/structure.py), model minus
    reference, over STRUCTURE_SIGMA. The model is measured on its own 5 m intervals; the reference lap's gaps
    (over 0.75 s without a sample) are left out of both laps.
    """
    excluded = reference_gaps(reference)
    model = trajectory_structure(trajectory, sparc=False, exclude=excluded).durations()
    real = reference_structure(reference, sparc=False, exclude=excluded).durations()
    return (model - real) / STRUCTURE_SIGMA


def round_residuals(params: Mapping[str, float], round_number: int, structure: bool = False) -> np.ndarray:
    """
    Residuals of one round: the speed trace (scaled so that it counts as SPEED_SAMPLES samples), then lap time,
    then with structure=True the four time-structure durations (structure_residuals).
    """
    reference = reference_lap(round_number)
    trajectory = model_lap(params, reference)
    if trajectory.solver_status != "optimal":
        raise RuntimeError(f"Round {round_number}: solver {trajectory.solver_status} for {dict(params)}")
    speed = model_speed_at(trajectory, reference.s, reference.track.total_length)
    weight = np.sqrt(SPEED_SAMPLES / len(reference.s))
    parts = [weight * (speed - reference.speed) / SPEED_SIGMA, [(trajectory.lap_time - reference.lap_time) / LAP_SIGMA]]
    if structure:
        parts.append(structure_residuals(trajectory, reference))
    return np.concatenate(parts)


FAILED = 1e3                 # Residual given to every entry when a solve fails, so that the step is rejected


def _job(args):
    """round_residuals, or None when the solve fails (reported, so that one bad point doesn't end a fit)."""
    params, round_number, structure = args
    try:
        return round_residuals(params, round_number, structure)
    except Exception as error:
        print(f"   ! round {round_number} failed ({error}) at " + ", ".join(f"{k} {v:.4g}" for k, v in params.items()), flush=True)
        return None


class Fit:
    """
    Least-squares fit of the named PARAMETERS (the others at their start values) to the given rounds; structure=True
    adds the time-structure residuals.
    """

    def __init__(self, rounds: Sequence[int], names: Sequence[str], fixed: Mapping[str, float] = None, workers: int = 8,
                 structure: bool = False):
        self.rounds, self.workers, self.structure = list(rounds), workers, structure
        self.parameters = [p for p in PARAMETERS if p.name in names]
        self.fixed = {p.name: p.start for p in PARAMETERS}
        self.fixed.update(fixed or {})
        self.history: List[Dict] = []
        self._cache = {}

    # Parameters in units of their range, so every step size means the same
    def params(self, x) -> Dict[str, float]:
        values = dict(self.fixed)
        for p, xi in zip(self.parameters, x):
            values[p.name] = p.lower + xi * (p.upper - p.lower)
        return values

    def x_of(self, values: Mapping[str, float]) -> np.ndarray:
        return np.array([(values[p.name] - p.lower) / (p.upper - p.lower) for p in self.parameters])

    def _evaluate_many(self, xs) -> List[np.ndarray]:
        """Residual vector for each point, or None where any round's solve failed."""
        jobs = [(self.params(x), r, self.structure) for x in xs for r in self.rounds]
        with ProcessPoolExecutor(self.workers) as pool:
            flat = list(pool.map(_job, jobs))
        n = len(self.rounds)
        rows = [flat[i * n:(i + 1) * n] for i in range(len(xs))]
        return [None if any(part is None for part in row) else np.concatenate(row) for row in rows]

    def residuals(self, x) -> np.ndarray:
        key = tuple(np.round(x, 12))
        if key not in self._cache:
            value = self._evaluate_many([x])[0]
            if value is None:
                if not self._cache:
                    raise RuntimeError("The start point doesn't solve")
                value = np.full(len(next(iter(self._cache.values()))), FAILED)
            self._cache[key] = value
            cost = 0.5 * float(self._cache[key] @ self._cache[key])
            self.history.append({"params": self.params(x), "cost": cost})
            print(f"   cost {cost:10.2f}  " + "  ".join(f"{p.name} {self.params(x)[p.name]:.4f}" for p in self.parameters), flush=True)
        return self._cache[key]

    def jacobian(self, x, step: float = 0.01) -> np.ndarray:
        """Forward differences, all perturbed solves at once (away from the upper bound)."""
        base = self.residuals(x)
        steps = [step if xi + step <= 1.0 else -step for xi in x]
        points = [x + h * np.eye(len(x))[i] for i, h in enumerate(steps)]
        values = self._evaluate_many(points)
        # A failed solve: try the step the other way, else leave that column out
        retry = [i for i, v in enumerate(values) if v is None]
        if retry:
            flipped = self._evaluate_many([x - steps[i] * np.eye(len(x))[i] for i in retry])
            for i, value in zip(retry, flipped):
                steps[i], values[i] = -steps[i], value
        columns = [np.zeros_like(base) if v is None else (v - base) / h for v, h in zip(values, steps)]
        return np.column_stack(columns)

    def run(self, max_evaluations: int = 20):
        t0 = time.time()
        x0 = self.x_of(self.fixed)
        result = least_squares(self.residuals, x0, jac=self.jacobian, bounds=(0.0, 1.0), method="trf",
                               x_scale=0.05, max_nfev=max_evaluations, verbose=0)
        self.result = result
        print(f"   {result.message} ({result.nfev} evaluations, {time.time() - t0:.0f} s)")
        return self.params(result.x)


def report(params: Mapping[str, float], rounds: Sequence[int], workers: int = 8) -> List[Dict]:
    """Lap time error, speed RMSE, top speeds and time structure per round for a parameter set."""
    with ProcessPoolExecutor(workers) as pool:
        rows = list(pool.map(_report_row, [(dict(params), r) for r in rounds]))
    return rows


def report_row(trajectory, reference) -> Dict:
    """One report row: lap time, speed trace, top speed, and the time structure of model and reference."""
    speed = model_speed_at(trajectory, reference.s, reference.track.total_length)
    row = {
        "round": reference.round, "name": reference.name, "status": trajectory.solver_status,
        "lap_model": trajectory.lap_time, "lap_real": reference.lap_time,
        "lap_error_pct": 100.0 * (trajectory.lap_time - reference.lap_time) / reference.lap_time,
        "speed_rmse_kmh": 3.6 * float(np.sqrt(np.mean((speed - reference.speed) ** 2))),
        "top_model_kmh": 3.6 * float(trajectory.v_opt.max()), "top_real_kmh": 3.6 * reference.top_speed,
    }
    # The model on its own intervals (what the residuals use) and sampled at the reference's distances
    track = reference.track
    for label, measured in (("model", trajectory_structure(trajectory, track=track)),
                            ("model_at_ref", trajectory_structure(trajectory, at=reference.s, track=track)),
                            ("real", reference_structure(reference))):
        row.update({f"{k}_{label}": v for k, v in measured.as_dict().items()})
    # What the residuals see: the reference's gaps left out of both laps
    row["gap_s"] = row["total_real"] - reference_structure(reference, sparc=False, exclude=reference_gaps(reference)).total
    row["structure_residuals"] = structure_residuals(trajectory, reference).tolist()
    return row


def _report_row(args):
    params, round_number = args
    reference = reference_lap(round_number)
    return report_row(model_lap(params, reference), reference)


def print_report(rows) -> None:
    for x in rows:
        print(f"   R{x['round']:02d} {x['name']:14s} lap {x['lap_model']:7.3f} vs {x['lap_real']:7.3f} ({x['lap_error_pct']:+6.2f} %)  "
              f"speed RMSE {x['speed_rmse_kmh']:5.1f} km/h  top {x['top_model_kmh']:4.0f} vs {x['top_real_kmh']:4.0f}  {x['status']}")
    errors = np.abs([x["lap_error_pct"] for x in rows])
    print(f"   |lap error|: median {np.median(errors):.2f} %, max {errors.max():.2f} %")
    if rows and "full_model" in rows[0]:
        print_structure(rows)


def print_structure(rows) -> None:
    """
    Seconds per lap at full throttle, part throttle, braking and clipping, model vs real (whole laps), the SPARC
    clip, the reference's gaps, and the residuals the fit sees (gaps left out of both laps, in units of
    STRUCTURE_SIGMA) with their cost.
    """
    print("   time structure (s per lap, model/real)  " + "  ".join(f"{k:>11s}" for k in DURATIONS)
          + "   SPARC clip  gaps | residuals " + " ".join(f"{k:>5s}" for k in DURATIONS) + "   cost")
    for x in rows:
        cells = "  ".join(f"{x[f'{k}_model']:5.1f}/{x[f'{k}_real']:5.1f}" for k in DURATIONS)
        r = np.array(x["structure_residuals"])
        print(f"   R{x['round']:02d} {x['name']:14s}{'':23s}  {cells}  {x['clip_sparc_model']:5.1f}/{x['clip_sparc_real']:5.1f}"
              f"  {x['gap_s']:4.1f} |           " + " ".join(f"{v:+5.1f}" for v in r) + f"  {0.5 * float(r @ r):5.1f}")


def save(params: Mapping[str, float], path) -> None:
    with open(path, "w") as f:
        json.dump(dict(params), f, indent=2)
