"""
Fit the shared car parameters to reference laps: least squares on a residual vector per round, the speed trace
at the measured samples and the lap time, with finite-difference Jacobians whose solves run in parallel.
"""
import json
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Mapping, Sequence

import numpy as np
from scipy.optimize import least_squares

from .dataset import reference_lap
from .model import PARAMETERS, model_lap, model_speed_at

SPEED_SIGMA = 8.0 / 3.6      # (m/s) Expected speed-trace error of a good model (ROADMAP Phase 4 target)
LAP_SIGMA = 0.2              # (s)
SPEED_SAMPLES = 100          # A round's speed trace weighs like this many independent samples


def round_residuals(params: Mapping[str, float], round_number: int) -> np.ndarray:
    """Residuals of one round: the speed trace (scaled so that it counts as SPEED_SAMPLES samples), then lap time."""
    reference = reference_lap(round_number)
    trajectory = model_lap(params, reference)
    if trajectory.solver_status != "optimal":
        raise RuntimeError(f"Round {round_number}: solver {trajectory.solver_status} for {dict(params)}")
    speed = model_speed_at(trajectory, reference.s, reference.track.total_length)
    weight = np.sqrt(SPEED_SAMPLES / len(reference.s))
    return np.concatenate([
        weight * (speed - reference.speed) / SPEED_SIGMA,
        [(trajectory.lap_time - reference.lap_time) / LAP_SIGMA],
    ])


def _job(args):
    params, round_number = args
    return round_residuals(params, round_number)


class Fit:
    """Least-squares fit of the named PARAMETERS (the others at their start values) to the given rounds."""

    def __init__(self, rounds: Sequence[int], names: Sequence[str], fixed: Mapping[str, float] = None, workers: int = 8):
        self.rounds, self.workers = list(rounds), workers
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
        jobs = [(self.params(x), r) for x in xs for r in self.rounds]
        with ProcessPoolExecutor(self.workers) as pool:
            flat = list(pool.map(_job, jobs))
        n = len(self.rounds)
        return [np.concatenate(flat[i * n:(i + 1) * n]) for i in range(len(xs))]

    def residuals(self, x) -> np.ndarray:
        key = tuple(np.round(x, 12))
        if key not in self._cache:
            self._cache[key] = self._evaluate_many([x])[0]
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
        for point, value in zip(points, values):
            self._cache[tuple(np.round(point, 12))] = value
        return np.column_stack([(v - base) / h for v, h in zip(values, steps)])

    def run(self, max_evaluations: int = 20):
        t0 = time.time()
        x0 = self.x_of(self.fixed)
        result = least_squares(self.residuals, x0, jac=self.jacobian, bounds=(0.0, 1.0), method="trf",
                               x_scale=0.05, max_nfev=max_evaluations, verbose=0)
        self.result = result
        print(f"   {result.message} ({result.nfev} evaluations, {time.time() - t0:.0f} s)")
        return self.params(result.x)


def report(params: Mapping[str, float], rounds: Sequence[int], workers: int = 8) -> List[Dict]:
    """Lap time error, speed RMSE and top speeds per round for a parameter set."""
    with ProcessPoolExecutor(workers) as pool:
        rows = list(pool.map(_report_row, [(dict(params), r) for r in rounds]))
    return rows


def _report_row(args):
    params, round_number = args
    reference = reference_lap(round_number)
    trajectory = model_lap(params, reference)
    speed = model_speed_at(trajectory, reference.s, reference.track.total_length)
    return {
        "round": round_number, "name": reference.name, "status": trajectory.solver_status,
        "lap_model": trajectory.lap_time, "lap_real": reference.lap_time,
        "lap_error_pct": 100.0 * (trajectory.lap_time - reference.lap_time) / reference.lap_time,
        "speed_rmse_kmh": 3.6 * float(np.sqrt(np.mean((speed - reference.speed) ** 2))),
        "top_model_kmh": 3.6 * float(trajectory.v_opt.max()), "top_real_kmh": 3.6 * reference.top_speed,
    }


def print_report(rows) -> None:
    for x in rows:
        print(f"   R{x['round']:02d} {x['name']:14s} lap {x['lap_model']:7.3f} vs {x['lap_real']:7.3f} ({x['lap_error_pct']:+6.2f} %)  "
              f"speed RMSE {x['speed_rmse_kmh']:5.1f} km/h  top {x['top_model_kmh']:4.0f} vs {x['top_real_kmh']:4.0f}  {x['status']}")
    errors = np.abs([x["lap_error_pct"] for x in rows])
    print(f"   |lap error|: median {np.median(errors):.2f} %, max {errors.max():.2f} %")


def save(params: Mapping[str, float], path) -> None:
    with open(path, "w") as f:
        json.dump(dict(params), f, indent=2)
