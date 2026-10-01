"""
Joint fit: the shared car parameters plus one aero level per track.

Teams bring low-drag wings to Monza and Spa and high-downforce ones to Budapest, so one car can't match every
track's straights and fast corners. Each track gets an aero level k: ClA × (1 + k) and CdA × (1 + DRAG_PER_DOWNFORCE·k).
Every round's k is set by the fastest lap of its sessions before qualifying (calibration/practice.py), solved under
that session's energy rules: practice caps are up to 2.5 MJ higher than qualifying ones, so a practice lap clips
less on the straights, and fitting it with the qualifying rules would read that as less drag. The pole laps of the
training rounds set the shared parameters. A held-out round's pole lap is never used, so its lap time stays a
prediction (ROADMAP Phase 4, per-track scalars from non-pole sessions).
"""
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Mapping, Sequence

import numpy as np
from scipy.optimize import least_squares

from .dataset import reference_lap
from .fit import FAILED, LAP_SIGMA, SPEED_SAMPLES, SPEED_SIGMA
from .model import PARAMETERS, model_lap, model_speed_at
from .practice import practice_lap

DRAG_PER_DOWNFORCE = 0.5     # Relative drag change per relative downforce change between wing levels
AERO_LEVEL_BOUNDS = (-0.3, 0.3)


def round_params(params: Mapping[str, float], round_number: int) -> Dict[str, float]:
    """The car for one round: the shared parameters with that round's aero level applied."""
    values = {k: v for k, v in params.items() if "@" not in k}
    level = params.get(f"aero_level@{round_number}", 0.0)
    values["c_z_a"] = values["c_z_a"] * (1.0 + level)
    values["c_w_a"] = values["c_w_a"] * (1.0 + DRAG_PER_DOWNFORCE * level)
    return values


def trace_residuals(trajectory, reference) -> np.ndarray:
    """The model's speed against a reference lap's samples, scaled so that the trace counts as SPEED_SAMPLES samples."""
    speed = model_speed_at(trajectory, reference.s, reference.track.total_length)
    return np.sqrt(SPEED_SAMPLES / len(reference.s)) * (speed - reference.speed) / SPEED_SIGMA


def solved_lap(params: Mapping[str, float], reference):
    trajectory = model_lap(params, reference)
    if trajectory.solver_status != "optimal":
        raise RuntimeError(f"{reference.name}: solver {trajectory.solver_status}")
    return trajectory


def round_residuals(params: Mapping[str, float], round_number: int, with_pole: bool) -> np.ndarray:
    """
    The practice lap's speed trace against the model under the practice session's rules, then, for a training
    round, the pole-lap speed trace and lap time as in calibration/fit.py.
    """
    car = round_params(params, round_number)
    practice = practice_lap(round_number)
    parts = [trace_residuals(solved_lap(car, practice), practice)]
    if with_pole:
        reference = reference_lap(round_number)
        trajectory = solved_lap(car, reference)
        parts.append(trace_residuals(trajectory, reference))
        parts.append([(trajectory.lap_time - reference.lap_time) / LAP_SIGMA])
    return np.concatenate(parts)


def _job(args):
    """round_residuals, or None when the solve fails."""
    try:
        return round_residuals(*args)
    except Exception as error:
        print(f"   ! round {args[1]} failed ({error})", flush=True)
        return None


class JointFit:
    """Least squares over the named shared PARAMETERS and one aero level per round."""

    def __init__(self, train: Sequence[int], held_out: Sequence[int], names: Sequence[str], start: Mapping[str, float] = None,
                 workers: int = 8):
        self.train, self.held_out, self.workers = list(train), list(held_out), workers
        self.rounds = self.train + self.held_out
        self.shared = [p for p in PARAMETERS if p.name in names]
        self.fixed = {p.name: p.start for p in PARAMETERS}
        self.fixed.update(start or {})
        self.fixed.update({f"aero_level@{r}": 0.0 for r in self.rounds})
        self.history: List[Dict] = []
        self._cache = {}

    @property
    def size(self) -> int:
        return len(self.shared) + len(self.rounds)

    def params(self, x) -> Dict[str, float]:
        values = dict(self.fixed)
        for p, xi in zip(self.shared, x):
            values[p.name] = p.lower + xi * (p.upper - p.lower)
        lo, hi = AERO_LEVEL_BOUNDS
        for r, xi in zip(self.rounds, x[len(self.shared):]):
            values[f"aero_level@{r}"] = lo + xi * (hi - lo)
        return values

    def x_of(self, values: Mapping[str, float]) -> np.ndarray:
        lo, hi = AERO_LEVEL_BOUNDS
        return np.array([(values[p.name] - p.lower) / (p.upper - p.lower) for p in self.shared]
                        + [(values[f"aero_level@{r}"] - lo) / (hi - lo) for r in self.rounds])

    def _solve(self, jobs):
        with ProcessPoolExecutor(self.workers) as pool:
            return list(pool.map(_job, jobs))

    def _job_list(self, x, rounds):
        values = self.params(x)
        return [(values, r, r in self.train) for r in rounds]

    def residuals(self, x) -> np.ndarray:
        key = tuple(np.round(x, 12))
        if key not in self._cache:
            parts = self._solve(self._job_list(x, self.rounds))
            if any(part is None for part in parts):
                if not self._cache:
                    raise RuntimeError("The start point doesn't solve")
                sizes = {r: len(v) for r, v in next(iter(self._cache.values())).items()}
                parts = [np.full(sizes[r], FAILED) if part is None else part for r, part in zip(self.rounds, parts)]
            self._cache[key] = dict(zip(self.rounds, parts))
            vector = np.concatenate(parts)
            cost = 0.5 * float(vector @ vector)
            values = self.params(x)
            self.history.append({"params": values, "cost": cost})
            print(f"   cost {cost:10.2f}  " + "  ".join(f"{p.name} {values[p.name]:.4f}" for p in self.shared)
                  + "  levels " + " ".join(f"{r}:{values[f'aero_level@{r}']:+.3f}" for r in self.rounds), flush=True)
        return np.concatenate([self._cache[key][r] for r in self.rounds])

    def jacobian(self, x, step: float = 0.01) -> np.ndarray:
        """Forward differences; an aero level only moves its own round, so it costs one solve."""
        base = self.residuals(x)
        by_round = self._cache[tuple(np.round(x, 12))]
        offsets = np.cumsum([0] + [len(by_round[r]) for r in self.rounds])
        steps = [step if xi + step <= 1.0 else -step for xi in x]
        jobs, owners = [], []
        for i, h in enumerate(steps):
            point = x + h * np.eye(len(x))[i]
            rounds = self.rounds if i < len(self.shared) else [self.rounds[i - len(self.shared)]]
            jobs += self._job_list(point, rounds)
            owners += [(i, r) for r in rounds]
        values = self._solve(jobs)
        J = np.zeros((len(base), len(x)))
        for (i, r), value in zip(owners, values):
            if value is None:
                continue                  # Failed solve: that entry stays 0
            k = self.rounds.index(r)
            J[offsets[k]:offsets[k + 1], i] = (value - by_round[r]) / steps[i]
        return J

    def run(self, max_evaluations: int = 15):
        t0 = time.time()
        result = least_squares(self.residuals, self.x_of(self.fixed), jac=self.jacobian, bounds=(0.0, 1.0),
                               method="trf", x_scale=0.05, max_nfev=max_evaluations)
        self.result = result
        print(f"   {result.message} ({result.nfev} evaluations, {time.time() - t0:.0f} s)")
        return self.params(result.x)


def _report_row(args):
    params, round_number = args
    reference = reference_lap(round_number)
    trajectory = model_lap(round_params(params, round_number), reference)
    speed = model_speed_at(trajectory, reference.s, reference.track.total_length)
    return {
        "round": round_number, "name": reference.name, "status": trajectory.solver_status,
        "lap_model": trajectory.lap_time, "lap_real": reference.lap_time,
        "lap_error_pct": 100.0 * (trajectory.lap_time - reference.lap_time) / reference.lap_time,
        "speed_rmse_kmh": 3.6 * float(np.sqrt(np.mean((speed - reference.speed) ** 2))),
        "top_model_kmh": 3.6 * float(trajectory.v_opt.max()), "top_real_kmh": 3.6 * reference.top_speed,
    }


def report(params: Mapping[str, float], rounds: Sequence[int], workers: int = 8) -> List[Dict]:
    """Lap time error, speed RMSE and top speeds per round, each with its own aero level."""
    with ProcessPoolExecutor(workers) as pool:
        return list(pool.map(_report_row, [(dict(params), r) for r in rounds]))
