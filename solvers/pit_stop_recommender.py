"""One-stop pit recommendation over a fixed multi-lap horizon."""

from dataclasses import dataclass

import numpy as np

from utils.tire_degradation import build_lap_grip_scales

from .base import OptimalTrajectory
from .multi_lap_nlp import MultiLapSpatialNLPSolver


@dataclass
class PitStrategyCandidate:
    pit_lap_end: int | None
    lap_grip_scales: np.ndarray
    driving_time_s: float
    pit_loss_s: float
    total_time_s: float
    solver_status: str
    trajectory: OptimalTrajectory
    delta_vs_no_stop_s: float = 0.0

    def to_dict(self) -> dict:
        """Serialize candidate without trajectory payload."""
        return {
            "pit_lap_end": self.pit_lap_end,
            "lap_grip_scales": self.lap_grip_scales,
            "driving_time_s": self.driving_time_s,
            "pit_loss_s": self.pit_loss_s,
            "total_time_s": self.total_time_s,
            "delta_vs_no_stop_s": self.delta_vs_no_stop_s,
            "solver_status": self.solver_status,
        }


@dataclass
class PitRecommendationResult:
    best_candidate: PitStrategyCandidate
    no_stop_candidate: PitStrategyCandidate
    candidates_ranked: list[PitStrategyCandidate]
    pit_loss_time_s: float
    pit_window_start_lap: int
    pit_window_end_lap: int

    def to_dict(self) -> dict:
        return {
            "pit_loss_time_s": self.pit_loss_time_s,
            "pit_window_start_lap": self.pit_window_start_lap,
            "pit_window_end_lap": self.pit_window_end_lap,
            "best_candidate": self.best_candidate.to_dict(),
            "no_stop_candidate": self.no_stop_candidate.to_dict(),
            "candidates_ranked": [candidate.to_dict() for candidate in self.candidates_ranked],
        }


def _resolve_pit_window(
    n_laps: int,
    pit_window_start_lap: int,
    pit_window_end_lap: int | None,
) -> tuple[int, int]:
    """Resolve and clamp pit window to valid one-stop lap range."""
    if n_laps < 2:
        return (1, 0)  # empty window

    start = max(1, pit_window_start_lap)
    default_end = n_laps - 3
    end = default_end if pit_window_end_lap is None else pit_window_end_lap
    end = min(n_laps - 1, end)
    return (start, end)


def recommend_one_stop_pit(
    solver: MultiLapSpatialNLPSolver,
    v_limit_profile: np.ndarray,
    n_laps: int,
    initial_soc: float,
    final_soc_min: float,
    is_flying_lap: bool,
    per_lap_final_soc_min: float | None,
    wear_rate_per_lap: float,
    min_grip_scale: float,
    pit_loss_time_s: float,
    pit_window_start_lap: int,
    pit_window_end_lap: int | None,
    pit_eval_step_lap: int,
) -> PitRecommendationResult:
    """
    Enumerate no-stop and one-stop candidates and select minimum total race time.
    """
    if n_laps < 2:
        raise ValueError("n_laps must be >= 2 for pit-stop recommendation")
    if pit_eval_step_lap < 1:
        raise ValueError("pit_eval_step_lap must be >= 1")
    if pit_loss_time_s < 0.0:
        raise ValueError("pit_loss_time_s must be >= 0")

    start_lap, end_lap = _resolve_pit_window(
        n_laps=n_laps,
        pit_window_start_lap=pit_window_start_lap,
        pit_window_end_lap=pit_window_end_lap,
    )

    def evaluate_candidate(pit_lap_end: int | None) -> PitStrategyCandidate:
        lap_grip_scales = build_lap_grip_scales(
            n_laps=n_laps,
            wear_rate=wear_rate_per_lap,
            min_scale=min_grip_scale,
            pit_lap_end=pit_lap_end,
        )

        trajectory = solver.solve(
            v_limit_profile=v_limit_profile,
            n_laps=n_laps,
            initial_soc=initial_soc,
            final_soc_min=final_soc_min,
            is_flying_lap=is_flying_lap,
            per_lap_final_soc_min=per_lap_final_soc_min,
            lap_grip_scales=lap_grip_scales,
        )

        pit_loss = pit_loss_time_s if pit_lap_end is not None else 0.0
        total_time = trajectory.lap_time + pit_loss

        return PitStrategyCandidate(
            pit_lap_end=pit_lap_end,
            lap_grip_scales=lap_grip_scales,
            driving_time_s=float(trajectory.lap_time),
            pit_loss_s=float(pit_loss),
            total_time_s=float(total_time),
            solver_status=trajectory.solver_status,
            trajectory=trajectory,
        )

    candidates: list[PitStrategyCandidate] = [evaluate_candidate(None)]

    if start_lap <= end_lap:
        for pit_lap in range(start_lap, end_lap + 1, pit_eval_step_lap):
            candidates.append(evaluate_candidate(pit_lap))

    no_stop = candidates[0]
    for candidate in candidates:
        candidate.delta_vs_no_stop_s = float(candidate.total_time_s - no_stop.total_time_s)

    ranked = sorted(candidates, key=lambda c: c.total_time_s)
    best = ranked[0]

    return PitRecommendationResult(
        best_candidate=best,
        no_stop_candidate=no_stop,
        candidates_ranked=ranked,
        pit_loss_time_s=pit_loss_time_s,
        pit_window_start_lap=start_lap,
        pit_window_end_lap=end_lap,
    )
