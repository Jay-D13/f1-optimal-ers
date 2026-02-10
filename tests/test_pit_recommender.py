import unittest

import numpy as np

from solvers.pit_stop_recommender import recommend_one_stop_pit


class _DummyTrajectory:
    def __init__(self, lap_time: float):
        self.lap_time = lap_time
        self.solver_status = "optimal"


class _FakeMultiLapSolver:
    """
    Fast deterministic surrogate for pit recommendation tests.
    """

    def __init__(self, base_lap_time_s: float = 80.0, wear_penalty_s: float = 12.0):
        self.base_lap_time_s = base_lap_time_s
        self.wear_penalty_s = wear_penalty_s

    def solve(
        self,
        v_limit_profile: np.ndarray,
        n_laps: int,
        initial_soc: float,
        final_soc_min: float,
        is_flying_lap: bool,
        per_lap_final_soc_min: float | None,
        lap_grip_scales: np.ndarray | None = None,
    ):
        if lap_grip_scales is None:
            lap_grip_scales = np.ones(n_laps, dtype=float)
        wear_penalty = self.wear_penalty_s * float(np.sum(1.0 - lap_grip_scales))
        lap_time = self.base_lap_time_s * n_laps + wear_penalty
        return _DummyTrajectory(lap_time=lap_time)


class PitRecommendationTests(unittest.TestCase):
    def setUp(self):
        self.v_limit = np.ones(10, dtype=float) * 80.0

    def test_high_wear_low_pit_loss_prefers_one_stop(self):
        solver = _FakeMultiLapSolver(base_lap_time_s=80.0, wear_penalty_s=22.0)
        result = recommend_one_stop_pit(
            solver=solver,  # type: ignore[arg-type]
            v_limit_profile=self.v_limit,
            n_laps=10,
            initial_soc=0.55,
            final_soc_min=0.45,
            is_flying_lap=True,
            per_lap_final_soc_min=0.40,
            wear_rate_per_lap=0.03,
            min_grip_scale=0.85,
            pit_loss_time_s=5.0,
            pit_window_start_lap=3,
            pit_window_end_lap=None,
            pit_eval_step_lap=1,
        )

        self.assertIsNotNone(result.best_candidate)
        self.assertIsNotNone(result.no_stop_candidate)
        self.assertIsNotNone(result.best_candidate.pit_lap_end)

    def test_low_wear_high_pit_loss_prefers_no_stop(self):
        solver = _FakeMultiLapSolver(base_lap_time_s=80.0, wear_penalty_s=10.0)
        result = recommend_one_stop_pit(
            solver=solver,  # type: ignore[arg-type]
            v_limit_profile=self.v_limit,
            n_laps=10,
            initial_soc=0.55,
            final_soc_min=0.45,
            is_flying_lap=True,
            per_lap_final_soc_min=0.40,
            wear_rate_per_lap=0.01,
            min_grip_scale=0.90,
            pit_loss_time_s=35.0,
            pit_window_start_lap=3,
            pit_window_end_lap=None,
            pit_eval_step_lap=1,
        )

        self.assertIsNone(result.best_candidate.pit_lap_end)
        self.assertAlmostEqual(result.best_candidate.delta_vs_no_stop_s, 0.0, places=9)


if __name__ == "__main__":
    unittest.main()
