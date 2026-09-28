"""
Multi-lap spatial NLP solver for ERS strategy optimization.

Extends the single-lap NLP to a full stint horizon with:
- SOC continuity across lap boundaries
- Per-lap deployment/recovery constraints
- Optional per-lap final SOC floor (charge-sustaining races)

The NLP itself is built by SpatialNLPSolver._build_and_solve, shared with the single-lap solver.
"""

import time

import numpy as np

from .base import OptimalTrajectory
from .spatial_nlp import SpatialNLPSolver


class MultiLapSpatialNLPSolver(SpatialNLPSolver):
    """Spatial-domain NLP solver over multiple consecutive laps."""

    @property
    def name(self) -> str:
        return "SpatialNLP-MultiLap"

    def solve(
        self,
        v_limit_profile: np.ndarray,
        n_laps: int = 2,
        initial_soc: float = 0.5,
        final_soc_min: float = 0.3,
        is_flying_lap: bool = True,
        per_lap_final_soc_min: float | None = None,
        lap_grip_scales: np.ndarray | None = None,
    ) -> OptimalTrajectory:
        """
        Solve a multi-lap ERS optimal control problem.

        Args:
            v_limit_profile: Single-lap velocity limit profile from forward-backward solver.
            n_laps: Number of consecutive laps in the horizon.
            initial_soc: Starting SOC for lap 1.
            final_soc_min: Final SOC lower bound at end of last lap.
            is_flying_lap: If True, enforce V_start == V_end over the full horizon.
            per_lap_final_soc_min: Optional SOC floor at each lap boundary.
            lap_grip_scales: Optional per-lap grip multipliers (shape [n_laps]).

        Raises:
            SolverError: if the NLP solver does not converge
        """
        if n_laps < 1:
            raise ValueError("n_laps must be >= 1")
        lap_grip_scales_arr: np.ndarray | None = None
        if lap_grip_scales is not None:
            lap_grip_scales_arr = np.asarray(lap_grip_scales, dtype=float).reshape(-1)
            if lap_grip_scales_arr.shape[0] != n_laps:
                raise ValueError("lap_grip_scales must have length n_laps")
            if np.any(lap_grip_scales_arr <= 0.0):
                raise ValueError("lap_grip_scales values must be > 0")
        if n_laps == 1:
            return super().solve(
                v_limit_profile=v_limit_profile,
                initial_soc=initial_soc,
                final_soc_min=final_soc_min,
                is_flying_lap=is_flying_lap,
            )

        self._log(
            f"Setting up multi-lap NLP ({n_laps} laps, {self.N * n_laps} intervals) "
            f"using {self.collocation_method.value} collocation.."
        )
        start_time = time.time()

        try:
            trajectory = self._build_and_solve(
                v_limit_profile=self._sample_on_grid(v_limit_profile),
                initial_soc=initial_soc,
                final_soc_min=final_soc_min,
                is_flying_lap=is_flying_lap,
                n_laps=n_laps,
                per_lap_final_soc_min=per_lap_final_soc_min,
                lap_grip_scales=lap_grip_scales_arr,
            )
            trajectory.solve_time = time.time() - start_time

            self._log(f"✓ Solved in {trajectory.solve_time:.2f}s")
            self._log(
                f"  Total time: {trajectory.lap_time:.3f}s "
                f"(avg {trajectory.lap_time / n_laps:.3f}s/lap)"
            )
            for i in range(n_laps):
                self._log(
                    f"  Lap {i + 1}: {trajectory.lap_times[i]:.3f}s | "
                    f"SOC {trajectory.lap_start_soc[i] * 100:.1f}% -> "
                    f"{trajectory.lap_end_soc[i] * 100:.1f}%"
                )
            self._log(f"  Status: {trajectory.solver_status}")
            return trajectory

        except RuntimeError as e:
            self._log(f"❌ Optimization failed: {e}")
            raise
