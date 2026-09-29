"""
Multi-lap spatial NLP solver for ERS strategy optimization.

Extends the single-lap NLP to a full stint horizon with:
- SOC continuity across lap boundaries
- Per-lap deployment/recovery constraints
- Optional per-lap final SOC floor (charge-sustaining races)
- Optional dynamic tire thermal + degradation state evolution

The NLP itself is built by SpatialNLPSolver._build_and_solve, shared with the single-lap solver.
"""

import time
from typing import Literal

import numpy as np

from config import TireCompoundConfig, TireThermalConfig

from .base import OptimalTrajectory
from .spatial_nlp import DynamicTireSettings, SpatialNLPSolver


class MultiLapSpatialNLPSolver(SpatialNLPSolver):
    """Spatial-domain NLP solver over multiple consecutive laps."""

    @property
    def name(self) -> str:
        return "SpatialNLP-MultiLap"

    def solve(
        self,
        v_guess: np.ndarray | None = None,
        n_laps: int = 2,
        initial_soc: float = 0.5,
        final_soc_min: float = 0.3,
        is_flying_lap: bool = True,
        per_lap_final_soc_min: float | None = None,
        lap_grip_scales: np.ndarray | None = None,
        tire_model: Literal["scalar", "dynamic"] = "scalar",
        tire_thermal_config: TireThermalConfig | None = None,
        tire_compound_config: TireCompoundConfig | None = None,
        ambient_temp_c: float = 25.0,
        track_temp_c: float = 35.0,
        tire_init_temp_c: float = 80.0,
    ) -> OptimalTrajectory:
        """
        Solve a multi-lap ERS optimal control problem.

        Args:
            v_guess: Single-lap speed profile for the initial guess (default: forward-backward without ERS).
            n_laps: Number of consecutive laps in the horizon.
            initial_soc: Starting SOC for lap 1.
            final_soc_min: Final SOC lower bound at end of last lap.
            is_flying_lap: If True, enforce V_start == V_end over the full horizon.
            per_lap_final_soc_min: Optional SOC floor at each lap boundary.
            lap_grip_scales: Optional per-lap grip multipliers (shape [n_laps]) for scalar mode.
            tire_model: "scalar" (legacy per-lap scale) or "dynamic" (stateful thermal/wear).

        Raises:
            SolverError: if the NLP solver does not converge
        """
        if n_laps < 1:
            raise ValueError("n_laps must be >= 1")

        dynamic_enabled = tire_model == "dynamic"
        if tire_model not in ("scalar", "dynamic"):
            raise ValueError("tire_model must be one of {'scalar', 'dynamic'}")

        lap_grip_scales_arr: np.ndarray | None = None
        if lap_grip_scales is not None:
            lap_grip_scales_arr = np.asarray(lap_grip_scales, dtype=float).reshape(-1)
            if lap_grip_scales_arr.shape[0] != n_laps:
                raise ValueError("lap_grip_scales must have length n_laps")
            if np.any(lap_grip_scales_arr <= 0.0):
                raise ValueError("lap_grip_scales values must be > 0")

        if dynamic_enabled and lap_grip_scales_arr is not None:
            self._log("Dynamic tire model active: ignoring lap_grip_scales scalar schedule.")
            lap_grip_scales_arr = None

        if n_laps == 1:
            if dynamic_enabled:
                self._log("Dynamic tire model is multi-lap only in this rollout; using single-lap scalar solve.")
            return super().solve(
                v_guess=v_guess,
                initial_soc=initial_soc,
                final_soc_min=final_soc_min,
                is_flying_lap=is_flying_lap,
            )

        self._log(
            f"Setting up multi-lap NLP ({n_laps} laps, {self.N * n_laps} intervals) "
            f"using {self.collocation_method.value} collocation.."
        )
        if dynamic_enabled:
            self._log("Dynamic tire model: enabled")

        start_time = time.time()

        tire = None
        if dynamic_enabled:
            tire = DynamicTireSettings(
                thermal=tire_thermal_config or TireThermalConfig(),
                compound=tire_compound_config or TireCompoundConfig(),
                ambient_temp_c=ambient_temp_c,
                track_temp_c=track_temp_c,
                init_temp_c=tire_init_temp_c,
            )

        try:
            trajectory = self._build_and_solve(
                v_guess=self._guess_on_grid(v_guess, is_flying_lap),
                initial_soc=initial_soc,
                final_soc_min=final_soc_min,
                is_flying_lap=is_flying_lap,
                n_laps=n_laps,
                per_lap_final_soc_min=per_lap_final_soc_min,
                lap_grip_scales=lap_grip_scales_arr,
                tire=tire,
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
