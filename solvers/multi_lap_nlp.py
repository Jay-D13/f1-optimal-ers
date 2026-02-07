"""
Multi-lap spatial NLP solver for ERS strategy optimization.

Extends the single-lap NLP to a full stint horizon with:
- SOC continuity across lap boundaries
- Per-lap deployment/recovery constraints
- Optional per-lap final SOC floor (charge-sustaining races)
"""

import time

import casadi as ca
import numpy as np

from .base import OptimalTrajectory
from .spatial_nlp import CollocationMethod, SpatialNLPSolver


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
        """
        if n_laps < 1:
            raise ValueError("n_laps must be >= 1")
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

        if len(v_limit_profile) != self.N + 1:
            v_limit_profile = np.interp(
                self.s_grid,
                np.linspace(0, self.track.total_length, len(v_limit_profile)),
                v_limit_profile,
            )

        try:
            trajectory = self._build_and_solve_multi_lap(
                v_limit_profile=v_limit_profile,
                n_laps=n_laps,
                initial_soc=initial_soc,
                final_soc_min=final_soc_min,
                is_flying_lap=is_flying_lap,
                per_lap_final_soc_min=per_lap_final_soc_min,
            )
            trajectory.solve_time = time.time() - start_time

            self._log(f"✓ Solved in {trajectory.solve_time:.2f}s")
            self._log(
                f"  Total time: {trajectory.lap_time:.3f}s "
                f"(avg {trajectory.lap_time / n_laps:.3f}s/lap)"
            )
            if trajectory.lap_times is not None:
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

    def _build_and_solve_multi_lap(
        self,
        v_limit_profile: np.ndarray,
        n_laps: int,
        initial_soc: float,
        final_soc_min: float,
        is_flying_lap: bool,
        per_lap_final_soc_min: float | None,
    ) -> OptimalTrajectory:
        """Build and solve the multi-lap CasADi optimization problem."""
        opti = ca.Opti()

        veh = self.vehicle.vehicle
        ers = self.vehicle.ers

        track_data = self.track.track_data
        gradient_arr = np.resize(track_data.gradient, self.N + 1)
        radius_arr = np.resize(track_data.radius, self.N + 1)

        n_intervals_total = self.N * n_laps
        s_grid_total = np.linspace(
            0.0,
            self.track.total_length * n_laps,
            n_intervals_total + 1,
        )

        # =================================================================
        # DECISION VARIABLES
        # =================================================================
        V = opti.variable(n_intervals_total + 1)
        SOC = opti.variable(n_intervals_total + 1)

        P_DEPLOY = opti.variable(n_intervals_total)
        P_HARVEST = opti.variable(n_intervals_total)
        THROTTLE = opti.variable(n_intervals_total)
        BRAKE = opti.variable(n_intervals_total)

        if self.collocation_method == CollocationMethod.HERMITE_SIMPSON:
            V_MID = opti.variable(n_intervals_total)
            SOC_MID = opti.variable(n_intervals_total)

        # =================================================================
        # OBJECTIVE
        # =================================================================
        T_total = 0
        lap_deployment = [ca.MX(0) for _ in range(n_laps)]
        lap_recovery = [ca.MX(0) for _ in range(n_laps)]

        for i in range(n_intervals_total):
            lap_idx = i // self.N
            k = i % self.N

            if self.collocation_method == CollocationMethod.HERMITE_SIMPSON:
                v_k_safe = ca.fmax(V[i], 1.0)
                v_mid_safe = ca.fmax(V_MID[i], 1.0)
                v_k1_safe = ca.fmax(V[i + 1], 1.0)
                T_total += (self.ds / 6.0) * (1.0 / v_k_safe + 4.0 / v_mid_safe + 1.0 / v_k1_safe)
            else:
                v_avg = 0.5 * (V[i] + V[i + 1])
                v_safe = ca.fmax(v_avg, 1.0)
                T_total += self.ds / v_safe

            # --- 2026 Regulation Logic (Speed Dependent Taper) ---
            self._apply_ers_power_limits(opti, P_DEPLOY[i], V[i], ers)
            opti.subject_to(P_HARVEST[i] <= ers.max_recovery_power)
            opti.subject_to(P_DEPLOY[i] >= 0)
            opti.subject_to(P_HARVEST[i] >= 0)

            # --- Derivatives at node k ---
            dv_ds_k, dsoc_ds_k, F_prop_k, F_brake_k, F_grip_k = self._compute_derivatives(
                V[i],
                SOC[i],
                P_DEPLOY[i],
                P_HARVEST[i],
                THROTTLE[i],
                BRAKE[i],
                gradient_arr[k],
                radius_arr[k],
                veh,
                ers,
            )

            opti.subject_to(F_prop_k - F_brake_k <= F_grip_k)
            opti.subject_to(F_prop_k - F_brake_k >= -F_grip_k)

            if self.collocation_method == CollocationMethod.EULER:
                opti.subject_to(V[i + 1] == V[i] + self.ds * dv_ds_k)
                opti.subject_to(SOC[i + 1] == SOC[i] + self.ds * dsoc_ds_k)

            elif self.collocation_method == CollocationMethod.TRAPEZOIDAL:
                dv_ds_k1, dsoc_ds_k1, F_prop_k1, F_brake_k1, F_grip_k1 = self._compute_derivatives(
                    V[i + 1],
                    SOC[i + 1],
                    P_DEPLOY[i],
                    P_HARVEST[i],
                    THROTTLE[i],
                    BRAKE[i],
                    gradient_arr[k + 1],
                    radius_arr[k + 1],
                    veh,
                    ers,
                )

                if i == n_intervals_total - 1:
                    opti.subject_to(F_prop_k1 - F_brake_k1 <= F_grip_k1)
                    opti.subject_to(F_prop_k1 - F_brake_k1 >= -F_grip_k1)

                opti.subject_to(V[i + 1] == V[i] + (self.ds / 2.0) * (dv_ds_k + dv_ds_k1))
                opti.subject_to(SOC[i + 1] == SOC[i] + (self.ds / 2.0) * (dsoc_ds_k + dsoc_ds_k1))

            elif self.collocation_method == CollocationMethod.HERMITE_SIMPSON:
                dv_ds_k1, dsoc_ds_k1, F_prop_k1, F_brake_k1, F_grip_k1 = self._compute_derivatives(
                    V[i + 1],
                    SOC[i + 1],
                    P_DEPLOY[i],
                    P_HARVEST[i],
                    THROTTLE[i],
                    BRAKE[i],
                    gradient_arr[k + 1],
                    radius_arr[k + 1],
                    veh,
                    ers,
                )

                if i == n_intervals_total - 1:
                    opti.subject_to(F_prop_k1 - F_brake_k1 <= F_grip_k1)
                    opti.subject_to(F_prop_k1 - F_brake_k1 >= -F_grip_k1)

                v_mid_hermite = 0.5 * (V[i] + V[i + 1]) + (self.ds / 8.0) * (dv_ds_k - dv_ds_k1)
                soc_mid_hermite = 0.5 * (SOC[i] + SOC[i + 1]) + (self.ds / 8.0) * (dsoc_ds_k - dsoc_ds_k1)

                opti.subject_to(V_MID[i] == v_mid_hermite)
                opti.subject_to(SOC_MID[i] == soc_mid_hermite)

                grad_mid = 0.5 * (gradient_arr[k] + gradient_arr[k + 1])
                radius_mid = 0.5 * (radius_arr[k] + radius_arr[k + 1])

                dv_ds_mid, dsoc_ds_mid, F_prop_mid, F_brake_mid, F_grip_mid = self._compute_derivatives(
                    V_MID[i],
                    SOC_MID[i],
                    P_DEPLOY[i],
                    P_HARVEST[i],
                    THROTTLE[i],
                    BRAKE[i],
                    grad_mid,
                    radius_mid,
                    veh,
                    ers,
                )

                opti.subject_to(F_prop_mid - F_brake_mid <= F_grip_mid)
                opti.subject_to(F_prop_mid - F_brake_mid >= -F_grip_mid)

                opti.subject_to(V[i + 1] == V[i] + (self.ds / 6.0) * (dv_ds_k + 4.0 * dv_ds_mid + dv_ds_k1))
                opti.subject_to(
                    SOC[i + 1] == SOC[i] + (self.ds / 6.0) * (dsoc_ds_k + 4.0 * dsoc_ds_mid + dsoc_ds_k1)
                )

            dt_step = self.ds / ca.fmax(V[i], 5.0)
            lap_deployment[lap_idx] += P_DEPLOY[i] * dt_step
            lap_recovery[lap_idx] += P_HARVEST[i] * dt_step

            opti.subject_to(THROTTLE[i] * BRAKE[i] <= 0.01)

        opti.minimize(T_total)

        # =================================================================
        # CONSTRAINTS
        # =================================================================
        opti.subject_to(SOC[0] == initial_soc)
        opti.subject_to(SOC[-1] >= final_soc_min)

        if per_lap_final_soc_min is not None:
            for lap_idx in range(n_laps):
                lap_end_idx = (lap_idx + 1) * self.N
                opti.subject_to(SOC[lap_end_idx] >= per_lap_final_soc_min)

        for lap_idx in range(n_laps):
            opti.subject_to(lap_deployment[lap_idx] <= ers.deployment_limit_per_lap)
            opti.subject_to(lap_recovery[lap_idx] <= ers.recovery_limit_per_lap)

        v_limit_nodes = np.concatenate([np.tile(v_limit_profile[:-1], n_laps), [v_limit_profile[-1]]])

        opti.subject_to(opti.bounded(ers.min_soc, SOC, ers.max_soc))
        opti.subject_to(opti.bounded(5.0, V, v_limit_nodes * 1.02))
        opti.subject_to(opti.bounded(0, THROTTLE, 1))
        opti.subject_to(opti.bounded(0, BRAKE, 1))

        if is_flying_lap:
            opti.subject_to(V[0] == V[-1])
        else:
            opti.subject_to(V[0] == v_limit_profile[0])

        if self.collocation_method == CollocationMethod.HERMITE_SIMPSON:
            v_limit_mid_single = 0.5 * (v_limit_profile[:-1] + v_limit_profile[1:])
            v_limit_mid = np.tile(v_limit_mid_single, n_laps)
            opti.subject_to(opti.bounded(5.0, V_MID, v_limit_mid * 1.02))
            opti.subject_to(opti.bounded(ers.min_soc, SOC_MID, ers.max_soc))

        # =================================================================
        # SOLVE
        # =================================================================
        self._configure_solver(opti)

        v_guess = np.concatenate([np.tile(v_limit_profile[:-1], n_laps), [v_limit_profile[-1]]]) * 0.95
        soc_target = max(final_soc_min, per_lap_final_soc_min or ers.min_soc)
        soc_guess = np.linspace(initial_soc, soc_target, n_intervals_total + 1)
        opti.set_initial(V, v_guess)
        opti.set_initial(SOC, soc_guess)
        opti.set_initial(THROTTLE, np.ones(n_intervals_total) * 0.8)
        opti.set_initial(BRAKE, np.zeros(n_intervals_total))
        opti.set_initial(P_DEPLOY, np.zeros(n_intervals_total))
        opti.set_initial(P_HARVEST, np.zeros(n_intervals_total))

        if self.collocation_method == CollocationMethod.HERMITE_SIMPSON:
            v_mid_guess = np.tile(0.5 * (v_limit_profile[:-1] + v_limit_profile[1:]), n_laps) * 0.95
            soc_mid_guess = 0.5 * (soc_guess[:-1] + soc_guess[1:])
            opti.set_initial(V_MID, v_mid_guess)
            opti.set_initial(SOC_MID, soc_mid_guess)

        try:
            sol = opti.solve()
            status = "optimal"
        except Exception as e:
            print(f"Solver Warning: {e}")
            status = "suboptimal"
            sol = opti.debug

        # =================================================================
        # EXTRACT RESULTS
        # =================================================================
        v_opt = sol.value(V)
        soc_opt = sol.value(SOC)
        p_deploy_opt = sol.value(P_DEPLOY)
        p_harvest_opt = sol.value(P_HARVEST)
        P_ers_opt = p_deploy_opt - p_harvest_opt

        throttle_opt = sol.value(THROTTLE)
        brake_opt = sol.value(BRAKE)

        t_opt = np.zeros_like(v_opt)
        for i in range(1, len(t_opt)):
            ds_step = s_grid_total[i] - s_grid_total[i - 1]
            v_avg = 0.5 * (v_opt[i] + v_opt[i - 1])
            t_opt[i] = t_opt[i - 1] + ds_step / max(v_avg, 1.0)

        lap_times = np.zeros(n_laps)
        lap_start_soc = np.zeros(n_laps)
        lap_end_soc = np.zeros(n_laps)
        for lap_idx in range(n_laps):
            start = lap_idx * self.N
            end = (lap_idx + 1) * self.N
            lap_times[lap_idx] = t_opt[end] - t_opt[start]
            lap_start_soc[lap_idx] = soc_opt[start]
            lap_end_soc[lap_idx] = soc_opt[end]

        lap_energy_deployed = np.array([float(sol.value(expr)) for expr in lap_deployment])
        lap_energy_recovered = np.array([float(sol.value(expr)) for expr in lap_recovery])

        return OptimalTrajectory(
            s=s_grid_total,
            ds=self.ds,
            n_points=n_intervals_total + 1,
            v_opt=v_opt,
            soc_opt=soc_opt,
            P_ers_opt=P_ers_opt,
            throttle_opt=throttle_opt,
            brake_opt=brake_opt,
            t_opt=t_opt,
            lap_time=t_opt[-1],
            energy_deployed=float(np.sum(lap_energy_deployed)),
            energy_recovered=float(np.sum(lap_energy_recovered)),
            solve_time=0.0,
            solver_status=status,
            solver_name=f"{self.name}({self._resolved_nlp_solver})",
            n_laps=n_laps,
            lap_length=self.track.total_length,
            lap_times=lap_times,
            lap_energy_deployed=lap_energy_deployed,
            lap_energy_recovered=lap_energy_recovered,
            lap_start_soc=lap_start_soc,
            lap_end_soc=lap_end_soc,
        )
