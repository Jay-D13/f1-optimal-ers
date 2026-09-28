"""
Spatial-Domain NLP Solver for ERS Optimization

Offline optimizer using direct collocation in spatial domain.

Different integration schemes:
- Euler (1st order) - Fast, less accurate
- Trapezoidal (2nd order) - Good balance
- Hermite-Simpson (4th order) - High accuracy

Problem Formulation:
    minimize    T = ∑(ds / v[k])              (lap time)
    subject to:
        v[k] ≤ v_limit[k]                     (grip limit from Forward-Backward)
        v dynamics from ERS + ICE power
        SOC dynamics from ERS power
        E_deploy, E_recover ≤ per-lap limits  (regulatory limits; running totals are states)
        SOC_min ≤ SOC ≤ SOC_max               (battery limits)
"""

import time
from enum import Enum
from typing import Literal, Tuple

import casadi as ca
import numpy as np

from solvers import BaseSolver, OptimalTrajectory, SolverError

# Ipopt return statuses that count as a solution
_SUCCESS_STATUS = {"Solve_Succeeded": "optimal", "Solved_To_Acceptable_Level": "acceptable"}


def deploy_power_limit(v, ers):
    """
    Maximum MGU-K deployment power (W) at speed v (m/s). Works on floats and CasADi expressions.

    2026: 350 kW up to 290 km/h, then 1800 - 5v kW, then 6900 - 20v kW from 340 km/h,
    reaching 0 at 345 km/h (v in km/h). Earlier years: a flat limit.
    """
    if ers.regulation_year < 2026:
        return ers.max_deployment_power

    v_kph = v * 3.6
    p_taper1 = (1800.0 - 5.0 * v_kph) * 1000.0    # Linear drop 290-340 km/h
    p_taper2 = (6900.0 - 20.0 * v_kph) * 1000.0   # Sharp drop 340-345 km/h
    return ca.fmax(0, ca.fmin(ers.max_deployment_power, ca.fmin(p_taper1, p_taper2)))


class CollocationMethod(Enum):
    """Available collocation/integration methods."""
    EULER = "euler"                    # 1st order - explicit Euler
    TRAPEZOIDAL = "trapezoidal"        # 2nd order - implicit trapezoidal
    HERMITE_SIMPSON = "hermite_simpson" # 4th order - Hermite-Simpson


class SpatialNLPSolver(BaseSolver):
    """
    Finds the globally (offline) optimal ERS deployment strategy for a single lap.
    Optimizes lap time subject to energy budget and thermal/grip limits.
    """

    # ERS powers are solved in units of 100 kW so that every variable is O(1)
    POWER_SCALE = 1e5

    def __init__(
        self,
        vehicle_model,
        track_model,
        ers_config,
        ds: float = 5.0,
        collocation_method: Literal["euler", "trapezoidal", "hermite_simpson"] = "euler",
        nlp_solver: Literal["auto", "ipopt", "fatrop", "sqpmethod"] = "auto",
        ipopt_linear_solver: str = "mumps",
        ipopt_hessian_approximation: Literal["limited-memory", "exact"] = "exact",
    ):
        super().__init__(vehicle_model, track_model, ers_config)

        self.collocation_method = CollocationMethod(collocation_method)
        self.nlp_solver = nlp_solver
        self.ipopt_linear_solver = ipopt_linear_solver
        self.ipopt_hessian_approximation = ipopt_hessian_approximation
        self._resolved_nlp_solver = self._resolve_nlp_solver()

        # Discretization: N equal steps covering exactly one lap (ds is stretched slightly to fit).
        # Rounding down keeps one node per track point, which the comparison plots rely on.
        self.N = max(1, int(track_model.total_length / ds))
        self.ds = track_model.total_length / self.N
        self.s_grid = np.linspace(0, track_model.total_length, self.N + 1)

    @property
    def name(self) -> str:
        return "SpatialNLP"

    def _resolve_nlp_solver(self) -> Literal["ipopt", "fatrop", "sqpmethod"]:
        """Resolve auto solver mode to a concrete backend: Ipopt everywhere, fatrop and sqpmethod are opt-in."""
        if self.nlp_solver == "auto":
            return "ipopt"
        return self.nlp_solver

    def solve(
        self,
        v_limit_profile: np.ndarray,
        initial_soc: float = 0.5,
        final_soc_min: float = 0.3,
        is_flying_lap: bool = True
    ) -> OptimalTrajectory:
        """
        Solve the optimal control problem for ERS deployment.

        Args:
            v_limit_profile: Maximum velocity profile from grip limits
            initial_soc: Starting state of charge (0-1)
            final_soc_min: Minimum final state of charge
            is_flying_lap: If True, enforce V[0] == V[-1]

        Returns:
            OptimalTrajectory containing solution

        Raises:
            SolverError: if the NLP solver does not converge
        """
        self._log(f"Setting up NLP with {self.N} nodes using {self.collocation_method.value} collocation..")

        if self.nlp_solver == "auto":
            self._log(f"Auto-selected NLP backend: {self._resolved_nlp_solver}")
        start_time = time.time()

        # Build and solve NLP
        try:
            trajectory = self._build_and_solve(
                v_limit_profile=self._sample_on_grid(v_limit_profile),
                initial_soc=initial_soc,
                final_soc_min=final_soc_min,
                is_flying_lap=is_flying_lap
            )
            trajectory.solve_time = time.time() - start_time

            self._log(f"✓ Solved in {trajectory.solve_time:.2f}s")
            self._log(f"  Lap time: {trajectory.lap_time:.3f}s")
            self._log(f"  Status: {trajectory.solver_status}")
            return trajectory

        except RuntimeError as e:
            self._log(f"❌ Optimization failed: {e}")
            raise

    def _sample_on_grid(self, values: np.ndarray) -> np.ndarray:
        """Sample a per-lap profile at the NLP nodes. It is given on the track's points, or evenly spaced over the lap."""
        values = np.asarray(values, dtype=float)
        track_data = self.track.track_data
        if len(values) == track_data.n_points:
            # Track points stop short of the finish line, so wrap around the lap
            return np.interp(self.s_grid, track_data.s, values, period=self.track.total_length)
        return np.interp(self.s_grid, np.linspace(0, self.track.total_length, len(values)), values)

    def _compute_derivatives(
        self,
        v, soc,
        P_deploy, P_harvest, throttle, brake,
        gradient, radius,
        veh, ers
    ) -> Tuple[ca.MX, ca.MX, ca.MX, ca.MX, ca.MX]:
        """
        Returns: (dv_ds, dsoc_ds, F_prop, F_brake, F_grip) - derivatives and forces for constraint checking
        """
        grad = np.clip(gradient, -0.2, 0.2) if isinstance(gradient, (int, float)) else gradient
        v_safe = ca.fmax(v, 5.0)

        # --- Forces ---
        c_w_a_eff = self._get_effective_drag_coefficient(veh, ers, radius)
        F_drag = 0.5 * veh.rho_air * v**2 * c_w_a_eff
        F_roll = veh.mass * veh.g * veh.cr
        F_grav = veh.mass * veh.g * ca.sin(grad)

        # Propulsion
        P_net_ers = P_deploy - P_harvest
        P_ice = throttle * veh.pow_max_ice
        F_prop = (P_ice + P_net_ers) / v_safe

        # Braking
        F_brake = brake * veh.max_brake_force

        # Net Force
        F_net = F_prop - F_brake - F_drag - F_roll - F_grav

        # Grip limit
        F_norm = veh.mass * veh.g * ca.cos(grad) + 0.5 * veh.rho_air * v**2 * (
            veh.c_z_a_f + veh.c_z_a_r
        )
        F_grip = veh.mu_longitudinal * F_norm

        # State derivatives (spatial domain)
        dv_ds = F_net / (veh.mass * v_safe)

        # Battery dynamics
        P_bat_out = (P_deploy / ers.deployment_efficiency) - (P_harvest * ers.recovery_efficiency)
        dsoc_ds = -P_bat_out / (ers.battery_capacity * v_safe)

        return dv_ds, dsoc_ds, F_prop, F_brake, F_grip

    @staticmethod
    def _add_grip_limit(opti, F_prop, F_brake, F_grip, grip_scale: float = 1.0):
        """Friction limit on the net longitudinal force."""
        opti.subject_to(F_prop - F_brake <= F_grip * grip_scale)
        opti.subject_to(F_prop - F_brake >= -F_grip * grip_scale)

    def _build_and_solve(
        self,
        v_limit_profile: np.ndarray,
        initial_soc: float,
        final_soc_min: float,
        is_flying_lap: bool,
        n_laps: int = 1,
        per_lap_final_soc_min: float | None = None,
        lap_grip_scales: np.ndarray | None = None,
    ) -> OptimalTrajectory:
        """Build and solve the CasADi optimization problem over n_laps consecutive laps."""
        opti = ca.Opti()

        # Get vehicle parameters
        veh = self.vehicle.vehicle
        ers = self.vehicle.ers
        PS = self.POWER_SCALE
        method = self.collocation_method

        # Track data arrays
        gradient_arr = self._sample_on_grid(self.track.track_data.gradient)
        radius_arr = self._sample_on_grid(self.track.track_data.radius)

        degradation_enabled = lap_grip_scales is not None
        if lap_grip_scales is None:
            lap_grip_scales = np.ones(n_laps)

        n = self.N * n_laps  # Intervals over the whole horizon

        # =================================================================
        # DECISION VARIABLES
        # =================================================================

        # States at node points
        V = opti.variable(n + 1)          # Velocity (m/s)
        SOC = opti.variable(n + 1)        # State of Charge (0-1)
        E_DEPLOY = opti.variable(n + 1)   # ERS energy deployed since the start (MJ)
        E_RECOVER = opti.variable(n + 1)  # ERS energy recovered since the start (MJ)

        # Controls (piecewise constant over intervals)
        P_DEPLOY = opti.variable(n)       # ERS discharge power (≥0, units of POWER_SCALE)
        P_HARVEST = opti.variable(n)      # ERS recovery power (≥0, units of POWER_SCALE)
        THROTTLE = opti.variable(n)       # Throttle position (0-1)
        BRAKE = opti.variable(n)          # Brake position (0-1)

        # For Hermite-Simpson: midpoint states
        if method == CollocationMethod.HERMITE_SIMPSON:
            V_MID = opti.variable(n)      # Velocity at midpoints
            SOC_MID = opti.variable(n)    # SOC at midpoints

        # =================================================================
        # OBJECTIVE, DYNAMICS & PHYSICS
        # =================================================================

        T_total = 0

        for i in range(n):
            k = i % self.N  # Node index within the lap
            grip_scale = float(lap_grip_scales[i // self.N])
            p_deploy = P_DEPLOY[i] * PS
            p_harvest = P_HARVEST[i] * PS

            # --- Lap time ---
            if method == CollocationMethod.HERMITE_SIMPSON:
                # Simpson's rule for integrating 1/v:
                # T = integral of (1/v) ds ≈ (ds/6) * (1/v_k + 4/v_mid + 1/v_{k+1})
                v_k_safe = ca.fmax(V[i], 1.0)
                v_mid_safe = ca.fmax(V_MID[i], 1.0)
                v_k1_safe = ca.fmax(V[i + 1], 1.0)
                T_total += (self.ds / 6.0) * (1.0 / v_k_safe + 4.0 / v_mid_safe + 1.0 / v_k1_safe)
            else:
                # Euler and Trapezoidal
                v_avg = 0.5 * (V[i] + V[i + 1])
                T_total += self.ds / ca.fmax(v_avg, 1.0)

            # --- 2026 Regulation Logic (Speed Dependent Taper) ---
            opti.subject_to(P_DEPLOY[i] <= deploy_power_limit(V[i], ers) / PS)

            # --- Compute derivatives at node k ---
            dv_ds_k, dsoc_ds_k, F_prop_k, F_brake_k, F_grip_k = self._compute_derivatives(
                V[i], SOC[i],
                p_deploy, p_harvest, THROTTLE[i], BRAKE[i],
                gradient_arr[k], radius_arr[k], veh, ers
            )

            # --- Grip Limits at node k (Friction Circle) ---
            self._add_grip_limit(opti, F_prop_k, F_brake_k, F_grip_k, grip_scale)

            # --- Apply collocation constraints ---
            # dt: time over the interval, integrated like the SOC (same 5 m/s speed floor as the dynamics)
            if method == CollocationMethod.EULER:
                # Explicit Euler: x[k+1] = x[k] + h * f(x[k])
                opti.subject_to(V[i + 1] == V[i] + self.ds * dv_ds_k)
                opti.subject_to(SOC[i + 1] == SOC[i] + self.ds * dsoc_ds_k)
                dt = self.ds / ca.fmax(V[i], 5.0)

            else:
                # Compute derivatives at node k+1
                dv_ds_k1, dsoc_ds_k1, F_prop_k1, F_brake_k1, F_grip_k1 = self._compute_derivatives(
                    V[i + 1], SOC[i + 1],
                    p_deploy, p_harvest, THROTTLE[i], BRAKE[i],  # Same control
                    gradient_arr[k + 1], radius_arr[k + 1], veh, ers
                )

                # Grip limits at k+1: only add for the final node (others handled when they become k)
                if i == n - 1:
                    self._add_grip_limit(opti, F_prop_k1, F_brake_k1, F_grip_k1, grip_scale)

                if method == CollocationMethod.TRAPEZOIDAL:
                    # Trapezoidal: x[k+1] = x[k] + (h/2) * (f(x[k]) + f(x[k+1]))
                    opti.subject_to(V[i + 1] == V[i] + (self.ds / 2.0) * (dv_ds_k + dv_ds_k1))
                    opti.subject_to(SOC[i + 1] == SOC[i] + (self.ds / 2.0) * (dsoc_ds_k + dsoc_ds_k1))
                    dt = (self.ds / 2.0) * (1.0 / ca.fmax(V[i], 5.0) + 1.0 / ca.fmax(V[i + 1], 5.0))

                else:
                    # 1. Midpoint state from Hermite interpolation
                    # x_mid = (x[k] + x[k+1])/2 + (h/8) * (f[k] - f[k+1])
                    v_mid_hermite = 0.5 * (V[i] + V[i + 1]) + (self.ds / 8.0) * (dv_ds_k - dv_ds_k1)
                    soc_mid_hermite = 0.5 * (SOC[i] + SOC[i + 1]) + (self.ds / 8.0) * (dsoc_ds_k - dsoc_ds_k1)

                    opti.subject_to(V_MID[i] == v_mid_hermite)
                    opti.subject_to(SOC_MID[i] == soc_mid_hermite)

                    # 2. Compute derivatives at midpoint
                    grad_mid = 0.5 * (gradient_arr[k] + gradient_arr[k + 1])
                    radius_mid = 0.5 * (radius_arr[k] + radius_arr[k + 1])

                    dv_ds_mid, dsoc_ds_mid, F_prop_mid, F_brake_mid, F_grip_mid = self._compute_derivatives(
                        V_MID[i], SOC_MID[i],
                        p_deploy, p_harvest, THROTTLE[i], BRAKE[i],
                        grad_mid, radius_mid, veh, ers
                    )

                    # Grip limits at midpoint
                    self._add_grip_limit(opti, F_prop_mid, F_brake_mid, F_grip_mid, grip_scale)

                    # 3. Simpson quadrature: x[k+1] = x[k] + (h/6) * (f[k] + 4*f_mid + f[k+1])
                    opti.subject_to(
                        V[i + 1] == V[i] + (self.ds / 6.0) * (dv_ds_k + 4.0 * dv_ds_mid + dv_ds_k1)
                    )
                    opti.subject_to(
                        SOC[i + 1] == SOC[i] + (self.ds / 6.0) * (dsoc_ds_k + 4.0 * dsoc_ds_mid + dsoc_ds_k1)
                    )
                    dt = (self.ds / 6.0) * (
                        1.0 / ca.fmax(V[i], 5.0) + 4.0 / ca.fmax(V_MID[i], 5.0) + 1.0 / ca.fmax(V[i + 1], 5.0)
                    )

            # --- Energy totals: states rather than lap-long sums, so each constraint only couples
            # neighbouring nodes. Same quadrature as the SOC, so the battery bookkeeping is exact.
            opti.subject_to(E_DEPLOY[i + 1] == E_DEPLOY[i] + p_deploy * dt / 1e6)
            opti.subject_to(E_RECOVER[i + 1] == E_RECOVER[i] + p_harvest * dt / 1e6)

            # Control interlock (prevent simultaneous throttle + brake)
            opti.subject_to(THROTTLE[i] * BRAKE[i] <= 0.01)

        opti.minimize(T_total)

        # =================================================================
        # CONSTRAINTS
        # =================================================================

        # Boundary conditions
        opti.subject_to(SOC[0] == initial_soc)
        opti.subject_to(SOC[-1] >= final_soc_min)
        opti.subject_to(E_DEPLOY[0] == 0)
        opti.subject_to(E_RECOVER[0] == 0)

        # Per-lap limits
        for lap_idx in range(n_laps):
            start, end = lap_idx * self.N, (lap_idx + 1) * self.N
            opti.subject_to(E_DEPLOY[end] - E_DEPLOY[start] <= ers.deployment_limit_per_lap / 1e6)
            opti.subject_to(E_RECOVER[end] - E_RECOVER[start] <= ers.recovery_limit_per_lap / 1e6)
            if per_lap_final_soc_min is not None:
                opti.subject_to(SOC[end] >= per_lap_final_soc_min)

        # State bounds (a worn tyre lowers the speed limit by sqrt(grip scale))
        v_limit_nodes = np.concatenate(
            [v_limit_profile[:-1] * np.sqrt(scale) for scale in lap_grip_scales]
            + [v_limit_profile[-1:] * np.sqrt(lap_grip_scales[-1])]
        )
        v_limit_scale = 1.00 if degradation_enabled else 1.02
        opti.subject_to(opti.bounded(ers.min_soc, SOC, ers.max_soc))
        opti.subject_to(opti.bounded(5.0, V, v_limit_nodes * v_limit_scale))

        # Control bounds
        opti.subject_to(P_DEPLOY >= 0)
        opti.subject_to(opti.bounded(0, P_HARVEST, ers.max_recovery_power / PS))
        opti.subject_to(opti.bounded(0, THROTTLE, 1))
        opti.subject_to(opti.bounded(0, BRAKE, 1))

        # Velocity boundary condition
        if is_flying_lap:
            opti.subject_to(V[0] == V[-1])
        else:
            opti.subject_to(V[0] == v_limit_profile[0])

        # Midpoint bounds for Hermite-Simpson
        if method == CollocationMethod.HERMITE_SIMPSON:
            v_limit_mid = np.concatenate(
                [0.5 * (v_limit_profile[:-1] + v_limit_profile[1:]) * np.sqrt(scale) for scale in lap_grip_scales]
            )
            opti.subject_to(opti.bounded(5.0, V_MID, v_limit_mid * v_limit_scale))
            opti.subject_to(opti.bounded(ers.min_soc, SOC_MID, ers.max_soc))

        # =================================================================
        # SOLVE
        # =================================================================

        self._configure_solver(opti)

        # Initial guess: follow the speed limit closely, drain the battery linearly
        soc_target = max(final_soc_min, per_lap_final_soc_min or ers.min_soc)
        soc_guess = np.linspace(initial_soc, soc_target, n + 1)
        opti.set_initial(V, v_limit_nodes * 0.95)
        opti.set_initial(SOC, soc_guess)
        opti.set_initial(THROTTLE, 0.8)
        if method == CollocationMethod.HERMITE_SIMPSON:
            opti.set_initial(V_MID, v_limit_mid * 0.95)
            opti.set_initial(SOC_MID, 0.5 * (soc_guess[:-1] + soc_guess[1:]))

        variables = dict(
            V=V, SOC=SOC, E_DEPLOY=E_DEPLOY, E_RECOVER=E_RECOVER,
            P_DEPLOY=P_DEPLOY, P_HARVEST=P_HARVEST, THROTTLE=THROTTLE, BRAKE=BRAKE,
        )

        try:
            sol = opti.solve()
        except RuntimeError as e:
            return_status = opti.debug.stats().get("return_status", "unknown")
            last_iterate = self._extract_trajectory(
                opti.debug, variables, f"failed: {return_status}", n_laps, lap_grip_scales
            )
            raise SolverError(
                f"{self.name}({self._resolved_nlp_solver}) did not converge: {return_status}", last_iterate
            ) from e

        return_status = sol.stats().get("return_status", "Solve_Succeeded")
        status = _SUCCESS_STATUS.get(return_status, return_status)
        return self._extract_trajectory(sol, variables, status, n_laps, lap_grip_scales)

    def _get_effective_drag_coefficient(self, veh, ers, radius_k: float):
        """
        Calculate effective drag coefficient with 2026 Active Aerodynamics.

        Logic:
        - 2025: Fixed Drag (Standard)
        - 2026:
            - Z-Mode (High Drag) in corners (radius < threshold)
            - X-Mode (Low Drag) on straights (radius > threshold)
        """
        c_w_a = veh.c_w_a

        if ers.regulation_year >= 2026:
            # Threshold: If radius > 400m, assume we are on a straight (X-Mode)
            # 2026 regs allow X-mode generally on straights.
            if abs(radius_k) > 400.0:
                return c_w_a * 0.65  # X-Mode: ~35% Drag Reduction -> this is an understimate "The FIA has explicitly stated a target of 55% lower drag in low-drag configuration (X-Mode) compared to today's cars"
            else:
                return c_w_a         # Z-Mode: Full Drag for corners

        return c_w_a

    def _configure_solver(self, opti):
        """Configure NLP solver backend with appropriate options."""
        backend = self._resolved_nlp_solver

        if backend == "fatrop":
            opti.solver("fatrop")
            return

        if backend == "sqpmethod":
            opti.solver("sqpmethod", {"qpsol": "qpoases"})
            return

        if backend == "ipopt":
            opts = {
                "expand": True,  # SX graph: the exact Hessian is much cheaper to evaluate
                "print_time": self.verbose,
                "ipopt.max_iter": 3000,
                "ipopt.print_level": 4 if self.verbose else 0,
                "ipopt.tol": 1e-8,
                "ipopt.linear_solver": self.ipopt_linear_solver,
            }
            if not self.verbose:
                opts["ipopt.sb"] = "yes"  # Hide the Ipopt banner too
            if self.ipopt_linear_solver == "mumps":
                # AMD ordering: MUMPS's default ordering segfaults on Apple Silicon.
                # Set on every platform so macOS and Linux take the same iterations.
                opts["ipopt.mumps_pivot_order"] = 0
            if self.ipopt_hessian_approximation == "limited-memory":
                # Opt-in only: on this problem L-BFGS stops at "optimal" laps that are seconds too slow
                opts["ipopt.hessian_approximation"] = "limited-memory"

            opti.solver("ipopt", opts)
            return

        raise ValueError(f"Unknown NLP solver backend: {backend}")

    def _extract_trajectory(self, sol, variables, status, n_laps, lap_grip_scales):
        """Extract and package the optimization results."""
        PS = self.POWER_SCALE
        v_opt = sol.value(variables["V"])
        soc_opt = sol.value(variables["SOC"])
        e_deploy = sol.value(variables["E_DEPLOY"]) * 1e6   # J
        e_recover = sol.value(variables["E_RECOVER"]) * 1e6

        # Reconstruct net P_ERS for visualization
        P_ers_opt = (sol.value(variables["P_DEPLOY"]) - sol.value(variables["P_HARVEST"])) * PS

        # Compute time array
        s = np.linspace(0.0, self.track.total_length * n_laps, len(v_opt))
        v_avg = np.maximum(0.5 * (v_opt[1:] + v_opt[:-1]), 1.0)
        t_opt = np.concatenate([[0.0], np.cumsum(self.ds / v_avg)])

        trajectory = OptimalTrajectory(
            s=s,
            ds=self.ds,
            n_points=len(v_opt),
            v_opt=v_opt,
            soc_opt=soc_opt,
            P_ers_opt=P_ers_opt,
            throttle_opt=sol.value(variables["THROTTLE"]),
            brake_opt=sol.value(variables["BRAKE"]),
            t_opt=t_opt,
            lap_time=t_opt[-1],
            energy_deployed=e_deploy[-1],
            energy_recovered=e_recover[-1],
            solve_time=0.0,
            solver_status=status,
            solver_name=f"{self.name}({self._resolved_nlp_solver})",
        )

        if n_laps > 1:
            ends = np.arange(n_laps + 1) * self.N  # Node index of each lap boundary
            trajectory.n_laps = n_laps
            trajectory.lap_length = self.track.total_length
            trajectory.lap_times = np.diff(t_opt[ends])
            trajectory.lap_energy_deployed = np.diff(e_deploy[ends])
            trajectory.lap_energy_recovered = np.diff(e_recover[ends])
            trajectory.lap_start_soc = soc_opt[ends[:-1]]
            trajectory.lap_end_soc = soc_opt[ends[1:]]
            trajectory.lap_grip_scales = lap_grip_scales

        return trajectory
