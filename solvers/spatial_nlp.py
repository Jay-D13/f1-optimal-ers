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
        ∑(P_ers_deploy) ≤ 4 MJ                (regulatory limit)
        SOC_min ≤ SOC ≤ SOC_max               (battery limits)
"""

import platform
import time
from enum import Enum
from typing import Literal, Tuple

import casadi as ca
import numpy as np

from solvers import BaseSolver, OptimalTrajectory


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

    def __init__(
        self, 
        vehicle_model, 
        track_model, 
        ers_config, 
        ds: float = 5.0,
        collocation_method: Literal["euler", "trapezoidal", "hermite_simpson"] = "euler",
        nlp_solver: Literal["auto", "ipopt", "fatrop", "sqpmethod"] = "auto",
        ipopt_linear_solver: str = "mumps",
        ipopt_hessian_approximation: Literal["limited-memory", "exact"] = "limited-memory",
    ):
        super().__init__(vehicle_model, track_model, ers_config)
        self.ds = ds

        self.collocation_method = CollocationMethod(collocation_method)
        self.nlp_solver = nlp_solver
        self.ipopt_linear_solver = ipopt_linear_solver
        self.ipopt_hessian_approximation = ipopt_hessian_approximation
        self._resolved_nlp_solver = self._resolve_nlp_solver()

        # Discretization
        self.N = int(track_model.total_length / ds)
        self.s_grid = np.linspace(0, track_model.total_length, self.N + 1)

    @property
    def name(self) -> str:
        return "SpatialNLP"

    def _resolve_nlp_solver(self) -> Literal["ipopt", "fatrop", "sqpmethod"]:
        """Resolve auto solver mode to a concrete backend."""
        if self.nlp_solver == "auto":
            if platform.system() == "Darwin" and platform.machine() == "arm64":
                return "fatrop"
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
        """
        self._log(f"Setting up NLP with {self.N} nodes using {self.collocation_method.value} collocation..")

        if self.nlp_solver == "auto":
            self._log(f"Auto-selected NLP backend: {self._resolved_nlp_solver}")

        if (
            self._resolved_nlp_solver == "ipopt"
            and platform.system() == "Darwin"
            and platform.machine() == "arm64"
            and self.N >= 1000
        ):
            self._log(
                "⚠ Apple Silicon + Ipopt(MUMPS) can segfault on large NLPs. "
                "Try nlp_solver='fatrop' if this crashes."
            )
        start_time = time.time()

        # Ensure v_limit is the right size
        if len(v_limit_profile) != self.N + 1:
            v_limit_profile = np.interp(
                self.s_grid,
                np.linspace(0, self.track.total_length, len(v_limit_profile)),
                v_limit_profile,
            )

        # Build and solve NLP
        try:
            trajectory = self._build_and_solve(
                v_limit_profile=v_limit_profile,
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

    def _build_and_solve(
        self,
        v_limit_profile: np.ndarray,
        initial_soc: float,
        final_soc_min: float,
        is_flying_lap: bool
    ) -> OptimalTrajectory:
        """Build and solve the CasADi optimization problem."""
        opti = ca.Opti()

        # Get vehicle parameters
        veh = self.vehicle.vehicle
        ers = self.vehicle.ers

        # Track data arrays
        track_data = self.track.track_data
        gradient_arr = np.resize(track_data.gradient, self.N + 1)
        radius_arr = np.resize(track_data.radius, self.N + 1)

        # =================================================================
        # DECISION VARIABLES
        # =================================================================

        # States at node points
        V = opti.variable(self.N + 1)      # Velocity (m/s)
        SOC = opti.variable(self.N + 1)    # State of Charge (0-1)

        # Controls (piecewise constant over intervals)
        P_DEPLOY = opti.variable(self.N)   # ERS discharge power (≥0)
        P_HARVEST = opti.variable(self.N)  # ERS recovery power (≥0)
        THROTTLE = opti.variable(self.N)   # Throttle position (0-1)
        BRAKE = opti.variable(self.N)      # Brake position (0-1)

        # For Hermite-Simpson: midpoint states
        if self.collocation_method == CollocationMethod.HERMITE_SIMPSON:
            V_MID = opti.variable(self.N)      # Velocity at midpoints
            SOC_MID = opti.variable(self.N)    # SOC at midpoints

        # =================================================================
        # OBJECTIVE: Minimize Lap Time
        # =================================================================

        T_lap = 0
        for k in range(self.N):
            if self.collocation_method == CollocationMethod.HERMITE_SIMPSON:
                # Simpson's rule for integrating 1/v:
                # T = integral of (1/v) ds ≈ (ds/6) * (1/v_k + 4/v_mid + 1/v_{k+1})
                v_k_safe = ca.fmax(V[k], 1.0)
                v_mid_safe = ca.fmax(V_MID[k], 1.0)
                v_k1_safe = ca.fmax(V[k + 1], 1.0)
                T_lap += (self.ds / 6.0) * (1.0/v_k_safe + 4.0/v_mid_safe + 1.0/v_k1_safe)
            else:
                # Euler and Trapezoidal
                v_avg = 0.5 * (V[k] + V[k + 1])
                v_safe = ca.fmax(v_avg, 1.0)
                T_lap += self.ds / v_safe

        opti.minimize(T_lap)

        # =================================================================
        # DYNAMICS & PHYSICS
        # =================================================================

        total_deployment = 0
        total_recovery = 0

        for k in range(self.N):
            # --- Vehicle State at node k ---
            v_k = V[k]
            v_safe_k = ca.fmax(v_k, 5.0)
            grad_k = np.clip(gradient_arr[k], -0.2, 0.2)

            # --- 2026 Regulation Logic (Speed Dependent Taper) ---
            self._apply_ers_power_limits(opti, P_DEPLOY[k], v_k, ers)

            # Harvest and non-negativity limits
            opti.subject_to(P_HARVEST[k] <= ers.max_recovery_power)
            opti.subject_to(P_DEPLOY[k] >= 0)
            opti.subject_to(P_HARVEST[k] >= 0)

            # --- Compute derivatives at node k ---
            dv_ds_k, dsoc_ds_k, F_prop_k, F_brake_k, F_grip_k = self._compute_derivatives(
                V[k], SOC[k],
                P_DEPLOY[k], P_HARVEST[k], THROTTLE[k], BRAKE[k],
                gradient_arr[k], radius_arr[k], veh, ers
            )

            # --- Grip Limits at node k (Friction Circle) ---
            opti.subject_to(F_prop_k - F_brake_k <= F_grip_k)
            opti.subject_to(F_prop_k - F_brake_k >= -F_grip_k)

            # --- Apply collocation constraints ---
            if self.collocation_method == CollocationMethod.EULER:
                # Explicit Euler: x[k+1] = x[k] + h * f(x[k])
                opti.subject_to(V[k + 1] == V[k] + self.ds * dv_ds_k)
                opti.subject_to(SOC[k + 1] == SOC[k] + self.ds * dsoc_ds_k)

            elif self.collocation_method == CollocationMethod.TRAPEZOIDAL:
                # Compute derivatives at node k+1
                dv_ds_k1, dsoc_ds_k1, F_prop_k1, F_brake_k1, F_grip_k1 = self._compute_derivatives(
                    V[k + 1], SOC[k + 1],
                    P_DEPLOY[k], P_HARVEST[k], THROTTLE[k], BRAKE[k],  # Same control
                    gradient_arr[k + 1], radius_arr[k + 1], veh, ers
                )
                
                # Grip limits at k+1: only add for the final node (others handled when they become k)
                if k == self.N - 1:
                    opti.subject_to(F_prop_k1 - F_brake_k1 <= F_grip_k1)
                    opti.subject_to(F_prop_k1 - F_brake_k1 >= -F_grip_k1)

                # Trapezoidal: x[k+1] = x[k] + (h/2) * (f(x[k]) + f(x[k+1]))
                opti.subject_to(V[k + 1] == V[k] + (self.ds / 2.0) * (dv_ds_k + dv_ds_k1))
                opti.subject_to(SOC[k + 1] == SOC[k] + (self.ds / 2.0) * (dsoc_ds_k + dsoc_ds_k1))

            elif self.collocation_method == CollocationMethod.HERMITE_SIMPSON:
                # Compute derivatives at node k+1
                dv_ds_k1, dsoc_ds_k1, F_prop_k1, F_brake_k1, F_grip_k1 = self._compute_derivatives(
                    V[k + 1], SOC[k + 1],
                    P_DEPLOY[k], P_HARVEST[k], THROTTLE[k], BRAKE[k],
                    gradient_arr[k + 1], radius_arr[k + 1], veh, ers
                )
                
                # Grip limits at k+1 (only for final node)
                if k == self.N - 1:
                    opti.subject_to(F_prop_k1 - F_brake_k1 <= F_grip_k1)
                    opti.subject_to(F_prop_k1 - F_brake_k1 >= -F_grip_k1)

                # 1. Midpoint state from Hermite interpolation
                # x_mid = (x[k] + x[k+1])/2 + (h/8) * (f[k] - f[k+1])
                v_mid_hermite = 0.5 * (V[k] + V[k + 1]) + (self.ds / 8.0) * (dv_ds_k - dv_ds_k1)
                soc_mid_hermite = 0.5 * (SOC[k] + SOC[k + 1]) + (self.ds / 8.0) * (dsoc_ds_k - dsoc_ds_k1)
                
                opti.subject_to(V_MID[k] == v_mid_hermite)
                opti.subject_to(SOC_MID[k] == soc_mid_hermite)

                # 2. Compute derivatives at midpoint
                grad_mid = 0.5 * (gradient_arr[k] + gradient_arr[k + 1])
                radius_mid = 0.5 * (radius_arr[k] + radius_arr[k + 1])
                
                dv_ds_mid, dsoc_ds_mid, F_prop_mid, F_brake_mid, F_grip_mid = self._compute_derivatives(
                    V_MID[k], SOC_MID[k],
                    P_DEPLOY[k], P_HARVEST[k], THROTTLE[k], BRAKE[k],
                    grad_mid, radius_mid, veh, ers
                )

                # Grip limits at midpoint
                opti.subject_to(F_prop_mid - F_brake_mid <= F_grip_mid)
                opti.subject_to(F_prop_mid - F_brake_mid >= -F_grip_mid)

                # 3. Simpson quadrature: x[k+1] = x[k] + (h/6) * (f[k] + 4*f_mid + f[k+1])
                opti.subject_to(
                    V[k + 1] == V[k] + (self.ds / 6.0) * (dv_ds_k + 4.0 * dv_ds_mid + dv_ds_k1)
                )
                opti.subject_to(
                    SOC[k + 1] == SOC[k] + (self.ds / 6.0) * (dsoc_ds_k + 4.0 * dsoc_ds_mid + dsoc_ds_k1)
                )

            # --- Energy Accumulation ---
            dt_step = self.ds / v_safe_k
            total_deployment += P_DEPLOY[k] * dt_step
            total_recovery += P_HARVEST[k] * dt_step

            # Control interlock (prevent simultaneous throttle + brake)
            opti.subject_to(THROTTLE[k] * BRAKE[k] <= 0.01)

        # =================================================================
        # CONSTRAINTS
        # =================================================================

        # Boundary conditions
        opti.subject_to(SOC[0] == initial_soc)
        opti.subject_to(SOC[-1] >= final_soc_min)

        # Global limits
        opti.subject_to(total_deployment <= ers.deployment_limit_per_lap)
        opti.subject_to(total_recovery <= ers.recovery_limit_per_lap)

        # State bounds
        opti.subject_to(opti.bounded(ers.min_soc, SOC, ers.max_soc))
        opti.subject_to(opti.bounded(5.0, V, v_limit_profile * 1.02))

        # Control bounds
        opti.subject_to(opti.bounded(0, THROTTLE, 1))
        opti.subject_to(opti.bounded(0, BRAKE, 1))

        # Velocity boundary condition
        if is_flying_lap:
            opti.subject_to(V[0] == V[-1])
        else:
            opti.subject_to(V[0] == v_limit_profile[0])

        # Midpoint bounds for Hermite-Simpson
        if self.collocation_method == CollocationMethod.HERMITE_SIMPSON:
            v_limit_mid = 0.5 * (v_limit_profile[:-1] + v_limit_profile[1:])
            opti.subject_to(opti.bounded(5.0, V_MID, v_limit_mid * 1.02))
            opti.subject_to(opti.bounded(ers.min_soc, SOC_MID, ers.max_soc))

        # =================================================================
        # 5. SOLVE
        # =================================================================

        self._configure_solver(opti)
        self._set_initial_guess(opti, V, SOC, P_DEPLOY, THROTTLE, v_limit_profile, initial_soc)
        
        # Initial guess for midpoints
        if self.collocation_method == CollocationMethod.HERMITE_SIMPSON:
            v_mid_guess = 0.5 * (v_limit_profile[:-1] + v_limit_profile[1:]) * 0.95
            soc_mid_guess = np.linspace(initial_soc, 0.3, self.N)
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
        # 6. EXTRACT RESULTS
        # =================================================================

        return self._extract_trajectory(
            sol, V, SOC, P_DEPLOY, P_HARVEST, THROTTLE, BRAKE, 
            total_deployment, total_recovery, status
        )

    def _apply_ers_power_limits(self, opti, P_deploy_k, v_k, ers):
        """Apply speed-dependent ERS power limits for 2026+ regulations."""
        if ers.regulation_year >= 2026:
            v_kph = v_k * 3.6
            p_base = ers.max_deployment_power

            # Taper logic: Linear drop 290-340kph, Sharp drop 340-345kph
            p_taper1 = (1800.0 - 5.0 * v_kph) * 1000.0
            p_taper2 = (6900.0 - 20.0 * v_kph) * 1000.0

            limit_k = ca.fmax(0, ca.fmin(p_base, ca.fmin(p_taper1, p_taper2)))
            opti.subject_to(P_deploy_k <= limit_k)
        else:
            opti.subject_to(P_deploy_k <= ers.max_deployment_power)

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
            try:
                opti.solver("fatrop")
                return
            except Exception:
                if self.nlp_solver != "auto":
                    raise
                self._log("⚠ fatrop unavailable, falling back to ipopt")
                backend = "ipopt"
                self._resolved_nlp_solver = backend

        if backend == "sqpmethod":
            opti.solver("sqpmethod", {"qpsol": "qpoases"})
            return

        if backend == "ipopt":
            opts = {
                "ipopt.max_iter": 3000,
                "ipopt.print_level": 4,
                "ipopt.tol": 1e-4,
                "ipopt.acceptable_tol": 1e-3,
                "ipopt.linear_solver": self.ipopt_linear_solver,
                "ipopt.nlp_scaling_method": "gradient-based",
            }
            if self.ipopt_hessian_approximation == "limited-memory":
                opts["ipopt.hessian_approximation"] = "limited-memory"

            opti.solver("ipopt", opts)
            return

        raise ValueError(f"Unknown NLP solver backend: {backend}")

    def _set_initial_guess(self, opti, V, SOC, P_DEPLOY, THROTTLE, v_limit_profile, initial_soc):
        """Set initial guess for optimization variables."""
        # Velocity: Follow limit closely
        opti.set_initial(V, v_limit_profile * 0.95)

        # SOC: Linear drain
        soc_guess = np.linspace(initial_soc, 0.3, self.N + 1)
        opti.set_initial(SOC, soc_guess)

        # Controls
        opti.set_initial(THROTTLE, np.ones(self.N) * 0.8)
        opti.set_initial(P_DEPLOY, np.zeros(self.N))

    def _extract_trajectory(
        self, sol, V, SOC, P_DEPLOY, P_HARVEST, THROTTLE, BRAKE, total_deployment, total_recovery, status
    ):
        """Extract and package the optimization results."""
        v_opt = sol.value(V)
        soc_opt = sol.value(SOC)
        p_deploy_opt = sol.value(P_DEPLOY)
        p_harvest_opt = sol.value(P_HARVEST)

        # Reconstruct net P_ERS for visualization
        P_ers_opt = p_deploy_opt - p_harvest_opt

        throttle_opt = sol.value(THROTTLE)
        brake_opt = sol.value(BRAKE)

        # Compute time array
        t_opt = np.zeros_like(v_opt)
        for i in range(1, len(t_opt)):
            v_avg = 0.5 * (v_opt[i] + v_opt[i - 1])
            t_opt[i] = t_opt[i - 1] + self.ds / max(v_avg, 1.0)

        return OptimalTrajectory(
            s=self.s_grid,
            ds=self.ds,
            n_points=self.N + 1,
            v_opt=v_opt,
            soc_opt=soc_opt,
            P_ers_opt=P_ers_opt,
            throttle_opt=throttle_opt,
            brake_opt=brake_opt,
            t_opt=t_opt,
            lap_time=t_opt[-1],
            energy_deployed=sol.value(total_deployment),
            energy_recovered=sol.value(total_recovery),
            solve_time=0.0,
            solver_status=status,
            solver_name=f"{self.name}({self._resolved_nlp_solver})",
        )
