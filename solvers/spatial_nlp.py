"""
Spatial-Domain NLP Solver for ERS Optimization

Offline optimizer using direct collocation in spatial domain.

Problem Formulation:
    minimize    T = ∑(ds / v[k])              (lap time)
    subject to: 
        v[k] ≤ v_limit[k]                     (grip limit from Forward-Backward)
        v dynamics from ERS + ICE power
        SOC dynamics from ERS power
        ∑(P_ers_deploy) ≤ 4 MJ                (regulatory limit)
        SOC_min ≤ SOC ≤ SOC_max               (battery limits)
"""

import time

import casadi as ca
import numpy as np

from solvers import BaseSolver, OptimalTrajectory


class SpatialNLPSolver(BaseSolver):
    """
    Finds the globally (offline) optimal ERS deployment strategy for a single lap.
    Optimizes lap time subject to energy budget and thermal/grip limits.
    """

    def __init__(self, vehicle_model, track_model, ers_config, ds: float = 5.0):
        super().__init__(vehicle_model, track_model, ers_config)
        self.ds = ds

        # Discretization
        self.N = int(track_model.total_length / ds)
        self.s_grid = np.linspace(0, track_model.total_length, self.N + 1)

    @property
    def name(self) -> str:
        return "SpatialNLP"

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

        Returns:
            OptimalTrajectory containing solution
        """
        self._log(f"Setting up NLP with {self.N} nodes...")
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
        gradient_arr = np.resize(track_data.gradient, self.N)
        
        radius_arr = np.resize(track_data.radius, self.N)

        # =================================================================
        # DECISION VARIABLES
        # =================================================================

        # States
        V = opti.variable(self.N + 1)  # Velocity (m/s)
        SOC = opti.variable(self.N + 1)  # State of Charge (0-1)

        # Controls (split to avoid max() in constraints)
        P_DEPLOY = opti.variable(self.N)  # ERS discharge power (≥0)
        P_HARVEST = opti.variable(self.N)  # ERS recovery power (≥0)
        THROTTLE = opti.variable(self.N)  # Throttle position (0-1)
        BRAKE = opti.variable(self.N)  # Brake position (0-1)

        # =================================================================
        # OBJECTIVE: Minimize Lap Time
        # =================================================================

        T_lap = 0
        for k in range(self.N):
            # Trapezoidal integration for time
            v_avg = 0.5 * (V[k] + V[k + 1])
            v_safe = ca.fmax(v_avg, 1.0)  # Avoid singularity
            T_lap += self.ds / v_safe

        opti.minimize(T_lap)

        # =================================================================
        # DYNAMICS & PHYSICS
        # =================================================================

        total_deployment = 0
        total_recovery = 0

        for k in range(self.N):
            # --- Environment ---
            grad = np.clip(gradient_arr[k], -0.2, 0.2)

            # --- Vehicle State ---
            v_k = V[k]
            v_safe = ca.fmax(v_k, 5.0)

            # --- 2026 Regulation Logic (Speed Dependent Taper) ---
            self._apply_ers_power_limits(opti, P_DEPLOY[k], v_k, ers)

            # Harvest and non-negativity limits
            opti.subject_to(P_HARVEST[k] <= ers.max_recovery_power)
            opti.subject_to(P_DEPLOY[k] >= 0)
            opti.subject_to(P_HARVEST[k] >= 0)

            # --- Forces ---
            # Drag (with 2026 active aero adjustment)
            c_w_a_eff = self._get_effective_drag_coefficient(veh, ers, radius_arr[k])
            F_drag = 0.5 * veh.rho_air * v_k**2 * c_w_a_eff

            # Resistances
            F_roll = veh.mass * veh.g * veh.cr
            F_grav = veh.mass * veh.g * ca.sin(grad)

            # Propulsion
            P_net_ers = P_DEPLOY[k] - P_HARVEST[k]
            P_ice = THROTTLE[k] * veh.pow_max_ice
            F_prop = (P_ice + P_net_ers) / v_safe

            # Braking
            F_brake = BRAKE[k] * veh.max_brake_force

            # Net Force
            F_net = F_prop - F_brake - F_drag - F_roll - F_grav

            # --- Grip Limits (Friction Circle) ---
            F_norm = veh.mass * veh.g * ca.cos(grad) + 0.5 * veh.rho_air * v_k**2 * (
                veh.c_z_a_f + veh.c_z_a_r
            )
            F_grip = veh.mu_longitudinal * F_norm

            opti.subject_to(F_prop - F_brake <= F_grip)
            opti.subject_to(F_prop - F_brake >= -F_grip)

            # --- Dynamics Integration (Euler) ---
            dv_ds = F_net / (veh.mass * v_safe)
            opti.subject_to(V[k + 1] == V[k] + dv_ds * self.ds)

            # --- Battery Dynamics ---
            P_bat_out = (P_DEPLOY[k] / ers.deployment_efficiency) - (
                P_HARVEST[k] * ers.recovery_efficiency
            )
            dsoc_ds = -P_bat_out / (ers.battery_capacity * v_safe)
            opti.subject_to(SOC[k + 1] == SOC[k] + dsoc_ds * self.ds)

            # --- Energy Accumulation ---
            dt_step = self.ds / v_safe
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
        # opti.subject_to(V[0] == v_limit_profile[0])

        # Global limits
        opti.subject_to(total_deployment <= ers.deployment_limit_per_lap)
        opti.subject_to(total_recovery <= ers.recovery_limit_per_lap)

        # State bounds
        opti.subject_to(opti.bounded(ers.min_soc, SOC, ers.max_soc))
        opti.subject_to(opti.bounded(5.0, V, v_limit_profile * 1.02))

        # Control bounds
        opti.subject_to(opti.bounded(0, THROTTLE, 1))
        opti.subject_to(opti.bounded(0, BRAKE, 1))
        
        if is_flying_lap:
            opti.subject_to(V[0] == V[-1])
        else:
            opti.subject_to(V[0] == v_limit_profile[0])

        # =================================================================
        # 5. SOLVE
        # =================================================================

        self._configure_solver(opti)
        self._set_initial_guess(opti, V, SOC, P_DEPLOY, THROTTLE, v_limit_profile, initial_soc)

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
            sol, V, SOC, P_DEPLOY, P_HARVEST, THROTTLE, BRAKE, total_deployment, total_recovery, status
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
        """Configure IPOPT solver with appropriate options."""
        opts = {
            "ipopt.max_iter": 3000,
            "ipopt.print_level": 4,
            "ipopt.tol": 1e-4,
            "ipopt.acceptable_tol": 1e-3,
            "ipopt.linear_solver": "ma97",
            "ipopt.nlp_scaling_method": "gradient-based",
        }

        try:
            opti.solver("ipopt", opts)
        except Exception:
            print("⚠ MA97 not found, falling back to MUMPS")
            opts["ipopt.linear_solver"] = "mumps"
            opti.solver("ipopt", opts)

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
            solver_name=self.name,
        )