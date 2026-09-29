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
        v dynamics from the car model (models/car.py): ICE + ERS power, brakes, drag, rolling resistance
        per-axle friction ellipses at every collocation point (grip limit, including cornering)
        SOC dynamics from ERS power
        E_deploy, E_recover ≤ per-lap limits  (regulatory limits; running totals are states)
        SOC_min ≤ SOC ≤ SOC_max               (battery limits)
        tyre temperatures and wear            (optional dynamic tyre model, multi-lap)
"""

import time
from dataclasses import dataclass
from enum import Enum
from typing import Literal

import casadi as ca
import numpy as np

from config import TireCompoundConfig, TireThermalConfig
from models.car import V_MAX, deploy_power_limit
from models.tire_thermals import (
    core_temp_rate_ca,
    heat_generation_ca,
    mu_scale_ca,
    mu_scale_np,
    surface_temp_rate_ca,
    utilization_ca,
    wear_rate_ca,
)
from solvers import BaseSolver, OptimalTrajectory, SolverError

# Ipopt return statuses that count as a solution
_SUCCESS_STATUS = {"Solve_Succeeded": "optimal", "Solved_To_Acceptable_Level": "acceptable"}

# Dynamic tyre states: surface temperature, core temperature (°C) and wear (0-1), front then rear
TIRE_STATES = ("TSF", "TCF", "WF", "TSR", "TCR", "WR")
_WEAR_STATES = ("WF", "WR")

V_MIN = 5.0  # Lower speed bound (m/s)

# NLP controls (see _build_and_solve)
CONTROLS = ("P_DEPLOY", "P_HARVEST", "THROTTLE", "BRAKE_F", "BRAKE_R")


@dataclass
class DynamicTireSettings:
    """Inputs of the dynamic tyre model, which adds tyre temperature and wear states to the NLP."""
    thermal: TireThermalConfig
    compound: TireCompoundConfig
    ambient_temp_c: float
    track_temp_c: float
    init_temp_c: float


class CollocationMethod(Enum):
    """Available collocation/integration methods."""
    EULER = "euler"                    # 1st order - explicit Euler
    TRAPEZOIDAL = "trapezoidal"        # 2nd order - implicit trapezoidal
    HERMITE_SIMPSON = "hermite_simpson" # 4th order - Hermite-Simpson


class SpatialNLPSolver(BaseSolver):
    """
    Finds the globally (offline) optimal ERS deployment strategy for a single lap.
    Optimizes lap time subject to energy budget, battery and per-axle grip limits.
    """

    # ERS powers are solved in units of 100 kW so that every variable is O(1)
    POWER_SCALE = 1e5
    # Objective cost of friction braking (s per m at full brake force); see _build_and_solve
    BRAKE_COST = 1e-5

    def __init__(
        self,
        vehicle_model,
        track_model,
        ers_config,
        ds: float = 5.0,
        collocation_method: Literal["euler", "trapezoidal", "hermite_simpson"] = "trapezoidal",
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

    @property
    def car(self):
        return self.vehicle.car

    def _resolve_nlp_solver(self) -> Literal["ipopt", "fatrop", "sqpmethod"]:
        """Resolve auto solver mode to a concrete backend: Ipopt everywhere, fatrop and sqpmethod are opt-in."""
        if self.nlp_solver == "auto":
            return "ipopt"
        return self.nlp_solver

    def solve(
        self,
        v_guess: np.ndarray | None = None,
        initial_soc: float = 0.5,
        final_soc_min: float = 0.3,
        is_flying_lap: bool = True
    ) -> OptimalTrajectory:
        """
        Solve the optimal control problem for ERS deployment.

        Args:
            v_guess: Speed profile for the initial guess, on the track points (default: the forward-backward
                     profile without ERS). It is not a constraint: the grip limits are inside the NLP.
            initial_soc: Starting state of charge (0-1)
            final_soc_min: Minimum final state of charge
            is_flying_lap: If True, enforce V[0] == V[-1]; otherwise the lap starts at v_guess[0]

        Returns:
            OptimalTrajectory containing solution

        Raises:
            SolverError: if the NLP solver does not converge
        """
        self._log(f"Setting up NLP with {self.N} nodes using {self.collocation_method.value} collocation..")

        if self.nlp_solver == "auto":
            self._log(f"Auto-selected NLP backend: {self._resolved_nlp_solver}")
        start_time = time.time()

        trajectory = self._build_and_solve(
            v_guess=self._guess_on_grid(v_guess, is_flying_lap),
            initial_soc=initial_soc,
            final_soc_min=final_soc_min,
            is_flying_lap=is_flying_lap
        )
        trajectory.solve_time = time.time() - start_time

        self._log(f"✓ Solved in {trajectory.solve_time:.2f}s")
        self._log(f"  Lap time: {trajectory.lap_time:.3f}s")
        self._log(f"  Status: {trajectory.solver_status}")
        return trajectory

    def _guess_on_grid(self, v_guess: np.ndarray | None, is_flying_lap: bool) -> np.ndarray:
        """Initial speed guess at the NLP nodes: the given profile, or the forward-backward one without ERS."""
        if v_guess is None:
            from solvers.forward_backward import ForwardBackwardSolver
            v_guess = ForwardBackwardSolver(self.vehicle, self.track, use_ers_power=False).solve(
                flying_lap=is_flying_lap
            ).v
        return np.clip(self._sample_on_grid(v_guess), V_MIN, V_MAX)

    def _sample_on_grid(self, values: np.ndarray) -> np.ndarray:
        """Sample a per-lap profile at the NLP nodes. It is given on the track's points, or evenly spaced over the lap."""
        values = np.asarray(values, dtype=float)
        s_track = getattr(self.track.track_data, "s", None)
        if s_track is not None and len(values) == len(s_track):
            # Track points stop short of the finish line, so wrap around the lap
            return np.interp(self.s_grid, s_track, values, period=self.track.total_length)
        return np.interp(self.s_grid, np.linspace(0, self.track.total_length, len(values)), values)

    def _track_on_grid(self):
        """Curvature (1/m), gradient (rad) and aero mode at the NLP nodes of one lap."""
        radius = self._sample_on_grid(self.track.track_data.radius)
        gradient = self._sample_on_grid(self.track.track_data.gradient)
        return 1.0 / np.abs(radius), gradient, self.car.aero_mode(radius)

    def _point_dynamics(self, opti, x, u, kappa, gradient, w, grip_scale, tire, add_constraints):
        """
        State derivatives d/ds at one collocation point, keyed like the states.

        u holds the physical controls (deploy and harvest power in W, throttle, front and rear brake force in N).
        With add_constraints, also adds the point's per-axle friction ellipses. With dynamic tyres, the tyre
        temperature and wear scale each axle's friction and drive the tyre states.
        """
        car = self.car
        p_deploy, p_harvest, throttle, brake_front, brake_rear = u
        v = x["V"]
        drive = car.drive_force(v, throttle, p_deploy, p_harvest)

        if tire is None:
            grip_front = grip_rear = grip_scale
        else:
            grip_front = mu_scale_ca(x["TCF"], x["WF"], tire.thermal, tire.compound)
            grip_rear = mu_scale_ca(x["TCR"], x["WR"], tire.thermal, tire.compound)

        f = car.point(v, kappa, gradient, w, drive, brake_front, brake_rear, grip_front, grip_rear)
        if add_constraints:
            opti.subject_to(f.usage_front <= 1.0)
            opti.subject_to(f.usage_rear <= 1.0)

        derivatives = {
            "V": f.a_x / v,
            "SOC": -car.battery_power(p_deploy, p_harvest) / (self.vehicle.ers.battery_capacity * v),
        }
        if tire is not None:
            derivatives.update(self._tire_derivatives(x, f, v, tire))
        return derivatives

    @staticmethod
    def _tire_derivatives(x, f, v, tire: DynamicTireSettings):
        """Spatial derivatives of the tyre temperatures and wear, driven by each axle's friction usage."""
        thermal, compound = tire.thermal, tire.compound
        p = thermal.utilization_ellipse_p
        derivatives = {}
        for axle, F_x, F_y, F_x_max, F_y_max, F_z in (
            ("F", f.F_x_front, f.F_y_front, f.F_x_max_front, f.F_y_max_front, f.F_z_front),
            ("R", f.F_x_rear, f.F_y_rear, f.F_x_max_rear, f.F_y_max_rear, f.F_z_rear),
        ):
            t_surface, t_core, wear = x["TS" + axle], x["TC" + axle], x["W" + axle]
            usage = utilization_ca(F_x, F_y, F_x_max, F_y_max, p)
            q_gen = heat_generation_ca(F_z, v, usage, thermal, compound)
            q_surface_core = thermal.k_surface_core * (t_surface - t_core)
            derivatives["TS" + axle] = surface_temp_rate_ca(
                q_gen, t_surface, t_core, tire.ambient_temp_c, tire.track_temp_c, thermal
            ) / v
            derivatives["TC" + axle] = core_temp_rate_ca(q_surface_core, t_core, tire.ambient_temp_c, thermal) / v
            derivatives["W" + axle] = wear_rate_ca(usage, t_core, F_z, thermal, compound) / v
        return derivatives

    def _build_and_solve(
        self,
        v_guess: np.ndarray,
        initial_soc: float,
        final_soc_min: float,
        is_flying_lap: bool,
        n_laps: int = 1,
        per_lap_final_soc_min: float | None = None,
        lap_grip_scales: np.ndarray | None = None,
        tire: DynamicTireSettings | None = None,
    ) -> OptimalTrajectory:
        """
        Build and solve the CasADi optimization problem over n_laps consecutive laps.
        v_guess is the initial speed guess at the nodes of one lap.
        With `tire`, tyre temperatures and wear are states that set the grip (dynamic tyre model).
        """
        opti = ca.Opti()

        ers = self.vehicle.ers
        car = self.car
        PS = self.POWER_SCALE
        F_BRAKE = self.vehicle.vehicle.max_brake_force  # Brake controls are fractions of the brake system's force
        method = self.collocation_method

        kappa_arr, gradient_arr, w_arr = self._track_on_grid()

        if lap_grip_scales is None:
            lap_grip_scales = np.ones(n_laps)

        n = self.N * n_laps  # Intervals over the whole horizon

        # =================================================================
        # DECISION VARIABLES
        # =================================================================

        # States at node points, integrated by the collocation scheme
        X = {
            "V": opti.variable(n + 1),    # Velocity (m/s)
            "SOC": opti.variable(n + 1),  # State of Charge (0-1)
        }
        if tire is not None:
            X.update({name: opti.variable(n + 1) for name in TIRE_STATES})
        V, SOC = X["V"], X["SOC"]

        # Running energy totals (MJ): states, so each constraint only couples neighbouring nodes
        E_DEPLOY = opti.variable(n + 1)   # ERS energy deployed since the start
        E_RECOVER = opti.variable(n + 1)  # ERS energy recovered since the start

        # Controls at the nodes, linear in between (trapezoidal collocation); Hermite-Simpson adds midpoint
        # controls. Each node's grip limit then uses exactly the control its dynamics apply.
        #   P_DEPLOY, P_HARVEST: ERS discharge and recovery power (≥0, units of POWER_SCALE)
        #   THROTTLE: throttle position (0-1)
        #   BRAKE_F, BRAKE_R: front and rear brake force (fractions of max_brake_force)
        U = {name: opti.variable(n + 1) for name in CONTROLS}

        # For Hermite-Simpson: midpoint states and controls
        U_MID = {}
        if method == CollocationMethod.HERMITE_SIMPSON:
            X_MID = {name: opti.variable(n) for name in X}
            V_MID = X_MID["V"]
            U_MID = {name: opti.variable(n) for name in CONTROLS}

        def physical(u, j):
            """Controls at index j in physical units: deploy and harvest power (W), throttle, brake forces (N)."""
            return (
                u["P_DEPLOY"][j] * PS, u["P_HARVEST"][j] * PS, u["THROTTLE"][j],
                u["BRAKE_F"][j] * F_BRAKE, u["BRAKE_R"][j] * F_BRAKE,
            )

        # =================================================================
        # OBJECTIVE, DYNAMICS & PHYSICS
        # =================================================================

        # Derivatives and grip limits at every node, each with its own control
        f_nodes = []
        for j in range(n + 1):
            k = j % self.N if j < n else self.N   # Node index within the lap
            f_nodes.append(self._point_dynamics(
                opti, {name: x[j] for name, x in X.items()}, physical(U, j),
                kappa_arr[k], gradient_arr[k], w_arr[k], float(lap_grip_scales[min(j // self.N, n_laps - 1)]),
                tire, add_constraints=True,
            ))
            # --- 2026 Regulation Logic (Speed Dependent Taper) ---
            opti.subject_to(U["P_DEPLOY"][j] <= deploy_power_limit(V[j], ers) / PS)

        T_total = 0

        for i in range(n):
            k = i % self.N  # Node index within the lap
            f_k, f_k1 = f_nodes[i], f_nodes[i + 1]
            # Deploy and recovery power over speed: integrated like the SOC, so the energy bookkeeping is exact
            rate_k = (U["P_DEPLOY"][i] / V[i], U["P_HARVEST"][i] / V[i])
            rate_k1 = (U["P_DEPLOY"][i + 1] / V[i + 1], U["P_HARVEST"][i + 1] / V[i + 1])

            if method == CollocationMethod.EULER:
                # Explicit Euler: x[k+1] = x[k] + h * f(x[k])
                T_total += self.ds / (0.5 * (V[i] + V[i + 1]))
                for name, x in X.items():
                    opti.subject_to(x[i + 1] == x[i] + self.ds * f_k[name])
                energy = [self.ds * r for r in rate_k]

            elif method == CollocationMethod.TRAPEZOIDAL:
                # Trapezoidal: x[k+1] = x[k] + (h/2) * (f(x[k]) + f(x[k+1]))
                T_total += self.ds / (0.5 * (V[i] + V[i + 1]))
                for name, x in X.items():
                    opti.subject_to(x[i + 1] == x[i] + (self.ds / 2.0) * (f_k[name] + f_k1[name]))
                energy = [(self.ds / 2.0) * (a + b) for a, b in zip(rate_k, rate_k1)]

            else:
                # Simpson's rule for integrating 1/v:
                # T = integral of (1/v) ds ≈ (ds/6) * (1/v_k + 4/v_mid + 1/v_{k+1})
                T_total += (self.ds / 6.0) * (1.0 / V[i] + 4.0 / V_MID[i] + 1.0 / V[i + 1])

                # 1. Midpoint states from Hermite interpolation
                # x_mid = (x[k] + x[k+1])/2 + (h/8) * (f[k] - f[k+1])
                for name, x in X.items():
                    opti.subject_to(
                        X_MID[name][i] == 0.5 * (x[i] + x[i + 1]) + (self.ds / 8.0) * (f_k[name] - f_k1[name])
                    )

                # 2. Derivatives and grip limits at the midpoint, with the midpoint controls
                f_mid = self._point_dynamics(
                    opti, {name: x[i] for name, x in X_MID.items()}, physical(U_MID, i),
                    0.5 * (kappa_arr[k] + kappa_arr[k + 1]),
                    0.5 * (gradient_arr[k] + gradient_arr[k + 1]),
                    0.5 * (w_arr[k] + w_arr[k + 1]),
                    float(lap_grip_scales[i // self.N]), tire, add_constraints=True,
                )
                opti.subject_to(U_MID["P_DEPLOY"][i] <= deploy_power_limit(V_MID[i], ers) / PS)

                # 3. Simpson quadrature: x[k+1] = x[k] + (h/6) * (f[k] + 4*f_mid + f[k+1])
                for name, x in X.items():
                    opti.subject_to(
                        x[i + 1] == x[i] + (self.ds / 6.0) * (f_k[name] + 4.0 * f_mid[name] + f_k1[name])
                    )
                rate_mid = (U_MID["P_DEPLOY"][i] / V_MID[i], U_MID["P_HARVEST"][i] / V_MID[i])
                energy = [(self.ds / 6.0) * (a + 4.0 * m + b) for a, m, b in zip(rate_k, rate_mid, rate_k1)]

            # --- Energy totals (MJ) ---
            opti.subject_to(E_DEPLOY[i + 1] == E_DEPLOY[i] + energy[0] * PS / 1e6)
            opti.subject_to(E_RECOVER[i + 1] == E_RECOVER[i] + energy[1] * PS / 1e6)

        # A tiny cost on friction braking, so that where only the net rear force matters the solver lifts
        # instead of braking against the throttle. Braking 1 m at full force costs BRAKE_COST seconds.
        braking = sum(ca.sum1(u["BRAKE_F"] + u["BRAKE_R"]) for u in (U, U_MID) if u)
        opti.minimize(T_total + self.BRAKE_COST * self.ds * braking)

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

        # State bounds. Speed has only loose physical bounds: the grip limits are the ellipses above.
        opti.subject_to(opti.bounded(ers.min_soc, SOC, ers.max_soc))
        opti.subject_to(opti.bounded(V_MIN, V, V_MAX))

        if tire is not None:
            for name in TIRE_STATES:
                wear = name in _WEAR_STATES
                opti.subject_to(opti.bounded(0.0, X[name], 1.0) if wear else opti.bounded(20.0, X[name], 220.0))
                opti.subject_to(X[name][0] == (0.0 if wear else tire.init_temp_c))

        # Control bounds
        for u in (U, U_MID):
            if not u:
                continue
            opti.subject_to(u["P_DEPLOY"] >= 0)
            opti.subject_to(opti.bounded(0, u["P_HARVEST"], ers.max_recovery_power / PS))
            opti.subject_to(opti.bounded(0, u["THROTTLE"], 1))
            opti.subject_to(u["BRAKE_F"] >= 0)
            opti.subject_to(u["BRAKE_R"] >= 0)
            opti.subject_to(u["BRAKE_F"] + u["BRAKE_R"] <= 1)

        # Velocity boundary condition
        if is_flying_lap:
            opti.subject_to(V[0] == V[-1])
        elif tire is not None:
            # Allow a cold-tyre start below the guessed start speed
            opti.subject_to(V[0] <= v_guess[0])
        else:
            opti.subject_to(V[0] == v_guess[0])

        # Midpoint bounds for Hermite-Simpson
        if method == CollocationMethod.HERMITE_SIMPSON:
            opti.subject_to(opti.bounded(V_MIN, V_MID, V_MAX))
            opti.subject_to(opti.bounded(ers.min_soc, X_MID["SOC"], ers.max_soc))
            if tire is not None:
                for name in TIRE_STATES:
                    wear = name in _WEAR_STATES
                    opti.subject_to(
                        opti.bounded(0.0, X_MID[name], 1.0) if wear else opti.bounded(20.0, X_MID[name], 220.0)
                    )

        # =================================================================
        # SOLVE
        # =================================================================

        self._configure_solver(opti)

        # Initial guess (SOL-8): the guessed speed profile with ERS off, which the car model can drive,
        # with the throttle and brakes it needs; tyres warm and wearing slowly
        v_nodes = np.concatenate([v_guess[:-1]] * n_laps + [v_guess[-1:]])
        interval_guess = dict(zip(("THROTTLE", "BRAKE_F", "BRAKE_R"), self._controls_for(v_nodes, kappa_arr, gradient_arr, w_arr)))
        guesses = {"V": v_nodes, "SOC": np.full(n + 1, np.clip(initial_soc, ers.min_soc, ers.max_soc))}
        if tire is not None:
            for name in TIRE_STATES:
                wear = name in _WEAR_STATES
                guesses[name] = np.linspace(0.0, 0.15, n + 1) if wear else np.full(n + 1, tire.init_temp_c)
        for name, guess in guesses.items():
            opti.set_initial(X[name], guess)
            if method == CollocationMethod.HERMITE_SIMPSON:
                opti.set_initial(X_MID[name], 0.5 * (guess[:-1] + guess[1:]))
        for name, guess in interval_guess.items():
            opti.set_initial(U[name], np.append(guess, guess[-1]))
            if U_MID:
                opti.set_initial(U_MID[name], guess)

        variables = dict(X, E_DEPLOY=E_DEPLOY, E_RECOVER=E_RECOVER, U=U, U_MID=U_MID)

        try:
            sol = opti.solve()
        except RuntimeError as e:
            return_status = opti.debug.stats().get("return_status", "unknown")
            last_iterate = self._extract_trajectory(
                opti.debug, variables, f"failed: {return_status}", n_laps, lap_grip_scales, tire
            )
            raise SolverError(
                f"{self.name}({self._resolved_nlp_solver}) did not converge: {return_status}", last_iterate
            ) from e

        return_status = sol.stats().get("return_status", "Solve_Succeeded")
        status = _SUCCESS_STATUS.get(return_status, return_status)
        return self._extract_trajectory(sol, variables, status, n_laps, lap_grip_scales, tire)

    def _controls_for(self, v_nodes, kappa_arr, gradient_arr, w_arr):
        """Throttle and brakes (fractions) that drive the speed profile v_nodes with ERS off, per interval."""
        car = self.car
        veh = self.vehicle.vehicle
        k = np.arange(len(v_nodes) - 1) % self.N
        v_mid = 0.5 * (v_nodes[1:] + v_nodes[:-1])
        a_x = (v_nodes[1:] ** 2 - v_nodes[:-1] ** 2) / (2.0 * self.ds)
        w_mid = 0.5 * (w_arr[k] + w_arr[k + 1])
        needed = car.mass * a_x + car.resistance(v_mid, kappa_arr[k], gradient_arr[k], w_mid)
        throttle = np.clip(needed * v_mid / veh.pow_max_ice, 0.0, 1.0)
        brake = np.clip(-needed / veh.max_brake_force, 0.0, 1.0)
        return throttle, veh.brake_balance_front * brake, (1.0 - veh.brake_balance_front) * brake

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

    def _node_forces(self, v, controls, grip_scales):
        """Car-model forces at the nodes (NumPy), from the node controls in physical units."""
        car = self.car
        kappa_arr, gradient_arr, w_arr = self._track_on_grid()
        n = len(v) - 1
        k = np.where(np.arange(n + 1) < n, np.arange(n + 1) % self.N, self.N)
        drive = car.drive_force(v, controls["throttle"], controls["P_deploy"], controls["P_harvest"])
        return car.point(
            v, kappa_arr[k], gradient_arr[k], w_arr[k], drive,
            controls["brake_front"], controls["brake_rear"], grip_scales, grip_scales,
        )

    def _extract_trajectory(self, sol, variables, status, n_laps, lap_grip_scales, tire=None):
        """Extract and package the optimization results."""
        PS = self.POWER_SCALE
        F_BRAKE = self.vehicle.vehicle.max_brake_force
        method = self.collocation_method
        v_opt = sol.value(variables["V"])
        soc_opt = sol.value(variables["SOC"])
        e_deploy = sol.value(variables["E_DEPLOY"]) * 1e6   # J
        e_recover = sol.value(variables["E_RECOVER"]) * 1e6

        # Controls in physical units: power in W, brakes as fractions of max_brake_force
        units = {"P_DEPLOY": PS, "P_HARVEST": PS, "THROTTLE": 1.0, "BRAKE_F": 1.0, "BRAKE_R": 1.0}
        names = {"P_DEPLOY": "P_deploy", "P_HARVEST": "P_harvest", "THROTTLE": "throttle",
                 "BRAKE_F": "brake_front", "BRAKE_R": "brake_rear"}
        node = {names[c]: np.atleast_1d(sol.value(variables["U"][c])) * units[c] for c in CONTROLS}
        mid = None
        if variables["U_MID"]:
            mid = {names[c]: np.atleast_1d(sol.value(variables["U_MID"][c])) * units[c] for c in CONTROLS}

        def interval(name):
            """Mean of a control over each interval, with the collocation's quadrature weights."""
            a, b = node[name][:-1], node[name][1:]
            if method == CollocationMethod.EULER:
                return a
            if mid is not None:
                return (a + 4.0 * mid[name] + b) / 6.0
            return 0.5 * (a + b)

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
            P_ers_opt=interval("P_deploy") - interval("P_harvest"),
            throttle_opt=interval("throttle"),
            brake_opt=interval("brake_front") + interval("brake_rear"),
            t_opt=t_opt,
            lap_time=t_opt[-1],
            energy_deployed=e_deploy[-1],
            energy_recovered=e_recover[-1],
            solve_time=0.0,
            solver_status=status,
            solver_name=f"{self.name}({self._resolved_nlp_solver})",
            P_deploy_opt=interval("P_deploy"),
            P_harvest_opt=interval("P_harvest"),
            brake_front_opt=interval("brake_front"),
            brake_rear_opt=interval("brake_rear"),
            node_controls=dict(node, mid=mid),
        )

        if tire is None:
            # Friction-ellipse usage at the nodes (≤ 1 where the solution respects grip)
            lap_of_node = np.minimum(np.arange(len(v_opt)) // self.N, n_laps - 1)
            forces = self._node_forces(
                v_opt, dict(node, brake_front=node["brake_front"] * F_BRAKE, brake_rear=node["brake_rear"] * F_BRAKE),
                np.asarray(lap_grip_scales, dtype=float)[lap_of_node],
            )
            trajectory.grip_usage_front = np.asarray(forces.usage_front)
            trajectory.grip_usage_rear = np.asarray(forces.usage_rear)

        if n_laps > 1:
            ends = np.arange(n_laps + 1) * self.N  # Node index of each lap boundary
            trajectory.n_laps = n_laps
            trajectory.lap_length = self.track.total_length
            trajectory.lap_times = np.diff(t_opt[ends])
            trajectory.lap_energy_deployed = np.diff(e_deploy[ends])
            trajectory.lap_energy_recovered = np.diff(e_recover[ends])
            trajectory.lap_start_soc = soc_opt[ends[:-1]]
            trajectory.lap_end_soc = soc_opt[ends[1:]]
            trajectory.lap_grip_scales = None if tire is not None else lap_grip_scales

        if tire is not None:
            states = {name: np.asarray(sol.value(variables[name]), dtype=float) for name in TIRE_STATES}
            trajectory.tire_temp_surface_front = states["TSF"]
            trajectory.tire_temp_surface_rear = states["TSR"]
            trajectory.tire_temp_core_front = states["TCF"]
            trajectory.tire_temp_core_rear = states["TCR"]
            trajectory.tire_wear_front = states["WF"]
            trajectory.tire_wear_rear = states["WR"]
            trajectory.tire_mu_scale_front = np.array(
                [mu_scale_np(t, w, tire.thermal, tire.compound) for t, w in zip(states["TCF"], states["WF"])]
            )
            trajectory.tire_mu_scale_rear = np.array(
                [mu_scale_np(t, w, tire.thermal, tire.compound) for t, w in zip(states["TCR"], states["WR"])]
            )

        return trajectory

    def reintegrate(self, trajectory: OptimalTrajectory, substeps: int = 10) -> dict:
        """
        Re-integrate the optimal controls with the car model on a grid `substeps` times finer (RK4 in distance).

        The controls follow the collocation's own profile inside each interval (constant for Euler, linear for
        trapezoidal, linear through the midpoint for Hermite-Simpson) and the track data are interpolated
        linearly. This checks the collocation: the lap time and speeds should match the NLP's, and the grip
        usage should stay near 1 or below between the nodes.

        Returns: dict with lap_time (s), s and v (fine grid), and the largest grip usage per axle.
        Not available for the dynamic tyre model.
        """
        if trajectory.tire_temp_core_front is not None:
            raise NotImplementedError("reintegrate does not model the dynamic tyre states")

        car = self.car
        F_BRAKE = self.vehicle.vehicle.max_brake_force
        method = self.collocation_method
        kappa_arr, gradient_arr, w_arr = self._track_on_grid()
        node, mid = trajectory.node_controls, trajectory.node_controls["mid"]
        names = ("P_deploy", "P_harvest", "throttle", "brake_front", "brake_rear")
        n = len(trajectory.v_opt) - 1
        n_laps = max(1, n // self.N)
        scales = np.ones(n_laps) if trajectory.lap_grip_scales is None else np.asarray(trajectory.lap_grip_scales, float)
        h = self.ds / substeps

        def controls(i, frac):
            """Controls inside interval i at fraction frac of it."""
            if method == CollocationMethod.EULER:
                values = [node[c][i] for c in names]
            elif mid is None:
                values = [(1.0 - frac) * node[c][i] + frac * node[c][i + 1] for c in names]
            elif frac <= 0.5:
                values = [node[c][i] + 2.0 * frac * (mid[c][i] - node[c][i]) for c in names]
            else:
                values = [mid[c][i] + (2.0 * frac - 1.0) * (node[c][i + 1] - mid[c][i]) for c in names]
            p_deploy, p_harvest, throttle, brake_front, brake_rear = values
            return p_deploy, p_harvest, throttle, brake_front * F_BRAKE, brake_rear * F_BRAKE

        def forces(i, v, frac):
            k = i % self.N
            track = [(1.0 - frac) * a[k] + frac * a[k + 1] for a in (kappa_arr, gradient_arr, w_arr)]
            p_deploy, p_harvest, throttle, brake_front, brake_rear = controls(i, frac)
            drive = car.drive_force(v, throttle, p_deploy, p_harvest)
            scale = scales[i // self.N]
            return car.point(v, *track, drive, brake_front, brake_rear, scale, scale)

        v = float(trajectory.v_opt[0])
        s_fine, v_fine = [0.0], [v]
        lap_time = 0.0
        usage_front = usage_rear = 0.0

        for i in range(n):
            for j in range(substeps):
                frac0, frac1 = j / substeps, (j + 1) / substeps
                frac_mid = 0.5 * (frac0 + frac1)
                k1 = forces(i, v, frac0).a_x / v
                v2 = v + 0.5 * h * k1
                k2 = forces(i, v2, frac_mid).a_x / v2
                v3 = v + 0.5 * h * k2
                k3 = forces(i, v3, frac_mid).a_x / v3
                v4 = v + h * k3
                k4 = forces(i, v4, frac1).a_x / v4
                v_next = max(v + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4), V_MIN)
                lap_time += h / (0.5 * (v + v_next))
                v = v_next
                f = forces(i, v, frac1)
                usage_front = max(usage_front, float(f.usage_front))
                usage_rear = max(usage_rear, float(f.usage_rear))
                s_fine.append(self.ds * (i + frac1))
                v_fine.append(v)

        return {
            "lap_time": lap_time,
            "s": np.array(s_fine),
            "v": np.array(v_fine),
            "max_usage_front": usage_front,
            "max_usage_rear": usage_rear,
        }
