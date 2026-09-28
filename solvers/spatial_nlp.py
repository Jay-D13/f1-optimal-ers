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
        tyre temperatures and wear            (optional dynamic tyre model, multi-lap)
"""

import time
from dataclasses import dataclass
from enum import Enum
from typing import Literal, Tuple

import casadi as ca
import numpy as np

from config import TireCompoundConfig, TireThermalConfig
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


@dataclass
class DynamicTireSettings:
    """Inputs of the dynamic tyre model, which adds tyre temperature and wear states to the NLP."""
    thermal: TireThermalConfig
    compound: TireCompoundConfig
    ambient_temp_c: float
    track_temp_c: float
    init_temp_c: float


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
        s_track = getattr(self.track.track_data, "s", None)
        if s_track is not None and len(values) == len(s_track):
            # Track points stop short of the finish line, so wrap around the lap
            return np.interp(self.s_grid, s_track, values, period=self.track.total_length)
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

    def _compute_axle_normal_loads(self, v, gradient, veh):
        """Estimate front/rear axle normal loads with static + aero contribution."""
        wb = veh.wheelbase
        front_static_ratio = veh.lr / wb
        rear_static_ratio = veh.lf / wb

        F_weight = veh.mass * veh.g * ca.cos(gradient)
        q = 0.5 * veh.rho_air * v**2
        F_aero_f = q * veh.c_z_a_f
        F_aero_r = q * veh.c_z_a_r

        F_z_f = front_static_ratio * F_weight + F_aero_f
        F_z_r = rear_static_ratio * F_weight + F_aero_r
        return F_z_f, F_z_r

    def _split_lateral_force_by_load(self, F_lat_total, F_z_f, F_z_r):
        """Split lateral demand front/rear proportionally to axle normal loads."""
        F_z_sum = ca.fmax(F_z_f + F_z_r, 1.0)
        front_ratio = F_z_f / F_z_sum
        rear_ratio = F_z_r / F_z_sum
        return front_ratio * F_lat_total, rear_ratio * F_lat_total

    def _compute_load_sensitive_mu(self, F_z, mu0: float, dmu_dfz: float):
        tires = self.vehicle.tires
        mu = mu0 + dmu_dfz * (F_z - tires.fz_0)
        return ca.fmax(mu, 0.5)

    def _compute_axle_force_potentials(self, F_z_f, F_z_r, mu_scale_f, mu_scale_r):
        """Return axle force limits (Fx/Fy potentials) including load sensitivity."""
        tires = self.vehicle.tires

        mu_x_f = self._compute_load_sensitive_mu(F_z_f, tires.mux_f, tires.dmux_dfz_f) * mu_scale_f
        mu_y_f = self._compute_load_sensitive_mu(F_z_f, tires.muy_f, tires.dmuy_dfz_f) * mu_scale_f
        mu_x_r = self._compute_load_sensitive_mu(F_z_r, tires.mux_r, tires.dmux_dfz_r) * mu_scale_r
        mu_y_r = self._compute_load_sensitive_mu(F_z_r, tires.muy_r, tires.dmuy_dfz_r) * mu_scale_r

        F_x_max_f = ca.fmax(mu_x_f * F_z_f, 1.0)
        F_y_max_f = ca.fmax(mu_y_f * F_z_f, 1.0)
        F_x_max_r = ca.fmax(mu_x_r * F_z_r, 1.0)
        F_y_max_r = ca.fmax(mu_y_r * F_z_r, 1.0)
        return F_x_max_f, F_x_max_r, F_y_max_f, F_y_max_r

    def _split_longitudinal_force(self, F_long):
        """
        Split net longitudinal tire force between axles.
        Positive force (traction) is rear-biased, braking uses both axles with front bias.
        """
        sigma_acc = 0.5 * (1.0 + ca.tanh(F_long / 1000.0))
        sigma_brk = 1.0 - sigma_acc

        # F1 traction is heavily rear-biased; braking has front bias.
        front_ratio = sigma_acc * 0.05 + sigma_brk * 0.60
        rear_ratio = sigma_acc * 0.95 + sigma_brk * 0.40
        return front_ratio * F_long, rear_ratio * F_long

    def _compute_dynamic_quantities(
        self,
        v,
        soc,
        tsf,
        tcf,
        wf,
        tsr,
        tcr,
        wr,
        p_deploy,
        p_harvest,
        throttle,
        brake,
        gradient,
        radius,
        veh,
        ers,
        thermal_cfg: TireThermalConfig,
        compound_cfg: TireCompoundConfig,
        ambient_temp_c: float,
        track_temp_c: float,
    ):
        """
        Compute dynamics and dynamic tire constraints at one collocation state.
        """
        dv_ds, dsoc_ds, F_prop, F_brake, _ = self._compute_derivatives(
            v,
            soc,
            p_deploy,
            p_harvest,
            throttle,
            brake,
            gradient,
            radius,
            veh,
            ers,
        )
        v_safe = ca.fmax(v, 5.0)

        # Axle loads and lateral demand split
        F_z_f, F_z_r = self._compute_axle_normal_loads(v, gradient, veh)
        safe_radius = ca.fmax(ca.fabs(radius), 15.0)
        F_lat_total = veh.mass * v * v / safe_radius
        F_y_f, F_y_r = self._split_lateral_force_by_load(F_lat_total, F_z_f, F_z_r)

        mu_scale_f = mu_scale_ca(tcf, wf, thermal_cfg, compound_cfg)
        mu_scale_r = mu_scale_ca(tcr, wr, thermal_cfg, compound_cfg)

        F_x_max_f, F_x_max_r, F_y_max_f, F_y_max_r = self._compute_axle_force_potentials(
            F_z_f, F_z_r, mu_scale_f, mu_scale_r
        )

        F_long = F_prop - F_brake
        F_x_f, F_x_r = self._split_longitudinal_force(F_long)

        p = thermal_cfg.utilization_ellipse_p
        u_f = utilization_ca(F_x_f, F_y_f, F_x_max_f, F_y_max_f, p)
        u_r = utilization_ca(F_x_r, F_y_r, F_x_max_r, F_y_max_r, p)

        q_gen_f = heat_generation_ca(F_z_f, v_safe, u_f, thermal_cfg, compound_cfg)
        q_gen_r = heat_generation_ca(F_z_r, v_safe, u_r, thermal_cfg, compound_cfg)

        d_tsf_dt = surface_temp_rate_ca(q_gen_f, tsf, tcf, ambient_temp_c, track_temp_c, thermal_cfg)
        d_tsr_dt = surface_temp_rate_ca(q_gen_r, tsr, tcr, ambient_temp_c, track_temp_c, thermal_cfg)

        q_sc_f = thermal_cfg.k_surface_core * (tsf - tcf)
        q_sc_r = thermal_cfg.k_surface_core * (tsr - tcr)
        d_tcf_dt = core_temp_rate_ca(q_sc_f, tcf, ambient_temp_c, thermal_cfg)
        d_tcr_dt = core_temp_rate_ca(q_sc_r, tcr, ambient_temp_c, thermal_cfg)

        d_wf_dt = wear_rate_ca(u_f, tcf, F_z_f, thermal_cfg, compound_cfg)
        d_wr_dt = wear_rate_ca(u_r, tcr, F_z_r, thermal_cfg, compound_cfg)

        d_tsf_ds = d_tsf_dt / v_safe
        d_tsr_ds = d_tsr_dt / v_safe
        d_tcf_ds = d_tcf_dt / v_safe
        d_tcr_ds = d_tcr_dt / v_safe
        d_wf_ds = d_wf_dt / v_safe
        d_wr_ds = d_wr_dt / v_safe

        ratio_y_f = ca.fmin(ca.fabs(F_y_f) / ca.fmax(F_y_max_f, 1e-6), 0.999)
        ratio_y_r = ca.fmin(ca.fabs(F_y_r) / ca.fmax(F_y_max_r, 1e-6), 0.999)
        avail_ratio_f = ca.power(ca.fmax(1.0 - ca.power(ratio_y_f, p), 0.0), 1.0 / p)
        avail_ratio_r = ca.power(ca.fmax(1.0 - ca.power(ratio_y_r, p), 0.0), 1.0 / p)
        F_x_avail_f = F_x_max_f * avail_ratio_f
        F_x_avail_r = F_x_max_r * avail_ratio_r

        F_long_upper = 0.10 * F_x_avail_f + F_x_avail_r
        F_long_lower = -(F_x_avail_f + F_x_avail_r)

        mu_lat_scale_avg = (mu_scale_f * F_z_f + mu_scale_r * F_z_r) / ca.fmax(F_z_f + F_z_r, 1.0)

        return {
            "dv_ds": dv_ds,
            "dsoc_ds": dsoc_ds,
            "d_tsf_ds": d_tsf_ds,
            "d_tsr_ds": d_tsr_ds,
            "d_tcf_ds": d_tcf_ds,
            "d_tcr_ds": d_tcr_ds,
            "d_wf_ds": d_wf_ds,
            "d_wr_ds": d_wr_ds,
            "F_long": F_long,
            "F_long_upper": F_long_upper,
            "F_long_lower": F_long_lower,
            "mu_lat_scale_avg": mu_lat_scale_avg,
        }

    def _point_dynamics(self, opti, x, u, gradient, radius, v_limit, grip_scale, tire, add_constraints):
        """
        State derivatives d/ds at one collocation point, keyed like the states.

        With add_constraints, also adds the point's grip limits: the friction limit on the net
        longitudinal force, or with dynamic tyres the per-axle limits and a cornering-speed limit
        scaled by the tyres' current grip.
        """
        veh = self.vehicle.vehicle
        ers = self.vehicle.ers
        p_deploy, p_harvest, throttle, brake = u

        if tire is None:
            dv_ds, dsoc_ds, F_prop, F_brake, F_grip = self._compute_derivatives(
                x["V"], x["SOC"], p_deploy, p_harvest, throttle, brake, gradient, radius, veh, ers
            )
            if add_constraints:
                self._add_grip_limit(opti, F_prop, F_brake, F_grip, grip_scale)
            return {"V": dv_ds, "SOC": dsoc_ds}

        q = self._compute_dynamic_quantities(
            x["V"], x["SOC"], x["TSF"], x["TCF"], x["WF"], x["TSR"], x["TCR"], x["WR"],
            p_deploy, p_harvest, throttle, brake, gradient, radius, veh, ers,
            tire.thermal, tire.compound, tire.ambient_temp_c, tire.track_temp_c,
        )
        if add_constraints:
            opti.subject_to(q["F_long"] <= q["F_long_upper"])
            opti.subject_to(q["F_long"] >= q["F_long_lower"])
            opti.subject_to(x["V"] <= v_limit * ca.sqrt(ca.fmax(q["mu_lat_scale_avg"], 0.20)) * 1.02)
        return {
            "V": q["dv_ds"], "SOC": q["dsoc_ds"],
            "TSF": q["d_tsf_ds"], "TCF": q["d_tcf_ds"], "WF": q["d_wf_ds"],
            "TSR": q["d_tsr_ds"], "TCR": q["d_tcr_ds"], "WR": q["d_wr_ds"],
        }

    def _build_and_solve(
        self,
        v_limit_profile: np.ndarray,
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
        With `tire`, tyre temperatures and wear are states that set the grip (dynamic tyre model).
        """
        opti = ca.Opti()

        # Get vehicle parameters
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

        # Controls (piecewise constant over intervals)
        P_DEPLOY = opti.variable(n)       # ERS discharge power (≥0, units of POWER_SCALE)
        P_HARVEST = opti.variable(n)      # ERS recovery power (≥0, units of POWER_SCALE)
        THROTTLE = opti.variable(n)       # Throttle position (0-1)
        BRAKE = opti.variable(n)          # Brake position (0-1)

        # For Hermite-Simpson: midpoint states
        if method == CollocationMethod.HERMITE_SIMPSON:
            X_MID = {name: opti.variable(n) for name in X}
            V_MID = X_MID["V"]

        # =================================================================
        # OBJECTIVE, DYNAMICS & PHYSICS
        # =================================================================

        T_total = 0

        for i in range(n):
            k = i % self.N  # Node index within the lap
            grip_scale = float(lap_grip_scales[i // self.N])
            u = (P_DEPLOY[i] * PS, P_HARVEST[i] * PS, THROTTLE[i], BRAKE[i])  # Powers in W

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

            # --- Derivatives and grip limits at node k ---
            f_k = self._point_dynamics(
                opti, {name: x[i] for name, x in X.items()}, u,
                gradient_arr[k], radius_arr[k], v_limit_profile[k], grip_scale, tire, add_constraints=True,
            )

            # --- Apply collocation constraints ---
            # dt: time over the interval, integrated like the SOC (same 5 m/s speed floor as the dynamics)
            if method == CollocationMethod.EULER:
                # Explicit Euler: x[k+1] = x[k] + h * f(x[k])
                for name, x in X.items():
                    opti.subject_to(x[i + 1] == x[i] + self.ds * f_k[name])
                dt = self.ds / ca.fmax(V[i], 5.0)

            else:
                # Derivatives at node k+1 (same control). Its grip limits are only added for the
                # final node; the others are added when they become node k.
                f_k1 = self._point_dynamics(
                    opti, {name: x[i + 1] for name, x in X.items()}, u,
                    gradient_arr[k + 1], radius_arr[k + 1], v_limit_profile[k + 1], grip_scale, tire,
                    add_constraints=(i == n - 1),
                )

                if method == CollocationMethod.TRAPEZOIDAL:
                    # Trapezoidal: x[k+1] = x[k] + (h/2) * (f(x[k]) + f(x[k+1]))
                    for name, x in X.items():
                        opti.subject_to(x[i + 1] == x[i] + (self.ds / 2.0) * (f_k[name] + f_k1[name]))
                    dt = (self.ds / 2.0) * (1.0 / ca.fmax(V[i], 5.0) + 1.0 / ca.fmax(V[i + 1], 5.0))

                else:
                    # 1. Midpoint states from Hermite interpolation
                    # x_mid = (x[k] + x[k+1])/2 + (h/8) * (f[k] - f[k+1])
                    for name, x in X.items():
                        opti.subject_to(
                            X_MID[name][i] == 0.5 * (x[i] + x[i + 1]) + (self.ds / 8.0) * (f_k[name] - f_k1[name])
                        )

                    # 2. Derivatives and grip limits at the midpoint
                    f_mid = self._point_dynamics(
                        opti, {name: x[i] for name, x in X_MID.items()}, u,
                        0.5 * (gradient_arr[k] + gradient_arr[k + 1]),
                        0.5 * (radius_arr[k] + radius_arr[k + 1]),
                        0.5 * (v_limit_profile[k] + v_limit_profile[k + 1]),
                        grip_scale, tire, add_constraints=True,
                    )

                    # 3. Simpson quadrature: x[k+1] = x[k] + (h/6) * (f[k] + 4*f_mid + f[k+1])
                    for name, x in X.items():
                        opti.subject_to(
                            x[i + 1] == x[i] + (self.ds / 6.0) * (f_k[name] + 4.0 * f_mid[name] + f_k1[name])
                        )
                    dt = (self.ds / 6.0) * (
                        1.0 / ca.fmax(V[i], 5.0) + 4.0 / ca.fmax(V_MID[i], 5.0) + 1.0 / ca.fmax(V[i + 1], 5.0)
                    )

            # --- Energy totals: same quadrature as the SOC, so the battery bookkeeping is exact ---
            opti.subject_to(E_DEPLOY[i + 1] == E_DEPLOY[i] + u[0] * dt / 1e6)
            opti.subject_to(E_RECOVER[i + 1] == E_RECOVER[i] + u[1] * dt / 1e6)

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

        # State bounds (a worn tyre lowers the speed limit by sqrt(grip scale)). Dynamic tyres
        # set their own cornering limit at each point, so the envelope is only a loose bound.
        v_limit_nodes = np.concatenate(
            [v_limit_profile[:-1] * np.sqrt(scale) for scale in lap_grip_scales]
            + [v_limit_profile[-1:] * np.sqrt(lap_grip_scales[-1])]
        )
        if tire is not None:
            v_limit_scale = 1.10
        else:
            v_limit_scale = 1.00 if degradation_enabled else 1.02
        opti.subject_to(opti.bounded(ers.min_soc, SOC, ers.max_soc))
        opti.subject_to(opti.bounded(5.0, V, v_limit_nodes * v_limit_scale))

        if tire is not None:
            for name in TIRE_STATES:
                wear = name in _WEAR_STATES
                opti.subject_to(opti.bounded(0.0, X[name], 1.0) if wear else opti.bounded(20.0, X[name], 220.0))
                opti.subject_to(X[name][0] == (0.0 if wear else tire.init_temp_c))

        # Control bounds
        opti.subject_to(P_DEPLOY >= 0)
        opti.subject_to(opti.bounded(0, P_HARVEST, ers.max_recovery_power / PS))
        opti.subject_to(opti.bounded(0, THROTTLE, 1))
        opti.subject_to(opti.bounded(0, BRAKE, 1))

        # Velocity boundary condition
        if is_flying_lap:
            opti.subject_to(V[0] == V[-1])
        elif tire is not None:
            # Allow a cold-tyre start below the nominal single-lap limit
            opti.subject_to(V[0] <= v_limit_profile[0])
        else:
            opti.subject_to(V[0] == v_limit_profile[0])

        # Midpoint bounds for Hermite-Simpson
        if method == CollocationMethod.HERMITE_SIMPSON:
            v_limit_mid = np.concatenate(
                [0.5 * (v_limit_profile[:-1] + v_limit_profile[1:]) * np.sqrt(scale) for scale in lap_grip_scales]
            )
            opti.subject_to(opti.bounded(5.0, V_MID, v_limit_mid * v_limit_scale))
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

        # Initial guess: follow the speed limit closely, drain the battery linearly, warm tyres wearing slowly
        soc_target = max(final_soc_min, per_lap_final_soc_min or ers.min_soc)
        guesses = {"V": v_limit_nodes * 0.95, "SOC": np.linspace(initial_soc, soc_target, n + 1)}
        if tire is not None:
            for name in TIRE_STATES:
                wear = name in _WEAR_STATES
                guesses[name] = np.linspace(0.0, 0.15, n + 1) if wear else np.full(n + 1, tire.init_temp_c)
        for name, guess in guesses.items():
            opti.set_initial(X[name], guess)
            if method == CollocationMethod.HERMITE_SIMPSON:
                opti.set_initial(X_MID[name], 0.5 * (guess[:-1] + guess[1:]))
        opti.set_initial(THROTTLE, 0.8)

        variables = dict(
            X, E_DEPLOY=E_DEPLOY, E_RECOVER=E_RECOVER,
            P_DEPLOY=P_DEPLOY, P_HARVEST=P_HARVEST, THROTTLE=THROTTLE, BRAKE=BRAKE,
        )

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

    def _extract_trajectory(self, sol, variables, status, n_laps, lap_grip_scales, tire=None):
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
