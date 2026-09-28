"""
Multi-lap spatial NLP solver for ERS strategy optimization.

Extends the single-lap NLP to a full stint horizon with:
- SOC continuity across lap boundaries
- Per-lap deployment/recovery constraints
- Optional per-lap final SOC floor (charge-sustaining races)
- Optional dynamic tire thermal + degradation state evolution
"""

import time
from typing import Literal

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
            v_limit_profile: Single-lap velocity limit profile from forward-backward solver.
            n_laps: Number of consecutive laps in the horizon.
            initial_soc: Starting SOC for lap 1.
            final_soc_min: Final SOC lower bound at end of last lap.
            is_flying_lap: If True, enforce V_start == V_end over the full horizon.
            per_lap_final_soc_min: Optional SOC floor at each lap boundary.
            lap_grip_scales: Optional per-lap grip multipliers (shape [n_laps]) for scalar mode.
            tire_model: "scalar" (legacy per-lap scale) or "dynamic" (stateful thermal/wear).
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
                v_limit_profile=v_limit_profile,
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
                lap_grip_scales=lap_grip_scales_arr,
                dynamic_enabled=dynamic_enabled,
                tire_thermal_config=tire_thermal_config or TireThermalConfig(),
                tire_compound_config=tire_compound_config or TireCompoundConfig(),
                ambient_temp_c=ambient_temp_c,
                track_temp_c=track_temp_c,
                tire_init_temp_c=tire_init_temp_c,
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

    def _build_and_solve_multi_lap(
        self,
        v_limit_profile: np.ndarray,
        n_laps: int,
        initial_soc: float,
        final_soc_min: float,
        is_flying_lap: bool,
        per_lap_final_soc_min: float | None,
        lap_grip_scales: np.ndarray | None,
        dynamic_enabled: bool,
        tire_thermal_config: TireThermalConfig,
        tire_compound_config: TireCompoundConfig,
        ambient_temp_c: float,
        track_temp_c: float,
        tire_init_temp_c: float,
    ) -> OptimalTrajectory:
        """Build and solve the multi-lap CasADi optimization problem."""
        opti = ca.Opti()

        veh = self.vehicle.vehicle
        ers = self.vehicle.ers

        track_data = self.track.track_data
        gradient_arr = np.resize(track_data.gradient, self.N + 1)
        radius_arr = np.resize(track_data.radius, self.N + 1)

        degradation_enabled = (lap_grip_scales is not None) and (not dynamic_enabled)
        if lap_grip_scales is None:
            lap_grip_scales = np.ones(n_laps, dtype=float)

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

        if dynamic_enabled:
            TSF = opti.variable(n_intervals_total + 1)
            TSR = opti.variable(n_intervals_total + 1)
            TCF = opti.variable(n_intervals_total + 1)
            TCR = opti.variable(n_intervals_total + 1)
            WF = opti.variable(n_intervals_total + 1)
            WR = opti.variable(n_intervals_total + 1)

        if self.collocation_method == CollocationMethod.HERMITE_SIMPSON:
            V_MID = opti.variable(n_intervals_total)
            SOC_MID = opti.variable(n_intervals_total)
            if dynamic_enabled:
                TSF_MID = opti.variable(n_intervals_total)
                TSR_MID = opti.variable(n_intervals_total)
                TCF_MID = opti.variable(n_intervals_total)
                TCR_MID = opti.variable(n_intervals_total)
                WF_MID = opti.variable(n_intervals_total)
                WR_MID = opti.variable(n_intervals_total)

        # =================================================================
        # OBJECTIVE
        # =================================================================
        T_total = 0
        lap_deployment = [ca.MX(0) for _ in range(n_laps)]
        lap_recovery = [ca.MX(0) for _ in range(n_laps)]

        for i in range(n_intervals_total):
            lap_idx = i // self.N
            k = i % self.N
            grip_scale = float(lap_grip_scales[lap_idx])

            if self.collocation_method == CollocationMethod.HERMITE_SIMPSON:
                v_k_safe = ca.fmax(V[i], 1.0)
                v_mid_safe = ca.fmax(V_MID[i], 1.0)
                v_k1_safe = ca.fmax(V[i + 1], 1.0)
                T_total += (self.ds / 6.0) * (1.0 / v_k_safe + 4.0 / v_mid_safe + 1.0 / v_k1_safe)
            else:
                v_avg = 0.5 * (V[i] + V[i + 1])
                v_safe = ca.fmax(v_avg, 1.0)
                T_total += self.ds / v_safe

            self._apply_ers_power_limits(opti, P_DEPLOY[i], V[i], ers)
            opti.subject_to(P_HARVEST[i] <= ers.max_recovery_power)
            opti.subject_to(P_DEPLOY[i] >= 0)
            opti.subject_to(P_HARVEST[i] >= 0)

            if dynamic_enabled:
                q_k = self._compute_dynamic_quantities(
                    V[i],
                    SOC[i],
                    TSF[i],
                    TCF[i],
                    WF[i],
                    TSR[i],
                    TCR[i],
                    WR[i],
                    P_DEPLOY[i],
                    P_HARVEST[i],
                    THROTTLE[i],
                    BRAKE[i],
                    gradient_arr[k],
                    radius_arr[k],
                    veh,
                    ers,
                    tire_thermal_config,
                    tire_compound_config,
                    ambient_temp_c,
                    track_temp_c,
                )
                opti.subject_to(q_k["F_long"] <= q_k["F_long_upper"])
                opti.subject_to(q_k["F_long"] >= q_k["F_long_lower"])
                v_dyn_limit_k = v_limit_profile[k] * ca.sqrt(ca.fmax(q_k["mu_lat_scale_avg"], 0.20)) * 1.02
                opti.subject_to(V[i] <= v_dyn_limit_k)
            else:
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
                q_k = {
                    "dv_ds": dv_ds_k,
                    "dsoc_ds": dsoc_ds_k,
                }
                opti.subject_to(F_prop_k - F_brake_k <= F_grip_k * grip_scale)
                opti.subject_to(F_prop_k - F_brake_k >= -F_grip_k * grip_scale)

            if self.collocation_method == CollocationMethod.EULER:
                opti.subject_to(V[i + 1] == V[i] + self.ds * q_k["dv_ds"])
                opti.subject_to(SOC[i + 1] == SOC[i] + self.ds * q_k["dsoc_ds"])
                if dynamic_enabled:
                    opti.subject_to(TSF[i + 1] == TSF[i] + self.ds * q_k["d_tsf_ds"])
                    opti.subject_to(TSR[i + 1] == TSR[i] + self.ds * q_k["d_tsr_ds"])
                    opti.subject_to(TCF[i + 1] == TCF[i] + self.ds * q_k["d_tcf_ds"])
                    opti.subject_to(TCR[i + 1] == TCR[i] + self.ds * q_k["d_tcr_ds"])
                    opti.subject_to(WF[i + 1] == WF[i] + self.ds * q_k["d_wf_ds"])
                    opti.subject_to(WR[i + 1] == WR[i] + self.ds * q_k["d_wr_ds"])

            elif self.collocation_method == CollocationMethod.TRAPEZOIDAL:
                if dynamic_enabled:
                    q_k1 = self._compute_dynamic_quantities(
                        V[i + 1],
                        SOC[i + 1],
                        TSF[i + 1],
                        TCF[i + 1],
                        WF[i + 1],
                        TSR[i + 1],
                        TCR[i + 1],
                        WR[i + 1],
                        P_DEPLOY[i],
                        P_HARVEST[i],
                        THROTTLE[i],
                        BRAKE[i],
                        gradient_arr[k + 1],
                        radius_arr[k + 1],
                        veh,
                        ers,
                        tire_thermal_config,
                        tire_compound_config,
                        ambient_temp_c,
                        track_temp_c,
                    )
                    if i == n_intervals_total - 1:
                        opti.subject_to(q_k1["F_long"] <= q_k1["F_long_upper"])
                        opti.subject_to(q_k1["F_long"] >= q_k1["F_long_lower"])
                        v_dyn_limit_k1 = (
                            v_limit_profile[-1] * ca.sqrt(ca.fmax(q_k1["mu_lat_scale_avg"], 0.20)) * 1.02
                        )
                        opti.subject_to(V[i + 1] <= v_dyn_limit_k1)
                else:
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
                    q_k1 = {"dv_ds": dv_ds_k1, "dsoc_ds": dsoc_ds_k1}
                    if i == n_intervals_total - 1:
                        opti.subject_to(F_prop_k1 - F_brake_k1 <= F_grip_k1 * grip_scale)
                        opti.subject_to(F_prop_k1 - F_brake_k1 >= -F_grip_k1 * grip_scale)

                opti.subject_to(V[i + 1] == V[i] + (self.ds / 2.0) * (q_k["dv_ds"] + q_k1["dv_ds"]))
                opti.subject_to(SOC[i + 1] == SOC[i] + (self.ds / 2.0) * (q_k["dsoc_ds"] + q_k1["dsoc_ds"]))
                if dynamic_enabled:
                    opti.subject_to(TSF[i + 1] == TSF[i] + (self.ds / 2.0) * (q_k["d_tsf_ds"] + q_k1["d_tsf_ds"]))
                    opti.subject_to(TSR[i + 1] == TSR[i] + (self.ds / 2.0) * (q_k["d_tsr_ds"] + q_k1["d_tsr_ds"]))
                    opti.subject_to(TCF[i + 1] == TCF[i] + (self.ds / 2.0) * (q_k["d_tcf_ds"] + q_k1["d_tcf_ds"]))
                    opti.subject_to(TCR[i + 1] == TCR[i] + (self.ds / 2.0) * (q_k["d_tcr_ds"] + q_k1["d_tcr_ds"]))
                    opti.subject_to(WF[i + 1] == WF[i] + (self.ds / 2.0) * (q_k["d_wf_ds"] + q_k1["d_wf_ds"]))
                    opti.subject_to(WR[i + 1] == WR[i] + (self.ds / 2.0) * (q_k["d_wr_ds"] + q_k1["d_wr_ds"]))

            elif self.collocation_method == CollocationMethod.HERMITE_SIMPSON:
                if dynamic_enabled:
                    q_k1 = self._compute_dynamic_quantities(
                        V[i + 1],
                        SOC[i + 1],
                        TSF[i + 1],
                        TCF[i + 1],
                        WF[i + 1],
                        TSR[i + 1],
                        TCR[i + 1],
                        WR[i + 1],
                        P_DEPLOY[i],
                        P_HARVEST[i],
                        THROTTLE[i],
                        BRAKE[i],
                        gradient_arr[k + 1],
                        radius_arr[k + 1],
                        veh,
                        ers,
                        tire_thermal_config,
                        tire_compound_config,
                        ambient_temp_c,
                        track_temp_c,
                    )
                    if i == n_intervals_total - 1:
                        opti.subject_to(q_k1["F_long"] <= q_k1["F_long_upper"])
                        opti.subject_to(q_k1["F_long"] >= q_k1["F_long_lower"])
                        v_dyn_limit_k1 = (
                            v_limit_profile[-1] * ca.sqrt(ca.fmax(q_k1["mu_lat_scale_avg"], 0.20)) * 1.02
                        )
                        opti.subject_to(V[i + 1] <= v_dyn_limit_k1)
                else:
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
                    q_k1 = {"dv_ds": dv_ds_k1, "dsoc_ds": dsoc_ds_k1}
                    if i == n_intervals_total - 1:
                        opti.subject_to(F_prop_k1 - F_brake_k1 <= F_grip_k1 * grip_scale)
                        opti.subject_to(F_prop_k1 - F_brake_k1 >= -F_grip_k1 * grip_scale)

                v_mid_hermite = 0.5 * (V[i] + V[i + 1]) + (self.ds / 8.0) * (q_k["dv_ds"] - q_k1["dv_ds"])
                soc_mid_hermite = 0.5 * (SOC[i] + SOC[i + 1]) + (self.ds / 8.0) * (q_k["dsoc_ds"] - q_k1["dsoc_ds"])
                opti.subject_to(V_MID[i] == v_mid_hermite)
                opti.subject_to(SOC_MID[i] == soc_mid_hermite)

                grad_mid = 0.5 * (gradient_arr[k] + gradient_arr[k + 1])
                radius_mid = 0.5 * (radius_arr[k] + radius_arr[k + 1])

                if dynamic_enabled:
                    tsf_mid_hermite = 0.5 * (TSF[i] + TSF[i + 1]) + (self.ds / 8.0) * (
                        q_k["d_tsf_ds"] - q_k1["d_tsf_ds"]
                    )
                    tsr_mid_hermite = 0.5 * (TSR[i] + TSR[i + 1]) + (self.ds / 8.0) * (
                        q_k["d_tsr_ds"] - q_k1["d_tsr_ds"]
                    )
                    tcf_mid_hermite = 0.5 * (TCF[i] + TCF[i + 1]) + (self.ds / 8.0) * (
                        q_k["d_tcf_ds"] - q_k1["d_tcf_ds"]
                    )
                    tcr_mid_hermite = 0.5 * (TCR[i] + TCR[i + 1]) + (self.ds / 8.0) * (
                        q_k["d_tcr_ds"] - q_k1["d_tcr_ds"]
                    )
                    wf_mid_hermite = 0.5 * (WF[i] + WF[i + 1]) + (self.ds / 8.0) * (
                        q_k["d_wf_ds"] - q_k1["d_wf_ds"]
                    )
                    wr_mid_hermite = 0.5 * (WR[i] + WR[i + 1]) + (self.ds / 8.0) * (
                        q_k["d_wr_ds"] - q_k1["d_wr_ds"]
                    )

                    opti.subject_to(TSF_MID[i] == tsf_mid_hermite)
                    opti.subject_to(TSR_MID[i] == tsr_mid_hermite)
                    opti.subject_to(TCF_MID[i] == tcf_mid_hermite)
                    opti.subject_to(TCR_MID[i] == tcr_mid_hermite)
                    opti.subject_to(WF_MID[i] == wf_mid_hermite)
                    opti.subject_to(WR_MID[i] == wr_mid_hermite)

                    q_mid = self._compute_dynamic_quantities(
                        V_MID[i],
                        SOC_MID[i],
                        TSF_MID[i],
                        TCF_MID[i],
                        WF_MID[i],
                        TSR_MID[i],
                        TCR_MID[i],
                        WR_MID[i],
                        P_DEPLOY[i],
                        P_HARVEST[i],
                        THROTTLE[i],
                        BRAKE[i],
                        grad_mid,
                        radius_mid,
                        veh,
                        ers,
                        tire_thermal_config,
                        tire_compound_config,
                        ambient_temp_c,
                        track_temp_c,
                    )
                    opti.subject_to(q_mid["F_long"] <= q_mid["F_long_upper"])
                    opti.subject_to(q_mid["F_long"] >= q_mid["F_long_lower"])
                    v_dyn_limit_mid = (
                        0.5 * (v_limit_profile[k] + v_limit_profile[k + 1])
                        * ca.sqrt(ca.fmax(q_mid["mu_lat_scale_avg"], 0.20))
                        * 1.02
                    )
                    opti.subject_to(V_MID[i] <= v_dyn_limit_mid)
                else:
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
                    q_mid = {"dv_ds": dv_ds_mid, "dsoc_ds": dsoc_ds_mid}
                    opti.subject_to(F_prop_mid - F_brake_mid <= F_grip_mid * grip_scale)
                    opti.subject_to(F_prop_mid - F_brake_mid >= -F_grip_mid * grip_scale)

                opti.subject_to(V[i + 1] == V[i] + (self.ds / 6.0) * (q_k["dv_ds"] + 4.0 * q_mid["dv_ds"] + q_k1["dv_ds"]))
                opti.subject_to(
                    SOC[i + 1] == SOC[i] + (self.ds / 6.0) * (q_k["dsoc_ds"] + 4.0 * q_mid["dsoc_ds"] + q_k1["dsoc_ds"])
                )
                if dynamic_enabled:
                    opti.subject_to(
                        TSF[i + 1] == TSF[i] + (self.ds / 6.0) * (q_k["d_tsf_ds"] + 4.0 * q_mid["d_tsf_ds"] + q_k1["d_tsf_ds"])
                    )
                    opti.subject_to(
                        TSR[i + 1] == TSR[i] + (self.ds / 6.0) * (q_k["d_tsr_ds"] + 4.0 * q_mid["d_tsr_ds"] + q_k1["d_tsr_ds"])
                    )
                    opti.subject_to(
                        TCF[i + 1] == TCF[i] + (self.ds / 6.0) * (q_k["d_tcf_ds"] + 4.0 * q_mid["d_tcf_ds"] + q_k1["d_tcf_ds"])
                    )
                    opti.subject_to(
                        TCR[i + 1] == TCR[i] + (self.ds / 6.0) * (q_k["d_tcr_ds"] + 4.0 * q_mid["d_tcr_ds"] + q_k1["d_tcr_ds"])
                    )
                    opti.subject_to(
                        WF[i + 1] == WF[i] + (self.ds / 6.0) * (q_k["d_wf_ds"] + 4.0 * q_mid["d_wf_ds"] + q_k1["d_wf_ds"])
                    )
                    opti.subject_to(
                        WR[i + 1] == WR[i] + (self.ds / 6.0) * (q_k["d_wr_ds"] + 4.0 * q_mid["d_wr_ds"] + q_k1["d_wr_ds"])
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

        v_limit_nodes = np.concatenate(
            [
                *(v_limit_profile[:-1] * np.sqrt(lap_grip_scales[lap_idx]) for lap_idx in range(n_laps)),
                [v_limit_profile[-1] * np.sqrt(lap_grip_scales[-1])],
            ]
        )
        v_limit_scale = 1.00 if degradation_enabled else 1.02

        opti.subject_to(opti.bounded(ers.min_soc, SOC, ers.max_soc))
        opti.subject_to(opti.bounded(0, THROTTLE, 1))
        opti.subject_to(opti.bounded(0, BRAKE, 1))

        if dynamic_enabled:
            opti.subject_to(opti.bounded(5.0, V, v_limit_nodes * 1.10))
            opti.subject_to(opti.bounded(20.0, TSF, 220.0))
            opti.subject_to(opti.bounded(20.0, TSR, 220.0))
            opti.subject_to(opti.bounded(20.0, TCF, 220.0))
            opti.subject_to(opti.bounded(20.0, TCR, 220.0))
            opti.subject_to(opti.bounded(0.0, WF, 1.0))
            opti.subject_to(opti.bounded(0.0, WR, 1.0))
            opti.subject_to(TSF[0] == tire_init_temp_c)
            opti.subject_to(TSR[0] == tire_init_temp_c)
            opti.subject_to(TCF[0] == tire_init_temp_c)
            opti.subject_to(TCR[0] == tire_init_temp_c)
            opti.subject_to(WF[0] == 0.0)
            opti.subject_to(WR[0] == 0.0)
        else:
            opti.subject_to(opti.bounded(5.0, V, v_limit_nodes * v_limit_scale))

        if is_flying_lap:
            opti.subject_to(V[0] == V[-1])
        else:
            if dynamic_enabled:
                # Allow a cold-tire start below the nominal single-lap limit.
                opti.subject_to(V[0] <= v_limit_profile[0])
            else:
                opti.subject_to(V[0] == v_limit_profile[0])

        if self.collocation_method == CollocationMethod.HERMITE_SIMPSON:
            v_limit_mid_single = 0.5 * (v_limit_profile[:-1] + v_limit_profile[1:])
            v_limit_mid = np.concatenate(
                [v_limit_mid_single * np.sqrt(lap_grip_scales[lap_idx]) for lap_idx in range(n_laps)]
            )
            opti.subject_to(opti.bounded(ers.min_soc, SOC_MID, ers.max_soc))
            if dynamic_enabled:
                opti.subject_to(opti.bounded(5.0, V_MID, v_limit_mid * 1.10))
                opti.subject_to(opti.bounded(20.0, TSF_MID, 220.0))
                opti.subject_to(opti.bounded(20.0, TSR_MID, 220.0))
                opti.subject_to(opti.bounded(20.0, TCF_MID, 220.0))
                opti.subject_to(opti.bounded(20.0, TCR_MID, 220.0))
                opti.subject_to(opti.bounded(0.0, WF_MID, 1.0))
                opti.subject_to(opti.bounded(0.0, WR_MID, 1.0))
            else:
                opti.subject_to(opti.bounded(5.0, V_MID, v_limit_mid * v_limit_scale))

        # =================================================================
        # SOLVE
        # =================================================================
        self._configure_solver(opti)

        v_guess = v_limit_nodes * 0.95
        soc_target = max(final_soc_min, per_lap_final_soc_min or ers.min_soc)
        soc_guess = np.linspace(initial_soc, soc_target, n_intervals_total + 1)
        opti.set_initial(V, v_guess)
        opti.set_initial(SOC, soc_guess)
        opti.set_initial(THROTTLE, np.ones(n_intervals_total) * 0.8)
        opti.set_initial(BRAKE, np.zeros(n_intervals_total))
        opti.set_initial(P_DEPLOY, np.zeros(n_intervals_total))
        opti.set_initial(P_HARVEST, np.zeros(n_intervals_total))

        if dynamic_enabled:
            opti.set_initial(TSF, np.ones(n_intervals_total + 1) * tire_init_temp_c)
            opti.set_initial(TSR, np.ones(n_intervals_total + 1) * tire_init_temp_c)
            opti.set_initial(TCF, np.ones(n_intervals_total + 1) * tire_init_temp_c)
            opti.set_initial(TCR, np.ones(n_intervals_total + 1) * tire_init_temp_c)
            wear_guess = np.linspace(0.0, 0.15, n_intervals_total + 1)
            opti.set_initial(WF, wear_guess)
            opti.set_initial(WR, wear_guess)

        if self.collocation_method == CollocationMethod.HERMITE_SIMPSON:
            v_mid_guess = v_limit_mid * 0.95
            soc_mid_guess = 0.5 * (soc_guess[:-1] + soc_guess[1:])
            opti.set_initial(V_MID, v_mid_guess)
            opti.set_initial(SOC_MID, soc_mid_guess)
            if dynamic_enabled:
                temp_mid_guess = np.ones(n_intervals_total) * tire_init_temp_c
                wear_mid_guess = 0.5 * (wear_guess[:-1] + wear_guess[1:])
                opti.set_initial(TSF_MID, temp_mid_guess)
                opti.set_initial(TSR_MID, temp_mid_guess)
                opti.set_initial(TCF_MID, temp_mid_guess)
                opti.set_initial(TCR_MID, temp_mid_guess)
                opti.set_initial(WF_MID, wear_mid_guess)
                opti.set_initial(WR_MID, wear_mid_guess)

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

        tire_temp_surface_front = None
        tire_temp_surface_rear = None
        tire_temp_core_front = None
        tire_temp_core_rear = None
        tire_wear_front = None
        tire_wear_rear = None
        tire_mu_scale_front = None
        tire_mu_scale_rear = None

        if dynamic_enabled:
            tire_temp_surface_front = np.asarray(sol.value(TSF), dtype=float)
            tire_temp_surface_rear = np.asarray(sol.value(TSR), dtype=float)
            tire_temp_core_front = np.asarray(sol.value(TCF), dtype=float)
            tire_temp_core_rear = np.asarray(sol.value(TCR), dtype=float)
            tire_wear_front = np.asarray(sol.value(WF), dtype=float)
            tire_wear_rear = np.asarray(sol.value(WR), dtype=float)
            tire_mu_scale_front = np.array(
                [
                    mu_scale_np(tire_temp_core_front[idx], tire_wear_front[idx], tire_thermal_config, tire_compound_config)
                    for idx in range(tire_temp_core_front.shape[0])
                ],
                dtype=float,
            )
            tire_mu_scale_rear = np.array(
                [
                    mu_scale_np(tire_temp_core_rear[idx], tire_wear_rear[idx], tire_thermal_config, tire_compound_config)
                    for idx in range(tire_temp_core_rear.shape[0])
                ],
                dtype=float,
            )

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
            lap_grip_scales=None if dynamic_enabled else lap_grip_scales,
            tire_temp_surface_front=tire_temp_surface_front,
            tire_temp_surface_rear=tire_temp_surface_rear,
            tire_temp_core_front=tire_temp_core_front,
            tire_temp_core_rear=tire_temp_core_rear,
            tire_wear_front=tire_wear_front,
            tire_wear_rear=tire_wear_rear,
            tire_mu_scale_front=tire_mu_scale_front,
            tire_mu_scale_rear=tire_mu_scale_rear,
        )
