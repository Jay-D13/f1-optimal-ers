from __future__ import annotations

import casadi as ca
import numpy as np

from config import TireCompoundConfig, TireThermalConfig


def utilization_np(F_x: float, F_y: float, F_x_max: float, F_y_max: float, p: float) -> float:
    fx_ratio = abs(F_x) / max(F_x_max, 1e-6)
    fy_ratio = abs(F_y) / max(F_y_max, 1e-6)
    u_raw = (fx_ratio**p + fy_ratio**p) ** (1.0 / p)
    return float(np.clip(u_raw, 0.0, 1.0))


def utilization_ca(F_x, F_y, F_x_max, F_y_max, p: float):
    fx_ratio = ca.fabs(F_x) / ca.fmax(F_x_max, 1e-6)
    fy_ratio = ca.fabs(F_y) / ca.fmax(F_y_max, 1e-6)
    u_raw = ca.power(ca.power(fx_ratio, p) + ca.power(fy_ratio, p), 1.0 / p)
    return ca.fmin(1.0, ca.fmax(0.0, u_raw))


def heat_generation_np(F_z: float, v: float, utilization: float, cfg: TireThermalConfig, cmp: TireCompoundConfig) -> float:
    return cfg.heat_generation_coeff * cmp.heat_generation_scale * F_z * max(v, 0.0) * (
        utilization**cfg.heat_generation_utilization_exp
    )


def heat_generation_ca(F_z, v, utilization, cfg: TireThermalConfig, cmp: TireCompoundConfig):
    return (
        cfg.heat_generation_coeff
        * cmp.heat_generation_scale
        * F_z
        * ca.fmax(v, 0.0)
        * ca.power(utilization, cfg.heat_generation_utilization_exp)
    )


def surface_temp_rate_np(
    q_gen: float,
    t_surface: float,
    t_core: float,
    t_air: float,
    t_track: float,
    cfg: TireThermalConfig,
) -> float:
    return (
        q_gen
        - cfg.h_air * (t_surface - t_air)
        - cfg.h_track * (t_surface - t_track)
        - cfg.k_surface_core * (t_surface - t_core)
    ) / cfg.surface_heat_capacity


def surface_temp_rate_ca(q_gen, t_surface, t_core, t_air: float, t_track: float, cfg: TireThermalConfig):
    return (
        q_gen
        - cfg.h_air * (t_surface - t_air)
        - cfg.h_track * (t_surface - t_track)
        - cfg.k_surface_core * (t_surface - t_core)
    ) / cfg.surface_heat_capacity


def core_temp_rate_np(q_sc: float, t_core: float, t_air: float, cfg: TireThermalConfig) -> float:
    return (q_sc - cfg.h_rim * (t_core - t_air)) / cfg.core_heat_capacity


def core_temp_rate_ca(q_sc, t_core, t_air: float, cfg: TireThermalConfig):
    return (q_sc - cfg.h_rim * (t_core - t_air)) / cfg.core_heat_capacity


def wear_rate_np(utilization: float, t_core: float, F_z: float, cfg: TireThermalConfig, cmp: TireCompoundConfig) -> float:
    rate = (
        cfg.wear_coeff
        * cmp.wear_rate_scale
        * (utilization**cfg.wear_utilization_exp)
        * np.exp(cfg.wear_temp_exp_coeff * (t_core - cfg.wear_temp_ref_c))
        * (max(F_z, 1e-6) / cfg.wear_fz_ref_n) ** cfg.wear_load_exp
    )
    return float(max(rate, 0.0))


def wear_rate_ca(utilization, t_core, F_z, cfg: TireThermalConfig, cmp: TireCompoundConfig):
    rate = (
        cfg.wear_coeff
        * cmp.wear_rate_scale
        * ca.power(utilization, cfg.wear_utilization_exp)
        * ca.exp(cfg.wear_temp_exp_coeff * (t_core - cfg.wear_temp_ref_c))
        * ca.power(ca.fmax(F_z, 1e-6) / cfg.wear_fz_ref_n, cfg.wear_load_exp)
    )
    return ca.fmax(rate, 0.0)


def mu_temp_scale_np(t_core: float, cfg: TireThermalConfig, cmp: TireCompoundConfig) -> float:
    sigma = max(cmp.temp_sigma_c, 1e-3)
    z = (t_core - cmp.temp_opt_core_c) / sigma
    gauss = np.exp(-0.5 * z * z)
    raw = cmp.min_temp_scale + (1.02 - cmp.min_temp_scale) * gauss
    return float(np.clip(raw, cmp.min_temp_scale, cfg.max_mu_scale))


def mu_temp_scale_ca(t_core, cfg: TireThermalConfig, cmp: TireCompoundConfig):
    sigma = max(cmp.temp_sigma_c, 1e-3)
    z = (t_core - cmp.temp_opt_core_c) / sigma
    gauss = ca.exp(-0.5 * z * z)
    raw = cmp.min_temp_scale + (1.02 - cmp.min_temp_scale) * gauss
    return ca.fmin(cfg.max_mu_scale, ca.fmax(cmp.min_temp_scale, raw))


def mu_wear_scale_np(wear: float, cfg: TireThermalConfig, cmp: TireCompoundConfig) -> float:
    raw = 1.0 - cmp.wear_quadratic_coeff * (wear**2)
    return float(np.clip(raw, cmp.min_wear_scale, cfg.max_mu_scale))


def mu_wear_scale_ca(wear, cfg: TireThermalConfig, cmp: TireCompoundConfig):
    raw = 1.0 - cmp.wear_quadratic_coeff * ca.power(wear, 2)
    return ca.fmin(cfg.max_mu_scale, ca.fmax(cmp.min_wear_scale, raw))


def mu_scale_np(t_core: float, wear: float, cfg: TireThermalConfig, cmp: TireCompoundConfig) -> float:
    raw = mu_temp_scale_np(t_core, cfg, cmp) * mu_wear_scale_np(wear, cfg, cmp)
    return float(np.clip(raw, cfg.min_mu_scale, cfg.max_mu_scale))


def mu_scale_ca(t_core, wear, cfg: TireThermalConfig, cmp: TireCompoundConfig):
    raw = mu_temp_scale_ca(t_core, cfg, cmp) * mu_wear_scale_ca(wear, cfg, cmp)
    return ca.fmin(cfg.max_mu_scale, ca.fmax(cfg.min_mu_scale, raw))

