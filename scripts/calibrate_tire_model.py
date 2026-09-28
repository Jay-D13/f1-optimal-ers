#!/usr/bin/env python3
"""
Fit dynamic tire-model coefficients to high-level target behavior.

This script calibrates a small subset of thermal/degradation parameters and
exports a YAML file that can be used as an override set.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import minimize

from config import TireThermalConfig, get_tire_compound_config
from models.tire_thermals import (
    core_temp_rate_np,
    heat_generation_np,
    mu_scale_np,
    surface_temp_rate_np,
    utilization_np,
    wear_rate_np,
)


def _simulate_stint(
    thermal_cfg: TireThermalConfig,
    compound_cfg,
    ambient_temp_c: float,
    track_temp_c: float,
    init_temp_c: float,
    representative_load_n: float,
    representative_speed_ms: float,
    n_laps: int = 12,
    steps_per_lap: int = 40,
) -> dict[str, np.ndarray | float]:
    # Axle-averaged surrogate workload profile.
    dt = 1.0 / steps_per_lap
    ts = init_temp_c
    tc = init_temp_c
    wear = 0.0

    core_temp_lap = np.zeros(n_laps)
    mu_lap = np.zeros(n_laps)

    for lap in range(n_laps):
        mu_samples = []
        tc_samples = []
        for step in range(steps_per_lap):
            # Higher utilization during warmup laps, then settle.
            if lap <= 1:
                u_nom = 0.68
            elif lap <= 4:
                u_nom = 0.62
            else:
                u_nom = 0.56
            u = float(np.clip(u_nom + 0.03 * np.sin(2 * np.pi * step / steps_per_lap), 0.0, 1.0))

            fx = representative_load_n * 0.40 * u
            fy = representative_load_n * 0.55 * u
            fmax = representative_load_n * 1.5
            u_eff = utilization_np(fx, fy, fmax, fmax, thermal_cfg.utilization_ellipse_p)

            q_gen = heat_generation_np(
                representative_load_n,
                representative_speed_ms,
                u_eff,
                thermal_cfg,
                compound_cfg,
            )
            dts_dt = surface_temp_rate_np(q_gen, ts, tc, ambient_temp_c, track_temp_c, thermal_cfg)
            q_sc = thermal_cfg.k_surface_core * (ts - tc)
            dtc_dt = core_temp_rate_np(q_sc, tc, ambient_temp_c, thermal_cfg)
            dw_dt = wear_rate_np(u_eff, tc, representative_load_n, thermal_cfg, compound_cfg)

            ts += dts_dt * dt
            tc += dtc_dt * dt
            wear = float(np.clip(wear + dw_dt * dt, 0.0, 1.0))

            mu_samples.append(mu_scale_np(tc, wear, thermal_cfg, compound_cfg))
            tc_samples.append(tc)

        mu_lap[lap] = float(np.mean(mu_samples))
        core_temp_lap[lap] = float(np.mean(tc_samples))

    peak_idx = int(np.argmax(mu_lap))
    warmup_laps = float(peak_idx + 1)
    peak_core_temp = float(np.max(core_temp_lap))

    overheat_threshold = compound_cfg.temp_opt_core_c + 10.0
    overheat_lap = np.where(core_temp_lap >= overheat_threshold)[0]
    overheat_onset_temp = float(core_temp_lap[overheat_lap[0]]) if overheat_lap.size else float(core_temp_lap[-1])

    # Convert grip decay to equivalent laptime dropoff proxy.
    lap_penalty = (1.0 - mu_lap) * 4.0
    if n_laps - peak_idx > 2:
        x = np.arange(peak_idx, n_laps, dtype=float)
        y = lap_penalty[peak_idx:]
        slope = float(np.polyfit(x, y, deg=1)[0])
    else:
        slope = 0.0

    return {
        "warmup_laps_to_peak": warmup_laps,
        "peak_core_temp_c": peak_core_temp,
        "overheat_onset_c": overheat_onset_temp,
        "stint_dropoff_s_per_lap": max(0.0, slope),
        "core_temp_lap": core_temp_lap,
        "mu_lap": mu_lap,
    }


def _loss(x, base_thermal: TireThermalConfig, base_compound, targets: dict, fit_defaults: dict) -> float:
    thermal_cfg = replace(
        base_thermal,
        heat_generation_coeff=float(x[0]),
        wear_coeff=float(x[1]),
        wear_temp_exp_coeff=float(x[2]),
    )
    compound_cfg = replace(
        base_compound,
        wear_rate_scale=float(x[3]),
        temp_sigma_c=float(x[4]),
    )

    sim = _simulate_stint(
        thermal_cfg=thermal_cfg,
        compound_cfg=compound_cfg,
        ambient_temp_c=float(fit_defaults["ambient_temp_c"]),
        track_temp_c=float(fit_defaults["track_temp_c"]),
        init_temp_c=float(fit_defaults["init_temp_c"]),
        representative_load_n=float(fit_defaults["representative_load_n"]),
        representative_speed_ms=float(fit_defaults["representative_speed_ms"]),
    )

    err = 0.0
    err += ((sim["warmup_laps_to_peak"] - targets["warmup_laps_to_peak"]) / 1.0) ** 2
    err += ((sim["peak_core_temp_c"] - targets["peak_core_temp_c"]) / 8.0) ** 2
    err += ((sim["overheat_onset_c"] - targets["overheat_onset_c"]) / 8.0) ** 2
    err += ((sim["stint_dropoff_s_per_lap"] - targets["stint_dropoff_s_per_lap"]) / 0.08) ** 2
    return float(err)


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate dynamic tire-model coefficients.")
    parser.add_argument(
        "--targets",
        type=Path,
        default=Path("config/tire_calibration_targets.yaml"),
        help="Target behavior YAML path.",
    )
    parser.add_argument(
        "--compound",
        choices=["soft", "medium", "hard"],
        default="medium",
        help="Compound to calibrate.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output YAML path (default: config/tire_model_fit_<compound>.yaml).",
    )
    args = parser.parse_args()

    data = yaml.safe_load(args.targets.read_text())
    targets = data["compound_targets"][args.compound]
    fit_defaults = data["fit_defaults"]

    base_thermal = TireThermalConfig()
    base_compound = get_tire_compound_config(args.compound)

    x0 = np.array(
        [
            base_thermal.heat_generation_coeff,
            base_thermal.wear_coeff,
            base_thermal.wear_temp_exp_coeff,
            base_compound.wear_rate_scale,
            base_compound.temp_sigma_c,
        ],
        dtype=float,
    )

    bounds = [
        (0.001, 0.030),  # heat_generation_coeff
        (1e-6, 1e-3),    # wear_coeff
        (0.005, 0.060),  # wear_temp_exp_coeff
        (0.3, 2.0),      # wear_rate_scale
        (8.0, 35.0),     # temp_sigma_c
    ]

    res = minimize(
        _loss,
        x0=x0,
        args=(base_thermal, base_compound, targets, fit_defaults),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 300, "ftol": 1e-10},
    )

    best = res.x
    fitted_thermal = replace(
        base_thermal,
        heat_generation_coeff=float(best[0]),
        wear_coeff=float(best[1]),
        wear_temp_exp_coeff=float(best[2]),
    )
    fitted_compound = replace(
        base_compound,
        wear_rate_scale=float(best[3]),
        temp_sigma_c=float(best[4]),
    )

    sim = _simulate_stint(
        thermal_cfg=fitted_thermal,
        compound_cfg=fitted_compound,
        ambient_temp_c=float(fit_defaults["ambient_temp_c"]),
        track_temp_c=float(fit_defaults["track_temp_c"]),
        init_temp_c=float(fit_defaults["init_temp_c"]),
        representative_load_n=float(fit_defaults["representative_load_n"]),
        representative_speed_ms=float(fit_defaults["representative_speed_ms"]),
    )

    out = {
        "compound": args.compound,
        "optimizer_success": bool(res.success),
        "optimizer_message": str(res.message),
        "loss": float(res.fun),
        "fitted_thermal": asdict(fitted_thermal),
        "fitted_compound": asdict(fitted_compound),
        "simulated_metrics": {
            k: float(v) for k, v in sim.items() if not isinstance(v, np.ndarray)
        },
        "targets": {k: float(v) for k, v in targets.items()},
    }

    output_path = args.output or Path(f"config/tire_model_fit_{args.compound}.yaml")
    output_path.write_text(yaml.safe_dump(out, sort_keys=False))
    print(f"Saved calibration: {output_path}")
    print(f"Optimizer success: {res.success} | loss={res.fun:.6f}")


if __name__ == "__main__":
    main()
