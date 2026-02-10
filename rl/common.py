"""
Shared helpers for RL-based ERS control.

This module intentionally stays dependency-light so both training code and
runtime strategy inference can reuse the same action projection and features.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from config import ERSConfig


def get_reference_arrays(reference_profile: Any) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract distance and speed arrays from a velocity-like profile.

    Supports:
    - VelocityProfile: fields ``s`` and ``v``
    - OptimalTrajectory: fields ``s`` and ``v_opt``
    """
    if reference_profile is None:
        raise ValueError("reference_profile must not be None")

    if not hasattr(reference_profile, "s"):
        raise AttributeError("reference_profile must expose 's'")

    s_ref = np.asarray(reference_profile.s, dtype=float)

    if hasattr(reference_profile, "v_opt"):
        v_ref = np.asarray(reference_profile.v_opt, dtype=float)
    elif hasattr(reference_profile, "v"):
        v_ref = np.asarray(reference_profile.v, dtype=float)
    else:
        raise AttributeError("reference_profile must expose 'v_opt' or 'v'")

    if s_ref.ndim != 1 or v_ref.ndim != 1 or s_ref.shape[0] != v_ref.shape[0]:
        raise ValueError("reference profile arrays must be 1D and have equal length")
    if s_ref.shape[0] < 2:
        raise ValueError("reference profile must contain at least two points")

    return s_ref, v_ref


def sample_reference_speed(
    position_m: float,
    total_length_m: float,
    s_ref: np.ndarray,
    v_ref: np.ndarray,
) -> float:
    """Sample target speed from a spatial reference profile with lap wrapping."""
    pos = float(position_m % max(total_length_m, 1.0))
    return float(np.interp(pos, s_ref, v_ref))


def compute_speed_limited_deploy_power(speed_mps: float, ers: ERSConfig) -> float:
    """
    Maximum deploy power [W] after regulation-dependent speed taper.

    2025: constant cap
    2026+: FIA taper model used in the main dynamics implementation.
    """
    if ers.regulation_year < 2026:
        return float(ers.max_deployment_power)

    v_kph = float(speed_mps * 3.6)
    p_base = float(ers.max_deployment_power)
    p_taper_1 = (1800.0 - 5.0 * v_kph) * 1000.0
    p_taper_2 = (6900.0 - 20.0 * v_kph) * 1000.0
    return float(max(0.0, min(p_base, p_taper_1, p_taper_2)))


def action_to_ers_power(
    action_value: float,
    speed_mps: float,
    soc: float,
    ers: ERSConfig,
    dt: float,
    deployed_energy_j: float,
    recovered_energy_j: float,
) -> float:
    """
    Convert normalized action [-1, 1] to feasible ERS power [W].

    Positive action deploys, negative action harvests. Projection includes:
    - speed-dependent deployment cap (2026 taper)
    - SOC bounds
    - remaining per-lap energy budgets
    """
    action = float(np.clip(action_value, -1.0, 1.0))
    dt_safe = max(float(dt), 1e-4)

    deploy_speed_cap = compute_speed_limited_deploy_power(speed_mps, ers)
    deploy_remaining = max(float(ers.deployment_limit_per_lap) - float(deployed_energy_j), 0.0)
    deploy_budget_cap = deploy_remaining / dt_safe
    deploy_cap = min(deploy_speed_cap, deploy_budget_cap)

    recover_power_cap = float(ers.max_recovery_power)
    recover_remaining = max(float(ers.recovery_limit_per_lap) - float(recovered_energy_j), 0.0)
    recover_budget_cap = recover_remaining / dt_safe
    recover_cap = min(recover_power_cap, recover_budget_cap)

    if soc <= ers.min_soc + 1e-4:
        deploy_cap = 0.0
    if soc >= ers.max_soc - 1e-4:
        recover_cap = 0.0

    if action >= 0.0:
        power = action * deploy_cap
    else:
        power = action * recover_cap

    return float(np.clip(power, -recover_cap, deploy_cap))


def build_observation(
    state: np.ndarray,
    track_info: dict[str, float],
    total_length_m: float,
    reference_speed_mps: float,
    deploy_remaining_j: float,
    recover_remaining_j: float,
    deployment_limit_j: float,
    recovery_limit_j: float,
    throttle_base: float,
    brake_base: float,
) -> np.ndarray:
    """
    Build normalized observation vector for RL policy.

    Order:
    [s_norm, v_norm, soc, grad_norm, curvature_norm, v_ref_norm,
     v_error_norm, deploy_remaining_norm, recover_remaining_norm, drive_balance]
    """
    s_norm = (float(state[0]) % max(total_length_m, 1.0)) / max(total_length_m, 1.0)
    v_norm = np.clip(float(state[1]) / 110.0, 0.0, 1.5)
    soc = np.clip(float(state[2]), 0.0, 1.0)

    grad = float(track_info.get("gradient", 0.0))
    grad_norm = np.clip(grad / 0.2, -1.0, 1.0)

    curvature = track_info.get("curvature")
    if curvature is None:
        radius = max(float(track_info.get("radius", 1e6)), 1.0)
        curvature = 1.0 / radius
    curvature_norm = np.clip(abs(float(curvature)) * 250.0, 0.0, 1.0)

    v_ref_norm = np.clip(float(reference_speed_mps) / 110.0, 0.0, 1.5)
    v_err_norm = np.clip((float(reference_speed_mps) - float(state[1])) / 40.0, -1.5, 1.5)

    deploy_norm = np.clip(
        float(deploy_remaining_j) / max(float(deployment_limit_j), 1.0),
        0.0,
        1.0,
    )
    recover_norm = np.clip(
        float(recover_remaining_j) / max(float(recovery_limit_j), 1.0),
        0.0,
        1.0,
    )

    drive_balance = np.clip(float(throttle_base) - float(brake_base), -1.0, 1.0)

    return np.array(
        [
            s_norm,
            v_norm,
            soc,
            grad_norm,
            curvature_norm,
            v_ref_norm,
            v_err_norm,
            deploy_norm,
            recover_norm,
            drive_balance,
        ],
        dtype=np.float32,
    )
