from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal


@dataclass
class TireThermalConfig:
    """Global thermal/degradation coefficients for the dynamic tire model."""

    surface_heat_capacity: float = 14_000.0
    core_heat_capacity: float = 42_000.0

    h_air: float = 65.0
    h_track: float = 120.0
    h_rim: float = 12.0
    k_surface_core: float = 48.0

    heat_generation_coeff: float = 0.005
    heat_generation_utilization_exp: float = 1.6
    utilization_ellipse_p: float = 2.0

    wear_coeff: float = 1.8e-5
    wear_utilization_exp: float = 2.2
    wear_load_exp: float = 1.0
    wear_temp_exp_coeff: float = 0.018
    wear_temp_ref_c: float = 95.0
    wear_fz_ref_n: float = 6_000.0

    min_mu_scale: float = 0.70
    max_mu_scale: float = 1.08


@dataclass
class TireCompoundConfig:
    """Compound-specific grip-window and degradation behavior."""

    name: Literal["soft", "medium", "hard"] = "medium"
    temp_opt_core_c: float = 100.0
    temp_sigma_c: float = 18.0
    min_temp_scale: float = 0.76
    wear_quadratic_coeff: float = 0.45
    min_wear_scale: float = 0.72
    heat_generation_scale: float = 1.0
    wear_rate_scale: float = 1.0


_COMPOUND_PRESETS: dict[str, TireCompoundConfig] = {
    "soft": TireCompoundConfig(
        name="soft",
        temp_opt_core_c=102.0,
        temp_sigma_c=14.0,
        min_temp_scale=0.73,
        wear_quadratic_coeff=0.62,
        min_wear_scale=0.68,
        heat_generation_scale=1.12,
        wear_rate_scale=1.28,
    ),
    "medium": TireCompoundConfig(
        name="medium",
        temp_opt_core_c=100.0,
        temp_sigma_c=18.0,
        min_temp_scale=0.76,
        wear_quadratic_coeff=0.45,
        min_wear_scale=0.72,
        heat_generation_scale=1.0,
        wear_rate_scale=1.0,
    ),
    "hard": TireCompoundConfig(
        name="hard",
        temp_opt_core_c=96.0,
        temp_sigma_c=24.0,
        min_temp_scale=0.80,
        wear_quadratic_coeff=0.32,
        min_wear_scale=0.76,
        heat_generation_scale=0.90,
        wear_rate_scale=0.72,
    ),
}


def get_tire_compound_config(compound: str) -> TireCompoundConfig:
    """Return an immutable-ish copy of the requested compound preset."""
    try:
        preset = _COMPOUND_PRESETS[compound.lower()]
    except KeyError as e:
        raise ValueError(f"Unknown tire compound: {compound}") from e
    return replace(preset)


__all__ = [
    "TireThermalConfig",
    "TireCompoundConfig",
    "get_tire_compound_config",
]
