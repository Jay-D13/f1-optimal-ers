from dataclasses import dataclass, replace
from typing import Dict, Mapping, Optional
import numpy as np


@dataclass 
class VehicleConfig:
    """
    Based on FIA rules and TUMFTM F1_Shanghai.ini 
    configuration: https://github.com/TUMFTM/laptime-simulation/blob/master/laptimesim/input/vehicles/F1_Shanghai.ini
    """
    regulation_year: int = 2025  # Default regulation year
    
    # ==================== Mass and Geometry ====================
    mass: float = 798.0           # [kg] Minimum with driver (2024 regs)
    fuel_mass: float = 0.0        # [kg] Fuel on board, added to the mass
    lf: float = 1.968             # [m] Front axle to CoG
    lr: float = 1.632             # [m] Rear axle to CoG
    h_cog: float = 0.335          # [m] CoG height
    sf: float = 1.6               # [m] Front track width
    sr: float = 1.6               # [m] Rear track width
    
    # ==================== Aerodynamics ====================
    frontal_area: float = 1.5     # [m²] For reference (not directly used)
    c_w_a: float = 1.56           # [m²] Cd × A (drag coefficient × area)
    c_z_a_f: float = 2.20         # [m²] Cl × A front (downforce)
    c_z_a_r: float = 2.68         # [m²] Cl × A rear (downforce)
    rho_air: float = 1.18         # [kg/m³] Air density
    drs_factor: float = 0.17      # [-] Drag reduction from DRS
    cornering_drag_coeff: float = 0.0  # [kg] Extra drag c·|κ|·v² in corners (ETH/Ferrari model; 0 = off)

    # 2026 active aero until the FIA zones are modelled (ROADMAP Phase 2): Straight Mode on any stretch
    # with a radius above straight_mode_min_radius, scaling the drag and downforce areas
    straight_mode_min_radius: float = 400.0   # [m]
    straight_mode_transition: float = 0.4     # [s] Longest opening or closing time (C3.10.10o)
    straight_mode_drag_factor: float = 0.65   # [-]
    straight_mode_downforce_factor: float = 1.0  # [-]
    
    # ==================== Rolling Resistance ====================
    f_roll: float = 0.03          # [-] Rolling resistance coefficient
    cr: float = 0.03              # [-] Alias for f_roll (compatibility)
    
    # ==================== Powertrain ====================
    topology: str = "RWD"         # Rear wheel drive
    pow_max_ice: float = 575e3    # [W] ICE max power (~770 HP)
    pow_max_ers: float = 120e3    # [W] ERS max power (MGU-K)
    
    # Engine RPM characteristics (for gearbox model)
    n_begin: float = 10500.0 / 60.0  # [1/s] RPM at pow_max - pow_diff
    n_max: float = 11400.0 / 60.0    # [1/s] RPM at pow_max
    n_end: float = 12200.0 / 60.0    # [1/s] RPM at pow_max - pow_diff
    pow_diff: float = 41e3           # [W] Power drop from max
    
    # ==================== Braking ====================
    max_brake_force: float = 50_000  # [N] Total braking force
    brake_balance_front: float = 0.6  # [-] Front share of a single brake command (simulator; the NLP splits freely)
    
    # ==================== Physical Constants ====================
    g: float = 9.81               # [m/s²] Gravitational acceleration
    
    # ==================== Simplified Model ====================
    
    @property
    def wheelbase(self) -> float:
        """Total wheelbase [m]"""
        return self.lf + self.lr
    
    @property
    def pow_max_total(self) -> float:
        """Total power (ICE + ERS) [W]"""
        return self.pow_max_ice + self.pow_max_ers
    
    @property
    def cd(self) -> float:
        """Drag coefficient (estimated from c_w_a)"""
        return self.c_w_a / self.frontal_area
    
    @property
    def cl(self) -> float:
        """Lift coefficient (estimated from c_z_a)"""
        return (self.c_z_a_f + self.c_z_a_r) / self.frontal_area
    
    # ==================== Track-Specific Configurations ====================
    
    @classmethod
    def for_monaco(cls) -> 'VehicleConfig':
        """High downforce for Monaco (tight corners)"""
        config = cls()
        config.c_w_a = 1.8       # Higher drag
        config.c_z_a_f = 2.8     # More front downforce
        config.c_z_a_r = 3.2     # More rear downforce
        return config
    
    @classmethod
    def for_monza(cls) -> 'VehicleConfig':
        """Low downforce for Monza (high speed)"""
        config = cls()
        config.c_w_a = 1.2       # Minimum drag
        config.c_z_a_f = 1.6     # Reduced front downforce
        config.c_z_a_r = 2.0     # Reduced rear downforce
        return config
    
    @classmethod
    def for_spa(cls) -> 'VehicleConfig':
        """Medium downforce for Spa"""
        # Default is already medium
        return cls()
    
    @classmethod
    def for_silverstone(cls) -> 'VehicleConfig':
        """Medium-high downforce for Silverstone"""
        config = cls()
        config.c_z_a_f = 2.4
        config.c_z_a_r = 2.9
        return config
    
    @classmethod
    def for_montreal(cls) -> 'VehicleConfig':
        """Medium downforce for Montreal"""
        # Default is already medium
        return cls()
    
    @classmethod
    def for_shanghai(cls) -> 'VehicleConfig':
        # already have Shanghai defaults by default 
        return cls()
    
# ==================== Regulation-specific variants ====================

_REGULATION_OVERRIDES: Mapping[str, Dict[str, float]] = {
    # V6 Turbo Hybrid era (2014-2025)
    "2025": {
        "mass": 798.0,          # [kg] min with driver
        "pow_max_ice": 575e3,   # [W] ~770 HP
        "pow_max_ers": 120e3,   # [W] MGU-K power (120 kW)
        "regulation_year": 2025,
    },
    # 2026 new regulations
    "2026": {
        "mass": 772.0,          # [kg] 726 kg qualifying minimum + ~46 kg of tyres (C4.1; tyre mass unverified)
        "pow_max_ice": 400e3,   # [W] ~536 HP (reduced ICE)
        "pow_max_ers": 350e3,   # [W] MGU-K power (350 kW - tripled!)
        "regulation_year": 2026,
        # Starting values for the Phase 4 fit. Corner Mode aero from 2026 telemetry fits (REFERENCE_2026.md §2, §7),
        # the same at every track: scaling the 2025 presets (TUM estimates) by the FIA's published targets
        # made the 2026 car far too fast on the straights, since the 2025 car itself isn't calibrated.
        "c_w_a": 0.95,          # [m²] Corner Mode CdA
        "c_z_a_f": 3.45 * 0.451,  # [m²] Corner Mode ClA 3.45, 45 % front
        "c_z_a_r": 3.45 * 0.549,
        # Straight Mode vs Corner Mode, from CFD estimates: about -20 % drag, -28 % downforce
        "straight_mode_drag_factor": 0.80,
        "straight_mode_downforce_factor": 0.72,
    },
}

# Scaling of the base car's aero per regulation set (applied to any track preset). None for now:
# the 2026 car takes absolute values from 2026 data instead (see _REGULATION_OVERRIDES).
_REGULATION_SCALES: Mapping[str, Dict[str, float]] = {}

def get_vehicle_config(regulation_set: str = "2025", *, base: Optional["VehicleConfig"] = None) -> "VehicleConfig":
    """Return a VehicleConfig for a given regulation set.

    This avoids duplicating all the shared parameters: we create a base VehicleConfig
    (defaults to VehicleConfig()) and then override only the fields that change.
    The 2026 aero areas are scaled from the base (e.g. a track preset) rather than replaced.
    """
    cfg = base or VehicleConfig()
    try:
        overrides = _REGULATION_OVERRIDES[regulation_set]
    except KeyError as e:
        raise ValueError(f"Unknown regulation set: {regulation_set}") from e
    scaled = {name: getattr(cfg, name) * factor for name, factor in _REGULATION_SCALES.get(regulation_set, {}).items()}
    return replace(cfg, **overrides, **scaled)


# used for grip
# since we have it from TUMFTM might as well use it
@dataclass
class TireParameters:
    """
    TUMFTM tire parameters.
    
    The tire model includes load-dependent friction:
    μ(F_z) = μ_0 + dμ/dF_z · (F_z - F_z0)
    
    This models the reduction in friction coefficient at higher loads.
    """
    fz_0: float = 3000.0          # Nominal tire load (N)
    
    # Front tire
    mux_f: float = 1.65           # Longitudinal friction at fz_0
    muy_f: float = 1.85           # Lateral friction at fz_0
    dmux_dfz_f: float = -5.0e-5   # Friction reduction with load
    dmuy_dfz_f: float = -5.0e-5
    
    # Rear tire  
    mux_r: float = 1.95           # Rear has more grip (wider tires)
    muy_r: float = 2.15           # Lateral friction at fz_0
    dmux_dfz_r: float = -5.0e-5
    dmuy_dfz_r: float = -5.0e-5
    
    # Friction circle exponent (2.0 = pure circle, <2.0 = diamond-ish)
    tire_model_exp: float = 2.0