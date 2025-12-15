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
    
    @property
    def mu_longitudinal(self) -> float:
        """Average longitudinal friction (for simplified models)"""
        return 1.8  # Average of front and rear
    
    @property
    def mu_lateral(self) -> float:
        """Average lateral friction (for simplified models)"""
        return 2.0  # Average of front and rear
    
    def get_aero_forces(self, velocity: float) -> dict:
        """Calculate aerodynamic forces at given velocity."""
        q = 0.5 * self.rho_air * velocity**2
        return {
            'drag': q * self.c_w_a,
            'downforce': q * (self.c_z_a_f + self.c_z_a_r),
            'downforce_front': q * self.c_z_a_f,
            'downforce_rear': q * self.c_z_a_r,
        }    

    
    def get_max_cornering_speed(self, radius: float, tire_params: Optional['TireParameters'] = None) -> float:
        """
        Calculate maximum cornering speed.
        
        Uses CONSTANT friction coefficient (mu_lateral = 2.0) for consistency
        across all solvers (FB and NLP). Downforce effect is calculated iteratively.
        
        Output is capped at V_MAX_PHYSICAL to prevent divergence.
        """
        # Use default parameters if none provided
        tires = tire_params if tire_params is not None else TireParameters()

        # Physical speed cap - F1 cars max out around 370 km/h = 103 m/s
        # Use slightly higher for theoretical limit
        V_MAX_PHYSICAL = 110.0  # m/s (~396 km/h)
        
        # For very large radii (essentially straights), return physical max
        if np.isinf(radius) or radius > 1000:
            return V_MAX_PHYSICAL
        
        radius = max(radius, 10.0)
        
        # Initial guess
        v = 50.0 
        
        for _ in range(15):  # Iterative solver
            # Cap v during iteration to prevent divergence
            v = min(v, V_MAX_PHYSICAL)
            
            # 1. Calculate Downforce
            aero = self.get_aero_forces(v)
            downforce = aero['downforce']
            
            # 2. Total Vertical Load (Mass + Downforce)
            # We ignore banking (gradient) here for the 'general' limit, 
            # or you can assume flat ground (cos(0)=1).
            F_z_total = self.mass * self.g + downforce
            
            # 3. Maximum lateral force (constant friction for consistency)
            # Using constant mu_lateral = 2.0 for consistency across all solvers
            mu_lat = self.mu_lateral  # 2.0
            
            # 4. Solve for new velocity
            # F_lat = m * v^2 / R  <=  mu * F_z_total
            # v = sqrt( mu * F_z_total * R / m )
            
            F_lat_max = mu_lat * F_z_total
            v_new = np.sqrt(F_lat_max * radius / self.mass)
            
            # Cap to physical max
            v_new = min(v_new, V_MAX_PHYSICAL)
            
            # Damped update for stability
            if abs(v_new - v) < 0.01:
                v = v_new
                break
            v = 0.6 * v + 0.4 * v_new
        
        # Final safety cap
        return min(v, V_MAX_PHYSICAL)
    
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
        "mass": 768.0,          # [kg] 30kg lighter
        "pow_max_ice": 400e3,   # [W] ~536 HP (reduced ICE)
        "pow_max_ers": 350e3,   # [W] MGU-K power (350 kW - tripled!)
        "regulation_year": 2026,
    },
}

def get_vehicle_config(regulation_set: str = "2025", *, base: Optional["VehicleConfig"] = None) -> "VehicleConfig":
    """Return a VehicleConfig for a given regulation set.

    This avoids duplicating all the shared parameters: we create a base VehicleConfig
    (defaults to VehicleConfig()) and then override only the fields that change.
    """
    cfg = base or VehicleConfig()
    try:
        overrides = _REGULATION_OVERRIDES[regulation_set]
    except KeyError as e:
        raise ValueError(f"Unknown regulation set: {regulation_set}") from e
    return replace(cfg, **overrides)


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