from dataclasses import dataclass
from typing import Dict
import numpy as np


@dataclass 
class VehicleConfig:
    """
    Based on FIA rules and TUMFTM F1_Shanghai.ini 
    configuration: https://github.com/TUMFTM/laptime-simulation/blob/master/laptimesim/input/vehicles/F1_Shanghai.ini
    """
    
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
    def max_ice_power(self) -> float:
        """Alias for pow_max_ice for compatibility"""
        return self.pow_max_ice
    
    @property
    def max_total_power(self) -> float:
        """Alias for pow_max_total for compatibility"""
        return self.pow_max_total
    
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

    
    def get_max_cornering_speed(self, radius: float) -> float:
        """
        Calculate maximum cornering speed for given radius.
        
        Iterative solution accounting for downforce.
        """
        mu = self.mu_lateral
            
        if np.isinf(radius) or radius > 2000:
            return 100.0  # Max on straights
        
        radius = max(radius, 10.0)
        
        # Start with no-downforce estimate
        v = np.sqrt(mu * self.g * radius)
        
        # Iterate (downforce depends on speed)
        for _ in range(5):
            aero = self.get_aero_forces(v)
            normal_force = self.mass * self.g + aero['downforce']
            friction_force = mu * normal_force
            v_new = np.sqrt(friction_force * radius / self.mass)
            
            if abs(v_new - v) < 0.1:
                break
            v = 0.7 * v + 0.3 * v_new
        
        return min(v, 100.0)  # Cap at max speed
    
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
