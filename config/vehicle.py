from dataclasses import dataclass
from typing import Dict


@dataclass
class VehicleConfig:
    """Vehicle parameters for dynamics model"""
    
    # Mass and inertia
    mass: float = 798.0                     # kg (minimum with driver)
    
    # Aerodynamics
    frontal_area: float = 1.5               # m² (typical F1)
    cd: float = 0.9                         # Drag coefficient (medium downforce)
    cl: float = 2.5                         # Lift coefficient (downforce)
    
    # Tire characteristics
    cr: float = 0.015                       # Rolling resistance coefficient
    mu_longitudinal: float = 1.8            # Peak longitudinal friction (dry)
    mu_lateral: float = 1.8                 # Peak lateral friction (dry)
    
    # Powertrain
    max_ice_power: float = 560_000          # W (~750 HP from ICE)
    max_total_power: float = 680_000        # W (ICE + MGU-K)
    
    # Braking
    max_brake_force: float = 50_000         # N (pure mechanical)
    max_brake_power: float = 2_000_000      # W (at high speed)
    
    # Physical constants
    g: float = 9.81                         # Gravitational acceleration
    rho_air: float = 1.225                  # Air density at sea level
    
    @classmethod
    def for_monaco(cls) -> 'VehicleConfig':
        """High downforce configuration for Monaco"""
        return cls(
            cd=1.0,                         # Higher drag
            cl=3.5,                         # Maximum downforce
            mu_lateral=1.9,                 # Better mechanical grip setup
        )
    
    @classmethod
    def for_monza(cls) -> 'VehicleConfig':
        """Low downforce configuration for Monza"""
        return cls(
            cd=0.7,                         # Minimum drag
            cl=2.0,                         # Reduced downforce
        )
    
    @classmethod
    def for_spa(cls) -> 'VehicleConfig':
        """Medium downforce for Spa"""
        return cls(
            cd=0.85,
            cl=2.5,
        )
    
    def get_aero_forces(self, velocity: float) -> Dict[str, float]:
        """Calculate aerodynamic forces at given velocity"""
        q = 0.5 * self.rho_air * velocity**2  # Dynamic pressure
        
        return {
            'drag': q * self.cd * self.frontal_area,
            'downforce': q * self.cl * self.frontal_area,
        }
    
    def get_max_cornering_speed(self, radius: float) -> float:
        """
        Calculate maximum cornering speed for given radius.
        Uses simplified model: v = sqrt(mu * g * R * (1 + Cl*rho*A*R/(2*m)))
        
        With downforce, the normal force increases with speed, allowing higher speeds.
        """
        import numpy as np
        
        if np.isinf(radius) or radius > 5000:
            return 100.0  # Max speed for straights
        
        # Iterative solution (downforce depends on speed)
        # Start with no-downforce estimate
        v = np.sqrt(self.mu_lateral * self.g * radius)
        
        for _ in range(5):  # Converge quickly
            downforce = 0.5 * self.rho_air * v**2 * self.cl * self.frontal_area
            normal_force = self.mass * self.g + downforce
            friction_force = self.mu_lateral * normal_force
            v_new = np.sqrt(friction_force * radius / self.mass)
            if abs(v_new - v) < 0.1:
                break
            v = 0.5 * (v + v_new)  # Damped update
        
        return min(v, 100.0)  # Cap at max speed