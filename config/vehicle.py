from dataclasses import dataclass

@dataclass
class VehicleConfig:
    """Vehicle parameters for dynamics model"""
    mass: float = 798.0
    frontal_area: float = 1.4  # m^2 (more typical than 1.5)
    cd: float = 0.9            # drag coefficient (medium downforce)
    cl: float = 2.5            # lift coefficient (negative = downforce)
    cd_low_df: float = 0.7     # Monza configuration
    cd_high_df: float = 1.0    # Monaco configuration
    cr: float = 0.015          # rolling resistance (F1 slicks)
    max_ice_power: float = 560000  # W (~750 HP ICE only, conservative)
    # Note: Total system with MGU-K = 560kW + 120kW = 680kW (~910 HP)
    # With MGU-H in reality, total exceeds 750kW (1000+ HP)
    max_brake_force: float = 50000   # N
    
    @classmethod
    def for_monaco(cls):
        return cls(cd=1.0, cl=3.0)
    
    @classmethod
    def for_monza(cls):
        return cls(cd=0.7, cl=2.0)
    