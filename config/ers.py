"""
ERS (Energy Recovery System) Configuration

F1 ERS Regulations (2024):
- MGU-K: 120kW max power, 2MJ recovery limit per lap
- MGU-H: Unlimited (connected to turbo)
- Total deployment: 4MJ per lap
- Battery capacity: ~4.5MJ physical, 4MJ usable
"""

from dataclasses import dataclass

@dataclass
class ERSConfig:
    """Configuration parameters for the ERS system"""
    
    # Power limits (Watts)
    max_deployment_power: float = 120_000   # 120kW (MGU-K limit)
    max_recovery_power: float = 120_000     # 120kW (MGU-K limit)
    
    # Energy limits (Joules)
    battery_capacity: float = 4.5e6         # 4.5 MJ physical capacity
    battery_usable_energy: float = 4.0e6    # 4.0 MJ regulatory limit per lap
    recovery_limit_per_lap: float = 2.0e6   # 2.0 MJ MGU-K limit
    deployment_limit_per_lap: float = 4.0e6 # 4.0 MJ deployment limit
    
    # Efficiency factors
    deployment_efficiency: float = 0.95     # Battery → wheel
    recovery_efficiency: float = 0.85       # Kinetic → battery
    
    # Electric turbocharger recovery efficiency (MGU-H)
    etc_recovery_efficiency: float = 0.10   # From TUMFTM
    
    # State of Charge limits (battery health protection)
    min_soc: float = 0.20                   # Never fully discharge (never below 20%)
    max_soc: float = 0.90                   # Never fully charge (never above 90%)
    
    # Initial conditions
    default_initial_soc: float = 0.50       # Start at 50%
    
    # Minimum velocity for e-motor use
    vel_min_e_motor: float = 27.777         # [m/s] ~100 km/h
    
    # Maximum e-motor torque
    torque_e_motor_max: float = 200.0       # [Nm]
    
    @property
    def usable_soc_range(self) -> float:
        """Available SOC swing for optimization"""
        return self.max_soc - self.min_soc
    
    @property
    def usable_energy(self) -> float:
        """Usable energy in battery (J)"""
        return self.battery_capacity * self.usable_soc_range
    
    @property
    def energy_per_percent_soc(self) -> float:
        """Joules per 1% SOC change"""
        return self.battery_capacity / 100

@dataclass
class ERSConfigQualifying(ERSConfig):
    """
    ERS configuration optimized for qualifying.
    
    In qualifying:
    - Start with full charge
    - Use all available energy
    - End SOC doesn't matter (only one lap)
    """
    default_initial_soc: float = 0.90
    min_final_soc: float = 0.20
    
@dataclass
class ERSConfigRace(ERSConfig):
    """
    ERS configuration for race.
    
    In race:
    - Must be charge-sustaining over each lap
    - Energy budget management across stint
    - Consider tire/fuel degradation effects
    """
    default_initial_soc: float = 0.50
    min_final_soc: float = 0.45  # End close to where started