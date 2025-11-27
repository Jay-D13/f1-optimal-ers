from dataclasses import dataclass

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
    
    # Efficiency factors
    deployment_efficiency: float = 0.95     # Battery to motor
    recovery_efficiency: float = 0.85       # Kinetic to battery (includes motor + inverter losses)
    
    # State of Charge limits (battery health protection)
    min_soc: float = 0.20                   # Never fully discharge
    max_soc: float = 0.90                   # Never fully charge
    
    # TODO: do we consider thermal constraints? (simplified)
    max_battery_temp: float = 60.0          # °C
    max_motor_temp: float = 150.0           # °C
    
    @property
    def usable_soc_range(self) -> float:
        """Available SOC swing for optimization"""
        return self.max_soc - self.min_soc
    
    @property
    def energy_per_percent_soc(self) -> float:
        """Joules per 1% SOC change"""
        return self.battery_capacity / 100
