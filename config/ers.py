from dataclasses import dataclass

@dataclass
class ERSConfig:
    """Configuration parameters for the ERS system"""
    max_deployment_power: float = 120000  # W (160 HP)
    max_recovery_power: float = 120000   # W
    battery_capacity: float = 4.5e6      # J (physical capacity)
    battery_usable_energy: float = 4e6   # J (regulatory limit)
    deployment_efficiency: float = 0.95
    recovery_efficiency: float = 0.85
    min_soc: float = 0.10  # Don't fully discharge (battery health)
    max_soc: float = 0.90  # Don't fully charge (battery health)