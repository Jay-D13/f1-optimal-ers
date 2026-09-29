"""
ERS (Energy Recovery System) Configuration

F1 ERS Regulations (2024):
- MGU-K: 120kW max power, 2MJ recovery limit per lap
- MGU-H: Unlimited (connected to turbo)
- Total deployment: 4MJ per lap
- Battery capacity: ~4.5MJ physical, 4MJ usable
"""

from dataclasses import dataclass, replace
from typing import Optional, Mapping
import math

@dataclass
class ERSConfig:
    """Configuration parameters for the ERS system"""
    
    regulation_year: int = 2025  # Default regulation year
    
    # Power limits (Watts)
    # 2025: 120kW (MGU-K)
    # 2026: 350kW (MGU-K)
    max_deployment_power: float = 120_000   
    max_recovery_power: float = 120_000     
    
    # Energy limits (Joules)
    battery_capacity: float = 4.5e6         # 4.5 MJ physical capacity
    
    # "Usable Energy": The permitted SOC swing in the battery.
    # 2025: 4 MJ limit. 2026: Still effectively 4 MJ ES sizing/rules.
    battery_usable_energy: float = 4.0e6    

    # Recovery Limit (MGU-K Harvesting limit)
    # 2025: 2 MJ. 2026: 8.5 MJ.
    recovery_limit_per_lap: float = 2.0e6   

    # Deployment Limit (ES -> MGU-K limit)
    # 2025: 4 MJ cap. 
    # 2026: Unlimited (physically constrained by recovery + battery capacity).
    deployment_limit_per_lap: float = 4.0e6 
    
    # Efficiency factors
    deployment_efficiency: float = 0.95     # Battery → wheel
    recovery_efficiency: float = 0.85       # Kinetic → battery
    
    # Electric turbocharger recovery efficiency (MGU-H)
    # 2025: ~10% (TUMFTM value)
    # 2026: 0% (MGU-H Removed)
    etc_recovery_efficiency: float = 0.10   
    
    # State of Charge limits (battery health protection)
    min_soc: float = 0.20                   
    max_soc: float = 0.90                   
    
    # Initial conditions
    default_initial_soc: float = 0.50       
    
    # Minimum velocity for e-motor use
    vel_min_e_motor: float = 27.777         # [m/s] ~100 km/h
    
    # Maximum e-motor torque
    torque_e_motor_max: float = 200.0       # [Nm]

    # 2026 rules (defaults keep the 2025 behaviour)
    # Deploy limit vs speed: "flat" (max_deployment_power), "standard" (race) or "overtake" (qualifying, C5.2.8)
    deploy_curve: str = "flat"
    # MGU-K and inverter efficiency, wheel <-> DC bus. The 2026 limits (deploy curve, ±350 kW, recharge cap)
    # are DC powers and energies; 1.0 applies them to the mechanical power at the wheel.
    mgu_k_efficiency: float = 1.0
    # Max - min stored energy while on track (J, C5.2.9). None: fixed min_soc/max_soc bounds instead.
    soc_window: Optional[float] = None
    # Harvest allowed at full throttle (W, DC; "super-clip"). None: max_recovery_power.
    superclip_power: Optional[float] = None
    # Qualifying lap: start with a full store, include the run-up from the last corner, finish anywhere
    qualifying: bool = False
    
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
        return self.battery_capacity / 100.0
    
_REG_OVERRIDES: Mapping[str, dict] = {
    "2025": {
        "max_deployment_power": 120_000.0,
        "max_recovery_power": 120_000.0,
        "recovery_limit_per_lap": 2.0e6,      # 2 MJ
        "deployment_limit_per_lap": 4.0e6,    # 4 MJ
        "etc_recovery_efficiency": 0.10,      # MGU-H exists
    },
    "2026": {
        "max_deployment_power": 350_000.0,    # 350 kW (Tripled!)
        "max_recovery_power": 350_000.0,      # 350 kW
        "recovery_limit_per_lap": 8.5e6,      # 8.5 MJ (Doubled+)
        "deployment_limit_per_lap": 100.0e6,  # Effectively Unlimited (use huge number)
        "etc_recovery_efficiency": 0.0,       # MGU-H Removed
        "regulation_year": 2026,
        "deploy_curve": "standard",
        # Starting values, calibrated in ROADMAP Phase 4: MGU-K + inverter 0.95, battery 0.97 each way
        "mgu_k_efficiency": 0.95,
        "deployment_efficiency": 0.95 * 0.97,
        "recovery_efficiency": 0.95 * 0.97,
    },
}

def get_ers_config(
    regulation_set: str = "2025",
    base: ERSConfig | None = None,
    session: str = "race",
    event=None,
) -> ERSConfig:
    """
    ERS rules for a regulation set. For 2026 qualifying (session="qualifying"): the Overtake deploy curve,
    the 4 MJ window with a full store at the start, and the event's recharge cap and super-clip limit
    (`event` is a 2026 round number or track name; unknown events keep the base 8.5 MJ and 350 kW).
    Sessions don't change the 2025 rules.
    """
    cfg = base or ERSConfig()
    
    if regulation_set not in _REG_OVERRIDES:
        raise ValueError(f"Unknown regulation set: {regulation_set}. Options: {list(_REG_OVERRIDES.keys())}")
    if session not in ("race", "qualifying"):
        raise ValueError(f"Unknown session: {session}. Options: race, qualifying")

    cfg = replace(cfg, **_REG_OVERRIDES[regulation_set])
    if regulation_set != "2026" or session != "qualifying":
        return cfg

    from config.events import find_event_2026

    found = find_event_2026(event)
    return replace(
        cfg,
        deploy_curve="overtake",
        soc_window=4.0e6,
        min_soc=0.0,
        max_soc=1.0,
        qualifying=True,
        recovery_limit_per_lap=(found.quali_recharge_mj if found else 8.5) * 1e6,
        superclip_power=(found.superclip_kw if found else 350.0) * 1e3,
    )


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