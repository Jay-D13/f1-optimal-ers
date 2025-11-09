from __future__ import annotations
from dataclasses import dataclass

@dataclass
class CarParams:
    m: float = 800.0           # kg (car + driver)
    rho: float = 1.2           # air density
    CdA: float = 1.60          # drag area [m^2]
    ClA: float = 3.00          # downforce area [m^2]
    Cr: float = 0.012          # rolling resistance coef
    g: float = 9.81
    eta_drv: float = 0.95      # driveline eff (wheel)

@dataclass
class ERSParams:
    Pk_max: float = 120e3      # MGU-K abs power limit [W]
    Pice_max: float = 520e3    # ICE max shaft power to gearbox [W] (placeholder)
    Ebatt_max: float = 4.0e6   # energy capacity [J] (approx 4 MJ usable)
    soc_min: float = 0.1
    soc_max: float = 0.95
    eta_k_deploy: float = 0.95 # ES->MGU-K eff
    eta_k_harv: float = 0.75   # MGU-K->ES eff

@dataclass
class SimParams:
    v_min: float = 5.0/3.6     # to avoid divide-by-zero in spatial dynamics [m/s]
    ds: float = 2.0            # grid step [m], keep in sync with resampling