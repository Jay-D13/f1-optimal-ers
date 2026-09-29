"""
The calibration parameters, and the model's qualifying lap for a set of them on a reference lap's path.
"""
import contextlib
import io
from dataclasses import dataclass, replace
from typing import Dict, Mapping

import numpy as np

from config import TireParameters, get_ers_config, get_vehicle_config
from models import VehicleDynamicsModel
from solvers import SpatialNLPSolver

from .dataset import ReferenceLap


@dataclass(frozen=True)
class Parameter:
    name: str
    start: float
    lower: float
    upper: float
    doc: str


# Shared by every track (decision D2: one generic pole car). Bounds are physical ranges: when a fit ends on one,
# the model is missing something rather than the car being extreme.
PARAMETERS = (
    Parameter("c_w_a", 0.95, 0.7, 1.3, "Corner Mode drag area CdA (m²)"),
    Parameter("c_z_a", 3.45, 2.5, 5.5, "Corner Mode downforce area ClA (m²)"),
    Parameter("front_downforce_share", 0.451, 0.38, 0.52, "Share of the downforce on the front axle"),
    Parameter("straight_mode_drag_factor", 0.80, 0.6, 0.95, "Straight Mode drag / Corner Mode drag"),
    Parameter("mu_scale", 1.0, 0.8, 1.3, "Scale on every tyre friction coefficient"),
    Parameter("pow_max_ice", 400e3, 330e3, 430e3, "ICE power (W); confounded with drag on the straights"),
)


def start_values() -> Dict[str, float]:
    return {p.name: p.start for p in PARAMETERS}


def car_for(params: Mapping[str, float], air_density=None):
    """VehicleConfig and TireParameters for a parameter set (unknown names are ignored)."""
    vehicle = get_vehicle_config("2026")
    changes = {k: v for k, v in params.items() if hasattr(vehicle, k)}
    if "c_z_a" in params or "front_downforce_share" in params:
        total = params.get("c_z_a", vehicle.c_z_a_f + vehicle.c_z_a_r)
        front = params.get("front_downforce_share", vehicle.c_z_a_f / (vehicle.c_z_a_f + vehicle.c_z_a_r))
        changes["c_z_a_f"] = total * front
        changes["c_z_a_r"] = total * (1.0 - front)
    if air_density:
        changes["rho_air"] = air_density
    vehicle = replace(vehicle, **changes)
    tires = TireParameters()
    scale = params.get("mu_scale", 1.0)
    tires = replace(tires, **{k: getattr(tires, k) * scale for k in ("mux_f", "muy_f", "mux_r", "muy_r")})
    return vehicle, tires


def model_lap(params: Mapping[str, float], reference: ReferenceLap):
    """The model's optimal qualifying lap with these parameters, on the reference lap's path and weather."""
    vehicle, tires = car_for(params, reference.air_density)
    ers = get_ers_config("2026", session="qualifying", event=reference.round)
    solver = SpatialNLPSolver(VehicleDynamicsModel(vehicle, ers, tires), reference.track, ers)
    solver.verbose = False
    with contextlib.redirect_stdout(io.StringIO()):
        return solver.solve()


def model_speed_at(trajectory, s, length: float) -> np.ndarray:
    """Model speed (m/s) at lap distances s (the timed lap, from the line)."""
    return np.interp(np.mod(s, length), trajectory.s, trajectory.v_opt, period=length)
