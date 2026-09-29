"""
One car model for the optimiser, the forward-backward pass and the time-domain simulator.

A point mass on a fixed path, with the tyre forces resolved per axle:
- aero drag and downforce per aero mode (w = 0 is Corner Mode, w = 1 Straight Mode), with a front/rear split;
- axle normal loads from weight, downforce and longitudinal load transfer;
- axle lateral forces from the steady yaw balance;
- a load-sensitive friction ellipse per axle, with traction on the rear axle only and brakes on both;
- ICE and MGU-K power, and the battery.

Every method works on floats, NumPy arrays and CasADi symbols and uses only smooth operations,
so the optimiser can use the same equations inside Ipopt.

References: TUMFTM laptime-simulation (tyre model and parameters); Duhr, "Time-optimal operation of the
hybrid Formula 1 powertrain", ETH Diss. 28783, 2022 (per-axle force limits with separate power limits).
"""
from dataclasses import dataclass
from typing import Any, Optional

import casadi as ca
import numpy as np

from config import ERSConfig, TireParameters, VehicleConfig

# Upper speed bound shared by the solvers (m/s, ~396 km/h)
V_MAX = 110.0


def _is_casadi(value) -> bool:
    return isinstance(value, (ca.SX, ca.MX, ca.DM))


def _sin(x):
    return ca.sin(x) if _is_casadi(x) else np.sin(x)


def _cos(x):
    return ca.cos(x) if _is_casadi(x) else np.cos(x)


def _abs(x):
    return ca.fabs(x) if _is_casadi(x) else np.abs(x)


def deploy_power_limit(v, ers: ERSConfig):
    """
    Maximum MGU-K deployment power (W) at speed v (m/s). Works on floats and CasADi expressions.

    2026: 350 kW up to 290 km/h, then 1800 - 5v kW, then 6900 - 20v kW from 340 km/h,
    reaching 0 at 345 km/h (v in km/h). Earlier years: a flat limit.
    """
    if ers.regulation_year < 2026:
        return ers.max_deployment_power

    v_kph = v * 3.6
    p_taper1 = (1800.0 - 5.0 * v_kph) * 1000.0    # Linear drop 290-340 km/h
    p_taper2 = (6900.0 - 20.0 * v_kph) * 1000.0   # Sharp drop 340-345 km/h
    return ca.fmax(0, ca.fmin(ers.max_deployment_power, ca.fmin(p_taper1, p_taper2)))


def air_density(temp_c: float, pressure_hpa: float, humidity_pct: float = 50.0) -> float:
    """Density of moist air (kg/m³) from temperature, pressure and relative humidity (Tetens saturation pressure)."""
    temp_k = temp_c + 273.15
    vapour_hpa = humidity_pct / 100.0 * 6.1078 * 10.0 ** (7.5 * temp_c / (temp_c + 237.3))
    dry_hpa = pressure_hpa - vapour_hpa
    return (dry_hpa * 100.0) / (287.058 * temp_k) + (vapour_hpa * 100.0) / (461.495 * temp_k)


@dataclass
class PointForces:
    """Acceleration and per-axle tyre forces at one point on the path (N, m/s²)."""
    a_x: Any                # Longitudinal acceleration
    F_z_front: Any          # Normal loads
    F_z_rear: Any
    F_x_front: Any          # Longitudinal tyre forces (positive forwards)
    F_x_rear: Any
    F_y_front: Any          # Lateral tyre forces
    F_y_rear: Any
    F_x_max_front: Any      # Friction limits
    F_y_max_front: Any
    F_x_max_rear: Any
    F_y_max_rear: Any

    @property
    def usage_front(self):
        """Front friction-ellipse usage: feasible when ≤ 1."""
        return CarModel.grip_usage(self.F_x_front, self.F_y_front, self.F_x_max_front, self.F_y_max_front)

    @property
    def usage_rear(self):
        """Rear friction-ellipse usage: feasible when ≤ 1."""
        return CarModel.grip_usage(self.F_x_rear, self.F_y_rear, self.F_x_max_rear, self.F_y_max_rear)


class CarModel:
    """Vehicle equations shared by every solver. See the module docstring."""

    def __init__(self, vehicle: VehicleConfig, tires: Optional[TireParameters] = None, ers: Optional[ERSConfig] = None):
        self.vehicle = vehicle
        self.tires = tires or TireParameters()
        self.ers = ers or ERSConfig()

        self.mass = vehicle.mass + vehicle.fuel_mass
        wheelbase = vehicle.lf + vehicle.lr
        self.front_weight_share = vehicle.lr / wheelbase          # Static load on the front axle
        self.front_lateral_share = vehicle.lr / wheelbase         # Yaw balance: F_yf·lf = F_yr·lr
        self.front_downforce_share = vehicle.c_z_a_f / (vehicle.c_z_a_f + vehicle.c_z_a_r)
        self.load_transfer = self.mass * vehicle.h_cog / wheelbase  # Load moved to the rear axle per m/s² of a_x

    # ------------------------------------------------------------------ aero
    def aero_mode(self, radius):
        """
        Straight-Mode fraction w at each track point (0 = Corner Mode, 1 = Straight Mode).

        Until the FIA activation zones are modelled (ROADMAP Phase 2, REG-5), 2026 cars use Straight Mode
        wherever the radius exceeds straight_mode_min_radius. Earlier cars have one mode.
        """
        threshold = self.vehicle.straight_mode_min_radius
        if _is_casadi(radius):
            return 0.0 if self.vehicle.regulation_year < 2026 else ca.if_else(ca.fabs(radius) > threshold, 1.0, 0.0)
        radius = np.asarray(radius, dtype=float)
        if self.vehicle.regulation_year < 2026:
            return np.zeros_like(radius)
        return (np.abs(radius) > threshold).astype(float)

    def drag_area(self, w=0.0):
        """Drag area CdA (m²) in aero mode w."""
        veh = self.vehicle
        return veh.c_w_a * (1.0 + (veh.straight_mode_drag_factor - 1.0) * w)

    def downforce_area(self, w=0.0):
        """Downforce area ClA (m²) in aero mode w."""
        veh = self.vehicle
        return (veh.c_z_a_f + veh.c_z_a_r) * (1.0 + (veh.straight_mode_downforce_factor - 1.0) * w)

    def downforce(self, v, w=0.0):
        """Total aerodynamic downforce (N)."""
        return 0.5 * self.vehicle.rho_air * v**2 * self.downforce_area(w)

    def resistance(self, v, kappa=0.0, gradient=0.0, w=0.0):
        """Forces opposing motion (N): drag (plus the optional cornering term), rolling resistance and the slope."""
        veh = self.vehicle
        drag = 0.5 * veh.rho_air * v**2 * self.drag_area(w) + veh.cornering_drag_coeff * _abs(kappa) * v**2
        rolling = veh.f_roll * (self.mass * veh.g * _cos(gradient) + self.downforce(v, w))
        slope = self.mass * veh.g * _sin(gradient)
        return drag + rolling + slope

    # ----------------------------------------------------------------- tyres
    def axle_loads(self, v, a_x, gradient=0.0, w=0.0):
        """Normal load on the front and rear axle (N). Accelerating moves load to the rear."""
        veh = self.vehicle
        weight = self.mass * veh.g * _cos(gradient)
        downforce = self.downforce(v, w)
        transfer = self.load_transfer * a_x
        F_z_front = self.front_weight_share * weight + self.front_downforce_share * downforce - transfer
        F_z_rear = (1.0 - self.front_weight_share) * weight + (1.0 - self.front_downforce_share) * downforce + transfer
        return F_z_front, F_z_rear

    def axle_lateral_forces(self, v, kappa):
        """Lateral force each axle supplies in a steady turn of curvature kappa (N)."""
        F_y = self.mass * kappa * v**2
        return self.front_lateral_share * F_y, (1.0 - self.front_lateral_share) * F_y

    def axle_grip(self, F_z_front, F_z_rear, grip_front=1.0, grip_rear=1.0):
        """
        Longitudinal and lateral friction limits per axle (N): (F_x_max_front, F_y_max_front, F_x_max_rear, F_y_max_rear).

        Load-sensitive friction per tyre, mu = mu0 + dmu/dFz·(F_z_tyre − F_z0), with two tyres per axle.
        grip_front and grip_rear scale the friction (tyre temperature, wear or a track grip level).
        """
        t = self.tires

        def limit(F_z, mu0, dmu_dfz, scale):
            return scale * (mu0 + dmu_dfz * (0.5 * F_z - t.fz_0)) * F_z

        return (
            limit(F_z_front, t.mux_f, t.dmux_dfz_f, grip_front),
            limit(F_z_front, t.muy_f, t.dmuy_dfz_f, grip_front),
            limit(F_z_rear, t.mux_r, t.dmux_dfz_r, grip_rear),
            limit(F_z_rear, t.muy_r, t.dmuy_dfz_r, grip_rear),
        )

    @staticmethod
    def grip_usage(F_x, F_y, F_x_max, F_y_max):
        """Friction-ellipse usage (F_x/F_x_max)² + (F_y/F_y_max)²: feasible when ≤ 1."""
        return (F_x / F_x_max) ** 2 + (F_y / F_y_max) ** 2

    # ------------------------------------------------------------ powertrain
    def drive_force(self, v, throttle, p_deploy, p_harvest):
        """Net powertrain force at the rear wheels (N): ICE plus MGU-K deploy, minus MGU-K harvest (powers in W)."""
        return (throttle * self.vehicle.pow_max_ice + p_deploy - p_harvest) / v

    def battery_power(self, p_deploy, p_harvest):
        """Power drawn from the battery (W) for the given MGU-K deploy and harvest power at the wheels."""
        return p_deploy / self.ers.deployment_efficiency - p_harvest * self.ers.recovery_efficiency

    # -------------------------------------------------------------- the point
    def point(self, v, kappa, gradient, w, drive_force, brake_front, brake_rear, grip_front=1.0, grip_rear=1.0) -> PointForces:
        """
        Acceleration and per-axle forces at one point on the path.

        drive_force is the net powertrain force at the rear wheels; brake_front and brake_rear are friction
        brake forces (N, ≥ 0). The normal loads use the resulting acceleration, so the load transfer is exact.
        """
        F_x_front = -brake_front
        F_x_rear = drive_force - brake_rear
        a_x = (F_x_front + F_x_rear - self.resistance(v, kappa, gradient, w)) / self.mass
        F_z_front, F_z_rear = self.axle_loads(v, a_x, gradient, w)
        F_y_front, F_y_rear = self.axle_lateral_forces(v, kappa)
        F_x_max_front, F_y_max_front, F_x_max_rear, F_y_max_rear = self.axle_grip(F_z_front, F_z_rear, grip_front, grip_rear)
        return PointForces(
            a_x=a_x,
            F_z_front=F_z_front, F_z_rear=F_z_rear,
            F_x_front=F_x_front, F_x_rear=F_x_rear,
            F_y_front=F_y_front, F_y_rear=F_y_rear,
            F_x_max_front=F_x_max_front, F_y_max_front=F_y_max_front,
            F_x_max_rear=F_x_max_rear, F_y_max_rear=F_y_max_rear,
        )

    def longitudinal_limits(self, v, kappa, gradient, w, a_x, grip_front=1.0, grip_rear=1.0):
        """
        Longitudinal force each axle has left after its lateral force, at acceleration a_x (N, NumPy only).

        Returns (lateral_ok, available_front, available_rear); lateral_ok is False where an axle can't even carry
        its lateral force.
        """
        F_z_front, F_z_rear = self.axle_loads(v, a_x, gradient, w)
        F_y_front, F_y_rear = self.axle_lateral_forces(v, kappa)
        F_x_max_front, F_y_max_front, F_x_max_rear, F_y_max_rear = self.axle_grip(F_z_front, F_z_rear, grip_front, grip_rear)
        lateral_front = (F_y_front / F_y_max_front) ** 2
        lateral_rear = (F_y_rear / F_y_max_rear) ** 2
        lateral_ok = (lateral_front <= 1.0) & (lateral_rear <= 1.0) & (F_z_front > 0.0) & (F_z_rear > 0.0)
        available_front = F_x_max_front * np.sqrt(np.clip(1.0 - lateral_front, 0.0, None))
        available_rear = F_x_max_rear * np.sqrt(np.clip(1.0 - lateral_rear, 0.0, None))
        return lateral_ok, available_front, available_rear

    def is_feasible(self, v, kappa, gradient, w, a_x, max_power, grip_front=1.0, grip_rear=1.0):
        """
        Whether acceleration a_x is possible at speed v (NumPy, vectorised).

        Traction goes through the rear axle only, up to max_power (W). Braking is split between the axles
        as needed, up to the brake system's max_brake_force.
        """
        needed = self.mass * a_x + self.resistance(v, kappa, gradient, w)   # Net tyre force (N)
        lateral_ok, available_front, available_rear = self.longitudinal_limits(
            v, kappa, gradient, w, a_x, grip_front, grip_rear
        )
        traction_ok = needed <= np.minimum(available_rear, max_power / v)
        braking_ok = -needed <= np.minimum(available_front + available_rear, self.vehicle.max_brake_force)
        return lateral_ok & np.where(needed >= 0.0, traction_ok, braking_ok)
