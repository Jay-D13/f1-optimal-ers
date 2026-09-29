"""
Vehicle dynamics in the time and spatial domains, on the shared car model (models/car.py).

The time-domain function drives the lap simulator and the baseline strategies. Its forces come from the
same CarModel as the NLP and the forward-backward pass; on top, it saturates commands the way a car's
electronics would (deploy curve, traction control, ABS). What the simulator still lacks (exact lap end,
clipping, a lateral grip check) is ROADMAP Phase 5 (SIM-1).

References:
- TUMFTM laptime-simulation (car.py, car_hybrid.py): https://github.com/TUMFTM/laptime-simulation
- Heilmeier et al. "Application of Monte Carlo Methods" (2020)
"""
import casadi as ca
from typing import Dict, Optional

from config import VehicleConfig, ERSConfig, TireParameters
from models.car import CarModel, V_MAX, deploy_power_limit


class VehicleDynamicsModel:
    """
    Vehicle model shared by every solver.

    `car` holds the equations (CarModel); this class adds the time- and spatial-domain CasADi functions
    used by the simulator.
    """

    def __init__(self,
                 vehicle_params: VehicleConfig,
                 ers_config: ERSConfig,
                 tire_params: Optional[TireParameters] = None):

        self.vehicle = vehicle_params
        self.ers = ers_config
        self.tires = tire_params or TireParameters()
        self.car = CarModel(self.vehicle, self.tires, self.ers)

        # Cache for CasADi functions
        self._dynamics_time_func = None
        self._dynamics_spatial_func = None

    def create_time_domain_dynamics(self) -> ca.Function:
        """
        Create CasADi function for time-domain dynamics.

        State: x = [s, v, soc]
        Control: u = [P_ers, throttle, brake]  (P_ers > 0 deploys, < 0 harvests; brake is a fraction of max_brake_force)
        Parameters: p = [gradient, radius]

        Returns: dx/dt
        """
        if self._dynamics_time_func is not None:
            return self._dynamics_time_func

        s = ca.MX.sym('s')
        v = ca.MX.sym('v')
        soc = ca.MX.sym('soc')
        P_ers = ca.MX.sym('P_ers')
        throttle = ca.MX.sym('throttle')
        brake = ca.MX.sym('brake')
        gradient = ca.MX.sym('gradient')
        radius = ca.MX.sym('radius')

        car = self.car
        veh = self.vehicle
        ers = self.ers
        v_safe = ca.fmax(v, 5.0)
        kappa = 1.0 / ca.fmax(ca.fabs(radius), 1.0)
        w = car.aero_mode(radius)

        # ERS: deploy capped by the speed-dependent limit, harvest by the MGU-K limit
        P_deploy = ca.fmin(ca.fmax(P_ers, 0), deploy_power_limit(v_safe, ers))
        P_harvest = ca.fmin(ca.fmax(-P_ers, 0), ers.max_recovery_power)

        # Commanded forces, then each axle capped by the grip its ellipse leaves after the lateral force
        # (traction control and ABS). The loads use the commanded acceleration.
        F_drive = car.drive_force(v_safe, throttle, P_deploy, P_harvest)
        F_brake = brake * veh.max_brake_force
        F_brake_front = veh.brake_balance_front * F_brake
        F_brake_rear = (1.0 - veh.brake_balance_front) * F_brake
        cmd = car.point(v_safe, kappa, gradient, w, F_drive, F_brake_front, F_brake_rear)

        def available(F_x_max, F_y, F_y_max):
            return F_x_max * ca.sqrt(ca.fmax(1.0 - (F_y / F_y_max) ** 2, 0.0))

        avail_front = available(cmd.F_x_max_front, cmd.F_y_front, cmd.F_y_max_front)
        avail_rear = available(cmd.F_x_max_rear, cmd.F_y_rear, cmd.F_y_max_rear)
        F_x_front = -ca.fmin(F_brake_front, avail_front)
        F_x_rear = ca.fmax(ca.fmin(cmd.F_x_rear, avail_rear), -avail_rear)

        dv_dt = (F_x_front + F_x_rear - car.resistance(v_safe, kappa, gradient, w)) / car.mass
        dsoc_dt = -car.battery_power(P_deploy, P_harvest) / ers.battery_capacity

        x = ca.vertcat(s, v, soc)
        u = ca.vertcat(P_ers, throttle, brake)
        p = ca.vertcat(gradient, radius)
        x_dot = ca.vertcat(v, dv_dt, dsoc_dt)

        self._dynamics_time_func = ca.Function(
            'dynamics_time', [x, u, p], [x_dot],
            ['x', 'u', 'p'], ['x_dot']
        )

        return self._dynamics_time_func

    def create_spatial_domain_dynamics(self) -> ca.Function:
        """
        Create CasADi function for spatial-domain dynamics.

        State: x = [v, soc]
        Control: u = [P_ers, throttle, brake]
        Parameters: p = [gradient, radius]

        Returns: [dx/ds, dt/ds]
        """
        if self._dynamics_spatial_func is not None:
            return self._dynamics_spatial_func

        time_dynamics = self.create_time_domain_dynamics()

        v = ca.MX.sym('v')
        soc = ca.MX.sym('soc')
        P_ers = ca.MX.sym('P_ers')
        throttle = ca.MX.sym('throttle')
        brake = ca.MX.sym('brake')
        gradient = ca.MX.sym('gradient')
        radius = ca.MX.sym('radius')

        u = ca.vertcat(P_ers, throttle, brake)
        p = ca.vertcat(gradient, radius)
        x_dot = time_dynamics(ca.vertcat(0, v, soc), u, p)

        # dx/ds = (dx/dt) / (ds/dt)
        v_safe = ca.fmax(v, 5.0)
        dx_ds = ca.vertcat(x_dot[1] / v_safe, x_dot[2] / v_safe)
        dt_ds = 1.0 / v_safe

        self._dynamics_spatial_func = ca.Function(
            'dynamics_spatial',
            [ca.vertcat(v, soc), u, p], [dx_ds, dt_ds],
            ['x', 'u', 'p'], ['dx_ds', 'dt_ds']
        )

        return self._dynamics_spatial_func

    def get_constraints(self) -> Dict:
        """Return system constraints"""
        return {
            # ERS power limits
            'P_ers_min': -self.ers.max_recovery_power,
            'P_ers_max': self.ers.max_deployment_power,

            # SOC limits
            'soc_min': self.ers.min_soc,
            'soc_max': self.ers.max_soc,

            # Velocity limits
            'v_min': 15.0,   # m/s (~54 km/h)
            'v_max': V_MAX,

            # Control limits
            'throttle_min': 0.0,
            'throttle_max': 1.0,
            'brake_min': 0.0,
            'brake_max': 1.0,
        }
