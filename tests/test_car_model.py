"""The shared car model (models/car.py) and the solvers that use it."""
import unittest

import casadi as ca
import numpy as np

from config import ERSConfig, TireParameters, VehicleConfig, get_ers_config, get_vehicle_config
from models import CarModel, VehicleDynamicsModel, air_density
from solvers.forward_backward import V_GRID, ForwardBackwardSolver


class _Track:
    """Minimal track: constant radius and gradient."""

    class _Data:
        pass

    def __init__(self, radius: float, length: float, ds: float = 5.0, gradient: float = 0.0):
        n = int(length / ds)
        data = self._Data()
        data.s = np.arange(n) * ds
        data.radius = np.full(n, radius)
        data.gradient = np.full(n, gradient)
        data.ds = ds
        data.total_length = n * ds
        self.track_data = data
        self.total_length = data.total_length


class CarModelTests(unittest.TestCase):
    def setUp(self):
        self.car = CarModel(VehicleConfig(), TireParameters(), ERSConfig())

    def test_loads_conserve_weight_and_downforce(self):
        for a_x in (-40.0, 0.0, 12.0):
            F_zf, F_zr = self.car.axle_loads(70.0, a_x)
            self.assertAlmostEqual(F_zf + F_zr, self.car.mass * 9.81 + self.car.downforce(70.0), places=6)
        # Accelerating moves load to the rear
        self.assertLess(self.car.axle_loads(70.0, 10.0)[0], self.car.axle_loads(70.0, 0.0)[0])

    def test_lateral_forces_balance(self):
        F_yf, F_yr = self.car.axle_lateral_forces(50.0, 1 / 120.0)
        self.assertAlmostEqual(F_yf + F_yr, self.car.mass * 50.0**2 / 120.0, places=6)
        veh = self.car.vehicle
        self.assertAlmostEqual(F_yf * veh.lf, F_yr * veh.lr, places=6)   # No yaw moment

    def test_casadi_and_numpy_agree(self):
        v, kappa, drive, bf, br = ca.SX.sym("v"), ca.SX.sym("kappa"), ca.SX.sym("drive"), ca.SX.sym("bf"), ca.SX.sym("br")
        sym = self.car.point(v, kappa, 0.02, 1.0, drive, bf, br, 0.97, 1.01)
        f = ca.Function("f", [v, kappa, drive, bf, br], [sym.a_x, sym.usage_front, sym.usage_rear])
        args = (63.0, 1 / 150.0, 9000.0, 4000.0, 1500.0)
        num = self.car.point(*args[:2], 0.02, 1.0, *args[2:], 0.97, 1.01)
        for got, expected in zip(f(*args), (num.a_x, num.usage_front, num.usage_rear)):
            self.assertAlmostEqual(float(got), float(expected), places=9)

    def test_air_density(self):
        self.assertAlmostEqual(air_density(15.0, 1013.25, 0.0), 1.225, delta=0.001)   # ISA sea level
        self.assertAlmostEqual(air_density(25.0, 1013.25, 50.0), 1.177, delta=0.002)
        self.assertLess(air_density(20.0, 780.0, 40.0), 0.95)                         # Mexico City altitude

    def test_straight_mode_only_for_2026(self):
        radius = np.array([50.0, 399.0, 401.0, 5000.0])
        np.testing.assert_array_equal(CarModel(get_vehicle_config("2025")).aero_mode(radius), [0, 0, 0, 0])
        np.testing.assert_array_equal(CarModel(get_vehicle_config("2026")).aero_mode(radius), [0, 0, 1, 1])


class SharedForcesTests(unittest.TestCase):
    """The forward-backward pass, the NLP and the time-domain simulator use the same forces."""

    def setUp(self):
        self.model = VehicleDynamicsModel(get_vehicle_config("2026"), get_ers_config("2026"))
        self.car = self.model.car

    def test_simulator_matches_car_model(self):
        # Inputs inside every axle's grip, so traction control and ABS stay inactive
        v, radius, gradient, throttle, p_ers, brake = 60.0, 300.0, 0.01, 0.6, 150e3, 0.0
        x_dot = self.model.create_time_domain_dynamics()([0.0, v, 0.5], [p_ers, throttle, brake], [gradient, radius])
        drive = self.car.drive_force(v, throttle, p_ers, 0.0)
        expected = self.car.point(v, 1 / radius, gradient, self.car.aero_mode(radius), drive, 0.0, 0.0)
        self.assertLess(float(expected.usage_rear), 1.0)
        self.assertAlmostEqual(float(x_dot[1]), float(expected.a_x), places=9)
        battery = self.car.battery_power(p_ers, 0.0)
        self.assertAlmostEqual(float(x_dot[2]), -battery / self.model.ers.battery_capacity, places=12)

    def test_forward_backward_limits_use_the_grip_ellipse(self):
        # The traction limit uses the rear ellipse fully unless the power runs out first;
        # the braking limit uses an axle ellipse fully unless the brakes run out first
        fb = ForwardBackwardSolver(self.model, None)
        veh = self.car.vehicle
        for v, radius in ((25.0, 40.0), (50.0, 150.0), (80.0, 1000.0)):
            with self.subTest(v=v, radius=radius):
                kappa = 1.0 / radius
                a_max, a_min = fb._acceleration_limits(np.array([kappa]), np.zeros(1), np.zeros(1))
                a_up = float(np.interp(v, V_GRID, a_max[0]))
                a_down = float(np.interp(v, V_GRID, a_min[0]))
                resistance = self.car.resistance(v, kappa, 0.0, 0.0)

                F_up = self.car.mass * a_up + resistance
                up = self.car.point(v, kappa, 0.0, 0.0, F_up, 0.0, 0.0)
                grip_bound = abs(float(up.usage_rear) - 1.0) < 0.01
                power_bound = abs(F_up * v / veh.pow_max_ice - 1.0) < 0.01
                self.assertTrue(grip_bound or power_bound, (float(up.usage_rear), F_up * v / veh.pow_max_ice))

                # Split the braking force like the NLP may: in proportion to what each axle has left
                _, avail_f, avail_r = self.car.longitudinal_limits(v, kappa, 0.0, 0.0, a_down)
                F_down = -(self.car.mass * a_down + resistance)
                down = self.car.point(
                    v, kappa, 0.0, 0.0, 0.0, F_down * avail_f / (avail_f + avail_r), F_down * avail_r / (avail_f + avail_r)
                )
                grip_bound = abs(max(float(down.usage_front), float(down.usage_rear)) - 1.0) < 0.01
                brake_bound = abs(F_down / veh.max_brake_force - 1.0) < 0.01
                self.assertTrue(grip_bound or brake_bound)

    def test_steady_cornering_speed(self):
        # On a circle, the cornering speed holds speed with the rear ellipse at its limit
        fb = ForwardBackwardSolver(self.model, _Track(radius=100.0, length=2 * np.pi * 100.0))
        v = float(fb._cornering_speeds(np.array([0.01]), np.zeros(1), np.zeros(1))[0])
        resistance = self.car.resistance(v, 0.01, 0.0, 0.0)
        at_limit = self.car.point(v, 0.01, 0.0, 0.0, resistance, 0.0, 0.0)
        self.assertAlmostEqual(float(at_limit.a_x), 0.0, places=9)
        self.assertAlmostEqual(max(float(at_limit.usage_front), float(at_limit.usage_rear)), 1.0, delta=1e-6)


if __name__ == "__main__":
    unittest.main()
