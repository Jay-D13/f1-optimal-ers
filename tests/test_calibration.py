import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from calibration.dataset import good_rounds, lap_distances
from calibration.fit import Fit
from calibration.model import PARAMETERS, car_for, start_values
from models import F1TrackModel
from models.geometry import TrackGeometry


def circle_track(radius=500.0, ds=5.0):
    """A 2π·500 m circle driven anticlockwise, starting at (radius, 0)."""
    n = 3142
    s = np.arange(n) * (2 * np.pi * radius / n)
    angle = s / radius
    geometry = TrackGeometry(s, radius * np.cos(angle), radius * np.sin(angle), np.zeros(n), np.full(n, 1 / radius),
                             np.zeros(n), np.zeros(n), 2 * np.pi * radius, "circle")
    return F1TrackModel(2026, "circle", ds=ds).load_from_geometry(geometry)


class DatasetTests(unittest.TestCase):
    def test_good_rounds_have_racelines(self):
        rounds = good_rounds()
        self.assertEqual(sorted(rounds), [2, 3, 5, 8, 9, 10, 11, 13])

    def test_lap_distances_run_through_the_line_and_skip_strays(self):
        track = circle_track()
        length = track.total_length
        # A lap at 50 m/s that starts 40 m before the line, sampled every 0.25 s, with one stray position
        seconds = np.arange(0.0, length / 50.0, 0.25)
        s_true = seconds * 50.0 - 40.0
        angle = s_true / 500.0
        xy = np.column_stack([500 * np.cos(angle), 500 * np.sin(angle)])
        xy[100] = [-500.0, 0.0]                                  # Half a lap away
        s = lap_distances(xy, np.full(len(seconds), 50.0), seconds, track)
        self.assertLess(np.abs(s - s_true).max(), 3.0)
        self.assertTrue(np.all(np.diff(s) > 0))


class ModelTests(unittest.TestCase):
    def test_car_for_maps_the_parameters(self):
        params = dict(start_values(), c_z_a=4.0, front_downforce_share=0.4, mu_scale=1.1, c_w_a=1.0)
        vehicle, tires = car_for(params, air_density=1.1)
        self.assertAlmostEqual(vehicle.c_z_a_f, 1.6)
        self.assertAlmostEqual(vehicle.c_z_a_r, 2.4)
        self.assertAlmostEqual(vehicle.c_w_a, 1.0)
        self.assertAlmostEqual(vehicle.rho_air, 1.1)
        self.assertEqual(vehicle.regulation_year, 2026)
        self.assertAlmostEqual(tires.muy_r, 2.15 * 1.1)

    def test_start_values_are_inside_the_bounds(self):
        for p in PARAMETERS:
            self.assertLess(p.lower, p.start)
            self.assertLess(p.start, p.upper)


class FitTests(unittest.TestCase):
    def test_parameters_round_trip_through_unit_scale(self):
        fit = Fit([13], ["c_w_a", "mu_scale"])
        x = fit.x_of(fit.fixed)
        self.assertTrue(np.all((x > 0) & (x < 1)))
        values = fit.params(np.array([0.25, 0.75]))
        self.assertAlmostEqual(values["c_w_a"], 0.7 + 0.25 * 0.6)
        self.assertAlmostEqual(values["mu_scale"], 0.8 + 0.75 * 0.5)
        self.assertAlmostEqual(values["c_z_a"], 3.45)                # Not fitted: start value


if __name__ == "__main__":
    unittest.main()
