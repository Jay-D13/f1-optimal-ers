import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from calibration.dataset import (TOW_DISTANCE, TOW_SECONDS, ReferenceLap, distance_ahead, in_pits,
                                 tow_exposure)
from calibration.model import PARAMETERS
from config import get_vehicle_config


class TowExposureTests(unittest.TestCase):
    """The tow detector on a synthetic lap sampled every 0.25 s for 60 s."""

    def setUp(self):
        self.seconds = np.arange(0.0, 60.0, 0.25)
        t = self.seconds
        self.full = ((t >= 10.0) & (t < 30.0)) | ((t >= 40.0) & (t < 50.0))

    def test_a_car_close_ahead_at_full_throttle_is_a_tow(self):
        t = self.seconds
        aaa = np.where((t >= 12.0) & (t < 17.0), 50.0, 200.0)       # 5 s within 80 m, at full throttle
        bbb = np.where((t >= 32.0) & (t < 38.0), 30.0, np.inf)      # Close, but while braking and cornering
        tow = tow_exposure(t, self.full, {"AAA": aaa, "BBB": bbb})
        self.assertAlmostEqual(tow.seconds, 5.0, delta=0.01)
        self.assertGreater(tow.seconds, TOW_SECONDS)
        self.assertEqual(tow.driver, "AAA")
        self.assertEqual(tow.gap, 50.0)

    def test_a_short_pass_is_not_a_tow(self):
        t = self.seconds
        # A slow car passed on the straight: 2 s within 80 m, then behind (just under a lap ahead)
        gap = np.interp(t, [40.0, 41.0, 42.0, 42.25], [120.0, 40.0, 1.0, 5000.0])
        gap[t >= 42.25] = 5000.0
        tow = tow_exposure(t, self.full, {"SLO": gap, "FAR": np.full(len(t), 300.0)})
        self.assertLess(tow.seconds, TOW_SECONDS)
        self.assertGreater(tow.seconds, 1.0)
        self.assertEqual(tow.driver, "SLO")

    def test_without_a_car_close_the_driver_is_the_nearest_ahead(self):
        t = self.seconds
        gaps = {"AAA": np.full(len(t), 400.0), "BBB": np.where(t < 20.0, 150.0, np.nan)}
        tow = tow_exposure(t, self.full, gaps)
        self.assertEqual(tow.seconds, 0.0)
        self.assertEqual(tow.driver, "BBB")
        self.assertEqual(tow.gap, 150.0)
        self.assertEqual(tow_exposure(t, self.full, {}), (0.0, "", None))

    def test_only_the_nearest_car_counts(self):
        t = self.seconds
        inside = (t >= 10.0) & (t < 14.0)
        tow = tow_exposure(t, self.full, {"AAA": np.where(inside, 20.0, np.inf), "BBB": np.where(inside, 60.0, np.inf)})
        self.assertAlmostEqual(tow.seconds, 4.0, delta=0.01)       # Not 8: two cars ahead don't double the time
        self.assertEqual(tow.driver, "AAA")


class DistanceAheadTests(unittest.TestCase):
    def test_distance_along_a_closed_line(self):
        radius = 500.0
        length = 2 * np.pi * radius
        line_angle = np.arange(0.0, 2 * np.pi, 20.0 / radius)       # One position every 20 m, as at speed
        line = radius * np.column_stack([np.cos(line_angle), np.sin(line_angle)])
        own_angle = np.array([0.1, 3.0, 6.2])
        on_circle = lambda a, r=radius: r * np.column_stack([np.cos(a), np.sin(a)])
        own = on_circle(own_angle)
        np.testing.assert_allclose(distance_ahead(line, own, on_circle(own_angle + 60.0 / radius)), 60.0, atol=1.5)
        # Behind by 30 m: almost a lap ahead
        np.testing.assert_allclose(distance_ahead(line, own, on_circle(own_angle - 30.0 / radius)), length - 30.0, atol=1.5)
        # 10 m off the line (the pit lane): not on track
        self.assertTrue(np.all(np.isinf(distance_ahead(line, own, on_circle(own_angle + 0.1, radius + 10.0)))))


class PitTests(unittest.TestCase):
    def test_in_pits_between_pit_in_and_pit_out(self):
        s = lambda *v: pd.Series(pd.to_timedelta(v, unit="s"))
        laps = pd.DataFrame({"PitOutTime": s(100.0, None, None, 500.0), "PitInTime": s(None, None, 400.0, None)})
        seconds = np.array([50.0, 150.0, 399.0, 450.0, 600.0])
        np.testing.assert_array_equal(in_pits(laps, seconds), [True, False, False, True, False])
        empty = pd.DataFrame({"PitOutTime": s(None), "PitInTime": s(None)})
        self.assertFalse(in_pits(empty, seconds).any())


class CacheTests(unittest.TestCase):
    def test_version_1_pickles_load_without_a_tow_figure(self):
        reference = ReferenceLap.__new__(ReferenceLap)          # As unpickled from a cache without the fields
        self.assertIsNone(reference.tow_seconds)
        self.assertIsNone(reference.frozen_share)
        self.assertFalse(reference.towed)


class StartValueTests(unittest.TestCase):
    def test_physical_start_values_and_bounds(self):
        params = {p.name: p for p in PARAMETERS}
        self.assertEqual((params["pow_max_ice"].start, params["pow_max_ice"].lower, params["pow_max_ice"].upper),
                         (420e3, 380e3, 440e3))
        self.assertEqual((params["c_w_a"].lower, params["c_w_a"].upper), (0.85, 1.05))

    def test_qualifying_fuel_mass(self):
        self.assertEqual(get_vehicle_config("2026").fuel_mass, 4.0)
        self.assertEqual(get_vehicle_config("2025").fuel_mass, 0.0)


if __name__ == "__main__":
    unittest.main()
