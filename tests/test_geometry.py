import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import get_vehicle_config, get_ers_config
from models import CarModel, F1TrackModel
from models.geometry import TrackGeometry, _PeriodicSpline, fit_track
from models import telemetry


def oval_laps(n_laps=100, spacing=15.0, noise=0.3, seed=0):
    """Noisy laps of an ellipse (semi-axes 400 m and 150 m) with a 10 m hill, driven anticlockwise."""
    rng = np.random.default_rng(seed)
    a, b = 400.0, 150.0
    t = np.linspace(0, 2 * np.pi, 20001)[:-1]
    points = np.column_stack([a * np.cos(t), b * np.sin(t)])
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(np.vstack([points, points[:1]]), axis=0), axis=1))])
    length = s[-1]
    laps, speeds = [], []
    for _ in range(n_laps):
        s_samples = np.sort(rng.uniform(0, length, int(length / spacing)))
        u = np.interp(s_samples, s[:-1], t)
        xyz = np.column_stack([a * np.cos(u), b * np.sin(u), 5.0 * np.sin(2 * np.pi * s_samples / length)])
        laps.append(xyz + rng.normal(0, noise, xyz.shape) * [1, 1, 0.3])
        speeds.append(np.full(len(s_samples), 60.0))
    return laps, speeds, length, (a, b)


class PeriodicSplineTests(unittest.TestCase):
    def test_smooth_periodic_function_and_seam(self):
        rng = np.random.default_rng(1)
        length = 1000.0
        s = rng.uniform(0, length, 4000)
        values = np.sin(2 * np.pi * 3 * s / length) + rng.normal(0, 0.05, len(s))
        spline = _PeriodicSpline(s, values, length, smoothing=50.0)
        x = np.linspace(0, length, 500, endpoint=False)
        self.assertLess(np.sqrt(np.mean((spline(x) - np.sin(2 * np.pi * 3 * x / length)) ** 2)), 0.02)
        # Value and derivatives match across the start/finish seam
        for der in range(3):
            self.assertAlmostEqual(float(spline(0.0, der)), float(spline(length - 1e-9, der)), places=6)

    def test_smoothing_follows_the_local_length(self):
        # A short bump survives where the local smoothing is short, and is flattened where it is long
        length = 2000.0
        s = np.linspace(0, length, 8000, endpoint=False)
        bump = lambda centre: np.exp(-0.5 * ((s - centre) / 8.0) ** 2)
        values = bump(500.0) + bump(1500.0)
        local = np.where(s < length / 2, 10.0, 200.0)
        spline = _PeriodicSpline(s, values, length, smoothing=local)
        self.assertGreater(float(spline(500.0)), 0.8)
        self.assertLess(float(spline(1500.0)), 0.3)


class FitTrackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        laps, speeds, cls.length, (cls.a, cls.b) = oval_laps()
        cls.geometry = fit_track(laps, speeds=speeds, source="oval")

    def test_closed_line_and_length(self):
        g = self.geometry
        self.assertAlmostEqual(g.heading_turns, 1.0, places=3)      # Anticlockwise: one left turn
        self.assertLess(abs(g.length - self.length) / self.length, 1e-3)
        self.assertAlmostEqual(g.step, 1.0, places=2)

    def test_curvature(self):
        # Ellipse curvature ab / (a² sin²t + b² cos²t)^1.5: from b/a² on the flanks to a/b² at the ends
        g, a, b = self.geometry, self.a, self.b
        t = np.arctan2(g.y / b, g.x / a)
        exact = a * b / (a**2 * np.sin(t) ** 2 + b**2 * np.cos(t) ** 2) ** 1.5
        self.assertAlmostEqual(g.kappa.max(), a / b**2, delta=0.05 * a / b**2)
        self.assertLess(np.sqrt(np.mean((g.kappa - exact) ** 2)), 0.08 * np.mean(exact))   # Noise-limited (0.3 m, 6 samples/m)

    def test_elevation(self):
        # z = 5 sin(2πs/L): gradient peaks at 5·2π/L and the vertical curvature is its derivative
        g = self.geometry
        peak_gradient = 5.0 * 2 * np.pi / self.length
        self.assertAlmostEqual(np.tan(g.gradient).max(), peak_gradient, delta=0.1 * peak_gradient)
        # Vertical curvature -5·(2π/L)²·sin(2πs/L), within 3e-5 1/m RMS (0.02 g at 80 m/s)
        # (the fit starts at lap 1's first sample, within one sample spacing of s = 0)
        exact = -5.0 * (2 * np.pi / self.length) ** 2 * np.sin(2 * np.pi * g.s / g.length)
        self.assertLess(np.sqrt(np.mean((g.kappa_v - exact) ** 2)), 3e-5)
        self.assertAlmostEqual(np.sum(np.sin(g.gradient)) * g.step, 0.0, delta=0.05)   # Closes in height (m)

    def test_csv_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "oval.csv"
            self.geometry.to_csv(path)
            loaded = TrackGeometry.from_csv(path)
        self.assertEqual(loaded.source, "oval")
        self.assertAlmostEqual(loaded.length, self.geometry.length, places=3)
        np.testing.assert_allclose(loaded.kappa, self.geometry.kappa, rtol=1e-6, atol=1e-9)

    def test_track_model_keeps_the_turning(self):
        # Averaging over each NLP step keeps the total heading change and the vertical curvature
        track = F1TrackModel(2026, "oval", ds=5.0).load_from_geometry(self.geometry)
        td = track.track_data
        # The whole cells cover [-ds/2, n·ds - ds/2]: the lap minus a short tail before the line
        g, n_full = self.geometry, int(track.total_length // track.ds)
        tail = (g.s >= n_full * track.ds - 0.5 * track.ds) & (g.s < g.length - 0.5 * track.ds)
        missing = np.sum(g.kappa[tail]) * g.step / (2 * np.pi)
        self.assertAlmostEqual(np.sum(td.curvature[:n_full]) * track.ds / (2 * np.pi), 1.0 - missing, delta=3e-3)
        self.assertAlmostEqual(td.vertical_curvature.max(), self.geometry.kappa_v.max(), delta=0.05 * self.geometry.kappa_v.max())


class VerticalCurvatureTests(unittest.TestCase):
    def test_dip_adds_load_and_crest_removes_it(self):
        car = CarModel(get_vehicle_config("2025"), get_ers_config("2025"))
        v = 60.0
        flat = sum(car.axle_loads(v, 0.0))
        dip = sum(car.axle_loads(v, 0.0, kappa_v=0.002))
        crest = sum(car.axle_loads(v, 0.0, kappa_v=-0.002))
        self.assertAlmostEqual(dip - flat, car.mass * v**2 * 0.002, places=6)
        self.assertAlmostEqual(flat - crest, car.mass * v**2 * 0.002, places=6)


SCHEDULE = pd.DataFrame({
    "RoundNumber": [7, 13, 14, 16, 23],
    "Location": ["Barcelona", "Monza", "Madrid", "Kuala Lumpur", "Yas Marina"],
    "Country": ["Spain", "Italy", "Spain", "Bahrain", "United Arab Emirates"],
    "EventName": ["Barcelona Grand Prix", "Italian Grand Prix", "Spanish Grand Prix", "Bahrain Grand Prix", "Abu Dhabi Grand Prix"],
})


class FindRoundTests(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(telemetry.ff1, "get_event_schedule", return_value=SCHEDULE)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_exact_names_and_aliases(self):
        self.assertEqual(telemetry.find_round(2026, "monza"), (13, "Monza"))
        self.assertEqual(telemetry.find_round(2026, 13), (13, "Monza"))
        self.assertEqual(telemetry.find_round(2026, "Catalunya"), (7, "Barcelona"))
        self.assertEqual(telemetry.find_round(2026, "Sepang"), (16, "Kuala Lumpur"))
        self.assertEqual(telemetry.find_round(2026, "YasMarina"), (23, "Yas Marina"))
        self.assertEqual(telemetry.find_round(2026, "Spanish Grand Prix"), (14, "Madrid"))

    def test_ambiguous_or_unknown_names_fail(self):
        with self.assertRaises(ValueError):
            telemetry.find_round(2026, "Spain")
        with self.assertRaises(ValueError):
            telemetry.find_round(2026, "Sakhir")
        with self.assertRaises(ValueError):
            telemetry.find_round(2026, "Monz")


class StraightModeZoneTests(unittest.TestCase):
    CORNERS = {"1": 900.0, "2": 950.0, "3": 1450.0, "4": 2100.0, "11": 5300.0}
    LENGTH = 5760.0

    def test_zones_end_at_the_next_marker_and_wrap(self):
        zones = telemetry.zone_intervals((("11", 30), ("3", 70)), self.CORNERS, self.LENGTH)
        self.assertEqual(zones, [(5330.0, 900.0), (1520.0, 2100.0)])
        with self.assertRaises(ValueError):
            telemetry.zone_intervals((("12", 0),), self.CORNERS, self.LENGTH)

    def test_mask_and_aero_mode(self):
        track = F1TrackModel(2026, "test")
        track.total_length = self.LENGTH
        self.assertIsNone(track.straight_mode_mask([0.0]))
        track.straight_mode_zones = [(5330.0, 900.0), (1520.0, 2100.0)]
        s = np.array([0.0, 500.0, 1000.0, 1600.0, 3000.0, 5500.0, self.LENGTH + 100.0])
        np.testing.assert_array_equal(track.straight_mode_mask(s), [1, 1, 0, 1, 0, 1, 1])
        # Only 2026 cars have a Straight Mode; the mask replaces the radius heuristic
        radius = np.full(len(s), 5000.0)
        car_2026 = CarModel(get_vehicle_config("2026"), get_ers_config("2026"))
        car_2025 = CarModel(get_vehicle_config("2025"), get_ers_config("2025"))
        np.testing.assert_array_equal(car_2026.aero_mode(radius, track.straight_mode_mask(s)), [1, 1, 0, 1, 0, 1, 1])
        np.testing.assert_array_equal(car_2025.aero_mode(radius, track.straight_mode_mask(s)), np.zeros(len(s)))


if __name__ == "__main__":
    unittest.main()
