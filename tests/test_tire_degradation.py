import unittest

import numpy as np

from utils.tire_degradation import build_lap_grip_scales


class TireDegradationTests(unittest.TestCase):
    def test_no_pit_monotonic_and_floor(self):
        scales = build_lap_grip_scales(
            n_laps=6,
            wear_rate=0.05,
            min_scale=0.88,
            pit_lap_end=None,
        )

        np.testing.assert_allclose(scales, np.array([1.0, 0.95, 0.90, 0.88, 0.88, 0.88]), atol=1e-12)
        self.assertTrue(np.all(np.diff(scales) <= 1e-12))
        self.assertGreaterEqual(float(scales.min()), 0.88)

    def test_pit_reset_applies_to_next_lap(self):
        scales = build_lap_grip_scales(
            n_laps=6,
            wear_rate=0.03,
            min_scale=0.85,
            pit_lap_end=3,
        )

        np.testing.assert_allclose(scales, np.array([1.0, 0.97, 0.94, 1.0, 0.97, 0.94]), atol=1e-12)
        self.assertAlmostEqual(float(scales[2]), 0.94, places=12)
        self.assertAlmostEqual(float(scales[3]), 1.0, places=12)


if __name__ == "__main__":
    unittest.main()
