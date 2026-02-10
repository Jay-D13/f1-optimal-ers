import unittest

import numpy as np

from utils.tire_degradation import build_lap_grip_scales


class TireDegradationTests(unittest.TestCase):
    def test_monotonic_and_floor(self):
        scales = build_lap_grip_scales(
            n_laps=6,
            wear_rate=0.05,
            min_scale=0.88,
        )

        np.testing.assert_allclose(scales, np.array([1.0, 0.95, 0.90, 0.88, 0.88, 0.88]), atol=1e-12)
        self.assertTrue(np.all(np.diff(scales) <= 1e-12))
        self.assertGreaterEqual(float(scales.min()), 0.88)


if __name__ == "__main__":
    unittest.main()
