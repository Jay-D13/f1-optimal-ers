import unittest

from config import TireThermalConfig, get_tire_compound_config
from models.tire_thermals import (
    core_temp_rate_np,
    heat_generation_np,
    mu_temp_scale_np,
    surface_temp_rate_np,
    wear_rate_np,
)


class TireThermalModelTests(unittest.TestCase):
    def setUp(self):
        self.thermal = TireThermalConfig()
        self.compound = get_tire_compound_config("medium")

    def test_higher_utilization_increases_surface_heating(self):
        q_low = heat_generation_np(6000.0, 60.0, 0.25, self.thermal, self.compound)
        q_high = heat_generation_np(6000.0, 60.0, 0.85, self.thermal, self.compound)
        dts_low = surface_temp_rate_np(q_low, 85.0, 80.0, 25.0, 35.0, self.thermal)
        dts_high = surface_temp_rate_np(q_high, 85.0, 80.0, 25.0, 35.0, self.thermal)
        self.assertGreater(dts_high, dts_low)

    def test_low_utilization_can_cool_hot_tire(self):
        q = heat_generation_np(5500.0, 45.0, 0.05, self.thermal, self.compound)
        dts = surface_temp_rate_np(q, 130.0, 115.0, 22.0, 30.0, self.thermal)
        self.assertLess(dts, 0.0)

    def test_wear_rate_is_non_negative_and_monotonic_when_integrated(self):
        wear = 0.0
        tc = 96.0
        for _ in range(100):
            dw = wear_rate_np(0.60, tc, 6200.0, self.thermal, self.compound)
            self.assertGreaterEqual(dw, 0.0)
            wear_next = min(1.0, wear + dw * 0.1)
            self.assertGreaterEqual(wear_next, wear)
            wear = wear_next
            q_sc = self.thermal.k_surface_core * (100.0 - tc)
            tc += core_temp_rate_np(q_sc, tc, 25.0, self.thermal) * 0.1

    def test_temperature_multiplier_peaks_near_optimum(self):
        t_opt = self.compound.temp_opt_core_c
        scale_opt = mu_temp_scale_np(t_opt, self.thermal, self.compound)
        scale_cold = mu_temp_scale_np(t_opt - 20.0, self.thermal, self.compound)
        scale_hot = mu_temp_scale_np(t_opt + 20.0, self.thermal, self.compound)
        self.assertGreater(scale_opt, scale_cold)
        self.assertGreater(scale_opt, scale_hot)


if __name__ == "__main__":
    unittest.main()
