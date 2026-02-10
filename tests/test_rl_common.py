import unittest

import numpy as np

from config import ERSConfig
from rl.common import (
    action_to_ers_power,
    build_observation,
    compute_speed_limited_deploy_power,
)


class RLCommonTests(unittest.TestCase):
    def test_2026_speed_taper_applies(self):
        ers_2026 = ERSConfig(
            regulation_year=2026,
            max_deployment_power=350_000.0,
        )
        p_limit = compute_speed_limited_deploy_power(speed_mps=300.0 / 3.6, ers=ers_2026)
        self.assertAlmostEqual(p_limit, 300_000.0, places=3)

    def test_action_projection_respects_soc_and_budget(self):
        ers = ERSConfig(
            regulation_year=2025,
            max_deployment_power=120_000.0,
            max_recovery_power=120_000.0,
            deployment_limit_per_lap=1000.0,
            recovery_limit_per_lap=1000.0,
            min_soc=0.2,
            max_soc=0.9,
        )

        # No deployment at min SOC
        p_deploy = action_to_ers_power(
            action_value=1.0,
            speed_mps=60.0,
            soc=0.2,
            ers=ers,
            dt=0.1,
            deployed_energy_j=0.0,
            recovered_energy_j=0.0,
        )
        self.assertEqual(p_deploy, 0.0)

        # No recovery at max SOC
        p_recover = action_to_ers_power(
            action_value=-1.0,
            speed_mps=60.0,
            soc=0.9,
            ers=ers,
            dt=0.1,
            deployed_energy_j=0.0,
            recovered_energy_j=0.0,
        )
        self.assertEqual(p_recover, 0.0)

        # Budget-limited deployment: remaining energy 100J over dt=0.1 -> 1000W cap
        p_budget = action_to_ers_power(
            action_value=1.0,
            speed_mps=60.0,
            soc=0.5,
            ers=ers,
            dt=0.1,
            deployed_energy_j=900.0,
            recovered_energy_j=0.0,
        )
        self.assertAlmostEqual(p_budget, 1000.0, places=6)

    def test_observation_vector_shape(self):
        obs = build_observation(
            state=np.array([250.0, 75.0, 0.45]),
            track_info={"gradient": 0.05, "radius": 120.0, "curvature": 1.0 / 120.0},
            total_length_m=5000.0,
            reference_speed_mps=80.0,
            deploy_remaining_j=2.0e6,
            recover_remaining_j=1.0e6,
            deployment_limit_j=4.0e6,
            recovery_limit_j=2.0e6,
            throttle_base=0.6,
            brake_base=0.0,
        )
        self.assertEqual(obs.shape, (10,))
        self.assertEqual(obs.dtype, np.float32)
        self.assertGreaterEqual(float(obs[0]), 0.0)
        self.assertLessEqual(float(obs[0]), 1.0)


if __name__ == "__main__":
    unittest.main()
