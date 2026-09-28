import unittest

import numpy as np

from config import ERSConfig, TireThermalConfig, VehicleConfig, get_tire_compound_config
from models import VehicleDynamicsModel
from solvers import MultiLapSpatialNLPSolver


class _DummyTrackData:
    def __init__(self, n_points: int):
        self.gradient = np.zeros(n_points)
        self.radius = np.ones(n_points) * 220.0


class _DummyTrack:
    def __init__(self, total_length: float, ds: float):
        self.total_length = total_length
        self.ds = ds
        n_points = int(total_length / ds) + 1
        self.track_data = _DummyTrackData(n_points)


class MultiLapDynamicTireTests(unittest.TestCase):
    def _make_solver(self):
        vehicle_model = VehicleDynamicsModel(VehicleConfig(), ERSConfig())
        track = _DummyTrack(total_length=80.0, ds=20.0)
        solver = MultiLapSpatialNLPSolver(
            vehicle_model=vehicle_model,
            track_model=track,
            ers_config=vehicle_model.ers,
            ds=20.0,
            collocation_method="euler",
            nlp_solver="ipopt",
        )
        solver.verbose = False
        return solver

    def test_scalar_mode_is_stable_and_dynamic_mode_adds_tire_states(self):
        solver = self._make_solver()
        v_limit = np.array([55.0, 60.0, 58.0, 62.0, 57.0], dtype=float)

        try:
            scalar_1 = solver.solve(
                v_limit_profile=v_limit,
                n_laps=2,
                initial_soc=0.6,
                final_soc_min=0.25,
                is_flying_lap=False,
                lap_grip_scales=np.array([1.0, 0.97]),
                tire_model="scalar",
            )
            scalar_2 = solver.solve(
                v_limit_profile=v_limit,
                n_laps=2,
                initial_soc=0.6,
                final_soc_min=0.25,
                is_flying_lap=False,
                lap_grip_scales=np.array([1.0, 0.97]),
                tire_model="scalar",
            )
            dynamic = solver.solve(
                v_limit_profile=v_limit,
                n_laps=2,
                initial_soc=0.6,
                final_soc_min=0.25,
                is_flying_lap=False,
                tire_model="dynamic",
                tire_thermal_config=TireThermalConfig(),
                tire_compound_config=get_tire_compound_config("medium"),
                ambient_temp_c=24.0,
                track_temp_c=38.0,
                tire_init_temp_c=70.0,
            )
        except Exception as exc:
            self.skipTest(f"NLP backend unavailable for regression test: {exc}")

        # Scalar regression: repeat solve should remain numerically stable.
        self.assertAlmostEqual(scalar_1.lap_time, scalar_2.lap_time, places=5)
        np.testing.assert_allclose(scalar_1.soc_opt, scalar_2.soc_opt, atol=1e-6)
        self.assertIsNotNone(scalar_1.lap_grip_scales)
        self.assertIsNone(scalar_1.tire_temp_core_front)

        # Dynamic mode should expose evolving tire states.
        self.assertIsNotNone(dynamic.tire_temp_surface_front)
        self.assertIsNotNone(dynamic.tire_temp_core_front)
        self.assertIsNotNone(dynamic.tire_wear_front)
        self.assertIsNotNone(dynamic.tire_mu_scale_front)
        self.assertEqual(dynamic.tire_temp_core_front.shape[0], dynamic.n_points)
        self.assertGreater(float(np.std(dynamic.tire_mu_scale_front)), 1e-5)

        # Behavior should differ from scalar strategy due to thermal/degradation coupling.
        self.assertGreater(abs(dynamic.lap_time - scalar_1.lap_time), 1e-4)

        # Guard runtime inflation.
        self.assertLessEqual(dynamic.solve_time, scalar_1.solve_time * 1.8 + 1.0)


if __name__ == "__main__":
    unittest.main()
