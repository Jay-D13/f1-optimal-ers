import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from simulation.lap import LapResult, simulate_multiple_laps
from solvers.base import OptimalTrajectory
from visualization.animation import _compute_frame_times, _trail_window_mask
from visualization.results_viz import _compute_cumulative_ers_energy_mj


class VisualizationAndMultiLapTests(unittest.TestCase):
    def test_animation_physical_timing_constant_speed_progression(self):
        times = np.array([0.0, 10.0])
        frame_times = _compute_frame_times(times, timing_mode="physical", fps=5, playback_rate=1.0)

        self.assertTrue(np.allclose(np.diff(frame_times[:-1]), 0.2, atol=1e-12))

        s = np.interp(frame_times, [0.0, 10.0], [0.0, 100.0])  # 10 m/s
        self.assertTrue(np.allclose(np.diff(s[:-1]), 2.0, atol=1e-9))

    def test_animation_trail_window_is_distance_based(self):
        frame_s = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0])
        mask = _trail_window_mask(frame_s, frame_idx=5, trail_length_m=25.0)
        self.assertTrue(np.array_equal(mask, np.array([False, False, False, True, True, True])))

    def test_cumulative_energy_uses_nonuniform_time_steps(self):
        traj = OptimalTrajectory(
            s=np.array([0.0, 5.0, 10.0, 15.0]),
            ds=5.0,
            n_points=4,
            v_opt=np.array([30.0, 30.0, 30.0, 30.0]),
            soc_opt=np.array([0.6, 0.58, 0.57, 0.55]),
            P_ers_opt=np.array([100_000.0, -50_000.0, 100_000.0]),
            throttle_opt=np.array([0.8, 0.7, 0.8]),
            brake_opt=np.array([0.0, 0.1, 0.0]),
            t_opt=np.array([0.0, 0.5, 1.0, 2.0]),
            lap_time=2.0,
            energy_deployed=0.0,
            energy_recovered=0.0,
            solve_time=0.1,
            solver_status="optimal",
            solver_name="test",
        )

        deployed, recovered, net = _compute_cumulative_ers_energy_mj(traj)
        self.assertAlmostEqual(float(deployed[-1]), 0.15, places=9)
        self.assertAlmostEqual(float(recovered[-1]), 0.025, places=9)
        self.assertAlmostEqual(float(net[-1]), 0.125, places=9)

    def test_multilap_soc_carry_over(self):
        call_inputs = []

        class DummySimulator:
            def __init__(self, vehicle_model, track_model, controller, dt=0.1):
                self.track_model = track_model

            def simulate_lap(self, initial_soc=0.5, initial_velocity=30.0, max_time=200.0, reference=None):
                call_inputs.append((initial_soc, initial_velocity))
                final_soc = initial_soc - 0.05
                v_exit = initial_velocity + 2.0
                return LapResult(
                    times=np.array([0.0, 1.0]),
                    positions=np.array([0.0, self.track_model.total_length]),
                    velocities=np.array([initial_velocity, v_exit]),
                    socs=np.array([initial_soc, final_soc]),
                    P_ers_history=np.array([10_000.0]),
                    throttle_history=np.array([0.5]),
                    brake_history=np.array([0.0]),
                    lap_time=1.0,
                    final_soc=final_soc,
                    energy_deployed=10_000.0,
                    energy_recovered=0.0,
                    completed=True,
                )

        with patch("simulation.lap.LapSimulator", DummySimulator):
            controller = SimpleNamespace(reset=lambda: None)
            track_model = SimpleNamespace(total_length=1000.0)

            result = simulate_multiple_laps(
                vehicle_model=object(),
                track_model=track_model,
                controller=controller,
                num_laps=3,
                initial_soc=0.6,
                initial_velocity=30.0,
            )

        self.assertEqual(result.completed_laps, 3)
        self.assertTrue(np.allclose([x[0] for x in call_inputs], [0.6, 0.55, 0.5], atol=1e-12))
        self.assertTrue(np.allclose([x[1] for x in call_inputs], [30.0, 32.0, 34.0], atol=1e-12))
        self.assertTrue(np.array_equal(np.unique(result.lap_index_states), np.array([1, 2, 3])))


if __name__ == "__main__":
    unittest.main()
