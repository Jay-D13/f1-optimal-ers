"""
Solver tests on the bundled TUMFTM racelines (no network or FastF1 cache needed).

The track sweep solves every bundled track under both rule sets (~2 min):
    RUN_TRACK_SWEEP=1 python -m unittest tests.test_spatial_nlp
"""
import os
import time
import unittest
from pathlib import Path

import numpy as np

from config import get_ers_config, get_track_config, get_vehicle_config
from models import F1TrackModel, VehicleDynamicsModel, find_tumftm_raceline
from solvers import ForwardBackwardSolver, MultiLapSpatialNLPSolver, SolverError, SpatialNLPSolver
from solvers.spatial_nlp import deploy_power_limit

RACELINES = Path(__file__).resolve().parents[1] / "data" / "racelines"

# Optimal Monza lap (bundled raceline, 2025 rules, trapezoidal, ds = 5 m) for the current physics.
# Update it only when the physics change on purpose.
MONZA_2025_LAP = 79.7745


def solve_tumftm(track_name, regulations, ds=5.0, collocation="trapezoidal", n_laps=1, track_ds=None, **solve_kwargs):
    """Same pipeline as main.py: forward-backward speed envelope, then the NLP. The track grid defaults to ds."""
    ers_config = get_ers_config(regulations)
    vehicle_config = get_vehicle_config(regulations, base=get_track_config(track_name))
    track = F1TrackModel(year=2024, gp=track_name, ds=track_ds or ds)
    track.load_from_tumftm_raceline(str(find_tumftm_raceline(track_name, RACELINES)))
    vehicle_model = VehicleDynamicsModel(vehicle_config, ers_config)
    v_limit = ForwardBackwardSolver(vehicle_model, track, use_ers_power=True).solve(flying_lap=True).v

    solver_class = SpatialNLPSolver if n_laps == 1 else MultiLapSpatialNLPSolver
    solver = solver_class(vehicle_model, track, ers_config, ds=ds, collocation_method=collocation)
    solver.verbose = False
    if n_laps > 1:
        solve_kwargs["n_laps"] = n_laps
    return solver.solve(v_limit, **solve_kwargs)


class DeployPowerLimitTests(unittest.TestCase):
    def test_2025_is_flat(self):
        ers = get_ers_config("2025")
        for v_kph in (100.0, 300.0, 350.0):
            self.assertEqual(deploy_power_limit(v_kph / 3.6, ers), 120_000)

    def test_2026_taper(self):
        ers = get_ers_config("2026")
        expected_kw = {200: 350, 290: 350, 300: 300, 330: 150, 340: 100, 342: 60, 345: 0, 360: 0}
        for v_kph, p_kw in expected_kw.items():
            with self.subTest(v_kph=v_kph):
                self.assertAlmostEqual(float(deploy_power_limit(v_kph / 3.6, ers)), p_kw * 1e3, delta=1e-3)


class SpatialNLPTests(unittest.TestCase):
    """Monza, 2025 rules."""

    @classmethod
    def setUpClass(cls):
        cls.ers = get_ers_config("2025")
        cls.trajectory = solve_tumftm("Monza", "2025")

    def test_reference_lap(self):
        self.assertEqual(self.trajectory.solver_status, "optimal")
        self.assertAlmostEqual(self.trajectory.lap_time, MONZA_2025_LAP, delta=0.005)

    def test_energy_limits(self):
        self.assertLessEqual(self.trajectory.energy_recovered, self.ers.recovery_limit_per_lap + 1.0)
        self.assertLessEqual(self.trajectory.energy_deployed, self.ers.deployment_limit_per_lap + 1.0)
        self.assertGreaterEqual(self.trajectory.soc_opt[-1], 0.3 - 1e-6)

    def test_energy_bookkeeping(self):
        # The SOC change equals the energy that went through the battery
        stored = self.ers.battery_capacity * (self.trajectory.soc_opt[0] - self.trajectory.soc_opt[-1])
        moved = (
            self.trajectory.energy_deployed / self.ers.deployment_efficiency
            - self.trajectory.energy_recovered * self.ers.recovery_efficiency
        )
        self.assertAlmostEqual(stored, moved, delta=1e3)  # J

    def test_grid_refinement(self):
        # Halve the NLP step on a fixed track grid. The track grid also moves the speed envelope
        # that bounds the NLP (up to 0.11 % at Spa), which is a separate effect.
        coarse = solve_tumftm("Monza", "2025", ds=5.0, track_ds=1.25)
        fine = solve_tumftm("Monza", "2025", ds=2.5, track_ds=1.25)
        self.assertLess(abs(fine.lap_time - coarse.lap_time) / coarse.lap_time, 1e-3)

    def test_failed_solve_raises(self):
        with self.assertRaises(SolverError):
            solve_tumftm("Monza", "2025", final_soc_min=0.95)  # Above max_soc: infeasible


class MultiLapTests(unittest.TestCase):
    def test_two_laps_with_tyre_wear(self):
        ers = get_ers_config("2025")
        trajectory = solve_tumftm(
            "Monza", "2025", n_laps=2, lap_grip_scales=[1.0, 0.95], per_lap_final_soc_min=0.3
        )
        self.assertEqual(trajectory.solver_status, "optimal")
        self.assertEqual(len(trajectory.lap_times), 2)
        self.assertGreater(trajectory.lap_times[1], trajectory.lap_times[0])  # Less grip on lap 2
        self.assertAlmostEqual(trajectory.lap_times.sum(), trajectory.lap_time, delta=1e-9)
        np.testing.assert_array_less(trajectory.lap_energy_recovered, ers.recovery_limit_per_lap + 1.0)
        np.testing.assert_array_less(0.3 - 1e-6, trajectory.lap_end_soc)


@unittest.skipUnless(os.environ.get("RUN_TRACK_SWEEP"), "slow (~2 min): set RUN_TRACK_SWEEP=1")
class TrackSweepTests(unittest.TestCase):
    def test_all_bundled_tracks(self):
        for path in sorted(RACELINES.glob("*.csv")):
            for regulations in ("2025", "2026"):
                with self.subTest(track=path.stem, regulations=regulations):
                    start = time.time()
                    trajectory = solve_tumftm(path.stem, regulations)
                    self.assertEqual(trajectory.solver_status, "optimal")
                    self.assertLess(time.time() - start, 30.0)


if __name__ == "__main__":
    unittest.main()
