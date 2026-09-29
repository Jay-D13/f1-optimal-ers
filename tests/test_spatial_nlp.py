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
MONZA_2025_LAP = 81.7815


def make_solver(track_name, regulations, ds=5.0, collocation="trapezoidal", n_laps=1, track_ds=None):
    """The NLP solver for a bundled track, as in main.py. The track grid defaults to ds."""
    ers_config = get_ers_config(regulations)
    vehicle_config = get_vehicle_config(regulations, base=get_track_config(track_name))
    track = F1TrackModel(year=2024, gp=track_name, ds=track_ds or ds)
    track.load_from_tumftm_raceline(str(find_tumftm_raceline(track_name, RACELINES)))
    vehicle_model = VehicleDynamicsModel(vehicle_config, ers_config)

    solver_class = SpatialNLPSolver if n_laps == 1 else MultiLapSpatialNLPSolver
    solver = solver_class(vehicle_model, track, ers_config, ds=ds, collocation_method=collocation)
    solver.verbose = False
    return solver


def solve_tumftm(track_name, regulations, ds=5.0, collocation="trapezoidal", n_laps=1, track_ds=None, **solve_kwargs):
    """Solve a bundled track. The initial guess is the forward-backward profile without ERS."""
    solver = make_solver(track_name, regulations, ds, collocation, n_laps, track_ds)
    if n_laps > 1:
        solve_kwargs["n_laps"] = n_laps
    return solver.solve(**solve_kwargs)


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


class QualifyingRulesTests(unittest.TestCase):
    """2026 qualifying: Overtake curve, per-event caps on the DC side, 4 MJ window, run-up from the last apex."""

    @classmethod
    def setUpClass(cls):
        cls.ers = get_ers_config("2026", session="qualifying", event="Monza")
        vehicle_model = VehicleDynamicsModel(get_vehicle_config("2026", base=get_track_config("Monza")), cls.ers)
        track = F1TrackModel(year=2024, gp="Monza", ds=5.0)
        track.load_from_tumftm_raceline(str(find_tumftm_raceline("Monza", RACELINES)))
        cls.solver = SpatialNLPSolver(vehicle_model, track, cls.ers, ds=5.0)
        cls.solver.verbose = False
        cls.trajectory = cls.solver.solve()

    def test_overtake_curve(self):
        expected_kw = {200: 350, 337.5: 350, 340: 300, 350: 100, 355: 0, 360: 0}
        for v_kph, p_kw in expected_kw.items():
            with self.subTest(v_kph=v_kph):
                self.assertAlmostEqual(float(deploy_power_limit(v_kph / 3.6, self.ers)), p_kw * 1e3, delta=1e-3)

    def test_event_rules(self):
        self.assertEqual(self.ers.recovery_limit_per_lap, 5.0e6)                                   # Monza
        self.assertEqual(get_ers_config("2026", session="qualifying", event=3).superclip_power, 250e3)   # Suzuka, before Miami
        self.assertEqual(get_ers_config("2026", session="qualifying", event="Miami").superclip_power, 350e3)
        self.assertEqual(get_ers_config("2025", session="qualifying").deploy_curve, "flat")        # 2025 unchanged

    def test_lap_uses_the_cap_and_the_window(self):
        trajectory = self.trajectory
        self.assertEqual(trajectory.solver_status, "optimal")
        # The recharge cap binds, counted on the DC side from the timing line
        self.assertAlmostEqual(trajectory.energy_recovered, self.ers.recovery_limit_per_lap, delta=1e3)
        # The run-up starts full; the lap ends at the bottom of the 4 MJ window
        self.assertIsNotNone(trajectory.run_up)
        self.assertAlmostEqual(trajectory.run_up["soc_start"], 1.0, places=9)
        floor = 1.0 - self.ers.soc_window / self.ers.battery_capacity
        self.assertAlmostEqual(trajectory.soc_opt[-1], floor, delta=1e-6)
        self.assertGreaterEqual(trajectory.soc_opt.min(), floor - 1e-6)

    def test_dc_energy_bookkeeping(self):
        # The stored energy changes by the DC flows, through the MGU-K and battery efficiencies
        ers, t = self.ers, self.trajectory
        stored = ers.battery_capacity * (t.soc_opt[0] - t.soc_opt[-1])
        eta_k = ers.mgu_k_efficiency
        moved = (t.energy_deployed * eta_k / ers.deployment_efficiency
                 - t.energy_recovered / eta_k * ers.recovery_efficiency)
        self.assertAlmostEqual(stored, moved, delta=1e3)  # J

    def test_ramp_down(self):
        # Above 210 km/h at full throttle, the deploy never drops by more than the first step between nodes
        # (C5.12.4); the deploy curve's own cuts (C5.2.8) are exempt
        t, nodes = self.trajectory, self.trajectory.node_controls
        deploy = nodes["P_deploy"] / self.ers.mgu_k_efficiency
        limit = np.array([float(deploy_power_limit(v, self.ers)) for v in t.v_opt])
        for k in range(len(deploy) - 1):
            at_speed = min(t.v_opt[k], t.v_opt[k + 1]) > 215 / 3.6
            full = min(nodes["throttle"][k], nodes["throttle"][k + 1]) > 0.999
            if at_speed and full and deploy[k + 1] < limit[k + 1] - 5e3:
                self.assertLessEqual(deploy[k] - deploy[k + 1], self.ers.ramp_first_step + 5e3, k)

    def test_deploy_follows_the_curve(self):
        nodes = self.trajectory.node_controls
        limit = np.array([float(deploy_power_limit(v, self.ers)) for v in self.trajectory.v_opt])
        # Mechanical deploy ≤ MGU-K efficiency × DC limit (plus the NLP's 2 kW corner smoothing)
        self.assertTrue(np.all(nodes["P_deploy"] <= self.ers.mgu_k_efficiency * limit + 2e3))


class SpatialNLPTests(unittest.TestCase):
    """Monza, 2025 rules."""

    @classmethod
    def setUpClass(cls):
        cls.ers = get_ers_config("2025")
        cls.solver = make_solver("Monza", "2025")
        cls.trajectory = cls.solver.solve()

    def test_reference_lap(self):
        self.assertEqual(self.trajectory.solver_status, "optimal")
        self.assertAlmostEqual(self.trajectory.lap_time, MONZA_2025_LAP, delta=0.005)

    def test_grip_limits_hold_at_the_nodes(self):
        for usage in (self.trajectory.grip_usage_front, self.trajectory.grip_usage_rear):
            self.assertLessEqual(usage.max(), 1.0 + 1e-6)
        # Grip is what limits the corners: some node is at the limit on each axle
        self.assertGreater(self.trajectory.grip_usage_front.max(), 0.999)
        self.assertGreater(self.trajectory.grip_usage_rear.max(), 0.999)

    def test_pedal_rates(self):
        # A full throttle or brake travel takes at least the rise time, and the limits bind at the braking points
        t, nodes, car = self.trajectory, self.trajectory.node_controls, self.solver.vehicle.vehicle
        dt = self.solver.ds / (0.5 * (t.v_opt[1:] + t.v_opt[:-1]))
        pedals = ((nodes["throttle"], car.throttle_rise_time), (nodes["brake_front"] + nodes["brake_rear"], car.brake_rise_time))
        for pedal, rise in pedals:
            rate = np.abs(np.diff(pedal)) / dt
            self.assertLessEqual(rate.max(), (1.0 + 1e-6) / rise)
            self.assertGreater(rate.max(), 0.99 / rise)

    def test_reintegrated_controls_reproduce_the_lap(self):
        # Replay the optimal controls through the car model on a 10x finer grid
        replay = self.solver.reintegrate(self.trajectory, substeps=10)
        self.assertLess(abs(replay["lap_time"] - self.trajectory.lap_time), 0.05)
        self.assertLess(max(replay["max_usage_front"], replay["max_usage_rear"]), 1.05)
        v_replay = np.interp(self.trajectory.s, replay["s"], replay["v"])
        self.assertLess(np.max(np.abs(v_replay - self.trajectory.v_opt)), 1.0)   # m/s

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
        # Halve the NLP step on a fixed track grid. The lap time converges at first order, because braking
        # and deployment switch abruptly and a switch can only move by whole nodes (BUGS.md SOL-13):
        # 5 -> 2.5 m adds ~0.11 %, 2.5 -> 1.25 m ~0.05 %.
        coarse = solve_tumftm("Monza", "2025", ds=5.0, track_ds=1.25)
        fine = solve_tumftm("Monza", "2025", ds=2.5, track_ds=1.25)
        self.assertLess(abs(fine.lap_time - coarse.lap_time) / coarse.lap_time, 1.5e-3)

    def test_failed_solve_raises(self):
        with self.assertRaises(SolverError):
            solve_tumftm("Monza", "2025", final_soc_min=0.95)  # Above max_soc: infeasible


class _Track:
    """Minimal track: constant radius, flat."""

    class _Data:
        pass

    def __init__(self, radius: float, length: float, ds: float = 5.0):
        n = int(length / ds)
        data = self._Data()
        data.s = np.arange(n) * ds
        data.radius = np.full(n, radius)
        data.gradient = np.zeros(n)
        data.ds = ds
        data.total_length = n * ds
        self.track_data = data
        self.total_length = data.total_length


class ConstantRadiusTests(unittest.TestCase):
    """On a circle, the optimal lap holds the steady cornering speed of the car model."""

    def test_circle(self):
        ers_config = get_ers_config("2025")
        vehicle_model = VehicleDynamicsModel(get_vehicle_config("2025"), ers_config)
        track = _Track(radius=100.0, length=2 * np.pi * 100.0)
        solver = SpatialNLPSolver(vehicle_model, track, ers_config, ds=5.0)
        solver.verbose = False
        trajectory = solver.solve()

        v_corner = ForwardBackwardSolver(vehicle_model, track)._cornering_speeds(
            np.array([0.01]), np.zeros(1), np.zeros(1), hold=True
        )[0]
        np.testing.assert_allclose(trajectory.v_opt, v_corner, rtol=1e-4)
        self.assertAlmostEqual(trajectory.lap_time, track.total_length / v_corner, delta=1e-3)


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
