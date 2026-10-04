"""
Solver tests on the bundled TUMFTM racelines (no network or FastF1 cache needed).

The track sweep solves every bundled track under both rule sets (~2 min):
    RUN_TRACK_SWEEP=1 python -m unittest tests.test_spatial_nlp
"""
import os
import time
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from config import get_ers_config, get_track_config, get_vehicle_config
from config.events import RampWindow
from models import F1TrackModel, VehicleDynamicsModel, find_tumftm_raceline
from solvers import ForwardBackwardSolver, MultiLapSpatialNLPSolver, SolverError, SpatialNLPSolver
from solvers.spatial_nlp import deploy_power_limit

RACELINES = Path(__file__).resolve().parents[1] / "data" / "racelines"

# Optimal Monza lap (bundled raceline, 2025 rules, trapezoidal, ds = 5 m) for the current physics.
# Update it only when the physics change on purpose.
MONZA_2025_LAP = 81.7838


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

    def test_practice_rules(self):
        practice = get_ers_config("2026", session="practice", event="Monza")
        self.assertEqual(practice.recovery_limit_per_lap, 7.5e6)                                   # Against 5.0 in qualifying
        self.assertEqual(practice.deploy_curve, "overtake")
        self.assertEqual(practice.soc_window, 4.0e6)
        self.assertTrue(practice.qualifying)                                                        # A push lap: start full, run-up

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
        # In each derate run at full throttle: no increase, the first step held, then the ramp rate (C5.12.4-6)
        net, runs = _derate_runs(self.solver, self.trajectory)
        self.assertTrue(runs)
        step_drops, _ = _check_ramp_rules(self, net, self.trajectory.t_opt, runs, self.ers)
        self.assertGreater(max(step_drops), self.ers.ramp_first_step - 15e3)   # Some run steps down fully

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

    def test_no_pedal_overlap(self):
        # Throttle and brake are never pressed together (the overlap cost)
        nodes = self.trajectory.node_controls
        overlap = nodes["throttle"] * (nodes["brake_front"] + nodes["brake_rear"])
        self.assertLess(overlap.max(), 1e-3)

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


def _oval(straight: float = 1500.0, radius: float = 40.0, ds: float = 5.0) -> _Track:
    """Two straights joined by hairpins, flat."""
    radii = []
    for length, r in ((straight, 1e5), (np.pi * radius, radius), (straight, 1e5), (np.pi * radius, radius)):
        radii.extend([r] * int(round(length / ds)))
    track = _Track(radius=1.0, length=len(radii) * ds, ds=ds)
    track.track_data.radius = np.array(radii)
    return track


def _quali_solver(track, windows=(), drag_scale=1.0):
    """The NLP solver for a synthetic track under 2026 qualifying rules (unknown event), with these ramp windows."""
    ers = replace(get_ers_config("2026", session="qualifying"), ramp_windows=tuple(windows))
    vehicle = get_vehicle_config("2026")
    vehicle = replace(vehicle, c_w_a=vehicle.c_w_a * drag_scale)
    solver = SpatialNLPSolver(VehicleDynamicsModel(vehicle, ers), track, ers, ds=5.0)
    solver.verbose = False
    return solver


def _derate_runs(solver, trajectory):
    """Net DC ERS-K power at the nodes (W), and each derate run inside the timed lap as (lap nodes, near the taper)."""
    ers = solver.vehicle.ers
    nodes = trajectory.node_controls
    net = nodes["P_deploy"] / ers.mgu_k_efficiency - nodes["P_harvest"] * ers.mgu_k_efficiency
    r = int(round(trajectory.run_up["distance"] / solver.ds)) if trajectory.run_up else 0
    runs = []
    for run in solver.ramp_runs:
        keep = run["nodes"] >= r
        runs.append((run["nodes"][keep] - r, run["taper"][keep]))
    return net, [run for run in runs if len(run[0]) > 1]


def _check_ramp_rules(test, net, t, runs, ers, tol=15e3):
    """
    Assert the ramp-down rules in each derate run (tolerance tol W for the rounded corners): the net power never
    rises (C5.12.5) away from the deploy curve's taper, and there, while it is above the release level, its reduction
    from the run's start stays within the first step or its drop over the last ramp_hold (less a node) within the
    ramp rate (C5.12.4, C5.12.6). Returns the reductions from the start, and those 1 s drops beyond the first step.
    """
    step_drops, ramp_drops = [], []
    for lap_nodes, cut in runs:
        y, tr = net[lap_nodes], t[lap_nodes]
        for k in range(1, len(y)):
            if not (cut[k] or cut[k - 1]):
                test.assertLessEqual(y[k], y[k - 1] + 2e3, lap_nodes[k])
        for m in range(len(y)):
            if y[m] <= ers.ramp_release + tol or cut[m]:
                continue   # Released, or where the deploy curve may hold the ERS-K below the demand
            # The NLP takes its 1 s windows from the first solve's times, which may differ here by a node
            j = int(np.searchsorted(tr, tr[m] - ers.ramp_hold + 0.1))
            lower = min(y[0] - ers.ramp_first_step, y[j] - ers.ramp_rate * ers.ramp_hold)
            test.assertGreaterEqual(y[m], lower - tol, lap_nodes[m])
            step_drops.append(y[0] - y[m])
            if y[0] - y[m] > ers.ramp_first_step + tol:
                ramp_drops.append(y[j] - y[m])
    return step_drops, ramp_drops


class RampRunTests(unittest.TestCase):
    """Where the ramp-down rules apply (C5.12.4-8): the derate runs found from a solution's speed and throttle."""

    def runs(self, v_kph, throttle, windows=()):
        solver = _quali_solver(_Track(radius=1e5, length=5.0 * (len(v_kph) - 1)), windows)
        solver._horizon = {"v": np.asarray(v_kph, float) / 3.6, "throttle": np.asarray(throttle, float)}
        return solver._full_throttle_runs(run_up=0)

    def test_trigger_hold_and_threshold(self):
        # 252 km/h = 70 m/s: one node every 1/14 s. Full throttle from node 10 to 200.
        v = np.full(301, 252.0)
        v[:30] = 200.0                                     # Below 210 km/h until node 30
        throttle = np.where((np.arange(301) >= 10) & (np.arange(301) <= 200), 1.0, 0.5)
        (run,) = self.runs(v, throttle)
        self.assertEqual(run["nodes"][0], 30)              # Triggered 1 s after node 10 (node 24), then above 210 km/h
        self.assertEqual(run["nodes"][-1], 200)
        # The hold window: each node's reference is the earliest node at most 1 s before it, in the run
        m = 50
        self.assertEqual(run["ref"][m], m - 14)
        self.assertEqual(run["ref"][5], 0)
        self.assertFalse(run["exempt"].any())
        self.assertFalse((run["extra_step"] > 0).any())

    def test_windows(self):
        v = np.full(301, 252.0)
        throttle = np.ones(301)
        windows = (
            RampWindow("reset", 600, 700),                 # Nodes 120-140 split the run
            RampWindow("speed_threshold", 300, 400, 270),  # 252 km/h is below it: free drops at nodes 60-80
            RampWindow("first_step_350", 1000, 1100, 350),
        )
        first, second = self.runs(v, throttle, windows)
        self.assertEqual((first["nodes"][0], first["nodes"][-1]), (14, 119))
        self.assertEqual((second["nodes"][0], second["nodes"][-1]), (141, 300))   # No new trigger after a reset
        exempt = first["nodes"][first["exempt"]]
        self.assertEqual((exempt[0], exempt[-1]), (60, 80))
        extra = second["nodes"][second["extra_step"] > 0]
        self.assertEqual((extra[0], extra[-1]), (200, 220))
        self.assertAlmostEqual(second["extra_step"].max(), 200e3)

    def test_qualifying_only_windows(self):
        quali = get_ers_config("2026", session="qualifying", event="Sepang").ramp_windows
        practice = get_ers_config("2026", session="practice", event="Sepang").ramp_windows
        self.assertEqual(len(quali), 10)
        self.assertEqual(len(practice), 8)                  # Exit T15 first step and reset are SQ and Q only
        self.assertEqual(get_ers_config("2026", session="qualifying", event="Monza").ramp_windows, ())


class RampRuleTests(unittest.TestCase):
    """
    The ramp-down rules on a synthetic oval with two 1.5 km straights (drag doubled, so the car stays below the
    deploy curve's taper): each straight ends in a derate, and each rule binds.
    """

    @staticmethod
    def solve(windows=()):
        solver = _quali_solver(_oval(), windows, drag_scale=2.0)
        trajectory = solver.solve()
        net, runs = _derate_runs(solver, trajectory)
        return solver, trajectory, net, runs

    @classmethod
    def setUpClass(cls):
        cls.solver, cls.trajectory, cls.net, cls.runs = cls.solve()

    @staticmethod
    def largest_drop(net, runs, above=125e3):
        """The largest drop between neighbouring run nodes that starts above `above` (W), and its node."""
        best, where = 0.0, None
        for lap_nodes, _ in runs:
            for a, b in zip(lap_nodes[:-1], lap_nodes[1:]):
                if net[a] > above and net[a] - net[b] > best:
                    best, where = net[a] - net[b], b
        return best, where

    def test_step_hold_ramp_harvest(self):
        t, net, ers = self.trajectory.t_opt, self.net, self.solver.vehicle.ers
        self.assertEqual(self.trajectory.solver_status, "optimal")
        self.assertEqual(len(self.runs), 2)
        tol = 15e3  # The rules' corners are rounded over RAMP_SMOOTHING
        step_drops, ramp_drops = _check_ramp_rules(self, net, t, self.runs, ers, tol)
        # The first step and the ramp rate bind
        self.assertGreater(max(step_drops), ers.ramp_first_step - tol)
        self.assertGreater(max(ramp_drops), 0.8 * ers.ramp_rate * ers.ramp_hold)
        # Below the release level the ERS-K drops straight into super-clip harvest at full throttle
        self.assertLess(net.min(), -300e3)
        (first, _), (second, _) = self.runs
        self.assertLess(net[first].min(), -300e3)
        throttle = self.trajectory.node_controls["throttle"]
        self.assertGreater(throttle[first][np.argmin(net[first])], 0.99)
        # The largest drop that starts above the release level is the first step, not a shortcut of the ramp
        drop, _ = self.largest_drop(net, self.runs)
        self.assertLess(drop, ers.ramp_first_step)

    def test_first_step_window(self):
        # A 350 kW first-step window (C5.12.4) late on the first straight: the derate cuts more than a first step
        # in one go there, and the lap is no slower
        solver, trajectory, net, runs = self.solve((RampWindow("first_step_350", 700, 1300, 350),))
        drop, node = self.largest_drop(net, runs)
        self.assertGreater(drop, 100e3)
        self.assertTrue(700 <= trajectory.s[node] <= 1300)
        self.assertLessEqual(trajectory.lap_time, self.trajectory.lap_time + 1e-3)

    def test_speed_threshold_window(self):
        # A 400 km/h threshold (C5.12.7) over the end of the first straight: there the ERS-K may drop freely
        solver, trajectory, net, runs = self.solve((RampWindow("speed_threshold", 850, 1450, 400),))
        drop, node = self.largest_drop(net, runs)
        self.assertGreater(drop, 100e3)
        self.assertTrue(850 <= trajectory.s[node] <= 1450)
        self.assertLessEqual(trajectory.lap_time, self.trajectory.lap_time + 1e-3)


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
