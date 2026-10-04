"""Time-structure residuals and clip detectors (calibration/structure.py) on a synthetic lap with known durations."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from calibration.dataset import ReferenceLap
from calibration.fit import STRUCTURE_SIGMA, structure_residuals
from calibration.structure import (DURATIONS, SPARC_DRAG, SPARC_ROLL, disagreements, reference_gaps, reference_samples,
                                   reference_structure, runs, trajectory_samples, trajectory_structure)
from models import F1TrackModel
from models.geometry import TrackGeometry
from solvers.base import OptimalTrajectory

FINE = 0.05                  # (m) Step of the exact synthetic lap
NODE = 5.0                   # (m) Model step, as in the calibration solves


def circle_track(radius=500.0, ds=5.0):
    """A flat 2π·500 m circle (as in test_calibration)."""
    n = 3142
    s = np.arange(n) * (2 * np.pi * radius / n)
    angle = s / radius
    geometry = TrackGeometry(s, radius * np.cos(angle), radius * np.sin(angle), np.zeros(n), np.full(n, 1 / radius),
                             np.zeros(n), np.zeros(n), 2 * np.pi * radius, "circle")
    return F1TrackModel(2026, "circle", ds=ds).load_from_geometry(geometry)


def _power_run(v0, length, power_per_kg):
    """Speed along a full-throttle run at a constant wheel power (W/kg) with the SPARC drag shape."""
    n = int(round(length / FINE))
    v = np.empty(n)
    v[0] = v0
    for i in range(1, n):
        a = power_per_kg / v[i - 1] - SPARC_DRAG * v[i - 1] ** 2 - SPARC_ROLL
        v[i] = v[i - 1] + FINE * a / v[i - 1]
    return v


def synthetic_lap(length):
    """
    One lap on the fine grid: speed, throttle (fraction), model brake (fraction of the maximum), and the state
    each point is in. Segment ends sit on the 5 m model grid.
    """
    segments = []                                            # (name, speeds, throttle, brake)
    v1 = _power_run(40.0, 700.0, 900.0)                      # Full power: ICE plus the whole MGU-K deploy
    segments.append(("accel", v1, 1.0, 0.0))
    v2 = v1[-1] - 0.10 / 3.6 * FINE * np.arange(1, int(300 / FINE) + 1)   # Clipping: -0.10 (km/h)/m at full throttle
    segments.append(("clip", v2, 1.0, 0.0))
    v3 = np.linspace(v2[-1], 52.0, int(150 / FINE) + 1)[1:]  # Braking
    segments.append(("brake", v3, 0.0, 0.6))
    segments.append(("part", np.full(int(250 / FINE), 52.0), 0.4, 0.0))
    v5 = _power_run(52.0, 500.0, 400.0)                      # A derate that still accelerates: 500 W/kg short of the envelope
    segments.append(("derate", v5, 1.0, 0.0))
    end6 = NODE * np.floor((length - 200.0) / NODE)          # The last brake zone starts on the model grid
    segments.append(("part", np.full(int(round((end6 - 1900.0) / FINE)), v5[-1]), 0.5, 0.0))
    n7 = int(round(length / FINE)) - sum(len(x[1]) for x in segments)
    segments.append(("brake", np.linspace(v5[-1], 40.0, n7 + 1)[:-1], 0.0, 0.6))
    v = np.concatenate([x[1] for x in segments])
    state = np.concatenate([[x[0]] * len(x[1]) for x in segments])
    throttle = np.concatenate([np.full(len(x[1]), x[2]) for x in segments])
    brake = np.concatenate([np.full(len(x[1]), x[3]) for x in segments])
    s = FINE * np.arange(len(v))
    return s, v, throttle, brake, state


def known_durations(s, v, state):
    """Exact seconds in each state (the integral of ds/v on the fine grid)."""
    dt = FINE / v
    seconds = {k: float(dt[state == k].sum()) for k in ("accel", "clip", "brake", "part", "derate")}
    return {
        "full": seconds["accel"] + seconds["clip"] + seconds["derate"], "part": seconds["part"],
        "brake": seconds["brake"], "clip": seconds["clip"], "sparc": seconds["clip"] + seconds["derate"],
        "lap": float(dt.sum()),
    }


def model_trajectory(s, v, throttle, brake, length):
    """An OptimalTrajectory on the 5 m grid as the NLP returns it: node speeds, interval controls, t from mean speeds."""
    nodes = np.arange(0.0, length + 1e-9, NODE)
    nodes[-1] = length
    v_nodes = np.interp(nodes, np.append(s, length), np.append(v, v[0]))
    mid = np.minimum(((nodes[:-1] + nodes[1:]) / 2 / FINE).astype(int), len(s) - 1)
    t = np.concatenate([[0.0], np.cumsum(np.diff(nodes) / (0.5 * (v_nodes[1:] + v_nodes[:-1])))])
    n = len(nodes)
    return OptimalTrajectory(
        s=nodes, ds=NODE, n_points=n, v_opt=v_nodes, soc_opt=np.full(n, 0.5), P_ers_opt=np.zeros(n - 1),
        throttle_opt=throttle[mid], brake_opt=brake[mid], t_opt=t, lap_time=float(t[-1]), energy_deployed=0.0,
        energy_recovered=0.0, solve_time=0.0, solver_status="optimal", solver_name="synthetic",
    )


def reference_lap_of(s, v, throttle, brake, track, hz=4.0):
    """The same lap as car data: samples every 1/hz s, throttle in %, the brake as a 0/1 flag."""
    t = np.concatenate([[0.0], np.cumsum(FINE / v)])[:-1]
    when = np.arange(0.0, t[-1], 1.0 / hz)
    k = np.searchsorted(t, when, side="right") - 1
    return ReferenceLap(
        round=0, name="synthetic", driver="SYN", lap_time=float(np.sum(FINE / v)), pole_time=float(np.sum(FINE / v)),
        s=s[k], speed=v[k], throttle=100.0 * throttle[k], brake=(brake[k] > 0.5).astype(float), air_density=None,
        track=track,
    )


class StructureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.track = circle_track()
        cls.length = cls.track.total_length
        s, v, throttle, brake, state = synthetic_lap(cls.length)
        cls.known = known_durations(s, v, state)
        cls.trajectory = model_trajectory(s, v, throttle, brake, cls.length)
        cls.reference = reference_lap_of(s, v, throttle, brake, cls.track)

    def assertDurations(self, measured, tolerance):
        for key in DURATIONS:
            with self.subTest(metric=key):
                self.assertAlmostEqual(getattr(measured, key), self.known[key], delta=tolerance)
        self.assertAlmostEqual(measured.clip_sparc, self.known["sparc"], delta=tolerance)

    def test_model_on_its_own_grid_finds_the_known_durations(self):
        measured = trajectory_structure(self.trajectory, track=self.track)
        self.assertAlmostEqual(measured.total, self.trajectory.lap_time, places=9)   # Weights are the model's dt
        self.assertAlmostEqual(measured.total, self.known["lap"], delta=0.01)
        self.assertDurations(measured, tolerance=0.15)       # About two 5 m intervals per transition

    def test_reference_at_4_hz_finds_the_known_durations(self):
        measured = reference_structure(self.reference)
        self.assertAlmostEqual(measured.total, self.known["lap"], delta=0.05)
        self.assertDurations(measured, tolerance=0.5)        # About two 0.25 s samples per transition

    def test_model_and_reference_definitions_agree_on_the_same_samples(self):
        """Sampled at the reference's distances, the model gives the reference's numbers: the definitions are one."""
        model = trajectory_structure(self.trajectory, at=self.reference.s, track=self.track)
        real = reference_structure(self.reference)
        for key in DURATIONS + ("clip_sparc", "clip_both", "total"):
            with self.subTest(metric=key):
                self.assertAlmostEqual(getattr(model, key), getattr(real, key), delta=0.02)
        np.testing.assert_array_equal(trajectory_samples(self.trajectory, at=self.reference.s).braking,
                                      reference_samples(self.reference).braking)

    def test_model_brake_counts_above_two_percent_of_the_maximum(self):
        light = model_trajectory(*synthetic_lap(self.length)[:4], self.length)
        light.brake_opt = np.where(light.brake_opt > 0, 0.015, 0.0)    # Below 2 %: not braking
        measured = trajectory_structure(light, sparc=False)
        self.assertEqual(measured.brake, 0.0)
        self.assertAlmostEqual(measured.part, self.known["part"] + self.known["brake"], delta=0.15)

    def test_the_detectors_disagree_on_a_derate_that_still_accelerates(self):
        model = trajectory_structure(self.trajectory, track=self.track)
        self.assertAlmostEqual(model.clip_both, self.known["clip"], delta=0.15)
        stretches = disagreements(trajectory_samples(self.trajectory, track=self.track))
        self.assertEqual(stretches[0]["which"], "sparc")              # The derate, from 1400 to 1900 m
        self.assertAlmostEqual(stretches[0]["start_m"], 1400.0, delta=15.0)
        self.assertAlmostEqual(stretches[0]["end_m"], 1900.0, delta=15.0)
        self.assertLess(sum(x["seconds"] for x in stretches if x["which"] == "dv"), 0.15)

    def test_structure_residuals_are_model_minus_reference_over_sigma(self):
        r = structure_residuals(self.trajectory, self.reference)
        model = trajectory_structure(self.trajectory, sparc=False).durations()
        real = reference_structure(self.reference, sparc=False).durations()
        np.testing.assert_allclose(r, (model - real) / STRUCTURE_SIGMA)
        self.assertLess(np.abs(r).max(), 0.5 / STRUCTURE_SIGMA)       # Same lap: within the sampling tolerance

    def test_a_gap_in_the_reference_is_left_out_of_both_laps(self):
        """A dropout over the first braking zone: the sample before it would count the zone as full throttle."""
        ref = self.reference
        keep = (ref.s < 990.0) | (ref.s >= 1160.0)
        holed = ReferenceLap(round=0, name="holed", driver="SYN", lap_time=ref.lap_time, pole_time=ref.pole_time,
                             s=ref.s[keep], speed=ref.speed[keep], throttle=ref.throttle[keep], brake=ref.brake[keep],
                             air_density=None, track=self.track)
        stretches = reference_gaps(holed)
        self.assertEqual(len(stretches), 1)
        self.assertLess(stretches[0][0], 990.0)
        self.assertGreaterEqual(stretches[0][1], 1160.0)
        whole = reference_structure(holed, sparc=False)
        self.assertLess(whole.brake, self.known["brake"] - 1.0)           # The zone's braking is lost
        model = trajectory_structure(self.trajectory, sparc=False, exclude=stretches)
        real = reference_structure(holed, sparc=False, exclude=stretches)
        self.assertLess(model.total, self.known["lap"] - 2.0)
        for key in DURATIONS:
            with self.subTest(metric=key):
                self.assertAlmostEqual(getattr(model, key), getattr(real, key), delta=0.5)
        self.assertLess(np.abs(structure_residuals(self.trajectory, holed)).max(), 0.5 / STRUCTURE_SIGMA)

    def test_runs_keep_only_long_runs_and_wrap_through_the_line(self):
        flag = np.array([1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1], dtype=bool)
        kept = runs(flag, 5)
        np.testing.assert_array_equal(np.flatnonzero(kept), [0, 1, 3, 4, 5, 6, 7, 14, 15, 16])


if __name__ == "__main__":
    unittest.main()
