import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from calibration.straights import (Car, Prepared, TimedLap, acceleration, fit, fit_ceiling, full_throttle_runs,
                                   overtake_curve, segment, smooth)


def synthetic_lap(car: Car, superclip=350e3, seed=0) -> TimedLap:
    """
    Two straights driven by the model's own equations, sampled like FastF1 car data (about 4 Hz, 1 km/h steps):
    a corner at part throttle, full throttle with the full deploy, full harvest at full throttle (super-clip),
    then braking. Straight Mode is open from 150 m into each straight.
    """
    rng = np.random.default_rng(seed)
    phases = [("corner", 2.0, 120), ("deploy", 7.0, None), ("clip", 3.0, None), ("brake", 2.0, None),
              ("corner", 2.0, 140), ("deploy", 9.0, None), ("clip", 2.5, None), ("brake", 2.0, None)]
    dt = 0.01
    t, s, v, phase_of, w_of = [0.0], [0.0], [120 / 3.6], ["corner"], [0.0]
    straight_start = 0.0
    stub = TimedLap(0, "test", "test", np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1),
                    np.zeros(1), np.zeros(1), np.zeros(1), 1e9, 1.2, superclip, "qualifying")
    for name, seconds, corner_kph in phases:
        if name == "deploy":
            straight_start = s[-1]
        for _ in range(int(round(seconds / dt))):
            w = 1.0 if name in ("deploy", "clip") and s[-1] - straight_start > 150.0 else 0.0
            stub.w[0] = w
            if name == "corner":
                a = (corner_kph / 3.6 - v[-1]) / 0.5
            elif name == "brake":
                a = -30.0
            else:
                a = float(acceleration(stub, np.array([0]), np.array([v[-1]]), car, name)[0])
            v.append(max(v[-1] + a * dt, 20.0))
            s.append(s[-1] + v[-1] * dt)
            t.append(t[-1] + dt)
            phase_of.append(name)
            w_of.append(w)
    t, s, v = np.array(t), np.array(s), np.array(v)
    # Car-data sampling: irregular 0.22–0.28 s steps, speed in whole km/h
    times = np.cumsum(rng.uniform(0.22, 0.28, size=int(t[-1] / 0.22)))
    times = times[times < t[-1]]
    k = np.searchsorted(t, times)
    phase = np.array(phase_of)[k]
    zone = np.array(w_of)[k]
    throttle = np.where(np.isin(phase, ("deploy", "clip")), 100.0, np.where(phase == "corner", 30.0, 0.0))
    return TimedLap(
        round=0, kind="test", label="synthetic", t=times - times[0], s=s[k], v=np.round(v[k] * 3.6) / 3.6,
        throttle=throttle, brake=(phase == "brake").astype(float), gap=np.full(len(k), np.nan), w=zone,
        gradient=np.zeros(len(k)), kappa_v=np.zeros(len(k)), length=1e9, rho=1.2, superclip=superclip,
        rules="qualifying",
    )


class StraightsTests(unittest.TestCase):
    def test_overtake_curve(self):
        kph = np.array([200.0, 337.5, 345.0, 355.0, 360.0])
        np.testing.assert_allclose(overtake_curve(kph / 3.6) / 1e3, [350.0, 350.0, 200.0, 0.0, 0.0], atol=1e-9)

    def test_smoothing_recovers_a_linear_ramp_on_irregular_samples(self):
        t = np.cumsum(np.random.default_rng(1).uniform(0.2, 0.3, 40))
        v, a = smooth(t, 50.0 + 2.5 * t)
        known = np.isfinite(a)                        # The ends may have fewer than 3 samples in the window
        self.assertGreaterEqual(known.sum(), 36)
        np.testing.assert_allclose(a[known], 2.5, atol=1e-9)
        np.testing.assert_allclose(v, 50.0 + 2.5 * t, atol=1e-9)

    def test_runs_need_full_throttle_brake_off_and_two_seconds(self):
        lap = synthetic_lap(Car(380e3, 1.0))
        for i0, i1 in full_throttle_runs(lap):
            self.assertGreaterEqual(lap.t[i1] - lap.t[i0], 2.0)
            self.assertTrue(np.all(lap.throttle[i0:i1 + 1] >= 98.0))
            self.assertTrue(np.all(lap.brake[i0:i1 + 1] < 0.5))
        self.assertEqual(len(full_throttle_runs(lap)), 2)

    def test_segments_find_a_deploy_and_a_clip_stretch_on_each_straight(self):
        lap = synthetic_lap(Car(380e3, 1.0))
        runs, _, _ = segment(lap)
        for run in runs:
            kinds = [st.kind for st in run.stretches]
            self.assertIn("deploy", kinds)
            self.assertIn("clip", kinds)
            deploy = next(st for st in run.stretches if st.kind == "deploy")
            self.assertGreaterEqual(lap.v[deploy.idx[0]] * 3.6, 150.0)
            self.assertLessEqual(lap.t[deploy.idx[-1]] - lap.t[deploy.idx[0]], 2.0 + 1e-9)

    def test_fit_recovers_the_car_that_drove_the_lap(self):
        truth = Car(380e3, 1.05, 0.80)
        lap = synthetic_lap(truth)
        runs, v_s, a_s = segment(lap)
        result, used = fit([Prepared(lap, runs, v_s, a_s)], 0.80)
        self.assertLess(abs(result.p_ice - truth.p_ice), 25e3)
        self.assertLess(abs(result.cda - truth.cda), 0.15)
        self.assertEqual(len(used), result.n_deploy + result.n_clip)

    def test_ceiling_at_fixed_cda_recovers_the_ice_power(self):
        truth = Car(300e3, 0.95, 0.80)
        lap = synthetic_lap(truth, seed=3)
        runs, v_s, a_s = segment(lap)
        result = fit_ceiling([Prepared(lap, runs, v_s, a_s)], 0.80, top=2, cda=0.95)
        self.assertLess(abs(result.p_ice - truth.p_ice), 15e3)


if __name__ == "__main__":
    unittest.main()
