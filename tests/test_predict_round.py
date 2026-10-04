"""scripts/predict_round.py: parameter sets, file names, the record and the ledger (no FastF1, no solver)."""
import csv
import datetime as dt
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("predict_round", ROOT / "scripts" / "predict_round.py")
predict_round = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(predict_round)


def synthetic_trajectory():
    """A 1 km 'lap': accelerate at full throttle, clip (full throttle, slowing), brake, then lift."""
    s = np.linspace(0.0, 1000.0, 201)
    v = np.interp(s, [0.0, 400.0, 600.0, 800.0, 1000.0], [50.0, 80.0, 70.0, 40.0, 50.0])
    t = np.concatenate([[0.0], np.cumsum(np.diff(s) / (0.5 * (v[1:] + v[:-1])))])
    n = len(s) - 1
    throttle = np.zeros(n)
    throttle[:120] = 1.0
    brake = np.zeros(n)
    brake[120:160] = 1.0
    P_harvest = np.where(np.arange(n) < 120, 0.0, 200e3)
    P_harvest[80:120] = 350e3
    P_deploy = np.where(np.arange(n) < 80, 350e3, 0.0)
    return SimpleNamespace(
        s=s, v_opt=v, t_opt=t, soc_opt=np.linspace(1.0, 0.5, len(s)), throttle_opt=throttle, brake_opt=brake,
        P_ers_opt=P_deploy - P_harvest, P_deploy_opt=P_deploy, P_harvest_opt=P_harvest, lap_time=float(t[-1]),
        energy_deployed=4e6, energy_recovered=3e6, solver_status="optimal", solve_time=12.0,
        run_up={"distance": 300.0, "v_start": 30.0},
    )


ERS = SimpleNamespace(mgu_k_efficiency=0.95, recovery_limit_per_lap=7.5e6, superclip_power=350e3, ramp_rate=100e3)


class ParameterSetTest(unittest.TestCase):
    def test_hash_ignores_order_and_sees_changes(self):
        a = {"c_w_a": 0.95, "mu_scale": 1.0}
        self.assertEqual(predict_round.params_hash(a), predict_round.params_hash(dict(reversed(list(a.items())))))
        self.assertNotEqual(predict_round.params_hash(a), predict_round.params_hash({**a, "mu_scale": 0.99}))
        self.assertEqual(len(predict_round.params_hash(a)), 8)

    def test_builtin_sets(self):
        name, start = predict_round.parameter_set("start")
        self.assertEqual(name, "start")
        from calibration.model import start_values
        self.assertEqual(start["pow_max_ice"], start_values()["pow_max_ice"])
        name, fit2 = predict_round.parameter_set("fit2")
        self.assertEqual((name, fit2["c_z_a"], fit2["g_max"]), ("fit2", 5.3, 0.0))
        fit2["c_z_a"] = 0.0                       # A copy: the module's set is unchanged
        self.assertEqual(predict_round.FIT2["c_z_a"], 5.3)

    def test_json_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "wp8_car.json"
            path.write_text(json.dumps({"c_w_a": 1.0, "pow_max_ice": 380000}))
            name, values = predict_round.parameter_set(str(path))
            self.assertEqual(name, "wp8_car")
            self.assertEqual(values, {"c_w_a": 1.0, "pow_max_ice": 380000.0})
            path.write_text(json.dumps({"c_w_a": "high"}))
            with self.assertRaises(ValueError):
                predict_round.parameter_set(str(path))

    def test_file_name(self):
        created = dt.datetime(2026, 10, 2, 22, 30, 5, tzinfo=dt.timezone.utc)
        self.assertEqual(predict_round.prediction_name(16, "abcd1234", created), "R16_abcd1234_20261002T223005Z.json")
        self.assertEqual(predict_round.prediction_name(16, "abcd1234", created, "FP3"),
                         "R16_abcd1234_FP3_20261002T223005Z.json")

    def test_qualifying_sessions_refused(self):
        for name in ("Q", "R", "Sprint"):
            with self.assertRaises(ValueError):
                predict_round.available_sessions(16, [name])


class RecordTest(unittest.TestCase):
    def test_lap_structure(self):
        out = predict_round.lap_structure(synthetic_trajectory(), ERS)
        traj = synthetic_trajectory()
        dt_s = np.diff(traj.t_opt)
        self.assertAlmostEqual(out["full_throttle_s"], dt_s[:120].sum())
        self.assertAlmostEqual(out["braking_s"], dt_s[120:160].sum())
        self.assertAlmostEqual(out["clip_s"], dt_s[80:120].sum())        # Full throttle and slowing
        self.assertAlmostEqual(out["harvest_full_throttle_MJ"], (350e3 * 0.95 * dt_s[80:120]).sum() / 1e6)

    def test_record_and_ledger(self):
        traj = synthetic_trajectory()
        created = dt.datetime(2026, 10, 2, 22, 30, 5, tzinfo=dt.timezone.utc)
        placement = {"session": "FP2", "raceline": "Sepang.csv", "gap_m": 7.9, "air_density": 1.13,
                     "length_m": 1000.0, "zones": [[900.0, 100.0]], "zones_source": "test"}
        params = {"c_w_a": 0.95}
        record = predict_round.prediction_record(
            round_number=16, event_name="Bahrain (Sepang)", params_name="start", params=params, label="",
            created=created, sessions=["FP1", "FP2"], placement=placement, g_max=5.0,
            kept=np.array([1.0, 0.8, 1.0]), ers=ERS, traj=traj, git={"commit": "abc", "dirty": False})
        self.assertEqual(record["params_hash"], predict_round.params_hash(params))
        self.assertEqual(record["path"]["points_bounded"], 1)
        self.assertEqual(len(record["trace"]["speed_kmh"]), len(traj.s))
        self.assertAlmostEqual(record["result"]["lap_time_s"], traj.lap_time)
        json.dumps(record)                                                 # Serialisable

        with tempfile.TemporaryDirectory() as d:
            directory = Path(d)
            name = predict_round.prediction_name(16, record["params_hash"], created)
            (directory / name).write_text(json.dumps(record))
            (directory / "R13_other.json").write_text(json.dumps({**record, "round": 13}))
            pole = traj.lap_time - 0.5
            settled = dt.datetime(2026, 10, 3, 9, 0, tzinfo=dt.timezone.utc)
            rows = predict_round.ledger_rows(16, directory, pole, "RUS", "manual", settled, set())
            self.assertEqual([r["file"] for r in rows], [name])
            self.assertEqual(rows[0]["error_s"], "+0.500")
            ledger = directory / "ledger.csv"
            predict_round.append_ledger(ledger, rows)
            self.assertEqual(predict_round.ledger_files(ledger), {name})
            # Settling again adds nothing
            again = predict_round.ledger_rows(16, directory, pole, "RUS", "manual", settled,
                                              predict_round.ledger_files(ledger))
            self.assertEqual(again, [])
            with open(ledger, newline="") as f:
                self.assertEqual(len(list(csv.DictReader(f))), 1)


if __name__ == "__main__":
    unittest.main()
