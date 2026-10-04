"""
Tests for the wide minimum-curvature racelines (models/raceline.py) and the raceline sets (models/track.py).

The sweep solves every wide line under both rule sets, like tests.test_spatial_nlp does for TUM's (~2 min):
    RUN_TRACK_SWEEP=1 python -m unittest tests.test_raceline
"""
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from calibration.practice import lateral_g_max
from config import get_ers_config, get_track_config, get_vehicle_config
from config.events import EVENTS_2026
from models import F1TrackModel, VehicleDynamicsModel, find_tumftm_raceline
from models import raceline
from models.raceline import (box_qp, closed_length, line_curvature, load_corridor, make_corridor,
                             min_curvature_offset, write_raceline)
from models.track import RACELINE_ENV, RACELINE_SETS, tum_sibling

WIDE = RACELINE_SETS["wide"]


def stadium(straight=400.0, radius=80.0, step=2.0):
    """An anticlockwise stadium: two straights joined by two semicircles, as a closed (n × 2) centreline."""
    def arc(cx, start):
        a = start + np.arange(0.0, np.pi, step / radius)
        return np.column_stack([cx + radius * np.cos(a), radius * np.sin(a)])
    bottom = np.column_stack([np.arange(0.0, straight, step), np.full(int(straight / step), -radius)])
    right = arc(straight, -0.5 * np.pi)
    top = np.column_stack([np.arange(straight, 0.0, -step), np.full(int(straight / step), radius)])
    left = arc(0.0, 0.5 * np.pi)
    return np.vstack([bottom, right, top, left])


class BoxQPTests(unittest.TestCase):
    def test_separable_problem_is_the_clipped_minimum(self):
        target = np.array([-3.0, -0.5, 0.2, 2.5])
        x = box_qp(sparse.identity(4), -target, np.full(4, -1.0), np.full(4, 1.0))
        np.testing.assert_allclose(x, np.clip(target, -1.0, 1.0), atol=1e-6)

    def test_coupled_problem_meets_the_kkt_conditions(self):
        rng = np.random.default_rng(1)
        M = rng.normal(size=(30, 20))
        H, g = M.T @ M, rng.normal(size=20) * 10
        lower, upper = np.full(20, -0.3), np.full(20, 0.4)
        x = box_qp(sparse.csc_matrix(H), g, lower, upper)
        gradient = H @ x + g
        free = (x > lower + 1e-6) & (x < upper - 1e-6)
        self.assertLess(np.abs(gradient[free]).max(), 1e-5)
        self.assertTrue(np.all(gradient[x <= lower + 1e-6] >= -1e-5))   # Pushing down against the lower bound
        self.assertTrue(np.all(gradient[x >= upper - 1e-6] <= 1e-5))


def circle_corridor(radius=300.0, width=6.0):
    angle = np.arange(0.0, 2 * np.pi, 5.0 / radius)
    centre = radius * np.column_stack([np.cos(angle), np.sin(angle)])
    return make_corridor(centre, np.full(len(angle), width), np.full(len(angle), width), kerb=1.0, margin=0.75)


def total_cost(corridor, offset, weight=raceline.LENGTH_WEIGHT):
    _, _, spacing = raceline._linearised_curvature(corridor, np.zeros(len(corridor)))
    xy = corridor.points(offset)
    return float(np.sum(spacing * line_curvature(xy) ** 2)) + weight * closed_length(xy)


class MinCurvatureTests(unittest.TestCase):
    def test_length_gradient_and_hessian_match_finite_differences(self):
        centre = stadium()
        corridor = make_corridor(centre, np.full(len(centre), 6.0), np.full(len(centre), 6.0))
        rng = np.random.default_rng(2)
        offset = rng.uniform(-3.0, 3.0, len(corridor))
        length, gradient, hessian = raceline._length_terms(corridor, offset)
        self.assertAlmostEqual(length, closed_length(corridor.points(offset)), places=6)
        step = 1e-5 * rng.normal(size=len(corridor))
        change = raceline._length_terms(corridor, offset + step)[1] - raceline._length_terms(corridor, offset - step)[1]
        np.testing.assert_allclose(change / 2, hessian @ step, atol=1e-9)
        up = closed_length(corridor.points(offset + step)) - closed_length(corridor.points(offset - step))
        self.assertAlmostEqual(up / 2, gradient @ step, places=9)

    def test_circle_line_runs_along_the_outside_edge_without_the_length_term(self):
        # On a circle the curvature 1/(R + d) is least on the outside edge (right of an anticlockwise lap)
        corridor = circle_corridor()
        offset = min_curvature_offset(corridor, length_weight=0.0)
        np.testing.assert_allclose(offset, corridor.lower, atol=1e-3)
        np.testing.assert_allclose(np.abs(line_curvature(corridor.points(offset))), 1.0 / (300.0 + 6.25), rtol=1e-3)

    def test_heavy_length_weight_takes_the_inside_edge(self):
        corridor = circle_corridor()
        offset = min_curvature_offset(corridor, length_weight=1.0)
        np.testing.assert_allclose(offset, corridor.upper, atol=1e-3)

    def test_stadium_line_stays_inside_and_beats_the_centreline(self):
        centre = stadium()
        width = np.full(len(centre), 6.0)
        for kerb in (0.0, 1.0):
            corridor = make_corridor(centre, width, width, kerb=kerb, margin=0.75)
            offset = min_curvature_offset(corridor)
            self.assertTrue(np.all(offset >= corridor.lower - 1e-6) and np.all(offset <= corridor.upper + 1e-6))
            self.assertLess(total_cost(corridor, offset), total_cost(corridor, np.zeros(len(corridor))))

    def test_tum_linearisation_with_no_kerb_reproduces_tum(self):
        # With no kerb allowance the corridor is TUM's own, so TUM's formulation should give TUM's raceline
        corridor = load_corridor("Spielberg", kerb=0.0)
        line = corridor.points(min_curvature_offset(corridor, exact=False))
        tum = np.loadtxt(find_tumftm_raceline("Spielberg", RACELINE_SETS["tum"]), delimiter=",", comments="#")
        self.assertLess(abs(closed_length(line) / closed_length(tum) - 1.0), 0.002)
        from scipy.spatial import cKDTree
        self.assertLess(np.median(cKDTree(tum).query(line)[0]), 1.5)


class RacelineFileTests(unittest.TestCase):
    def test_written_file_reads_back_with_the_raceline_loaders(self):
        xy = stadium()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "Stadium.csv"
            write_raceline(path, xy, "test line")
            self.assertEqual(path.read_text().splitlines()[1], "# x_m,y_m")
            np.testing.assert_allclose(np.loadtxt(path, delimiter=",", comments="#"), xy, atol=1e-6)
            self.assertEqual(find_tumftm_raceline("stadium", directory), path)

    def test_every_event_raceline_has_a_wide_line_inside_its_corridor(self):
        for event in EVENTS_2026.values():
            if not event.raceline:
                continue
            with self.subTest(event=event.name):
                path = find_tumftm_raceline(event.raceline, variant="wide")
                self.assertIsNotNone(path)
                self.assertEqual(path.parent, WIDE)
                xy = np.loadtxt(path, delimiter=",", comments="#")
                corridor = load_corridor(event.raceline, kerb=raceline.KERB_ALLOWANCE)
                offset = np.einsum("ij,ij->i", xy - np.column_stack([corridor.x, corridor.y]),
                                   np.column_stack([corridor.nx, corridor.ny]))
                self.assertTrue(np.all(offset >= corridor.lower - 0.01) and np.all(offset <= corridor.upper + 0.01))

    def test_wide_line_names_its_tum_sibling_for_the_layout_check(self):
        self.assertEqual(tum_sibling(find_tumftm_raceline("monza", variant="wide")),
                         find_tumftm_raceline("monza", variant="tum"))
        self.assertIsNone(tum_sibling(find_tumftm_raceline("monza", variant="tum")))

    def test_environment_variable_picks_the_set(self):
        saved = os.environ.get(RACELINE_ENV)
        try:
            os.environ[RACELINE_ENV] = "wide"
            self.assertEqual(find_tumftm_raceline("Suzuka").parent, WIDE)
            os.environ[RACELINE_ENV] = "tum"
            self.assertEqual(find_tumftm_raceline("Suzuka").parent, RACELINE_SETS["tum"])
            self.assertEqual(find_tumftm_raceline("Suzuka", variant="wide").parent, WIDE)
            os.environ[RACELINE_ENV] = "bogus"
            with self.assertRaises(ValueError):
                find_tumftm_raceline("Suzuka")
        finally:
            if saved is None:
                os.environ.pop(RACELINE_ENV, None)
            else:
                os.environ[RACELINE_ENV] = saved


class GripFromBrakingTests(unittest.TestCase):
    def test_lateral_grip_is_the_braking_peak_less_drag_times_the_friction_ratio(self):
        g = lateral_g_max(3.9, 313.0)
        self.assertGreater(g, 3.5)
        self.assertLess(g, 3.9)
        self.assertLess(lateral_g_max(3.9, 330.0), g)   # More drag in the braking peak at a higher speed


@unittest.skipUnless(os.environ.get("RUN_TRACK_SWEEP"), "slow (~2 min): set RUN_TRACK_SWEEP=1")
class WideTrackSweepTests(unittest.TestCase):
    def test_all_wide_lines(self):
        from solvers import SpatialNLPSolver

        for path in sorted(WIDE.glob("*.csv")):
            for regulations in ("2025", "2026"):
                with self.subTest(track=path.stem, regulations=regulations):
                    start = time.time()
                    ers = get_ers_config(regulations)
                    track = F1TrackModel(year=2024, gp=path.stem, ds=5.0)
                    track.load_from_tumftm_raceline(str(path))
                    vehicle = VehicleDynamicsModel(get_vehicle_config(regulations, base=get_track_config(path.stem)), ers)
                    solver = SpatialNLPSolver(vehicle, track, ers, ds=5.0)
                    solver.verbose = False
                    trajectory = solver.solve()
                    self.assertEqual(trajectory.solver_status, "optimal")
                    self.assertLess(time.time() - start, 30.0)
                    print(f"{path.stem} {regulations}: {trajectory.lap_time:.3f} s in {time.time() - start:.1f} s", flush=True)


if __name__ == "__main__":
    unittest.main()
