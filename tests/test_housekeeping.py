import os
import re
import sys
import tomllib
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("MPLBACKEND", "Agg")

from config import ERSConfig, VehicleConfig, find_track, get_ers_config, get_track_config, track_raceline  # noqa: E402
from config.events import EVENTS_2026  # noqa: E402
from config.tracks import TRACKS, normalise  # noqa: E402
from models import find_tumftm_raceline  # noqa: E402
from visualization.results_viz import power_limits, soc_band  # noqa: E402

RACELINES = ROOT / "data" / "racelines"


class TestTrackRegistry(unittest.TestCase):
    def test_names_are_unique(self):
        seen = {}
        for track in TRACKS:
            for name in track.names:
                self.assertNotIn(name, seen, f"{name} is both {seen.get(name)} and {track.name}")
                seen[name] = track.name

    def test_every_bundled_raceline_belongs_to_one_track(self):
        stems = sorted(path.stem for path in RACELINES.glob("*.csv"))
        self.assertEqual(stems, sorted(t.raceline for t in TRACKS if t.raceline))
        for track in TRACKS:
            if track.raceline:
                self.assertIsNotNone(find_tumftm_raceline(track.raceline, RACELINES), track.name)

    def test_every_2026_event_is_a_track(self):
        for number, event in EVENTS_2026.items():
            track = find_track(number)
            self.assertIsNotNone(track, event.name)
            self.assertIs(track.event_2026, event)
            for name in event.tracks:
                self.assertIs(find_track(name, 2026), track, name)
            # The event's raceline (current layout only) is the track's bundled one
            if event.raceline:
                self.assertEqual(track.raceline, event.raceline)

    def test_bahrain_depends_on_the_year(self):
        self.assertEqual(find_track("Bahrain", 2026).name, "Sepang")
        self.assertEqual(find_track("Bahrain", 2024).name, "Sakhir")
        self.assertIsNone(find_track("Sakhir").event_2026)

    def test_names_are_normalised(self):
        self.assertEqual(find_track("Montréal").name, "Montreal")
        self.assertEqual(find_track("CANADA").name, "Montreal")
        self.assertEqual(find_track("Mexico City").raceline, "MexicoCity")
        self.assertEqual(find_track("São Paulo").raceline, "SaoPaulo")
        self.assertEqual(normalise("Spa-Francorchamps"), "spafrancorchamps")

    def test_presets_and_racelines(self):
        self.assertEqual(get_track_config("Monaco").c_w_a, VehicleConfig.for_monaco().c_w_a)
        self.assertEqual(get_track_config("Italy").c_w_a, VehicleConfig.for_monza().c_w_a)
        self.assertEqual(get_track_config("Atlantis").c_w_a, VehicleConfig().c_w_a)
        self.assertEqual(track_raceline("Canada"), "montreal")
        self.assertIsNone(track_raceline("Monaco"))
        self.assertEqual(track_raceline("Atlantis"), "Atlantis")


class TestRequirements(unittest.TestCase):
    def test_requirements_match_pyproject(self):
        with open(ROOT / "pyproject.toml", "rb") as f:
            pyproject = tomllib.load(f)["project"]["dependencies"]
        lines = (ROOT / "requirements.txt").read_text().splitlines()
        requirements = [line.strip() for line in lines if line.strip() and not line.startswith("#")]

        def spec(entries):
            return {re.split(r"[<>=!~ ]", e, maxsplit=1)[0].lower(): e.replace(" ", "") for e in entries}

        self.assertEqual(spec(requirements), spec(pyproject))


class TestPlotLimits(unittest.TestCase):
    def test_2025_limits(self):
        self.assertEqual(soc_band(ERSConfig()), (20.0, 90.0))
        deploy, harvest = power_limits(np.array([50.0, 90.0]), get_ers_config("2025"))
        np.testing.assert_allclose(deploy, [120.0, 120.0])
        self.assertEqual(harvest, -120.0)

    def test_2026_qualifying_limits(self):
        ers = get_ers_config("2026", session="qualifying", event="monza")
        low, high = soc_band(ers)
        self.assertAlmostEqual(low, 100.0 * (1.0 - 4.0 / 4.5))
        self.assertEqual(high, 100.0)
        # Overtake curve: 350 kW up to 337.5 km/h, 0 from 355 km/h
        deploy, harvest = power_limits(np.array([300.0, 346.0, 360.0]) / 3.6, ers)
        np.testing.assert_allclose(deploy, [350.0, 180.0, 0.0], atol=1e-9)
        self.assertEqual(harvest, -350.0)


if __name__ == "__main__":
    unittest.main()
