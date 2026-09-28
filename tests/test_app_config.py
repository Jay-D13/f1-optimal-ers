import tempfile
import unittest
from pathlib import Path
from unittest import mock

from typer.testing import CliRunner

import main
from config.app_config import app
from models import find_tumftm_raceline

RACELINES = Path(__file__).resolve().parents[1] / "data" / "racelines"


class ConfigFileTests(unittest.TestCase):
    def run_cli(self, yaml_text, *cli_args):
        """Invoke the CLI with a YAML config; return the AppConfig handed to main()."""
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "run.yaml"
            config_path.write_text(yaml_text)
            with mock.patch.object(main, "main") as run_main:
                result = CliRunner().invoke(app, ["--config", str(config_path), *cli_args])
        return result, (run_main.call_args.args[0] if run_main.called else None)

    def test_yaml_overrides_defaults(self):
        result, args = self.run_cli("track: Monza\nds: 2.5\nipopt_hessian: limited-memory\n")
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(args.track, "Monza")
        self.assertEqual(args.ds, 2.5)
        self.assertEqual(args.ipopt_hessian, "limited-memory")
        self.assertEqual(args.year, 2024)  # Not in the YAML: default

    def test_command_line_overrides_yaml(self):
        result, args = self.run_cli("track: Monza\nds: 2.5\n", "--ds", "10", "--no-plot")
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(args.track, "Monza")
        self.assertEqual(args.ds, 10.0)
        self.assertFalse(args.plot)

    def test_unknown_yaml_key_is_rejected(self):
        result, args = self.run_cli("trakc: Monza\n")
        self.assertNotEqual(result.exit_code, 0)
        self.assertIsNone(args)


class RacelineLookupTests(unittest.TestCase):
    def test_case_insensitive(self):
        self.assertEqual(find_tumftm_raceline("suzuka", RACELINES).name, "Suzuka.csv")
        self.assertEqual(find_tumftm_raceline("MONZA", RACELINES).name, "monza.csv")
        self.assertIsNone(find_tumftm_raceline("Atlantis", RACELINES))


if __name__ == "__main__":
    unittest.main()
