"""The solver lock wrapper runs wrapped commands one at a time and releases the lock afterwards."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "solver_lock.py"
CHILD = "import json, time; t0 = time.time(); time.sleep(0.5); print(json.dumps([t0, time.time()]))"


class SolverLockTest(unittest.TestCase):
    def test_wrapped_commands_never_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = dict(os.environ, ERS_SOLVER_LOCK=os.path.join(tmp, "lock"))
            procs = [subprocess.Popen([sys.executable, str(SCRIPT), "--label", f"t{i}", "--", sys.executable, "-c", CHILD],
                                      env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                     for i in range(3)]
            spans = []
            for proc in procs:
                out, err = proc.communicate(timeout=60)
                self.assertEqual(proc.returncode, 0, err)
                spans.append(json.loads(out.strip().splitlines()[-1]))
            spans.sort()
            for (_, end), (start, _) in zip(spans, spans[1:]):
                self.assertGreaterEqual(start, end, "two wrapped commands ran at the same time")
            free = subprocess.run([sys.executable, str(SCRIPT), "--status"], env=env, capture_output=True, text=True)
            self.assertEqual(free.returncode, 0, free.stdout + free.stderr)
            self.assertIn("is free", free.stdout)

    def test_exit_code_is_forwarded(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = dict(os.environ, ERS_SOLVER_LOCK=os.path.join(tmp, "lock"))
            result = subprocess.run([sys.executable, str(SCRIPT), "--", sys.executable, "-c", "import sys; sys.exit(3)"],
                                    env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 3, result.stderr)


if __name__ == "__main__":
    unittest.main()
