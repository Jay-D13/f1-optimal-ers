#!/usr/bin/env python
"""Run a command while holding the machine-wide solver lock, so one Ipopt job runs at a time.

Usage::

    .venv/bin/python scripts/solver_lock.py --label WP2 -- .venv/bin/python -m calibration.model ...
    .venv/bin/python scripts/solver_lock.py --status

Model laps, sweeps, calibration fits, predictions and the unit tests all call Ipopt. Two of them
together on this machine are slower than one after the other, so every agent and session wraps
such commands in this script. It waits until the lock is free (saying who holds it), records
itself as the holder, runs the command and releases the lock when the command exits, also on
Ctrl-C or SIGTERM. A crashed holder releases it too: flock locks die with their process.

The lock file is data/cache/solver.lock in the main checkout, found through the shared git
directory so every worktree uses the same file. ERS_SOLVER_LOCK overrides the path.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

POLL = 2.0            # seconds between lock attempts
REPORT_EVERY = 60.0   # seconds between "still waiting" lines


def lock_path() -> Path:
    override = os.environ.get("ERS_SOLVER_LOCK")
    if override:
        return Path(override)
    here = Path(__file__).resolve().parent.parent
    try:
        common = subprocess.run(["git", "rev-parse", "--git-common-dir"], cwd=here, capture_output=True,
                                text=True, check=True).stdout.strip()
        root = (here / common).resolve().parent
    except (OSError, subprocess.CalledProcessError):
        root = here
    return root / "data" / "cache" / "solver.lock"


def read_holder(path: Path) -> dict:
    try:
        text = path.read_text().strip()
        return json.loads(text) if text else {}
    except (OSError, ValueError):
        return {}


def describe(holder: dict) -> str:
    if not holder:
        return "an unknown process"
    since = holder.get("since", "?")
    return f"{holder.get('label', '?')} (pid {holder.get('pid', '?')}, since {since}): {holder.get('command', '')}"


def open_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return os.fdopen(os.open(path, os.O_RDWR | os.O_CREAT, 0o644), "r+")


def status(path: Path) -> int:
    with open_lock(path) as fh:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            print(f"solver lock {path} is held by {describe(read_holder(path))}")
            return 1
        fcntl.flock(fh, fcntl.LOCK_UN)
    print(f"solver lock {path} is free")
    return 0


def run(label: str, timeout: float | None, command: list[str]) -> int:
    path = lock_path()
    fh = open_lock(path)
    t0 = time.monotonic()
    last_report = -REPORT_EVERY
    while True:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except OSError:
            waited = time.monotonic() - t0
            if timeout is not None and waited > timeout:
                print(f"solver_lock: gave up after {waited:.0f} s; the lock is held by "
                      f"{describe(read_holder(path))}", file=sys.stderr, flush=True)
                fh.close()
                return 75
            if waited - last_report >= REPORT_EVERY:
                print(f"solver_lock: waiting ({waited:.0f} s so far) for the solver lock held by "
                      f"{describe(read_holder(path))}", file=sys.stderr, flush=True)
                last_report = waited
            time.sleep(POLL)
    waited = time.monotonic() - t0
    fh.seek(0)
    fh.truncate()
    json.dump({"pid": os.getpid(), "label": label, "cwd": str(Path.cwd()), "command": " ".join(command),
               "since": dt.datetime.now().isoformat(timespec="seconds")}, fh)
    fh.flush()
    if waited > POLL:
        print(f"solver_lock: acquired after {waited:.0f} s", file=sys.stderr, flush=True)
    proc = subprocess.Popen(command)   # close_fds=True: the child does not inherit the lock

    def forward(signum, _frame):
        proc.send_signal(signum)

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, forward)
    try:
        code = proc.wait()
    finally:
        fh.seek(0)
        fh.truncate()
        fh.flush()
        fcntl.flock(fh, fcntl.LOCK_UN)
        fh.close()
    return code if code >= 0 else 128 - code


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run a command while holding the machine-wide solver lock.")
    parser.add_argument("--label", default="", help="who holds the lock, e.g. WP2 (default: the current directory's name)")
    parser.add_argument("--timeout", type=float, default=None, help="give up after this many seconds (default: wait)")
    parser.add_argument("--status", action="store_true", help="show who holds the lock and exit")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="the command to run, after --")
    args = parser.parse_args(argv)
    if args.status:
        return status(lock_path())
    command = args.command[1:] if args.command and args.command[0] == "--" else args.command
    if not command:
        parser.error("give the command to run after --, or use --status")
    return run(args.label or Path.cwd().name, args.timeout, command)


if __name__ == "__main__":
    sys.exit(main())
