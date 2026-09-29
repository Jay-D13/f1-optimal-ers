"""
API for the web UI: the 2026 season, the model's runs, and runs started in the background.

A run is `main.py` in a subprocess (no plots); its output streams into the job's log, and its results land in
results/<track>/<run id>/ like any CLI run.
"""
import json
import os
import re
import subprocess
import sys
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config.events import EVENTS_2026  # noqa: E402
from backend.season import RESULTS_DIR, round_runs, run_info, season, track_key  # noqa: E402

app = FastAPI(title="ERS Pole Lab API")
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def health():
    return {"status": "ok"}


@app.get("/season")
def get_season():
    return {"rounds": season()}


@app.get("/rounds/{round_number}/runs")
def get_round_runs(round_number: int):
    if round_number not in EVENTS_2026:
        raise HTTPException(404, "Unknown round")
    return {"runs": round_runs(round_number)}


@app.get("/rounds/{round_number}/raceline")
def get_raceline(round_number: int):
    """The round's TUM raceline (x, y in its own frame), for a map before the first run."""
    from models import find_tumftm_raceline

    event = EVENTS_2026.get(round_number)
    if event is None:
        raise HTTPException(404, "Unknown round")
    path = find_tumftm_raceline(event.raceline) if event.raceline else None
    if path is None:
        return {"x": None, "y": None}
    xy = np.loadtxt(path, delimiter=",", comments="#")
    return {"x": xy[:, 0].tolist(), "y": xy[:, 1].tolist()}


def _load(data_dir: Path, name: str) -> Optional[list]:
    path = data_dir / f"{name}.npy"
    return np.load(path).tolist() if path.exists() else None


def _pole_trace(round_number: int, ds: float) -> Optional[dict]:
    """The real pole lap's speed on the model's path (the calibration reference lap), if it can be had."""
    if not EVENTS_2026[round_number].raceline:
        return None
    try:
        from calibration.dataset import reference_lap
        ref = reference_lap(round_number, ds=ds)
    except Exception as e:  # FastF1 unavailable, no clean lap...
        print(f"No pole trace for round {round_number}: {e}")
        return None
    return {"driver": ref.driver, "lap_time": ref.lap_time, "s": ref.s.tolist(), "v": ref.speed.tolist()}


@app.get("/runs/{track}/{run_id}")
def get_run(track: str, run_id: str, round: Optional[int] = None):
    if not re.fullmatch(r"[\w.-]+", track) or not re.fullmatch(r"[\w.-]+", run_id):
        raise HTTPException(400, "Bad run path")
    run_dir = RESULTS_DIR / track.lower() / run_id
    summary_path = run_dir / "data" / "results_summary.json"
    if not summary_path.is_file():
        raise HTTPException(404, "Run not found")
    summary = json.loads(summary_path.read_text())
    data = run_dir / "data"
    series = {name: _load(data, file) for name, file in (
        ("s", "distance"), ("t", "time"), ("v", "velocity_optimal"), ("soc", "soc_optimal"),
        ("power", "ers_power"), ("throttle", "throttle"), ("brake", "brake"), ("x", "x"), ("y", "y"),
    )}
    if series["s"] is None or series["v"] is None:
        raise HTTPException(404, "Run data files missing")
    meta = summary.get("metadata", {})
    pole = None
    if round in EVENTS_2026 and meta.get("year") == 2026:
        pole = _pole_trace(round, float(meta.get("ds") or 5.0))
    return {
        "info": run_info(run_id, summary, round),
        "summary": summary,
        "series": series,
        "straight_mode_zones": summary.get("track_info", {}).get("straight_mode_zones", []),
        "pole_trace": pole,
    }


# ---------------------------------------------------------------------------
# Background runs
# ---------------------------------------------------------------------------

class RunRequest(BaseModel):
    round: int
    regulations: Literal["2025", "2026"] = "2026"
    session: Literal["qualifying", "race"] = "qualifying"
    laps: int = Field(default=1, ge=1, le=20)
    ds: float = Field(default=5.0, ge=1.0, le=50.0)
    collocation: Literal["euler", "trapezoidal", "hermite_simpson"] = "trapezoidal"
    nlp_solver: Literal["auto", "ipopt", "fatrop", "sqpmethod"] = "auto"
    ipopt_linear_solver: str = Field(default="mumps", pattern=r"^[a-z0-9_]+$")
    ipopt_hessian: Literal["limited-memory", "exact"] = "exact"
    initial_soc: float = Field(default=0.5, ge=0.0, le=1.0)
    final_soc_min: float = Field(default=0.3, ge=0.0, le=1.0)
    per_lap_final_soc_min: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    flying_lap: bool = True
    tire_model: Literal["scalar", "dynamic"] = "scalar"
    tire_compound: Literal["soft", "medium", "hard"] = "medium"
    enable_tire_degradation: bool = False


def build_command(req: RunRequest) -> list:
    """main.py on the event's placed TUM raceline, with the event's 2026 energy rules."""
    event = EVENTS_2026[req.round]
    cmd = [
        sys.executable, "main.py",
        "--track", track_key(event),
        "--year", "2026",
        "--event", str(req.round),
        "--regulations", req.regulations,
        "--session", req.session,
        "--laps", str(req.laps),
        "--ds", str(req.ds),
        "--collocation", req.collocation,
        "--nlp-solver", req.nlp_solver,
        "--ipopt-linear-solver", req.ipopt_linear_solver,
        "--ipopt-hessian", req.ipopt_hessian,
        "--initial-soc", str(req.initial_soc),
        "--final-soc-min", str(req.final_soc_min),
        "--tire-model", req.tire_model,
        "--tire-compound", req.tire_compound,
        "--flying-lap" if req.flying_lap else "--no-flying-lap",
        "--enable-tire-degradation" if req.enable_tire_degradation else "--no-tire-degradation",
        "--use-tumftm",
        "--no-plot",
    ]
    if req.per_lap_final_soc_min is not None:
        cmd += ["--per-lap-final-soc-min", str(req.per_lap_final_soc_min)]
    return cmd


RUN_DIR_LINE = re.compile(r"Results directory: (\S+)")
LOG_LINES = 400
jobs: dict = {}
jobs_lock = threading.Lock()


def _job_view(job: dict) -> dict:
    return {k: v for k, v in job.items() if k != "process"}


def _follow(job: dict):
    """Stream the run's output into its log until it exits."""
    process = job["process"]
    for line in process.stdout:
        line = line.rstrip()
        with jobs_lock:
            job["log"] = (job["log"] + [line])[-LOG_LINES:]
            match = RUN_DIR_LINE.search(line)
            if match:
                job["run_id"] = Path(match.group(1)).name
    code = process.wait()
    with jobs_lock:
        job["finished"] = datetime.now().isoformat()
        summary = RESULTS_DIR / job["track"].lower() / (job["run_id"] or "-") / "data" / "results_summary.json"
        if job["cancelled"]:
            job["status"] = "cancelled"
        elif code == 0 and summary.is_file():
            job["status"] = "done"
        else:
            job["status"] = "failed"
            if code == 0:
                job["log"].append("The run ended without saving results.")


@app.post("/jobs")
def start_job(req: RunRequest):
    event = EVENTS_2026.get(req.round)
    if event is None:
        raise HTTPException(404, "Unknown round")
    if not event.raceline:
        raise HTTPException(400, f"{event.name} has no racing line for its current layout yet")
    with jobs_lock:
        if any(j["status"] == "running" for j in jobs.values()):
            raise HTTPException(409, "A run is already in progress")
        cmd = build_command(req)
        process = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1, env={**os.environ, "PYTHONUNBUFFERED": "1"})
        job = {
            "id": uuid.uuid4().hex[:12], "round": req.round, "track": track_key(event),
            "request": req.model_dump(), "status": "running", "cancelled": False,
            "started": datetime.now().isoformat(), "finished": None, "run_id": None,
            "log": ["main.py " + " ".join(cmd[2:])], "process": process,
        }
        jobs[job["id"]] = job
    threading.Thread(target=_follow, args=(job,), daemon=True).start()
    return _job_view(job)


@app.get("/jobs")
def list_jobs():
    with jobs_lock:
        return {"jobs": [_job_view(j) for j in sorted(jobs.values(), key=lambda j: j["started"], reverse=True)]}


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "Unknown job")
        return _job_view(job)


@app.delete("/jobs/{job_id}")
def cancel_job(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "Unknown job")
        if job["status"] == "running":
            job["cancelled"] = True
            job["process"].terminate()
    return {"ok": True}
