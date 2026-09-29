from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import sys

# Add project root to sys.path to allow imports from main project
sys.path.insert(0, str(Path(__file__).parent.parent))

app = FastAPI(title="F1 ERS Optimal Control API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For local development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import pandas as pd
import subprocess
import json
import fastf1
from pydantic import BaseModel, Field
from typing import Optional, Literal
import numpy as np

DATA_DIR = Path(__file__).parent.parent / "data"

class SimulationRequest(BaseModel):
    track: str
    year: int = Field(default=2024, ge=2018)
    laps: int = Field(default=1, ge=1)
    regulations: Literal["2025", "2026"] = "2025"
    initial_soc: float = Field(default=0.5, ge=0.0, le=1.0)
    final_soc_min: float = Field(default=0.3, ge=0.0, le=1.0)
    per_lap_final_soc_min: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    ds: float = Field(default=5.0, gt=0.0)
    collocation: Literal["euler", "trapezoidal", "hermite_simpson"] = "trapezoidal"
    nlp_solver: Literal["auto", "ipopt", "fatrop", "sqpmethod"] = "auto"
    flying_lap: bool = True
    enable_tire_degradation: bool = False
    tire_wear_rate_per_lap: float = Field(default=0.012, ge=0.0)
    tire_min_grip_scale: float = Field(default=0.88, gt=0.0, le=1.0)
    tire_model: Literal["scalar", "dynamic"] = "scalar"
    tire_compound: Literal["soft", "medium", "hard"] = "medium"
    ambient_temp_c: float = 25.0
    track_temp_c: float = 35.0
    tire_init_temp_c: float = 80.0
    ipopt_linear_solver: str = "mumps"
    ipopt_hessian: Literal["limited-memory", "exact"] = "exact"
    use_tumftm: bool = False
    driver: Optional[str] = None


def _build_simulation_command(req: SimulationRequest) -> list[str]:
    cmd = [
        "uv", "run", "python", "main.py",
        "--track", req.track,
        "--year", str(req.year),
        "--laps", str(req.laps),
        "--regulations", req.regulations,
        "--initial-soc", str(req.initial_soc),
        "--final-soc-min", str(req.final_soc_min),
        "--ds", str(req.ds),
        "--collocation", req.collocation,
        "--nlp-solver", req.nlp_solver,
        "--ipopt-linear-solver", req.ipopt_linear_solver,
        "--ipopt-hessian", req.ipopt_hessian,
        "--tire-wear-rate-per-lap", str(req.tire_wear_rate_per_lap),
        "--tire-min-grip-scale", str(req.tire_min_grip_scale),
        "--tire-model", req.tire_model,
        "--tire-compound", req.tire_compound,
        "--ambient-temp-c", str(req.ambient_temp_c),
        "--track-temp-c", str(req.track_temp_c),
        "--tire-init-temp-c", str(req.tire_init_temp_c),
        "--flying-lap" if req.flying_lap else "--no-flying-lap",
        "--enable-tire-degradation" if req.enable_tire_degradation else "--no-tire-degradation",
    ]

    if req.use_tumftm:
        cmd.append("--use-tumftm")

    if req.per_lap_final_soc_min is not None:
        cmd.extend(["--per-lap-final-soc-min", str(req.per_lap_final_soc_min)])

    if req.driver:
        cmd.extend(["--driver", req.driver])

    return cmd

# Configure FastF1 cache
CACHE_DIR = DATA_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE_DIR))

@app.get("/")
async def root():
    return {"message": "F1 ERS Optimal Control API is running"}

@app.get("/tracks")
async def list_tracks():
    """List all available local tracks from data/tracks directory."""
    tracks_dir = DATA_DIR / "tracks"
    if not tracks_dir.exists():
        return {"tracks": []}
    
    tracks = []
    for file in tracks_dir.glob("*.csv"):
        tracks.append(file.stem)
    
    return {"tracks": sorted(tracks)}

@app.get("/fastf1/years")
async def list_fastf1_years():
    """List available years for FastF1."""
    # FastF1 supports from 2018 usually for good ERS data, but 2018+ is safe
    return {"years": list(range(2018, 2026))}

@app.get("/fastf1/tracks/{year}")
async def list_fastf1_tracks(year: int):
    """List tracks for a given year using FastF1."""
    try:
        schedule = fastf1.get_event_schedule(year)
        # Filter for completed events or valid ones
        events = []
        for _, event in schedule.iterrows():
            if event['EventName'] != "Pre-Season Testing":
                events.append({
                    "name": event['EventName'],
                    "location": event['Location'],
                    "round": event['RoundNumber'],
                    "country": event['Country'],
                    "official_name": event['OfficialEventName']
                })
        return {"tracks": events}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/fastf1/drivers/{year}/{location}")
async def list_fastf1_drivers(year: int, location: str):
    """List drivers for a session."""
    try:
        # We need to fuzzy match location to event
        schedule = fastf1.get_event_schedule(year)
        # Try to find event by location or name
        # Simple match for now
        event = schedule[schedule['Location'].str.contains(location, case=False, regex=False)]
        if event.empty:
             event = schedule[schedule['EventName'].str.contains(location, case=False, regex=False)]
        
        if event.empty:
            raise HTTPException(status_code=404, detail="Event not found")
            
        # Load session (Qualifying is best for pure speed)
        session = fastf1.get_session(year, event.iloc[0]['RoundNumber'], 'Q')
        session.load(telemetry=False, weather=False, messages=False)
        
        drivers = []
        for driver in session.drivers:
            drivers.append({
                "id": driver,
                "code": session.get_driver(driver)["Abbreviation"],
                "team": session.get_driver(driver)["TeamName"]
            })
            
        return {"drivers": drivers}
    except Exception as e:
        # Fallback to simple list if session load fails (e.g. future event)
        print(f"Error loading drivers: {e}")
        return {"drivers": [
            {"id": "VER", "code": "VER", "team": "Red Bull Racing"},
            {"id": "HAM", "code": "HAM", "team": "Mercedes"},
            {"id": "LEC", "code": "LEC", "team": "Ferrari"},
            {"id": "NOR", "code": "NOR", "team": "McLaren"},
            {"id": "ALO", "code": "ALO", "team": "Aston Martin"},
        ]}

@app.get("/track/{track_id}")
async def get_track_data(track_id: str):
    """Get track centerline and width data."""
    csv_path = DATA_DIR / "tracks" / f"{track_id}.csv"
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="Track not found")
    
    try:
        df = pd.read_csv(csv_path)
        # Clean column names: remove leading/trailing whitespace and #
        df.columns = df.columns.str.strip().str.replace('^#\s*', '', regex=True)
        
        # Ensure correct columns (handle potential variations if any)
        # Columns: x_m, y_m, w_tr_right_m, w_tr_left_m
        
        # Determine center line (x_m, y_m)
        # We return a list of points
        data = []
        for _, row in df.iterrows():
            data.append({
                "x": row["x_m"],
                "y": row["y_m"],
                "w_left": row["w_tr_left_m"],
                "w_right": row["w_tr_right_m"]
            })
        
        return {"track_data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load track data: {str(e)}")

@app.get("/raceline/{track_id}")
async def get_raceline(track_id: str, type: str = "tumftm"):
    """Get raceline data."""
    if type == "tumftm":
        csv_path = DATA_DIR / "racelines" / f"{track_id.lower()}.csv"
        if not csv_path.exists():
             # Fallback to check if Uppercase exists or something, but standard seems to be lowercase for racelines
             racelines = list((DATA_DIR / "racelines").glob("*.csv"))
             # Try case insensitive match
             match = next((p for p in racelines if p.stem.lower() == track_id.lower()), None)
             if not match:
                 return {"raceline": None, "message": "No TUMFTM raceline found"}
             csv_path = match

        try:
            # TUMFTM raceline format: x_m, y_m, w_tr_right_m, w_tr_left_m
            # Sometimes it might just be x, y. 
            # The models/track.py says: x = data[:, 0], y = data[:, 1]
            data = pd.read_csv(csv_path, header=None) # TUMFTM racelines often don't have headers or have comments
            # Need to handle comment lines if read_csv doesn't automatically
            
            # Let's inspect racelines in next step if this fails, but for now assume standard csv
            # We'll use numpy to be safer if it has comments
            arr = np.loadtxt(csv_path, delimiter=',', comments='#')
            
            points = []
            for i in range(arr.shape[0]):
                points.append({"x": arr[i, 0], "y": arr[i, 1]})
            
            return {"raceline": points, "source": "tumftm"}
        except Exception as e:
             raise HTTPException(status_code=500, detail=f"Failed to load raceline: {str(e)}")
    
    return {"raceline": None, "message": "FastF1 raceline not yet implemented via API"}

@app.post("/simulation")
async def run_simulation(req: SimulationRequest):
    """Run the offline simulation."""
    cmd = _build_simulation_command(req)
    
    # We want JSON output. The main.py saves to `results/track/timestamp/data/results_summary.json`.
    # We need to capture the output directory from stdout or just look for the most recent one.
    # main.py prints: "📁 All results saved to: results/..."
    
    try:
        # Use Popen to stream? No, wait for completion for now
        process = subprocess.run(cmd, cwd=str(Path(__file__).parent.parent), capture_output=True, text=True)
        
        if process.returncode != 0:
            return {"status": "error", "message": process.stderr, "stdout": process.stdout}
            
        # Fix: Ensure track name is lowercase to match RunManager's behavior
        results_dir = DATA_DIR.parent / "results" / req.track.lower()
        if not results_dir.exists():
             return {"status": "error", "message": "Results directory not found", "stdout": process.stdout}
        
        # Get latest subdir
        subdirs = sorted([d for d in results_dir.iterdir() if d.is_dir()], key=lambda x: x.stat().st_mtime)
        if not subdirs:
             return {"status": "error", "message": "No results found in directory", "stdout": process.stdout}
             
        latest_run = subdirs[-1]
        summary_file = latest_run / "data" / "results_summary.json"
        
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                results = json.load(f)
            return {"status": "success", "results": results, "run_id": latest_run.name}
        else:
             return {"status": "error", "message": "Results summary file missing", "stdout": process.stdout}
             
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/simulation/{track}/{run_id}/data")
async def get_simulation_data(track: str, run_id: str):
    """Get detailed simulation data for visualization."""
    try:
        data_dir = DATA_DIR.parent / "results" / track.lower() / run_id / "data"
        if not data_dir.exists():
            raise HTTPException(status_code=404, detail="Run data not found")
        
        # Load required numpy files
        # We need: s (distance), v_opt (velocity), soc_opt (SOC), P_ers_opt (power), t_opt (time if available, else construct)
        
        try:
            s = np.load(data_dir / "distance.npy")
            t = np.load(data_dir / "time.npy")
            v = np.load(data_dir / "velocity_optimal.npy")
            soc = np.load(data_dir / "soc_optimal.npy")
            power = np.load(data_dir / "ers_power.npy")
            throttle = np.load(data_dir / "throttle.npy")
            brake = np.load(data_dir / "brake.npy")
        except FileNotFoundError:
             raise HTTPException(status_code=404, detail="Essential data files missing")

        # Construct response
        # We'll return arrays as lists
        return {
            "s": s.tolist(),
            "t": t.tolist(),
            "v": v.tolist(),
            "soc": soc.tolist(),
            "power": power.tolist(),
            "throttle": throttle.tolist(),
            "brake": brake.tolist()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
