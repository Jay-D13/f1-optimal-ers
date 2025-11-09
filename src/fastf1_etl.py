from __future__ import annotations
import numpy as np
import pandas as pd
import fastf1 as ff1
from scipy.interpolate import interp1d

# --- FastF1 helpers ---------------------------------------------------------

def load_lap(year: int, grand_prix: str, session_code: str, driver: str,
             cache_dir: str = "data/cache", pick: str = "fastest"):
    """Load a session and return a single lap telemetry for a driver.
    session_code: 'R' (Race), 'Q' (Quali), 'FP1', 'FP2', ...
    pick: 'fastest' | lap number like '15' | 'median'
    Returns: (lap_df, meta)
    """
    ff1.Cache.enable_cache(cache_dir)
    ses = ff1.get_session(year, grand_prix, session_code)
    ses.load()

    laps = ses.laps.pick_driver(driver)
    if pick == "fastest":
        lap = laps.pick_fastest()
    elif pick == "median":
        lap = laps.sort_values("LapTime").iloc[len(laps)//2]
    else:
        lap = laps[laps["LapNumber"] == int(pick)].iloc[0]

    tel = lap.get_telemetry()  # columns: Time, Distance, Speed, X, Y, Z, RPM, nGear, Throttle, Brake, DRS ...

    # Ensure a clean increasing distance (m)
    if "Distance" not in tel:
        # fallback (most modern FastF1 versions include Distance)
        dxy = np.sqrt(np.diff(tel["X"])**2 + np.diff(tel["Y"])**2)
        dist = np.concatenate([[0.0], np.cumsum(dxy)])
        tel = tel.copy()
        tel["Distance"] = dist

    meta = {
        "year": year,
        "gp": grand_prix,
        "session": session_code,
        "driver": driver,
        "lap_time_s": float(lap["LapTime"].total_seconds()),
    }
    return tel.reset_index(drop=True), meta


def resample_uniform_s(tel: pd.DataFrame, N: int = 1200):
    """Resample telemetry onto a uniform arc-length grid s in [0, L].
    Returns grid dict with s, x, y, z, speed, drs, etc. Missing channels filled with NaN.
    """
    L = float(tel["Distance"].iloc[-1])
    s_grid = np.linspace(0.0, L, N)

    def interp(col, kind="linear"):
        if col not in tel:
            return np.full_like(s_grid, np.nan)
        f = interp1d(tel["Distance"].values, tel[col].values, kind=kind, fill_value="extrapolate")
        return f(s_grid)

    grid = {
        "s": s_grid,
        "L": L,
        "x": interp("X"),
        "y": interp("Y"),
        "z": interp("Z"),
        "v_ref": interp("Speed") / 3.6,  # m/s
        "throttle": interp("Throttle"),
        "brake": interp("Brake"),
        "drs": interp("DRS"),
        "gear": interp("nGear"),
        "rpm": interp("RPM"),
    }
    return grid