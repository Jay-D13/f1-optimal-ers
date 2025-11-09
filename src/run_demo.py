from __future__ import annotations
import numpy as np
from fastf1_etl import load_lap, resample_uniform_s
from track_model import curvature_from_xy, lateral_limit
from params import CarParams, ERSParams, SimParams
from ocp_single_lap import build_single_lap_ocp
from mpc_within_lap import run_spatial_mpc
from plotting import plot_traces

# --- Choose a session -------------------------------------------------------
YEAR = 2024
GP = "Monza"          # try: "Monza", "Spa", "Baku" ...
SESSION = "Q"         # 'R' for race, 'Q' for quali
DRIVER = "LEC"        # driver code

# --- Load & preprocess ------------------------------------------------------
telemetry, meta = load_lap(YEAR, GP, SESSION, DRIVER, pick="fastest")
G = resample_uniform_s(telemetry, N=1200)

s = G["s"]; x = G["x"]; y = G["y"]
v_ref = G["v_ref"]

kappa = curvature_from_xy(x, y)

# --- Params ----------------------------------------------------------------
cp = CarParams()
ep = ERSParams()
sp = SimParams(ds=float(np.mean(np.diff(s))))

# --- Single-lap direct OCP (baseline) --------------------------------------
# Initialize v0 from first samples; SOC from mid-band
v0 = max(float(v_ref[0]), sp.v_min)
soc0 = 0.6
socT = 0.6  # cyclic lap target

opti, var = build_single_lap_ocp(s, kappa, cp, ep, sp, v0, soc0, soc_final_target=socT)
sol = opti.solve()
res_ocp = {
    "v": sol.value(var["v"]),
    "soc": sol.value(var["soc"]),
    "P_ice": sol.value(var["P_ice"]),
    "P_k": sol.value(var["P_k"]),
}
res_ocp["lap_time"] = float(np.sum(sp.ds / np.maximum(res_ocp["v"], 1e-3)))
print(f"Direct OCP lap time: {res_ocp['lap_time']:.3f} s (FastF1 lap ~{meta['lap_time_s']:.3f} s)")

plot_traces(s, res_ocp, label_prefix="ocp")

# --- Within-lap NMPC (example) ---------------------------------------------
res_mpc = run_spatial_mpc(s, kappa, cp, ep, sp, v0, soc0, H=150, step=50, soc_terminal=socT)
print(f"MPC lap time: {res_mpc['lap_time']:.3f} s")
plot_traces(s, res_mpc, label_prefix="mpc")