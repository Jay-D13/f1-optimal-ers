from __future__ import annotations
import numpy as np
import casadi as ca
from ocp_single_lap import build_single_lap_ocp


def run_spatial_mpc(s_grid, kappa, cp, ep, sp, v0, soc0,
                    H: int = 120, step: int = 40, soc_terminal: float | None = None):
    """Slide a window of length H over the lap, re-solving every `step` nodes.
    Returns dict with stitched trajectories and total lap time.
    """
    N = len(s_grid)
    v_trj = np.zeros(N)
    soc_trj = np.zeros(N)
    Pice_trj = np.zeros(N-1)
    Pk_trj = np.zeros(N-1)

    idx = 0
    v_curr = float(v0)
    soc_curr = float(soc0)

    while idx < N-1:
        end = min(N, idx + H)
        sub_s = s_grid[idx:end]
        sub_k = kappa[idx:end]
        # Terminal SOC target in final window if requested
        soc_T = soc_terminal if end == N and soc_terminal is not None else None

        opti, vars = build_single_lap_ocp(sub_s, sub_k, cp, ep, sp, v_curr, soc_curr, soc_T)
        sol = opti.solve()

        v_sol = sol.value(vars["v"])
        soc_sol = sol.value(vars["soc"])
        Pice_sol = sol.value(vars["P_ice"])
        Pk_sol = sol.value(vars["P_k"])

        take = min(step, len(v_sol) - 1)  # commit `step` points (except last is boundary)
        v_trj[idx:idx+take] = v_sol[:take]
        soc_trj[idx:idx+take] = soc_sol[:take]
        Pice_trj[idx:idx+take] = Pice_sol[:take]
        Pk_trj[idx:idx+take] = Pk_sol[:take]

        # Update state for next window at boundary `idx+take`
        v_curr = float(v_sol[take])
        soc_curr = float(soc_sol[take])
        idx += take

    ds = float(np.mean(np.diff(s_grid)))
    lap_time = float(np.sum(ds / np.maximum(v_trj, 1e-3)))
    return {
        "v": v_trj, "soc": soc_trj, "P_ice": Pice_trj, "P_k": Pk_trj, "lap_time": lap_time
    }