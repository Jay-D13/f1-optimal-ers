from __future__ import annotations
import casadi as ca
import numpy as np
from params import CarParams, ERSParams, SimParams
from dyn_models import build_spatial_rhs


def build_single_lap_ocp(s_grid: np.ndarray, kappa: np.ndarray,
                         cp: CarParams, ep: ERSParams, sp: SimParams,
                         v_init: float, soc_init: float, soc_final_target: float | None = None):
    N = len(s_grid)
    ds = float(np.mean(np.diff(s_grid)))

    f_fun, a_fun, _ = build_spatial_rhs(cp, ep, sp)

    opti = ca.Opti()

    v = opti.variable(N)         # m/s
    soc = opti.variable(N)       # 0..1
    P_ice = opti.variable(N-1)   # W
    P_k   = opti.variable(N-1)   # W (signed)

    # Bounds
    opti.subject_to(v >= sp.v_min)
    opti.subject_to(soc >= ep.soc_min)
    opti.subject_to(soc <= ep.soc_max)
    opti.subject_to(P_ice >= 0)
    opti.subject_to(P_ice <= ep.Pice_max)
    opti.subject_to(P_k >= -ep.Pk_max)
    opti.subject_to(P_k <=  ep.Pk_max)

    # Initial conditions
    opti.subject_to(v[0] == v_init)
    opti.subject_to(soc[0] == soc_init)

    # Dynamics (explicit Euler; swap to RK4 later)
    for k in range(N-1):
        xk = ca.vertcat(v[k], soc[k])
        uk = ca.vertcat(P_ice[k], P_k[k])
        fk = f_fun(xk, uk, kappa[k])
        x_next = xk + ds * fk
        opti.subject_to(v[k+1] == x_next[0])
        opti.subject_to(soc[k+1] == x_next[1])

        # Traction-cap constraint: a_free <= a_x_cap
        a_free, a_cap = a_fun(xk, uk, kappa[k])
        opti.subject_to(a_free <= a_cap)

    # Optional: terminal SOC target (e.g., cyclic lap)
    if soc_final_target is not None:
        opti.subject_to(soc[-1] == soc_final_target)

    # Objective: minimize lap time = ∫ ds / v
    time_cost = ca.sums1(ds / v)

    # Small regularization to avoid bang-bang
    reg = 1e-10 * ca.sumsqr(P_k[1:] - P_k[:-1]) + 1e-12 * ca.sumsqr(P_ice)

    opti.minimize(time_cost + reg)

    # Solver
    p_opts = {"expand": True}
    s_opts = {"max_iter": 3000, "tol": 1e-6}
    opti.solver("ipopt", p_opts, s_opts)

    # Warm start: constant guesses
    opti.set_initial(v, np.maximum(50/3.6, np.ones(N)*80/3.6))
    opti.set_initial(soc, (soc_init + (soc_final_target or soc_init))/2)
    opti.set_initial(P_ice, 0.7*ep.Pice_max)
    opti.set_initial(P_k, 0.0)

    return opti, {"v": v, "soc": soc, "P_ice": P_ice, "P_k": P_k}