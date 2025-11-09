from __future__ import annotations
import casadi as ca

from params import CarParams, ERSParams, SimParams


def build_spatial_rhs(cp: CarParams, ep: ERSParams, sp: SimParams):
    """Return CasADi functions for spatial dynamics x' = f(s, x, u) with x' = dx/ds.
    States x = [v, soc]; Controls u = [P_ice, P_k].
    dv/ds = a_x / v,  where a_x = (F_drive - F_res)/m and limited by traction circle via a constraint.
    dsoc/ds = - (power_out_ES)/ (Emax * v)
    """
    v = ca.SX.sym('v')
    soc = ca.SX.sym('soc')
    x = ca.vertcat(v, soc)

    P_ice = ca.SX.sym('P_ice')   # >=0
    P_k   = ca.SX.sym('P_k')     # signed
    u = ca.vertcat(P_ice, P_k)

    # parameters that vary along s (as "online data"): kappa(s)
    kappa = ca.SX.sym('kappa')

    # Resistive forces
    F_drag = 0.5*cp.rho*cp.CdA*v*v
    F_roll = cp.m*cp.g*cp.Cr
    F_res = F_drag + F_roll

    # Wheel power available
    P_wheel = cp.eta_drv * (P_ice + P_k)
    # Longitudinal accel without traction cap
    a_free = (P_wheel - F_res * ca.fmax(v, sp.v_min)) / (cp.m * ca.fmax(v, sp.v_min))

    # Traction-circle longitudinal accel cap
    a_total = 1.6 * (cp.g + 0.5*cp.rho*3.0/cp.m * v*v)  # keep consistent w/ Track model defaults
    a_y = v*v*ca.fabs(kappa)
    a_x_cap = ca.sqrt(ca.fmax(0, a_total*a_total - a_y*a_y))

    # Use constraint a_free <= a_x_cap instead of hard min (smoother)
    a_x = a_free

    # SOC dynamics (positive P_k = deploy to wheels)
    P_es_out = ca.if_else(P_k >= 0, P_k/ep.eta_k_deploy, ep.eta_k_harv*P_k)  # P_es_out signed
    dsoc = - P_es_out / (ep.Ebatt_max * ca.fmax(v, sp.v_min))

    dv = a_x / ca.fmax(v, sp.v_min)

    f = ca.vertcat(dv, dsoc)

    f_fun = ca.Function('f_fun', [x, u, kappa], [f])
    a_constraints = ca.Function('a_constr', [x, u, kappa], [a_free, a_x_cap])
    res_forces = ca.Function('res_forces', [v], [F_drag, F_roll])
    return f_fun, a_constraints, res_forces