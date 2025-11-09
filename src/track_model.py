from __future__ import annotations
import numpy as np
from dataclasses import dataclass

@dataclass
class CorneringModel:
    mu: float = 1.6              # baseline friction coefficient (constant)
    ClA: float = 3.0             # downforce coefficient area [m^2]
    rho: float = 1.2             # air density [kg/m^3]
    m: float = 800.0             # mass [kg]
    g: float = 9.81


def curvature_from_xy(x: np.ndarray, y: np.ndarray, eps: float = 1e-6):
    """Discrete curvature κ(s) from position arrays (m^-1)."""
    x = np.asarray(x); y = np.asarray(y)
    dx = np.gradient(x); dy = np.gradient(y)
    ddx = np.gradient(dx); ddy = np.gradient(dy)
    denom = (dx*dx + dy*dy)**1.5 + eps
    kappa = (dx*ddy - dy*ddx) / denom
    return kappa


def lateral_limit(v: np.ndarray, kappa: np.ndarray, cm: CorneringModel):
    """Return feasible longitudinal acceleration allowance via a traction-circle-like rule.
    a_total_max = mu * (g + 0.5*rho*ClA/m * v^2)
    a_y = v^2 * |kappa|
    max a_x satisfies a_x^2 + a_y^2 <= a_total_max^2
    """
    a_total = cm.mu * (cm.g + 0.5*cm.rho*cm.ClA/cm.m * v**2)
    a_y = (v**2) * np.abs(kappa)
    a_x_max = np.sqrt(np.maximum(0.0, a_total**2 - a_y**2))
    return a_x_max