import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict

@dataclass
class OptimalTrajectory:   
    # Spatial discretization
    s: np.ndarray              # Distance points (m)
    ds: float                  # Step size (m)
    n_points: int              # Number of points
    
    # State trajectories
    v_opt: np.ndarray          # Optimal velocity (m/s)
    soc_opt: np.ndarray        # Optimal SOC (0-1)
    
    # Control trajectories  
    P_ers_opt: np.ndarray      # Optimal ERS power (W), + = deploy, - = harvest
    throttle_opt: np.ndarray   # Optimal throttle (0-1)
    brake_opt: np.ndarray      # Optimal brake (0-1)
    
    # Derived quantities
    t_opt: np.ndarray          # Cumulative time (s)
    lap_time: float            # Total lap time (s)
    energy_deployed: float     # Total ERS energy deployed (J)
    energy_recovered: float    # Total ERS energy recovered (J)
    
    # Solver metadata
    solve_time: float          # Computation time (s)
    solver_status: str         # 'optimal', 'suboptimal', 'failed'
    solver_name: str           # Name of solver used
    
    def get_reference_at_distance(self, distance: float) -> Dict:
        """reference values at given distance (with lap wrapping)."""
        distance = distance % self.s[-1]
        idx = np.searchsorted(self.s, distance)
        idx = min(idx, self.n_points - 2)
        
        return {
            'v_ref': self.v_opt[idx],
            'soc_ref': self.soc_opt[idx],
            'P_ers_ref': self.P_ers_opt[idx] if idx < len(self.P_ers_opt) else 0,
            'throttle_ref': self.throttle_opt[idx] if idx < len(self.throttle_opt) else 0,
            'brake_ref': self.brake_opt[idx] if idx < len(self.brake_opt) else 0,
        }
    
    def compute_energy_stats(self) -> Dict:
        """detailed energy statistics"""
        return {
            'total_deployed_MJ': self.energy_deployed / 1e6,
            'total_recovered_MJ': self.energy_recovered / 1e6,
            'net_energy_MJ': (self.energy_deployed - self.energy_recovered) / 1e6,
            'initial_soc': self.soc_opt[0],
            'final_soc': self.soc_opt[-1],
            'soc_swing': self.soc_opt.max() - self.soc_opt.min(),
        }

class BaseSolver(ABC):

    def __init__(self, vehicle_model, track_model, ers_config):

        self.vehicle = vehicle_model
        self.track = track_model
        self.ers = ers_config
        
        # Solver parameters (can be overridden by subclasses)
        self.ds = 5.0  # Spatial discretization (m)
        self.verbose = True
        
    @property
    @abstractmethod
    def name(self) -> str:
        """Solver name for logging and identification."""
        pass
    
    @abstractmethod
    def solve(self, 
              v_limit_profile: np.ndarray,
              initial_soc: float = 0.5,
              final_soc_min: float = 0.3,
              energy_limit: float = 4e6) -> OptimalTrajectory:

        pass
    
    def _log(self, message: str):
        """Print message if verbose mode is on"""
        if self.verbose:
            print(f"   [{self.name}] {message}")
    
    def _compute_lap_time(self, s: np.ndarray, v: np.ndarray) -> tuple:
        """
        Compute cumulative time from velocity profile
        """
        t_opt = np.zeros(len(s))
        for k in range(len(s) - 1):
            v_avg = max(0.5 * (v[k] + v[k+1]), 5.0)
            ds = s[k+1] - s[k] if k < len(s) - 1 else self.ds
            t_opt[k+1] = t_opt[k] + ds / v_avg
        return t_opt, t_opt[-1]
    
    def _compute_energy_totals(self, P_ers: np.ndarray, v: np.ndarray) -> tuple:
        """
        Compute total energy deployed and recovered
        """
        energy_deployed = 0.0
        energy_recovered = 0.0
        
        for k in range(len(P_ers)):
            v_avg = max(v[k], 5.0) if k < len(v) else 50.0
            dt = self.ds / v_avg
            
            if P_ers[k] > 0:
                energy_deployed += P_ers[k] * dt
            else:
                energy_recovered += -P_ers[k] * dt
                
        return energy_deployed, energy_recovered


class VelocityProfileSolver(ABC):
    """Abstract base for velocity profile solvers (no ERS, just grip limits)"""
    
    def __init__(self, vehicle_model, track_model):
        self.vehicle = vehicle_model
        self.track = track_model
        
    @abstractmethod
    def solve(self) -> 'VelocityProfile':
        """Solve for grip-limited velocity profile."""
        pass


@dataclass  
class VelocityProfile:
    """Outputs of velocity profile solvers"""
    s: np.ndarray       # Distance points
    v: np.ndarray       # Velocity at each point
    a_x: np.ndarray     # Longitudinal acceleration
    t: np.ndarray       # Cumulative time
    lap_time: float     # Total lap time (theoretical minimum)
