import numpy as np
from dataclasses import dataclass
from models import F1TrackModel, VehicleDynamicsModel

@dataclass
class VelocityProfile:
    s: np.ndarray
    v: np.ndarray
    ax: np.ndarray
    t: np.ndarray
    lap_time: float

class ApexVelocitySolver:
    """
    Calculates the ultimate grip-limited velocity profile
    using Forward-Backward integration.
    """
    def __init__(self, vehicle: VehicleDynamicsModel, track: F1TrackModel):
        self.vehicle = vehicle.vehicle
        self.track = track
        
    def solve(self) -> VelocityProfile:
        print("   Solving Grip-Limited Velocity (Apex Solver)...")
        
        # track data
        s = self.track.track_data.s
        ds = self.track.ds
        radii = self.track.track_data.radius
        gradients = self.track.track_data.gradient
        N = len(s)
        
        # calculation of apex speeds (friction limit)
        # limit the "ceiling" of velocity at every point
        v_limit = np.zeros(N)
        for i in range(N):
            # iterative solver already in VehicleConfig
            v_limit[i] = self.vehicle.get_max_cornering_speed(radii[i])
            
        # forward integration (max acceleration)
        v_fwd = np.zeros(N)
        v_fwd[0] = v_limit[0] # limit start
        
        for i in range(N - 1):
            # max possible acceleration at current speed (assuming full power)
            # assuming SOC is full for to find the physical limit
            ax_max = self._get_max_accel(v_fwd[i], gradients[i], radii[i])
            
            # Kinematics: v_next = sqrt(v^2 + 2*a*d)
            next_sq = v_fwd[i]**2 + 2 * ax_max * ds
            v_next = np.sqrt(max(0, next_sq))
            
            # Clamp to corner limit
            v_fwd[i+1] = min(v_next, v_limit[i+1])
            
        # backward integration (max braking)
        v_bwd = np.zeros(N)
        v_bwd[-1] = v_limit[-1] # limit end
        
        for i in range(N - 1, 0, -1):
            # max possible deceleration
            ax_min = self._get_max_brake(v_bwd[i], gradients[i], radii[i]) 
            # Note: ax_min should be negative
            
            # kinematics backwards: v_prev = sqrt(v^2 - 2*a*d) -> minus because going back
            prev_sq = v_bwd[i]**2 - 2 * ax_min * ds
            v_prev = np.sqrt(max(0, prev_sq))
            
            # clamp
            v_bwd[i-1] = min(v_prev, v_limit[i-1])
            
        # final profile is the minimum of both
        v_final = np.minimum(v_fwd, v_bwd)
        
        # time
        t = np.zeros(N)
        for i in range(1, N):
            v_avg = 0.5 * (v_final[i] + v_final[i-1])
            t[i] = t[i-1] + ds / max(v_avg, 1.0)
            
        print(f"   ✓ Theoretical Lap Time: {t[-1]:.3f}s")
        
        return VelocityProfile(s, v_final, np.zeros(N), t, t[-1])

    def _get_max_accel(self, v, grad, radius):
        # simplified helper: full power (ICE + ERS)
        forces = self.vehicle.get_aero_forces(v)
        drag = forces['drag']
        
        # max power force (P = F*v -> F = P/v)
        P_total = self.vehicle.max_total_power
        F_prop = P_total / max(v, 5.0)
        
        # grip limit
        normal = self.vehicle.mass * 9.81 * np.cos(grad) + forces['downforce']
        F_tire_max = self.vehicle.mu_longitudinal * normal
        F_prop = min(F_prop, F_tire_max)
        
        m = self.vehicle.mass
        F_grav = m * 9.81 * np.sin(grad)
        
        return (F_prop - drag - F_grav) / m

    def _get_max_brake(self, v, grad, radius):
        # simplified helper: max braking
        forces = self.vehicle.get_aero_forces(v)
        drag = forces['drag']
        
        normal = self.vehicle.mass * 9.81 * np.cos(grad) + forces['downforce']
        F_brake_max = self.vehicle.mu_longitudinal * normal
        
        m = self.vehicle.mass
        F_grav = m * 9.81 * np.sin(grad)
        
        # deceleration is negative
        return (-F_brake_max - drag - F_grav) / m