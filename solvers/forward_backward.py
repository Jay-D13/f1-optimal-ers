"""
Forward-backward velocity solver
computes the grip limited velocity profile WITHOUT ERS optimization.

Based on TUMFTM's laptime simulation approach.
"""
import numpy as np
from dataclasses import dataclass
from typing import Optional

from solvers import VelocityProfileSolver, VelocityProfile
from models import VehicleDynamicsModel
from models import F1TrackModel


class ForwardBackwardSolver(VelocityProfileSolver):
    """
    Forward-backward solver for grip-limited velocity profile
    
    Algorithm:
    1. Compute apex (maximum cornering) speeds at each point
    2. Forward pass: Accelerate from each point respecting grip
    3. Backward pass: Decelerate to each point respecting grip  
    4. Final profile = min(forward, backward) at each point
    """
   
    def __init__(self, 
                vehicle_model : VehicleDynamicsModel,
                track_model: F1TrackModel,
                use_ers_power: bool = False
            ):
        
        super().__init__(vehicle_model, track_model)
        self.use_ers_power = use_ers_power
        
    def solve(self) -> VelocityProfile:
        """Solve for grip-limited velocity profile"""
        print("==== Solving grip-limited velocity profile...====")
        
        track_data = self.track.track_data
        s = track_data.s
        ds = self.track.ds
        N = len(s)
        
        radii = track_data.radius
        gradients = track_data.gradient
        
        veh = self.vehicle.vehicle
        
        # apex speeds (cornering limits)
        v_apex = np.zeros(N)
        for i in range(N):
            v_apex[i] = self._compute_cornering_speed(
                radius=radii[i],
                velocity_guess=50.0,  # TODO Initial guess because depends on if continuing lap or starting lap
                gradient=gradients[i]
            )
        
        print(f"   Apex speeds: {v_apex.min():.1f} - {v_apex.max():.1f} m/s")
        
        # forward pass (maximum acceleration)
        v_fwd = np.zeros(N)
        v_fwd[0] = min(v_apex[0], 50.0)
        
        for i in range(N - 1):
            # max acceleration at current state
            a_x_max = self._get_max_acceleration(
                v=v_fwd[i],
                radius=radii[i],
                gradient=gradients[i]
            )
            
            # Kinematics: v² = v₀² + 2·a·Δs
            v_next_sq = v_fwd[i]**2 + 2 * a_x_max * ds
            v_next = np.sqrt(max(0, v_next_sq))
            
            # Clamp to apex speed
            v_fwd[i+1] = min(v_next, v_apex[i+1])
        
        # backward pass (maximum braking)
        v_bwd = np.zeros(N)
        v_bwd[-1] = min(v_apex[-1], v_apex[0])  # For lap continuity
        
        for i in range(N - 1, 0, -1):
            # max deceleration (negative)
            a_x_min = self._get_max_deceleration(
                v=v_bwd[i],
                radius=radii[i],
                gradient=gradients[i]
            )
            
            # Kinematics backwards: v_prev² = v² - 2·a·Δs
            # Since a_x_min is negative: v_prev² = v² + 2·|a|·Δs
            v_prev_sq = v_bwd[i]**2 - 2 * a_x_min * ds
            v_prev = np.sqrt(max(0, v_prev_sq))
            
            # Clamp to apex speed
            v_bwd[i-1] = min(v_prev, v_apex[i-1])
        
        # final profile is minimum of both
        v_final = np.minimum(v_fwd, v_bwd)
        
        # Compute acceleration profile
        a_x = np.zeros(N)
        for i in range(N - 1):
            a_x[i] = (v_final[i+1]**2 - v_final[i]**2) / (2 * ds)
        a_x[-1] = a_x[-2]  # Extend last value
        
        # Compute time
        t = np.zeros(N)
        for i in range(1, N):
            v_avg = 0.5 * (v_final[i] + v_final[i-1])
            t[i] = t[i-1] + ds / max(v_avg, 1.0)
        
        lap_time = t[-1]
        
        print(f"   ✓ Theoretical lap time: {lap_time:.3f}s")
        print(f"   Velocity range: {v_final.min():.1f} - {v_final.max():.1f} m/s")
        print(f"   Acceleration range: {a_x.min():.1f} - {a_x.max():.1f} m/s²")
        
        return VelocityProfile(
            s=s,
            v=v_final,
            a_x=a_x,
            t=t,
            lap_time=lap_time
        )
    
    def _compute_cornering_speed(self, radius: float, velocity_guess: float, 
                                  gradient: float = 0.0) -> float:
        """
        Maximum cornering speed iteratively (downforce depends on speed)
        a_lat = v²/R ≤ μ·(m·g + F_downforce) / m
        """
        veh = self.vehicle.vehicle
        
        # Handle straights
        if radius > 2000 or np.isinf(radius):
            return 100.0  # Max straight-line speed
        
        radius = max(radius, 10.0)  # Minimum radius
        
        # Iterative solution (downforce depends on speed)
        v = velocity_guess
        for _ in range(10):
            # Aerodynamic forces
            aero = veh.get_aero_forces(v)
            downforce = aero['downforce']
            
            # Normal force (weight + downforce)
            F_normal = veh.mass * veh.g * np.cos(gradient) + downforce
            
            # Maximum lateral force (friction)
            F_lat_max = veh.mu_lateral * F_normal
            
            # Required lateral force for cornering
            # F_lat = m · v² / R
            # v_max = sqrt(F_lat_max · R / m)
            v_new = np.sqrt(F_lat_max * radius / veh.mass)
            
            if abs(v_new - v) < 0.1:
                break
            v = 0.7 * v + 0.3 * v_new  # Damped update
        
        return min(v, 100.0)
    
    def _get_max_acceleration(self, v: float, radius: float, gradient: float) -> float:
        """
        Maximum longitudinal acceleration
        
        Limited by:
        1. Available power (ICE + optional ERS)
        2. Traction (friction circle minus lateral force)
        """
        veh = self.vehicle.vehicle
        
        # Aerodynamic forces
        aero = veh.get_aero_forces(v)
        drag = aero['drag']
        downforce = aero['downforce']
        
        # Normal force
        F_normal = veh.mass * veh.g * np.cos(gradient) + downforce
        
        # Rolling resistance
        F_roll = veh.cr * F_normal
        
        # Gravity component
        F_gravity = veh.mass * veh.g * np.sin(gradient)
        
        # === Power limit ===
        if self.use_ers_power:
            P_max = veh.pow_max_total  # ICE + ERS
        else:
            P_max = veh.pow_max_ice    # ICE only
        
        F_power = P_max / max(v, 5.0)
        
        # === Traction limit (friction circle) ===
        # Lateral force from cornering
        radius = max(radius, 50.0)  # Avoid division issues on straights
        a_lat = v**2 / radius if radius < 2000 else 0
        F_lat = veh.mass * a_lat
        
        # Total grip available
        F_grip_total = veh.mu_longitudinal * F_normal
        
        # Remaining grip for longitudinal (friction circle)
        F_grip_remaining_sq = F_grip_total**2 - F_lat**2
        F_grip_remaining = np.sqrt(max(F_grip_remaining_sq, 100))  # Min 10N
        
        # Propulsive force is minimum of power and grip limits
        F_prop = min(F_power, F_grip_remaining)
        
        # Net acceleration
        F_net = F_prop - drag - F_roll - F_gravity
        a_x = F_net / veh.mass
        
        return max(a_x, 0)  # Can't have negative acceleration in forward pass
    
    def _get_max_deceleration(self, v: float, radius: float, gradient: float) -> float:
        """
        Maximum deceleration (braking) (negative acceleration)
        """
        veh = self.vehicle.vehicle
        
        # Aerodynamic forces
        aero = veh.get_aero_forces(v)
        drag = aero['drag']
        downforce = aero['downforce']
        
        # Normal force
        F_normal = veh.mass * veh.g * np.cos(gradient) + downforce
        
        # Rolling resistance
        F_roll = veh.cr * F_normal
        
        # Gravity component
        F_gravity = veh.mass * veh.g * np.sin(gradient)
        
        # === Braking limit (friction circle) ===
        radius = max(radius, 50.0)
        a_lat = v**2 / radius if radius < 2000 else 0
        F_lat = veh.mass * a_lat
        
        # Total grip available
        F_grip_total = veh.mu_longitudinal * F_normal
        
        # Remaining grip for braking
        F_grip_remaining_sq = F_grip_total**2 - F_lat**2
        F_grip_remaining = np.sqrt(max(F_grip_remaining_sq, 100))
        
        # Maximum braking force
        F_brake_max = min(F_grip_remaining, veh.max_brake_force)
        
        # Net deceleration (negative)
        # Note: drag helps braking, gravity depends on slope
        F_net = -F_brake_max - drag - F_roll - F_gravity
        a_x = F_net / veh.mass
        
        return min(a_x, 0) # should be negative


# class ForwardBackwardSolverTUMFTM(ForwardBackwardSolver):
#     """
#     Solver like TUMFTM tire model (has tire load dependent friction)
#     """
    
#     def __init__(self, vehicle_model, track_model, tire_params: Optional[dict] = None):
#         super().__init__(vehicle_model, track_model, use_ers_power=True)
        
#         # TUMFTM tire parameters (from F1_Shanghai.ini)
#         self.tire_params = tire_params or {
#             'fz_0': 3000.0,          # Nominal tire load (N)
#             'mux_f': 1.65,           # Front longitudinal friction at fz_0
#             'mux_r': 1.95,           # Rear longitudinal friction at fz_0
#             'muy_f': 1.85,           # Front lateral friction at fz_0
#             'muy_r': 2.15,           # Rear lateral friction at fz_0
#             'dmux_dfz': -5.0e-5,     # Friction reduction with load
#             'dmuy_dfz': -5.0e-5,
#         }
    
#     def _compute_tire_force_potential(self, fz: float, mu_0: float, dmu_dfz: float) -> float:
#         """
#         Compute tire force potential with load-dependent friction.
        
#         F = μ(F_z) · F_z
#         μ(F_z) = μ_0 + dμ/dF_z · (F_z - F_z0)
        
#         This models the reduction in friction coefficient at higher loads.
#         """
#         fz_0 = self.tire_params['fz_0']
#         mu = mu_0 + dmu_dfz * (fz - fz_0)
#         return mu * fz
    
    # TODO: Override _get_max_acceleration and _get_max_deceleration
    # with 4-wheel load transfer calculations
