"""
Forward-backward velocity solver
computes the grip limited velocity profile WITHOUT ERS optimization.

Based on TUMFTM's laptime simulation approach.
"""
import numpy as np

from solvers import VelocityProfileSolver, VelocityProfile
from models import VehicleDynamicsModel
from models import F1TrackModel


class ForwardBackwardSolver(VelocityProfileSolver):
    """
    Forward-backward solver for grip-limited velocity profile
    """
   
    def __init__(self, 
                vehicle_model : VehicleDynamicsModel,
                track_model: F1TrackModel,
                use_ers_power: bool = False
            ):
        super().__init__(vehicle_model, track_model)
        self.use_ers_power = use_ers_power
        
    def solve(self, flying_lap: bool = False) -> VelocityProfile:
        """
        Solve for grip-limited velocity profile.
        
        Args:
            flying_lap: If True, simulates 2 laps to ensure V_start == V_end
                        (Cyclic boundary condition)
        """
        if flying_lap:
            return self._solve_flying()
        
        print("==== Solving grip-limited velocity profile (Standing Start)...====")
        return self._solve_core(self.track.track_data)

    def _solve_flying(self) -> VelocityProfile:
        """
        Simulate a flying lap by doubling the track data (2 laps),
        solving, and extracting the second lap.
        """
        print("==== Solving grip-limited velocity profile (Flying Lap)...====")
        
        # create a double rrack (lap 1 + lap 2)
        # concatenate the arrays to simulate two consecutive laps
        original_data = self.track.track_data
        
        # duble the arrays
        s_double = np.concatenate([original_data.s, original_data.s + original_data.total_length])
        radius_double = np.concatenate([original_data.radius, original_data.radius])
        grad_double = np.concatenate([original_data.gradient, original_data.gradient])
        
        #temporary data structure for solver
        class DoubleTrackData:
            s = s_double
            radius = radius_double
            gradient = grad_double
            ds = original_data.ds
            
        # solver on the double track
        # first lap acts as a "run-up" to get the correct entry speed for the second lap
        full_profile = self._solve_core(DoubleTrackData)
        
        # extract the second lap (the flying lap)
        N = len(original_data.s)
        
        # indice for the second lap
        v_flying = full_profile.v[N:]
        a_flying = full_profile.a_x[N:]
        
        # time for just one lap starting at t=0
        t_flying = np.zeros(N)
        ds = self.track.ds
        for i in range(1, N):
            v_avg = 0.5 * (v_flying[i] + v_flying[i-1])
            t_flying[i] = t_flying[i-1] + ds / max(v_avg, 1.0)
            
        print(f"   ✓ Flying lap time: {t_flying[-1]:.3f}s")
        print(f"   Entry Speed: {v_flying[0]*3.6:.1f} km/h")
        
        return VelocityProfile(
            s=original_data.s,
            v=v_flying,
            a_x=a_flying,
            t=t_flying,
            lap_time=t_flying[-1]
        )

    def _solve_core(self, track_data) -> VelocityProfile:
        """Core logic for forward-backward integration"""
        s = track_data.s
        ds = track_data.ds
        N = len(s)
        
        radii = track_data.radius
        gradients = track_data.gradient
        
        # apex speeds (cornering limits)
        v_apex = np.zeros(N)
        for i in range(N):
            v_apex[i] = self._compute_cornering_speed(
                radius=radii[i],
                velocity_guess=50.0, # TODO take from FastF1 real data
                gradient=gradients[i]
            )
        
        print(f"   Apex speeds: {v_apex.min():.1f} - {v_apex.max():.1f} m/s")
        
        # forward pass (maximum acceleration)
        v_fwd = np.zeros(N)
        v_fwd[0] = v_apex[0] # Optimistic start, will be clamped by backward pass if needed
        
        for i in range(N - 1):
            a_x_max = self._get_max_acceleration(v_fwd[i], radii[i], gradients[i])
            v_next_sq = v_fwd[i]**2 + 2 * a_x_max * ds
            v_fwd[i+1] = min(np.sqrt(max(0, v_next_sq)), v_apex[i+1])
        
        # backward pass (maximum braking)
        v_bwd = np.zeros(N)
        v_bwd[-1] = v_apex[-1] # End at limit
        
        for i in range(N - 1, 0, -1):
            a_x_min = self._get_max_deceleration(v_bwd[i], radii[i], gradients[i])
            # Reverse kinematics: v_prev² = v² - 2*a*ds (a is negative)
            v_prev_sq = v_bwd[i]**2 - 2 * a_x_min * ds 
            v_bwd[i-1] = min(np.sqrt(max(0, v_prev_sq)), v_apex[i-1])
        
        # final profile is minimum of both
        v_final = np.minimum(v_fwd, v_bwd)
        
        # Compute acceleration profile
        a_x = np.zeros(N)
        for i in range(N - 1):
            a_x[i] = (v_final[i+1]**2 - v_final[i]**2) / (2 * ds)
        
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
        
        # Physical speed cap
        V_MAX_PHYSICAL = 110.0  # m/s
        
        # Handle straights
        if radius > 1000 or np.isinf(radius):
            return V_MAX_PHYSICAL
        
        radius = max(radius, 10.0)  # Minimum radius
        
        # Use constant lateral friction (consistent with NLP solver)
        mu_lat = veh.mu_lateral  # 2.0
        
        # Iterative solution (downforce depends on speed)
        v = velocity_guess
        for _ in range(15):
            v = min(v, V_MAX_PHYSICAL)
            
            # Aerodynamic forces
            aero = veh.get_aero_forces(v)
            downforce = aero['downforce']
            
            # Normal force (weight + downforce)
            F_normal = veh.mass * veh.g * np.cos(gradient) + downforce
            
            # Maximum lateral force (constant friction)
            F_lat_max = mu_lat * F_normal
            
            # Required lateral force for cornering
            # F_lat = m · v² / R
            # v_max = sqrt(F_lat_max · R / m)
            v_new = np.sqrt(F_lat_max * radius / veh.mass)
            v_new = min(v_new, V_MAX_PHYSICAL)
            
            if abs(v_new - v) < 0.1:
                break
            v = 0.7 * v + 0.3 * v_new  # Damped update
        
        return min(v, V_MAX_PHYSICAL)
    
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