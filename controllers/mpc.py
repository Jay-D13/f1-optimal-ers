import numpy as np
import casadi as ca
from typing import Dict, Tuple, Optional
import time

from models import VehicleDynamicsModel, F1TrackModel
from .offline_optimizer import OptimalTrajectory


class OnlineMPController:
    """
    Online MPC controller that tracks the globally optimal trajectory.
    
    The MPC minimizes:
        J = Σ[ w_v*(v - v_ref)² + w_soc*(soc - soc_ref)² + w_u*Δu² ]
    
    Subject to:
        - Vehicle dynamics
        - State/control constraints
        - Friction circle (lateral acceleration limit)
    """
    
    def __init__(self,
                 vehicle_model: VehicleDynamicsModel,
                 track_model: F1TrackModel,
                 horizon_distance: float = 200.0,
                 dt: float = 0.1):
        """
        Initialize the online MPC controller.
        
        Args:
            vehicle_model: Vehicle dynamics model
            track_model: Track model with geometry data
            horizon_distance: Prediction horizon in meters
            dt: Time discretization step (seconds)
        """
        self.vehicle = vehicle_model
        self.track = track_model
        self.horizon_distance = horizon_distance
        self.dt = dt
        
        # Reference trajectory (set by set_reference())
        self.reference: Optional[OptimalTrajectory] = None
        
        # MPC horizon adapts based on velocity
        self.N_max = 50  # Maximum horizon steps
        self.N_min = 10  # Minimum horizon steps
        
        # CasADi problem (rebuilt for each solve for flexibility)
        self.opti = None
        self.solution = None
        self.fallback_counter = 0 # Track how many times we've fallen back in a row
        
        # Warm start data
        self.prev_x_opt = None
        self.prev_u_opt = None
        
        # Performance tracking
        self.solve_count = 0
        self.fail_count = 0
        self.total_solve_time = 0.0
        
        # Weights (tuning parameters)
        self.w_v = 1.0          # Velocity tracking weight
        self.w_soc = 50.0       # SOC tracking weight
        self.w_P_ers = 0.001    # ERS power smoothness
        self.w_throttle = 0.01  # Throttle smoothness
        self.w_brake = 0.01     # Brake smoothness
        self.w_terminal = 10.0  # Terminal cost weight
        
        print(f"   Online MPC initialized:")
        print(f"     Horizon: {horizon_distance} m")
        print(f"     Time step: {dt} s")
    
    def set_reference(self, reference: OptimalTrajectory) -> None:
        """Set the reference trajectory from global optimizer"""
        self.reference = reference
        print(f"   Reference trajectory loaded: {reference.lap_time:.2f}s lap")
    
    def solve_mpc_step(self,
                       current_state: np.ndarray,
                       track_position: float,
                       soc_reference_trajectory: np.ndarray = None
                       ) -> Tuple[np.ndarray, Dict]:
        """
        Solve one MPC iteration.
        
        Args:
            current_state: Current state [position, velocity, soc]
            track_position: Current distance along track (m)
            soc_reference_trajectory: Optional explicit SOC reference
        
        Returns:
            control: Optimal control [P_ers, throttle, brake]
            info: Dictionary with solve information
        """
        
        start_time = time.time()
        
        # Get current velocity to determine horizon
        v_current = max(current_state[1], 20.0)
        
        # Adaptive horizon based on velocity
        # At high speed, we need more look-ahead distance
        N = int(self.horizon_distance / (v_current * self.dt))
        N = max(self.N_min, min(N, self.N_max))
        
        # Build and solve MPC
        try:
            control, info = self._solve_mpc(
                current_state, track_position, N, 
                soc_reference_trajectory
            )
            self.solve_count += 1
            self.fallback_counter = 0
            
        except Exception as e:
            # Fallback to simple rule-based control
            self.fallback_counter += 1
            control, fallback_type = self._fallback_control(current_state, track_position)
            
            info = {
                'success': False,
                'error': str(e),
                'fallback': True,
                'fallback_type': fallback_type,
                'predicted_states': None,
                'predicted_controls': None
            }
            self.fail_count += 1
        
        solve_time = time.time() - start_time
        self.total_solve_time += solve_time
        info['solve_time'] = solve_time
        
        return control, info
    
    def _solve_mpc(self,
                   current_state: np.ndarray,
                   track_position: float,
                   N: int,
                   soc_ref_explicit: Optional[np.ndarray] = None
                   ) -> Tuple[np.ndarray, Dict]:
        """Internal MPC solve"""
        
        opti = ca.Opti()
        constraints = self.vehicle.get_constraints()
        dynamics = self.vehicle.create_time_domain_dynamics()
        
        # === Decision Variables ===
        X = opti.variable(3, N + 1)  # [s, v, soc]
        U = opti.variable(3, N)      # [P_ers, throttle, brake]
        
        # === Parameters ===
        x0 = opti.parameter(3)
        v_ref = opti.parameter(N + 1)
        soc_ref = opti.parameter(N + 1)
        track_params = opti.parameter(2, N)  # [gradient, radius]
        
        # === Objective Function ===
        obj = 0
        for k in range(N):
            # Tracking errors
            v_error = X[1, k] - v_ref[k]
            soc_error = X[2, k] - soc_ref[k]
            
            obj += self.w_v * v_error**2
            obj += self.w_soc * soc_error**2
            
            # Control smoothness
            if k > 0:
                obj += self.w_P_ers * (U[0, k] - U[0, k-1])**2
                obj += self.w_throttle * (U[1, k] - U[1, k-1])**2
                obj += self.w_brake * (U[2, k] - U[2, k-1])**2
        
        # Terminal cost
        obj += self.w_terminal * (X[1, N] - v_ref[N])**2
        obj += self.w_terminal * (X[2, N] - soc_ref[N])**2
        
        # prevents the solver from crashing if we are slightly over speed limit
        slacks = opti.variable(N + 1)
        obj += 1e2 * ca.sumsqr(slacks) # high penalty
        
        opti.minimize(obj)
        
        SLACK_DYN = opti.variable(3, N)
        opti.minimize(obj + 1e4 * ca.sumsqr(SLACK_DYN))
        
        # === Dynamics Constraints (RK4) ===
        for k in range(N):
            x_k = X[:, k]
            u_k = U[:, k]
            p_k = track_params[:, k]
            
            # RK4 integration
            k1 = dynamics(x_k, u_k, p_k)
            k2 = dynamics(x_k + self.dt/2 * k1, u_k, p_k)
            k3 = dynamics(x_k + self.dt/2 * k2, u_k, p_k)
            k4 = dynamics(x_k + self.dt * k3, u_k, p_k)
            x_next = x_k + self.dt/6 * (k1 + 2*k2 + 2*k3 + k4)
            
            # opti.subject_to(X[:, k+1] == x_next)
            opti.subject_to(X[:, k+1] == x_next + SLACK_DYN[:, k])
            
            # Control bounds
            opti.subject_to(opti.bounded(constraints['P_ers_min'], U[0, k], constraints['P_ers_max']))
            opti.subject_to(opti.bounded(0, U[1, k], 1))
            opti.subject_to(opti.bounded(0, U[2, k], 1))
            
            # Friction Circle / Speed Limit (Soft Constraint)
            # radius_k = track_params[1, k]
            # v_max_corner = ca.sqrt(1.8 * 9.81 * ca.fmin(radius_k, 2000))
            
            # # Relaxed constraint: v <= v_max + slack
            # opti.subject_to(X[1, k] <= ca.fmin(v_max_corner, constraints['v_max']) + slacks[k])
            opti.subject_to(slacks[k] >= 0)
        
        # === State Constraints ===
        for k in range(N + 1):
            opti.subject_to(X[1, k] >= constraints['v_min'])
            opti.subject_to(X[2, k] >= constraints['soc_min'])
            opti.subject_to(X[2, k] <= constraints['soc_max'])
            
            # Speed limit from track (using reference as proxy)
            # opti.subject_to(X[1, k] <= v_ref[k] * 1.1 + 5)  # Allow 10% overspeed margin
        
        # === Control Constraints ===
        # for k in range(N):
        #     opti.subject_to(opti.bounded(
        #         constraints['P_ers_min'], U[0, k], constraints['P_ers_max']
        #     ))
        #     opti.subject_to(opti.bounded(0, U[1, k], 1))
        #     opti.subject_to(opti.bounded(0, U[2, k], 1))
            
        #     # Friction circle approximation via speed limit
        #     # (more sophisticated version would use explicit lateral force)
        #     radius_k = track_params[1, k]
        #     v_max_corner = ca.sqrt(1.8 * 9.81 * ca.fmin(radius_k, 2000))
            
        #     slacks = opti.variable(N + 1)
        #     opti.minimize(obj + 100000 * ca.sumsqr(slacks)) # Huge penalty

        #     opti.subject_to(X[1, k] <= ca.fmin(v_max_corner, constraints['v_max']) + slacks[k])
        #     opti.subject_to(slacks[k] >= 0)
        
        # === Initial Condition ===
        opti.subject_to(X[:, 0] == x0)
        
        # === Solver Options ===
        opts = {
            'ipopt.max_iter': 100,
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.linear_solver': 'ma97', # "mumps" if no license
            'ipopt.tol': 1e-3,
            'ipopt.warm_start_init_point': 'yes',
        }
        opti.solver('ipopt', opts)
        
        # === Set Parameter Values ===
        opti.set_value(x0, current_state)
        
        # Get track preview
        track_preview = self._get_track_preview(track_position, current_state[1], N)
        opti.set_value(track_params, track_preview)
        
        # Get reference trajectory
        if soc_ref_explicit is not None:
             opti.set_value(soc_ref, soc_ref_explicit[:N+1])
        else:
             opti.set_value(soc_ref, self._get_soc_reference(track_position, current_state[1], N))
        
        v_reference = self._get_velocity_reference(track_position, current_state[1], N)
        
        opti.set_value(v_ref, v_reference)
        # opti.set_value(soc_ref, soc_reference)
        
        # === Warm Start ===
        if self.prev_x_opt is not None and self.prev_x_opt.shape[1] >= N + 1:
            try:
                # Shift previous solution
                x_warm = np.hstack([
                    self.prev_x_opt[:, 1:N+1], 
                    self.prev_x_opt[:, -1:]
                ])
                u_warm = np.hstack([
                    self.prev_u_opt[:, 1:N], 
                    self.prev_u_opt[:, -1:]
                ])
                
                if x_warm.shape[1] == N + 1 and u_warm.shape[1] == N:
                    opti.set_initial(X, x_warm)
                    opti.set_initial(U, u_warm)
            except:
                pass
        
        # === Solve ===
        sol = opti.solve()
        
        # Extract solution
        x_opt = sol.value(X)
        u_opt = sol.value(U)
        
        # Store for warm start
        self.prev_x_opt = x_opt
        self.prev_u_opt = u_opt
        
        info = {
            'success': True,
            'predicted_states': x_opt,
            'predicted_controls': u_opt,
            'horizon': N,
        }
        
        return u_opt[:, 0], info
    
    def _get_track_preview(self, current_pos: float, current_vel: float, 
                           N: int) -> np.ndarray:
        """Get track parameters for prediction horizon"""
        
        track_data = np.zeros((2, N))
        
        for k in range(N):
            # Estimate future position
            future_pos = current_pos + k * self.dt * current_vel
            future_pos = future_pos % self.track.total_length
            
            segment = self.track.get_segment_at_distance(future_pos)
            
            track_data[0, k] = segment.gradient
            track_data[1, k] = min(segment.radius, 5000)  # Cap for numerics
        
        return track_data
    
    def _get_velocity_reference(self, current_pos: float, current_vel: float,
                                 N: int) -> np.ndarray:
        """Get velocity reference from global optimizer"""
        
        if self.reference is None:
            # Fallback: use track speed limits
            v_ref = np.ones(N + 1) * current_vel
            return v_ref
        
        v_ref = np.zeros(N + 1)
        
        for k in range(N + 1):
            future_pos = current_pos + k * self.dt * current_vel
            ref = self.reference.get_reference_at_distance(future_pos)
            v_ref[k] = ref['v_ref']
        
        return v_ref
    
    def _get_soc_reference(self, current_pos: float, current_vel: float,
                            N: int) -> np.ndarray:
        """Get SOC reference from global optimizer"""
        
        if self.reference is None:
            # Fallback: target 50% SOC
            return np.ones(N + 1) * 0.5
        
        soc_ref = np.zeros(N + 1)
        
        for k in range(N + 1):
            future_pos = current_pos + k * self.dt * current_vel
            ref = self.reference.get_reference_at_distance(future_pos)
            soc_ref[k] = ref['soc_ref']
        
        return soc_ref
    
    def _fallback_control(self, state: np.ndarray, position: float) -> np.ndarray:
        """
        Two-stage fallback strategy:
        1. Shifted Horizon: Use the valid plan from the previous step.
        2. PID Tracker: If no previous plan, track reference velocity with simple logic.
        """
        
        # SHIFTED HORIZON (The "Ghost" Solution)
        # If we solved successfully recently, the previous plan is still valid, just shifted by one dt.
        if self.prev_u_opt is not None and self.fallback_counter < self.prev_u_opt.shape[1]:
            # We index into the previous solution based on how many times we've failed
            # If fail_count is 1, we take index 1 (the second step of previous plan)
            idx = self.fallback_counter
            
            if idx < self.prev_u_opt.shape[1]:
                u_fallback = self.prev_u_opt[:, idx]
                return u_fallback, "shifted_horizon"

        # PID REFERENCE TRACKER (Emergency Mode)
        # If we have no history or have run out of buffer, use simple physics
        
        # Get target velocity from offline reference
        ref = self.reference.get_reference_at_distance(position)
        v_target = ref['v_ref']
        v_current = state[1]
        
        # Look ahead slightly (0.5s) to catch braking zones early
        lookahead_pos = position + v_current * 0.5
        ref_ahead = self.reference.get_reference_at_distance(lookahead_pos)
        v_target = min(v_target, ref_ahead['v_ref'])
        
        # P-Controller
        error = v_target - v_current
        Kp_accel = 0.5
        Kp_brake = 0.2
        
        throttle = 0.0
        brake = 0.0
        P_ers = 0.0
        
        if error > 0:
            # We are too slow -> Accelerate
            throttle = np.clip(Kp_accel * error, 0.0, 1.0)
            
            # Use ERS if throttle is high and we have SOC
            if throttle > 0.9 and state[2] > 0.3:
                P_ers = 120000 # Max deployment
        else:
            # We are too fast -> Brake
            # Note: error is negative here
            brake = np.clip(-Kp_brake * error, 0.0, 1.0)
            
            # Use ERS to help brake (Harvest)
            if state[2] < 0.9:
                P_ers = -120000 # Max harvest
        
        u_fallback = np.array([P_ers, throttle, brake])
        return u_fallback, "pid_tracker"
    
    def get_statistics(self) -> Dict:
        """Get performance statistics"""
        return {
            'solve_count': self.solve_count,
            'fail_count': self.fail_count,
            'success_rate': (self.solve_count / max(self.solve_count + self.fail_count, 1)) * 100,
            'avg_solve_time': self.total_solve_time / max(self.solve_count, 1),
        }