from cmath import sqrt
import numpy as np
import casadi as ca
from typing import Tuple, Dict

from models import VehicleDynamicsModel, F1TrackModel


class ERSOptimalController:
    """MPC-based optimal controller for ERS deployment"""
    
    def __init__(self, vehicle_model: VehicleDynamicsModel, 
                 track_model: F1TrackModel,
                 horizon_time: float = 3.0,
                 dt: float = 0.1):
        self.vehicle_model = vehicle_model
        self.track_model = track_model
        self.horizon_time = horizon_time
        self.dt = dt
        self.N = int(horizon_time / dt)  # Prediction horizon steps
        
        self.opti = None
        self.solution = None
        self.dynamics_func = None
        self.failed_solves = 0  # Track consecutive failures
        
    def setup_optimization(self):
        """Setup the CasADi optimization problem with Lateral G constraints"""
        self.opti = ca.Opti()
        
        # 1. Get dynamics and basic constraints
        self.dynamics_func = self.vehicle_model.create_casadi_function()
        constraints = self.vehicle_model.get_constraints()
        
        # 2. Decision variables
        X = self.opti.variable(3, self.N+1)  # States: [pos, vel, soc]
        U = self.opti.variable(3, self.N)    # Controls: [P_ers, throttle, brake]
        
        # 3. Parameters
        x0 = self.opti.parameter(3)  # Initial state
        track_params = self.opti.parameter(2, self.N)  # [gradient, radius]
        
        # 4. Objective Function Setup
        obj = 0
        
        # Constants for weights (tuning knobs)
        W_progress = 10.0      # Weight for maximizing velocity
        W_energy_diff = 50.0   # Weight for keeping SOC near target
        W_smooth = 0.1         # Weight for smooth control changes
        
        # Lateral physics constants
        mu_lat = 1.8  # Max lateral friction coefficient (Dry F1 tires)
        g = 9.81
        
        for k in range(self.N):
            # --- Cost: Maximize Velocity (Progress) ---
            obj -= W_progress * X[1, k]
            
            # --- Cost: Energy Management ---
            # Instead of a hard target of 0.5, we penalize being empty
            # and gently encourage staying in the middle.
            # This allows the car to use energy when needed.
            soc_deviation = (X[2, k] - 0.5)
            obj += W_energy_diff * soc_deviation**2
            
            # --- Cost: Smoothness ---
            if k > 0:
                # Penalize rapid changes in ERS and Throttle
                obj += W_smooth * (U[0, k] - U[0, k-1])**2 
                obj += W_smooth * (U[1, k] - U[1, k-1])**2
            
            # --- Constraint: The "Friction Circle" (The Fix) ---
            # Calculate max speed allowed by corner radius: v_max = sqrt(mu * g * R)
            
            # 1. Get radius, ensuring we handle straight lines (Inf) safely
            # We treat any radius > 2000m as effectively straight for grip purposes
            r_k = track_params[1, k]
            safe_r = ca.fmin(r_k, 2000.0) 
            
            # 2. Calculate the physical speed limit for this segment
            # v^2 / R <= mu * g  ->  v <= sqrt(mu * g * R)
            v_max_corner = ca.sqrt(mu_lat * g * safe_r)
            
            # 3. Apply as a constraint
            # We use fmin so the limit is never higher than the car's mechanical max speed
            limit = ca.fmin(v_max_corner, constraints['v_max'])
            self.opti.subject_to(X[1, k] <= limit)
            
            # --- Dynamics Integration (RK4) ---
            x_k = X[:, k]
            u_k = U[:, k]
            p_k = track_params[:, k]
            
            k1 = self.dynamics_func(x_k, u_k, p_k)
            k2 = self.dynamics_func(x_k + self.dt/2 * k1, u_k, p_k)
            k3 = self.dynamics_func(x_k + self.dt/2 * k2, u_k, p_k)
            k4 = self.dynamics_func(x_k + self.dt * k3, u_k, p_k)
            x_next = x_k + self.dt/6 * (k1 + 2*k2 + 2*k3 + k4)
            
            self.opti.subject_to(X[:, k+1] == x_next)
            
            # --- Actuator Constraints ---
            self.opti.subject_to(self.opti.bounded(constraints['P_ers_min'], U[0, k], constraints['P_ers_max']))
            self.opti.subject_to(self.opti.bounded(constraints['throttle_min'], U[1, k], constraints['throttle_max']))
            self.opti.subject_to(self.opti.bounded(constraints['brake_min'], U[2, k], constraints['brake_max']))
            
            # No simultaneous throttle and brake
            self.opti.subject_to(U[1, k] * U[2, k] <= 1e-4)
            
        # 5. Boundary Conditions
        self.opti.subject_to(X[:, 0] == x0) # Initial state
        
        # Terminal Constraints (Safety)
        # Ensure we don't end the horizon with an empty battery
        self.opti.subject_to(X[2, -1] >= 0.2) 
        
        self.opti.minimize(obj)
        
        # 6. Solver Settings
        opts = {
            'ipopt.max_iter': 500,
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.acceptable_tol': 1e-4,
            'ipopt.acceptable_obj_change_tol': 1e-4,
            'ipopt.tol': 1e-4,
            'ipopt.warm_start_init_point': 'yes'
        }
        self.opti.solver('ipopt', opts)
        
        # Save references
        self.X = X
        self.U = U
        self.x0_param = x0
        self.track_params = track_params
          
    def solve_mpc_step(self, current_state: np.ndarray, 
                       track_position: float) -> Tuple[np.ndarray, Dict]:
        """Solve one MPC iteration"""
        # Set current state
        self.opti.set_value(self.x0_param, current_state)
        
        # Set track parameters for horizon
        track_data = self._get_track_preview(track_position, current_state[1])
        self.opti.set_value(self.track_params, track_data)
        
        # Warm start from previous solution if available
        if self.solution is not None:
            try:
                # Shift previous solution for better warm start
                x_warm = np.hstack([self.solution['x_opt'][:, 1:], 
                                   self.solution['x_opt'][:, -1:]])
                u_warm = np.hstack([self.solution['u_opt'][:, 1:], 
                                   self.solution['u_opt'][:, -1:]])
                
                self.opti.set_initial(self.X, x_warm)
                self.opti.set_initial(self.U, u_warm)
            except:
                pass  # Ignore warm start errors
        else:
            # Initialize with reasonable values
            v_init = current_state[1]
            soc_init = current_state[2]
            
            # Simple initialization
            x_init = np.zeros((3, self.N+1))
            x_init[0, :] = np.linspace(current_state[0], 
                                       current_state[0] + v_init * self.horizon_time, 
                                       self.N+1)
            x_init[1, :] = v_init
            x_init[2, :] = soc_init
            
            u_init = np.zeros((3, self.N))
            u_init[1, :] = 0.7  # Moderate throttle
            
            self.opti.set_initial(self.X, x_init)
            self.opti.set_initial(self.U, u_init)
        
        # Solve
        try:
            sol = self.opti.solve()
            
            # Extract and store solution
            u_opt = sol.value(self.U)
            x_opt = sol.value(self.X)
            
            self.solution = {
                'x_opt': x_opt,
                'u_opt': u_opt
            }
            
            self.failed_solves = 0  # Reset failure counter
            
            info = {
                'success': True,
                'predicted_states': x_opt,
                'predicted_controls': u_opt,
                'solve_time': 0.0
            }
            
            # Return first control action
            return u_opt[:, 0], info
            
        except Exception as e:
            # Try to get debug solution if available
            try:
                u_opt = self.opti.debug.value(self.U)
                x_opt = self.opti.debug.value(self.X)
                
                # Store debug solution for next warm start
                self.solution = {
                    'x_opt': x_opt,
                    'u_opt': u_opt
                }
                
                # Use first control from debug solution
                return u_opt[:, 0], {'success': False, 'error': str(e), 'used_debug': True}
                
            except:
                # Complete failure - return safe default control
                self.failed_solves += 1
                
                # If too many consecutive failures, return conservative control
                if self.failed_solves > 5:
                    u_opt = np.array([0, 0.3, 0])  # Very conservative
                else:
                    u_opt = np.array([0, 0.6, 0])  # Moderate fallback
                
                info = {'success': False, 'error': str(e), 'failed_solves': self.failed_solves}
                return u_opt, info
        
    def _get_track_preview(self, current_position: float, current_velocity: float) -> np.ndarray:
        """Get track parameters for the prediction horizon"""
        track_data = np.zeros((2, self.N))
        
        # Use current velocity for better preview estimation
        estimated_speed = max(current_velocity, 30)  # At least 30 m/s
        
        for i in range(self.N):
            future_pos = (current_position + i * self.dt * estimated_speed) % self.track_model.total_length
            
            # Find corresponding segment
            cumulative_length = 0
            found = False
            for segment in self.track_model.segments:
                if cumulative_length + segment.length > future_pos:
                    track_data[0, i] = float(segment.gradient)
                    # Handle infinite radius (straights)
                    if np.isinf(segment.radius):
                        track_data[1, i] = 10000.0
                    else:
                        track_data[1, i] = float(segment.radius)
                    found = True
                    break
                cumulative_length += segment.length
            
            # Default values if segment not found
            if not found:
                track_data[0, i] = 0.0
                track_data[1, i] = 10000.0
            
        return track_data
    