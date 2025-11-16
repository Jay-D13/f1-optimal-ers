import numpy as np
import casadi as ca
from typing import Tuple, Dict

from models import VehicleDynamicsModel, F1TrackModel


class ERSOptimalController:
    """MPC-based optimal controller for ERS deployment"""
    
    def __init__(self, vehicle_model: VehicleDynamicsModel, 
                 track_model: F1TrackModel,
                 horizon_time: float = 3.0,  # Reduced from 5.0 for faster solves
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
        """Setup the CasADi optimization problem"""
        self.opti = ca.Opti()
        
        # Get dynamics function and constraints
        self.dynamics_func = self.vehicle_model.create_casadi_function()
        constraints = self.vehicle_model.get_constraints()
        
        # Decision variables (using Opti stack variables)
        X = self.opti.variable(3, self.N+1)  # States over horizon
        U = self.opti.variable(3, self.N)    # Controls over horizon
        
        # Parameters (will be set at each MPC iteration)
        x0 = self.opti.parameter(3)  # Initial state
        self.track_params = self.opti.parameter(2, self.N)  # Track parameters
        
        w_progress = 1.0
        q_soc = 1.0
        r_u = 1e-4
        r_du = 1e-2
        
        # Objective: Minimize lap time while managing energy
        obj = 0
        for k in range(self.N):
            # Maximize progress (minimize time)
            # Weight velocity instead of distance for smoother optimization
            # obj -= 5 * X[1, k]  # Maximize velocity (favors speed)
            
            
            # Deploying ERS at high speed is good (more gain)
            # ers_benefit = ca.fmax(0, U[0, k]) * X[1, k] / 50  # Normalized
            # obj -= 0.5 * ers_benefit
            
            # Penalize battery depletion
            # target_soc = 0.5
            # obj += 5 * (X[2, k] - target_soc)**2
            
            # Penalize excessive ERS switching (smoothness)
            # if k > 0:
            #     obj += 0.01 * (U[0, k] - U[0, k-1])**2
            
            obj += q_soc * (X[2, k] - 0.5)**2
            obj += r_u * ca.sumsqr(U[:, k])
            if k > 0:
                obj += r_du * (U[0,k] - U[0,k-1])**2
            
            # Small regularization on controls
            obj += 1e-5 * ca.sumsqr(U[:, k])
            
            # Soft penalty for SOC deviations from target (encourage energy neutrality)
            # target_soc = 0.5
            # obj += 0.1 * (X[2, k] - target_soc)**2
            
        progress = X[0, self.N] - X[0, 0]
        obj -= w_progress * progress
            
        self.opti.minimize(obj)
        
        # Dynamic constraints using RK4 integration
        for k in range(self.N):
            x_k = X[:, k]
            u_k = U[:, k]
            p_k = self.track_params[:, k]
            
            # RK4 integration
            k1 = self.dynamics_func(x_k, u_k, p_k)
            k2 = self.dynamics_func(x_k + self.dt/2 * k1, u_k, p_k)
            k3 = self.dynamics_func(x_k + self.dt/2 * k2, u_k, p_k)
            k4 = self.dynamics_func(x_k + self.dt * k3, u_k, p_k)
            
            x_next = x_k + self.dt/6 * (k1 + 2*k2 + 2*k3 + k4)
            
            self.opti.subject_to(X[:, k+1] == x_next)
            
        # Initial condition
        self.opti.subject_to(X[:, 0] == x0)
        
        # Path constraints
        for k in range(self.N):
            # Control constraints
            self.opti.subject_to(self.opti.bounded(
                constraints['P_ers_min'], U[0, k], constraints['P_ers_max']))
            self.opti.subject_to(self.opti.bounded(
                constraints['throttle_min'], U[1, k], constraints['throttle_max']))
            self.opti.subject_to(self.opti.bounded(
                constraints['brake_min'], U[2, k], constraints['brake_max']))
            
            # Mutual exclusivity: throttle + brake <= 1 (can't do both)
            self.opti.subject_to(U[1, k] + U[2, k] <= 1.0)
            
            a_lat_max = 1.5 * 9.81
            radius_k = self.track_params[1, k]

            # Avoid division by zero and handle straights
            v_corner_max = ca.sqrt(a_lat_max * radius_k)

            # Limit to some global max (e.g. 100 m/s) to stay safe numerically
            v_corner_max = ca.fmin(v_corner_max, constraints['v_max'])

            # Enforce lateral constraint
            self.opti.subject_to(X[1, k] <= v_corner_max)
            
            # v_corner_max_k = track_params[1,k]  # now parameter is v_max, not radius
            # self.opti.subject_to(X[1,k] <= v_corner_max_k)
            
            # State constraints with some margin for numerical stability
            self.opti.subject_to(self.opti.bounded(
                constraints['v_min'], X[1, k], constraints['v_max']))
            self.opti.subject_to(self.opti.bounded(
                constraints['soc_min'], X[2, k], constraints['soc_max']))
            
        # Terminal state constraints
        self.opti.subject_to(self.opti.bounded(
            constraints['soc_min'], X[2, -1], constraints['soc_max']))
        
        # Improved solver options for robustness
        opts = {
            'ipopt.max_iter': 300,  # Increased from 200
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.acceptable_tol': 1e-5,  # Relaxed from 1e-6
            'ipopt.acceptable_obj_change_tol': 1e-5,  # Relaxed from 1e-6
            'ipopt.acceptable_iter': 10,  # Accept solution after 10 acceptable iterations
            'ipopt.mu_strategy': 'adaptive',  # Better barrier parameter update
            'ipopt.warm_start_init_point': 'yes',  # Enable warm starting
            'ipopt.warm_start_bound_push': 1e-9,
            'ipopt.warm_start_mult_bound_push': 1e-9,
        }
        self.opti.solver('ipopt', opts)
        
        # Store variables for later use
        self.X = X
        self.U = U
        self.x0_param = x0
        # self.track_params = track_params
        
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