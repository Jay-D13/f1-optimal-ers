import numpy as np
import casadi as ca
from dataclasses import dataclass
from typing import Dict
import time

from models import F1TrackModel, VehicleDynamicsModel


@dataclass
class OptimalTrajectory:
    """Container for the globally optimal trajectory"""
    
    # Spatial discretization
    s: np.ndarray              # Distance points (m)
    ds: float                  # Step size (m)
    n_points: int              # Number of points
    
    # State trajectories
    v_opt: np.ndarray          # Optimal velocity (m/s)
    soc_opt: np.ndarray        # Optimal SOC (0-1)
    
    # Control trajectories
    P_ers_opt: np.ndarray      # Optimal ERS power (W)
    throttle_opt: np.ndarray   # Optimal throttle (0-1)
    brake_opt: np.ndarray      # Optimal brake (0-1)
    
    # Derived quantities
    t_opt: np.ndarray          # Cumulative time (s)
    lap_time: float            # Total lap time (s)
    energy_deployed: float     # Total ERS energy deployed (J)
    energy_recovered: float    # Total ERS energy recovered (J)
    
    # Solver info
    solve_time: float          # Computation time (s)
    solver_status: str         # 'optimal', 'suboptimal', 'failed'
    
    def get_reference_at_distance(self, distance: float) -> Dict:
        """Interpolate reference values at given distance"""
        # Wrap distance
        distance = distance % self.s[-1]
        
        # Find index
        idx = np.searchsorted(self.s, distance)
        idx = min(idx, self.n_points - 2)  # -2 to stay within P_ers bounds
        
        return {
            'v_ref': self.v_opt[idx],
            'soc_ref': self.soc_opt[idx],
            'P_ers_ref': self.P_ers_opt[idx],
            'throttle_ref': self.throttle_opt[idx],
            'brake_ref': self.brake_opt[idx],
        }
    
    def get_soc_trajectory_from_distance(self, current_distance: float, 
                                         horizon_length: float,
                                         n_points: int) -> np.ndarray:
        """Get SOC reference trajectory for MPC horizon"""
        distances = np.linspace(
            current_distance, 
            current_distance + horizon_length, 
            n_points
        )
        
        soc_ref = np.zeros(n_points)
        for i, d in enumerate(distances):
            d_wrapped = d % self.s[-1]
            idx = np.searchsorted(self.s, d_wrapped)
            idx = min(idx, self.n_points - 1)
            soc_ref[i] = self.soc_opt[idx]
        
        return soc_ref


class GlobalOfflineOptimizer:
    """
    Global optimizer using spatial-domain NLP formulation.
    
    Objective: Minimize lap time = ∑(ds / v[k])
    
    This is solved ONCE before the lap starts, providing:
    1. Optimal velocity profile v*(s)
    2. Optimal SOC trajectory soc*(s)
    3. Optimal control inputs P_ers*(s), throttle*(s), brake*(s)
    """
    
    def __init__(self, 
                 vehicle_model: VehicleDynamicsModel,
                 track_model: F1TrackModel,
                 ds: float = 5.0):
        """
        Initialize the global optimizer.
        
        Args:
            vehicle_model: Vehicle dynamics model
            track_model: Track model with curvature data
            ds: Spatial discretization step (meters)
        """
        self.vehicle = vehicle_model
        self.track = track_model
        self.ds = ds
        
        # Discretization
        self.N = int(track_model.total_length / ds)
        self.s_grid = np.linspace(0, track_model.total_length, self.N + 1)
        
        # CasADi optimization problem
        self.opti = None
        self.solution = None
        
        # Reference to decision variables
        self.V = None      # Velocity
        self.SOC = None    # State of charge
        self.P_ERS = None  # ERS power
        self.THROT = None  # Throttle
        self.BRAKE = None  # Brake
        
        print(f"   Offline optimizer initialized:")
        print(f"     Track length: {track_model.total_length:.0f} m")
        print(f"     Discretization: {self.N} points at {ds} m intervals")
    
    def setup_nlp(self, 
                  v_limit_profile: np.ndarray,
                  initial_soc: float = 0.5,
                  final_soc_min: float = 0.3,
                  energy_limit: float = 4e6) -> None:
        """
        Setup the NLP optimization problem.
        
        Args:
            initial_soc: Starting battery SOC
            final_soc_min: Minimum SOC at end of lap
            energy_limit: Maximum ERS deployment energy (J), default 4MJ
        """
        
        print("   Setting up global NLP...")
        
        self.opti = ca.Opti()
        constraints = self.vehicle.get_constraints()
        
        # Get track data
        track_data = self.track.track_data
        radius_array = track_data.radius
        gradient_array = track_data.gradient
        v_max_array = track_data.v_max_corner

        # Ensure we have speed limits computed
        if np.all(v_max_array == 0):
            v_max_array = self.track.compute_speed_limits(self.vehicle.vehicle)

        # Diagnostic checks for track data quality
        print(f"\n   Track data diagnostics:")
        print(f"     Radius: min={np.min(radius_array):.1f} m, max={np.max(radius_array):.1f} m")
        print(f"     Gradient: min={np.min(gradient_array):.4f}, max={np.max(gradient_array):.4f}")
        print(f"     v_max: min={np.min(v_max_array):.1f} m/s, max={np.max(v_max_array):.1f} m/s")

        # Check for problematic values
        n_zero_vmax = np.sum(v_max_array <= 0)
        n_tiny_radius = np.sum(radius_array < 10)
        if n_zero_vmax > 0:
            print(f"     WARNING: {n_zero_vmax} points have zero/negative v_max!")
        if n_tiny_radius > 0:
            print(f"     WARNING: {n_tiny_radius} points have radius < 10m!")
        
        # === Decision Variables ===
        # State variables at each spatial node
        V = self.opti.variable(self.N + 1)      # Velocity [m/s]
        SOC = self.opti.variable(self.N + 1)    # State of charge [0-1]
        
        for k in range(self.N + 1):
            self.opti.subject_to(self.opti.bounded(15.0, V[k], 90.0))
            self.opti.subject_to(self.opti.bounded(0.1, SOC[k], 0.95))
        
        # Control variables between nodes
        P_ERS = self.opti.variable(self.N)      # ERS power [W]
        THROT = self.opti.variable(self.N)      # Throttle [0-1]
        BRAKE = self.opti.variable(self.N)      # Brake [0-1]
        
        # Slack variable for soft constraints (robustness)
        SLACK_V = self.opti.variable(self.N + 1)  # Velocity slack
        
        # Store references
        self.V = V
        self.SOC = SOC
        self.P_ERS = P_ERS
        self.THROT = THROT
        self.BRAKE = BRAKE
        
        # === Objective Function ===
        # Minimize lap time: T = ∫(1/v)ds ≈ ∑(ds/v[k])
        lap_time = 0
        for k in range(self.N):
            # Use average velocity over segment for accuracy
            v_avg = 0.5 * (V[k] + V[k+1])
            v_safe = ca.fmax(v_avg, 5.0)  # Prevent division by zero
            lap_time += self.ds / v_safe
        
        # Add penalty for slack variables (soft constraint violation)
        slack_penalty = 1000 * ca.sum1(SLACK_V**2)
        
        self.opti.minimize(lap_time + slack_penalty)
        
        # === Dynamics Constraints ===
        dynamics = self.vehicle.create_spatial_domain_dynamics()
        
        for k in range(self.N):
            # Get track parameters for this segment
            # Handle index wrapping for safety
            track_idx = min(k, len(radius_array) - 1)
            radius_k = float(radius_array[track_idx])
            gradient_k = float(gradient_array[track_idx])
            v_max_k = float(v_max_array[track_idx])

            # Sanitize track data to prevent NaN
            radius_k = max(radius_k, 15.0)  # Minimum 15m radius
            gradient_k = np.clip(gradient_k, -0.15, 0.15)  # Max ±15% grade
            v_max_k = max(v_max_k, 20.0)  # Minimum 20 m/s speed limit

            # Current state and control
            x_k = ca.vertcat(V[k], SOC[k])
            u_k = ca.vertcat(P_ERS[k], THROT[k], BRAKE[k])
            p_k = ca.vertcat(gradient_k, radius_k)
            
            # Get spatial derivatives
            dx_ds, dt_ds = dynamics(x_k, u_k, p_k)
            
            # Euler integration in spatial domain
            # x[k+1] = x[k] + dx_ds * ds
            x_next = x_k + dx_ds * self.ds
            
            # Dynamics constraints
            self.opti.subject_to(V[k+1] == x_next[0])
            self.opti.subject_to(SOC[k+1] == x_next[1])
            
            # === State Constraints ===
            # Velocity limits (with soft slack)
            # self.opti.subject_to(V[k] >= constraints['v_min'] - SLACK_V[k])
            # self.opti.subject_to(V[k] <= v_max_k + SLACK_V[k])
            limit_k = v_limit_profile[k] * 0.99 
            self.opti.subject_to(V[k] <= limit_k + SLACK_V[k])
            
            # SOC limits (hard constraints for battery safety)
            self.opti.subject_to(SOC[k] >= constraints['soc_min'])
            self.opti.subject_to(SOC[k] <= constraints['soc_max'])
            
            # === Control Constraints ===
            self.opti.subject_to(self.opti.bounded(
                constraints['P_ers_min'], P_ERS[k], constraints['P_ers_max']
            ))
            self.opti.subject_to(self.opti.bounded(
                constraints['throttle_min'], THROT[k], constraints['throttle_max']
            ))
            self.opti.subject_to(self.opti.bounded(
                constraints['brake_min'], BRAKE[k], constraints['brake_max']
            ))
            
            # No simultaneous throttle and heavy braking
            self.opti.subject_to(THROT[k] * BRAKE[k] <= 0.1)
            
            # Slack must be non-negative
            self.opti.subject_to(SLACK_V[k] >= 0)
        
        # Final node constraints
        self.opti.subject_to(V[self.N] >= constraints['v_min'] - SLACK_V[self.N])
        self.opti.subject_to(SLACK_V[self.N] >= 0)
        
        # === Boundary Conditions ===
        # Initial conditions
        self.opti.subject_to(SOC[0] == initial_soc)
        
        # Starting velocity (use first corner speed limit as estimate)
        # maybe should use real starting data speed
        v_start = min(50.0, v_max_array[0])  # Conservative start
        self.opti.subject_to(V[0] >= v_start * 0.8)
        self.opti.subject_to(V[0] <= v_start * 1.2)
        
        # Terminal conditions
        # SOC should end at reasonable level (for next lap or race strategy)
        self.opti.subject_to(SOC[self.N] >= final_soc_min)
        
        # Periodic velocity? (for consistency)
        # self.opti.subject_to(V[self.N] == V[0])
        
        # === Energy Limit Constraint ===
        # Total ERS deployment must not exceed limit (e.g., 4MJ/lap)
        total_deployment = 0
        for k in range(self.N):
            # Only count positive P_ERS (deployment)
            v_avg = ca.fmax(0.5 * (V[k] + V[k+1]), 5.0)
            dt_segment = self.ds / v_avg
            deployment_k = ca.fmax(P_ERS[k], 0) * dt_segment
            total_deployment += deployment_k
        
        self.opti.subject_to(total_deployment <= energy_limit)
        
        # === Solver Configuration ===
        opts = {
            'ipopt.max_iter': 3000,
            'ipopt.print_level': 5,  # More verbose to see where NaN occurs
            'print_time': 1,
            'ipopt.tol': 1e-3,
            'ipopt.acceptable_tol': 1e-2,
            'ipopt.acceptable_iter': 15,
            'ipopt.linear_solver': 'ma97', # got a license yaaay! "mumps" or "ma27" if no license
            'ipopt.warm_start_init_point': 'yes',
            
            # Robustness settings to handle NaN
            'ipopt.mu_strategy': 'adaptive',
            'ipopt.nlp_scaling_method': 'gradient-based',
            'ipopt.bound_relax_factor': 1e-8,
            'ipopt.honor_original_bounds': 'yes',
            
            # Additional NaN prevention
            'ipopt.check_derivatives_for_naninf': 'yes',
            'ipopt.alpha_for_y': 'safer-min-dual-infeas',  # More conservative step
            'ipopt.recalc_y': 'yes',
            
            # Derivative checker (helpful for debugging)
            # 'ipopt.derivative_test': 'first-order',
            # 'ipopt.derivative_test_tol': 1e-4,
        }
        self.opti.solver('ipopt', opts)
        
        # === Initial Guess ===
        self._set_initial_guess(v_max_array, initial_soc)
        
        print("   ✓ NLP setup complete")
        print(f"     Variables: {(self.N+1)*2 + self.N*3 + (self.N+1)}")
        print(f"     Constraints: ~{self.N * 10}")
    
    def _set_initial_guess(self, v_max_array: np.ndarray, initial_soc: float):
        """Set warm-start initial guess for faster convergence"""

        # Ensure v_max_array has valid values
        # Replace any zeros, NaNs, or unrealistic values
        v_max_clean = np.copy(v_max_array)
        v_max_clean = np.nan_to_num(v_max_clean, nan=50.0, posinf=80.0, neginf=20.0)

        # IMPORTANT: Only clip the UPPER bound, not lower!
        # Low speeds in hairpins are physically necessary
        v_max_clean = np.clip(v_max_clean, None, 85.0)  # Max 85 m/s
        v_max_clean = np.maximum(v_max_clean, 15.0)  # But at least 15 m/s

        # Velocity: 70% of maximum speed limit (conservative but not too slow)
        v_init = 0.7 * v_max_clean

        # Only clip upper bound to avoid numerical issues at very high speeds
        v_init = np.minimum(v_init, 75.0)

        # Extend to N+1 points
        if len(v_init) < self.N + 1:
            v_init = np.resize(v_init, self.N + 1)

        self.opti.set_initial(self.V, v_init[:self.N + 1])

        # SOC: Linear from initial to slightly below
        soc_init = np.linspace(initial_soc, initial_soc - 0.1, self.N + 1)
        soc_init = np.clip(soc_init, 0.25, 0.75)  # Stay within safe bounds
        self.opti.set_initial(self.SOC, soc_init)

        # Controls: Moderate values
        self.opti.set_initial(self.P_ERS, np.zeros(self.N))
        self.opti.set_initial(self.THROT, 0.6 * np.ones(self.N))  # Higher throttle for acceleration
        self.opti.set_initial(self.BRAKE, np.zeros(self.N))

        # Debug output
        print(f"   Initial guess set:")
        print(f"     V: {v_init.min():.1f} - {v_init.max():.1f} m/s ({v_init.min()*3.6:.0f} - {v_init.max()*3.6:.0f} km/h)")
        print(f"     SOC: {soc_init.min():.2f} - {soc_init.max():.2f}")
    
    def solve(self) -> OptimalTrajectory:
        """
        Solve the global optimization problem.
        
        Returns:
            OptimalTrajectory containing the optimal reference trajectories
        """
        
        if self.opti is None:
            raise RuntimeError("NLP not setup. Call setup_nlp() first.")
        
        print("\n   Solving global optimization...")
        print("   " + "="*50)
        
        start_time = time.time()
        
        try:
            sol = self.opti.solve()
            status = 'optimal'
            
            # Extract solution
            v_opt = sol.value(self.V)
            soc_opt = sol.value(self.SOC)
            P_ers_opt = sol.value(self.P_ERS)
            throttle_opt = sol.value(self.THROT)
            brake_opt = sol.value(self.BRAKE)
            
        except Exception as e:
            print(f"   ⚠ Solver did not find optimal solution: {e}")
            print("   Attempting to extract suboptimal solution...")
            
            try:
                v_opt = self.opti.debug.value(self.V)
                soc_opt = self.opti.debug.value(self.SOC)
                P_ers_opt = self.opti.debug.value(self.P_ERS)
                throttle_opt = self.opti.debug.value(self.THROT)
                brake_opt = self.opti.debug.value(self.BRAKE)
                status = 'suboptimal'
            except:
                raise RuntimeError("Failed to extract any solution")
        
        solve_time = time.time() - start_time
        
        # === Compute Derived Quantities ===
        # Cumulative time
        t_opt = np.zeros(self.N + 1)
        for k in range(self.N):
            v_avg = max(0.5 * (v_opt[k] + v_opt[k+1]), 5.0)
            t_opt[k+1] = t_opt[k] + self.ds / v_avg
        
        lap_time = t_opt[-1]
        
        # Energy accounting
        energy_deployed = 0.0
        energy_recovered = 0.0
        
        for k in range(self.N):
            v_avg = max(0.5 * (v_opt[k] + v_opt[k+1]), 5.0)
            dt = self.ds / v_avg
            
            if P_ers_opt[k] > 0:
                energy_deployed += P_ers_opt[k] * dt
            else:
                energy_recovered += -P_ers_opt[k] * dt
        
        # === Create Result Object ===
        trajectory = OptimalTrajectory(
            s=self.s_grid,
            ds=self.ds,
            n_points=self.N + 1,
            v_opt=v_opt,
            soc_opt=soc_opt,
            P_ers_opt=P_ers_opt,
            throttle_opt=throttle_opt,
            brake_opt=brake_opt,
            t_opt=t_opt,
            lap_time=lap_time,
            energy_deployed=energy_deployed,
            energy_recovered=energy_recovered,
            solve_time=solve_time,
            solver_status=status,
        )
        
        self.solution = trajectory
        
        # Print summary
        print("   " + "="*50)
        print(f"   ✓ Global optimization complete!")
        print(f"     Status: {status}")
        print(f"     Lap time: {lap_time:.3f} s")
        print(f"     Solve time: {solve_time:.2f} s")
        print(f"     Energy deployed: {energy_deployed/1e6:.2f} MJ")
        print(f"     Energy recovered: {energy_recovered/1e6:.2f} MJ")
        print(f"     Final SOC: {soc_opt[-1]*100:.1f}%")
        print(f"     Velocity range: {v_opt.min():.1f} - {v_opt.max():.1f} m/s")
        
        return trajectory
    
    def get_reference_trajectory(self) -> OptimalTrajectory:
        """Get the stored optimal trajectory"""
        if self.solution is None:
            raise RuntimeError("No solution available. Call solve() first.")
        return self.solution