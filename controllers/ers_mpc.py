"""
ERS Model Predictive Controller with Co-State Extraction

Based on academic approaches:
- Serrao et al. (2009): ECMS equivalence factor = co-state from PMP
- ETH Zurich ELTMS: Equivalent Lap Time Minimization Strategy
- Oxford F1 research: Bang-bang optimal control for MGU-K

Key Features:
1. Extracts co-states (dual variables) from offline NLP solution
2. Uses threshold-based switching with PI adaptation
3. Warm-starting from previous solution
4. Handles tire degradation and fuel mass changes
"""

import numpy as np
import casadi as ca
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, List
import time

from models import VehicleDynamicsModel, F1TrackModel
from config import ERSConfig
from solvers import OptimalTrajectory


@dataclass
class MPCConfig:
    """Configuration for MPC controller"""
    # Horizon parameters
    horizon_distance: float = 200.0     # [m] Prediction horizon (reduced for speed)
    horizon_segments: int = 40          # Number of discretization points (reduced)
    
    # Tracking weights - INCREASED for better reference tracking
    w_laptime: float = 1.0              # Lap time minimization weight
    w_velocity_track: float = 0.1       # Velocity tracking weight (increased)
    w_soc_track: float = 1.0            # SOC tracking weight (increased significantly)
    w_terminal_soc: float = 5.0         # Terminal SOC weight (increased)
    
    # Smoothness weights
    w_P_ers_rate: float = 1e-6          # ERS power rate penalty
    w_throttle_rate: float = 1e-4       # Throttle rate penalty
    w_brake_rate: float = 1e-4          # Brake rate penalty
    
    # PI controller gains for equivalence factor adaptation
    Kp_soc: float = 0.5                 # Proportional gain
    Ki_soc: float = 0.1                 # Integral gain
    
    # Solver settings
    max_iter: int = 50                  # Reduced for speed
    tol: float = 1e-3                   # Relaxed for speed
    verbose: bool = False
    debug: bool = False                 # Enable debug prints


@dataclass 
class CoStateProfile:
    """
    Co-states extracted from offline NLP solution.
    
    The co-state λ_soc represents the marginal value of battery energy
    in terms of lap time. This is the key for online ECMS/ELTMS control.
    """
    s: np.ndarray                       # Distance points
    lambda_soc: np.ndarray              # SOC co-state (∂T/∂SOC)
    lambda_v: np.ndarray                # Velocity co-state  
    
    # Derived thresholds
    deploy_threshold: np.ndarray        # When to deploy ERS
    harvest_threshold: np.ndarray       # When to harvest
    
    def get_costate_at_distance(self, distance: float) -> Dict:
        """Interpolate co-states at given distance"""
        distance = distance % self.s[-1]
        idx = np.searchsorted(self.s, distance)
        idx = min(idx, len(self.s) - 1)
        
        # Linear interpolation
        if idx > 0 and idx < len(self.s):
            alpha = (distance - self.s[idx-1]) / (self.s[idx] - self.s[idx-1] + 1e-6)
            lambda_soc = (1 - alpha) * self.lambda_soc[idx-1] + alpha * self.lambda_soc[idx]
            lambda_v = (1 - alpha) * self.lambda_v[idx-1] + alpha * self.lambda_v[idx]
        else:
            lambda_soc = self.lambda_soc[idx]
            lambda_v = self.lambda_v[idx]
            
        return {
            'lambda_soc': lambda_soc,
            'lambda_v': lambda_v,
            'deploy_threshold': self.deploy_threshold[idx],
            'harvest_threshold': self.harvest_threshold[idx],
        }


class ERSMPCController:
    """
    MPC Controller for ERS Management
    
    Implements a receding-horizon optimal control that:
    1. Tracks the offline optimal trajectory
    2. Uses co-state information for energy management decisions
    3. Adapts to real-time disturbances (tire deg, fuel, etc.)
    
    The controller solves:
        min  Σ(ds/v) + w_track*||x - x_ref||² + w_smooth*||Δu||²
        s.t. Vehicle dynamics
             State/control constraints
             Energy constraints
    """
    
    def __init__(self,
                 vehicle_model: VehicleDynamicsModel,
                 track_model: F1TrackModel,
                 ers_config: ERSConfig,
                 config: Optional[MPCConfig] = None):
        
        self.vehicle = vehicle_model
        self.track = track_model
        self.ers = ers_config
        self.config = config or MPCConfig()
        
        # Compute segment length
        self.ds = self.config.horizon_distance / self.config.horizon_segments
        self.N = self.config.horizon_segments
        
        # Reference trajectory (set externally)
        self.reference: Optional[OptimalTrajectory] = None
        self.costate_profile: Optional[CoStateProfile] = None
        
        # Adaptive equivalence factor (for ELTMS)
        self.equivalence_factor = 1.0
        self.soc_error_integral = 0.0
        
        # Previous solution for warm starting
        self._prev_solution: Optional[Dict] = None
        self._prev_position: float = 0.0
        
        # Performance tracking
        self.solve_count = 0
        self.fail_count = 0
        self.total_solve_time = 0.0
        
        # Build the optimization problem
        self._build_mpc_problem()
        
        print(f"   ERS MPC Controller initialized:")
        print(f"     Horizon: {self.config.horizon_distance}m ({self.N} segments)")
        print(f"     ds: {self.ds:.1f}m")
        
    def set_reference(self, reference: OptimalTrajectory):
        """
        Set offline optimal trajectory as reference.
        Also extracts co-states for ELTMS control.
        """
        self.reference = reference
        
        # Extract co-states from the reference trajectory
        self.costate_profile = self._extract_costates(reference)
        
        # Reset adaptation state
        self.equivalence_factor = 1.0
        self.soc_error_integral = 0.0
        self._prev_solution = None
        
        print(f"   Reference trajectory set: {reference.lap_time:.3f}s")
        print(f"   Co-state range: λ_soc ∈ [{self.costate_profile.lambda_soc.min():.4f}, "
              f"{self.costate_profile.lambda_soc.max():.4f}]")
        
    def _extract_costates(self, trajectory: OptimalTrajectory) -> CoStateProfile:
        """
        Extract co-states from offline solution.
        
        The co-state λ_soc represents ∂T*/∂SOC - how much lap time changes
        per unit of SOC. This is approximated from the optimal trajectory
        by analyzing the relationship between SOC and velocity/time.
        
        For a properly solved NLP, we could get the actual dual variables.
        Here we estimate them from the trajectory characteristics.
        """
        s = trajectory.s
        v = trajectory.v_opt
        soc = trajectory.soc_opt
        P_ers = trajectory.P_ers_opt
        
        n = len(s)
        
        # Estimate λ_soc from the marginal value of energy
        # When P_ers > 0 (deploying), the car is faster
        # λ_soc ≈ (speed gain) / (energy cost)
        
        lambda_soc = np.zeros(n)
        lambda_v = np.zeros(n)
        
        for i in range(n - 1):
            # Look at how velocity responds to ERS power
            if abs(P_ers[i]) > 1000:  # Significant ERS use
                # Estimate time benefit per unit energy
                ds = s[i+1] - s[i] if i < n-1 else trajectory.ds
                v_avg = 0.5 * (v[i] + v[i+1])
                dt = ds / max(v_avg, 5.0)
                
                # Energy used in this segment
                dE = abs(P_ers[i]) * dt
                
                if P_ers[i] > 0:  # Deploying
                    # Higher SOC value where we deploy (we want to use it here)
                    lambda_soc[i] = 1.0 / max(dE, 1e-6) * dt
                else:  # Harvesting
                    # Lower value where we harvest (we gain energy)
                    lambda_soc[i] = -0.5 / max(dE, 1e-6) * dt
            else:
                # Coasting - interpolate
                lambda_soc[i] = 0.0
        
        # Smooth the co-state profile
        from scipy.ndimage import gaussian_filter1d
        lambda_soc = gaussian_filter1d(lambda_soc, sigma=5)
        
        # Normalize to reasonable range
        if np.max(np.abs(lambda_soc)) > 0:
            lambda_soc = lambda_soc / np.max(np.abs(lambda_soc))
        
        # Velocity co-state (approximated)
        lambda_v = -1.0 / (v + 1e-6)  # Higher value at low speeds
        
        # Compute deployment/harvest thresholds
        # Deploy when lap-time benefit > equivalence_factor * energy_cost
        deploy_threshold = np.ones(n) * 0.5   # Base threshold
        harvest_threshold = np.ones(n) * 0.3  # Harvest when below this
        
        # Adjust thresholds based on track position
        # Higher threshold (more aggressive) on straights
        for i in range(n):
            segment = self.track.get_segment_at_distance(s[i])
            if segment.radius > 300:  # Straight
                deploy_threshold[i] = 0.7
                harvest_threshold[i] = 0.2
            elif segment.radius < 100:  # Tight corner
                deploy_threshold[i] = 0.3
                harvest_threshold[i] = 0.5
        
        return CoStateProfile(
            s=s,
            lambda_soc=lambda_soc,
            lambda_v=lambda_v,
            deploy_threshold=deploy_threshold,
            harvest_threshold=harvest_threshold,
        )
    
    def _build_mpc_problem(self):
        """Build CasADi optimization problem for MPC"""
        
        N = self.N
        opti = ca.Opti()
        
        # =====================================================================
        # DECISION VARIABLES
        # =====================================================================
        
        V = opti.variable(N + 1)          # Velocity [m/s]
        SOC = opti.variable(N + 1)        # State of charge [0-1]
        P_ERS = opti.variable(N)          # ERS power [W] (+ deploy, - harvest)
        THROTTLE = opti.variable(N)       # Throttle [0-1]
        BRAKE = opti.variable(N)          # Brake [0-1]
        
        # =====================================================================
        # PARAMETERS (set at runtime)
        # =====================================================================
        
        # Initial conditions
        v_init = opti.parameter()
        soc_init = opti.parameter()
        
        # Track parameters (arrays for horizon)
        gradient = opti.parameter(N)
        radius = opti.parameter(N)
        v_limit = opti.parameter(N + 1)
        
        # Reference trajectory
        v_ref = opti.parameter(N + 1)
        soc_ref = opti.parameter(N + 1)
        P_ers_ref = opti.parameter(N)
        
        # Co-state profile
        lambda_soc = opti.parameter(N + 1)
        
        # Vehicle state modifiers
        tire_factor = opti.parameter()    # Tire degradation [0-1]
        mass = opti.parameter()           # Current vehicle mass [kg]
        
        # Adaptive equivalence factor
        equiv_factor = opti.parameter()
        
        # =====================================================================
        # OBJECTIVE FUNCTION
        # =====================================================================
        
        cfg = self.config
        veh = self.vehicle.vehicle
        ers = self.vehicle.ers
        
        lap_time = 0
        tracking_cost = 0
        smoothness_cost = 0
        ers_tracking_cost = 0
        
        for k in range(N):
            # Lap time (primary objective)
            v_avg = 0.5 * (V[k] + V[k+1])
            v_safe = ca.fmax(v_avg, 5.0)
            lap_time += self.ds / v_safe
            
            # Tracking cost (stay near reference)
            tracking_cost += cfg.w_velocity_track * (V[k] - v_ref[k])**2
            tracking_cost += cfg.w_soc_track * (SOC[k] - soc_ref[k])**2
            
            # ERS power tracking (KEY: follow the optimal deployment strategy)
            # Normalize by max power squared for reasonable scaling
            P_max_sq = (ers.max_deployment_power)**2
            ers_tracking_cost += 0.1 * (P_ERS[k] - P_ers_ref[k])**2 / P_max_sq
        
        # Control smoothness
        for k in range(N - 1):
            smoothness_cost += cfg.w_P_ers_rate * (P_ERS[k+1] - P_ERS[k])**2
            smoothness_cost += cfg.w_throttle_rate * (THROTTLE[k+1] - THROTTLE[k])**2
            smoothness_cost += cfg.w_brake_rate * (BRAKE[k+1] - BRAKE[k])**2
        
        # Terminal cost - track both velocity and SOC at end of horizon
        terminal_cost = cfg.w_terminal_soc * (SOC[N] - soc_ref[N])**2
        terminal_cost += cfg.w_velocity_track * (V[N] - v_ref[N])**2
        
        opti.minimize(
            cfg.w_laptime * lap_time + 
            tracking_cost + 
            smoothness_cost + 
            ers_tracking_cost +
            terminal_cost
        )
        
        # =====================================================================
        # DYNAMICS CONSTRAINTS
        # =====================================================================
        
        for k in range(N):
            v_k = V[k]
            v_safe = ca.fmax(v_k, 5.0)
            grad_k = gradient[k]
            radius_k = ca.fmax(radius[k], 15.0)
            
            # Aerodynamic forces
            q = 0.5 * veh.rho_air * v_k**2
            F_drag = q * veh.c_w_a
            F_downforce = q * (veh.c_z_a_f + veh.c_z_a_r)
            
            # Normal force
            F_normal = mass * veh.g * ca.cos(grad_k) + F_downforce
            F_roll = veh.cr * F_normal
            F_gravity = mass * veh.g * ca.sin(grad_k)
            
            # Grip limit (scaled by tire degradation)
            mu_eff = veh.mu_longitudinal * tire_factor
            F_grip_max = mu_eff * F_normal
            
            # Lateral force (cornering)
            a_lat = v_k**2 / radius_k
            F_lat = mass * a_lat
            
            # Friction circle: remaining longitudinal grip
            F_grip_long_sq = F_grip_max**2 - F_lat**2
            F_grip_long = ca.sqrt(ca.fmax(F_grip_long_sq, 100.0))
            
            # Propulsion
            P_ers_deploy = ca.fmax(P_ERS[k], 0)
            P_ice = THROTTLE[k] * veh.pow_max_ice
            P_total = P_ice + P_ers_deploy
            F_prop_request = P_total / v_safe
            F_prop = ca.fmin(F_prop_request, F_grip_long)
            
            # Braking
            P_ers_harvest = ca.fmin(P_ERS[k], 0)
            F_brake_mech = BRAKE[k] * veh.max_brake_force
            F_brake_regen = -P_ers_harvest / v_safe
            F_brake_total = ca.fmin(F_brake_mech + F_brake_regen, F_grip_long)
            
            # Net force and acceleration
            F_net = F_prop - F_drag - F_roll - F_gravity - F_brake_total
            dv_ds = F_net / (mass * v_safe)
            
            opti.subject_to(V[k+1] == V[k] + dv_ds * self.ds)
            
            # SOC dynamics
            sigma = 0.5 * (1 + ca.tanh(P_ERS[k] / 1000.0))
            P_battery = sigma * (P_ERS[k] / ers.deployment_efficiency) + \
                       (1 - sigma) * (P_ERS[k] * ers.recovery_efficiency)
            dsoc_ds = -P_battery / (ers.battery_capacity * v_safe)
            
            opti.subject_to(SOC[k+1] == SOC[k] + dsoc_ds * self.ds)
        
        # =====================================================================
        # CONSTRAINTS
        # =====================================================================
        
        cons = self.vehicle.get_constraints()
        
        # Initial conditions
        opti.subject_to(V[0] == v_init)
        opti.subject_to(SOC[0] == soc_init)
        
        # State bounds
        for k in range(N + 1):
            opti.subject_to(V[k] >= cons['v_min'])
            opti.subject_to(V[k] <= v_limit[k] * 1.05)
            opti.subject_to(SOC[k] >= cons['soc_min'])
            opti.subject_to(SOC[k] <= cons['soc_max'])
        
        # Control bounds
        for k in range(N):
            opti.subject_to(P_ERS[k] >= cons['P_ers_min'])
            opti.subject_to(P_ERS[k] <= cons['P_ers_max'])
            opti.subject_to(THROTTLE[k] >= 0)
            opti.subject_to(THROTTLE[k] <= 1)
            opti.subject_to(BRAKE[k] >= 0)
            opti.subject_to(BRAKE[k] <= 1)
            # Prevent simultaneous throttle and brake
            opti.subject_to(THROTTLE[k] * BRAKE[k] <= 0.1)
        
        # =====================================================================
        # SOLVER SETUP
        # =====================================================================
        
        opts = {
            'ipopt.max_iter': self.config.max_iter,
            'ipopt.print_level': 3 if self.config.verbose else 0,
            'print_time': 0,
            'ipopt.tol': self.config.tol,
            'ipopt.acceptable_tol': self.config.tol * 10,
            'ipopt.acceptable_iter': 5,
            'ipopt.warm_start_init_point': 'yes',
            'ipopt.warm_start_bound_push': 1e-6,
            'ipopt.warm_start_mult_bound_push': 1e-6,
            'ipopt.mu_init': 1e-4,
            'ipopt.linear_solver': 'mumps',
        }
        
        opti.solver('ipopt', opts)
        
        # Store problem components
        self.opti = opti
        self.vars = {
            'V': V, 'SOC': SOC, 'P_ERS': P_ERS,
            'THROTTLE': THROTTLE, 'BRAKE': BRAKE
        }
        self.params = {
            'v_init': v_init, 'soc_init': soc_init,
            'gradient': gradient, 'radius': radius, 'v_limit': v_limit,
            'v_ref': v_ref, 'soc_ref': soc_ref, 'P_ers_ref': P_ers_ref,
            'lambda_soc': lambda_soc,
            'tire_factor': tire_factor, 'mass': mass,
            'equiv_factor': equiv_factor,
        }
    
    def solve_mpc_step(self,
                       current_state: np.ndarray,
                       track_position: float,
                       tire_degradation: float = 1.0,
                       fuel_mass: float = None) -> Tuple[np.ndarray, Dict]:
        """
        Compatibility method for LapSimulator.
        
        Args:
            current_state: [position, velocity, soc] array
            track_position: Current distance along track (m)
            tire_degradation: Tire grip factor [0-1]
            fuel_mass: Current fuel mass [kg]
            
        Returns:
            control: [P_ers, throttle, brake]
            info: Dictionary with solver info
        """
        # Extract state components
        velocity = current_state[1]
        soc = current_state[2]
        
        return self.solve(
            position=track_position,
            velocity=velocity,
            soc=soc,
            tire_degradation=tire_degradation,
            fuel_mass=fuel_mass
        )
    
    def solve(self,
              position: float,
              velocity: float,
              soc: float,
              tire_degradation: float = 1.0,
              fuel_mass: float = None) -> Tuple[np.ndarray, Dict]:
        """
        Solve one MPC step (receding horizon).
        
        Args:
            position: Current track position [m]
            velocity: Current velocity [m/s]
            soc: Current state of charge [0-1]
            tire_degradation: Tire grip factor [0-1], 1.0 = fresh
            fuel_mass: Current fuel mass [kg] (for mass calculation)
            
        Returns:
            control: [P_ers, throttle, brake]
            info: Dictionary with solver info
        """
        
        if self.reference is None:
            raise RuntimeError("Reference trajectory not set. Call set_reference() first.")
        
        cfg = self.config
        start_time = time.time()
        
        # Calculate current mass
        if fuel_mass is None:
            current_mass = self.vehicle.vehicle.mass
        else:
            current_mass = self.vehicle.vehicle.mass - 110 + fuel_mass  # Base + fuel
        
        # Get track and reference data for horizon
        track_ahead = self._get_horizon_track_data(position)
        ref_ahead = self._get_horizon_reference(position)
        costate_ahead = self._get_horizon_costates(position)
        
        # Debug: Check reference values
        if cfg.debug and self.solve_count % 50 == 0:
            print(f"\n   [MPC Debug] Step {self.solve_count}, pos={position:.0f}m")
            print(f"     Current: v={velocity:.1f}m/s, soc={soc*100:.1f}%")
            print(f"     Ref: v={ref_ahead['v'][0]:.1f}m/s, soc={ref_ahead['soc'][0]*100:.1f}%")
            print(f"     Ref P_ers[0:3]: {ref_ahead['P_ers'][:3]/1000}")
        
        # Update equivalence factor (PI controller)
        self._update_equivalence_factor(soc, ref_ahead['soc'][0])
        
        # Set parameters
        self.opti.set_value(self.params['v_init'], velocity)
        self.opti.set_value(self.params['soc_init'], soc)
        self.opti.set_value(self.params['gradient'], track_ahead['gradient'])
        self.opti.set_value(self.params['radius'], track_ahead['radius'])
        self.opti.set_value(self.params['v_limit'], track_ahead['v_limit'])
        self.opti.set_value(self.params['v_ref'], ref_ahead['v'])
        self.opti.set_value(self.params['soc_ref'], ref_ahead['soc'])
        self.opti.set_value(self.params['P_ers_ref'], ref_ahead['P_ers'])
        self.opti.set_value(self.params['lambda_soc'], costate_ahead)
        self.opti.set_value(self.params['tire_factor'], tire_degradation)
        self.opti.set_value(self.params['mass'], current_mass)
        self.opti.set_value(self.params['equiv_factor'], self.equivalence_factor)
        
        # Warm start
        self._warm_start(ref_ahead, velocity, soc, position)
        
        # Solve
        try:
            sol = self.opti.solve()
            status = 'optimal'
            
            P_ers_opt = sol.value(self.vars['P_ERS'])
            throttle_opt = sol.value(self.vars['THROTTLE'])
            brake_opt = sol.value(self.vars['BRAKE'])
            V_opt = sol.value(self.vars['V'])
            SOC_opt = sol.value(self.vars['SOC'])
            
            # Debug: Check solution
            if cfg.debug and self.solve_count % 50 == 0:
                print(f"     Solution: P_ers[0]={P_ers_opt[0]/1000:.1f}kW, throttle={throttle_opt[0]:.2f}")
                print(f"     Predicted SOC: {SOC_opt[0]*100:.1f}% -> {SOC_opt[-1]*100:.1f}%")
            
            # Store for warm starting
            self._prev_solution = {
                'V': V_opt, 'SOC': SOC_opt,
                'P_ERS': P_ers_opt, 'THROTTLE': throttle_opt, 'BRAKE': brake_opt
            }
            self._prev_position = position
            self.solve_count += 1
            
        except Exception as e:
            if cfg.verbose or cfg.debug:
                print(f"   MPC solve failed: {e}")
            status = 'fallback'
            
            # Try debug values first
            try:
                P_ers_opt = self.opti.debug.value(self.vars['P_ERS'])
                throttle_opt = self.opti.debug.value(self.vars['THROTTLE'])
                brake_opt = self.opti.debug.value(self.vars['BRAKE'])
                V_opt = self.opti.debug.value(self.vars['V'])
                SOC_opt = self.opti.debug.value(self.vars['SOC'])
                
                if cfg.debug:
                    print(f"     Using debug values: P_ers={P_ers_opt[0]/1000:.1f}kW")
                    
            except:
                # Fallback to co-state based control (ELTMS style)
                control, status = self._fallback_control(position, velocity, soc, ref_ahead)
                solve_time = time.time() - start_time
                self.fail_count += 1
                
                if cfg.debug:
                    print(f"     Fallback control: {control}")
                
                return control, {
                    'status': status,
                    'solve_time': solve_time,
                    'equivalence_factor': self.equivalence_factor,
                }
        
        solve_time = time.time() - start_time
        self.total_solve_time += solve_time
        
        # Return first control (receding horizon)
        control = np.array([P_ers_opt[0], throttle_opt[0], brake_opt[0]])
        
        info = {
            'status': status,
            'solve_time': solve_time,
            'horizon_length': self.N,
            'equivalence_factor': self.equivalence_factor,
            'soc_error': soc - ref_ahead['soc'][0],
            'predicted_soc_final': SOC_opt[-1] if status == 'optimal' else None,
        }
        
        return control, info
    
    def _update_equivalence_factor(self, current_soc: float, ref_soc: float):
        """
        Update equivalence factor using PI controller.
        
        The equivalence factor s determines the threshold for ERS deployment.
        Higher s = more conservative (save energy)
        Lower s = more aggressive (use energy)
        
        When SOC > reference: decrease s (use more energy)
        When SOC < reference: increase s (save energy)
        """
        error = current_soc - ref_soc
        
        # PI update
        self.soc_error_integral += error
        self.soc_error_integral = np.clip(self.soc_error_integral, -1.0, 1.0)
        
        delta_s = (self.config.Kp_soc * error + 
                   self.config.Ki_soc * self.soc_error_integral)
        
        # Update equivalence factor
        self.equivalence_factor = np.clip(1.0 - delta_s, 0.5, 2.0)
    
    def _fallback_control(self, 
                          position: float,
                          velocity: float,
                          soc: float,
                          ref: Dict) -> Tuple[np.ndarray, str]:
        """
        Fallback control using ELTMS-style threshold logic.
        
        This is used when the MPC solver fails. It uses the co-state
        information to make deployment/harvest decisions.
        """
        if self.costate_profile is None:
            # No co-state info, use simple logic
            return self._simple_fallback(velocity, soc, ref), 'simple_fallback'
        
        # Get co-state info
        costate_info = self.costate_profile.get_costate_at_distance(position)
        
        # Reference values
        v_ref = ref['v'][0]
        soc_ref = ref['soc'][0]
        
        # Velocity error
        v_error = v_ref - velocity
        
        # SOC error (positive = above reference)
        soc_error = soc - soc_ref
        
        # Decision logic based on co-state and adaptive factor
        threshold = costate_info['deploy_threshold'] * self.equivalence_factor
        
        P_ers = 0.0
        throttle = 0.0
        brake = 0.0
        
        if v_error > 0:  # Need to speed up
            throttle = np.clip(0.3 + 0.7 * (v_error / 20.0), 0.3, 1.0)
            
            # Deploy ERS if above threshold and have charge
            if soc > threshold and soc > self.ers.min_soc + 0.05:
                P_ers = self.ers.max_deployment_power * min(1.0, v_error / 10.0)
                
        else:  # Need to slow down
            brake = np.clip(-v_error / 30.0, 0.0, 1.0)
            
            # Harvest if below capacity
            if soc < self.ers.max_soc - 0.05 and brake > 0.1:
                P_ers = -self.ers.max_recovery_power * min(1.0, brake)
        
        return np.array([P_ers, throttle, brake]), 'costate_fallback'
    
    def _simple_fallback(self, velocity: float, soc: float, ref: Dict) -> np.ndarray:
        """Simple rule-based fallback when no co-state info available"""
        v_ref = ref['v'][0]
        v_error = v_ref - velocity
        
        if v_error > 5:
            return np.array([60000.0, 0.8, 0.0])  # Accelerate with ERS
        elif v_error < -5:
            return np.array([-60000.0, 0.0, 0.5])  # Brake and harvest
        else:
            return np.array([0.0, 0.5, 0.0])  # Cruise
    
    def _get_horizon_track_data(self, position: float) -> Dict:
        """Get track parameters for prediction horizon"""
        
        track_data = self.track.track_data
        N = self.N
        
        gradient = np.zeros(N)
        radius = np.zeros(N)
        v_limit = np.zeros(N + 1)
        
        has_v_max = (track_data.v_max_corner is not None and 
                     len(track_data.v_max_corner) > 0 and
                     np.max(track_data.v_max_corner) > 1.0)
        
        for i in range(N + 1):
            s = (position + i * self.ds) % self.track.total_length
            segment = self.track.get_segment_at_distance(s)
            
            if i < N:
                gradient[i] = segment.gradient
                radius[i] = max(segment.radius, 15.0)
            
            if has_v_max:
                idx = int(s / self.track.ds) % len(track_data.v_max_corner)
                v_limit[i] = track_data.v_max_corner[idx]
            else:
                r = max(segment.radius, 15.0)
                mu = self.vehicle.vehicle.mu_lateral
                g = self.vehicle.vehicle.g
                v_limit[i] = min(100.0, np.sqrt(mu * g * r) * 1.3)
            
            v_limit[i] = np.clip(v_limit[i], 20.0, 100.0)
        
        return {'gradient': gradient, 'radius': radius, 'v_limit': v_limit}
    
    def _get_horizon_reference(self, position: float) -> Dict:
        """Get reference trajectory for horizon"""
        
        N = self.N
        v_ref = np.zeros(N + 1)
        soc_ref = np.zeros(N + 1)
        P_ers_ref = np.zeros(N)
        throttle_ref = np.zeros(N)
        brake_ref = np.zeros(N)
        
        if self.reference is None:
            v_ref[:] = 50.0
            soc_ref[:] = 0.5
            throttle_ref[:] = 0.7
        else:
            for i in range(N + 1):
                s = (position + i * self.ds) % self.track.total_length
                ref = self.reference.get_reference_at_distance(s)
                v_ref[i] = ref['v_ref']
                soc_ref[i] = ref['soc_ref']
                if i < N:
                    P_ers_ref[i] = ref['P_ers_ref']
                    throttle_ref[i] = max(ref['throttle_ref'], 0.3)
                    brake_ref[i] = ref['brake_ref']
        
        return {
            'v': v_ref, 'soc': soc_ref,
            'P_ers': P_ers_ref, 'throttle': throttle_ref, 'brake': brake_ref
        }
    
    def _get_horizon_costates(self, position: float) -> np.ndarray:
        """Get co-state profile for horizon"""
        
        N = self.N
        lambda_soc = np.zeros(N + 1)
        
        if self.costate_profile is None:
            return lambda_soc
        
        for i in range(N + 1):
            s = (position + i * self.ds) % self.track.total_length
            costate = self.costate_profile.get_costate_at_distance(s)
            lambda_soc[i] = costate['lambda_soc']
        
        return lambda_soc
    
    def _warm_start(self, ref: Dict, current_v: float, current_soc: float, position: float):
        """Initialize optimization with good starting point"""
        
        N = self.N
        cfg = self.config
        
        # Always start with reference trajectory as base
        V_init = ref['v'].copy()
        V_init[0] = current_v
        
        SOC_init = ref['soc'].copy()
        SOC_init[0] = current_soc
        
        P_ERS_init = ref['P_ers'].copy()
        THROTTLE_init = np.clip(ref['throttle'], 0.1, 1.0)
        BRAKE_init = ref['brake'].copy()
        
        # Try to improve with shifted previous solution if available
        if self._prev_solution is not None:
            delta_pos = position - self._prev_position
            
            # Only use previous solution if we moved forward by a reasonable amount
            if 0 < delta_pos < self.ds * 5:
                shift = max(1, int(delta_pos / self.ds))
                prev_N = len(self._prev_solution['V']) - 1
                
                if cfg.debug:
                    print(f"   [Warm] Using shifted prev solution: shift={shift}, delta={delta_pos:.1f}m")
                
                # Shift and pad previous solution
                if shift < prev_N:
                    # For states (N+1 points)
                    V_shifted = self._prev_solution['V'][shift:]
                    SOC_shifted = self._prev_solution['SOC'][shift:]
                    
                    # Pad with last values to reach N+1 points
                    V_padded = np.pad(V_shifted, (0, max(0, N + 1 - len(V_shifted))), 
                                      mode='edge')[:N+1]
                    SOC_padded = np.pad(SOC_shifted, (0, max(0, N + 1 - len(SOC_shifted))), 
                                        mode='edge')[:N+1]
                    
                    # Blend with reference (keep first few from current state)
                    blend_len = min(5, N)
                    for i in range(1, N + 1):
                        if i < blend_len:
                            alpha = i / blend_len
                            V_init[i] = (1 - alpha) * current_v + alpha * V_padded[i]
                            SOC_init[i] = (1 - alpha) * current_soc + alpha * SOC_padded[i]
                        else:
                            V_init[i] = V_padded[i]
                            SOC_init[i] = SOC_padded[i]
                    
                    # For controls (N points)
                    P_shifted = self._prev_solution['P_ERS'][min(shift, len(self._prev_solution['P_ERS'])-1):]
                    T_shifted = self._prev_solution['THROTTLE'][min(shift, len(self._prev_solution['THROTTLE'])-1):]
                    B_shifted = self._prev_solution['BRAKE'][min(shift, len(self._prev_solution['BRAKE'])-1):]
                    
                    P_ERS_init = np.pad(P_shifted, (0, max(0, N - len(P_shifted))), mode='edge')[:N]
                    THROTTLE_init = np.pad(T_shifted, (0, max(0, N - len(T_shifted))), mode='edge')[:N]
                    BRAKE_init = np.pad(B_shifted, (0, max(0, N - len(B_shifted))), mode='edge')[:N]
        
        # Ensure constraints are satisfied in initial guess
        V_init = np.clip(V_init, 10.0, 100.0)
        SOC_init = np.clip(SOC_init, self.ers.min_soc, self.ers.max_soc)
        P_ERS_init = np.clip(P_ERS_init, -self.ers.max_recovery_power, self.ers.max_deployment_power)
        THROTTLE_init = np.clip(THROTTLE_init, 0.0, 1.0)
        BRAKE_init = np.clip(BRAKE_init, 0.0, 1.0)
        
        self.opti.set_initial(self.vars['V'], V_init)
        self.opti.set_initial(self.vars['SOC'], SOC_init)
        self.opti.set_initial(self.vars['P_ERS'], P_ERS_init)
        self.opti.set_initial(self.vars['THROTTLE'], THROTTLE_init)
        self.opti.set_initial(self.vars['BRAKE'], BRAKE_init)
    
    def get_statistics(self) -> Dict:
        """Get controller performance statistics"""
        total = self.solve_count + self.fail_count
        return {
            'solve_count': self.solve_count,
            'fail_count': self.fail_count,
            'success_rate': self.solve_count / max(total, 1) * 100,
            'avg_solve_time': self.total_solve_time / max(self.solve_count, 1),
            'current_equiv_factor': self.equivalence_factor,
        }
    
    def get_control(self, state: np.ndarray, track_params: Dict) -> np.ndarray:
        """
        Simple control interface (no info returned).
        
        Args:
            state: [position, velocity, soc] array
            track_params: Dict with 'gradient' and 'radius'
            
        Returns:
            control: [P_ers, throttle, brake]
        """
        control, _ = self.solve_mpc_step(state, state[0])
        return control
    
    def reset(self):
        """Reset controller state for new lap/session"""
        self._prev_solution = None
        self._prev_position = 0.0
        self.equivalence_factor = 1.0
        self.soc_error_integral = 0.0
        self.solve_count = 0
        self.fail_count = 0
        self.total_solve_time = 0.0