"""
Fixed Model Predictive Control for ERS Management

Key fixes:
1. Better v_limit handling when track.v_max_corner not computed
2. Improved warm starting from reference + previous solution
3. More robust dynamics formulation
4. Fallback to reference controls on solver failure
"""
import numpy as np
import casadi as ca
from typing import Optional, Tuple, Dict

from models import VehicleDynamicsModel, F1TrackModel
from config import ERSConfig


class ERSModelPredictiveController:
    """
    MPC for online ERS control with receding horizon
    
    Key features:
    - Re-optimizes based on current measured state
    - Adapts to tire degradation mismatch
    - Corrects for SOC estimation errors
    - Handles fuel mass changes
    """
    
    def __init__(self,
                 vehicle_model: VehicleDynamicsModel,
                 track_model: F1TrackModel,
                 ers_config: ERSConfig,
                 horizon_segments: int = 50,
                 verbose: bool = False):
        self.vehicle = vehicle_model
        self.track = track_model
        self.ers = ers_config
        self.horizon = horizon_segments
        self.verbose = verbose
        
        self.ds = track_model.ds
        
        # Reference trajectory (set externally)
        self.reference_trajectory = None
        
        # Previous solution for warm starting
        self._prev_solution = None
        
        # Build MPC problem
        self._build_mpc_problem()
        
    def set_reference(self, reference_trajectory):
        """Set offline optimal trajectory as reference"""
        self.reference_trajectory = reference_trajectory
        self._prev_solution = None
        
    def _build_mpc_problem(self):
        """Build CasADi optimization problem for MPC"""
        
        N = self.horizon
        opti = ca.Opti()
        
        # Decision variables
        V = opti.variable(N + 1)      # Velocity
        SOC = opti.variable(N + 1)    # State of charge
        P_ERS = opti.variable(N)      # ERS power
        THROTTLE = opti.variable(N)   # Throttle
        BRAKE = opti.variable(N)      # Brake
        
        # Parameters (set at runtime)
        v_init = opti.parameter()     # Initial velocity
        soc_init = opti.parameter()   # Initial SOC
        
        # Track parameters (arrays)
        gradient = opti.parameter(N)
        radius = opti.parameter(N)
        v_limit = opti.parameter(N + 1)
        
        # Reference trajectory (for tracking)
        v_ref = opti.parameter(N + 1)
        soc_ref = opti.parameter(N + 1)
        P_ers_ref = opti.parameter(N)
        throttle_ref = opti.parameter(N)
        
        # Tire degradation factor (1.0 = fresh, 0.8 = 20% degraded)
        tire_factor = opti.parameter()
        
        # Fuel mass (decreases over laps)
        mass = opti.parameter()
        
        # =====================================================
        # OBJECTIVE: Minimize lap time + tracking + smoothness
        # =====================================================
        
        lap_time = 0
        tracking_cost = 0
        smoothness_cost = 0
        
        for k in range(N):
            v_avg = 0.5 * (V[k] + V[k+1])
            v_safe = ca.fmax(v_avg, 5.0)
            lap_time += self.ds / v_safe
            
            # Small penalty to stay near reference (helps convergence)
            tracking_cost += 0.0001 * (V[k] - v_ref[k])**2
            tracking_cost += 0.001 * (SOC[k] - soc_ref[k])**2
        
        # Control smoothness
        for k in range(N - 1):
            smoothness_cost += 1e-8 * (P_ERS[k+1] - P_ERS[k])**2
            smoothness_cost += 0.001 * (THROTTLE[k+1] - THROTTLE[k])**2
            smoothness_cost += 0.001 * (BRAKE[k+1] - BRAKE[k])**2
        
        opti.minimize(lap_time + tracking_cost + smoothness_cost)
        
        # =====================================================
        # DYNAMICS
        # =====================================================
        
        veh = self.vehicle.vehicle
        ers = self.vehicle.ers
        
        for k in range(N):
            gradient_k = gradient[k]
            radius_k = radius[k]
            
            v_k = V[k]
            soc_k = SOC[k]
            P_ers_k = P_ERS[k]
            throttle_k = THROTTLE[k]
            brake_k = BRAKE[k]
            
            # Aerodynamics
            q = 0.5 * veh.rho_air * v_k**2
            F_drag = q * veh.c_w_a
            F_downforce = q * (veh.c_z_a_f + veh.c_z_a_r)
            
            # Normal force (using actual mass parameter)
            F_normal = mass * veh.g * ca.cos(gradient_k) + F_downforce
            F_roll = veh.cr * F_normal
            F_gravity = mass * veh.g * ca.sin(gradient_k)
            
            # Grip limit (scaled by tire degradation)
            mu_eff = veh.mu_longitudinal * tire_factor
            F_grip_max = mu_eff * F_normal
            
            v_safe = ca.fmax(v_k, 5.0)
            
            # Propulsion: ICE + ERS deployment
            P_ice = throttle_k * veh.pow_max_ice
            P_ers_deploy = ca.fmax(P_ers_k, 0)
            P_total = P_ice + P_ers_deploy
            
            F_prop_request = P_total / v_safe
            F_prop = ca.fmin(F_prop_request, F_grip_max)
            
            # Braking: mechanical + regenerative
            F_brake_mech = brake_k * veh.max_brake_force
            P_ers_harvest = ca.fmin(P_ers_k, 0)
            F_brake_regen = -P_ers_harvest / v_safe
            F_brake_total = ca.fmin(F_brake_mech + F_brake_regen, F_grip_max)
            
            # Net force - KEY: F_brake_total is only applied when braking
            # Since we have throttle*brake <= 0.1, they're mostly exclusive
            F_net = F_prop - F_drag - F_roll - F_gravity - F_brake_total
            
            dv_dt = F_net / mass
            dv_ds = dv_dt / v_safe
            
            opti.subject_to(V[k+1] == V[k] + dv_ds * self.ds)
            
            # SOC dynamics
            sigma = 0.5 * (1 + ca.tanh(P_ers_k / 1000.0))
            P_battery = sigma * (P_ers_k / ers.deployment_efficiency) + \
                       (1 - sigma) * (P_ers_k * ers.recovery_efficiency)
            dsoc_ds = -P_battery / (ers.battery_capacity * v_safe)
            
            opti.subject_to(SOC[k+1] == SOC[k] + dsoc_ds * self.ds)
        
        # =====================================================
        # CONSTRAINTS
        # =====================================================
        
        cons = self.vehicle.get_constraints()
        
        # Initial conditions (set by parameters)
        opti.subject_to(V[0] == v_init)
        opti.subject_to(SOC[0] == soc_init)
        
        # State bounds
        for k in range(N + 1):
            opti.subject_to(V[k] >= cons['v_min'])
            opti.subject_to(V[k] <= v_limit[k] * 1.05)  # 5% margin
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
            opti.subject_to(THROTTLE[k] * BRAKE[k] <= 0.1)
        
        # =====================================================
        # SOLVER SETUP
        # =====================================================
        
        opts = {
            'ipopt.max_iter': 200,
            'ipopt.print_level': 0 if not self.verbose else 3,
            'print_time': 0,
            'ipopt.tol': 1e-4,
            'ipopt.acceptable_tol': 1e-3,
            'ipopt.acceptable_iter': 5,
            'ipopt.warm_start_init_point': 'yes',
            'ipopt.warm_start_bound_push': 1e-6,
            'ipopt.warm_start_mult_bound_push': 1e-6,
            'ipopt.mu_init': 1e-4,
            'ipopt.linear_solver': 'mumps',
        }
        
        opti.solver('ipopt', opts)
        
        # Store problem
        self.opti = opti
        self.vars = {
            'V': V, 'SOC': SOC, 'P_ERS': P_ERS,
            'THROTTLE': THROTTLE, 'BRAKE': BRAKE
        }
        self.params = {
            'v_init': v_init, 'soc_init': soc_init,
            'gradient': gradient, 'radius': radius, 'v_limit': v_limit,
            'v_ref': v_ref, 'soc_ref': soc_ref,
            'P_ers_ref': P_ers_ref, 'throttle_ref': throttle_ref,
            'tire_factor': tire_factor, 'mass': mass
        }
        
    def solve_step(self,
                   position: float,
                   velocity: float,
                   soc: float,
                   tire_degradation: float = 1.0,
                   fuel_mass: float = None) -> Tuple[np.ndarray, Dict]:
        """
        Solve one MPC step
        """
        
        if fuel_mass is None:
            fuel_mass = self.vehicle.vehicle.mass
        
        # Get horizon of track ahead
        track_ahead = self._get_horizon_data(position)
        
        # Get reference trajectory for horizon
        ref_ahead = self._get_reference_horizon(position)
        
        # Set parameters
        self.opti.set_value(self.params['v_init'], velocity)
        self.opti.set_value(self.params['soc_init'], soc)
        self.opti.set_value(self.params['gradient'], track_ahead['gradient'])
        self.opti.set_value(self.params['radius'], track_ahead['radius'])
        self.opti.set_value(self.params['v_limit'], track_ahead['v_limit'])
        self.opti.set_value(self.params['v_ref'], ref_ahead['v'])
        self.opti.set_value(self.params['soc_ref'], ref_ahead['soc'])
        self.opti.set_value(self.params['P_ers_ref'], ref_ahead['P_ers'])
        self.opti.set_value(self.params['throttle_ref'], ref_ahead['throttle'])
        self.opti.set_value(self.params['tire_factor'], tire_degradation)
        self.opti.set_value(self.params['mass'], fuel_mass)
        
        # Warm start with reference
        self._warm_start(ref_ahead, velocity, soc)
        
        # Solve
        import time
        start = time.time()
        
        try:
            sol = self.opti.solve()
            status = 'optimal'
            
            P_ers_opt = sol.value(self.vars['P_ERS'])
            throttle_opt = sol.value(self.vars['THROTTLE'])
            brake_opt = sol.value(self.vars['BRAKE'])
            V_opt = sol.value(self.vars['V'])
            SOC_opt = sol.value(self.vars['SOC'])
            
            # Store for warm starting
            self._prev_solution = {
                'V': V_opt, 'SOC': SOC_opt,
                'P_ERS': P_ers_opt, 'THROTTLE': throttle_opt, 'BRAKE': brake_opt
            }
            
        except Exception as e:
            if self.verbose:
                print(f"   MPC warning: {e}")
            status = 'suboptimal'
            
            # Try debug values first
            try:
                P_ers_opt = self.opti.debug.value(self.vars['P_ERS'])
                throttle_opt = self.opti.debug.value(self.vars['THROTTLE'])
                brake_opt = self.opti.debug.value(self.vars['BRAKE'])
            except:
                # Fallback to reference - THIS IS KEY
                P_ers_opt = ref_ahead['P_ers']
                throttle_opt = ref_ahead['throttle']
                brake_opt = ref_ahead['brake']
        
        solve_time = time.time() - start
        
        # Return first control only (receding horizon)
        control = np.array([P_ers_opt[0], throttle_opt[0], brake_opt[0]])
        
        info = {
            'status': status,
            'solve_time': solve_time,
            'horizon_length': self.horizon,
        }
        
        return control, info
    
    def _get_horizon_data(self, position: float) -> Dict:
        """Get track data for horizon ahead"""
        
        track_data = self.track.track_data
        N = self.horizon
        
        gradient = np.zeros(N)
        radius = np.zeros(N)
        v_limit = np.zeros(N + 1)
        
        # Check if v_max_corner is properly computed
        has_v_max = (track_data.v_max_corner is not None and 
                     len(track_data.v_max_corner) > 0 and
                     np.max(track_data.v_max_corner) > 1.0)
        
        for i in range(N + 1):
            s = (position + i * self.ds) % self.track.total_length
            segment = self.track.get_segment_at_distance(s)
            
            if i < N:
                gradient[i] = segment.gradient
                radius[i] = max(segment.radius, 15.0)
            
            # Get speed limit - FIX: compute from radius if not available
            if has_v_max:
                idx = int(s / self.ds) % len(track_data.v_max_corner)
                v_limit[i] = track_data.v_max_corner[idx]
            else:
                # Compute from radius using simplified formula
                r = segment.radius if segment.radius > 0 else 1000.0
                r = max(r, 15.0)
                # v = sqrt(mu * g * R) with downforce boost
                mu = self.vehicle.vehicle.mu_lateral
                g = self.vehicle.vehicle.g
                v_limit[i] = min(100.0, np.sqrt(mu * g * r) * 1.3)  # 30% boost for downforce
            
            # Ensure reasonable bounds
            v_limit[i] = np.clip(v_limit[i], 20.0, 100.0)
        
        return {
            'gradient': gradient,
            'radius': radius,
            'v_limit': v_limit,
        }
    
    def _get_reference_horizon(self, position: float) -> Dict:
        """Get reference trajectory for horizon ahead"""
        
        N = self.horizon
        v_ref = np.zeros(N + 1)
        soc_ref = np.zeros(N + 1)
        P_ers_ref = np.zeros(N)
        throttle_ref = np.zeros(N)
        brake_ref = np.zeros(N)
        
        if self.reference_trajectory is None:
            # No reference, use reasonable defaults
            v_ref[:] = 50.0
            soc_ref[:] = 0.5
            throttle_ref[:] = 0.7  # KEY: Default to 70% throttle
        else:
            ref = self.reference_trajectory
            for i in range(N + 1):
                s = (position + i * self.ds) % self.track.total_length
                ref_data = ref.get_reference_at_distance(s)
                v_ref[i] = ref_data['v_ref']
                soc_ref[i] = ref_data['soc_ref']
                if i < N:
                    P_ers_ref[i] = ref_data['P_ers_ref']
                    throttle_ref[i] = max(ref_data['throttle_ref'], 0.3)  # At least 30%
                    brake_ref[i] = ref_data['brake_ref']
        
        return {
            'v': v_ref, 
            'soc': soc_ref,
            'P_ers': P_ers_ref,
            'throttle': throttle_ref,
            'brake': brake_ref,
        }
    
    def _warm_start(self, ref_ahead: Dict, current_v: float, current_soc: float):
        """Initialize optimization with good starting point"""
        
        N = self.horizon
        
        if self._prev_solution is not None:
            # Shift previous solution
            V_init = np.zeros(N + 1)
            V_init[0] = current_v
            V_init[1:] = self._prev_solution['V'][1:]
            
            SOC_init = np.zeros(N + 1)
            SOC_init[0] = current_soc
            SOC_init[1:] = self._prev_solution['SOC'][1:]
            
            P_ERS_init = np.zeros(N)
            P_ERS_init[:-1] = self._prev_solution['P_ERS'][1:]
            P_ERS_init[-1] = self._prev_solution['P_ERS'][-1]
            
            THROTTLE_init = np.zeros(N)
            THROTTLE_init[:-1] = self._prev_solution['THROTTLE'][1:]
            THROTTLE_init[-1] = self._prev_solution['THROTTLE'][-1]
            
            BRAKE_init = np.zeros(N)
            BRAKE_init[:-1] = self._prev_solution['BRAKE'][1:]
            BRAKE_init[-1] = self._prev_solution['BRAKE'][-1]
        else:
            # Use reference
            V_init = ref_ahead['v'].copy()
            V_init[0] = current_v
            
            SOC_init = ref_ahead['soc'].copy()
            SOC_init[0] = current_soc
            
            P_ERS_init = ref_ahead['P_ers'].copy()
            THROTTLE_init = np.clip(ref_ahead['throttle'], 0.3, 0.95)  # KEY: reasonable throttle
            BRAKE_init = ref_ahead['brake'].copy()
        
        self.opti.set_initial(self.vars['V'], V_init)
        self.opti.set_initial(self.vars['SOC'], SOC_init)
        self.opti.set_initial(self.vars['P_ERS'], P_ERS_init)
        self.opti.set_initial(self.vars['THROTTLE'], THROTTLE_init)
        self.opti.set_initial(self.vars['BRAKE'], BRAKE_init)