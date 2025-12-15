"""
Simplified vehicle dynamics in both time and spatial domains
s
References:
- TUMFTM laptime-simulation (car.py, car_hybrid.py): https://github.com/TUMFTM/laptime-simulation
- Heilmeier et al. "Application of Monte Carlo Methods" (2020)
"""
import casadi as ca
from typing import Dict, Optional

from config import VehicleConfig, ERSConfig, TireParameters

class VehicleDynamicsModel:
    """
    Simplified vehicle dynamics model for optimization
    
    Provides dynamics in both:
    - Time domain: dx/dt (for MPC, simulation)
    - Spatial domain: dx/ds (for offline NLP)
    """
    
    def __init__(self, 
                 vehicle_params: VehicleConfig,
                 ers_config : ERSConfig,
                 tire_params: Optional[TireParameters] = None):

        self.vehicle = vehicle_params
        self.ers = ers_config
        self.tires = tire_params or TireParameters()
        
        # Cache for CasADi functions
        self._dynamics_time_func = None
        self._dynamics_spatial_func = None
        
        # Precompute static load distribution
        self._precompute_load_transfer_coefficients()
        
    def _precompute_load_transfer_coefficients(self):
        """Precompute coefficients for load transfer calculations"""
        veh = self.vehicle
        g = veh.g
        m = veh.mass
        L = veh.wheelbase
        
        # Static load on each axle
        self.static_load_front = m * g * veh.lr / L  # Total front axle
        self.static_load_rear = m * g * veh.lf / L   # Total rear axle
        
        # Longitudinal load transfer coefficient
        # ΔF_z = m × a_x × h_cog / L
        self.long_transfer_coef = m * veh.h_cog / L
        
        # Lateral load transfer coefficients (per axle)
        # ΔF_z_f = m × (lr/L) × a_y × h_cog / sf
        self.lat_transfer_coef_f = m * (veh.lr / L) * veh.h_cog / veh.sf
        self.lat_transfer_coef_r = m * (veh.lf / L) * veh.h_cog / veh.sr
        
    def compute_tire_loads(self, velocity: float, a_x: float, a_y: float) -> Dict:
        """
        Compute individual tire loads including load transfer.
        
        Returns loads for FL, FR, RL, RR wheels.
        
        Convention:
        - a_x > 0: accelerating (load transfers to rear)
        - a_y > 0: turning left (load transfers to right)
        """
        veh = self.vehicle
        
        # Aerodynamic downforce
        q = 0.5 * veh.rho_air * velocity**2
        F_down_f = q * veh.c_z_a_f  # Front downforce
        F_down_r = q * veh.c_z_a_r  # Rear downforce
        
        # Static + aero load per axle
        F_z_f_static = self.static_load_front + F_down_f
        F_z_r_static = self.static_load_rear + F_down_r
        
        # Longitudinal load transfer
        delta_F_z_long = self.long_transfer_coef * a_x
        F_z_f_total = F_z_f_static - delta_F_z_long  # Front loses load under accel
        F_z_r_total = F_z_r_static + delta_F_z_long  # Rear gains load under accel
        
        # Lateral load transfer (assuming 50/50 left/right split as baseline)
        delta_F_z_lat_f = self.lat_transfer_coef_f * abs(a_y)
        delta_F_z_lat_r = self.lat_transfer_coef_r * abs(a_y)
        
        # Individual wheel loads
        # Sign of a_y determines which side gains load
        if a_y >= 0:  # Turning left, right side gains
            F_z_fl = 0.5 * F_z_f_total - delta_F_z_lat_f
            F_z_fr = 0.5 * F_z_f_total + delta_F_z_lat_f
            F_z_rl = 0.5 * F_z_r_total - delta_F_z_lat_r
            F_z_rr = 0.5 * F_z_r_total + delta_F_z_lat_r
        else:  # Turning right, left side gains
            F_z_fl = 0.5 * F_z_f_total + delta_F_z_lat_f
            F_z_fr = 0.5 * F_z_f_total - delta_F_z_lat_f
            F_z_rl = 0.5 * F_z_r_total + delta_F_z_lat_r
            F_z_rr = 0.5 * F_z_r_total - delta_F_z_lat_r
        
        # Clamp to minimum (wheel can't push up)
        F_z_fl = max(F_z_fl, 50.0)
        F_z_fr = max(F_z_fr, 50.0)
        F_z_rl = max(F_z_rl, 50.0)
        F_z_rr = max(F_z_rr, 50.0)
        
        return {
            'F_z_fl': F_z_fl, 'F_z_fr': F_z_fr,
            'F_z_rl': F_z_rl, 'F_z_rr': F_z_rr,
            'F_z_f_total': F_z_fl + F_z_fr,
            'F_z_r_total': F_z_rl + F_z_rr,
            'F_z_total': F_z_fl + F_z_fr + F_z_rl + F_z_rr,
        }
    
    def compute_tire_force_potential(self, F_z: float, 
                                      mu_0: float, dmu_dfz: float) -> float:
        """
        Compute maximum tire force with load-dependent friction.
        
        F_max = μ(F_z) × F_z
        μ(F_z) = μ_0 + dμ/dF_z × (F_z - F_z0)
        """
        fz_0 = self.tires.fz_0
        mu = mu_0 + dmu_dfz * (F_z - fz_0)
        mu = max(mu, 0.5)  # Minimum friction
        return mu * F_z
    
    def compute_combined_grip(self, F_z_f: float, F_z_r: float,
                               F_y_f: float, F_y_r: float,
                               for_acceleration: bool = True) -> float:
        """
        Compute available longitudinal force using friction circle.
        
        For each axle:
        F_x² + F_y² ≤ F_max²
        F_x_available = sqrt(F_max² - F_y²)
        
        For RWD (F1), only rear axle contributes to traction.
        All wheels contribute to braking.
        """
        tires = self.tires
        
        # Front axle force potential
        F_x_pot_f = self.compute_tire_force_potential(
            F_z_f, tires.mux_f, tires.dmux_dfz_f)
        F_y_pot_f = self.compute_tire_force_potential(
            F_z_f, tires.muy_f, tires.dmuy_dfz_f)
        
        # Rear axle force potential
        F_x_pot_r = self.compute_tire_force_potential(
            F_z_r, tires.mux_r, tires.dmux_dfz_r)
        F_y_pot_r = self.compute_tire_force_potential(
            F_z_r, tires.muy_r, tires.dmuy_dfz_r)
        
        # Friction circle: remaining longitudinal potential
        exp = tires.tire_model_exp
        
        # Front axle
        ratio_f = min(abs(F_y_f) / F_y_pot_f, 0.99)
        radicand_f = 1 - ratio_f ** exp
        F_x_avail_f = F_x_pot_f * (radicand_f ** (1/exp))
        
        # Rear axle
        ratio_r = min(abs(F_y_r) / F_y_pot_r, 0.99)
        radicand_r = 1 - ratio_r ** exp
        F_x_avail_r = F_x_pot_r * (radicand_r ** (1/exp))
        
        if for_acceleration:
            # RWD: only rear contributes to traction
            return F_x_avail_r
        else:
            # Braking: all wheels contribute
            return F_x_avail_f + F_x_avail_r
    
    def create_time_domain_dynamics(self) -> ca.Function:
        """
        Create CasADi function for time-domain dynamics.
        
        State: x = [s, v, soc]
        Control: u = [P_ers, throttle, brake]
        Parameters: p = [gradient, radius]
        
        Returns: dx/dt
        """
        
        if self._dynamics_time_func is not None:
            return self._dynamics_time_func
        
        # === Symbolic Variables ===
        # States
        s = ca.MX.sym('s')          # Position (m)
        v = ca.MX.sym('v')          # Velocity (m/s)
        soc = ca.MX.sym('soc')      # State of charge (0-1)
        
        # Controls
        P_ers = ca.MX.sym('P_ers')      # ERS power (W)
        throttle = ca.MX.sym('throttle') # Throttle (0-1)
        brake = ca.MX.sym('brake')       # Brake (0-1)
        
        # Parameters
        gradient = ca.MX.sym('gradient')  # Road gradient (rad)
        radius = ca.MX.sym('radius')      # Corner radius (m)
        
        # Vehicle parameters
        veh = self.vehicle
        ers = self.ers
        
        # =====================================================================
        # REGULATION-SPECIFIC ERS LIMITS
        # =====================================================================
        
        # MGU-K peak power
        P_k_max_config = ers.max_deployment_power
        
        if ers.regulation_year >= 2026:
            # --- 2026 REGULATIONS (Variable Power Limit) ---
            # Power tapers off at high speed to prevent "infinite" energy deployment
            
            # Convert v to km/h for the formula
            v_kph = v * 3.6
            
            # 1. Base limit (350 kW or whatever is in config)
            p_limit_base = P_k_max_config
            
            # 2. Linear drop 290-340 kph: P(kW) = 1800 - 5*v
            #    (1800 - 5*290 = 350kW)  -> (1800 - 5*340 = 100kW)
            p_limit_taper1 = (1800.0 - 5.0 * v_kph) * 1000.0
            
            # 3. Sharp drop 340-345 kph: P(kW) = 6900 - 20*v
            #    (6900 - 20*340 = 100kW) -> (6900 - 20*345 = 0kW)
            p_limit_taper2 = (6900.0 - 20.0 * v_kph) * 1000.0
            
            # Apply the continuous min() of all curves
            # We clip at 0 to ensure this "ceiling" doesn't become negative
            P_ers_deploy_limit = ca.fmax(0, ca.fmin(p_limit_base, ca.fmin(p_limit_taper1, p_limit_taper2)))
            
        else:
            # --- 2025 REGULATIONS (Constant Power Limit) ---
            # The MGU-K can provide max power at any speed (RPM limits handled elsewhere/ignored simplified)
            P_ers_deploy_limit = P_k_max_config

        # Apply the Calculated Limit to the Control Input
        # P_ers > 0 is deployment. We enforce: P_ers <= P_ers_deploy_limit
        # (Recovery limits are usually constant 120kW/350kW, handled below)
        
        P_ers_deploy_cmd = ca.fmax(P_ers, 0)
        P_ers_deploy_limited = ca.fmin(P_ers_deploy_cmd, P_ers_deploy_limit)
        
        P_ers_harvest_cmd = ca.fmin(P_ers, 0)
        
        # Check recovery limit (usually symmetric or specified)
        P_k_rec_max = ers.max_recovery_power 
        # P_ers_harvest_cmd is negative, so we limit it to be >= -P_k_rec_max
        P_ers_harvest_limited = ca.fmax(P_ers_harvest_cmd, -P_k_rec_max)
        
        # Recombine for the actual power applied to the wheels
        P_ers_actual = P_ers_deploy_limited + P_ers_harvest_limited

        # =====================================================================
        # PHYSICS MODEL
        # =====================================================================
        
        # Aerodynamic Forces
        q = 0.5 * veh.rho_air * v**2
        F_drag = q * veh.c_w_a
        F_downforce = q * (veh.c_z_a_f + veh.c_z_a_r)
        
        # Normal force
        F_normal = veh.mass * veh.g * ca.cos(gradient) + F_downforce
        
        # Rolling resistance
        F_roll = veh.f_roll * F_normal
        
        # Gravity component
        F_gravity = veh.mass * veh.g * ca.sin(gradient)
        
        # Lateral force from cornering
        safe_radius = ca.fmax(radius, 15.0)
        a_lat = v**2 / safe_radius
        F_lat = veh.mass * a_lat

        # Grip available for longitudinal (simplified friction circle)
        mu_avg = 0.5 * (self.tires.mux_f + self.tires.mux_r)
        F_grip_total = mu_avg * F_normal
        
        # Remaining grip after lateral
        F_grip_long_sq = F_grip_total**2 - F_lat**2
        F_grip_long = ca.sqrt(ca.fmax(F_grip_long_sq, 100.0))
        
        # Propulsion (ICE + ERS)
        # Note: ICE power also changes by year, but that's handled by veh.pow_max_ice value
        P_ice = throttle * veh.pow_max_ice
        P_total = P_ice + ca.fmax(P_ers_actual, 0) # Only positive ERS adds to propulsion
        
        v_safe = ca.fmax(v, 5.0)
        F_prop_request = P_total / v_safe
        F_prop = ca.fmin(F_prop_request, F_grip_long)
        
        # Braking (mechanical + regen)
        F_brake_mech = brake * veh.max_brake_force
        # Regen force comes from the negative part of P_ers_actual
        F_brake_regen = -ca.fmin(P_ers_actual, 0) / v_safe
        F_brake_total = ca.fmin(F_brake_mech + F_brake_regen, F_grip_long)
        
        # Equations of motion
        ds_dt = v
        
        F_net = F_prop - F_drag - F_roll - F_gravity - F_brake_total
        dv_dt = F_net / veh.mass
        
        # =====================================================================
        # BATTERY DYNAMICS (SoC)
        # =====================================================================
        
        # Smooth efficiency transition between deployment and harvest
        sigma = 0.5 * (1 + ca.tanh(P_ers_actual / 1000.0))
        eta_d = ers.deployment_efficiency
        eta_r = ers.recovery_efficiency
        
        # Power leaving/entering the battery (internal)
        P_battery = sigma * (P_ers_actual / eta_d) + (1 - sigma) * (P_ers_actual * eta_r)
        
        dsoc_dt = -P_battery / ers.battery_capacity
        
        # Pack into function
        x = ca.vertcat(s, v, soc)
        u = ca.vertcat(P_ers, throttle, brake)
        p = ca.vertcat(gradient, radius)
        x_dot = ca.vertcat(ds_dt, dv_dt, dsoc_dt)
        
        self._dynamics_time_func = ca.Function(
            'dynamics_time', [x, u, p], [x_dot],
            ['x', 'u', 'p'], ['x_dot']
        )
        
        return self._dynamics_time_func
    
    def create_spatial_domain_dynamics(self) -> ca.Function:
        """
        Create CasADi function for spatial-domain dynamics.
        
        State: x = [v, soc]
        Control: u = [P_ers, throttle, brake]
        Parameters: p = [gradient, radius]
        
        Returns: [dx/ds, dt/ds]
        
        This is used for the offline NLP when we minimize:
        T = ∫(1/v)ds = ∫(dt/ds)ds
        """
        
        if self._dynamics_spatial_func is not None:
            return self._dynamics_spatial_func
        
        # Get time-domain dynamics
        time_dynamics = self.create_time_domain_dynamics()
        
        # States (spatial domain doesn't need position s)
        v = ca.MX.sym('v')
        soc = ca.MX.sym('soc')
        
        # Controls
        P_ers = ca.MX.sym('P_ers')
        throttle = ca.MX.sym('throttle')
        brake = ca.MX.sym('brake')
        
        # Parameters  
        gradient = ca.MX.sym('gradient')
        radius = ca.MX.sym('radius')
        
        # Call time-domain dynamics
        x_time = ca.vertcat(0, v, soc)  # s = 0 (dummy, not used)
        u = ca.vertcat(P_ers, throttle, brake)
        p = ca.vertcat(gradient, radius)
        
        x_dot = time_dynamics(x_time, u, p)
        
        # Extract derivatives
        # ds_dt = v, dv_dt, dsoc_dt from time domain
        ds_dt = x_dot[0]  # = v
        dv_dt = x_dot[1]
        dsoc_dt = x_dot[2]
        
        # Convert to spatial domain: dx/ds = (dx/dt) / (ds/dt)
        v_safe = ca.fmax(v, 5.0)
        
        dv_ds = dv_dt / v_safe      # dv/ds = (dv/dt) / v
        dsoc_ds = dsoc_dt / v_safe  # dsoc/ds = (dsoc/dt) / v
        dt_ds = 1.0 / v_safe        # dt/ds = 1/v (for time integration)
        
        # Outputs
        x_spatial = ca.vertcat(v, soc)
        dx_ds = ca.vertcat(dv_ds, dsoc_ds)
        
        self._dynamics_spatial_func = ca.Function(
            'dynamics_spatial',
            [x_spatial, u, p], [dx_ds, dt_ds],
            ['x', 'u', 'p'], ['dx_ds', 'dt_ds']
        )
        
        return self._dynamics_spatial_func
    
    def get_constraints(self) -> Dict:
        """Return system constraints"""
        return {
            # ERS power limits
            'P_ers_min': -self.ers.max_recovery_power,
            'P_ers_max': self.ers.max_deployment_power,
            
            # SOC limits
            'soc_min': self.ers.min_soc,
            'soc_max': self.ers.max_soc,
            
            # Velocity limits
            'v_min': 15.0,   # m/s (~54 km/h)
            'v_max': 110.0,  # m/s (~360 km/h)
            
            # Control limits
            'throttle_min': 0.0,
            'throttle_max': 1.0,
            'brake_min': 0.0,
            'brake_max': 1.0,
        }