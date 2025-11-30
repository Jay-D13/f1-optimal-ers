import casadi as ca
from typing import Dict

from config import VehicleConfig, ERSConfig


class VehicleDynamicsModel:
    """Simplified vehicle dynamics model for optimization"""
    
    def __init__(self, vehicle_config: VehicleConfig, ers_config: ERSConfig):
        self.vehicle = vehicle_config
        self.ers = ers_config
        
        # cache for CasADi functions
        self._dynamics_time_func = None
        self._dynamics_spatial_func = None
        
    def create_time_domain_dynamics(self) -> ca.Function:
        """
        Create CasADi functions for time-domain dynamics (dx/dt).
        
        State: x = [s, v, soc]
            s: position along track (m)
            v: velocity (m/s)
            soc: battery state of charge (0-1)
        
        Control: u = [P_ers, throttle, brake]
            P_ers: ERS power (W), positive = deployment
            throttle: throttle position (0-1)
            brake: brake force fraction (0-1)
        
        Parameters: p = [gradient, radius]
            gradient: road gradient (radians)
            radius: corner radius (m)
        
        Returns:
            CasADi Function: dynamics(x, u, p) -> x_dot
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
        
        # === Physical Constants ===
        g = self.vehicle.g
        rho = self.vehicle.rho_air
        A = self.vehicle.frontal_area
        m = self.vehicle.mass
        
        # === Aerodynamic Forces ===
        q = 0.5 * rho * v**2  # Dynamic pressure
        
        F_drag = q * self.vehicle.cd * A
        F_downforce = q * self.vehicle.cl * A
        
        # === Gravitational Force Component ===
        F_gravity = m * g * ca.sin(gradient)
        
        # === Normal Force (weight + downforce) ===
        F_normal = m * g * ca.cos(gradient) + F_downforce
        
        # === Rolling Resistance ===
        F_rolling = self.vehicle.cr * F_normal
        
        # === Friction Circle ===
        # Total available tire force
        mu = self.vehicle.mu_lateral
        F_tire_max = mu * F_normal

        # Lateral force from cornering (v²/R)
        safe_radius = ca.fmax(radius, 10.0)  # Prevent div by zero
        a_lateral = v**2 / safe_radius
        F_lateral = m * a_lateral

        # Remaining force for longitudinal (friction circle)
        # F_long² + F_lat² <= F_max²
        # F_long_available = sqrt(F_max² - F_lat²)
        # Add extra margin to prevent negative values under sqrt
        grip_margin_sq = F_tire_max**2 - F_lateral**2
        # Soft limit to avoid sudden gradient change
        grip_margin_sq_safe = ca.fmax(grip_margin_sq, 100.0)  # Min 10N available force
        F_long_available = ca.sqrt(grip_margin_sq_safe)
        
        # === Powertrain Forces ===
        # ICE power
        P_ice = throttle * self.vehicle.max_ice_power
        
        # Total propulsive power (ICE + ERS deployment)
        # P_ers > 0 means deployment (adds to propulsion)
        P_propulsion = P_ice + ca.fmax(P_ers, 0)
        
        # Traction force (limited by power and grip)
        v_safe = ca.fmax(v, 1.0)  # Prevent div by zero
        F_traction_request = P_propulsion / v_safe
        F_traction = ca.fmin(F_traction_request, F_long_available)
        
        # === Braking Forces ===
        # Mechanical braking
        F_brake_mech = brake * self.vehicle.max_brake_force
        
        # ERS recovery (regenerative braking)
        # P_ers < 0 means recovery (acts as additional braking)
        P_regen = ca.fmin(P_ers, 0)  # Negative or zero
        F_brake_regen = -P_regen / v_safe  # Convert power to force (positive)
        
        # Total braking (limited by grip)
        F_brake_total = ca.fmin(F_brake_mech + F_brake_regen, F_long_available)
        
        # === Equations of Motion ===
        # ds/dt = v
        ds_dt = v
        
        # dv/dt = (F_traction - F_drag - F_rolling - F_gravity - F_brake) / m
        F_net = F_traction - F_drag - F_rolling - F_gravity - F_brake_total
        dv_dt = F_net / m
        
        # === Battery Dynamics ===
        # sign convention:
        # - P_ers > 0: Deployment -> battery LOSES energy -> SOC DECREASES
        # - P_ers < 0: Recovery -> battery GAINS energy -> SOC INCREASES
        #
        # Power at battery terminals (accounting for efficiency):
        # - Deployment: P_battery = P_ers / η_deploy (more internal loss)
        # - Recovery: P_battery = P_ers * η_recover (less recovered)
        
        # P_battery = ca.if_else(
        #     P_ers >= 0,
        #     P_ers / self.ers.deployment_efficiency,   # Deployment: divide
        #     P_ers * self.ers.recovery_efficiency      # Recovery: multiply
        # )
        
        # sigmoid function to blend between the two efficiency modes continuously
        # k_smooth determines sharpness. 0.1 to 1.0 is usually good for Power in Watts.
        sigma = 0.5 * (1 + ca.tanh(P_ers / 1000.0)) 

        eta_d = self.ers.deployment_efficiency
        eta_r = self.ers.recovery_efficiency

        # Smooth blend: when P_ers > 0, sigma -> 1 (divide by eff). 
        # When P_ers < 0, sigma -> 0 (multiply by eff).
        P_battery = sigma * (P_ers / eta_d) + (1 - sigma) * (P_ers * eta_r)
        
        # dsoc/dt = -P_battery / E_capacity
        # Negative because P_battery > 0 means energy leaving battery
        dsoc_dt = -P_battery / self.ers.battery_capacity
        
        # === Create Function ===
        x = ca.vertcat(s, v, soc)
        u = ca.vertcat(P_ers, throttle, brake)
        p = ca.vertcat(gradient, radius)
        x_dot = ca.vertcat(ds_dt, dv_dt, dsoc_dt)
        
        self._dynamics_time_func = ca.Function(
            'dynamics_time',
            [x, u, p], [x_dot],
            ['x', 'u', 'p'], ['x_dot']
        )
        
        return self._dynamics_time_func
    
    def create_spatial_domain_dynamics(self) -> ca.Function:
        """
        Create CasADi function for spatial-domain dynamics (dx/ds).
        
        This formulation uses distance (s) as the independent variable,
        which is natural for lap time optimization:
        - Total time = integral(1/v, ds) from 0 to L
        
        State: x = [v, soc]
            v: velocity (m/s)
            soc: state of charge (0-1)
        
        Control: u = [P_ers, throttle, brake]
        
        Parameters: p = [gradient, radius]
        
        Returns:
            CasADi Function: dynamics(x, u, p) -> [dx_ds, dt_ds]
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
        v_safe = ca.fmax(v, 1.0)
        
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
            'v_min': 10.0,   # m/s (~36 km/h)
            'v_max': 100.0,  # m/s (~360 km/h)
            
            # Control limits
            'throttle_min': 0.0,
            'throttle_max': 1.0,
            'brake_min': 0.0,
            'brake_max': 1.0,
        }
        