import casadi as ca
from typing import Dict

from config import VehicleConfig, ERSConfig


class VehicleDynamicsModel:
    """Simplified vehicle dynamics model for optimization"""
    
    def __init__(self, vehicle_config: VehicleConfig, ers_config: ERSConfig):
        self.vehicle = vehicle_config
        self.ers = ers_config
        
    def create_casadi_function(self) -> ca.Function:
        """Create CasADi function for dynamics evaluation"""
        # States
        s = ca.MX.sym('s')      # position along track
        v = ca.MX.sym('v')      # velocity (m/s)
        soc = ca.MX.sym('soc')  # battery state of charge (0..1)
        
        # Control variables
        P_ers = ca.MX.sym('P_ers')  # W, ERS power (positive = deployment) (negative = harvest)
        throttle = ca.MX.sym('throttle')  # throttle position [0, 1]
        brake = ca.MX.sym('brake')  # brake force [0, 1]
        
        # Parameters (track-dependent)
        gradient = ca.MX.sym('gradient')
        radius = ca.MX.sym('radius')
                
        # Physical constants
        rho = 1.225  # Air density
        A = self.vehicle.frontal_area
        
        # Aerodynamic forces
        F_drag      = 0.5 * rho * self.vehicle.cd * A * v**2
        F_downforce = 0.5 * rho * self.vehicle.cl * A * v**2
        F_gravity   = self.vehicle.mass * 9.81 * ca.sin(gradient)
        
        # Normal force includes downforce
        F_normal = self.vehicle.mass * 9.81 * ca.cos(gradient) + F_downforce
        F_rolling = self.vehicle.cr * F_normal
        
        # Maximum traction force (simplified)
        mu = 1.8  # Peak tire coefficient (dry slicks)
        F_total_max = mu * F_normal # Total tyre force limit from friction
        
        # Lateral acceleration from curvature (avoid div by zero / inf)
        # Treat radius <= 0 or very large as "straight"
        safe_radius = ca.fmax(radius, 1.0)
        a_lat = v**2 / safe_radius             # [m/s^2]
        F_lat = self.vehicle.mass * a_lat      # lateral tyre force

        # Remaining longitudinal force from friction circle
        # Protected against Sqrt(0) singularity
        # We clamp the "grip remaining" so it never drops below a tiny epsilon (1.0 N)
        # This prevents the gradient from exploding to Infinity.
        grip_remaining_sq = ca.fmax(F_total_max**2 - F_lat**2, 1.0)
        F_long_abs_max = ca.sqrt(grip_remaining_sq)

        # Longitudinal traction / braking limited by remaining grip
        P_ice = throttle * self.vehicle.max_ice_power
        P_total = P_ice + ca.fmax(0, P_ers)
        
        # Protect against v=0 division
        F_traction_requested = P_total / ca.fmax(v, 1.0)
        F_traction = ca.fmin(F_traction_requested, F_long_abs_max)

        F_brake_requested = brake * self.vehicle.max_brake_force
        F_brake = ca.fmin(F_brake_requested, F_long_abs_max)

        # --- Dynamics ---
        dv_dt = (F_traction - F_drag - F_rolling - F_gravity - F_brake) / self.vehicle.mass
        ds_dt = v
        
        # Battery dynamics: 
        # P_ers > 0: Deployment (Energy flows OUT of battery)
        # P_ers < 0: Recovery (Energy flows INTO battery)
        
        P_internal = ca.if_else(
            P_ers < 0,  # Recovery mode
            P_ers * self.ers.recovery_efficiency, # e.g. -100kW * 0.85 = -85kW (Internal gain)
            P_ers / self.ers.deployment_efficiency # e.g. 100kW / 0.95 = 105kW (Internal cost)
        )
        dsoc_dt = P_internal / self.ers.battery_capacity
        
        # Create function
        x = ca.vertcat(s, v, soc)
        u = ca.vertcat(P_ers, throttle, brake)
        p = ca.vertcat(gradient, radius)
        x_dot = ca.vertcat(ds_dt, dv_dt, dsoc_dt)
        
        dynamics_func = ca.Function(
            'dynamics', [x, u, p], [x_dot],
            ['x', 'u', 'p'], ['x_dot']
        )
        
        return dynamics_func
    
    def get_constraints(self) -> Dict:
        """Return system constraints"""
        return {
            'P_ers_min': -self.ers.max_recovery_power,
            'P_ers_max': self.ers.max_deployment_power,
            'soc_min': self.ers.min_soc,
            'soc_max': self.ers.max_soc,
            'v_min': 10,   # m/s (36 km/h)
            'v_max': 100,  # m/s (360 km/h)
            'throttle_min': 0,
            'throttle_max': 1,
            'brake_min': 0,
            'brake_max': 1,
        }
        