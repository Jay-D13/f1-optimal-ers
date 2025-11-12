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
        
        # --- Dynamics ---
        # Lateral acceleration limit (simplified)
        # v^2/r <= mu_lat * g * (1 + downforce_factor)
        a_lat_max = 4.5 * 9.81  # g-limit; 4.5g lateral (with downforce)

        # Corner speed limit from lateral grip
        v_max_corner = ca.sqrt(a_lat_max * radius)

        # Limit velocity based on corner radius
        # For straights (large radius), this doesn't constrain
        # For corners, this enforces realistic speeds
        v_eff = ca.fmin(v, v_max_corner) # v constrained
        
        rho = 1.225
        A   = self.vehicle.frontal_area
        
        # Aerodynamic forces
        F_drag      = 0.5 * rho * self.vehicle.cd * A * v_eff**2
        F_downforce = 0.5 * rho * self.vehicle.cl * A * v_eff**2
        
        # Normal force includes downforce
        F_normal = self.vehicle.mass * 9.81 * ca.cos(gradient) + F_downforce
        
        # Rolling resistance depends on normal force
        F_rolling = self.vehicle.cr * F_normal
        F_gravity = self.vehicle.mass * 9.81 * ca.sin(gradient)
        
        # Maximum traction force (simplified)
        mu = 1.8  # Peak tire coefficient (dry slicks)
        F_traction_max = mu * F_normal
        
        # Actual traction force (limited by tire grip)
        P_ice = throttle * self.vehicle.max_ice_power
        P_total = P_ice + ca.fmax(0, P_ers)
        F_traction_requested = P_total / ca.fmax(v, 1.0)
        F_traction = ca.fmin(F_traction_requested, F_traction_max)
        
        # Braking force (also limited by tire grip and includes aero)
        F_brake_max = mu * F_normal
        F_brake_requested = brake * self.vehicle.max_brake_force
        F_brake = ca.fmin(F_brake_requested, F_brake_max)
        
        # Dynamics
        dv_dt = (F_traction - F_drag - F_rolling - F_gravity - F_brake) / self.vehicle.mass
        ds_dt = v_eff  # Use effective velocity for position change
        
        # Battery dynamics: P_ers >0 deployment, <0 recovery
        # Slightly realistic recovery with diminishing returns at high power
        P_recovery_actual = ca.if_else(
            P_ers < 0,  # Recovery mode
            P_ers * self.ers.recovery_efficiency * (1 - 0.1 * ca.fabs(P_ers / self.ers.max_recovery_power)),
            -P_ers / self.ers.deployment_efficiency
        )
        dsoc_dt = P_recovery_actual / self.ers.battery_capacity
        
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
            'v_min': 0,
            'v_max': 100,  # m/s (~360 km/h)
            'throttle_min': 0,
            'throttle_max': 1,
            'brake_min': 0,
            'brake_max': 1,
        }
        