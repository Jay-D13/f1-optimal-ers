import numpy as np
from typing import Dict, Tuple
from scipy.interpolate import interp1d

from strategies.base import BaseStrategy
from config import VehicleConfig, ERSConfig
from models import F1TrackModel

class TrackingStrategy(BaseStrategy):
    """
    Physics-Aware Tracking Strategy.
    Uses Feedforward (Inverse Dynamics) to calculate the exact Throttle/Brake 
    needed to match the reference acceleration and overcome drag.
    """
    def __init__(self, 
                 vehicle_config: VehicleConfig, 
                 ers_config: ERSConfig, 
                 track_model: F1TrackModel,
                 reference_profile = None): # Pass the full profile object!
        super().__init__(vehicle_config, ers_config, track_model)
        
        self.profile = reference_profile
        
        if reference_profile is not None:
            # Interpolate Velocity
            self.v_ref_interp = interp1d(
                reference_profile.s, 
                reference_profile.v, 
                kind='linear', fill_value="extrapolate"
            )
            # Interpolate Acceleration (Crucial for Feedforward)
            self.a_ref_interp = interp1d(
                reference_profile.s, 
                reference_profile.a_x, 
                kind='linear', fill_value="extrapolate"
            )
        else:
            self.v_ref_interp = None
            self.a_ref_interp = None
            
        # Gains can be lower now because Feedforward does the heavy lifting
        self.kp_accel = 0.5
        self.kp_brake = 0.5

    @property
    def name(self) -> str:
        return "TrackingStrategy (FF)"

    def get_driver_commands(self, state: np.ndarray, track_info: Dict) -> Tuple[float, float, float]:
        """
        Returns: (throttle, brake, v_target)
        Uses Inverse Dynamics: F_req = m*a_ref + F_drag + F_gravity
        """
        s, v, soc = state
        
        # 1. Get Reference State
        v_target = float(self.v_ref_interp(s))
        a_target = float(self.a_ref_interp(s))
        
        # 2. Feedforward Physics (Inverse Dynamics)
        # Calculate forces resisting the car RIGHT NOW
        mass = self.vehicle.mass
        g = self.vehicle.g
        
        # Aerodynamic Drag: F_drag = 0.5 * rho * Cd * A * v^2
        # (Simplified using your config constant c_w_a)
        f_drag = 0.5 * self.vehicle.rho_air * self.vehicle.c_w_a * v**2
        
        # Rolling Resistance
        # Approximate Normal force (ignoring aero downforce for rolling resistance approx is fine, 
        # or use full formula if you want perfection)
        grad = track_info.get('gradient', 0.0)
        f_roll = self.vehicle.cr * mass * g * np.cos(grad)
        
        # Gravity (Hill Climbing)
        f_grav = mass * g * np.sin(grad)
        
        # Newton's Second Law: F_net = m * a
        # F_powertrain - F_drag - F_roll - F_grav = m * a_target
        # F_powertrain_required = m * a_target + F_drag + F_roll + F_grav
        f_req = (mass * a_target) + f_drag + f_roll + f_grav
        
        # 3. Feedback Correction (PID)
        # Fix small drifts from numerical error
        v_error = v_target - v
        f_correction = 0.0
        
        if v_error > 0:
            f_correction = self.kp_accel * v_error * mass # Convert accel demand to Force
        else:
            f_correction = self.kp_brake * v_error * mass
            
        f_total_cmd = f_req + f_correction
        
        # 4. Map Force to Throttle/Brake
        throttle = 0.0
        brake = 0.0
        
        if f_total_cmd > 0:
            # Acceleration
            # Throttle % ~ Force / Max_Force_at_Current_Speed
            # Power limited: P = F*v -> F_max = P_max / v
            # Grip limited: F_max = mu * F_z (simplified)
            
            # Simple approx: F_max = Power / v
            p_avail = self.vehicle.pow_max_ice
            f_max_engine = p_avail / max(v, 5.0)
            
            throttle = np.clip(f_total_cmd / f_max_engine, 0.0, 1.0)
            
        else:
            # Braking
            # Brake % ~ -Force / Max_Braking_Force
            brake = np.clip(-f_total_cmd / self.vehicle.max_brake_force, 0.0, 1.0)
            
        return throttle, brake, v_target

    def get_control(self, state: np.ndarray, track_info: Dict) -> np.ndarray:
        return np.array([0.0, 0.0, 0.0])

# --- UPDATED BASELINES THAT INHERIT THE NEW PHYSICS ---

class GreedyStrategy(TrackingStrategy):
    @property
    def name(self) -> str: return "Greedy (KERS)"
    
    def get_control(self, state: np.ndarray, track_info: Dict) -> np.ndarray:
        s, v, soc = state
        throttle, brake, _ = self.get_driver_commands(state, track_info)
        
        P_ers = 0.0
        # ERS Logic (Same as before)
        if brake > 0.05 and soc < self.ers.max_soc:
            P_ers = -self.ers.max_recovery_power
        elif throttle > 0.85 and v > 20.0 and soc > self.ers.min_soc + 0.05:
            P_ers = self.ers.max_deployment_power
            
        return np.array([P_ers, throttle, brake])

class TargetSOCStrategy(TrackingStrategy):
    def __init__(self, vehicle_config, ers_config, track_model, reference_profile, target_soc=0.5):
        super().__init__(vehicle_config, ers_config, track_model, reference_profile)
        self.target_soc = target_soc
        self.kp_soc = 2e6 

    @property
    def name(self) -> str: return f"Target SOC ({self.target_soc*100:.0f}%)"

    def get_control(self, state: np.ndarray, track_info: Dict) -> np.ndarray:
        s, v, soc = state
        throttle, brake, _ = self.get_driver_commands(state, track_info)
        
        P_ers = 0.0
        if brake > 0.05 and soc < self.ers.max_soc:
            P_ers = -self.ers.max_recovery_power
        elif throttle > 0.5:
            soc_error = soc - self.target_soc
            P_request = soc_error * self.kp_soc
            P_ers = np.clip(P_request, -self.ers.max_recovery_power, self.ers.max_deployment_power)
            
            if soc <= self.ers.min_soc: P_ers = max(0, P_ers)
            if soc >= self.ers.max_soc: P_ers = min(0, P_ers)

        return np.array([P_ers, throttle, brake])

class AlwaysDeployStrategy(TrackingStrategy):
    @property
    def name(self) -> str: return "Always Deploy"
    
    def get_control(self, state: np.ndarray, track_info: Dict) -> np.ndarray:
        s, v, soc = state
        throttle, brake, _ = self.get_driver_commands(state, track_info)
        
        P_ers = 0.0
        if brake > 0.05 and soc < self.ers.max_soc:
            P_ers = -self.ers.max_recovery_power
        elif throttle > 0.1 and soc > self.ers.min_soc:
            P_ers = self.ers.max_deployment_power
                
        return np.array([P_ers, throttle, brake])

class SmartRuleBasedStrategy(TrackingStrategy):
    @property
    def name(self) -> str: return "Smart Heuristic"
    
    def get_control(self, state: np.ndarray, track_info: Dict) -> np.ndarray:
        s, v, soc = state
        throttle, brake, _ = self.get_driver_commands(state, track_info)
        
        P_ers = 0.0
        if brake > 0.05 and soc < self.ers.max_soc:
            P_ers = -self.ers.max_recovery_power
        elif throttle > 0.95 or (throttle > 0.5 and v < 50.0):
            is_efficient_speed = 60/3.6 < v < 320/3.6
            has_buffer = soc > self.ers.min_soc + 0.05
            if is_efficient_speed and has_buffer:
                P_ers = self.ers.max_deployment_power
                
        return np.array([P_ers, throttle, brake])