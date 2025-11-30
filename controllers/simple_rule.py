import numpy as np
from typing import Dict, Tuple

from models import F1TrackModel
from config import VehicleConfig, ERSConfig

# TODO still need to fix simple rule based controller so it follows propoer physics
class SimpleRuleBasedStrategy:
    """
    Robust baseline strategy using a P-controller with lookahead.
    
    Logic:
    1. Safety: Look ahead 150m. If a corner is coming, lower target speed.
    2. Longitudinal: Use P-control to hit target speed (Throttle/Brake).
    3. ERS: 
       - Harvest whenever we are braking.
       - Deploy whenever we are at full throttle.
    """
    
    def __init__(self, track_model: F1TrackModel, vehicle_config: VehicleConfig, ers_config: ERSConfig):
        self.track = track_model
        self.vehicle = vehicle_config
        self.ers = ers_config
        
        # Tuning parameters
        self.lookahead_distance = 150.0  # Look ahead 150m for corners
        self.kp_accel = 0.5              # Proportional gain for throttle
        self.kp_brake = 0.8              # Proportional gain for braking
    
    def get_control(self, state: np.ndarray, segment) -> np.ndarray:
        """
        Calculate control inputs based on lookahead physics.
        state: [s, v, soc]
        """
        s = state[0]
        v = state[1]
        soc = state[2]
        
        # --- 1. Determine Target Speed (Safety) ---
        # Get the speed limit at current location
        current_limit = self.vehicle.get_max_cornering_speed(segment.radius)
        
        # CRITICAL: Look ahead to find the minimum speed limit in the braking zone
        # We sample points ahead to see if a tight corner is approaching
        future_limit = 999.0
        for dist in [50, 100, 150]:
            future_s = s + dist
            future_seg = self.track.get_segment_at_distance(future_s)
            
            # Allow some braking distance: v^2 = u^2 + 2as
            # We estimate we can slow down effectively. 
            # If a 20m/s corner is 100m away, we don't need to be 20m/s NOW, 
            # but we shouldn't be 300m/s either.
            # Simplified approach: effectively "smear" the speed limit backwards
            limit_at_point = self.vehicle.get_max_cornering_speed(future_seg.radius)
            
            # Simple heuristic: we can brake at ~20 m/s^2 (2G)
            # v_allowed = sqrt(v_corner^2 + 2 * a_brake * distance)
            braking_capacity = 2.0 * 9.81 
            safe_approach_speed = np.sqrt(limit_at_point**2 + 2 * braking_capacity * dist)
            
            if safe_approach_speed < future_limit:
                future_limit = safe_approach_speed
        
        # The target is the lowest of current limit or approaching limit
        # We multiply by 0.95 to leave a 5% safety margin (it's a baseline, not optimal)
        v_target = min(current_limit, future_limit) * 0.95
        
        # --- 2. Longitudinal Control (Throttle/Brake) ---
        error = v_target - v
        
        throttle = 0.0
        brake = 0.0
        
        if error > 0:
            # We are slower than target -> Accelerate
            # Use tanh to smooth the throttle map (0.0 to 1.0)
            throttle = np.clip(self.kp_accel * error, 0.0, 1.0)
        else:
            # We are faster than target -> Brake
            brake = np.clip(-self.kp_brake * error, 0.0, 1.0)
            
        # --- 3. ERS Logic (Heuristic) ---
        P_ers = 0.0
        
        # STRATEGY: HARVEST
        # If we are braking mechanically, we SHOULD be harvesting energy first
        if brake > 0:
            # Max possible harvest (limited by component)
            P_ers = -self.ers.max_recovery_power
            
            # Reduce mechanical brake to account for drag from generator?
            # For this simple baseline, we just add the braking torque on top.
            # But we must stop harvesting if battery is full
            if soc >= self.ers.max_soc:
                P_ers = 0.0
        
        # STRATEGY: DEPLOY
        # Only deploy if:
        # 1. We are full throttle (exiting corner or on straight)
        # 2. We are not drag limited (top speed)
        # 3. We have sufficient SOC
        elif throttle > 0.9 and v < 340/3.6:
            if soc > self.ers.min_soc + 0.05: # Buffer of 5% above min
                P_ers = self.ers.max_deployment_power
        
        return np.array([P_ers, throttle, brake])

    def solve_mpc_step(self, current_state: np.ndarray, 
                       track_position: float,
                       soc_reference_trajectory: np.ndarray = None
                       ) -> Tuple[np.ndarray, Dict]:
        """Compatibility wrapper to use this class in the simulator"""
        segment = self.track.get_segment_at_distance(track_position)
        control = self.get_control(current_state, segment)
        return control, {'success': True, 'rule_based': True}