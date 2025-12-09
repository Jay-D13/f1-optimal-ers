import numpy as np
from typing import Dict, Tuple

from models import F1TrackModel
from config import VehicleConfig, ERSConfig

# TODO fix baselines to follow physics properly
# TODO fuse with simple rule based controller?
class PureGreedyStrategy:
    """
    Classic greedy baseline from ERS optimization literature.

    References:
    - Borhan et al. (2012) "MPC for Energy Management"
    - Serrao et al. (2011) "Optimal control of hybrid vehicles"
    """

    def __init__(self, track_model: F1TrackModel, vehicle_config: VehicleConfig, ers_config: ERSConfig):
        self.track = track_model
        self.vehicle = vehicle_config
        self.ers = ers_config

        # Simple P-controller gains
        self.kp_accel = 0.5
        self.kp_brake = 0.8

        # Threshold for "straight" detection
        self.straight_radius_threshold = 500.0  # meters

    def get_control(self, state: np.ndarray, segment) -> np.ndarray:
        """
        Greedy control: deploy on straights, harvest on braking.

        state: [s, v, soc]
        """
        s = state[0]
        v = state[1]
        soc = state[2]

        # --- 1. Target Speed (Simple, no lookahead) ---
        # Just use current segment's cornering speed limit
        v_target = self.vehicle.get_max_cornering_speed(segment.radius) * 0.95

        # --- 2. Throttle/Brake Control ---
        error = v_target - v

        throttle = 0.0
        brake = 0.0

        if error > 0:
            # Accelerate
            throttle = np.clip(self.kp_accel * error, 0.0, 1.0)
        else:
            # Brake
            brake = np.clip(-self.kp_brake * error, 0.0, 1.0)

        # --- 3. GREEDY ERS Logic ---
        P_ers = 0.0

        # HARVEST: Always harvest at max power when braking
        if brake > 0.1:  # Braking threshold
            if soc < self.ers.max_soc:
                P_ers = -self.ers.max_recovery_power

        # DEPLOY: Deploy max power on straights if battery available
        # This is the "greedy" part - no optimization of when/where
        elif segment.radius > self.straight_radius_threshold:
            if soc > self.ers.min_soc + 0.02:  # Small buffer
                P_ers = self.ers.max_deployment_power

        return np.array([P_ers, throttle, brake])

    def solve_mpc_step(self, current_state: np.ndarray,
                       track_position: float,
                       soc_reference_trajectory: np.ndarray = None
                       ) -> Tuple[np.ndarray, Dict]:
        """Compatibility wrapper for simulator"""
        segment = self.track.get_segment_at_distance(track_position)
        control = self.get_control(current_state, segment)
        return control, {'success': True, 'greedy': True}


class AlwaysDeployGreedy:
    """
    Even simpler: deploy whenever possible, harvest when braking.

    This is the absolute baseline - should perform worst of all.
    """

    def __init__(self, track_model: F1TrackModel, vehicle_config: VehicleConfig, ers_config: ERSConfig):
        self.track = track_model
        self.vehicle = vehicle_config
        self.ers = ers_config

        self.kp_accel = 0.5
        self.kp_brake = 0.8

    def get_control(self, state: np.ndarray, segment) -> np.ndarray:
        """Always deploy when not braking."""
        s = state[0]
        v = state[1]
        soc = state[2]

        # Target speed
        v_target = self.vehicle.get_max_cornering_speed(segment.radius) * 0.95

        # Throttle/Brake
        error = v_target - v

        throttle = 0.0
        brake = 0.0

        if error > 0:
            throttle = np.clip(self.kp_accel * error, 0.0, 1.0)
        else:
            brake = np.clip(-self.kp_brake * error, 0.0, 1.0)

        # ERS: Harvest when braking, otherwise always deploy
        P_ers = 0.0

        if brake > 0.1:
            if soc < self.ers.max_soc:
                P_ers = -self.ers.max_recovery_power
        else:
            # GREEDY: Deploy whenever not braking (very wasteful!)
            if soc > self.ers.min_soc + 0.02:
                P_ers = self.ers.max_deployment_power

        return np.array([P_ers, throttle, brake])

    def solve_mpc_step(self, current_state: np.ndarray,
                       track_position: float,
                       soc_reference_trajectory: np.ndarray = None
                       ) -> Tuple[np.ndarray, Dict]:
        """Compatibility wrapper for simulator"""
        segment = self.track.get_segment_at_distance(track_position)
        control = self.get_control(current_state, segment)
        return control, {'success': True, 'always_deploy': True}
