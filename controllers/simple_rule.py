import numpy as np

from .base import BaselineStrategy
from models import TrackSegment


class SimpleRuleBasedStrategy(BaselineStrategy):
    """Simple rule-based ERS strategy"""
    
    def get_control(self, state: np.ndarray, track_segment: TrackSegment) -> np.ndarray:
        """Deploy on straights, recover under braking"""
        v = state[1]  # velocity
        soc = state[2]  # battery charge
        
        # Initialize control
        P_ers = 0
        throttle = 0.8  # Default throttle
        brake = 0
        
        # Determine corner type based on radius
        is_straight = track_segment.radius > 500
        is_slow_corner = track_segment.radius < 50
        
        # Only deploy if have sufficient battery AND on straight AND at high speed
        if is_straight and v > 60 and soc > 0.3:
            deploy_factor = min((soc - 0.3) / 0.4, 1.0)  # 0 at 30%, 1 at 70%
            P_ers = self.ers_config.max_deployment_power * deploy_factor
            throttle = 0.95
            
        elif is_slow_corner and v > 30:
            # Recover under braking
            P_ers = -self.ers_config.max_recovery_power * 0.7
            brake = 0.3
            throttle = 0
            
        # Opportunistic recovery when SOC is low
        elif soc < 0.35 and v > 40:
            P_ers = -self.ers_config.max_recovery_power * 0.4
            throttle = 0.6
        else:
            P_ers = 0
            throttle = 0.7
            
        return np.array([P_ers, throttle, brake])
    