import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Tuple


class BaseStrategy(ABC):
    
    def __init__(self, vehicle_config, ers_config, track_model):
        self.vehicle = vehicle_config
        self.ers = ers_config
        self.track = track_model
        
    @property
    @abstractmethod
    def name(self) -> str:
        """name just for logging"""
        pass
    
    @abstractmethod
    def get_control(self, state: np.ndarray, track_info: Dict) -> np.ndarray:
        pass
    
    def reset(self):
        """Reset any internal state (for new lap)"""
        pass
    
    def solve_mpc_step(self, 
                       current_state: np.ndarray,
                       track_position: float,
                       soc_reference: np.ndarray = None) -> Tuple[np.ndarray, Dict]:
        """
        wrapper for LapSimulator.
        """
        # Get track info
        segment = self.track.get_segment_at_distance(track_position)
        track_info = {
            'gradient': segment.gradient,
            'radius': segment.radius,
            'curvature': segment.curvature,
            'sector': segment.sector,
        }
        
        control = self.get_control(current_state, track_info)
        
        return control, {'success': True, 'strategy': self.name}
