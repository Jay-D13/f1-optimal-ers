import numpy as np
from abc import ABC, abstractmethod

from config import ERSConfig
from models import TrackSegment


class BaselineStrategy(ABC):
    """Base class for baseline ERS strategies"""
    
    def __init__(self, ers_config: ERSConfig):
        self.ers_config = ers_config
        
    @abstractmethod
    def get_control(self, state: np.ndarray, track_segment: TrackSegment) -> np.ndarray:
        """Return full control vector [ERS power, throttle, brake]"""
        raise NotImplementedError