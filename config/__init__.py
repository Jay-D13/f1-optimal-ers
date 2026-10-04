from typing import Optional

from .ers import ERSConfig, ERSConfigQualifying, ERSConfigRace, get_ers_config
from .tire_model import TireCompoundConfig, TireThermalConfig, get_tire_compound_config
from .vehicle import VehicleConfig, TireParameters, get_vehicle_config
from .tracks import TRACKS, Track, find_track, track_raceline, track_vehicle_config


def get_default_config() -> tuple:
    return (
        VehicleConfig(),
        ERSConfig(),
        TireParameters(),
    )


def get_track_config(track_name: str, year: Optional[int] = None) -> VehicleConfig:
    """The track's 2025 aero preset from the track registry (config/tracks.py), or the default car."""
    return track_vehicle_config(track_name, year)


__all__ = [
    'VehicleConfig',
    'ERSConfig', 
    'ERSConfigQualifying',
    'ERSConfigRace',
    'TireParameters',
    # 'SimulationConfig',
    'get_default_config',
    'get_track_config',
    'TRACKS',
    'Track',
    'find_track',
    'track_raceline',
    'track_vehicle_config',
    'get_vehicle_config',
    'get_ers_config',
    'TireThermalConfig',
    'TireCompoundConfig',
    'get_tire_compound_config',
]
