from .ers import ERSConfig, ERSConfigQualifying, ERSConfigRace, get_ers_config
from .pit import PIT_LOSS_DEFAULTS_S, get_default_pit_loss
from .vehicle import VehicleConfig, TireParameters, get_vehicle_config

def get_default_config() -> tuple:
    return (
        VehicleConfig(),
        ERSConfig(),
        TireParameters(),
    )


def get_track_config(track_name: str) -> VehicleConfig:
    track_configs = {
        'monaco': VehicleConfig.for_monaco,
        'monza': VehicleConfig.for_monza,
        'montreal': VehicleConfig.for_montreal,
        'spa': VehicleConfig.for_spa,
        'silverstone': VehicleConfig.for_silverstone,
        'shanghai': VehicleConfig.for_shanghai,
    }
    
    config_func = track_configs.get(track_name.lower(), VehicleConfig)
    return config_func()

__all__ = [
    'VehicleConfig',
    'ERSConfig', 
    'ERSConfigQualifying',
    'ERSConfigRace',
    'TireParameters',
    # 'SimulationConfig',
    'get_default_config',
    'get_track_config',
    'get_vehicle_config',
    'get_ers_config',
    'PIT_LOSS_DEFAULTS_S',
    'get_default_pit_loss',
]
