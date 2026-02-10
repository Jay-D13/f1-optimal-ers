"""
Pit stop configuration defaults.

Values are approximate stationary race pit-lane losses (seconds),
used as pragmatic defaults for one-stop recommendation.
"""

from collections.abc import Mapping


PIT_LOSS_DEFAULTS_S: Mapping[str, float] = {
    "default": 22.0,
    "monaco": 19.5,
    "monza": 24.0,
    "montreal": 21.5,
    "spa": 21.0,
    "silverstone": 20.5,
}


def get_default_pit_loss(track_name: str) -> float:
    """Return default pit-lane loss for track, with fallback."""
    return PIT_LOSS_DEFAULTS_S.get(track_name.lower(), PIT_LOSS_DEFAULTS_S["default"])
