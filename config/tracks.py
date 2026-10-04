"""
The track registry: one entry per circuit with the names it goes by, its bundled TUM raceline and its 2025 aero
preset. The 2026 events (energy rules, Straight Mode zones, whether the raceline matches the current layout) stay
in config/events.py and are found from here with `Track.event_2026`.

Names are matched after normalising (lower case, no accents, letters and digits only), so 'Montréal', 'MONTREAL'
and 'Mexico City' / 'MexicoCity' all work. "Bahrain" is Sakhir before 2026 and Sepang in 2026 (the 2026 Bahrain
Grand Prix runs at Sepang): pass the year, or use the circuit's name.
"""
import unicodedata
from dataclasses import dataclass
from typing import Callable, Optional

from .events import Event2026, find_event_2026
from .vehicle import VehicleConfig


def normalise(name) -> str:
    """Lower case, no accents, letters and digits only: 'Montréal' → 'montreal', 'Mexico City' → 'mexicocity'."""
    ascii_name = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    return "".join(ch for ch in ascii_name.lower() if ch.isalnum())


@dataclass(frozen=True)
class Track:
    name: str                                 # Circuit name, as printed
    aliases: tuple = ()                       # Other names: city, country, event (any case)
    raceline: Optional[str] = None            # Stem of the bundled TUM raceline in data/racelines, if any
    preset: Callable[[], VehicleConfig] = VehicleConfig  # 2025 aero preset (2026 runs override the aero)

    @property
    def names(self) -> set:
        """Every normalised name this track answers to."""
        return {normalise(n) for n in (self.name, *self.aliases)}

    @property
    def event_2026(self) -> Optional[Event2026]:
        """The 2026 event held here (energy rules, Straight Mode zones), or None."""
        for name in (self.name, *self.aliases):
            event = find_event_2026(name.lower())
            if event is not None and normalise(event.tracks[0]) in self.names:
                return event
        return None

    def vehicle_config(self) -> VehicleConfig:
        return self.preset()


TRACKS = (
    Track("Melbourne", ("Albert Park", "Australia"), raceline="Melbourne"),
    Track("Shanghai", ("China",), raceline="Shanghai", preset=VehicleConfig.for_shanghai),
    Track("Suzuka", ("Japan",), raceline="Suzuka"),
    Track("Miami", ("Miami Gardens",)),
    Track("Montreal", ("Canada", "Circuit Gilles Villeneuve"), raceline="montreal", preset=VehicleConfig.for_montreal),
    Track("Monaco", ("Monte Carlo",), preset=VehicleConfig.for_monaco),
    Track("Catalunya", ("Barcelona", "Spain"), raceline="Catalunya"),
    Track("Spielberg", ("Austria", "Red Bull Ring"), raceline="Spielberg"),
    Track("Silverstone", ("Great Britain", "Britain"), raceline="silverstone", preset=VehicleConfig.for_silverstone),
    Track("Spa", ("Belgium", "Spa-Francorchamps"), raceline="spa", preset=VehicleConfig.for_spa),
    Track("Budapest", ("Hungary", "Hungaroring"), raceline="Budapest"),
    Track("Zandvoort", ("Netherlands",)),
    Track("Monza", ("Italy",), raceline="monza", preset=VehicleConfig.for_monza),
    Track("Madrid", ("Madring", "Spain (Madrid)")),
    Track("Baku", ("Azerbaijan",)),
    Track("Sepang", ("Kuala Lumpur", "Malaysia", "Bahrain (Sepang)"), raceline="Sepang"),
    Track("Sakhir", ("Bahrain",), raceline="Sakhir"),
    Track("Austin", ("COTA", "United States"), raceline="Austin"),
    Track("Mexico City", ("Mexico",), raceline="MexicoCity"),
    Track("Sao Paulo", ("Interlagos", "Brazil"), raceline="SaoPaulo"),
    Track("Yas Marina", ("Abu Dhabi", "Yas Island"), raceline="YasMarina"),
    Track("Hockenheim", ("Germany",), raceline="Hockenheim"),
)

_BY_NAME = {name: track for track in TRACKS for name in track.names}


def find_track(name, year: Optional[int] = None) -> Optional[Track]:
    """
    The registry entry for a circuit, city, country or event name, or None. In 2026 (and for a bare round
    number) the 2026 event's names come first, so 'Bahrain' is Sepang there and Sakhir otherwise.
    """
    if name is None:
        return None
    if year == 2026 or (year is None and str(name).isdigit()):
        event = find_event_2026(name if str(name).isdigit() else str(name).lower())
        if event is not None:
            return _BY_NAME.get(normalise(event.tracks[0]))
    return _BY_NAME.get(normalise(name))


def track_vehicle_config(name, year: Optional[int] = None) -> VehicleConfig:
    """The track's 2025 aero preset, or the default car for a track the registry doesn't know."""
    track = find_track(name, year)
    return track.vehicle_config() if track else VehicleConfig()


def track_raceline(name, year: Optional[int] = None) -> Optional[str]:
    """
    Stem of the bundled TUM raceline for a track (None if it has none), or the name itself for a track the
    registry doesn't know, so a raceline file named after it can still be found.
    """
    track = find_track(name, year)
    return track.raceline if track else str(name)
