"""
2026 events and their energy rules, from the FIA's per-event "Power Unit Information" documents.

Qualifying recharge caps are measured on the MGU-K's DC bus, timing line to timing line (C5.2.10).
Super-clip: the harvest allowed at full throttle was 250 kW until Miami and 350 kW from Miami on (C5.12).
The ramp rate applies after the first power step of a derate (C5.12.6).
Straight Mode zones: normal-grip activation points from the FIA "Competition Notes – Circuit Map" of each event,
as (corner, metres after it) with FastF1's corner labels. The maps draw each zone but give no end point in text;
models/telemetry.py ends it at the next corner marker.
Racelines: the bundled TUM raceline for the current layout, if there is one. The FastF1 positions are snapped to
the timing provider's map line, not the line the cars drive (TRK-10), so rounds without a raceline (None) have no
reliable path yet: Melbourne and Barcelona changed after the TUM data were made, and Miami, Monaco, Zandvoort,
Madrid and Baku are not in it. They are left out of calibration until that is sorted out.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Event2026:
    round: int
    name: str
    tracks: tuple                    # Track names that map to this event (lower case)
    quali_recharge_mj: float         # Qualifying recharge cap (MJ per lap, DC side)
    race_recharge_mj: float          # Race recharge cap without Overtake (MJ per lap)
    power_limited_distance_m: float  # FIA power-limited distance (m)
    ramp_rate_kw_s: float            # Ramp-down rate after the first step (kW/s)
    superclip_kw: float              # Harvest allowed at full throttle (kW)
    verified: bool = True            # False when the cap is not from a public FIA document
    straight_mode_zones: tuple = ()  # Activation points: (corner label, metres after the corner marker)
    raceline: Optional[str] = None   # Bundled TUM raceline (data/racelines) for the current layout, if any


EVENTS_2026 = {
    1: Event2026(1, "Australia", ("melbourne", "albert park"), 7.0, 8.0, 3518, 50, 250,
                    straight_mode_zones=(("14", 50), ("2", 20), ("5", 85), ("8", 35), ("10", 60))),
    2: Event2026(2, "China", ("shanghai",), 9.0, 8.5, 3125, 100, 250,
                    straight_mode_zones=(("16", 100), ("4", 60), ("10", 100), ("13", 60)), raceline="Shanghai"),
    3: Event2026(3, "Japan", ("suzuka",), 8.0, 8.5, 3472, 100, 250,
                    straight_mode_zones=(("18", 0), ("14", 0)), raceline="Suzuka"),  # "Exit of T18/T14": taken at the marker
    4: Event2026(4, "Miami", ("miami",), 8.0, 8.5, 3346, 100, 350,
                    straight_mode_zones=(("19", 0), ("8", 165), ("16", 90))),  # "Exit of T19": taken at the marker
    5: Event2026(5, "Canada", ("montreal", "canada"), 6.0, 8.0, 2682, 100, 350,
                    straight_mode_zones=(("14", 100), ("7", 50), ("9", 60), ("11", 30)), raceline="montreal"),
    6: Event2026(6, "Monaco", ("monaco",), 9.0, 8.5, 1388, 100, 350,
                    straight_mode_zones=()),
    7: Event2026(7, "Barcelona", ("catalunya", "barcelona"), 7.0, 8.5, 2440, 100, 350,
                    straight_mode_zones=(("14", 45), ("3", 0), ("5", 90), ("9", 40))),  # "40 m before the exit of T3": taken at the marker
    8: Event2026(8, "Austria", ("spielberg", "austria"), 6.0, 8.0, 2923, 100, 350,
                    straight_mode_zones=(("10", 110), ("1", 110), ("3", 90), ("8", 10)), raceline="Spielberg"),  # Pairing of points and corners unsure
    9: Event2026(9, "Great Britain", ("silverstone",), 6.5, 8.0, 3833, 50, 350,
                    straight_mode_zones=(("18", 65), ("5", 55), ("7", 155), ("14", 65)), raceline="silverstone"),
    10: Event2026(10, "Belgium", ("spa",), 7.0, 8.5, 4594, 50, 350,
                    straight_mode_zones=(("19", 190), ("1", 140), ("4", 60), ("15", 140), ("17", 80)), raceline="spa"),
    11: Event2026(11, "Hungary", ("budapest", "hungaroring"), 9.0, 8.5, 1885, 100, 350,
                    straight_mode_zones=(("14", 40), ("1A", 30), ("3", 50), ("11", 60)), raceline="Budapest"),  # "30 m after T1A entry": taken after the marker
    # The Dutch PUI is not public: 7.5 MJ from ScuderiaFans, 9 MJ per SomersF1 (REFERENCE_2026 §4)
    12: Event2026(12, "Netherlands", ("zandvoort",), 7.5, 8.5, 2411, 100, 350, verified=False,
                    straight_mode_zones=(("14", 20), ("10", 50))),
    13: Event2026(13, "Italy", ("monza",), 5.0, 7.0, 4218, 50, 350,
                    straight_mode_zones=(("11", 30), ("3", 70), ("7", 170), ("10", 130)), raceline="monza"),
    14: Event2026(14, "Spain (Madrid)", ("madrid", "madring"), 7.5, 8.5, 3206, 100, 350,
                    straight_mode_zones=(("22", 100), ("3", 40))),
    15: Event2026(15, "Azerbaijan", ("baku",), 8.5, 8.5, 3796, 50, 350,
                    straight_mode_zones=(("19", 45), ("2", 110))),
}


def find_event_2026(key) -> Optional[Event2026]:
    """The 2026 event for a round number or a track name, or None."""
    if key is None:
        return None
    if isinstance(key, int) or str(key).isdigit():
        return EVENTS_2026.get(int(key))
    name = str(key).lower()
    for event in EVENTS_2026.values():
        if name in event.tracks or name == event.name.lower():
            return event
    return None
