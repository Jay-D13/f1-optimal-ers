"""
2026 events and their energy rules, from the FIA's per-event "Power Unit Information" documents.

Qualifying recharge caps are measured on the MGU-K's DC bus, timing line to timing line (C5.2.10).
Practice and sprint qualifying run with Overtake on (B7.2.2a), so their caps (practice here, sprint qualifying
the same as qualifying) are the PUI values as published, which already include its +0.5 MJ.
Super-clip: the harvest allowed at full throttle was 250 kW until Miami and 350 kW from Miami on (C5.12).
The ramp rate applies after the first power step of a derate (C5.12.6).
Straight Mode zones: normal-grip activation points from the FIA "Competition Notes – Circuit Map" of each event,
as (corner, metres after it) with FastF1's corner labels. The maps draw each zone but give no end point in text;
models/telemetry.py ends it at the next corner marker.
Ramp windows: the PUI's per-sector exceptions to the derate rules (C5.12.4, .5, .7), as lap distances from the
timing line in the FIA's metres: "first_step_350" sectors where the first reduction may be 350 kW instead of
150 kW, "reset" sectors where the reduction may be reset (the deploy demand may rise again), and
"speed_threshold" sectors where the ERS-K follows the ramp rates only above `value` km/h instead of 210 km/h.
Windows the PUI marks for sprint qualifying and qualifying only have quali_only=True.
Racelines: the bundled TUM raceline for the current layout, if there is one. The FastF1 positions are snapped to
the timing provider's map line, not the line the cars drive (TRK-10), so rounds without a raceline (None) have no
reliable path yet: Melbourne and Barcelona changed after the TUM data were made, and Miami, Monaco, Zandvoort,
Madrid and Baku are not in it. They are left out of calibration until that is sorted out.
"""
from dataclasses import dataclass
from typing import Optional


RAMP_WINDOW_KINDS = ("first_step_350", "reset", "speed_threshold")


@dataclass(frozen=True)
class RampWindow:
    kind: str                        # One of RAMP_WINDOW_KINDS
    start_m: float                   # Lap distance from the timing line where the window starts (m)
    end_m: float                     # ... and ends (m)
    value: float = 0.0               # first_step_350: the first step (kW); speed_threshold: the threshold (km/h)
    quali_only: bool = False         # Sprint qualifying and qualifying only

    def __post_init__(self):
        if self.kind not in RAMP_WINDOW_KINDS:
            raise ValueError(f"Unknown ramp window kind: {self.kind}. Options: {RAMP_WINDOW_KINDS}")
        if not self.end_m > self.start_m:
            raise ValueError(f"Ramp window ends before it starts: {self}")


def _first_step(start, end, quali_only=False):
    return RampWindow("first_step_350", start, end, 350.0, quali_only)


def _reset(start, end, quali_only=False):
    return RampWindow("reset", start, end, 0.0, quali_only)


def _threshold(start, end, kph, quali_only=False):
    return RampWindow("speed_threshold", start, end, kph, quali_only)


@dataclass(frozen=True)
class Event2026:
    round: int
    name: str
    tracks: tuple                    # Track names that map to this event (lower case)
    quali_recharge_mj: float         # Qualifying recharge cap (MJ per lap, DC side)
    race_recharge_mj: float          # Race recharge cap without Overtake (MJ per lap)
    practice_recharge_mj: float      # Free practice recharge cap (MJ per lap, DC side)
    power_limited_distance_m: float  # FIA power-limited distance (m)
    ramp_rate_kw_s: float            # Ramp-down rate after the first step (kW/s)
    superclip_kw: float              # Harvest allowed at full throttle (kW)
    verified: bool = True            # False when the cap is not from a public FIA document
    straight_mode_zones: tuple = ()  # Activation points: (corner label, metres after the corner marker)
    raceline: Optional[str] = None   # Bundled TUM raceline (data/racelines) for the current layout, if any
    ramp_windows: tuple = ()         # RampWindow entries from the PUI (C5.12.4/5/7 sectors)

    def ramp_windows_for(self, session: str) -> tuple:
        """The ramp windows that apply in a session ("qualifying" includes sprint qualifying)."""
        return tuple(w for w in self.ramp_windows if session == "qualifying" or not w.quali_only)


EVENTS_2026 = {
    1: Event2026(1, "Australia", ("melbourne", "albert park"), 7.0, 8.0, 8.5, 3518, 50, 250,
                    straight_mode_zones=(("14", 50), ("2", 20), ("5", 85), ("8", 35), ("10", 60))),
    2: Event2026(2, "China", ("shanghai",), 9.0, 8.5, 9.0, 3125, 100, 250,
                    straight_mode_zones=(("16", 100), ("4", 60), ("10", 100), ("13", 60)), raceline="Shanghai"),
    3: Event2026(3, "Japan", ("suzuka",), 8.0, 8.5, 9.0, 3472, 100, 250,
                    straight_mode_zones=(("18", 0), ("14", 0)), raceline="Suzuka"),  # "Exit of T18/T14": taken at the marker
    4: Event2026(4, "Miami", ("miami",), 8.0, 8.5, 9.0, 3346, 100, 350,
                    straight_mode_zones=(("19", 0), ("8", 165), ("16", 90))),  # "Exit of T19": taken at the marker
    5: Event2026(5, "Canada", ("montreal", "canada"), 6.0, 8.0, 8.5, 2682, 100, 350,
                    straight_mode_zones=(("14", 100), ("7", 50), ("9", 60), ("11", 30)), raceline="montreal"),
    6: Event2026(6, "Monaco", ("monaco",), 9.0, 8.5, 9.0, 1388, 100, 350,
                    straight_mode_zones=()),
    7: Event2026(7, "Barcelona", ("catalunya", "barcelona"), 7.0, 8.5, 9.0, 2440, 100, 350,
                    straight_mode_zones=(("14", 45), ("3", 0), ("5", 90), ("9", 40))),  # "40 m before the exit of T3": taken at the marker
    # Barcelona's PUI V2 (Doc 31, 13 Jun) has C5.12.7 windows, but their positions are not transcribed yet.
    8: Event2026(8, "Austria", ("spielberg", "austria"), 6.0, 8.0, 8.5, 2923, 100, 350,
                    straight_mode_zones=(("10", 110), ("1", 110), ("3", 90), ("8", 10)), raceline="Spielberg"),  # Pairing of points and corners unsure
    9: Event2026(9, "Great Britain", ("silverstone",), 6.5, 8.0, 8.5, 3833, 50, 350,
                    straight_mode_zones=(("18", 65), ("5", 55), ("7", 155), ("14", 65)), raceline="silverstone"),
    10: Event2026(10, "Belgium", ("spa",), 7.0, 8.5, 9.0, 4594, 50, 350,
                    straight_mode_zones=(("19", 190), ("1", 140), ("4", 60), ("15", 140), ("17", 80)), raceline="spa"),
    11: Event2026(11, "Hungary", ("budapest", "hungaroring"), 9.0, 8.5, 9.0, 1885, 100, 350,
                    straight_mode_zones=(("14", 40), ("1A", 30), ("3", 50), ("11", 60)), raceline="Budapest"),  # "30 m after T1A entry": taken after the marker
    # Dutch PUI (Doc 3, 19 Aug 2026, under fia.com/system/files/documents/): SQ and Q 7.5 MJ, PLD 2411 m, 100 kW/s
    # Windows: 350 kW first step at T2-T3, T8-T10 and exit T13 (Q only), reset at exit T14 (bracketed in the PUI,
    # taken as Q only); no C5.12.7 window.
    12: Event2026(12, "Netherlands", ("zandvoort",), 7.5, 8.5, 9.0, 2411, 100, 350,
                    straight_mode_zones=(("14", 20), ("10", 50)),
                    ramp_windows=(_first_step(670, 800), _first_step(2000, 2450), _first_step(3450, 3650, True),
                                  _reset(3700, 4200, True))),
    13: Event2026(13, "Italy", ("monza",), 5.0, 7.0, 7.5, 4218, 50, 350,
                    straight_mode_zones=(("11", 30), ("3", 70), ("7", 170), ("10", 130)), raceline="monza"),
    # Madrid PUI V2 (Doc 25, 11 Sep): the C5.12.4 windows. Its C5.12.5 resets and per-corner C5.12.7 thresholds
    # (240-260 km/h) are not transcribed yet, and the Race Director's event notes moved windows mid-weekend.
    14: Event2026(14, "Spain (Madrid)", ("madrid", "madring"), 7.5, 8.5, 9.0, 3206, 100, 350,
                    straight_mode_zones=(("22", 100), ("3", 40)),
                    ramp_windows=(_first_step(1900, 1975), _first_step(2200, 2300), _first_step(4100, 4800))),
    # Baku PUI (Doc 5, 23 Sep): qualifying-only windows, a 350 kW first step from exit T16 and a reset at exit T20
    15: Event2026(15, "Azerbaijan", ("baku",), 8.5, 8.5, 9.0, 3796, 50, 350,
                    straight_mode_zones=(("19", 45), ("2", 110)),
                    ramp_windows=(_first_step(4050, 5300, True), _reset(5600, 6000, True))),
    # The 2026 Bahrain Grand Prix is held at Sepang (PUI Doc 4 and circuit map Doc 7, both 1 Oct 2026). Windows:
    # 350 kW first step at T1-T2, T5-T7, T9-T11, T12-T13 and exit T15 (SQ and Q); resets at exit T5, exit T6 and
    # exit T15 (SQ and Q); a 270 km/h threshold at T5-T6 and T12-T13.
    16: Event2026(16, "Bahrain (Sepang)", ("sepang", "kuala lumpur", "malaysia", "bahrain"), 7.5, 8.5, 9.0, 3365, 100, 350,
                    straight_mode_zones=(("15", 65), ("3", 10), ("8", 65), ("14", 65)), raceline="Sepang",
                    ramp_windows=(_first_step(600, 750), _first_step(1900, 2500), _first_step(3100, 3400),
                                  _first_step(3750, 4000), _first_step(5050, 5300, True),
                                  _reset(1950, 2100), _reset(2150, 2350), _reset(5200, 5500, True),
                                  _threshold(1750, 2200, 270), _threshold(3700, 3900, 270))),
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
