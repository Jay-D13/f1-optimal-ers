"""
The time structure of a lap: seconds at full throttle, at part throttle without brake, braking, and clipping.

These are position-free targets (BUGS CAL-1): the real 2026 pole laps spend 8-18 s per lap decelerating at full
throttle and 12-26 s at part throttle, and a fit that only looks at the speed trace and the lap time can match
those with the wrong split. The same function measures a model trajectory and a reference lap:

- full throttle: throttle >= 98 %;
- braking: the reference lap's brake flag; in the model, brake force above 2 % of the maximum;
- part throttle: neither of these;
- clipping: full throttle with dv/ds below -0.05 (km/h)/m (5-sample moving average), in runs of at least 5 samples
  (the j5t3313 super-clip detector, REFERENCE section 7).

Every sample weighs ds/v, the time to the next sample around the lap. A model trajectory is measured on its own
intervals (5 m, each weighing its exact dt), or sampled at a reference lap's distances, which makes the
smoothing and the run rule identical too. Reference laps have gaps (dropped frozen blocks, dropouts) where one
sample's state stands for up to 2.8 s; the fit's residuals leave those stretches out of both laps (gaps, measure).

A second clip detector follows SPARC (REFERENCE section 7): full throttle with the acceleration below the lap's own
power-limited envelope a(v) = P/(m v) - B v^2 - C by more than the MGU-K's deploy, so the car runs on the ICE alone
or harvests. P is an upper quantile of the lap's own full-throttle samples; the shape B, C is fixed. It catches
clipping that still accelerates, which the dv/ds detector can't see.
"""
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

FULL_THROTTLE = 0.98         # Throttle fraction counted as full
MODEL_BRAKE = 0.02           # Model brake force (fraction of the maximum) counted as braking
CLIP_SLOPE = -0.05           # (km/h)/m: speed falling at least this fast at full throttle is clipping
SMOOTH = 5                   # Samples in the moving average of dv/ds
MIN_RUN = 5                  # Samples in the shortest run counted as clipping
MAX_GAP = 0.75               # (s) A reference sample standing for longer than this marks a gap (car data is ~4 Hz)

SPARC_MIN_SPEED = 180 / 3.6  # (m/s) Below this a full-throttle car is traction-limited, not power-limited
SPARC_QUANTILE = 0.9         # Quantile of the full-throttle samples' wheel power that sets the lap's envelope
SPARC_MASS = 772.0           # (kg) 2026 qualifying mass (config/vehicle.py), to turn power into acceleration
SPARC_DEFICIT = 350e3        # (W) Below the envelope by the MGU-K's whole deploy: the ICE alone, or harvesting
# Envelope shape a(v) = P/(m v) - SPARC_DRAG v² - SPARC_ROLL, from the start-value car in Straight Mode
# (CdA 0.95 x 0.80, ClA about 2.4, rolling coefficient 0.03, 1.18 kg/m³), fixed so that model and reference share it
SPARC_DRAG = 0.5 * 1.18 * (0.76 + 0.03 * 2.4) / SPARC_MASS   # (1/m)
SPARC_ROLL = 0.03 * 9.81     # (m/s²)
G = 9.81

DURATIONS = ("full", "part", "brake", "clip")   # The residual entries, in this order


@dataclass
class TimeStructure:
    """Seconds per lap in each state, and the SPARC detector's view of clipping."""
    full: float              # Full throttle
    part: float              # Part throttle, no brake
    brake: float             # Braking
    clip: float              # Clipping, dv/ds detector
    total: float             # Sum of the sample weights (the lap time on the samples)
    clip_sparc: float = 0.0  # Clipping, SPARC detector (full throttle, below the lap's own envelope)
    clip_both: float = 0.0   # Seconds both detectors flag
    envelope: float = 0.0    # (kW) The SPARC envelope's wheel power, at SPARC_MASS

    def durations(self) -> np.ndarray:
        return np.array([getattr(self, k) for k in DURATIONS])

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class Samples:
    """A lap as samples around it, in the units the detectors use."""
    s: np.ndarray            # Lap distance (m), sorted, in [0, length)
    v: np.ndarray            # (m/s)
    throttle: np.ndarray     # Fraction, 0-1
    braking: np.ndarray      # bool
    length: float            # Lap length (m)
    weight: np.ndarray       # (s) Time each sample stands for
    gradient: Optional[np.ndarray] = None   # Road gradient (rad, + uphill) at the samples


def periodic_weights(s: np.ndarray, v: np.ndarray, length: float) -> np.ndarray:
    """ds/v, with ds the distance to the next sample around the lap."""
    ds = np.diff(np.append(s, s[0] + length))
    return ds / v


def _periodic_slope(s: np.ndarray, y: np.ndarray, length: float) -> np.ndarray:
    """dy/ds at each sample by central differences around the lap (np.gradient's rule, wrapped)."""
    s_ext = np.concatenate([[s[-1] - length], s, [s[0] + length]])
    y_ext = np.concatenate([[y[-1]], y, [y[0]]])
    return np.gradient(y_ext, s_ext)[1:-1]


def _moving_average(x: np.ndarray, n: int) -> np.ndarray:
    """Centred moving average over n samples, wrapped around the lap."""
    if n <= 1:
        return x.copy()
    half = n // 2
    ext = np.concatenate([x[-half:], x, x[:half]])
    return np.convolve(ext, np.ones(n) / n, mode="valid")[: len(x)]


def runs(flag: np.ndarray, min_len: int) -> np.ndarray:
    """flag with only its runs of at least min_len consecutive samples kept (a run may wrap through the line)."""
    flag = np.asarray(flag, dtype=bool)
    n = len(flag)
    if flag.all():
        return flag.copy()
    keep = np.zeros(n, dtype=bool)
    start = int(np.argmin(flag))            # A False sample: no run crosses it
    order = (np.arange(n) + start) % n
    f = flag[order]
    i = 0
    while i < n:
        if f[i]:
            j = i
            while j < n and f[j]:
                j += 1
            if j - i >= min_len:
                keep[order[i:j]] = True
            i = j
        else:
            i += 1
    return keep


def slope_kmh_per_m(samples: Samples) -> np.ndarray:
    """dv/ds in (km/h)/m, smoothed over SMOOTH samples."""
    return _moving_average(_periodic_slope(samples.s, 3.6 * samples.v, samples.length), SMOOTH)


def clip_flags(samples: Samples) -> np.ndarray:
    """The dv/ds detector."""
    full = samples.throttle >= FULL_THROTTLE
    return runs(full & (slope_kmh_per_m(samples) < CLIP_SLOPE), MIN_RUN)


def propulsive_acceleration(samples: Samples) -> np.ndarray:
    """a_x + g sin(gradient) (m/s²): the acceleration the car would have on the flat."""
    a_x = samples.v * slope_kmh_per_m(samples) / 3.6
    if samples.gradient is not None:
        a_x = a_x + G * np.sin(samples.gradient)
    return a_x


def envelope_level(v: np.ndarray, a: np.ndarray) -> float:
    """
    The lap's own power level P/m (W/kg): the SPARC_QUANTILE quantile of (a + B v² + C)·v over the samples, the
    power at the wheels that the drag shape SPARC_DRAG, SPARC_ROLL would need for each acceleration. Only the
    level is fitted: one lap holds too few full-power samples at high speed to fit the shape too (fitted, the
    drag runs to its bound, because every 2026 car clips at the top of its straights).
    """
    return float(np.quantile((a + SPARC_DRAG * v**2 + SPARC_ROLL) * v, SPARC_QUANTILE))


def sparc_flags(samples: Samples) -> Tuple[np.ndarray, float]:
    """
    The SPARC detector and the envelope level P/m (W/kg) it used: full throttle, no brake, above
    SPARC_MIN_SPEED, with the power at the wheels below the lap's own envelope by more than the MGU-K's deploy
    (so the ICE alone, or the MGU-K harvesting), in runs of at least MIN_RUN samples.
    """
    fast = (samples.throttle >= FULL_THROTTLE) & ~samples.braking & (samples.v >= SPARC_MIN_SPEED)
    if fast.sum() < 10:
        return np.zeros(len(samples.v), dtype=bool), 0.0
    a = propulsive_acceleration(samples)
    level = envelope_level(samples.v[fast], a[fast])
    power = (a + SPARC_DRAG * samples.v**2 + SPARC_ROLL) * samples.v
    return runs(fast & (power < level - SPARC_DEFICIT / SPARC_MASS), MIN_RUN), level


def gaps(samples: Samples, max_gap: float = MAX_GAP) -> np.ndarray:
    """
    (start, end) lap distances of the stretches a sample stands for alone for more than max_gap seconds: dropped
    frozen blocks and dropouts in car data, where one sample's state is carried over a whole braking zone
    (Monza's Roggia sits in a 2.3 s gap, Suzuka has a 2.8 s one).
    """
    i = np.flatnonzero(samples.weight > max_gap)
    end = np.append(samples.s[1:], samples.s[0] + samples.length)
    return np.column_stack([samples.s[i], end[i]]) if len(i) else np.zeros((0, 2))


def outside(s: np.ndarray, stretches: Optional[np.ndarray], length: float) -> np.ndarray:
    """True where a lap distance lies in none of the [start, end) stretches (which may run through the line)."""
    keep = np.ones(len(s), dtype=bool)
    for a, b in (stretches if stretches is not None else ()):
        keep &= ~(((s >= a) & (s < b)) | ((s + length >= a) & (s + length < b)))
    return keep


def measure(samples: Samples, sparc: bool = True, exclude: Optional[np.ndarray] = None) -> TimeStructure:
    """
    The time structure of a lap given as samples. `exclude` lists (start, end) stretches left out of every
    duration (e.g. a reference lap's gaps, applied to the model too); the detectors still run on the whole lap.
    """
    keep = outside(samples.s, exclude, samples.length)
    w = np.where(keep, samples.weight, 0.0)
    full = samples.throttle >= FULL_THROTTLE
    braking = samples.braking
    clip = clip_flags(samples)
    out = TimeStructure(
        full=float(w[full].sum()), part=float(w[~full & ~braking].sum()), brake=float(w[braking].sum()),
        clip=float(w[clip].sum()), total=float(w.sum()),
    )
    if sparc:
        flags, level = sparc_flags(samples)
        out.clip_sparc, out.clip_both = float(w[flags].sum()), float(w[flags & clip].sum())
        out.envelope = level * SPARC_MASS / 1e3
    return out


# ------------------------------------------------------------------------------------------------ the two sources
def reference_samples(reference) -> Samples:
    """A ReferenceLap as samples (throttle in % on the lap, brake as a 0/1 flag)."""
    s, v, length = reference.s, reference.speed, reference.track.total_length
    return Samples(s=s, v=v, throttle=reference.throttle / 100.0, braking=reference.brake > 0.5, length=length,
                   weight=periodic_weights(s, v, length), gradient=_gradient_at(reference.track, s))


def trajectory_samples(trajectory, at: Optional[np.ndarray] = None, track=None) -> Samples:
    """
    A model trajectory as samples. By default one sample per interval, at its midpoint, with the interval's mean
    controls and the speed ds/dt, so the weights sum to the lap time exactly. With `at` (a reference lap's
    distances), the speed is interpolated there and the controls taken from the interval each distance falls in,
    so the model is sampled like the reference. `track` gives the gradient for the SPARC detector.
    """
    s_nodes = np.asarray(trajectory.s, dtype=float)
    length = float(trajectory.lap_length or s_nodes[-1])
    throttle = np.asarray(trajectory.throttle_opt, dtype=float)
    braking = np.asarray(trajectory.brake_opt, dtype=float) > MODEL_BRAKE
    if at is None:
        dt = np.diff(trajectory.t_opt)
        s = 0.5 * (s_nodes[1:] + s_nodes[:-1])
        v = np.diff(s_nodes) / dt
        return Samples(s=s, v=v, throttle=throttle, braking=braking, length=length, weight=dt,
                       gradient=None if track is None else _gradient_at(track, s))
    s = np.mod(np.asarray(at, dtype=float), length)
    v = np.interp(s, s_nodes, trajectory.v_opt, period=length)
    k = np.clip(np.searchsorted(s_nodes, s, side="right") - 1, 0, len(throttle) - 1)
    return Samples(s=s, v=v, throttle=throttle[k], braking=braking[k], length=length,
                   weight=periodic_weights(s, v, length), gradient=None if track is None else _gradient_at(track, s))


def _gradient_at(track, s: np.ndarray) -> Optional[np.ndarray]:
    td = getattr(track, "track_data", None)
    if td is None or td.gradient is None:
        return None
    return np.interp(s, td.s, td.gradient, period=track.total_length)


def reference_gaps(reference, max_gap: float = MAX_GAP) -> np.ndarray:
    """The reference lap's gaps (see gaps), to leave out of both laps' durations."""
    return gaps(reference_samples(reference), max_gap)


def reference_structure(reference, sparc: bool = True, exclude: Optional[np.ndarray] = None) -> TimeStructure:
    return measure(reference_samples(reference), sparc, exclude)


def trajectory_structure(trajectory, at: Optional[np.ndarray] = None, track=None, sparc: bool = True,
                         exclude: Optional[np.ndarray] = None) -> TimeStructure:
    return measure(trajectory_samples(trajectory, at, track), sparc, exclude)


def structure(lap, at: Optional[np.ndarray] = None, track=None, sparc: bool = True,
              exclude: Optional[np.ndarray] = None) -> TimeStructure:
    """
    Time structure of a ReferenceLap or a model OptimalTrajectory (see trajectory_samples for `at` and `track`,
    measure for `exclude`).
    """
    if hasattr(lap, "throttle_opt"):
        return trajectory_structure(lap, at, track, sparc, exclude)
    return reference_structure(lap, sparc, exclude)


def disagreements(samples: Samples, min_seconds: float = 0.5) -> List[Dict]:
    """
    Stretches where exactly one detector flags clipping, as (start m, end m, seconds, which), longest first:
    'sparc' where the car is below its envelope but its speed doesn't fall fast enough for the dv/ds detector,
    'dv' where the speed falls but the power stays within the MGU-K's deploy of the envelope (or below
    SPARC_MIN_SPEED, or with braking).
    """
    dv = clip_flags(samples)
    sp, _ = sparc_flags(samples)
    out = []
    for which, flag in (("sparc", sp & ~dv), ("dv", dv & ~sp)):
        n, i = len(flag), 0
        while i < n:
            if flag[i]:
                j = i
                while j < n and flag[j]:
                    j += 1
                secs = float(samples.weight[i:j].sum())
                if secs >= min_seconds:
                    end = samples.s[j] if j < n else samples.length
                    out.append({"start_m": float(samples.s[i]), "end_m": float(end), "seconds": secs, "which": which})
                i = j
            else:
                i += 1
    return sorted(out, key=lambda x: -x["seconds"])
