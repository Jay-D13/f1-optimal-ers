"""
Track geometry from telemetry (ROADMAP Phase 3).

Builds a closed 3D curve of the driven line from the position samples of many clean laps, then evaluates
curvature, gradient and vertical curvature on a fine grid, independent of the NLP step.

The pooled samples are projected onto a reference lap to get their distance along the lap, and each
coordinate is fitted with a penalised periodic spline of that distance; the fit is repeated on its
own arc length.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
from scipy import sparse
from scipy.interpolate import BSpline
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree


@dataclass
class TrackGeometry:
    """A closed driven line sampled every `step` metres of arc length."""
    s: np.ndarray          # Arc length (m), [0, length)
    x: np.ndarray          # Position (m)
    y: np.ndarray
    z: np.ndarray          # Elevation (m)
    kappa: np.ndarray      # Signed heading change per metre driven (1/m), positive turning left
    gradient: np.ndarray   # Road gradient (rad), positive uphill
    kappa_v: np.ndarray    # Vertical curvature d(gradient)/ds (1/m): positive in a dip, negative over a crest
    length: float          # Lap length (m)
    source: str = ""

    @property
    def step(self) -> float:
        return self.length / len(self.s)

    @property
    def heading_turns(self) -> float:
        """Total heading change over the lap in turns: ±1 for a closed lap."""
        return float(np.sum(self.kappa) * self.step / (2.0 * np.pi))

    @property
    def radius(self) -> np.ndarray:
        """Unsigned radius (m), capped at 10 km on straights like the TUM loader."""
        return np.clip(1.0 / np.maximum(np.abs(self.kappa), 1e-6), 10.0, 10000.0)

    def to_csv(self, path) -> None:
        header = f"source={self.source}; length={self.length:.3f}\ns,x,y,z,kappa,gradient,kappa_v"
        data = np.column_stack([self.s, self.x, self.y, self.z, self.kappa, self.gradient, self.kappa_v])
        np.savetxt(path, data, delimiter=",", header=header, fmt="%.7g")

    @classmethod
    def from_csv(cls, path) -> "TrackGeometry":
        first = Path(path).read_text().splitlines()[0]
        meta = dict(item.split("=", 1) for item in first.lstrip("# ").split("; "))
        s, x, y, z, kappa, gradient, kappa_v = np.loadtxt(path, delimiter=",", comments="#").T
        return cls(s, x, y, z, kappa, gradient, kappa_v, float(meta["length"]), meta.get("source", ""))


def _closed_arc_length(points: np.ndarray) -> np.ndarray:
    """Cumulative chord length of a closed polyline, including the segment back to the start (length n + 1)."""
    closed = np.vstack([points, points[:1]])
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(closed, axis=0), axis=1))])


class _PeriodicSpline:
    """
    Penalised periodic spline v(s) of period `length` (a P-spline: Eilers & Marx, 1996).

    Knots sit every `spacing` metres, dense enough never to limit the shape; a penalty on the third differences
    of neighbouring coefficients sets the smoothness instead. `smoothing` is the half-power wavelength (m): the
    fit keeps shape changes longer than it and damps shorter ones. It is either one length or one per sample,
    averaged locally, so the smoothing can follow the car's speed. The penalty is scaled by the number of
    samples per coefficient, so the smoothing does not depend on how dense the data are.
    """

    ORDER = 3   # Difference order of the penalty

    def __init__(self, s: np.ndarray, values: np.ndarray, length: float, smoothing,
                 spacing: float = 2.0, k: int = 5):
        n = max(int(round(length / spacing)), 2 * k + 2)
        h = length / n
        self.length, self.k = length, k
        self.knots = h * np.arange(-k, n + k + 1)
        s = np.mod(np.asarray(s, dtype=float), length)

        # Basis functions j and j + n are the same periodic function: fold the n + k columns onto n
        design = BSpline.design_matrix(s, self.knots, k).tocoo()
        basis = sparse.csr_matrix((design.data, (design.row, design.col % n)), shape=(len(s), n))
        # Periodic difference operator of order ORDER
        difference = sparse.identity(n, format="csr")
        shift = sparse.csr_matrix((np.ones(n), (np.arange(n), (np.arange(n) + 1) % n)), shape=(n, n))
        for _ in range(self.ORDER):
            difference = (shift - sparse.identity(n)) @ difference
        # Half power where lam·(2π h / smoothing)^(2·ORDER) equals the data weight per coefficient
        lam = (len(s) / n) * (_local_mean(s, smoothing, length, n) / (2.0 * np.pi * h)) ** (2 * self.ORDER)
        # Difference row i spans coefficients i..i+ORDER, centred ORDER/2 intervals after coefficient i's knot
        lam = np.roll(lam, -(self.ORDER // 2))
        lhs = (basis.T @ basis + difference.T @ sparse.diags(lam) @ difference).tocsc()
        coefficients = spsolve(lhs, basis.T @ np.asarray(values, dtype=float))
        self.spline = BSpline(self.knots, np.concatenate([coefficients, coefficients[:k]]), k, extrapolate=False)

    def __call__(self, s, der: int = 0) -> np.ndarray:
        return self.spline(np.mod(np.asarray(s, dtype=float), self.length), nu=der)


def _local_mean(s: np.ndarray, values, length: float, n: int) -> np.ndarray:
    """Mean of per-sample values in each of n equal bins of a periodic lap; empty bins are interpolated."""
    if np.ndim(values) == 0:
        return np.full(n, float(values))
    bins = np.minimum((s / length * n).astype(int), n - 1)
    counts = np.bincount(bins, minlength=n)
    sums = np.bincount(bins, weights=np.asarray(values, dtype=float), minlength=n)
    filled = counts > 0
    centres = (np.arange(n) + 0.5) * length / n
    return np.interp(centres, centres[filled], sums[filled] / counts[filled], period=length)


def fit_track(
    laps: List[np.ndarray],
    step: float = 1.0,
    speeds: Optional[List[np.ndarray]] = None,
    smoothing_time: float = 0.8,
    min_smoothing: float = 20.0,
    xy_smoothing: float = 30.0,
    z_smoothing: float = 80.0,
    iterations: int = 3,
    source: str = "",
) -> TrackGeometry:
    """
    Fit a closed driven line to the position samples of several laps.

    Each coordinate is a penalised periodic spline of distance along the lap. The smoothing wavelength sets
    the resolution: tens of metres horizontally, to resolve corners without turning position noise into
    curvature on the straights (0.005 1/m at 100 m/s is 5 g), and a fixed, longer one for the elevation: its
    samples are clean (about 0.07 m RMS at Spa), but the vertical curvature is a second derivative.

    Args:
        laps: one (n × 3) array of x, y, z positions (m) per lap, in driving order; the first is the reference
        speeds: optional speed (m/s) of every sample, one array per lap. With speeds, the horizontal smoothing
            wavelength is the distance covered in smoothing_time, at least min_smoothing: the lap time is
            sensitive to curvature errors as v², and the car cannot follow shorter features at speed anyway
        smoothing_time, min_smoothing: see speeds
        step: output grid step along the lap (m)
        xy_smoothing, z_smoothing: half-power wavelength of the horizontal and vertical fits (m); xy_smoothing
            applies without speeds
        iterations: projection and fit rounds; each one re-projects the samples onto the latest curve
        source: description stored with the geometry
    """
    reference = np.asarray(laps[0], dtype=float)
    pooled = np.vstack([np.asarray(lap, dtype=float) for lap in laps])
    if speeds is not None:
        xy_smoothing = np.maximum(np.concatenate(speeds) * smoothing_time, min_smoothing)

    # Start from the reference lap as a closed polyline
    curve = reference
    s_curve = _closed_arc_length(curve)
    for _ in range(iterations):
        length = s_curve[-1]
        # Distance along the lap of every pooled sample: its nearest point on the current curve
        dense_s = np.arange(0.0, length, 0.25)
        dense = np.column_stack([np.interp(dense_s, s_curve, np.append(curve[:, c], curve[0, c])) for c in range(3)])
        _, nearest = cKDTree(dense[:, :2]).query(pooled[:, :2])
        s_samples = dense_s[nearest]

        splines = [
            _PeriodicSpline(s_samples, pooled[:, 0], length, xy_smoothing),
            _PeriodicSpline(s_samples, pooled[:, 1], length, xy_smoothing),
            _PeriodicSpline(s_samples, pooled[:, 2], length, z_smoothing),
        ]
        # The fitted curve, finely sampled, becomes the next projection target
        fine_s = np.arange(0.0, length, 0.5)
        curve = np.column_stack([spline(fine_s) for spline in splines])
        s_curve = _closed_arc_length(curve)

    # Evaluate on a periodic grid of true arc length, with the step adjusted to divide the lap
    length = s_curve[-1]
    n_points = max(int(round(length / step)), 1)
    step = length / n_points
    s = np.arange(n_points) * step
    u = np.interp(s, s_curve[:-1], fine_s)   # Spline parameter at each arc length
    x, y, z = (spline(u) for spline in splines)
    dx, dy, dz = (spline(u, der=1) for spline in splines)
    ddx, ddy = (spline(u, der=2) for spline in splines[:2])

    horizontal = np.hypot(dx, dy)
    gradient = np.arctan2(dz, horizontal)
    # Curvature of the plan view per horizontal metre, times cos(gradient): heading change per metre driven,
    # so that it sums to whole turns over the 3D arc length
    kappa = (dx * ddy - dy * ddx) / horizontal**3 * np.cos(gradient)
    # Vertical curvature: rate of change of the gradient along the lap (periodic central difference)
    kappa_v = (np.roll(gradient, -1) - np.roll(gradient, 1)) / (2.0 * step)
    return TrackGeometry(s, x, y, z, kappa, gradient, kappa_v, length, source)
