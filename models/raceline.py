"""
Racelines recomputed inside TUM's corridors widened for the kerbs (BUGS TRK-11).

TUM's racelines keep the car's centre about 0.75 m inside TUM's track edges, which leave out the kerbs. On those
lines the 2026 pole speeds need 6–8 g in a few fast corners. Here the line is recomputed inside the TUM corridor
widened by a kerb allowance on each side.

The line is the centreline moved sideways by an offset a_i at each node: r_i = p_i + a_i n_i, with n_i the unit
normal to the left. The bounds on a_i are the TUM widths plus the allowance, minus the margin TUM keeps. The
discrete curvature at node i, along the line's own normal N_i and with h-, h+ the lengths of its two segments, is

    k_i = 2 / (h- + h+) * N_i . [(r_i+1 - r_i) / h+ - (r_i - r_i-1) / h-]

and depends only on a_i-1, a_i and a_i+1. The cost is  sum_i k_i^2 ds_i + LENGTH_WEIGHT * length  (ds_i the
centreline spacing): the squared curvature, blended with the line's length as in Braghin et al. (2008), whose
blend of the minimum-curvature and the shortest line approximates a minimum-time line. Pure minimum curvature
runs to the outside of every long arc and gave laps 1–2.4 % longer than the distance the cars drive (∫ v dt);
LENGTH_WEIGHT = 5e-4 /m brings the 12 current and coming rounds to a median of 0.0 % (−0.6 to +0.9 %).

Each iteration linearises the curvature (exact derivatives of its three offsets: a Gauss-Newton step) and takes a
second-order model of the length (convex), solves the box-constrained quadratic programme with a small primal-
dual interior-point method (box_qp) inside a trust region, shortens the step until the cost falls, and repeats.
exact=False instead freezes N_i, h- and h+ at the current line (TUM's linearisation): with no kerb allowance
that reproduces TUM's lines within about 1 m, but it leaves out how a sideways shift stretches the line, which
biases the line to the inside of long arcs (its lines are 0.1–0.4 % shorter than TUM's).

Output files use TUM's raceline format (x_m, y_m in TUM's frame, '#' comments) so that every raceline loader
reads them: data/racelines_wide/<raceline name>.csv, picked with models.track.find_tumftm_raceline(variant="wide").
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from .geometry import _PeriodicSpline

ROOT = Path(__file__).resolve().parents[1]
TRACKS_DIR = ROOT / "data" / "tracks"            # TUM centrelines with right and left widths
RACELINES_DIR = ROOT / "data" / "racelines"      # TUM minimum-curvature racelines
WIDE_DIR = ROOT / "data" / "racelines_wide"      # This module's lines

TUM_MARGIN = 0.75     # (m) Distance TUM's racelines keep from TUM's edges (5th percentile over the 17 tracks: 0.75–0.81 m)
KERB_ALLOWANCE = 1.0  # (m) Extra width per side for the kerbs and the run-off the drivers use
STEP = 5.0            # (m) Node spacing along the centreline, as in TUM's files
SMOOTHING = 15.0      # (m) Half-power wavelength of the corridor's centreline and width fits
FOLD_FRACTION = 0.7   # Largest inside offset, as a fraction of the centreline's local radius
LENGTH_WEIGHT = 5e-4  # (1/m²) Weight of the line's length against ∑ κ² ds (see the module docstring)


@dataclass
class Corridor:
    """A closed centreline with unit left normals and the bounds on the lateral offset of the car's centre."""
    x: np.ndarray
    y: np.ndarray
    nx: np.ndarray        # Unit normal to the left of the driving direction
    ny: np.ndarray
    lower: np.ndarray     # (m) Smallest offset (to the right, negative)
    upper: np.ndarray     # (m) Largest offset (to the left, positive)
    width_right: np.ndarray   # (m) TUM's widths, for plots
    width_left: np.ndarray
    name: str = ""

    def __len__(self) -> int:
        return len(self.x)

    def points(self, offset: np.ndarray) -> np.ndarray:
        """(n × 2) line at the given lateral offsets."""
        return np.column_stack([self.x + offset * self.nx, self.y + offset * self.ny])

    def edges(self, extra: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """Right and left track edges (TUM widths plus extra), each (n × 2)."""
        return self.points(-(self.width_right + extra)), self.points(self.width_left + extra)


def find_track_file(name: str, directory: Path = TRACKS_DIR) -> Optional[Path]:
    """TUM centreline file for a track or raceline name, matched case-insensitively."""
    for path in sorted(Path(directory).glob("*.csv")):
        if path.stem.lower() == name.lower():
            return path
    return None


def closed_length(xy: np.ndarray) -> float:
    """Length (m) of a closed polyline, including the segment back to the start."""
    return float(np.linalg.norm(np.diff(np.vstack([xy, xy[:1]]), axis=0), axis=1).sum())


def make_corridor(centreline: np.ndarray, width_right: np.ndarray, width_left: np.ndarray,
                  kerb: float = KERB_ALLOWANCE, margin: float = TUM_MARGIN, step: float = STEP,
                  smoothing: float = SMOOTHING, name: str = "") -> Corridor:
    """
    A corridor from a closed centreline (n × 2, in driving order, not repeating the first point) and its widths.

    The centreline is refitted with a periodic P-spline (half-power wavelength `smoothing` m) and resampled every
    `step` m, so that the normals carry no point-to-point noise; widths are interpolated around the lap. The car's
    centre may go up to width + kerb - margin from the centreline on each side.
    """
    centreline = np.asarray(centreline, dtype=float)
    closed = np.vstack([centreline, centreline[:1]])
    s_raw = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(closed, axis=0), axis=1))])
    length = s_raw[-1]
    s_raw = s_raw[:-1]
    sx = _PeriodicSpline(s_raw, centreline[:, 0], length, smoothing)
    sy = _PeriodicSpline(s_raw, centreline[:, 1], length, smoothing)
    n = max(int(round(length / step)), 8)
    s = np.arange(n) * length / n
    x, y = sx(s), sy(s)
    dx, dy = sx(s, der=1), sy(s, der=1)
    norm = np.hypot(dx, dy)
    nx, ny = -dy / norm, dx / norm
    wr = _PeriodicSpline(s_raw, np.asarray(width_right, dtype=float), length, smoothing)(s)
    wl = _PeriodicSpline(s_raw, np.asarray(width_left, dtype=float), length, smoothing)(s)
    lower = -(wr + kerb - margin)
    upper = wl + kerb - margin
    # On the inside of a tight bend the normals cross at the centre of curvature: stay short of it, or the
    # offset nodes fold over each other
    kappa = (dx * sy(s, der=2) - dy * sx(s, der=2)) / norm**3
    reach = FOLD_FRACTION / np.maximum(np.abs(kappa), 1e-9)
    upper = np.where(kappa > 0, np.minimum(upper, reach), upper)
    lower = np.where(kappa < 0, np.maximum(lower, -reach), lower)
    if np.any(lower >= upper):
        raise ValueError(f"{name}: the corridor is empty at {np.sum(lower >= upper)} nodes (margin above the width)")
    return Corridor(x, y, nx, ny, lower, upper, wr, wl, name)


def load_corridor(track: str, kerb: float = KERB_ALLOWANCE, margin: float = TUM_MARGIN,
                  directory: Path = TRACKS_DIR, **options) -> Corridor:
    """The corridor of a bundled TUM track (data/tracks/<track>.csv: x, y, right width, left width)."""
    path = find_track_file(track, directory)
    if path is None:
        raise FileNotFoundError(f"No TUM centreline for '{track}' in {directory}")
    data = np.loadtxt(path, delimiter=",", comments="#")
    return make_corridor(data[:, :2], data[:, 2], data[:, 3], kerb=kerb, margin=margin, name=path.stem, **options)


def _three_point_curvature(prev: np.ndarray, cur: np.ndarray, nxt: np.ndarray) -> np.ndarray:
    """Signed curvature at `cur` of the polyline prev → cur → nxt (rows are points)."""
    h_minus = np.linalg.norm(cur - prev, axis=1)
    h_plus = np.linalg.norm(nxt - cur, axis=1)
    tangent = nxt - prev
    tangent = tangent / np.linalg.norm(tangent, axis=1)[:, None]
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
    second = (nxt - cur) / h_plus[:, None] - (cur - prev) / h_minus[:, None]
    return 2.0 / (h_minus + h_plus) * np.einsum("ij,ij->i", normal, second)


def line_curvature(xy: np.ndarray) -> np.ndarray:
    """Signed curvature (1/m, positive turning left) at each node of a closed polyline (three-point formula)."""
    return _three_point_curvature(np.roll(xy, 1, axis=0), xy, np.roll(xy, -1, axis=0))


def _curvature_jacobian(corridor: Corridor, offset: np.ndarray, eps: float = 1e-4):
    """
    Curvature at each node as b + C @ offset to first order around `offset` (exact derivatives of the three-point
    curvature in the three offsets it depends on, by central differences). Returns (C, b).
    """
    n = len(corridor)
    nn = np.column_stack([corridor.nx, corridor.ny])
    r = corridor.points(offset)
    prev, nxt = np.roll(r, 1, axis=0), np.roll(r, -1, axis=0)
    n_prev, n_next = np.roll(nn, 1, axis=0), np.roll(nn, -1, axis=0)
    k = _three_point_curvature
    c_prev = (k(prev + eps * n_prev, r, nxt) - k(prev - eps * n_prev, r, nxt)) / (2 * eps)
    c_self = (k(prev, r + eps * nn, nxt) - k(prev, r - eps * nn, nxt)) / (2 * eps)
    c_next = (k(prev, r, nxt + eps * n_next) - k(prev, r, nxt - eps * n_next)) / (2 * eps)
    i = np.arange(n)
    C = sparse.csr_matrix((np.concatenate([c_prev, c_self, c_next]),
                           (np.tile(i, 3), np.concatenate([(i - 1) % n, i, (i + 1) % n]))), shape=(n, n))
    return C, k(prev, r, nxt) - C @ offset


def _linearised_curvature(corridor: Corridor, offset: np.ndarray):
    """
    Curvature at each node as b + C @ offset, with the line's normals N_i and segment lengths h-, h+ frozen at
    `offset` (so the second difference along N_i is linear in the offsets). Returns (C, b, w): C sparse (n × n)
    with three diagonals, b (n,), and the integration weights w = (h- + h+) / 2 (m).
    """
    n = len(corridor)
    p = np.column_stack([corridor.x, corridor.y])
    nn = np.column_stack([corridor.nx, corridor.ny])
    r = corridor.points(offset)
    prev, nxt = np.roll(r, 1, axis=0), np.roll(r, -1, axis=0)
    h_minus = np.linalg.norm(r - prev, axis=1)
    h_plus = np.linalg.norm(nxt - r, axis=1)
    tangent = nxt - prev
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    N = np.column_stack([-tangent[:, 1], tangent[:, 0]])
    scale = 2.0 / (h_minus + h_plus)

    i = np.arange(n)
    ip, im = (i + 1) % n, (i - 1) % n
    c_next = scale / h_plus * np.einsum("ij,ij->i", N, nn[ip])
    c_self = -scale * (1.0 / h_plus + 1.0 / h_minus) * np.einsum("ij,ij->i", N, nn)
    c_prev = scale / h_minus * np.einsum("ij,ij->i", N, nn[im])
    C = sparse.csr_matrix(
        (np.concatenate([c_prev, c_self, c_next]), (np.tile(i, 3), np.concatenate([im, i, ip]))), shape=(n, n)
    )
    p_prev, p_next = np.roll(p, 1, axis=0), np.roll(p, -1, axis=0)
    b = scale * np.einsum("ij,ij->i", N, (p_next - p) / h_plus[:, None] - (p - p_prev) / h_minus[:, None])
    return C, b, 0.5 * (h_minus + h_plus)


def box_qp(H: sparse.spmatrix, g: np.ndarray, lower: np.ndarray, upper: np.ndarray,
           tolerance: float = 1e-10, max_iterations: int = 100) -> np.ndarray:
    """
    min ½ xᵀHx + gᵀx subject to lower ≤ x ≤ upper, for a sparse positive semi-definite H, by a primal-dual
    interior-point method. Each Newton step is one sparse solve with H plus a diagonal, so a banded H of a
    thousand variables takes milliseconds per step.
    """
    n = len(g)
    H = sparse.csc_matrix(H)
    x = 0.5 * (lower + upper)
    s1, s2 = x - lower, upper - x                 # Slacks to the bounds
    scale = max(1.0, float(np.abs(H.diagonal()).max()))
    z1, z2 = np.full(n, scale), np.full(n, scale)  # Multipliers of the lower and upper bounds
    for _ in range(max_iterations):
        residual = H @ x + g - z1 + z2
        mu = (s1 @ z1 + s2 @ z2) / (2 * n)
        if mu < tolerance * scale and np.max(np.abs(residual)) < tolerance * scale * 1e2:
            break
        sigma_mu = 0.1 * mu
        matrix = H + sparse.diags(z1 / s1 + z2 / s2)
        rhs = -residual + (sigma_mu / s1 - z1) - (sigma_mu / s2 - z2)
        dx = spsolve(matrix.tocsc(), rhs)
        dz1 = (sigma_mu - s1 * z1 - z1 * dx) / s1
        dz2 = (sigma_mu - s2 * z2 + z2 * dx) / s2
        ds1, ds2 = dx, -dx
        # Largest step that keeps slacks and multipliers positive, with a fraction-to-boundary rule
        step = 1.0
        for value, change in ((s1, ds1), (s2, ds2), (z1, dz1), (z2, dz2)):
            falling = change < 0
            if falling.any():
                step = min(step, 0.995 * float(np.min(-value[falling] / change[falling])))
        x, s1, s2 = x + step * dx, s1 + step * ds1, s2 + step * ds2
        z1, z2 = z1 + step * dz1, z2 + step * dz2
    return np.clip(x, lower, upper)


def _length_terms(corridor: Corridor, offset: np.ndarray):
    """Length (m) of the closed line at `offset`, its gradient in the offsets, and its Hessian (convex, sparse)."""
    n = len(corridor)
    nn = np.column_stack([corridor.nx, corridor.ny])
    r = corridor.points(offset)
    e = np.roll(r, -1, axis=0) - r                       # Segment i: node i → i + 1
    length = np.linalg.norm(e, axis=1)
    u = e / length[:, None]
    n_next = np.roll(nn, -1, axis=0)
    gradient = np.einsum("ij,ij->i", np.roll(u, 1, axis=0), nn) - np.einsum("ij,ij->i", u, nn)
    # Segment i's Hessian in (a_i, a_i+1) is Jᵀ (I - u uᵀ) J / |e| with J = [-n_i, n_i+1]
    cross = lambda a, b: (np.einsum("ij,ij->i", a, b) - np.einsum("ij,ij->i", a, u) * np.einsum("ij,ij->i", b, u)) / length
    h_ii, h_jj, h_ij = cross(nn, nn), cross(n_next, n_next), -cross(nn, n_next)
    i = np.arange(n)
    j = (i + 1) % n
    H = sparse.csr_matrix((np.concatenate([h_ii, h_jj, h_ij, h_ij]),
                           (np.concatenate([i, j, i, j]), np.concatenate([i, j, j, i]))), shape=(n, n))
    return float(length.sum()), gradient, H


def min_curvature_offset(corridor: Corridor, iterations: int = 100, tolerance: float = 1e-3, trust: float = 2.0,
                         start: Optional[np.ndarray] = None, exact: bool = True,
                         length_weight: float = LENGTH_WEIGHT) -> np.ndarray:
    """
    Lateral offsets (m) of the line inside the corridor that minimises ∑ κ_i² Δs_i + length_weight · length
    (Δs_i the centreline's spacing), by a sequence of convex quadratic programmes in the offsets (see the module
    docstring), each step within `trust` m of the last line and shortened until the cost falls. Stops when no
    node moves by more than `tolerance` m at a full step, or the cost stops falling.

    exact=False: TUM's linearisation, with the line's normals and segment lengths frozen at each iteration, steps
    accepted when ∫ κ² ds falls, and no length term.
    """
    lower, upper = corridor.lower, corridor.upper
    offset = np.clip(np.zeros(len(corridor)) if start is None else np.asarray(start, dtype=float), lower, upper)
    _, _, spacing = _linearised_curvature(corridor, np.zeros(len(corridor)))
    if exact:
        objective = lambda a: (float(np.sum(spacing * line_curvature(corridor.points(a)) ** 2))
                               + length_weight * closed_length(corridor.points(a)))
    else:
        objective = lambda a: curvature_cost(corridor.points(a))
    cost = objective(offset)
    for _ in range(iterations):
        if exact:
            (C, b), w = _curvature_jacobian(corridor, offset), spacing
        else:
            C, b, w = _linearised_curvature(corridor, offset)
        # ∑ w (b + C a)²  =  aᵀ (Cᵀ W C) a + 2 (Cᵀ W b)ᵀ a + const, scaled to order one. Each step stays within
        # `trust` m of the current line, where the frozen normals and segment lengths still hold
        W = sparse.diags(w)
        H = (C.T @ W @ C).tocsc()
        g = C.T @ (w * b)
        if exact and length_weight > 0:
            # Second-order model of the length around the current offsets, at half weight like the curvature term
            _, grad_l, hess_l = _length_terms(corridor, offset)
            H = (H + 0.5 * length_weight * hess_l).tocsc()
            g = g + 0.5 * length_weight * (grad_l - hess_l @ offset)
        norm = float(np.abs(H.diagonal()).max())
        lo, hi = np.maximum(lower, offset - trust), np.minimum(upper, offset + trust)
        direction = box_qp(H / norm + 1e-9 * sparse.identity(len(w)), g / norm, lo, hi) - offset
        # Backtrack on the nonlinear cost, so that it falls at every iteration
        step = 1.0
        while step > 1e-3:
            trial = offset + step * direction
            trial_cost = objective(trial)
            if trial_cost < cost:
                break
            step *= 0.5
        else:
            break
        moved = float(np.max(np.abs(trial - offset)))
        offset, cost, previous = trial, trial_cost, cost
        if (step == 1.0 and moved < tolerance) or previous - cost < 1e-9 * previous:
            break
    return offset


def curvature_cost(xy: np.ndarray) -> float:
    """∫ κ² ds (1/m) of a closed polyline."""
    kappa = line_curvature(xy)
    seg = np.linalg.norm(np.diff(np.vstack([xy, xy[:1]]), axis=0), axis=1)
    return float(np.sum(kappa**2 * 0.5 * (seg + np.roll(seg, 1))))


def wide_raceline(track: str, kerb: float = KERB_ALLOWANCE, margin: float = TUM_MARGIN,
                  smoothing: float = SMOOTHING, **options) -> Tuple[np.ndarray, Corridor, np.ndarray]:
    """The minimum-curvature line of a bundled track: (n × 2 points in TUM's frame, its corridor, the offsets)."""
    corridor = load_corridor(track, kerb=kerb, margin=margin, smoothing=smoothing)
    offset = min_curvature_offset(corridor, **options)
    return corridor.points(offset), corridor, offset


def write_raceline(path, xy: np.ndarray, comment: str = "") -> None:
    """A raceline in TUM's format: '# x_m,y_m' and one x,y row per point (a closed line, first point not repeated)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (f"{comment}\n" if comment else "") + "x_m,y_m"
    np.savetxt(path, np.asarray(xy, dtype=float), delimiter=",", fmt="%.6f", header=header, comments="# ")


def driven_distance(laps, max_frozen: float = 0.05, n_laps: int = 10) -> Tuple[float, float, int]:
    """
    Distance the cars drive per lap (m), ∫ v dt from the timing line to the timing line, over the fastest
    `n_laps` FastF1 laps with at most `max_frozen` frozen car-data samples (TRK-7; those samples are dropped and
    the speed interpolated across them). Returns (median, spread as max − min, laps used).
    """
    distances = []
    for _, lap in laps[laps["LapTime"].notna()].sort_values("LapTime").head(n_laps).iterrows():
        try:
            car = lap.get_car_data(pad=1, pad_side="both")
        except Exception:
            continue
        frozen = ((car["Throttle"] >= 104) & car["Brake"].astype(bool)).to_numpy()
        if len(car) < 50 or frozen.mean() > max_frozen:
            continue
        t = car["SessionTime"].dt.total_seconds().to_numpy()[~frozen]
        v = car["Speed"].to_numpy(dtype=float)[~frozen] / 3.6
        t0 = lap["LapStartTime"].total_seconds()
        grid = np.linspace(t0, t0 + lap["LapTime"].total_seconds(), 20001)
        speed = np.interp(grid, t, v)
        distances.append(float(np.sum(0.5 * (speed[1:] + speed[:-1]) * np.diff(grid))))
    if not distances:
        raise ValueError("No lap with a lap time and few enough frozen samples")
    return float(np.median(distances)), float(np.ptp(distances)), len(distances)


def build_wide_racelines(kerb: float = KERB_ALLOWANCE, out_dir: Path = WIDE_DIR,
                         racelines_dir: Path = RACELINES_DIR, verbose: bool = True, **options) -> dict:
    """
    Write a wide line for every bundled TUM raceline that has a centreline, named like the raceline file so that
    the same raceline name finds both. Returns {name: (TUM length, wide length) in m}.
    """
    lengths = {}
    exact = options.get("exact", True)
    weight = options.get("length_weight", LENGTH_WEIGHT) if exact else 0.0
    for path in sorted(Path(racelines_dir).glob("*.csv")):
        xy, corridor, offset = wide_raceline(path.stem, kerb=kerb, **options)
        tum = np.loadtxt(path, delimiter=",", comments="#")[:, :2]
        lengths[path.stem] = (closed_length(tum), closed_length(xy))
        comment = (f"Curvature-length line (length weight {weight:g} /m², {'exact' if exact else 'frozen'} "
                   f"linearisation) in the TUM corridor ({corridor.name}) widened by {kerb:g} m per side, "
                   f"{TUM_MARGIN:g} m margin (models/raceline.py)")
        write_raceline(Path(out_dir) / path.name, xy, comment)
        if verbose:
            at_bound = np.mean((offset <= corridor.lower + 0.01) | (offset >= corridor.upper - 0.01))
            print(f"{path.stem:12s} TUM {lengths[path.stem][0]:7.1f} m, wide {lengths[path.stem][1]:7.1f} m "
                  f"({100 * (lengths[path.stem][1] / lengths[path.stem][0] - 1):+.2f} %), "
                  f"peak |κ| {np.abs(line_curvature(tum)).max():.4f} → {np.abs(line_curvature(xy)).max():.4f} 1/m, "
                  f"{100 * at_bound:.0f} % of nodes on a bound")
    return lengths


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Write the wide racelines (data/racelines_wide)")
    parser.add_argument("--kerb", type=float, default=KERB_ALLOWANCE, help="Extra width per side (m)")
    parser.add_argument("--smoothing", type=float, default=SMOOTHING, help="Corridor smoothing wavelength (m)")
    parser.add_argument("--length-weight", type=float, default=LENGTH_WEIGHT, help="Weight of the length (1/m²)")
    parser.add_argument("--frozen", action="store_true", help="TUM's linearisation, no length term")
    parser.add_argument("--out", type=Path, default=WIDE_DIR)
    args = parser.parse_args()
    build_wide_racelines(kerb=args.kerb, out_dir=args.out, smoothing=args.smoothing, exact=not args.frozen,
                         length_weight=args.length_weight)
