"""
Supports:
1. FastF1 telemetry: a geometry fitted to every clean lap of a session (models/telemetry.py)
2. TUMFTM minimum curvature racelines
3. A fitted TrackGeometry (models/geometry.py)
"""
import numpy as np
import pandas as pd

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from scipy.interpolate import interp1d, UnivariateSpline

from models.car import air_density
from config.events import find_event_2026
from models.telemetry import corner_distances, load_session, place_raceline, session_geometry, zone_intervals

# Largest distance (m) between a raceline and the session's map line before the layout counts as changed
MAX_RACELINE_GAP = 15.0

@dataclass
class TrackSegment:
    """Represents a segment of the track"""
    distance: float        # distance from start (m)
    length: float          # meters
    radius: float          # meters (inf for straight)
    curvature: float           # 1/radius (1/m)
    gradient: float            # Road gradient (radians)
    x: float = 0.0             # GPS X coordinate
    y: float = 0.0             # GPS Y coordinate
    sector: int = 1            # Track sector (1, 2, or 3)
    vertical_curvature: float = 0.0  # d(gradient)/ds (1/m): positive in a dip, negative over a crest
    
    @property
    def is_straight(self) -> bool:
        return self.radius > 500 or np.isinf(self.radius)


@dataclass 
class TrackData:
    """Complete track data in array form for optimization"""
    
    # Spatial discretization
    s: np.ndarray              # Distance points (m)
    ds: float                  # Discretization step (m)
    n_points: int              # Number of discretization points
    total_length: float        # Total track length (m)
    
    # Track geometry (arrays indexed by distance)
    radius: np.ndarray         # Corner radius at each point
    curvature: np.ndarray      # Curvature (1/radius)
    gradient: np.ndarray       # Road gradient (radians)
    
    # Coordinates for visualization
    x: np.ndarray
    y: np.ndarray
    
    # Track features
    sector: np.ndarray
    is_braking_zone: np.ndarray
    is_acceleration_zone: np.ndarray
    vertical_curvature: Optional[np.ndarray] = None   # d(gradient)/ds (1/m); None on flat loaders


def find_tumftm_raceline(track: str, directory: str | Path = "data/racelines") -> Optional[Path]:
    """Bundled TUMFTM raceline for a track name. File names are matched case-insensitively (Linux disks are case-sensitive)."""
    for path in sorted(Path(directory).glob("*.csv")):
        if path.stem.lower() == track.lower():
            return path
    return None


class F1TrackModel:
    """Track model built from FastF1 telemetry data"""
    
    def __init__(self, year: int , gp: str, session: str = 'Q', ds: float = 5.0):
        self.year = year
        self.gp = gp
        self.session_type = session
        self.ds = ds # spatial discretization step (meters)

        self.segments: List[TrackSegment] = []
        self.track_data: Optional[TrackData] = None
        self.total_length: float = 0.0
        
        # Raw telemetry for visualization
        self.telemetry_data: Optional[pd.DataFrame] = None
        
        # Source info
        self.data_source: str = 'none'

        # Air density from the session's weather (kg/m³), when the session has weather data
        self.air_density: Optional[float] = None

        # FIA Straight Mode zones as (start, end) lap distances (m), when known (2026 FastF1 tracks)
        self.straight_mode_zones: Optional[List[Tuple[float, float]]] = None
        
    def load_from_tumftm_raceline(self, raceline_path: str,
                                   track_params: Optional[dict] = None):
        """
        Load track from TUMFTM racelines file: https://github.com/TUMFTM/racetrack-database
        """
        print(f"   Loading TUMFTM raceline from {raceline_path}...")
        
        # Load raceline
        data = np.loadtxt(raceline_path, delimiter=',', comments='#')
        
        if data.shape[1] >= 4:
            x = data[:, 0]
            y = data[:, 1]
            w_right = data[:, 2]
            w_left = data[:, 3]
        else:
            x = data[:, 0]
            y = data[:, 1]
            w_right = np.ones(len(x)) * 5.0  # Default 5m
            w_left = np.ones(len(x)) * 5.0
        
        # Compute cumulative distance
        dx = np.diff(x)
        dy = np.diff(y)
        ds_raw = np.sqrt(dx**2 + dy**2)
        s_raw = np.concatenate([[0], np.cumsum(ds_raw)])
        self.total_length = s_raw[-1]
        
        # Compute curvature from spline (cleaner than finite differences)
        curvature = self._compute_curvature_spline(x, y)
        radius = 1.0 / (np.abs(curvature) + 1e-6)
        radius = np.clip(radius, 10, 10000)
        
        # Resample to uniform ds
        s_uniform = np.arange(0, self.total_length, self.ds)
        n_points = len(s_uniform)
        
        x_interp = np.interp(s_uniform, s_raw, x)
        y_interp = np.interp(s_uniform, s_raw, y)
        radius_interp = np.interp(s_uniform, s_raw, radius)
        curvature_interp = np.interp(s_uniform, s_raw, curvature)
        
        # Gradient (we're gonna assume flat cause... time and where would we get elevation data? sniff)
        gradient = np.zeros(n_points)
        
        # Build segments
        self.segments = []
        for i in range(n_points):
            segment = TrackSegment(
                distance=s_uniform[i],
                length=self.ds,
                radius=radius_interp[i],
                curvature=curvature_interp[i],
                gradient=gradient[i],
                x=x_interp[i],
                y=y_interp[i],
                sector=self._get_sector(s_uniform[i]),
            )
            self.segments.append(segment)
        
        self._create_track_arrays()
        self.data_source = 'tumftm'
        
        print(f"   ✓ Loaded {n_points} points, {self.total_length:.0f}m total")
        return self
        
    def load_from_geometry(self, geometry):
        """
        Load a track from a fitted TrackGeometry (models/geometry.py): the driven line with its curvature and
        gradient, resampled at ds by averaging over each step so that short curvature peaks are kept.
        """
        print(f"   Loading fitted geometry ({geometry.source or 'unnamed'})...")
        self.total_length = geometry.length
        s_uniform = np.arange(0, self.total_length, self.ds)
        n_points = len(s_uniform)

        def cell_mean(values):
            # Mean over [s - ds/2, s + ds/2] via the periodic running integral on the fine grid
            closed_s = np.append(geometry.s, geometry.length)
            closed_v = np.append(values, values[0])
            integral = np.concatenate([[0.0], np.cumsum(0.5 * (closed_v[1:] + closed_v[:-1]) * np.diff(closed_s))])
            total = integral[-1]

            def running(x):
                laps, rest = np.divmod(x, geometry.length)
                return laps * total + np.interp(rest, closed_s, integral)

            return (running(s_uniform + 0.5 * self.ds) - running(s_uniform - 0.5 * self.ds)) / self.ds

        curvature = cell_mean(geometry.kappa)
        radius = np.clip(1.0 / (np.abs(curvature) + 1e-6), 10, 10000)
        gradient = cell_mean(geometry.gradient)
        vertical_curvature = cell_mean(geometry.kappa_v)
        x = np.interp(s_uniform, geometry.s, geometry.x)
        y = np.interp(s_uniform, geometry.s, geometry.y)

        self.segments = [
            TrackSegment(
                distance=s_uniform[i], length=self.ds, radius=radius[i], curvature=curvature[i],
                gradient=gradient[i], x=x[i], y=y[i], sector=self._get_sector(s_uniform[i]),
                vertical_curvature=vertical_curvature[i],
            )
            for i in range(n_points)
        ]
        self._create_track_arrays()
        self.data_source = 'geometry'
        print(f"   ✓ Loaded {n_points} points, {self.total_length:.0f}m total")
        return self

    def load_from_fastf1(self, driver: Optional[str] = None, refresh: bool = False, raceline: Optional[str] = None):
        """
        Load the track from a FastF1 session: the line fitted to every clean lap (models/telemetry.py), or with
        raceline, a TUM raceline file placed onto that line, which gives it the session's height, timing line
        and Straight Mode zones. The live-timing positions are snapped to the provider's map line, which is not
        the line the cars drive (TRK-10), so the raceline is the better path where its layout is current.
        The fastest lap (or the driver's) is kept in telemetry_data for plots and comparisons.
        """
        session, round_number, location = load_session(self.year, self.gp, self.session_type)
        print(f"   {self.year} round {round_number}: {session.event['EventName']} ({location}), session {self.session_type}")

        weather = getattr(session, 'weather_data', None)
        if weather is not None and len(weather) > 0:
            self.air_density = air_density(
                float(weather['AirTemp'].median()),
                float(weather['Pressure'].median()),
                float(weather['Humidity'].median()),
            )

        laps = session.laps.pick_drivers(driver) if driver else session.laps
        lap = laps.pick_fastest()
        self.telemetry_data = lap.get_telemetry()

        geometry = session_geometry(self.year, self.gp, self.session_type, refresh=refresh, loaded_session=session)
        self.data_source = 'fastf1'
        if raceline is not None:
            xy = np.loadtxt(raceline, delimiter=',', comments='#')[:, :2]
            placed, gap = place_raceline(xy, geometry, source=f"{Path(raceline).name} on {geometry.source}")
            if gap > MAX_RACELINE_GAP:
                raise ValueError(
                    f"{raceline} is up to {gap:.0f} m from the {self.year} layout: the circuit has changed since"
                )
            print(f"   Raceline {Path(raceline).name} placed on the session (largest gap {gap:.1f} m)")
            geometry = placed
            self.data_source = 'tumftm+fastf1'
        self.load_from_geometry(geometry)

        event = find_event_2026(round_number) if self.year == 2026 else None
        if event is not None and event.straight_mode_zones:
            corners = corner_distances(session, geometry)
            if corners is None:
                print("   ⚠ No corner markers in FastF1: Straight Mode zones fall back to the radius heuristic")
            else:
                self.straight_mode_zones = zone_intervals(event.straight_mode_zones, corners, geometry.length)
                spans = ", ".join(f"{a:.0f}–{b:.0f}" for a, b in self.straight_mode_zones)
                print(f"   Straight Mode zones (m): {spans}")
        elif event is not None:
            self.straight_mode_zones = []
        return self, lap['Driver']

    def straight_mode_mask(self, s) -> Optional[np.ndarray]:
        """1.0 where lap distance s lies in an FIA Straight Mode zone, else 0.0; None when the zones are unknown."""
        if self.straight_mode_zones is None:
            return None
        s = np.mod(np.asarray(s, dtype=float), self.total_length)
        mask = np.zeros_like(s)
        for start, end in self.straight_mode_zones:
            inside = (s >= start) & (s <= end) if start <= end else (s >= start) | (s <= end)
            mask[inside] = 1.0
        return mask

    def _compute_curvature_spline(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        cleaner than finite differences
        """
        dx = np.diff(x)
        dy = np.diff(y)
        ds = np.sqrt(dx**2 + dy**2)
        s = np.concatenate([[0], np.cumsum(ds)])
        
        # Fit splines
        # smoothing factor to reduce noise
        smoothing = len(x) * 0.1  # TODO see if needs adjustment
        
        try:
            spline_x = UnivariateSpline(s, x, s=smoothing)
            spline_y = UnivariateSpline(s, y, s=smoothing)
            
            # Compute derivatives
            dx_ds = spline_x.derivative(1)(s)
            dy_ds = spline_y.derivative(1)(s)
            d2x_ds2 = spline_x.derivative(2)(s)
            d2y_ds2 = spline_y.derivative(2)(s)
            
            # Curvature formula: κ = (x'y'' - y'x'') / (x'² + y'²)^(3/2)
            numerator = dx_ds * d2y_ds2 - dy_ds * d2x_ds2
            denominator = (dx_ds**2 + dy_ds**2)**(1.5)
            
            curvature = numerator / (denominator + 1e-10)
            
        except Exception:
            # fallback (should be rare though (hopefully))
            curvature = self._compute_curvature_finite_diff(x, y)
        
        return curvature
    
    def _compute_curvature_finite_diff(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """sorta fallback curvature computation using finite differences"""
        dx = np.gradient(x)
        dy = np.gradient(y)
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        
        numerator = dx * ddy - dy * ddx
        denominator = (dx**2 + dy**2)**(1.5)
        
        curvature = numerator / (denominator + 1e-10)
        return curvature
    
    def _create_track_arrays(self):
        n = len(self.segments)
        
        s = np.array([seg.distance for seg in self.segments])
        radius = np.array([seg.radius for seg in self.segments])
        curvature = np.array([seg.curvature for seg in self.segments])
        gradient = np.array([seg.gradient for seg in self.segments])
        x = np.array([seg.x for seg in self.segments])
        y = np.array([seg.y for seg in self.segments])
        sector = np.array([seg.sector for seg in self.segments])
        vertical_curvature = np.array([seg.vertical_curvature for seg in self.segments])
        
        # Identify braking/acceleration zones
        is_braking = np.zeros(n, dtype=bool)
        is_accel = np.zeros(n, dtype=bool)
        
        for i in range(n - 1):
            # Look ahead for braking zones
            look_ahead = min(15, n - i)
            future_radii = radius[i:i+look_ahead]
            min_future_r = np.min(future_radii)
            
            if radius[i] > 200 and min_future_r < 100:
                is_braking[i] = True
            
            # Look behind for acceleration zones
            if i > 5:
                past_radii = radius[i-5:i]
                if np.min(past_radii) < 100 and radius[i] > 150:
                    is_accel[i] = True
        
        self.track_data = TrackData(
            s=s,
            ds=self.ds,
            n_points=n,
            total_length=self.total_length,
            radius=radius,
            curvature=curvature,
            gradient=gradient,
            x=x,
            y=y,
            sector=sector,
            is_braking_zone=is_braking,
            is_acceleration_zone=is_accel,
            vertical_curvature=vertical_curvature,
        )
    
    def get_interpolators(self) -> dict:
        """Create interpolation functions for continuous access to track properties"""
        
        if self.track_data is None:
            raise RuntimeError("Track data not loaded")
        
        s = self.track_data.s
        
        return {
            'radius': interp1d(s, self.track_data.radius,
                              kind='linear', fill_value='extrapolate'),
            'curvature': interp1d(s, self.track_data.curvature,
                                  kind='linear', fill_value='extrapolate'),
            'gradient': interp1d(s, self.track_data.gradient,
                                 kind='linear', fill_value='extrapolate'),
        }
    
    def _get_sector(self, distance: float) -> int:
        """Determine track sector from distance"""
        if self.total_length <= 0:
            return 1
        sector_length = self.total_length / 3
        return min(int(distance / sector_length) + 1, 3)
    
    def get_segment_at_distance(self, distance: float) -> TrackSegment:
        """Get track segment at given distance (with wrapping)"""
        distance = distance % self.total_length
        idx = int(distance / self.ds) % len(self.segments)
        return self.segments[idx]
    
    def _print_track_stats(self):
        """Print track statistics for debugging"""
        radii = [seg.radius for seg in self.segments]
        corner_count = sum(1 for r in radii if r < 500)
        
        print(f"   Track Statistics:")
        print(f"     Total length: {self.total_length:.0f} m")
        print(f"     Segments: {len(self.segments)} (at {self.ds}m intervals)")
        print(f"     Corners: {corner_count} ({100*corner_count/len(self.segments):.0f}%)")
        print(f"     Tightest: {min(radii):.0f} m radius")
        
        corner_radii = [r for r in radii if r < 500]
        if corner_radii:
            print(f"     Avg corner: {np.mean(corner_radii):.0f} m radius")
    