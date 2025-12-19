"""
Supports:
1. FastF1 telemetry (GPS-derived curvature)
2. TUMFTM minimum curvature racelines
3. Manual track definitions
"""
import numpy as np
import pandas as pd
import fastf1 as ff1
from dataclasses import dataclass
from typing import List, Optional, Tuple
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d, UnivariateSpline

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
    
    # Speed limits (computed with vehicle model)
    v_max_corner: np.ndarray
    
    # Track features
    sector: np.ndarray
    is_braking_zone: np.ndarray
    is_acceleration_zone: np.ndarray

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
        
    def load_from_fastf1(self, driver: Optional[str] = None):
        """Load track data from FastF1 API's telemetry"""
        
        ff1.Cache.enable_cache('./data/cache')
        session = ff1.get_session(self.year, self.gp, self.session_type)
        session.load()
        
        if driver:
            lap = session.laps.pick_driver(driver).pick_fastest()
        else:
            lap = session.laps.pick_fastest()
            
        telemetry = lap.get_telemetry()
        self.telemetry_data = telemetry  # Store for visualization
        
        # Extract data
        x = telemetry['X'].values
        y = telemetry['Y'].values
        distances = telemetry['Distance'].values
        speeds = telemetry['Speed'].values
        
        # Clean NaNs
        valid = ~(np.isnan(x) | np.isnan(y) | np.isnan(distances))
        x, y, distances = x[valid], y[valid], distances[valid]
        speeds = speeds[valid] if len(speeds) == len(valid) else speeds[~np.isnan(speeds)]
        
        self.total_length = distances[-1]
        
        # Compute curvature with smoothing
        curvature = self._compute_curvature_smoothed(x, y, distances, speeds)
        radius = 1.0 / (np.abs(curvature) + 1e-6)
        radius = np.clip(radius, 10, 10000)
        
        # Resample to uniform ds
        s_uniform = np.arange(0, self.total_length, self.ds)
        n_points = len(s_uniform)
        
        x_interp = np.interp(s_uniform, distances, x)
        y_interp = np.interp(s_uniform, distances, y)
        radius_interp = np.interp(s_uniform, distances, radius)
        curvature_interp = np.interp(s_uniform, distances, curvature)
        
        # Build segments
        self.segments = []
        for i in range(n_points):
            segment = TrackSegment(
                distance=s_uniform[i],
                length=self.ds,
                radius=radius_interp[i],
                curvature=curvature_interp[i],
                gradient=0.0,
                x=x_interp[i],
                y=y_interp[i],
                sector=self._get_sector(s_uniform[i]),
            )
            self.segments.append(segment)
        
        self._create_track_arrays()
        self.data_source = 'fastf1'
        
        print(f"   ✓ Loaded {n_points} points from FastF1")
        return self, lap['Driver']

    def _compute_curvature_smoothed(self, 
                                    x: np.ndarray, 
                                    y: np.ndarray, 
                                    distances: np.ndarray, 
                                    speeds: np.ndarray
                                ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute curvature using hybrid method (GPS geometry + speed limits) -> physics basically
        """
        # Smooth coordinates
        window = min(15, len(x) // 10)
        if window % 2 == 0:
            window += 1
        window = max(window, 5)
        
        x_smooth = savgol_filter(x, window, 3)
        y_smooth = savgol_filter(y, window, 3)
        
        # Compute curvature from heading changes
        dx = np.gradient(x_smooth)
        dy = np.gradient(y_smooth)
        heading = np.arctan2(dy, dx)
        heading = np.unwrap(heading)
        
        ds = np.gradient(distances)
        ds = np.maximum(ds, 0.1)
        
        d_heading = np.gradient(heading)
        curvature_gps = d_heading / ds
        
        # Smooth curvature
        curvature_gps = savgol_filter(curvature_gps, window, 3)
        
        # Physics-based bound from speed
        # v²/R = a_lat -> R = v²/a_lat -> κ = a_lat/v²
        # gonna assume max 4G lateral cause sometimes in life... you just gotta pick a number
        g = 9.81
        a_lat_max = 4.0 * g
        speeds_ms = speeds / 3.6 if np.mean(speeds) > 50 else speeds
        speeds_ms = np.maximum(speeds_ms, 10)
        
        kappa_physics_max = a_lat_max / speeds_ms**2
        
        # bound GPS curvature by physics
        curvature = np.sign(curvature_gps) * np.minimum(np.abs(curvature_gps), kappa_physics_max)
        
        return curvature
    
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
            v_max_corner=np.zeros(n),  # Computed later with vehicle
            sector=sector,
            is_braking_zone=is_braking,
            is_acceleration_zone=is_accel,
        )
    
    def compute_speed_limits(self, vehicle_config, tire_params=None) -> np.ndarray:
        if self.track_data is None:
            raise RuntimeError("Track data not loaded")
        
        # Initialize tire params if not passed
        # (Assuming you can import TireParameters or it's attached to vehicle_config)
        # Ideally, pass the tire_params object that matches your vehicle model
        if tire_params is None:
            # If TireParameters is in vehicle.py, you might need to import it
            from config import TireParameters 
            tire_params = TireParameters()
            
        v_max = np.zeros(self.track_data.n_points)
        
        for i, radius in enumerate(self.track_data.radius):
            # Pass the tire params here
            v_max[i] = vehicle_config.get_max_cornering_speed(radius, tire_params)
        
        self.track_data.v_max_corner = v_max
        return v_max
    
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
    