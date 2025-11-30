import numpy as np
import pandas as pd
import fastf1 as ff1
from dataclasses import dataclass
from typing import List, Optional, Tuple
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

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
    
    @property
    def is_corner(self) -> bool:
        return not self.is_straight


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
    
    # Speed limits from curvature
    v_max_corner: np.ndarray   # Max speed from lateral grip
    
    # Sector information
    sector: np.ndarray
    
    # Braking/acceleration zones (precomputed)
    is_braking_zone: np.ndarray
    is_acceleration_zone: np.ndarray

class F1TrackModel:
    """Track model built from FastF1 telemetry data"""
    
    def __init__(self, year: int, gp: str, session: str = 'Q', ds: float = 5.0):
        self.year = year
        self.gp = gp
        self.session_type = session
        self.ds = ds # spatial discretization step (meters)

        self.segments: List[TrackSegment] = []
        self.track_data: Optional[TrackData] = None
        self.total_length: float = 0.0
        
        # Raw telemetry for visualization
        self.telemetry_raw: Optional[pd.DataFrame] = None
        
    def load_from_fastf1(self, driver: Optional[str] = None):
        """Load track data from FastF1 API"""
        
        ff1.Cache.enable_cache('./data/cache')
        session = ff1.get_session(self.year, self.gp, self.session_type)
        session.load()
        
        if driver:
            lap = session.laps.pick_driver(driver).pick_fastest()
        else:
            lap = session.laps.pick_fastest()
            
        telemetry = lap.get_telemetry()
        self.telemetry_data = telemetry  # Store for visualization
        
        # Process telemetry to extract track characteristics
        self._process_telemetry(telemetry)
        
        self._create_track_arrays()
        
        print(f"   Processed {len(self.segments)} track segments")
        return self
    
    def _process_telemetry(self, telemetry: pd.DataFrame):
        """Convert raw telemetry to track segments using interpolation"""
        
        # 1. Extract raw data
        x = telemetry['X'].values
        y = telemetry['Y'].values
        distances = telemetry['Distance'].values
        speeds = telemetry['Speed'].values
        
        # Clean NaNs
        valid = ~(np.isnan(x) | np.isnan(y) | np.isnan(distances) | np.isnan(speeds))
        x, y, distances, speeds = x[valid], y[valid], distances[valid], speeds[valid]
        
        self.total_length = distances[-1]
        
        # 2. Compute curvature on the raw telemetry points
        curvature_raw, radius_raw = self._compute_curvature(x, y, distances, speeds)
        
        # 3. Create a fixed grid of distances (0, 5, 10, 15...)
        # This ensures we never skip a segment
        s_grid = np.arange(0, self.total_length, self.ds)
        
        # 4. Interpolate all properties onto this grid
        # We use the raw 'distances' as the x-axis for interpolation
        radius_interp = np.interp(s_grid, distances, radius_raw)
        curvature_interp = np.interp(s_grid, distances, curvature_raw)
        x_interp = np.interp(s_grid, distances, x)
        y_interp = np.interp(s_grid, distances, y)
        
        # 5. Build segments from interpolated data
        self.segments = [] # Clear old segments
        
        for i, s in enumerate(s_grid):
            # Calculate length (last segment might be shorter)
            length = min(self.ds, self.total_length - s)
            
            # Determine sector (simple logic: 3 equal parts)
            sector = self._get_sector(s)
            
            segment = TrackSegment(
                distance=float(s),
                length=float(length),
                radius=float(radius_interp[i]),
                curvature=float(curvature_interp[i]),
                gradient=0.0,
                x=float(x_interp[i]),
                y=float(y_interp[i]),
                sector=sector
            )
            self.segments.append(segment)
            
        self._create_track_arrays()
        self._print_track_stats()
    
    def _compute_curvature(self, x: np.ndarray, y: np.ndarray, 
                           distances: np.ndarray, speeds: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute curvature using hybrid method (GPS geometry + Speed limits).
        """
        # 1. GPS-based Curvature
        # Smooth the coordinates first to reduce noise amplification
        window_coords = 9 # Small smoothing on raw coords
        x_smooth = savgol_filter(x, window_coords, 2)
        y_smooth = savgol_filter(y, window_coords, 2)

        dx = np.gradient(x_smooth)
        dy = np.gradient(y_smooth)
        heading = np.arctan2(dy, dx)
        heading_unwrap = np.unwrap(heading)
        
        ds = np.gradient(distances)
        ds = np.maximum(ds, 0.1)
        
        d_heading = np.gradient(heading_unwrap)
        curvature_gps = d_heading / ds
        
        # Smooth the GPS curvature
        window_curv = 21
        if len(curvature_gps) > window_curv:
            curvature_gps = savgol_filter(curvature_gps, window_curv, 3)
            
        radius_gps = 1.0 / (np.abs(curvature_gps) + 1e-6)

        # 2. Physics-based Radius Estimation
        # v^2 / r = a_lat  ->  r = v^2 / a_lat
        # We assume a max lateral G. F1 cars do ~4G to 5G. 
        # We use a conservative value to "clamp" the GPS noise.
        LATERAL_G_TARGET = 4.0 
        g = 9.81
        speeds_ms = speeds / 3.6
        radius_physics = (speeds_ms**2) / (LATERAL_G_TARGET * g + 1e-6)
        
        radius_combined = np.minimum(radius_gps, radius_physics * 1.5) 
        # Multiplier 1.5 allows for some variance (e.g. camber/banking/driver not pushing)
        
        # Clip to limits
        radius_combined = np.clip(radius_combined, 10, 10000)
        
        # Final smoothing
        radius_final = savgol_filter(radius_combined, 15, 2)
        curvature_final = 1.0 / radius_final
        
        return curvature_final, radius_final
    
    def _create_track_arrays(self):
        """Create numpy arrays for efficient optimization"""
        
        n = len(self.segments)
        
        s = np.array([seg.distance for seg in self.segments])
        radius = np.array([seg.radius for seg in self.segments])
        curvature = np.array([seg.curvature for seg in self.segments])
        gradient = np.array([seg.gradient for seg in self.segments])
        x = np.array([seg.x for seg in self.segments])
        y = np.array([seg.y for seg in self.segments])
        sector = np.array([seg.sector for seg in self.segments])
        
        v_max_corner = np.zeros(n)
        
        # Identify braking/acceleration zones using Telemetry Logic
        # We need to map the raw telemetry acceleration to our segments
        is_braking = np.zeros(n, dtype=bool)
        is_accel = np.zeros(n, dtype=bool)
        
        # Calculate acceleration profile from reference lap
        # Note: We need to reconstruct speed profile on the segment grid
        # The easiest way is to use the max speed allowed or observed
        
        # For a robust 'is_braking_zone', we look at the Radius Derivative
        # or simply where Radius transitions from Large -> Small
        
        for i in range(n - 1):
            curr_r = radius[i]
            next_r = radius[i+1]
            
            # Braking: Approaching a corner
            # Look ahead logic is okay if radius is clean
            look_ahead = min(15, n - i)
            future_radii = radius[i:i+look_ahead]
            min_future_r = np.min(future_radii)
            
            # If we are currently straight-ish (>150m) and approaching tight (<100m)
            if curr_r > 150 and min_future_r < 100:
                is_braking[i] = True
                
            # Acceleration: Leaving a corner
            # If we were tight (<100m) and now opening up (>150m)
            if i > 5:
                past_radii = radius[i-5:i]
                if np.min(past_radii) < 100 and curr_r > 120:
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
            v_max_corner=v_max_corner,
            sector=sector,
            is_braking_zone=is_braking,
            is_acceleration_zone=is_accel,
        )
    
    def compute_speed_limits(self, vehicle_config) -> np.ndarray:
        """
        Compute maximum cornering speeds using vehicle configuration.
        
        Uses the formula: v_max = sqrt(mu * g * R * (1 + aero_factor))
        where aero_factor accounts for downforce increasing normal force.
        """
        if self.track_data is None:
            raise RuntimeError("Track data not loaded")
        
        v_max = np.zeros(self.track_data.n_points)
        
        for i, radius in enumerate(self.track_data.radius):
            v_max[i] = vehicle_config.get_max_cornering_speed(radius)
        
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
            'v_max': interp1d(s, self.track_data.v_max_corner,
                              kind='linear', fill_value='extrapolate'),
        }
    
    def _get_sector(self, distance: float) -> int:
        """Determine track sector from distance"""
        if self.total_length <= 0:
            return 1
        sector_length = self.total_length / 3
        if sector_length <= 0:
            return 1
        return min(int(distance / sector_length) + 1, 3)
    
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
    
    def get_segment_at_distance(self, distance: float) -> TrackSegment:
        """Get track segment at given distance (with wrapping)"""
        distance = distance % self.total_length
        idx = int(distance / self.ds) % len(self.segments)
        return self.segments[idx]