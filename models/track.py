import numpy as np
import pandas as pd
import fastf1 as ff1
from dataclasses import dataclass
from typing import List, Optional
from scipy.signal import savgol_filter

@dataclass
class TrackSegment:
    """Represents a segment of the track"""
    length: float          # meters
    radius: float          # meters (inf for straight)
    gradient: float        # radians
    speed_limit: float     # m/s (from track limits/safety)
    sector: int            # sector number (1, 2, or 3)
    
    @property
    def is_straight(self) -> bool:
        return self.radius > 1000 or np.isinf(self.radius)
    
    @property
    def is_braking_zone(self) -> bool:
        return self.radius < 100  # Tight corners require heavy braking
    

class F1TrackModel:
    """Track model built from FastF1 telemetry data"""
    
    def __init__(self, year: int, gp: str, session: str = 'Q'):
        self.year = year
        self.gp = gp
        self.session_type = session
        self.segments: List[TrackSegment] = []
        self.total_length: float = 0.0
        
        # Store raw telemetry for visualization
        self.telemetry_data = None
        
    def load_from_fastf1(self, driver: Optional[str] = None):
        """Load track data from FastF1 API"""
        
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
        
        print(f"   Processed {len(self.segments)} track segments")
    
    def _process_telemetry(self, telemetry: pd.DataFrame):
        """Convert telemetry to track segments with real curvature"""        
        # Extract data
        x = telemetry['X'].values
        y = telemetry['Y'].values
        speeds = telemetry['Speed'].values  # km/h
        distances = telemetry['Distance'].values
        
        # Calculate curvature from GPS
        dx = np.gradient(x)
        dy = np.gradient(y)
        heading = np.arctan2(dy, dx)
        
        # Unwrap heading to handle 2π discontinuities
        heading_unwrap = np.unwrap(heading)
        
        # Curvature = rate of change of heading with distance
        d_heading = np.gradient(heading_unwrap)
        ds = np.gradient(distances)
        curvature = d_heading / (ds + 1e-10)
        
        # Smooth curvature (GPS is noisy)
        window = min(51, len(curvature) // 3)  # Adaptive window
        if window % 2 == 0:
            window += 1  # Must be odd
        curvature_smooth = savgol_filter(curvature, window, 3)
        
        # Convert curvature to radius
        radius_gps = np.abs(1 / (curvature_smooth + 1e-10))
        radius_gps = np.clip(radius_gps, 15, 10000)  # Physical limits
        
        # Verify with speed-based estimation
        speeds_ms = speeds / 3.6
        lateral_g = 4.0  # Typical F1 lateral acceleration
        radius_speed = speeds_ms**2 / (lateral_g * 9.81 + 1e-10)
        radius_speed = np.clip(radius_speed, 15, 10000)
        
        # Take minimum (most conservative - tightest corner)
        radius = np.minimum(radius_gps, radius_speed)
        
        # Smooth final radius
        radius = savgol_filter(radius, window, 3)
        
        # Create segments
        segment_length = 50  # Target segment length in meters
        self.total_length = distances[-1]
        
        current_distance = 0
        i = 0
        
        while current_distance < self.total_length and i < len(distances):
            # Find telemetry index for this segment
            segment_start = current_distance
            segment_end = current_distance + segment_length
            
            # Find indices in this segment
            mask = (distances >= segment_start) & (distances < segment_end)
            if not mask.any():
                i += 1
                current_distance += segment_length
                continue
            
            # Average properties over segment
            segment_radius = np.mean(radius[mask])
            segment_speed = np.mean(speeds[mask])
            print(segment_speed)
            
            # Create segment
            seg = TrackSegment(
                length=min(segment_length, self.total_length - current_distance),
                radius=float(segment_radius),
                gradient=0.0,  # Would need Z data for this
                speed_limit=1000.0,  # No regulatory limit
                sector=self._get_sector(current_distance)
            )
            self.segments.append(seg)
            
            current_distance += segment_length
            i += 1
        
        # Diagnostics
        corner_segments = sum(1 for seg in self.segments if seg.radius < 500)
        print(f"   Track statistics:")
        print(f"     Total length: {self.total_length:.0f}m")
        print(f"     Corner segments: {corner_segments}/{len(self.segments)} ({100*corner_segments/len(self.segments):.0f}%)")
        print(f"     Tightest corner: {min(seg.radius for seg in self.segments):.0f}m")
        print(f"     Average corner radius: {np.mean([seg.radius for seg in self.segments if seg.radius < 500]):.0f}m")
        
    def _get_sector(self, distance):
        """Determine track sector from distance"""
        # Monaco has 3 sectors (approximate thirds)
        sector_length = self.total_length / 3
        return min(int(distance / sector_length) + 1, 3)
    
