import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class TrackSegment:
    """Represents a segment of the track"""
    length: float          # meters
    radius: float          # meters (inf for straight)
    gradient: float        # radians
    speed_limit: float     # m/s (from track limits/safety)
    sector: int           # DRS zones, etc.
    
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
        try:
            import fastf1
            from scipy.signal import savgol_filter
            
            session = fastf1.get_session(self.year, self.gp, self.session_type)
            session.load()
            
            if driver:
                lap = session.laps.pick_driver(driver).pick_fastest()
            else:
                lap = session.laps.pick_fastest()
                
            telemetry = lap.get_telemetry()
            self.telemetry_data = telemetry  # Store for visualization
            
            # Process telemetry to extract track characteristics
            self._process_telemetry_improved(telemetry)
            
            print(f"   Processed {len(self.segments)} track segments")
            
        except ImportError:
            print("FastF1 not available, using synthetic track data")
            self.create_synthetic_track()
        except Exception as e:
            print(f"Error loading FastF1 data: {e}")
            print("Falling back to synthetic track")
            self.create_synthetic_track()
    
    def _process_telemetry_improved(self, telemetry: pd.DataFrame):
        """Convert telemetry to track segments with real curvature"""
        import numpy as np
        from scipy.signal import savgol_filter
        
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
    
    def visualize_track(self, save_path: str = None):
        """Visualize the track layout and properties"""
        if self.telemetry_data is None:
            print("No telemetry data available for visualization")
            return
        
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        x = self.telemetry_data['X'].values
        y = self.telemetry_data['Y'].values
        speeds = self.telemetry_data['Speed'].values
        distances = self.telemetry_data['Distance'].values
        
        # Plot 1: Track layout colored by speed
        ax = axes[0, 0]
        scatter = ax.scatter(x, y, c=speeds, cmap='RdYlGn', s=1, alpha=0.6)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Track Layout (colored by speed)')
        ax.set_aspect('equal')
        plt.colorbar(scatter, ax=ax, label='Speed (km/h)')
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Track layout colored by segment radius
        ax = axes[0, 1]
        segment_distances = np.array([sum(seg.length for seg in self.segments[:i]) 
                                     for i in range(len(self.segments))])
        segment_radii = np.array([seg.radius for seg in self.segments])
        
        # Map segment radii to telemetry points
        radii_at_points = np.interp(distances, segment_distances, segment_radii)
        
        scatter = ax.scatter(x, y, c=np.log10(radii_at_points), 
                           cmap='coolwarm', s=1, alpha=0.6, vmin=1, vmax=4)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Track Layout (colored by corner radius)')
        ax.set_aspect('equal')
        cbar = plt.colorbar(scatter, ax=ax, label='log10(Radius (m))')
        cbar.ax.set_yticklabels(['10m', '100m', '1km', '10km'])
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Speed profile vs distance
        ax = axes[1, 0]
        ax.plot(distances, speeds, 'b-', linewidth=1, alpha=0.7)
        ax.set_xlabel('Distance (m)')
        ax.set_ylabel('Speed (km/h)')
        ax.set_title('Speed Profile')
        ax.grid(True, alpha=0.3)
        
        # Highlight corners
        for seg in self.segments:
            if seg.radius < 100:  # Tight corners
                seg_start = sum(s.length for s in self.segments[:self.segments.index(seg)])
                ax.axvspan(seg_start, seg_start + seg.length, 
                          alpha=0.1, color='red', label='Hairpin' if seg == self.segments[0] else '')
        
        # Plot 4: Radius vs distance
        ax = axes[1, 1]
        ax.plot(segment_distances, segment_radii, 'r-', linewidth=2)
        ax.set_xlabel('Distance (m)')
        ax.set_ylabel('Corner Radius (m)')
        ax.set_title('Track Curvature')
        ax.set_ylim([0, 1000])
        ax.axhline(y=100, color='k', linestyle='--', alpha=0.3, label='Slow corner')
        ax.axhline(y=500, color='k', linestyle=':', alpha=0.3, label='Fast corner')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Track Analysis: {self.year} {self.gp}', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Track visualization saved to {save_path}")
        
        return fig
    