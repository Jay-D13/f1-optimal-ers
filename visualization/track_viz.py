import numpy as np
import matplotlib.pyplot as plt
from itertools import groupby

def visualize_track(track_model, track_name=None, driver_name=None, save_path=None):
    if track_model.telemetry_data is None:
        print("No telemetry data available for visualization")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Extract raw telemetry
    x = track_model.telemetry_data['X'].values
    y = track_model.telemetry_data['Y'].values
    speeds = track_model.telemetry_data['Speed'].values
    distances = track_model.telemetry_data['Distance'].values
    
    # --- Plot 1: Track layout colored by speed ---
    ax = axes[0, 0]
    scatter = ax.scatter(x, y, c=speeds, cmap='RdYlGn', s=1, alpha=0.6)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Track Layout (colored by speed)')
    ax.set_aspect('equal')
    plt.colorbar(scatter, ax=ax, label='Speed (km/h)')
    ax.grid(True, alpha=0.3)
    
    # --- Plot 2: Track layout colored by segment radius ---
    ax = axes[0, 1]
    
    # pre-calculated distance from the segment object
    segment_distances = np.array([seg.distance for seg in track_model.segments])
    segment_radii = np.array([seg.radius for seg in track_model.segments])
    
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
    
    # --- Plot 3: Speed Profile ---
    ax = axes[1, 0]
    ax.plot(distances, speeds, 'b-', linewidth=1, alpha=0.7)
    
    # which indices are corners
    is_corner = [seg.radius < 100 for seg in track_model.segments]
    
    current_idx = 0
    for is_tight, group in groupby(is_corner):
        count = sum(1 for _ in group)
        if is_tight:
            start_dist = track_model.segments[current_idx].distance
            # The end distance is the start of the next block
            end_dist = start_dist + (count * track_model.ds)
            
            ax.axvspan(start_dist, end_dist, alpha=0.1, color='red')
            
        current_idx += count

    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Speed (km/h)')
    ax.set_title('Speed Profile (Red = Radius < 100m)')
    ax.grid(True, alpha=0.3)
    
    # --- Plot 4: Radius vs Distance ---
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
    
    plt.suptitle(f'{track_name} Analysis of {driver_name} fastest lap in {track_model.year}', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Track visualization saved to {save_path}")
    
    return fig