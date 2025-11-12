import numpy as np
import matplotlib.pyplot as plt

from models import F1TrackModel


def visualize_track(track_model: F1TrackModel, save_path: str = "../figures/track_analysis.png"):
    """Visualize the track layout and properties"""
    if track_model.telemetry_data is None:
        print("No telemetry data available for visualization")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    x = track_model.telemetry_data['X'].values
    y = track_model.telemetry_data['Y'].values
    speeds = track_model.telemetry_data['Speed'].values
    distances = track_model.telemetry_data['Distance'].values
    
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
    segment_distances = np.array([sum(seg.length for seg in track_model.segments[:i]) 
                                 for i in range(len(track_model.segments))])
    segment_radii = np.array([seg.radius for seg in track_model.segments])
    
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
    for seg in track_model.segments:
        if seg.radius < 100:  # Tight corners
            seg_start = sum(s.length for s in track_model.segments[:track_model.segments.index(seg)])
            ax.axvspan(seg_start, seg_start + seg.length, 
                      alpha=0.1, color='red', label='Hairpin' if seg == track_model.segments[0] else '')
    
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
    
    plt.suptitle(f'Track Analysis: {track_model.year} {track_model.gp}', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Track visualization saved to {save_path}")
    
    return fig