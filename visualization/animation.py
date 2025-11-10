from typing import Dict
import numpy as np

from ..models import F1TrackModel


def visualize_lap_animated(track_model: F1TrackModel, 
                          results: Dict, 
                          strategy_name: str = "MPC",
                          save_path: str = "../figures/lap_animation.gif"):
    """Create animated visualization of car going around track"""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.patches import Circle
    
    if track_model.telemetry_data is None:
        print("No track data for visualization")
        return
    
    # Track layout
    x_track = track_model.telemetry_data['X'].values
    y_track = track_model.telemetry_data['Y'].values
    distances_track = track_model.telemetry_data['Distance'].values
    
    # Car position over time
    states = results['states']
    times = results['times']
    car_distances = states[:, 0]
    car_speeds = states[:, 1]
    car_soc = states[:, 2]
    
    # Interpolate car position to track coordinates
    car_x = np.interp(car_distances, distances_track, x_track)
    car_y = np.interp(car_distances, distances_track, y_track)
    
    # Create figure
    fig = plt.figure(figsize=(16, 9))
    
    # Main track view
    ax_track = plt.subplot2grid((2, 3), (0, 0), colspan=2, rowspan=2)
    
    # Telemetry panels
    ax_speed = plt.subplot2grid((2, 3), (0, 2))
    ax_soc = plt.subplot2grid((2, 3), (1, 2))
    
    # Plot track
    ax_track.plot(x_track, y_track, 'k-', linewidth=2, alpha=0.3, label='Track')
    ax_track.set_xlabel('X (m)')
    ax_track.set_ylabel('Y (m)')
    ax_track.set_title(f'{strategy_name} Strategy - Live Lap')
    ax_track.set_aspect('equal')
    ax_track.grid(True, alpha=0.3)
    
    # Car marker
    car_marker = Circle((car_x[0], car_y[0]), 20, color='red', zorder=10)
    ax_track.add_patch(car_marker)
    
    # Trail (show last 100m)
    trail_line, = ax_track.plot([], [], 'r-', linewidth=3, alpha=0.5)
    
    # Speed plot
    ax_speed.set_xlim([0, times[-1]])
    ax_speed.set_ylim([0, max(car_speeds) * 3.6 * 1.1])
    ax_speed.set_xlabel('Time (s)')
    ax_speed.set_ylabel('Speed (km/h)')
    ax_speed.set_title('Speed')
    ax_speed.grid(True, alpha=0.3)
    speed_line, = ax_speed.plot([], [], 'g-', linewidth=2)
    speed_point = ax_speed.scatter([], [], c='green', s=100, zorder=10)
    
    # SOC plot
    ax_soc.set_xlim([0, times[-1]])
    ax_soc.set_ylim([0, 100])
    ax_soc.set_xlabel('Time (s)')
    ax_soc.set_ylabel('SOC (%)')
    ax_soc.set_title('Battery State of Charge')
    ax_soc.axhline(y=10, color='r', linestyle='--', alpha=0.3)
    ax_soc.axhline(y=90, color='r', linestyle='--', alpha=0.3)
    ax_soc.grid(True, alpha=0.3)
    soc_line, = ax_soc.plot([], [], 'r-', linewidth=2)
    soc_point = ax_soc.scatter([], [], c='red', s=100, zorder=10)
    
    # Time and info text
    info_text = ax_track.text(0.02, 0.98, '', transform=ax_track.transAxes,
                             fontsize=12, verticalalignment='top',
                             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    def init():
        trail_line.set_data([], [])
        speed_line.set_data([], [])
        soc_line.set_data([], [])
        return trail_line, speed_line, soc_line, car_marker, info_text
    
    def animate(frame):
        # Update car position
        car_marker.center = (car_x[frame], car_y[frame])
        
        # Update trail (last 50 points)
        trail_start = max(0, frame - 50)
        trail_line.set_data(car_x[trail_start:frame+1], car_y[trail_start:frame+1])
        
        # Update speed plot
        speed_line.set_data(times[:frame+1], car_speeds[:frame+1] * 3.6)
        speed_point.set_offsets([[times[frame], car_speeds[frame] * 3.6]])
        
        # Update SOC plot
        soc_line.set_data(times[:frame+1], car_soc[:frame+1] * 100)
        soc_point.set_offsets([[times[frame], car_soc[frame] * 100]])
        
        # Update info text
        progress = (car_distances[frame] / track_model.total_length) * 100
        info_text.set_text(
            f'Time: {times[frame]:.1f}s\n'
            f'Distance: {car_distances[frame]:.0f}m ({progress:.1f}%)\n'
            f'Speed: {car_speeds[frame]*3.6:.0f} km/h\n'
            f'SOC: {car_soc[frame]*100:.0f}%'
        )
        
        return trail_line, speed_line, soc_line, car_marker, speed_point, soc_point, info_text
    
    # Create animation
    # Subsample frames for smoother playback (every 5th frame = 0.5s steps)
    frames = range(0, len(times), 5)
    
    anim = FuncAnimation(fig, animate, init_func=init, frames=frames,
                        interval=50, blit=True, repeat=True)
    
    plt.tight_layout()
    
    if save_path:
        print(f"Saving animation (this may take a while)...")
        anim.save(save_path, writer='pillow', fps=20, dpi=100)
        print(f"Animation saved to {save_path}")
    
    return fig, anim
