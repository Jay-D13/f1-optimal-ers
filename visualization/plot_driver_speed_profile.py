"""
Pretty Speed Profile Visualization for Specific Driver
"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import groupby
from pathlib import Path
import sys
from scipy.ndimage import uniform_filter1d

sys.path.insert(0, str(Path(__file__).parent))

from models import F1TrackModel

def plot_multi_driver_speed_profile(track_name='Monaco', year=2024, 
                                    drivers=['VER', 'LEC', 'NOR'], save_path=None):
    """
    Create a beautiful speed profile comparing multiple drivers.
    Shows each driver's speed and their individual braking zones.
    
    Args:
        track_name: Name of the track (e.g., 'Monaco', 'Monza')
        year: Season year
        drivers: List of driver codes (e.g., ['VER', 'HAM', 'LEC'])
        save_path: Optional path to save the figure
    
    Returns:
        matplotlib Figure
    """
    
    # Styling
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 14,
        'axes.titlesize': 20,
        'axes.labelsize': 16,
        'legend.fontsize': 13,
        'lines.linewidth': 3.5,
        'axes.linewidth': 1.5,
        'grid.alpha': 0.3,
    })
    
    # Driver colors (F1 team colors)
    driver_colors = {
        'VER': '#3671C6',  # Red Bull blue
        'PER': '#3671C6',
        'HAM': '#27F4D2',  # Mercedes teal
        'RUS': '#27F4D2',
        'LEC': '#E8002D',  # Ferrari red
        'SAI': '#E8002D',
        'NOR': '#FF8000',  # McLaren orange
        'PIA': '#FF8000',
        'ALO': '#229971',  # Aston Martin green
        'STR': '#229971',
        'ALB': '#012564',  # Williams blue
        'OCO': '#FF87BC',  # Alpine pink
        'GAS': '#FF87BC',
    }
    
    # Driver full names
    driver_names = {
        'VER': 'Max Verstappen',
        'LEC': 'Charles Leclerc',
        'NOR': 'Lando Norris',
        'HAM': 'Lewis Hamilton',
        'PER': 'Sergio Pérez',
        'SAI': 'Carlos Sainz',
        'RUS': 'George Russell',
        'ALO': 'Fernando Alonso',
    }
    
    print(f"\n{'='*70}")
    print(f"Loading {track_name} telemetry for {len(drivers)} drivers...")
    print(f"{'='*70}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(18, 10))
    
    # Store data for each driver
    driver_data = {}
    
    for driver in drivers:
        print(f"\n  Loading {driver}...")
        
        # Load track with specific driver
        track = F1TrackModel(year, track_name, 'Q', ds=5.0)
        try:
            track.load_from_fastf1(driver=driver)
        except Exception as e:
            print(f"  ⚠ Could not load {driver}: {e}")
            continue
        
        # Extract telemetry
        telemetry = track.telemetry_data
        speeds = telemetry['Speed'].values
        distances = telemetry['Distance'].values
        
        # Calculate lap time
        try:
            lap_time = (telemetry['Time'].iloc[-1] - telemetry['Time'].iloc[0]).total_seconds()
        except:
            lap_time = None
        
        driver_data[driver] = {
            'speeds': speeds,
            'distances': distances,
            'lap_time': lap_time,
            'track': track,
            'telemetry': telemetry,
        }
    
    if not driver_data:
        print("⚠ No driver data loaded!")
        return None
    
    # Find fastest driver for sorting
    fastest_driver = min(driver_data.keys(), 
                        key=lambda d: driver_data[d]['lap_time'] or float('inf'))
    
    print(f"\n  Fastest: {fastest_driver} ({driver_data[fastest_driver]['lap_time']:.3f}s)")
    
    # Plot each driver
    for i, driver in enumerate(drivers):
        if driver not in driver_data:
            continue
        
        data = driver_data[driver]
        driver_color = driver_colors.get(driver, '#2962FF')
        driver_name = driver_names.get(driver, driver)
        
        # Plot speed profile
        lap_time_str = f"{data['lap_time']:.3f}s" if data['lap_time'] else "N/A"
        label = f"{driver_name}: {lap_time_str}"
        if driver == fastest_driver:
            label += " 🏆"
        
        ax.plot(data['distances'] / 1000, data['speeds'], 
               color=driver_color, linewidth=4, 
               label=label, alpha=0.85, zorder=3 + i)
        
        # Add braking zones for this driver (more transparent, color-coded)
        # Detect braking: significant speed drops over a sustained distance
        speeds = data['speeds']
        distances_km = data['distances'] / 1000
        
        # Calculate speed derivative (change per km, not per point)
        # Smooth the speed first to avoid noise
        from scipy.ndimage import uniform_filter1d
        speeds_smooth = uniform_filter1d(speeds, size=10, mode='nearest')
        
        # Calculate deceleration (km/h per km)
        dist_diff = np.diff(distances_km)
        speed_diff = np.diff(speeds_smooth)
        decel_rate = np.divide(speed_diff, dist_diff, 
                              where=dist_diff>1e-6, 
                              out=np.zeros_like(speed_diff))
        
        # Braking threshold: losing >50 km/h per km traveled
        braking_threshold = -50
        
        is_braking = np.concatenate([[False], decel_rate < braking_threshold])
        
        # Group consecutive braking zones and filter short ones
        braking_zones = []
        in_zone = False
        zone_start = 0
        zone_start_speed = 0
        
        for idx, braking in enumerate(is_braking):
            if braking and not in_zone:
                zone_start = idx
                zone_start_speed = speeds[idx]
                in_zone = True
            elif not braking and in_zone:
                zone_end = idx - 1
                zone_length = distances_km[zone_end] - distances_km[zone_start]
                speed_drop = zone_start_speed - speeds[zone_end]
                
                # Only keep if: long enough OR significant speed drop
                if zone_length > 0.05 or speed_drop > 30:  # 50m or 30+ km/h drop
                    braking_zones.append((distances_km[zone_start], distances_km[zone_end]))
                in_zone = False
        
        # Plot braking zones
        for start, end in braking_zones:
            ax.axvspan(start, end, alpha=0.12, color=driver_color, 
                      zorder=1, linewidth=0)
    
    # Add legend entry for braking zones
    from matplotlib.patches import Patch
    legend_elements = ax.get_legend_handles_labels()
    
    # Styling
    ax.set_xlabel('Distance (km)', fontweight='bold', fontsize=18)
    ax.set_ylabel('Speed (km/h)', fontweight='bold', fontsize=18)
    
    # Build title
    driver_list = ', '.join([driver_names.get(d, d) for d in drivers if d in driver_data])
    ax.set_title(f'{track_name} {year} - Speed Profile Comparison\n{driver_list}', 
                fontweight='bold', fontsize=24, pad=25)
    
    # Custom legend with braking zone explanation
    ax.legend(loc='upper right', framealpha=0.95, fontsize=14, 
             fancybox=True, shadow=True, ncol=1)
    
    # Add braking zone explanation
    ax.text(0.02, 0.02, 'Shaded regions: Individual braking zones\n(Color-coded by driver)', 
           transform=ax.transAxes, fontsize=12, 
           verticalalignment='bottom',
           bbox=dict(boxstyle='round', facecolor='white', 
                    alpha=0.9, edgecolor='gray', linewidth=2))
    
    # Add light grid
    ax.grid(True, alpha=0.25, linestyle='-', linewidth=0.8, color='gray')
    
    # Set reasonable y-limits
    all_speeds = np.concatenate([driver_data[d]['speeds'] for d in driver_data])
    y_range = all_speeds.max() - all_speeds.min()
    ax.set_ylim([all_speeds.min() - 0.08 * y_range, all_speeds.max() + 0.08 * y_range])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"\n✓ Saved multi-driver speed profile to {save_path}")
    
    # Print comparison statistics
    print("\n" + "="*70)
    print(f"SPEED PROFILE COMPARISON - {track_name} {year}")
    print("="*70)
    print(f"{'Driver':<20} {'Lap Time':<12} {'Max Speed':<12} {'Avg Speed':<12}")
    print("-"*70)
    
    for driver in drivers:
        if driver not in driver_data:
            continue
        data = driver_data[driver]
        driver_name = driver_names.get(driver, driver)
        lap_time_str = f"{data['lap_time']:.3f}s" if data['lap_time'] else "N/A"
        max_speed = f"{data['speeds'].max():.1f} km/h"
        avg_speed = f"{data['speeds'].mean():.1f} km/h"
        
        marker = " 🏆" if driver == fastest_driver else ""
        print(f"{driver_name:<20} {lap_time_str:<12} {max_speed:<12} {avg_speed:<12}{marker}")
    
    print("="*70)
    
    return fig

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate pretty speed profile comparison')
    parser.add_argument('--track', type=str, default='Monaco', 
                       help='Track name')
    parser.add_argument('--year', type=int, default=2024, 
                       help='Season year')
    parser.add_argument('--drivers', type=str, nargs='+', 
                       default=['VER', 'LEC', 'NOR'],
                       help='Driver codes (e.g., VER LEC NOR)')
    parser.add_argument('--save', type=str, default=None, 
                       help='Save path (e.g., monaco_comparison.png)')
    
    args = parser.parse_args()
    
    save_path = args.save or f'{args.track.lower()}_drivers_comparison.png'
    
    fig = plot_multi_driver_speed_profile(
        track_name=args.track,
        year=args.year,
        drivers=args.drivers,
        save_path=save_path
    )
    
    if fig:
        plt.show()