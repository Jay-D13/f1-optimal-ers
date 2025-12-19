import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import VehicleConfig, get_vehicle_config, get_ers_config
from models import F1TrackModel, VehicleDynamicsModel
from solvers import ForwardBackwardSolver, SpatialNLPSolver

# ============================================================================
# STYLING
# ============================================================================

COLORS = {
    'monaco': '#E10600',
    'montreal': '#005AFF',
    'monza': '#009246',
    'spa': '#FFDD00',
    'deploy': '#00D856',
    'harvest': '#FF1744',
    'optimal': '#2962FF',
    'real_driver': '#FF6D00',
    '2025': "#09A2B7",
    '2026': '#9C27B0',
}

def setup_poster_style():
    """Configure matplotlib for poster-quality plots"""
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica'],
        'font.size': 14,
        'axes.titlesize': 18,
        'axes.labelsize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'lines.linewidth': 3.0,
        'axes.linewidth': 1.5,
        'grid.alpha': 0.3,
        'axes.grid': True,
    })

# ============================================================================
# SOC STRATEGY COMPARISON (OVERLAPPED)
# ============================================================================

def plot_soc_strategies_overlapped(strategies_data, track_name, regulation_year='2025', save_path=None):
    """
    All SOC strategies overlapped on single figure for direct comparison.
    Shows velocity, SOC, and ERS power profiles together.
    """
    fig, axes = plt.subplots(3, 1, figsize=(16, 14), sharex=True)
    
    # Color map for strategies
    strategy_colors = {
        'SOC_50_30': '#FF9800',
        'SOC_60_40': '#9C27B0',
        'SOC_50_80': '#2196F3',
        'SOC_50_50': '#4CAF50',
        'SOC_75_25': '#E91E63',
    }
    
    max_power = 120 if regulation_year == '2025' else 350
    
    # Plot all strategies
    for strat_name, data in strategies_data.items():
        s_km = data['s'] / 1000
        v_kph = data['v_opt'] * 3.6
        soc_pct = data['soc_opt'] * 100
        P_ers_kw = data['P_ers_opt'] / 1000
        
        color = strategy_colors.get(strat_name, 'gray')
        label = strat_name.replace('SOC_', '').replace('_', '% → ') + '%'
        
        # Velocity
        axes[0].plot(s_km, v_kph, color=color, linewidth=3, alpha=0.85, label=label)
        
        # SOC
        axes[1].plot(s_km, soc_pct, color=color, linewidth=3, alpha=0.85, label=label)
        
        # ERS Power (only positive for clarity)
        axes[2].plot(s_km[:-1], P_ers_kw, color=color, linewidth=2.5, alpha=0.75, label=label)
    
    # Velocity styling
    axes[0].set_ylabel('Velocity (km/h)', fontweight='bold', fontsize=16)
    axes[0].set_title(f'{track_name} - SOC Strategy Comparison ({regulation_year} Regulations)', 
                     fontweight='bold', fontsize=20)
    axes[0].legend(loc='best', framealpha=0.95, fontsize=13, ncol=2)
    axes[0].grid(True, alpha=0.3)
    
    # SOC styling
    axes[1].axhline(y=20, color='red', linestyle='--', alpha=0.5, linewidth=2, label='SOC Limits')
    axes[1].axhline(y=90, color='red', linestyle='--', alpha=0.5, linewidth=2)
    axes[1].fill_between([0, s_km[-1]], 20, 90, alpha=0.08, color='green')
    axes[1].set_ylabel('Battery SOC (%)', fontweight='bold', fontsize=16)
    axes[1].set_ylim([0, 100])
    axes[1].legend(
        loc='upper center',          # Which part of the legend to anchor
        bbox_to_anchor=(0.5, 0.9),  # (x, y) -> x=0.5 is center, y=0.85 is lower than the top
        framealpha=0.95, 
        fontsize=13, 
        ncol=2
    )
    axes[1].grid(True, alpha=0.3)
    
    # ERS Power styling
    axes[2].axhline(y=max_power, color='blue', linestyle='--', alpha=0.5, linewidth=2, label=f'±{max_power}kW Limit')
    axes[2].axhline(y=-max_power, color='blue', linestyle='--', alpha=0.5, linewidth=2)
    axes[2].axhline(y=0, color='black', linewidth=2, alpha=0.5)
    axes[2].set_ylabel('ERS Power (kW)', fontweight='bold', fontsize=16)
    axes[2].set_xlabel('Distance (km)', fontweight='bold', fontsize=16)
    axes[2].legend(loc='best', framealpha=0.95, fontsize=13, ncol=2)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved overlapped SOC strategies to {save_path}")
    
    return fig

# ============================================================================
# INDIVIDUAL TRACK LAYOUT PLOTS (ERS COLORED)
# ============================================================================

def plot_single_track_ers_layout(track_name, track_model, trajectory, regulation_year='2025', save_path=None):
    """
    Single track layout colored by ERS power.
    Perfect for individual poster placement.
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Get coordinates
    if track_model.track_data is not None:
        x = track_model.track_data.x
        y = track_model.track_data.y
        distances = track_model.track_data.s
    elif track_model.telemetry_data is not None:
        x = track_model.telemetry_data['X'].values
        y = track_model.telemetry_data['Y'].values
        distances = track_model.telemetry_data['Distance'].values
    else:
        print(f"⚠ No coordinate data for {track_name}")
        return None
    
    # Get ERS power
    s_opt = trajectory.s
    P_ers_kw = trajectory.P_ers_opt / 1000
    
    # Interpolate to track points
    P_ers_at_points = np.interp(distances, s_opt[:-1], P_ers_kw)
    
    # Normalize
    max_power = 120 if regulation_year == '2025' else 350
    P_ers_norm = np.clip(P_ers_at_points / max_power, -1, 1)
    
    # Create scatter plot
    scatter = ax.scatter(x, y, c=P_ers_norm, 
                        cmap='RdYlGn', s=20, alpha=0.9,
                        vmin=-1, vmax=1, edgecolors='none')
    
    # Start/finish marker
    ax.scatter([x[0]], [y[0]], c='black', s=500, marker='*', 
              edgecolors='gold', linewidths=4, zorder=10,
              label='Start/Finish')
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('ERS Power', fontweight='bold', fontsize=16)
    cbar.ax.tick_params(labelsize=14)
    
    # Set proper tick positions and labels
    # The colorbar goes from -1 to +1 (normalized)
    cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
    cbar.set_ticklabels([
        f'-{max_power}kW\nHarvest',
        f'-{max_power//2}kW',
        'Neutral',
        f'+{max_power//2}kW',
        f'+{max_power}kW\nDeploy'
    ])
    
    # Styling
    ax.set_aspect('equal')
    ax.set_xlabel('X Position (m)', fontweight='bold', fontsize=16)
    ax.set_ylabel('Y Position (m)', fontweight='bold', fontsize=16)
    ax.set_title(f'{track_name} - ERS Strategy ({regulation_year})\n'
                f'Lap Time: {trajectory.lap_time:.2f}s | Energy: {trajectory.energy_deployed/1e6:.2f} MJ', 
                 fontweight='bold', fontsize=20, 
                 color=COLORS.get(track_name.lower(), 'black'),
                 pad=20)  # Add padding between title and plot
    ax.legend(loc='upper left', fontsize=14, framealpha=0.95)
    ax.grid(True, alpha=0.2)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved {track_name} ERS layout to {save_path}")
    
    return fig

# ============================================================================
# BAR CHART COMPARISONS
# ============================================================================

def plot_regulation_velocity_comparison(track_name, data_2025, data_2026, save_path=None):
    """
    Compare velocity profiles between 2025 and 2026 regulations.
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    s_km_25 = data_2025['s'] / 1000
    s_km_26 = data_2026['s'] / 1000
    
    ax.plot(s_km_25, data_2025['v_opt'] * 3.6, 
           color=COLORS['2025'], linewidth=3.5, label='2025 (120kW)', alpha=0.9)
    ax.plot(s_km_26, data_2026['v_opt'] * 3.6, 
           color=COLORS['2026'], linewidth=3.5, label='2026 (350kW)', alpha=0.9)
    
    # Fill improvement area
    v_25_interp = np.interp(s_km_26, s_km_25, data_2025['v_opt'] * 3.6)
    ax.fill_between(s_km_26, v_25_interp, data_2026['v_opt'] * 3.6,
                    where=(data_2026['v_opt'] * 3.6 > v_25_interp),
                    color=COLORS['2026'], alpha=0.2, label='2026 Advantage')
    
    ax.set_ylabel('Velocity (km/h)', fontweight='bold', fontsize=16)
    ax.set_xlabel('Distance (km)', fontweight='bold', fontsize=16)
    ax.set_title(f'{track_name} - Velocity Comparison\n'
                f'2025: {data_2025["lap_time"]:.3f}s | 2026: {data_2026["lap_time"]:.3f}s '
                f'(Δ {data_2025["lap_time"] - data_2026["lap_time"]:.3f}s)', 
                fontweight='bold', fontsize=20)
    ax.legend(loc='best', framealpha=0.95, fontsize=16)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved regulation velocity comparison to {save_path}")
    
    return fig

def plot_regulation_ers_comparison(track_name, data_2025, data_2026, save_path=None):
    """
    Compare ERS power between 2025 and 2026 regulations (stacked).
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    s_km_25 = data_2025['s'] / 1000
    s_km_26 = data_2026['s'] / 1000
    P_ers_25 = data_2025['P_ers_opt'] / 1000
    P_ers_26 = data_2026['P_ers_opt'] / 1000
    
    # 2025
    ax1.fill_between(s_km_25[:-1], 0, P_ers_25, 
                    where=(P_ers_25 > 0), color=COLORS['2025'], 
                    alpha=0.7, label='Deploy', step='post')
    ax1.fill_between(s_km_25[:-1], 0, P_ers_25, 
                    where=(P_ers_25 < 0), color=COLORS['2025'], 
                    alpha=0.4, label='Harvest', step='post', hatch='///')
    ax1.axhline(y=120, color='blue', linestyle='--', linewidth=2, alpha=0.7)
    ax1.axhline(y=-120, color='blue', linestyle='--', linewidth=2, alpha=0.7)
    ax1.axhline(y=0, color='black', linewidth=2)
    ax1.set_ylabel('ERS Power (kW)', fontweight='bold', fontsize=16)
    ax1.set_title(f'{track_name} - 2025 Regulations (120kW limit)', 
                 fontweight='bold', fontsize=18)
    ax1.legend(loc='best', framealpha=0.95, fontsize=14)
    ax1.set_ylim([-150, 150])
    ax1.grid(True, alpha=0.3)
    
    # 2026
    ax2.fill_between(s_km_26[:-1], 0, P_ers_26, 
                    where=(P_ers_26 > 0), color=COLORS['2026'], 
                    alpha=0.7, label='Deploy', step='post')
    ax2.fill_between(s_km_26[:-1], 0, P_ers_26, 
                    where=(P_ers_26 < 0), color=COLORS['2026'], 
                    alpha=0.4, label='Harvest', step='post', hatch='\\\\\\')
    ax2.axhline(y=350, color='blue', linestyle='--', linewidth=2, alpha=0.7)
    ax2.axhline(y=-350, color='blue', linestyle='--', linewidth=2, alpha=0.7)
    ax2.axhline(y=0, color='black', linewidth=2)
    ax2.set_ylabel('ERS Power (kW)', fontweight='bold', fontsize=16)
    ax2.set_xlabel('Distance (km)', fontweight='bold', fontsize=16)
    ax2.set_title('2026 Regulations (350kW limit)', fontweight='bold', fontsize=18)
    ax2.legend(loc='best', framealpha=0.95, fontsize=14)
    ax2.set_ylim([-400, 400])
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved regulation ERS comparison to {save_path}")
    
    return fig

def plot_regulation_soc_comparison(track_name, data_2025, data_2026, save_path=None):
    """
    Compare battery management (SOC) between 2025 and 2026 regulations.
    Overlapped on single plot to show differences.
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    s_km_25 = data_2025['s'] / 1000
    s_km_26 = data_2026['s'] / 1000
    soc_25 = data_2025['soc_opt'] * 100
    soc_26 = data_2026['soc_opt'] * 100
    
    # Plot both
    ax.plot(s_km_25, soc_25, color=COLORS['2025'], linewidth=3.5, 
           label='2025 (120kW)', alpha=0.9)
    ax.plot(s_km_26, soc_26, color=COLORS['2026'], linewidth=3.5, 
           label='2026 (350kW)', alpha=0.9)
    
    # SOC limits
    ax.axhline(y=20, color='red', linestyle='--', alpha=0.5, linewidth=2)
    ax.axhline(y=90, color='red', linestyle='--', alpha=0.5, linewidth=2)
    ax.fill_between([0, max(s_km_25[-1], s_km_26[-1])], 20, 90, 
                    alpha=0.08, color='green', label='Operating Range')
    
    ax.set_ylabel('Battery SOC (%)', fontweight='bold', fontsize=16)
    ax.set_xlabel('Distance (km)', fontweight='bold', fontsize=16)
    ax.set_title(f'{track_name} - Battery Management Strategy Comparison\n'
                f'2025: {data_2025["lap_time"]:.3f}s | 2026: {data_2026["lap_time"]:.3f}s', 
                fontweight='bold', fontsize=20)
    ax.legend(loc='best', framealpha=0.95, fontsize=16)
    ax.set_ylim([0, 100])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved regulation SOC comparison to {save_path}")
    
    return fig

# ============================================================================
# REAL VS OPTIMAL INDIVIDUAL PLOTS
# ============================================================================

def plot_real_vs_optimal_velocity(track_name, real_data, optimal_data, regulation_year='2025', save_path=None):
    """
    Compare real driver velocity with optimal.
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Real data
    s_real = real_data['Distance'].values / 1000
    v_real = real_data['Speed'].values
    
    # Optimal data
    s_opt = optimal_data['s'] / 1000
    v_opt = optimal_data['v_opt'] * 3.6
    
    ax.plot(s_real, v_real, color=COLORS['real_driver'], 
           linewidth=3.5, label='Real Driver', alpha=0.9)
    ax.plot(s_opt, v_opt, color=COLORS['optimal'], 
           linewidth=3.5, label='Optimal Strategy', alpha=0.9, linestyle='--')
    
    # Highlight differences
    v_opt_interp = np.interp(s_real, s_opt, v_opt)
    ax.fill_between(s_real, v_real, v_opt_interp,
                    where=(v_opt_interp > v_real),
                    color='green', alpha=0.2, label='Potential Gain')
    
    ax.set_ylabel('Velocity (km/h)', fontweight='bold', fontsize=16)
    ax.set_xlabel('Distance (km)', fontweight='bold', fontsize=16)
    
    # Calculate lap time improvement
    real_lap_time = (real_data['Time'].values[-1] - real_data['Time'].values[0]).total_seconds()
    opt_lap_time = optimal_data['lap_time']
    
    ax.set_title(f'{track_name} - Real vs Optimal ({regulation_year})\n'
                f'Real: {real_lap_time:.3f}s | Optimal: {opt_lap_time:.3f}s '
                f'(Δ {real_lap_time - opt_lap_time:.3f}s)', 
                fontweight='bold', fontsize=20)
    ax.legend(loc='best', framealpha=0.95, fontsize=16)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved real vs optimal velocity to {save_path}")
    
    return fig

# ============================================================================
# BAR CHART COMPARISONS
# ============================================================================

def plot_soc_strategy_laptime_comparison(strategies_data, track_name, save_path=None):
    """
    Bar chart comparing lap times for different SOC strategies.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Color map for strategies
    strategy_colors = {
        'SOC_50_30': '#FF9800',
        'SOC_60_40': '#9C27B0',
        'SOC_50_80': '#2196F3',
        'SOC_50_50': '#4CAF50',
        'SOC_75_25': '#E91E63',
    }
    
    strategies = list(strategies_data.keys())
    lap_times = [strategies_data[s]['lap_time'] for s in strategies]
    colors = [strategy_colors.get(s, 'gray') for s in strategies]
    labels = [s.replace('SOC_', '').replace('_', '% → ') + '%' for s in strategies]
    
    bars = ax.barh(labels, lap_times, color=colors, alpha=0.85, edgecolor='black', linewidth=2)
    
    # Add time values on bars
    best_time = min(lap_times)
    for i, (bar, time) in enumerate(zip(bars, lap_times)):
        diff = time - best_time
        label_text = f"{time:.3f}s"
        if diff > 0.001:
            label_text += f" (+{diff:.3f}s)"
        ax.text(time + 0.03, bar.get_y() + bar.get_height()/2, 
               label_text, va='center', fontweight='bold', fontsize=14)
    
    ax.set_xlabel('Lap Time (seconds)', fontweight='bold', fontsize=16)
    ax.set_title(f'{track_name} - SOC Strategy Performance Comparison', 
                fontweight='bold', fontsize=20)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved SOC strategy comparison bar chart to {save_path}")
    
    return fig

# ============================================================================
# MAIN GENERATION FUNCTION
# ============================================================================

def generate_individual_plots():
    setup_poster_style()
    
    print("="*80)
    print("GENERATING INDIVIDUAL POSTER PLOTS")
    print("="*80)
    
    output_dir = Path("figures/poster_plots")
    output_dir.mkdir(exist_ok=True)
    
    print("\n[1/4] Generating individual track ERS layouts...")
    
    tracks = ['Monaco', 'Montreal', 'Monza', 'Spa']
    tracks_data = {}
    
    for track_name in tracks:
        print(f"\n  Processing {track_name}...")
        
        # Setup
        ers_config = get_ers_config('2025')
        vehicle_config = getattr(VehicleConfig, f'for_{track_name.lower()}', VehicleConfig)()
        vehicle_config = get_vehicle_config('2025', base=vehicle_config)
        
        # Load track
        track = F1TrackModel(2024, track_name, 'Q', ds=5.0)
        track.load_from_fastf1()

        # Optimize
        vehicle_model = VehicleDynamicsModel(vehicle_config, ers_config)
        fb_solver = ForwardBackwardSolver(vehicle_model, track, use_ers_power=True)
        velocity_profile = fb_solver.solve(flying_lap=True)
        
        nlp_solver = SpatialNLPSolver(vehicle_model, track, ers_config, ds=5.0)
        trajectory = nlp_solver.solve(
            v_limit_profile=velocity_profile.v,
            initial_soc=0.5,
            final_soc_min=0.3,
            is_flying_lap=True
        )
        
        # Save data
        class TrajectoryData:
            def __init__(self, traj):
                self.s = traj.s
                self.P_ers_opt = traj.P_ers_opt
                self.soc_opt = traj.soc_opt
                self.v_opt = traj.v_opt
                self.lap_time = traj.lap_time
                self.energy_deployed = traj.energy_deployed
        
        traj_obj = TrajectoryData(trajectory)
        
        tracks_data[track_name] = {
            'track_model': track,
            'trajectory': traj_obj,
            's': trajectory.s,
            'P_ers_opt': trajectory.P_ers_opt,
            'v_opt': trajectory.v_opt,
            'soc_opt': trajectory.soc_opt,
            'lap_time': trajectory.lap_time,
            'energy_deployed': trajectory.energy_deployed,
        }
        
        # Generate individual track plot
        fig = plot_single_track_ers_layout(
            track_name,
            track,
            traj_obj,
            regulation_year='2025',
            save_path=output_dir / f"track_{track_name.lower()}_ers_layout.png"
        )
        if fig:
            plt.close(fig)
    
    print("\n[2/4] Generating SOC strategy comparison plot...")
    
    track_name = 'Monaco'
    soc_strategies = [
        (0.5, 0.3),
        (0.6, 0.4),
        (0.5, 0.8),
        (0.5, 0.5),
        (0.75, 0.25),
    ]
    
    strategies_data = {}
    
    for initial_soc, final_soc in soc_strategies:
        print(f"  Running strategy: SOC {initial_soc*100:.0f}% → {final_soc*100:.0f}%")
        
        strat_name = f"SOC_{int(initial_soc*100)}_{int(final_soc*100)}"
        
        ers_config = get_ers_config('2025')
        vehicle_config = VehicleConfig.for_monaco()
        vehicle_config = get_vehicle_config('2025', base=vehicle_config)
        
        track = F1TrackModel(2024, track_name, 'Q', ds=5.0)
        track.load_from_fastf1()
        
        vehicle_model = VehicleDynamicsModel(vehicle_config, ers_config)
        fb_solver = ForwardBackwardSolver(vehicle_model, track, use_ers_power=True)
        velocity_profile = fb_solver.solve(flying_lap=True)
        
        nlp_solver = SpatialNLPSolver(vehicle_model, track, ers_config, ds=5.0)
        trajectory = nlp_solver.solve(
            v_limit_profile=velocity_profile.v,
            initial_soc=initial_soc,
            final_soc_min=final_soc,
            is_flying_lap=True
        )
        
        strategies_data[strat_name] = {
            's': trajectory.s,
            'v_opt': trajectory.v_opt,
            'soc_opt': trajectory.soc_opt,
            'P_ers_opt': trajectory.P_ers_opt,
            'lap_time': trajectory.lap_time,
        }
    
    # Generate overlapped strategies plot
    fig_overlap = plot_soc_strategies_overlapped(
        strategies_data,
        track_name,
        regulation_year='2025',
        save_path=output_dir / "soc_strategies_overlapped.png"
    )
    if fig_overlap:
        plt.close(fig_overlap)
    
    # SOC comparison bar chart
    fig_bar = plot_soc_strategy_laptime_comparison(
        strategies_data,
        track_name,
        save_path=output_dir / "soc_strategies_laptime_comparison.png"
    )
    if fig_bar:
        plt.close(fig_bar)
    
    print("\n[3/4] Generating regulation comparison plots...")
    
    track_name = 'Monaco'
    
    # 2025 data
    data_2025 = strategies_data['SOC_50_30']
    
    # 2026 data
    print("  Running 2026 regulations...")
    ers_config_2026 = get_ers_config('2026')
    vehicle_config_2026 = VehicleConfig.for_monaco()
    vehicle_config_2026 = get_vehicle_config('2026', base=vehicle_config_2026)
    
    track = F1TrackModel(2024, track_name, 'Q', ds=5.0)
    track.load_from_fastf1()
    
    vehicle_model_2026 = VehicleDynamicsModel(vehicle_config_2026, ers_config_2026)
    fb_solver = ForwardBackwardSolver(vehicle_model_2026, track, use_ers_power=True)
    velocity_profile = fb_solver.solve(flying_lap=True)
    
    nlp_solver_2026 = SpatialNLPSolver(vehicle_model_2026, track, ers_config_2026, ds=5.0)
    trajectory_2026 = nlp_solver_2026.solve(
        v_limit_profile=velocity_profile.v,
        initial_soc=0.5,
        final_soc_min=0.3,
        is_flying_lap=True
    )
    
    data_2026 = {
        's': trajectory_2026.s,
        'v_opt': trajectory_2026.v_opt,
        'soc_opt': trajectory_2026.soc_opt,
        'P_ers_opt': trajectory_2026.P_ers_opt,
        'lap_time': trajectory_2026.lap_time,
        'energy_deployed': trajectory_2026.energy_deployed,
    }
    
    data_2025['energy_deployed'] = strategies_data['SOC_50_30'].get('energy_deployed', 2.0e6)
    
    fig_vel = plot_regulation_velocity_comparison(
        track_name,
        data_2025,
        data_2026,
        save_path=output_dir / "regulation_velocity_comparison.png"
    )
    if fig_vel:
        plt.close(fig_vel)
    
    fig_ers = plot_regulation_ers_comparison(
        track_name,
        data_2025,
        data_2026,
        save_path=output_dir / "regulation_ers_comparison.png"
    )
    if fig_ers:
        plt.close(fig_ers)
    
    fig_soc = plot_regulation_soc_comparison(
        track_name,
        data_2025,
        data_2026,
        save_path=output_dir / "regulation_soc_comparison.png"
    )
    if fig_soc:
        plt.close(fig_soc)
    
    print("\n[4/4] Generating real vs optimal plot...")
    
    track_name = 'Monaco'
    track = F1TrackModel(2024, track_name, 'Q', ds=5.0)
    
    try:
        track.load_from_fastf1(driver='LEC')
        real_data = track.telemetry_data
        
        fig_real = plot_real_vs_optimal_velocity(
            track_name,
            real_data,
            data_2025,
            regulation_year='2025',
            save_path=output_dir / "real_vs_optimal_velocity.png"
        )
        if fig_real:
            plt.close(fig_real)
    except Exception as e:
        print(f"  ⚠ Could not generate real vs optimal plot: {e}")
    
    print("\n" + "="*80)
    print("✓ ALL INDIVIDUAL PLOTS GENERATED")
    print(f"✓ Saved to: {output_dir.absolute()}")
    print("="*80)

if __name__ == "__main__":
    generate_individual_plots()