import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Dict, Optional, List
import matplotlib.patches as mpatches

from simulation import LapResult
from controllers import OptimalTrajectory
from models import F1TrackModel

def plot_results(results: Dict, title: str = "Lap Simulation"):
    """Plot simulation results"""
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    
    times = results['times']
    states = results['states']
    controls = results['controls'] if len(results['controls']) > 0 else None
    
    # Position
    axes[0, 0].plot(times, states[:, 0], 'b-', linewidth=2)
    axes[0, 0].set_ylabel('Position (m)')
    axes[0, 0].set_title('Track Position')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Velocity
    axes[0, 1].plot(times, states[:, 1] * 3.6, 'g-', linewidth=2)  # Convert to km/h
    axes[0, 1].set_ylabel('Speed (km/h)')
    axes[0, 1].set_title('Speed Profile')
    axes[0, 1].grid(True, alpha=0.3)
    
    # SOC
    axes[1, 0].plot(times, states[:, 2] * 100, 'r-', linewidth=2)
    axes[1, 0].set_ylabel('SOC (%)')
    axes[1, 0].set_title('Battery State of Charge')
    axes[1, 0].set_ylim([0, 100])
    axes[1, 0].axhline(y=10, color='r', linestyle='--', alpha=0.3, label='Min SOC')
    axes[1, 0].axhline(y=90, color='r', linestyle='--', alpha=0.3, label='Max SOC')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    if controls is not None and len(controls) > 0:
        # ERS Power
        axes[1, 1].plot(times[:-1], controls[:, 0] / 1000, 'c-', linewidth=2)
        axes[1, 1].set_ylabel('ERS Power (kW)')
        axes[1, 1].set_title('ERS Deployment/Recovery')
        axes[1, 1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
        axes[1, 1].fill_between(times[:-1], 0, controls[:, 0] / 1000,
                                where=(controls[:, 0] > 0), alpha=0.3, color='green', label='Deploy')
        axes[1, 1].fill_between(times[:-1], 0, controls[:, 0] / 1000,
                                where=(controls[:, 0] < 0), alpha=0.3, color='red', label='Recover')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Throttle
        axes[2, 0].plot(times[:-1], controls[:, 1] * 100, 'm-', linewidth=2)
        axes[2, 0].set_ylabel('Throttle (%)')
        axes[2, 0].set_title('Throttle Position')
        axes[2, 0].set_ylim([0, 100])
        axes[2, 0].grid(True, alpha=0.3)
        
        # Brake
        axes[2, 1].plot(times[:-1], controls[:, 2] * 100, 'orange', linewidth=2)
        axes[2, 1].set_ylabel('Brake (%)')
        axes[2, 1].set_title('Brake Application')
        axes[2, 1].set_ylim([0, 100])
        axes[2, 1].grid(True, alpha=0.3)
    
    for ax in axes.flat:
        ax.set_xlabel('Time (s)')
        
    status = "Completed" if results.get('completed', False) else "DNF"
    plt.suptitle(f"{title} - Lap Time: {results['lap_time']:.2f}s - {status}", 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig

def plot_lap_results(result: LapResult, 
                     title: str = "Lap Simulation Results",
                     save_path: Optional[str] = None) -> plt.Figure:
    """
    Create comprehensive plot of lap simulation results.
    
    Args:
        result: LapResult from simulation
        title: Plot title
        save_path: Optional path to save figure
    
    Returns:
        matplotlib Figure
    """
    
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(4, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    times = result.times[:-1]  # Match control array length
    positions = result.positions[:-1] / 1000  # Convert to km
    
    # 1. Velocity profile
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(positions, result.velocities[:-1] * 3.6, 'b-', linewidth=1.5, label='Actual')
    if result.v_ref_history is not None:
        ax1.plot(positions, result.v_ref_history * 3.6, 'r--', alpha=0.7, label='Reference')
    ax1.set_ylabel('Velocity (km/h)')
    ax1.set_xlabel('Distance (km)')
    ax1.set_title('Velocity Profile')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. SOC profile
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(positions, result.socs[:-1] * 100, 'g-', linewidth=1.5, label='Actual')
    if result.soc_ref_history is not None:
        ax2.plot(positions, result.soc_ref_history * 100, 'r--', alpha=0.7, label='Reference')
    ax2.axhline(y=10, color='r', linestyle=':', alpha=0.5, label='Min SOC')
    ax2.axhline(y=90, color='r', linestyle=':', alpha=0.5, label='Max SOC')
    ax2.set_ylabel('State of Charge (%)')
    ax2.set_xlabel('Distance (km)')
    ax2.set_title('Battery SOC')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 100])
    
    # 3. ERS Power
    ax3 = fig.add_subplot(gs[1, 0])
    P_ers_kw = result.P_ers_history / 1000
    colors = ['green' if p >= 0 else 'red' for p in P_ers_kw]
    ax3.bar(positions, P_ers_kw, width=positions[1]-positions[0] if len(positions) > 1 else 0.01, 
            color=colors, alpha=0.7)
    ax3.axhline(y=120, color='b', linestyle='--', alpha=0.5, label='Max Deploy')
    ax3.axhline(y=-120, color='b', linestyle='--', alpha=0.5, label='Max Harvest')
    ax3.set_ylabel('ERS Power (kW)')
    ax3.set_xlabel('Distance (km)')
    ax3.set_title('ERS Deployment/Recovery')
    ax3.grid(True, alpha=0.3)
    
    # 4. Throttle & Brake
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.fill_between(positions, result.throttle_history * 100, alpha=0.5, 
                     color='green', label='Throttle')
    ax4.fill_between(positions, -result.brake_history * 100, alpha=0.5, 
                     color='red', label='Brake')
    ax4.set_ylabel('Pedal Position (%)')
    ax4.set_xlabel('Distance (km)')
    ax4.set_title('Throttle & Brake')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([-100, 100])
    
    # 5. Energy balance
    ax5 = fig.add_subplot(gs[2, 0])
    
    # Cumulative energy
    cumulative_deployed = np.cumsum(np.maximum(result.P_ers_history, 0) * 0.1) / 1e6
    cumulative_recovered = np.cumsum(np.maximum(-result.P_ers_history, 0) * 0.1) / 1e6
    cumulative_net = cumulative_deployed - cumulative_recovered
    
    ax5.plot(positions, cumulative_deployed, 'g-', label='Deployed', linewidth=1.5)
    ax5.plot(positions, cumulative_recovered, 'b-', label='Recovered', linewidth=1.5)
    ax5.plot(positions, cumulative_net, 'r--', label='Net', linewidth=1.5)
    ax5.axhline(y=4.0, color='k', linestyle=':', alpha=0.5, label='4MJ Limit')
    ax5.set_ylabel('Energy (MJ)')
    ax5.set_xlabel('Distance (km)')
    ax5.set_title('Cumulative Energy')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Solver performance (if available)
    ax6 = fig.add_subplot(gs[2, 1])
    if result.solve_times is not None and len(result.solve_times) > 0:
        ax6.plot(positions[:len(result.solve_times)], result.solve_times * 1000, 'b-', alpha=0.7)
        ax6.set_ylabel('Solve Time (ms)')
        ax6.set_xlabel('Distance (km)')
        ax6.set_title('MPC Solve Time')
        ax6.grid(True, alpha=0.3)
        
        # Mark failures
        if result.solve_success is not None:
            fail_idx = np.where(~result.solve_success)[0]
            if len(fail_idx) > 0:
                ax6.scatter(positions[fail_idx], result.solve_times[fail_idx] * 1000, 
                           c='red', s=20, marker='x', label='Failures')
                ax6.legend()
    else:
        ax6.text(0.5, 0.5, 'No solver data', ha='center', va='center', 
                transform=ax6.transAxes)
    
    # 7. Summary statistics (text)
    ax7 = fig.add_subplot(gs[3, :])
    ax7.axis('off')
    
    summary_text = (
        f"Lap Time: {result.lap_time:.3f} s    |    "
        f"Final SOC: {result.final_soc*100:.1f}%    |    "
        f"Energy Deployed: {result.energy_deployed/1e6:.2f} MJ    |    "
        f"Energy Recovered: {result.energy_recovered/1e6:.2f} MJ    |    "
        f"Status: {'✓ Complete' if result.completed else '✗ Incomplete'}"
    )
    
    ax7.text(0.5, 0.5, summary_text, ha='center', va='center', fontsize=11,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5),
             transform=ax7.transAxes)
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   Saved plot to {save_path}")
    
    return fig


def plot_strategy_comparison(results: Dict[str, LapResult],
                              title: str = "Strategy Comparison",
                              save_path: Optional[str] = None) -> plt.Figure:
    """
    Compare multiple strategy results side by side.
    
    Args:
        results: Dictionary mapping strategy name to LapResult
        title: Plot title
        save_path: Optional save path
    
    Returns:
        matplotlib Figure
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    colors = {'offline': 'blue', 'mpc': 'green', 'baseline': 'red'}
    
    # 1. Velocity comparison
    ax1 = axes[0, 0]
    for name, result in results.items():
        positions = result.positions[:-1] / 1000
        velocities = result.velocities[:-1] * 3.6
        ax1.plot(positions, velocities, color=colors.get(name, 'gray'), 
                label=name.capitalize(), alpha=0.8)
    ax1.set_ylabel('Velocity (km/h)')
    ax1.set_xlabel('Distance (km)')
    ax1.set_title('Velocity Profiles')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. SOC comparison
    ax2 = axes[0, 1]
    for name, result in results.items():
        positions = result.positions[:-1] / 1000
        socs = result.socs[:-1] * 100
        ax2.plot(positions, socs, color=colors.get(name, 'gray'),
                label=name.capitalize(), alpha=0.8)
    ax2.set_ylabel('State of Charge (%)')
    ax2.set_xlabel('Distance (km)')
    ax2.set_title('Battery SOC')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Lap time bar chart
    ax3 = axes[1, 0]
    names = list(results.keys())
    lap_times = [results[n].lap_time for n in names]
    bars = ax3.bar(names, lap_times, color=[colors.get(n, 'gray') for n in names], alpha=0.7)
    ax3.set_ylabel('Lap Time (s)')
    ax3.set_title('Lap Time Comparison')
    
    # Add values on bars
    for bar, lt in zip(bars, lap_times):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{lt:.2f}s', ha='center', va='bottom', fontsize=10)
    
    # 4. Energy comparison
    ax4 = axes[1, 1]
    x = np.arange(len(names))
    width = 0.35
    
    deployed = [results[n].energy_deployed / 1e6 for n in names]
    recovered = [results[n].energy_recovered / 1e6 for n in names]
    
    ax4.bar(x - width/2, deployed, width, label='Deployed', color='green', alpha=0.7)
    ax4.bar(x + width/2, recovered, width, label='Recovered', color='blue', alpha=0.7)
    ax4.axhline(y=4.0, color='r', linestyle='--', alpha=0.5, label='4MJ Limit')
    ax4.set_xticks(x)
    ax4.set_xticklabels(names)
    ax4.set_ylabel('Energy (MJ)')
    ax4.set_title('Energy Usage')
    ax4.legend()
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_track_with_ers(track: F1TrackModel,
                         result: LapResult,
                         title: str = "Track Analysis with ERS",
                         save_path: Optional[str] = None) -> plt.Figure:
    """
    Visualize ERS deployment/recovery on track map.
    
    Args:
        track: Track model with coordinates
        result: Lap simulation result
        title: Plot title
        save_path: Optional save path
    
    Returns:
        matplotlib Figure
    """
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Get track coordinates
    if track.track_data is not None:
        x = track.track_data.x
        y = track.track_data.y
    elif track.telemetry_raw is not None:
        x = track.telemetry_raw['X'].values
        y = track.telemetry_raw['Y'].values
    else:
        print("   Warning: No track coordinates available")
        return fig
    
    # 1. Track colored by curvature
    ax1 = axes[0]
    radii = track.track_data.radius if track.track_data is not None else np.ones(len(x)) * 1000
    
    # Normalize radius for coloring (log scale)
    radii_log = np.log10(np.clip(radii, 10, 10000))
    radii_norm = (radii_log - np.log10(10)) / (np.log10(10000) - np.log10(10))
    
    scatter1 = ax1.scatter(x, y, c=radii_norm, cmap='RdYlGn', s=5, alpha=0.8)
    ax1.set_aspect('equal')
    ax1.set_title('Track Curvature (Green=Straight, Red=Tight Corner)')
    cbar1 = plt.colorbar(scatter1, ax=ax1)
    cbar1.set_label('Curvature (log scale)')
    
    # Add start/finish
    ax1.scatter([x[0]], [y[0]], c='black', s=100, marker='o', zorder=5, label='Start')
    ax1.legend()
    
    # 2. Track colored by ERS
    ax2 = axes[1]
    
    # Interpolate ERS to track points
    n_track = len(x)
    n_result = len(result.P_ers_history)
    
    # Simple nearest-neighbor mapping
    ers_on_track = np.zeros(n_track)
    for i in range(n_track):
        result_idx = int(i / n_track * n_result)
        result_idx = min(result_idx, n_result - 1)
        ers_on_track[i] = result.P_ers_history[result_idx]
    
    # Normalize ERS for coloring
    ers_norm = ers_on_track / 120000  # Normalize to [-1, 1]
    ers_norm = np.clip(ers_norm, -1, 1)
    
    scatter2 = ax2.scatter(x, y, c=ers_norm, cmap='RdYlGn', s=5, alpha=0.8, vmin=-1, vmax=1)
    ax2.set_aspect('equal')
    ax2.set_title('ERS Strategy (Green=Deploy, Red=Harvest)')
    cbar2 = plt.colorbar(scatter2, ax=ax2)
    cbar2.set_label('ERS Power (normalized)')
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_offline_solution(trajectory: OptimalTrajectory,
                           title: str = "Offline Optimal Solution",
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    Visualize the global offline optimization solution.
    
    Args:
        trajectory: OptimalTrajectory from offline optimizer
        title: Plot title
        save_path: Optional save path
    
    Returns:
        matplotlib Figure
    """
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    
    s_km = trajectory.s / 1000
    
    # 1. Optimal velocity
    axes[0, 0].plot(s_km, trajectory.v_opt * 3.6, 'b-', linewidth=1.5)
    axes[0, 0].set_ylabel('Velocity (km/h)')
    axes[0, 0].set_title('Optimal Velocity Profile')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Optimal SOC
    axes[0, 1].plot(s_km, trajectory.soc_opt * 100, 'g-', linewidth=1.5)
    axes[0, 1].axhline(y=10, color='r', linestyle=':', alpha=0.5)
    axes[0, 1].axhline(y=90, color='r', linestyle=':', alpha=0.5)
    axes[0, 1].set_ylabel('State of Charge (%)')
    axes[0, 1].set_title('Optimal SOC Trajectory')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 100])
    
    # 3. Optimal ERS power
    P_ers_kw = trajectory.P_ers_opt / 1000
    colors = ['green' if p >= 0 else 'red' for p in P_ers_kw]
    axes[1, 0].bar(s_km[:-1], P_ers_kw, width=s_km[1]-s_km[0], color=colors, alpha=0.7)
    axes[1, 0].axhline(y=120, color='b', linestyle='--', alpha=0.5)
    axes[1, 0].axhline(y=-120, color='b', linestyle='--', alpha=0.5)
    axes[1, 0].set_ylabel('ERS Power (kW)')
    axes[1, 0].set_title('Optimal ERS Strategy')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Throttle & Brake
    axes[1, 1].fill_between(s_km[:-1], trajectory.throttle_opt * 100, alpha=0.5, 
                            color='green', label='Throttle')
    axes[1, 1].fill_between(s_km[:-1], -trajectory.brake_opt * 100, alpha=0.5,
                            color='red', label='Brake')
    axes[1, 1].set_ylabel('Pedal Position (%)')
    axes[1, 1].set_title('Optimal Throttle & Brake')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 5. Cumulative time
    axes[2, 0].plot(s_km, trajectory.t_opt, 'b-', linewidth=1.5)
    axes[2, 0].set_ylabel('Time (s)')
    axes[2, 0].set_xlabel('Distance (km)')
    axes[2, 0].set_title('Cumulative Lap Time')
    axes[2, 0].grid(True, alpha=0.3)
    
    # 6. Summary text
    axes[2, 1].axis('off')
    
    summary = (
        f"OFFLINE OPTIMIZATION RESULTS\n"
        f"{'='*40}\n\n"
        f"Lap Time:         {trajectory.lap_time:.3f} s\n"
        f"Solve Time:       {trajectory.solve_time:.2f} s\n"
        f"Status:           {trajectory.solver_status}\n\n"
        f"Energy Deployed:  {trajectory.energy_deployed/1e6:.2f} MJ\n"
        f"Energy Recovered: {trajectory.energy_recovered/1e6:.2f} MJ\n"
        f"Net Energy:       {(trajectory.energy_deployed-trajectory.energy_recovered)/1e6:.2f} MJ\n\n"
        f"Initial SOC:      {trajectory.soc_opt[0]*100:.1f}%\n"
        f"Final SOC:        {trajectory.soc_opt[-1]*100:.1f}%\n\n"
        f"Velocity Range:   {trajectory.v_opt.min()*3.6:.0f} - {trajectory.v_opt.max()*3.6:.0f} km/h"
    )
    
    axes[2, 1].text(0.1, 0.5, summary, fontsize=11, fontfamily='monospace',
                    va='center', ha='left',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig