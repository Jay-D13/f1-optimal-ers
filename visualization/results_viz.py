import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

from config import ERSConfig
from solvers import OptimalTrajectory
from models import F1TrackModel
from models.car import deploy_power_limit

# Above this the per-lap deployment limit is "no limit" (the 2026 rules set it to 100 MJ)
_NO_ENERGY_LIMIT = 50e6


def soc_band(ers: Optional[ERSConfig] = None) -> tuple:
    """
    The SOC band the solver enforces, in %: min_soc to max_soc, raised for a 2026 qualifying lap to the floor
    of the stored-energy window below its full start (C5.2.9).
    """
    ers = ers or ERSConfig()
    low, high = ers.min_soc, ers.max_soc
    if ers.qualifying and ers.soc_window is not None:
        low = max(low, high - ers.soc_window / ers.battery_capacity)
    return 100.0 * low, 100.0 * high


def power_limits(v: np.ndarray, ers: Optional[ERSConfig] = None) -> tuple:
    """Deploy limit at each speed (the regulation's deploy curve) and the harvest limit, in kW (DC from 2026)."""
    ers = ers or ERSConfig()
    v = np.asarray(v, dtype=float)
    deploy = np.broadcast_to(np.asarray(deploy_power_limit(v, ers), dtype=float).ravel(), v.shape)
    return deploy / 1000.0, -ers.max_recovery_power / 1000.0


def _draw_power_limits(ax, s_km: np.ndarray, v: np.ndarray, ers: Optional[ERSConfig], label=None):
    deploy_kw, harvest_kw = power_limits(v, ers)
    ax.plot(s_km, deploy_kw, color='b', linestyle='--', alpha=0.5, label=label)
    ax.axhline(y=harvest_kw, color='b', linestyle='--', alpha=0.5)


def _draw_soc_band(ax, s_km: np.ndarray, ers: Optional[ERSConfig], linestyle='--', label=None, fill=True):
    low, high = soc_band(ers)
    ax.axhline(y=low, color='r', linestyle=linestyle, alpha=0.5, label=label)
    ax.axhline(y=high, color='r', linestyle=linestyle, alpha=0.5)
    if fill:
        ax.fill_between(s_km, low, high, alpha=0.1, color='green')


def plot_offline_solution(trajectory: OptimalTrajectory,
                           title: str = "Offline Optimal Solution",
                           save_path: Optional[str] = None,
                           ers_config: Optional[ERSConfig] = None) -> plt.Figure:
    """Overview of a solution. The SOC and power limits come from ers_config (default: the 2025 rules)."""

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    
    s_km = trajectory.s / 1000
    
    # Optimal velocity
    axes[0, 0].plot(s_km, trajectory.v_opt * 3.6, 'b-', linewidth=1.5)
    axes[0, 0].set_ylabel('Velocity (km/h)')
    axes[0, 0].set_title('Optimal Velocity Profile')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Optimal SOC
    axes[0, 1].plot(s_km, trajectory.soc_opt * 100, 'g-', linewidth=1.5)
    _draw_soc_band(axes[0, 1], s_km, ers_config, linestyle=':', fill=False)
    axes[0, 1].set_ylabel('State of Charge (%)')
    axes[0, 1].set_title('Optimal SOC Trajectory')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 100])
    
    # Optimal ERS power
    P_ers_kw = trajectory.P_ers_opt / 1000
    colors = ['green' if p >= 0 else 'red' for p in P_ers_kw]
    axes[1, 0].bar(s_km[:-1], P_ers_kw, width=s_km[1]-s_km[0], color=colors, alpha=0.7)
    _draw_power_limits(axes[1, 0], s_km[:-1], trajectory.v_opt[:-1], ers_config)
    axes[1, 0].set_ylabel('ERS Power (kW)')
    axes[1, 0].set_title('Optimal ERS Strategy')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Throttle & Brake
    axes[1, 1].fill_between(s_km[:-1], trajectory.throttle_opt * 100, alpha=0.5, 
                            color='green', label='Throttle')
    axes[1, 1].fill_between(s_km[:-1], -trajectory.brake_opt * 100, alpha=0.5,
                            color='red', label='Brake')
    axes[1, 1].set_ylabel('Pedal Position (%)')
    axes[1, 1].set_title('Optimal Throttle & Brake')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Cumulative time
    axes[2, 0].plot(s_km, trajectory.t_opt, 'b-', linewidth=1.5)
    axes[2, 0].set_ylabel('Time (s)')
    axes[2, 0].set_xlabel('Distance (km)')
    axes[2, 0].set_title('Cumulative Lap Time')
    axes[2, 0].grid(True, alpha=0.3)
        
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig

def plot_simple_results(trajectory: OptimalTrajectory, 
                       velocity_profile,
                       track,
                       track_name: str,
                       ers_config: Optional[ERSConfig] = None) -> plt.Figure:
    """Speed, SOC, ERS power and curvature. The SOC limits come from ers_config (default: the 2025 rules)."""
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    
    s = trajectory.s / 1000  # Convert to km
    
    # Velocity
    axes[0].plot(s, trajectory.v_opt * 3.6, 'b-', label='Optimal with ERS', linewidth=1.5)
    axes[0].plot(s[:-1], velocity_profile.v[:-1] * 3.6, 'r--', 
                 label='Grip limit (no ERS)', alpha=0.7, linewidth=1.5)
    axes[0].set_ylabel('Velocity (km/h)', fontsize=11)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(f'{track_name} - ERS Optimization Results', fontsize=12, fontweight='bold')
    
    # SOC
    axes[1].plot(s, trajectory.soc_opt * 100, 'g-', linewidth=1.5)
    _draw_soc_band(axes[1], s, ers_config, label='SOC limits')
    axes[1].set_ylabel('SOC (%)', fontsize=11)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 100])
    
    # ERS Power
    axes[2].fill_between(s[:-1], trajectory.P_ers_opt/1000, 0,
                         where=trajectory.P_ers_opt > 0,
                         color='green', alpha=0.6, label='Deploy')
    axes[2].fill_between(s[:-1], trajectory.P_ers_opt/1000, 0,
                         where=trajectory.P_ers_opt < 0,
                         color='red', alpha=0.6, label='Harvest')
    axes[2].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    axes[2].set_ylabel('ERS Power (kW)', fontsize=11)
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3)
    
    # Track curvature
    segment_distances = np.array([seg.distance for seg in track.segments])
    segment_radii = np.array([seg.radius for seg in track.segments])
    curvature = 1.0 / np.maximum(segment_radii, 50)
    
    axes[3].fill_between(segment_distances/1000, curvature * 1000, 0, 
                        alpha=0.3, color='purple')
    axes[3].set_ylabel('Curvature (1/km)', fontsize=11)
    axes[3].set_xlabel('Distance (km)', fontsize=11)
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig

def create_comparison_plot(track: F1TrackModel,
                          velocity_no_ers: np.ndarray,
                          velocity_with_ers: np.ndarray,
                          optimal_trajectory: OptimalTrajectory,
                          track_name: str,
                          ers_config: Optional[ERSConfig] = None) -> plt.Figure:
    """
    Create a comprehensive comparison plot showing:
    - Theoretical max speed without ERS
    - Theoretical max speed with ERS
    - Optimal trajectory details
    The SOC, power and energy limits come from ers_config (default: the 2025 rules).
    """
    ers = ers_config or ERSConfig()
    
    fig = plt.figure(figsize=(16, 12))
    
    s_km = optimal_trajectory.s / 1000
    
    # Velocity comparison
    ax1 = plt.subplot(4, 2, 1)
    ax1.plot(s_km, velocity_no_ers * 3.6, 'r--', 
             linewidth=2, alpha=0.7, label='Max (No ERS)')
    ax1.plot(s_km, velocity_with_ers * 3.6, 'orange', 
             linewidth=2, alpha=0.7, label='Max (Unlim. ERS)')
    ax1.plot(s_km, optimal_trajectory.v_opt * 3.6, 'b-', 
             linewidth=1.5, label='Optimal Strategy')
    ax1.set_ylabel('Velocity (km/h)')
    ax1.set_xlabel('Distance (km)')
    ax1.set_title('Velocity Profile Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Velocity difference (with ERS vs without)
    ax2 = plt.subplot(4, 2, 2)
    v_diff = (velocity_with_ers - velocity_no_ers) * 3.6
    ax2.fill_between(s_km, 0, v_diff, where=(v_diff > 0), 
                     color='green', alpha=0.5, label='ERS advantage')
    ax2.fill_between(s_km, 0, v_diff, where=(v_diff < 0), 
                     color='red', alpha=0.5, label='ERS disadvantage')
    ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax2.set_ylabel('Speed Difference (km/h)')
    ax2.set_xlabel('Distance (km)')
    ax2.set_title('ERS Impact on Maximum Speed')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # State of Charge
    ax3 = plt.subplot(4, 2, 3)
    ax3.plot(s_km, optimal_trajectory.soc_opt * 100, 'g-', linewidth=2)
    _draw_soc_band(ax3, s_km, ers, label='SOC limits')
    ax3.set_ylabel('SOC (%)')
    ax3.set_xlabel('Distance (km)')
    ax3.set_title('Battery State of Charge')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 100])
    
    # ERS Power
    ax4 = plt.subplot(4, 2, 4)
    P_ers_kw = optimal_trajectory.P_ers_opt / 1000
    ax4.fill_between(s_km[:-1], 0, P_ers_kw, where=(P_ers_kw > 0),
                     color='green', alpha=0.6, label='Deploy')
    ax4.fill_between(s_km[:-1], 0, P_ers_kw, where=(P_ers_kw < 0),
                     color='red', alpha=0.6, label='Harvest')
    _draw_power_limits(ax4, s_km[:-1], optimal_trajectory.v_opt[:-1], ers, label='Limits')
    ax4.set_ylabel('ERS Power (kW)')
    ax4.set_xlabel('Distance (km)')
    ax4.set_title('ERS Deployment Strategy')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Throttle & Brake
    ax5 = plt.subplot(4, 2, 5)
    ax5.fill_between(s_km[:-1], 0, optimal_trajectory.throttle_opt * 100, 
                     alpha=0.6, color='green', label='Throttle')
    ax5.fill_between(s_km[:-1], 0, -optimal_trajectory.brake_opt * 100, 
                     alpha=0.6, color='red', label='Brake')
    ax5.set_ylabel('Pedal Position (%)')
    ax5.set_xlabel('Distance (km)')
    ax5.set_title('Throttle & Brake Application')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim([-100, 100])
    
    # Cumulative energy
    ax6 = plt.subplot(4, 2, 6)
    dt = np.diff(optimal_trajectory.t_opt)  # Time spent on each interval
    cumulative_deployed = np.cumsum(np.maximum(optimal_trajectory.P_ers_opt, 0) * dt) / 1e6
    cumulative_recovered = np.cumsum(np.maximum(-optimal_trajectory.P_ers_opt, 0) * dt) / 1e6
    cumulative_net = cumulative_deployed - cumulative_recovered
    
    ax6.plot(s_km[:-1], cumulative_deployed, 'g-', label='Deployed', linewidth=2)
    ax6.plot(s_km[:-1], cumulative_recovered, 'b-', label='Recovered', linewidth=2)
    ax6.plot(s_km[:-1], cumulative_net, 'r--', label='Net Used', linewidth=2)
    if ers.deployment_limit_per_lap < _NO_ENERGY_LIMIT:
        ax6.axhline(y=ers.deployment_limit_per_lap / 1e6, color='g', linestyle=':', alpha=0.5,
                    label=f'Deploy limit {ers.deployment_limit_per_lap / 1e6:.1f} MJ')
    ax6.axhline(y=ers.recovery_limit_per_lap / 1e6, color='b', linestyle=':', alpha=0.5,
                label=f'Recharge limit {ers.recovery_limit_per_lap / 1e6:.1f} MJ')
    ax6.set_ylabel('Energy (MJ)')
    ax6.set_xlabel('Distance (km)')
    ax6.set_title('Cumulative Energy')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # Track curvature
    ax7 = plt.subplot(4, 2, 7)
    segment_distances = np.array([seg.distance for seg in track.segments])
    segment_radii = np.array([seg.radius for seg in track.segments])
    
    curvature = 1000.0 / np.maximum(segment_radii, 50)  # Convert to 1/km
    
    ax7.fill_between(segment_distances / 1000, 0, curvature, alpha=0.4, color='purple')
    ax7.set_ylabel('Curvature (1/km)')
    ax7.set_xlabel('Distance (km)')
    ax7.set_title('Track Curvature')
    ax7.grid(True, alpha=0.3)
    
    # Lap time progression
    ax8 = plt.subplot(4, 2, 8)
    ax8.plot(s_km, optimal_trajectory.t_opt, 'b-', linewidth=2)
    ax8.set_ylabel('Time (s)')
    ax8.set_xlabel('Distance (km)')
    ax8.set_title('Cumulative Lap Time')
    ax8.grid(True, alpha=0.3)
    
    plt.suptitle(f'{track_name} - Complete ERS Analysis', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return fig