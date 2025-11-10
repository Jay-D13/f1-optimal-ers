from typing import Dict
import matplotlib.pyplot as plt

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
