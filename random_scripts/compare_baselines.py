import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ERSConfig, VehicleConfig
from models import F1TrackModel, VehicleDynamicsModel
from solvers import ForwardBackwardSolver, SpatialNLPSolver
from simulation.lap import LapSimulator, LapResult  # Imported LapResult

# Import your strategies
from strategies import (
    GreedyStrategy,
    TargetSOCStrategy,
    AlwaysDeployStrategy,
    SmartRuleBasedStrategy
)

# =============================================================================
#  CONFIGURATION & STYLE
# =============================================================================

# High-contrast palette
COLORS = {
    'Offline Optimal':      '#00E676',  # Bright Green (Best)
    'Smart Heuristic':      '#2979FF',  # Bright Blue (Smart)
    'Target SOC':           '#D500F9',  # Purple (Balanced)
    'Pure Greedy':          '#FFC400',  # Amber (Basic)
    'Always Deploy':        '#FF1744',  # Red (Aggressive/Bad)
    'No ERS (Mechanical)':  '#B0BEC5',  # Grey (Baseline)
}

def set_plot_style():
    """Apply a clean, dark F1-style theme"""
    plt.style.use('dark_background')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'axes.grid': True,
        'grid.alpha': 0.2,
        'grid.linestyle': '--',
        'lines.linewidth': 2.0,
        'figure.dpi': 150
    })

def profile_to_result(profile, initial_soc):
    """Convert a VelocityProfile (Forward-Backward result) to a LapResult."""
    n = len(profile.s)
    # Create dummy controls (0 ERS, crude throttle/brake estimate from accel)
    throttle_est = np.clip(profile.a_x / 10.0, 0, 1)
    brake_est = np.clip(-profile.a_x / 15.0, 0, 1)
    
    return LapResult(
        times=profile.t,
        positions=profile.s,
        velocities=profile.v,
        socs=np.full(n, initial_soc), # Constant SOC
        P_ers_history=np.zeros(n),    # 0 Power
        throttle_history=throttle_est,
        brake_history=brake_est,
        lap_time=profile.lap_time,
        final_soc=initial_soc,
        energy_deployed=0.0,
        energy_recovered=0.0,
        completed=True
    )

# =============================================================================
#  MAIN COMPARISON LOGIC
# =============================================================================

def compare_all_baselines():
    """Run and compare all baseline strategies"""
    
    set_plot_style()

    print("="*70)
    print("  F1 ERS BASELINE COMPARISON")
    print("="*70)
    print("Comparing strategies...")

    # 1. Setup
    ers_config = ERSConfig()
    vehicle_config = VehicleConfig.for_monaco() 

    print("\nLoading Monaco track...")
    track = F1TrackModel(2025, 'Monaco', 'Q', ds=5.0)
    try:
        track.load_from_fastf1('NOR')
    except:
        print("   ⚠ FastF1 failed or not cached. Ensure data exists.")
        return

    vehicle_model = VehicleDynamicsModel(vehicle_config, ers_config)
    initial_soc = 0.5
    
    results = {}

    # -------------------------------------------------------------------------
    # 2. Reference Profiles
    # -------------------------------------------------------------------------
    
    # A. "No ERS" Profile (Mechanical Limit) -> For the table/plot
    print("\nComputing 'No ERS' Mechanical Limit...")
    fb_solver_no_ers = ForwardBackwardSolver(vehicle_model, track, use_ers_power=False)
    no_ers_profile = fb_solver_no_ers.solve(flying_lap=False)
    results['No ERS (Mechanical)'] = profile_to_result(no_ers_profile, initial_soc)
    print(f"   No ERS Time: {no_ers_profile.lap_time:.3f} s")

    # B. "Fast" Profile (With ERS Power) -> For the Baselines to track
    print("\nComputing 'Fast' Reference (Target for Baselines)...")
    fb_solver_fast = ForwardBackwardSolver(vehicle_model, track, use_ers_power=True)
    ref_profile = fb_solver_fast.solve(flying_lap=False)
    print(f"   Target Time: {ref_profile.lap_time:.3f} s")

    # -------------------------------------------------------------------------
    # 3. Run Strategies
    # -------------------------------------------------------------------------
    # Definition list: (Name, StrategyClass, Kwargs)
    strategies_to_run = [
        ('Always Deploy', AlwaysDeployStrategy, {}),
        ('Pure Greedy', GreedyStrategy, {}),
        ('Target SOC', TargetSOCStrategy, {'target_soc': 0.3}),
        ('Smart Heuristic', SmartRuleBasedStrategy, {})
    ]

    for name, Cls, kwargs in strategies_to_run:
        print(f"\nRunning {name}...")
        # Initialize with the FAST reference profile so they drive hard
        strategy = Cls(
            vehicle_config, ers_config, track,
            reference_profile=ref_profile,
            **kwargs
        )
        sim = LapSimulator(vehicle_model, track, strategy)
        results[name] = sim.simulate_lap(initial_soc=initial_soc)
        print(f"   Time: {results[name].lap_time:.3f}s | SOC: {results[name].final_soc*100:.1f}%")

    # 4. Offline Optimal (NLP)
    print("\nRunning Offline Optimal (NLP)...")
    optimizer = SpatialNLPSolver(vehicle_model, track, ers_config, ds=5.0)
    trajectory = optimizer.solve(
        v_limit_profile=ref_profile.v,
        s_limit_profile=ref_profile.s,
        initial_soc=initial_soc,
        final_soc_min=0.3,
        is_flying_lap=False
    )
    simulator = LapSimulator(vehicle_model, track, None)
    results['Offline Optimal'] = simulator.replay_trajectory(trajectory)
    print(f"   Time: {results['Offline Optimal'].lap_time:.3f}s | SOC: {results['Offline Optimal'].final_soc*100:.1f}%")

    # 5. Summary & Plot
    print_summary_table(results)
    
    print("\nGenerating improved plots...")
    plot_baseline_comparison(results, track, ers_config)
    print("✓ Done. Saved to 'baseline_comparison.png'")


def print_summary_table(results):
    print("\n" + "="*85)
    print(f"{'Strategy':<20} {'Lap Time':<12} {'Diff':<12} {'Energy(MJ)':<15} {'Final SOC':<10}")
    print("-"*85)

    # Sort by lap time (fastest first)
    sorted_results = sorted(results.items(), key=lambda x: x[1].lap_time)
    best_time = sorted_results[0][1].lap_time

    for name, res in sorted_results:
        diff = res.lap_time - best_time
        net_energy = (res.energy_deployed - res.energy_recovered) / 1e6
        
        diff_str = f"+{diff:.3f}s" if diff > 0.001 else "BEST"
        
        # Color code output if terminal supports it (optional)
        print(f"{name:<20} {res.lap_time:<12.3f} {diff_str:<12} {net_energy:<15.3f} {res.final_soc*100:<10.1f}%")
    print("="*85)


def plot_baseline_comparison(results, track, ers_config):
    """
    Create a professional, publication-quality comparison plot.
    """
    
    # Sort strategies by performance to keep legend/colors consistent
    sorted_keys = sorted(results.keys(), key=lambda k: results[k].lap_time)
    
    # Define line styles to cycle through
    LINE_STYLES = {
        'Offline Optimal': '-',
        'Smart Heuristic': '-',
        'Target SOC':      '-.',
        'Pure Greedy':     '--',
        'Always Deploy':   ':',
        'No ERS (Mechanical)': '--' # Dashed grey
    }
    
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], hspace=0.3, wspace=0.2)

    # ---------------------------------------------------------
    # 1. VELOCITY PROFILE (Top Left)
    # ---------------------------------------------------------
    ax1 = plt.subplot(gs[0, 0])
    
    start_m = track.total_length * 0.2
    end_m   = track.total_length * 0.5
    
    for name in sorted_keys:
        res = results[name]
        mask = (res.positions >= start_m) & (res.positions <= end_m)
        
        ls = LINE_STYLES.get(name, '-')
        lw = 2.5 if name == 'Offline Optimal' else 1.5
        alpha = 0.9 if name == 'Offline Optimal' else 0.7
        
        ax1.plot(res.positions[mask], res.velocities[mask] * 3.6, 
                 label=name, color=COLORS.get(name, 'white'), linestyle=ls, linewidth=lw, alpha=alpha)
        
    ax1.set_title(f"Velocity Profile (Sector 2 Zoom)", fontweight='bold')
    ax1.set_ylabel("Speed (km/h)")
    ax1.set_xlabel("Track Position (m)")
    ax1.legend(loc='lower right', frameon=True, facecolor='#222')

    # ---------------------------------------------------------
    # 2. BATTERY SOC EVOLUTION (Top Right)
    # ---------------------------------------------------------
    ax2 = plt.subplot(gs[0, 1])
    
    ax2.axhline(ers_config.max_soc * 100, color='grey', linestyle=':', alpha=0.5, label='Max Limit')
    ax2.axhline(ers_config.min_soc * 100, color='grey', linestyle=':', alpha=0.5, label='Min Limit')

    for name in sorted_keys:
        # Skip No ERS in SOC plot (it's a flat line)
        if "No ERS" in name: continue
            
        res = results[name]
        ls = LINE_STYLES.get(name, '-')
        ax2.plot(res.positions, res.socs * 100, 
                 label=name, color=COLORS.get(name, 'white'), linestyle=ls, linewidth=2)

    ax2.set_title("Energy Management Strategy (SOC)", fontweight='bold')
    ax2.set_ylabel("Battery Charge (%)")
    ax2.set_xlabel("Track Position (m)")
    ax2.set_ylim(ers_config.min_soc * 100 - 5, ers_config.max_soc * 100 + 5)
    
    # ---------------------------------------------------------
    # 3. ERS POWER DEPLOYMENT (Bottom Left)
    # ---------------------------------------------------------
    ax3 = plt.subplot(gs[1, 0])
    
    start_m = track.total_length * 0.6
    end_m   = track.total_length * 0.9
    
    for name in sorted_keys:
        if "No ERS" in name: continue
            
        res = results[name]
        mask = (res.positions[:-1] >= start_m) & (res.positions[:-1] <= end_m)
        
        if np.any(mask):
            ls = LINE_STYLES.get(name, '-')
            ax3.step(res.positions[:-1][mask], res.P_ers_history[mask] / 1000, 
                     where='post', label=name, color=COLORS.get(name, 'white'), linestyle=ls, linewidth=1.5)

    ax3.axhline(0, color='white', linewidth=1, alpha=0.3)
    ax3.set_title("ERS Power Usage (Sector 3 Zoom)", fontweight='bold')
    ax3.set_ylabel("Power (kW) [+Deploy / -Harvest]")
    ax3.set_xlabel("Track Position (m)")
    
    # ---------------------------------------------------------
    # 4. LAP TIME COMPARISON (Bottom Right)
    # ---------------------------------------------------------
    ax4 = plt.subplot(gs[1, 1])
    
    names = sorted_keys
    times = [results[n].lap_time for n in names]
    best_time = min(times)
    diffs = [t - best_time for t in times]
    colors = [COLORS.get(n, 'white') for n in names]
    
    y_pos = np.arange(len(names))
    bars = ax4.barh(y_pos, diffs, color=colors, alpha=0.8, edgecolor='white')
    
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(names, fontweight='bold')
    ax4.set_xlabel(f"Time Loss vs Best (seconds) | Best: {best_time:.3f}s")
    ax4.set_title("Final Performance Delta", fontweight='bold')
    
    for i, bar in enumerate(bars):
        width = bar.get_width()
        label = f"+{width:.3f}s" if width > 0 else "BEST"
        ax4.text(width + 0.05, bar.get_y() + bar.get_height()/2, label, 
                 va='center', fontweight='bold', color=colors[i])

    plt.tight_layout()
    plt.savefig('baseline_comparison.png', facecolor='#121212', edgecolor='none')
    plt.close()

if __name__ == "__main__":
    compare_all_baselines()