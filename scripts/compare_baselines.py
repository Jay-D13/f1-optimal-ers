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
from simulation.lap import LapSimulator, LapResult

from strategies import (
    GreedyStrategy,
    TargetSOCStrategy,
    AlwaysDeployStrategy,
    SmartRuleBasedStrategy,
    OptimalTrackingStrategy
)


# =============================================================================
#  CONFIGURATION
# =============================================================================

COLORS = {
    'Offline Optimal':      '#00E676',
    'Optimal Tracking':     '#00BCD4',
    'Smart Heuristic':      '#2979FF',
    'Target SOC':           '#D500F9',
    'Pure Greedy':          '#FFC400',
    'Always Deploy':        '#FF1744',
    'No ERS (Mechanical)':  '#B0BEC5',
}

def set_plot_style():
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
    """Convert a VelocityProfile to a LapResult."""
    n = len(profile.s)
    throttle_est = np.clip(profile.a_x / 10.0, 0, 1)
    brake_est = np.clip(-profile.a_x / 15.0, 0, 1)
    
    return LapResult(
        times=profile.t,
        positions=profile.s,
        velocities=profile.v,
        socs=np.full(n, initial_soc),
        P_ers_history=np.zeros(n),
        throttle_history=throttle_est,
        brake_history=brake_est,
        lap_time=profile.lap_time,
        final_soc=initial_soc,
        energy_deployed=0.0,
        energy_recovered=0.0,
        completed=True
    )


def compare_all_baselines():
    """Run and compare all baseline strategies with correct physics."""
    
    set_plot_style()

    print("="*70)
    print("  F1 ERS BASELINE COMPARISON")
    print("="*70)

    ers_config = ERSConfig()
    vehicle_config = VehicleConfig.for_monaco()

    print("\nLoading Monaco track...")
    track = F1TrackModel(2024, 'Monaco', 'Q', ds=5.0)
    track.load_from_fastf1('LEC')
    
    # some problems sometimes with TUMFTM racelines, gotta fix
    # try:
    #     track.load_from_fastf1('VER')
    #     print("   ✓ Loaded from FastF1")
    # except Exception as e:
    #     print(f"   ⚠ FastF1 failed: {e}")
    #     print("   Attempting TUMFTM raceline...")
    #     try:
    #         track.load_from_tumftm_raceline('/path/to/monaco_raceline.csv')
    #     except:
    #         print("   ✗ Could not load track data")
    #         return

    vehicle_model = VehicleDynamicsModel(vehicle_config, ers_config)
    initial_soc = 0.5
    
    results = {}

    # -------------------------------------------------------------------------
    #  Reference Profiles
    # -------------------------------------------------------------------------
    
    # "No ERS" Profile (Mechanical Limit)
    print("\nComputing 'No ERS' Mechanical Limit...")
    fb_solver_no_ers = ForwardBackwardSolver(vehicle_model, track, use_ers_power=False)
    no_ers_profile = fb_solver_no_ers.solve(flying_lap=True)
    results['No ERS (Mechanical)'] = profile_to_result(no_ers_profile, initial_soc)
    print(f"   No ERS Time: {no_ers_profile.lap_time:.3f} s")

    # "Fast" Profile (With ERS Power) -> For baselines to track
    # IMPORTANT: Baselines will track this, accounting for ERS in their controls
    print("\nComputing 'Fast' Reference (ICE + Full ERS)...")
    fb_solver_fast = ForwardBackwardSolver(vehicle_model, track, use_ers_power=True)
    ref_profile_fast = fb_solver_fast.solve(flying_lap=True)
    print(f"   Fast Time: {ref_profile_fast.lap_time:.3f} s")

    # -------------------------------------------------------------------------
    #  Run NLP Optimal Solution First
    # -------------------------------------------------------------------------
    print("\nRunning Offline Optimal (NLP)...")
    optimizer = SpatialNLPSolver(vehicle_model, track, ers_config, ds=5.0)
    optimal_trajectory = optimizer.solve(
        v_limit_profile=ref_profile_fast.v,
        initial_soc=initial_soc,
        final_soc_min=0.3,
        is_flying_lap=False
    )
    simulator = LapSimulator(vehicle_model, track, None)
    results['Offline Optimal'] = simulator.replay_trajectory(optimal_trajectory)
    print(f"   NLP Time: {results['Offline Optimal'].lap_time:.3f}s")

    # -------------------------------------------------------------------------
    #  Run Baseline Strategies
    # -------------------------------------------------------------------------
    
    ref_for_baselines = ref_profile_fast
    
    strategies_to_run = [
        ('Always Deploy', AlwaysDeployStrategy, {}),
        ('Pure Greedy', GreedyStrategy, {}),
        ('Target SOC', TargetSOCStrategy, {'target_soc': 0.3}),
        ('Smart Heuristic', SmartRuleBasedStrategy, {}),
        ('Optimal Tracking', OptimalTrackingStrategy, {}),
    ]

    for name, Cls, kwargs in strategies_to_run:
        print(f"\nRunning {name}...")
        
        # optimal trajectory for OptimalTrackingStrategy, else FB profile for the rest
        if name == 'Optimal Tracking':
            ref = optimal_trajectory
            v_start_target = ref.v_opt[0] # Flying lap for baselines
        else:
            ref = ref_for_baselines
            v_start_target = ref.v[0] # Flying lap for baselines
        
        strategy = Cls(
            vehicle_config, ers_config, track,
            reference_profile=ref,
            **kwargs
        )

        sim = LapSimulator(vehicle_model, track, strategy)
        result = sim.simulate_lap(
            initial_soc=initial_soc,
            initial_velocity=v_start_target 
        )
        results[name] = result
        print(f"   Time: {result.lap_time:.3f}s | SOC: {result.final_soc*100:.1f}%")

    # -------------------------------------------------------------------------
    # 5. Summary & Analysis
    # -------------------------------------------------------------------------
    print_summary_table(results)
    
    print("\nGenerating comparison plots...")
    plot_baseline_comparison(results, track, ers_config)
    print("✓ Done. Saved to 'baseline_comparison.png'")
    
    # Diagnostic plots
    print("\nGenerating diagnostic plots...")
    plot_diagnostics(results, track)
    print("✓ Diagnostics saved to 'baseline_diagnostics.png'")
    
    return results


def print_summary_table(results):
    print("\n" + "="*90)
    print(f"{'Strategy':<20} {'Lap Time':<12} {'Δ vs Opt':<12} {'Net Energy(MJ)':<15} {'Final SOC':<10}")
    print("-"*90)

    sorted_results = sorted(results.items(), key=lambda x: x[1].lap_time)
    best_time = sorted_results[0][1].lap_time

    for name, res in sorted_results:
        diff = res.lap_time - best_time
        net_energy = (res.energy_deployed - res.energy_recovered) / 1e6
        diff_str = f"+{diff:.3f}s" if diff > 0.001 else "BEST"
        
        print(f"{name:<20} {res.lap_time:<12.3f} {diff_str:<12} {net_energy:<15.3f} {res.final_soc*100:<10.1f}%")
    print("="*90)


def plot_baseline_comparison(results, track, ers_config):
    
    sorted_keys = sorted(results.keys(), key=lambda k: results[k].lap_time)
    
    LINE_STYLES = {
        'Offline Optimal': '-',
        'Optimal Tracking': '-',
        'Smart Heuristic': '-',
        'Target SOC': '-.',
        'Pure Greedy': '--',
        'Always Deploy': ':',
        'No ERS (Mechanical)': '--'
    }
    
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], hspace=0.3, wspace=0.2)

    ax1 = plt.subplot(gs[0, 0])
    
    for name in sorted_keys:
        res = results[name]
        ls = LINE_STYLES.get(name, '-')
        lw = 2.5 if name == 'Offline Optimal' else 1.5
        alpha = 0.9 if name == 'Offline Optimal' else 0.7
        
        ax1.plot(res.positions, res.velocities * 3.6, 
                 label=name, color=COLORS.get(name, 'white'), 
                 linestyle=ls, linewidth=lw, alpha=alpha)
        
    ax1.set_title("Velocity Profile (Full Lap)", fontweight='bold')
    ax1.set_ylabel("Speed (km/h)")
    ax1.set_xlabel("Track Position (m)")
    ax1.legend(loc='lower right', frameon=True, facecolor='#222')

    ax2 = plt.subplot(gs[0, 1])
    
    ax2.axhline(ers_config.max_soc * 100, color='grey', linestyle=':', alpha=0.5)
    ax2.axhline(ers_config.min_soc * 100, color='grey', linestyle=':', alpha=0.5)

    for name in sorted_keys:
        if "No ERS" in name:
            continue
        res = results[name]
        ls = LINE_STYLES.get(name, '-')
        ax2.plot(res.positions, res.socs * 100, 
                 label=name, color=COLORS.get(name, 'white'), 
                 linestyle=ls, linewidth=2)

    ax2.set_title("Energy Management (SOC)", fontweight='bold')
    ax2.set_ylabel("Battery Charge (%)")
    ax2.set_xlabel("Track Position (m)")
    ax2.set_ylim(ers_config.min_soc * 100 - 5, ers_config.max_soc * 100 + 5)
    ax2.legend(loc='upper right', frameon=True, facecolor='#222')

    ax3 = plt.subplot(gs[1, 0])
    
    for name in sorted_keys:
        if "No ERS" in name:
            continue
        res = results[name]
        ls = LINE_STYLES.get(name, '-')
        # positions[:-1] since P_ers has one fewer point
        n_ers = len(res.P_ers_history)
        ax3.step(res.positions[:n_ers], res.P_ers_history / 1000, 
                 where='post', label=name, color=COLORS.get(name, 'white'), 
                 linestyle=ls, linewidth=1.5, alpha=0.7)

    ax3.axhline(0, color='white', linewidth=1, alpha=0.3)
    ax3.set_title("ERS Power Usage", fontweight='bold')
    ax3.set_ylabel("Power (kW) [+Deploy / -Harvest]")
    ax3.set_xlabel("Track Position (m)")

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
    ax4.set_xlabel(f"Time Loss vs Best (s) | Best: {best_time:.3f}s")
    ax4.set_title("Performance Delta", fontweight='bold')
    
    for i, bar in enumerate(bars):
        width = bar.get_width()
        label = f"+{width:.3f}s" if width > 0.001 else "BEST"
        ax4.text(width + 0.02, bar.get_y() + bar.get_height()/2, label, 
                 va='center', fontweight='bold', color=colors[i], fontsize=9)

    plt.tight_layout()
    plt.savefig('figures/baseline_comparison.png', 
                facecolor='#121212', edgecolor='none', dpi=150)
    plt.close()


def plot_diagnostics(results, track):
    """Plot diagnostic information to verify physics consistency."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    diag_strategies = ['Offline Optimal', 'Smart Heuristic', 'Pure Greedy']
    colors = ['#00E676', '#2979FF', '#FFC400']
    
    # Throttle comparison
    ax1 = axes[0, 0]
    for name, color in zip(diag_strategies, colors):
        if name in results:
            res = results[name]
            n = len(res.throttle_history)
            ax1.plot(res.positions[:n], res.throttle_history, 
                    label=name, color=color, alpha=0.7)
    ax1.set_title("Throttle Command")
    ax1.set_ylabel("Throttle (0-1)")
    ax1.set_xlabel("Position (m)")
    ax1.legend()
    ax1.set_ylim(-0.05, 1.05)
    
    # Brake comparison  
    ax2 = axes[0, 1]
    for name, color in zip(diag_strategies, colors):
        if name in results:
            res = results[name]
            n = len(res.brake_history)
            ax2.plot(res.positions[:n], res.brake_history,
                    label=name, color=color, alpha=0.7)
    ax2.set_title("Brake Command")
    ax2.set_ylabel("Brake (0-1)")
    ax2.set_xlabel("Position (m)")
    ax2.legend()
    ax2.set_ylim(-0.05, 1.05)
    
    # Velocity tracking error
    ax3 = axes[1, 0]
    optimal = results.get('Offline Optimal')
    if optimal:
        v_opt_interp = np.interp
        for name, color in zip(diag_strategies[1:], colors[1:]):
            if name in results:
                res = results[name]
                # Interpolate optimal velocity at result positions
                v_opt_at_pos = np.interp(res.positions, optimal.positions, optimal.velocities)
                v_error = res.velocities - v_opt_at_pos
                ax3.plot(res.positions, v_error * 3.6, label=name, color=color, alpha=0.7)
    ax3.axhline(0, color='white', linestyle='--', alpha=0.3)
    ax3.set_title("Velocity Error vs Optimal")
    ax3.set_ylabel("Error (km/h)")
    ax3.set_xlabel("Position (m)")
    ax3.legend()
    
    # Cumulative time difference
    ax4 = axes[1, 1]
    if optimal:
        for name, color in zip(diag_strategies[1:], colors[1:]):
            if name in results:
                res = results[name]
                # Compare times at same positions
                t_opt_at_pos = np.interp(res.positions, optimal.positions, optimal.times)
                t_diff = res.times - t_opt_at_pos
                ax4.plot(res.positions, t_diff, label=name, color=color, alpha=0.7)
    ax4.axhline(0, color='white', linestyle='--', alpha=0.3)
    ax4.set_title("Cumulative Time Delta vs Optimal")
    ax4.set_ylabel("Time Difference (s)")
    ax4.set_xlabel("Position (m)")
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('figures/baseline_diagnostics.png',
                facecolor='#121212', edgecolor='none', dpi=150)
    plt.close()


if __name__ == "__main__":
    compare_all_baselines()
