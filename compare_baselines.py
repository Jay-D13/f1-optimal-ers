"""
Compare Different Baseline Strategies

This script compares:
1. Offline Optimal (theoretical best)
2. Smart Heuristic (your current baseline with lookahead)
3. Pure Greedy (classic literature baseline)
4. Always Deploy (worst case baseline)

This shows the spectrum from worst to best and validates that your
optimal strategy makes sense.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import ERSConfig, VehicleConfig
from models import F1TrackModel, VehicleDynamicsModel
from controllers import (
    GlobalOfflineOptimizer,
    SimpleRuleBasedStrategy,
    PureGreedyStrategy,
    AlwaysDeployGreedy
)
from simulation import LapSimulator


def compare_all_baselines():
    """Run and compare all baseline strategies"""

    print("="*70)
    print("  F1 ERS BASELINE COMPARISON")
    print("="*70)
    print("\nComparing 4 strategies from worst to best:")
    print("  1. Always Deploy (worst - no intelligence)")
    print("  2. Pure Greedy (literature baseline)")
    print("  3. Smart Heuristic (your current baseline)")
    print("  4. Offline Optimal (theoretical best)")
    print("="*70)

    # Setup
    ers_config = ERSConfig()
    vehicle_config = VehicleConfig.for_monaco()

    print("\nLoading Monaco track...")
    track = F1TrackModel(2025, 'Monaco', 'Q', ds=10.0)
    track.load_from_fastf1('NOR')
    v_max = track.compute_speed_limits(vehicle_config)
    print(f"   ✓ Track loaded: {track.total_length:.0f}m")

    vehicle_model = VehicleDynamicsModel(vehicle_config, ers_config)
    print("   ✓ Vehicle model ready")

    initial_soc = 0.5

    # Results storage
    results = {}

    # 1. Always Deploy (Worst)
    print("\n" + "="*70)
    print("1. ALWAYS DEPLOY GREEDY (Worst)")
    print("="*70)
    print("Strategy: Deploy max power whenever not braking")

    always_deploy = AlwaysDeployGreedy(track, vehicle_config, ers_config)
    sim_always = LapSimulator(vehicle_model, track, always_deploy)
    results['always_deploy'] = sim_always.simulate_lap(initial_soc=initial_soc)

    print(f"   Lap time:      {results['always_deploy'].lap_time:.3f} s")
    print(f"   Final SOC:     {results['always_deploy'].final_soc*100:.1f}%")
    print(f"   Energy used:   {(results['always_deploy'].energy_deployed - results['always_deploy'].energy_recovered)/1e6:.2f} MJ")

    # 2. Pure Greedy
    print("\n" + "="*70)
    print("2. PURE GREEDY (Literature Baseline)")
    print("="*70)
    print("Strategy: Deploy on straights (R>500m), harvest when braking")

    pure_greedy = PureGreedyStrategy(track, vehicle_config, ers_config)
    sim_greedy = LapSimulator(vehicle_model, track, pure_greedy)
    results['pure_greedy'] = sim_greedy.simulate_lap(initial_soc=initial_soc)

    improvement = results['always_deploy'].lap_time - results['pure_greedy'].lap_time
    print(f"   Lap time:      {results['pure_greedy'].lap_time:.3f} s ({improvement:+.3f}s vs Always Deploy)")
    print(f"   Final SOC:     {results['pure_greedy'].final_soc*100:.1f}%")
    print(f"   Energy used:   {(results['pure_greedy'].energy_deployed - results['pure_greedy'].energy_recovered)/1e6:.2f} MJ")

    # 3. Smart Heuristic
    print("\n" + "="*70)
    print("3. SMART HEURISTIC (Your Baseline)")
    print("="*70)
    print("Strategy: Lookahead, deploy at full throttle, avoid drag-limited speeds")

    smart_heuristic = SimpleRuleBasedStrategy(track, vehicle_config, ers_config)
    sim_smart = LapSimulator(vehicle_model, track, smart_heuristic)
    results['smart_heuristic'] = sim_smart.simulate_lap(initial_soc=initial_soc)

    improvement = results['pure_greedy'].lap_time - results['smart_heuristic'].lap_time
    print(f"   Lap time:      {results['smart_heuristic'].lap_time:.3f} s ({improvement:+.3f}s vs Pure Greedy)")
    print(f"   Final SOC:     {results['smart_heuristic'].final_soc*100:.1f}%")
    print(f"   Energy used:   {(results['smart_heuristic'].energy_deployed - results['smart_heuristic'].energy_recovered)/1e6:.2f} MJ")

    # 4. Offline Optimal
    print("\n" + "="*70)
    print("4. OFFLINE OPTIMAL (Theoretical Best)")
    print("="*70)
    print("Strategy: Global NLP optimization over entire lap")

    optimizer = GlobalOfflineOptimizer(vehicle_model, track, ds=10.0)
    optimizer.setup_nlp(initial_soc=initial_soc, final_soc_min=0.3, energy_limit=4e6)

    print("\nSolving (this takes ~1-2 minutes)...")
    trajectory = optimizer.solve()

    simulator = LapSimulator(vehicle_model, track, None)
    results['offline_optimal'] = simulator.replay_trajectory(trajectory)

    improvement = results['smart_heuristic'].lap_time - results['offline_optimal'].lap_time
    print(f"   Lap time:      {results['offline_optimal'].lap_time:.3f} s ({improvement:+.3f}s vs Smart Heuristic)")
    print(f"   Final SOC:     {results['offline_optimal'].final_soc*100:.1f}%")
    print(f"   Energy used:   {(results['offline_optimal'].energy_deployed - results['offline_optimal'].energy_recovered)/1e6:.2f} MJ")

    # Summary Table
    print("\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)
    print(f"\n{'Strategy':<25} {'Lap Time (s)':<15} {'vs Best':<15} {'Final SOC':<15}")
    print("-"*70)

    best_time = results['offline_optimal'].lap_time

    strategies_ordered = [
        ('4. Offline Optimal', 'offline_optimal'),
        ('3. Smart Heuristic', 'smart_heuristic'),
        ('2. Pure Greedy', 'pure_greedy'),
        ('1. Always Deploy', 'always_deploy'),
    ]

    for name, key in strategies_ordered:
        result = results[key]
        delta = result.lap_time - best_time
        delta_pct = (delta / best_time) * 100
        print(f"{name:<25} {result.lap_time:<15.3f} +{delta:.3f}s ({delta_pct:+.1f}%) {result.final_soc*100:<15.1f}%")

    print("\n" + "="*70)
    print("KEY INSIGHTS")
    print("="*70)

    total_improvement = results['always_deploy'].lap_time - results['offline_optimal'].lap_time
    total_pct = (total_improvement / results['always_deploy'].lap_time) * 100

    print(f"\n✓ Total improvement (worst to best): {total_improvement:.3f}s ({total_pct:.1f}%)")

    greedy_gap = results['pure_greedy'].lap_time - results['offline_optimal'].lap_time
    greedy_pct = (greedy_gap / results['pure_greedy'].lap_time) * 100
    print(f"✓ Optimal vs Pure Greedy (literature): {greedy_gap:.3f}s ({greedy_pct:.1f}%)")

    smart_gap = results['smart_heuristic'].lap_time - results['offline_optimal'].lap_time
    smart_pct = (smart_gap / results['smart_heuristic'].lap_time) * 100
    print(f"✓ Optimal vs Smart Heuristic: {smart_gap:.3f}s ({smart_pct:.1f}%)")

    print("\n" + "="*70)
    print("VALIDATION")
    print("="*70)

    # Check expected improvements
    if greedy_pct > 1.0 and greedy_pct < 10.0:
        print("✓ PASS: Optimal is 1-10% better than Pure Greedy (literature expectation)")
    else:
        print(f"⚠ WARNING: Optimal improvement ({greedy_pct:.1f}%) outside typical 1-10% range")

    if results['smart_heuristic'].lap_time < results['pure_greedy'].lap_time:
        print("✓ PASS: Smart Heuristic beats Pure Greedy (validates lookahead logic)")
    else:
        print("⚠ WARNING: Smart Heuristic should be better than Pure Greedy")

    if results['offline_optimal'].lap_time < results['smart_heuristic'].lap_time:
        print("✓ PASS: Offline Optimal is the fastest (as expected)")
    else:
        print("⚠ WARNING: Optimal should be the fastest strategy")

    # Create comparison plot
    plot_baseline_comparison(results, track)

    print("\n" + "="*70)
    print("✓ Comparison complete! See baseline_comparison.png")
    print("="*70)

    return results


def plot_baseline_comparison(results, track):
    """Create comprehensive comparison plots"""

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('F1 ERS Baseline Strategy Comparison', fontsize=14, fontweight='bold')

    strategies = [
        ('Offline Optimal', 'offline_optimal', '#4ECDC4'),
        ('Smart Heuristic', 'smart_heuristic', '#95E1D3'),
        ('Pure Greedy', 'pure_greedy', '#FFD93D'),
        ('Always Deploy', 'always_deploy', '#FF6B6B'),
    ]

    # Plot 1: Lap time comparison
    ax = axes[0, 0]
    names = [s[0] for s in strategies]
    times = [results[s[1]].lap_time for s in strategies]
    colors = [s[2] for s in strategies]

    bars = ax.barh(names, times, color=colors, alpha=0.8, edgecolor='black')
    ax.set_xlabel('Lap Time (s)', fontweight='bold')
    ax.set_title('Lap Time Comparison')
    ax.grid(axis='x', alpha=0.3)

    for bar, time in zip(bars, times):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2,
                f' {time:.2f}s', ha='left', va='center', fontweight='bold')

    # Plot 2: Energy usage
    ax = axes[0, 1]
    deploy = [results[s[1]].energy_deployed / 1e6 for s in strategies]
    recover = [results[s[1]].energy_recovered / 1e6 for s in strategies]

    x = np.arange(len(strategies))
    width = 0.35

    ax.bar(x - width/2, deploy, width, label='Deployed', color='#FF6B6B', alpha=0.7)
    ax.bar(x + width/2, recover, width, label='Recovered', color='#95E1D3', alpha=0.7)
    ax.set_ylabel('Energy (MJ)', fontweight='bold')
    ax.set_title('ERS Energy Usage')
    ax.set_xticks(x)
    ax.set_xticklabels([s[0] for s in strategies], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Plot 3: Final SOC
    ax = axes[0, 2]
    socs = [results[s[1]].final_soc * 100 for s in strategies]
    bars = ax.bar(names, socs, color=colors, alpha=0.8, edgecolor='black')
    ax.set_ylabel('SOC (%)', fontweight='bold')
    ax.set_title('Final Battery SOC')
    ax.grid(axis='y', alpha=0.3)
    ax.set_xticklabels(names, rotation=45, ha='right')

    # Plot 4: Speed profiles
    ax = axes[1, 0]
    for name, key, color in strategies:
        result = results[key]
        ax.plot(result.positions, result.velocities * 3.6,
                label=name, color=color, alpha=0.7, linewidth=2)
    ax.set_xlabel('Track Position (m)', fontweight='bold')
    ax.set_ylabel('Speed (km/h)', fontweight='bold')
    ax.set_title('Speed Profiles')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Plot 5: ERS power profiles
    ax = axes[1, 1]
    for name, key, color in strategies:
        result = results[key]
        ax.plot(result.positions[:-1], result.P_ers_history / 1000,
                label=name, color=color, alpha=0.7, linewidth=2)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    ax.set_xlabel('Track Position (m)', fontweight='bold')
    ax.set_ylabel('ERS Power (kW)', fontweight='bold')
    ax.set_title('ERS Power Strategies')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Plot 6: SOC evolution
    ax = axes[1, 2]
    for name, key, color in strategies:
        result = results[key]
        ax.plot(result.positions, result.socs * 100,
                label=name, color=color, alpha=0.7, linewidth=2)
    ax.set_xlabel('Track Position (m)', fontweight='bold')
    ax.set_ylabel('Battery SOC (%)', fontweight='bold')
    ax.set_title('Battery State of Charge')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('baseline_comparison.png', dpi=150, bbox_inches='tight')
    print("\n   ✓ Comparison plots saved to baseline_comparison.png")


if __name__ == "__main__":
    results = compare_all_baselines()
