"""
F1 ERS Optimization - Project Validation Suite

This script validates that the optimal ERS project makes sense by checking:
1. Physical plausibility (energy conservation, power limits)
2. Performance metrics (lap time improvements, ERS usage)
3. Strategy intelligence (deploy on straights, harvest on braking)
4. Different objectives (lap time vs energy efficiency)
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).parent))

from config import ERSConfig, VehicleConfig
from models import F1TrackModel, VehicleDynamicsModel
from controllers import GlobalOfflineOptimizer, SimpleRuleBasedStrategy, PureGreedyStrategy
from simulation import LapSimulator


class ValidationReport:
    """Container for validation results"""

    def __init__(self):
        self.tests = {}
        self.passed = 0
        self.failed = 0

    def add_test(self, name: str, passed: bool, message: str, value=None):
        """Add a test result"""
        self.tests[name] = {
            'passed': passed,
            'message': message,
            'value': value
        }
        if passed:
            self.passed += 1
        else:
            self.failed += 1

    def print_summary(self):
        """Print validation summary"""
        print("\n" + "="*70)
        print("VALIDATION REPORT")
        print("="*70)

        for name, result in self.tests.items():
            status = "✓ PASS" if result['passed'] else "✗ FAIL"
            print(f"\n{status}: {name}")
            print(f"  {result['message']}")
            if result['value'] is not None:
                print(f"  Value: {result['value']}")

        print("\n" + "="*70)
        print(f"SUMMARY: {self.passed} passed, {self.failed} failed")
        print("="*70)

        return self.failed == 0


def validate_physics(result, ers_config: ERSConfig, report: ValidationReport):
    """Validate physical plausibility"""

    print("\n" + "="*70)
    print("1. PHYSICS VALIDATION")
    print("="*70)

    # Test 1: Energy conservation
    net_energy = result.energy_deployed - result.energy_recovered
    max_allowed = ers_config.battery_usable_energy

    report.add_test(
        "Energy Conservation",
        net_energy <= max_allowed * 1.01,  # Allow 1% numerical error
        f"Net energy deployed: {net_energy/1e6:.2f} MJ (limit: {max_allowed/1e6:.2f} MJ)",
        f"{net_energy/1e6:.2f} MJ"
    )

    # Test 2: Power limits respected
    max_deployment = np.max(result.P_ers_history[result.P_ers_history > 0])
    max_recovery = np.abs(np.min(result.P_ers_history[result.P_ers_history < 0]))

    report.add_test(
        "Deployment Power Limit",
        max_deployment <= ers_config.max_deployment_power * 1.01,
        f"Max deployment: {max_deployment/1000:.1f} kW (limit: {ers_config.max_deployment_power/1000:.1f} kW)",
        f"{max_deployment/1000:.1f} kW"
    )

    report.add_test(
        "Recovery Power Limit",
        max_recovery <= ers_config.max_recovery_power * 1.01,
        f"Max recovery: {max_recovery/1000:.1f} kW (limit: {ers_config.max_recovery_power/1000:.1f} kW)",
        f"{max_recovery/1000:.1f} kW"
    )

    # Test 3: SOC bounds
    min_soc = np.min(result.socs)
    max_soc = np.max(result.socs)

    report.add_test(
        "SOC Bounds",
        min_soc >= 0.0 and max_soc <= 1.0,
        f"SOC range: {min_soc*100:.1f}% - {max_soc*100:.1f}%",
        f"{min_soc*100:.1f}% - {max_soc*100:.1f}%"
    )

    # Test 4: Velocity plausibility
    max_v = np.max(result.velocities)
    min_v = np.min(result.velocities)

    report.add_test(
        "Velocity Plausibility",
        min_v > 5.0 and max_v < 110.0,  # Reasonable F1 speeds in m/s
        f"Speed range: {min_v*3.6:.1f} - {max_v*3.6:.1f} km/h",
        f"{min_v*3.6:.1f} - {max_v*3.6:.1f} km/h"
    )


def validate_performance(offline_result, baseline_result, report: ValidationReport):
    """Validate performance improvements"""

    print("\n" + "="*70)
    print("2. PERFORMANCE VALIDATION")
    print("="*70)

    # Test 5: Lap time improvement
    improvement = baseline_result.lap_time - offline_result.lap_time
    improvement_pct = (improvement / baseline_result.lap_time) * 100

    report.add_test(
        "Lap Time Improvement vs Baseline",
        improvement > 0,  # Should be faster than baseline
        f"Offline: {offline_result.lap_time:.2f}s, Baseline: {baseline_result.lap_time:.2f}s, Improvement: {improvement:.2f}s ({improvement_pct:.1f}%)",
        f"{improvement:.2f}s ({improvement_pct:.1f}%)"
    )

    # Test 6: Reasonable lap time
    # Monaco quali lap ~70-80s in real life
    report.add_test(
        "Realistic Lap Time",
        60 < offline_result.lap_time < 100,
        f"Offline lap time: {offline_result.lap_time:.2f}s (Monaco typical: 70-80s)",
        f"{offline_result.lap_time:.2f}s"
    )

    # Test 7: ERS actually used
    offline_deployed = offline_result.energy_deployed / 1e6
    baseline_deployed = baseline_result.energy_deployed / 1e6

    report.add_test(
        "ERS Actually Deployed",
        offline_deployed > 0.5,  # Should use significant ERS (>0.5 MJ)
        f"Offline deployed: {offline_deployed:.2f} MJ, Baseline: {baseline_deployed:.2f} MJ",
        f"{offline_deployed:.2f} MJ"
    )

    # Test 8: Energy recovered
    offline_recovered = offline_result.energy_recovered / 1e6

    report.add_test(
        "ERS Recovery Active",
        offline_recovered > 0.5,  # Should recover energy
        f"Energy recovered: {offline_recovered:.2f} MJ",
        f"{offline_recovered:.2f} MJ"
    )


def validate_strategy_intelligence(result, track: F1TrackModel, report: ValidationReport):
    """Validate that the strategy makes intelligent decisions"""

    print("\n" + "="*70)
    print("3. STRATEGY INTELLIGENCE VALIDATION")
    print("="*70)

    # Analyze where ERS is deployed vs recovered
    positions = result.positions % track.total_length

    # Find straight sections (large radius)
    straight_mask = []
    corner_mask = []
    accel_mask = []
    brake_mask = []

    for pos in positions[:-1]:  # Exclude last to match array sizes
        segment = track.get_segment_at_distance(pos)
        if segment.radius > 500:  # Straight
            straight_mask.append(True)
            corner_mask.append(False)
        else:  # Corner
            straight_mask.append(False)
            corner_mask.append(True)

    straight_mask = np.array(straight_mask)
    corner_mask = np.array(corner_mask)

    # Calculate velocity changes (acceleration vs braking)
    dv = np.diff(result.velocities)
    accel_mask = dv > 0.1
    brake_mask = dv < -0.1

    # ERS deployment/recovery masks
    deploy_mask = result.P_ers_history > 1000  # >1kW deployment
    recover_mask = result.P_ers_history < -1000  # >1kW recovery

    # Test 9: Deploy on straights preferentially
    if np.sum(straight_mask) > 0 and np.sum(deploy_mask) > 0:
        deploy_on_straight_pct = np.sum(deploy_mask & straight_mask) / np.sum(deploy_mask) * 100

        report.add_test(
            "Deploy Preferentially on Straights",
            deploy_on_straight_pct > 30,  # At least 30% on straights
            f"{deploy_on_straight_pct:.1f}% of deployment on straights",
            f"{deploy_on_straight_pct:.1f}%"
        )
    else:
        report.add_test(
            "Deploy Preferentially on Straights",
            False,
            "Insufficient straight sections or deployment detected",
            "N/A"
        )

    # Test 10: Recover during braking
    if np.sum(brake_mask) > 0 and np.sum(recover_mask) > 0:
        recover_while_braking_pct = np.sum(recover_mask & brake_mask) / np.sum(recover_mask) * 100

        report.add_test(
            "Recover During Braking",
            recover_while_braking_pct > 20,  # At least 20% during braking
            f"{recover_while_braking_pct:.1f}% of recovery during braking",
            f"{recover_while_braking_pct:.1f}%"
        )
    else:
        report.add_test(
            "Recover During Braking",
            False,
            "Insufficient braking or recovery detected",
            "N/A"
        )

    # Test 11: Don't deploy in tight corners (wastes energy fighting drag at low speed)
    tight_corners = []
    for pos in positions[:-1]:
        segment = track.get_segment_at_distance(pos)
        if segment.radius < 50:  # Very tight corner
            tight_corners.append(True)
        else:
            tight_corners.append(False)

    tight_corners = np.array(tight_corners)

    if np.sum(tight_corners) > 0 and np.sum(deploy_mask) > 0:
        deploy_in_tight_pct = np.sum(deploy_mask & tight_corners) / np.sum(deploy_mask) * 100

        report.add_test(
            "Avoid Deployment in Tight Corners",
            deploy_in_tight_pct < 20,  # Less than 20% in tight corners
            f"Only {deploy_in_tight_pct:.1f}% of deployment in tight corners (<50m radius)",
            f"{deploy_in_tight_pct:.1f}%"
        )
    else:
        report.add_test(
            "Avoid Deployment in Tight Corners",
            True,  # Pass if no tight corners
            "No tight corners or no deployment detected",
            "N/A"
        )


def validate_different_objectives(vehicle_model, track, ers_config, initial_soc, report: ValidationReport):
    """Test different optimization objectives"""

    print("\n" + "="*70)
    print("4. MULTI-OBJECTIVE VALIDATION")
    print("="*70)

    print("\nTesting different final SOC targets to validate flexibility...")

    # Scenario 1: Minimum SOC (maximize performance)
    print("\n  Scenario 1: Minimize final SOC (max performance)...")
    optimizer_min = GlobalOfflineOptimizer(vehicle_model, track, ds=10.0)  # Faster solve
    optimizer_min.setup_nlp(initial_soc=initial_soc, final_soc_min=0.2, energy_limit=4e6)

    try:
        traj_min = optimizer_min.solve()
        simulator = LapSimulator(vehicle_model, track, None)
        result_min = simulator.replay_trajectory(traj_min)

        # Scenario 2: Higher final SOC (save energy for next lap)
        print("\n  Scenario 2: Higher final SOC (energy conservation)...")
        optimizer_max = GlobalOfflineOptimizer(vehicle_model, track, ds=10.0)
        optimizer_max.setup_nlp(initial_soc=initial_soc, final_soc_min=0.6, energy_limit=4e6)
        traj_max = optimizer_max.solve()
        result_max = simulator.replay_trajectory(traj_max)

        # Test 12: Trade-off exists
        lap_time_diff = result_max.lap_time - result_min.lap_time
        soc_diff = result_max.final_soc - result_min.final_soc

        report.add_test(
            "Performance vs Energy Trade-off",
            lap_time_diff > 0 and soc_diff > 0,
            f"Higher SOC constraint → {lap_time_diff:.2f}s slower but {soc_diff*100:.1f}% more energy saved",
            f"+{lap_time_diff:.2f}s, +{soc_diff*100:.1f}% SOC"
        )

        # Test 13: Both strategies complete lap
        report.add_test(
            "All Strategies Complete Lap",
            result_min.completed and result_max.completed,
            "Both min-SOC and max-SOC strategies complete the lap",
            "Both completed"
        )

    except Exception as e:
        report.add_test(
            "Multi-Objective Optimization",
            False,
            f"Failed to test different objectives: {str(e)}",
            "Error"
        )


def plot_validation_results(offline_result, baseline_result, greedy_result, track, save_path='validation_analysis.png'):
    """Create comprehensive validation plots"""

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle('F1 ERS Optimization - Validation Analysis', fontsize=14, fontweight='bold')

    # Plot 1: Lap time comparison (all 3 strategies)
    ax = axes[0, 0]
    strategies = ['Offline Optimal', 'Smart Heuristic', 'Pure Greedy']
    lap_times = [offline_result.lap_time, baseline_result.lap_time, greedy_result.lap_time]
    colors = ['#4ECDC4', '#95E1D3', '#FFD93D']
    bars = ax.bar(strategies, lap_times, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Lap Time (s)', fontweight='bold')
    ax.set_title('Lap Time Comparison')
    ax.grid(axis='y', alpha=0.3)
    for bar, time in zip(bars, lap_times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height,
                f'{time:.2f}s', ha='center', va='bottom', fontweight='bold', fontsize=9)

    # Plot 2: Energy usage (all 3 strategies)
    ax = axes[0, 1]
    x = np.arange(3)
    width = 0.35
    deploy_vals = [offline_result.energy_deployed / 1e6,
                   baseline_result.energy_deployed / 1e6,
                   greedy_result.energy_deployed / 1e6]
    recover_vals = [offline_result.energy_recovered / 1e6,
                    baseline_result.energy_recovered / 1e6,
                    greedy_result.energy_recovered / 1e6]

    ax.bar(x - width/2, deploy_vals, width, label='Deployed', color='#FF6B6B', alpha=0.7)
    ax.bar(x + width/2, recover_vals, width, label='Recovered', color='#95E1D3', alpha=0.7)
    ax.set_ylabel('Energy (MJ)', fontweight='bold')
    ax.set_title('ERS Energy Usage')
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, fontsize=9)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Plot 3: Speed profile comparison
    ax = axes[1, 0]
    ax.plot(baseline_result.positions, baseline_result.velocities * 3.6,
            label='Baseline', color='#FF6B6B', alpha=0.7, linewidth=2)
    ax.plot(offline_result.positions, offline_result.velocities * 3.6,
            label='Offline Optimal', color='#4ECDC4', alpha=0.7, linewidth=2)
    ax.set_xlabel('Track Position (m)', fontweight='bold')
    ax.set_ylabel('Speed (km/h)', fontweight='bold')
    ax.set_title('Speed Profile Around Track')
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 4: ERS power profile
    ax = axes[1, 1]
    ax.plot(offline_result.positions[:-1], offline_result.P_ers_history / 1000,
            color='#4ECDC4', linewidth=2)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    ax.fill_between(offline_result.positions[:-1],
                     0, offline_result.P_ers_history / 1000,
                     where=(offline_result.P_ers_history > 0),
                     color='#FF6B6B', alpha=0.5, label='Deploy')
    ax.fill_between(offline_result.positions[:-1],
                     0, offline_result.P_ers_history / 1000,
                     where=(offline_result.P_ers_history < 0),
                     color='#95E1D3', alpha=0.5, label='Harvest')
    ax.set_xlabel('Track Position (m)', fontweight='bold')
    ax.set_ylabel('ERS Power (kW)', fontweight='bold')
    ax.set_title('ERS Power Strategy')
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 5: SOC evolution
    ax = axes[2, 0]
    ax.plot(baseline_result.positions, baseline_result.socs * 100,
            label='Baseline', color='#FF6B6B', alpha=0.7, linewidth=2)
    ax.plot(offline_result.positions, offline_result.socs * 100,
            label='Offline Optimal', color='#4ECDC4', alpha=0.7, linewidth=2)
    ax.set_xlabel('Track Position (m)', fontweight='bold')
    ax.set_ylabel('Battery SOC (%)', fontweight='bold')
    ax.set_title('Battery State of Charge')
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 6: Track radius (context)
    ax = axes[2, 1]
    positions_track = np.linspace(0, track.total_length, len(track.segments))
    radii = [seg.radius for seg in track.segments]
    ax.fill_between(positions_track, 0, radii, color='#F38181', alpha=0.3)
    ax.plot(positions_track, radii, color='#F38181', linewidth=2)
    ax.set_xlabel('Track Position (m)', fontweight='bold')
    ax.set_ylabel('Corner Radius (m)', fontweight='bold')
    ax.set_title('Track Geometry (Lower = Tighter Corner)')
    ax.set_ylim(0, max(radii) * 1.1)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n   ✓ Validation plots saved to {save_path}")

    return fig


def main():
    """Run complete validation suite"""

    print("="*70)
    print("  F1 ERS OPTIMIZATION - PROJECT VALIDATION")
    print("="*70)
    print("\nThis script validates that your optimal ERS project makes sense")
    print("by checking physics, performance, and strategy intelligence.")
    print("="*70)

    # Initialize report
    report = ValidationReport()

    # Configuration
    print("\n" + "="*70)
    print("SETUP")
    print("="*70)

    ers_config = ERSConfig()
    vehicle_config = VehicleConfig.for_monaco()

    print(f"\nLoading Monaco track...")
    track = F1TrackModel(2025, 'Monaco', 'Q', ds=10.0)  # Faster with 10m discretization

    try:
        track.load_from_fastf1('NOR')
        print("   ✓ Track loaded")
    except Exception as e:
        print(f"   ✗ Failed to load track: {e}")
        return False

    v_max = track.compute_speed_limits(vehicle_config)
    print(f"   ✓ Speed limits computed: {v_max.min()*3.6:.0f} - {v_max.max()*3.6:.0f} km/h")

    # Create models
    vehicle_model = VehicleDynamicsModel(vehicle_config, ers_config)
    print("   ✓ Vehicle model ready")

    initial_soc = 0.5

    # Run offline optimization
    print("\n" + "="*70)
    print("RUNNING OFFLINE OPTIMIZATION")
    print("="*70)

    offline_optimizer = GlobalOfflineOptimizer(vehicle_model, track, ds=10.0)
    offline_optimizer.setup_nlp(
        initial_soc=initial_soc,
        final_soc_min=0.3,
        energy_limit=4e6
    )

    print("\nSolving offline optimal trajectory...")
    offline_trajectory = offline_optimizer.solve()

    # Create simulator and replay
    simulator = LapSimulator(vehicle_model, track, None)
    offline_result = simulator.replay_trajectory(offline_trajectory)
    print(f"   ✓ Offline lap time: {offline_result.lap_time:.2f}s")

    # Run baselines
    print("\n" + "="*70)
    print("RUNNING BASELINE STRATEGIES")
    print("="*70)

    # Pure Greedy (literature baseline)
    print("\n1. Pure Greedy (Literature Baseline)...")
    greedy_controller = PureGreedyStrategy(track, vehicle_config, ers_config)
    greedy_simulator = LapSimulator(vehicle_model, track, greedy_controller)
    greedy_result = greedy_simulator.simulate_lap(initial_soc=initial_soc)
    print(f"   ✓ Pure Greedy lap time: {greedy_result.lap_time:.2f}s")

    # Smart Heuristic (your baseline)
    print("\n2. Smart Heuristic (Your Baseline)...")
    baseline_controller = SimpleRuleBasedStrategy(track, vehicle_config, ers_config)
    baseline_simulator = LapSimulator(vehicle_model, track, baseline_controller)
    baseline_result = baseline_simulator.simulate_lap(initial_soc=initial_soc)
    print(f"   ✓ Smart Heuristic lap time: {baseline_result.lap_time:.2f}s")

    # Run validation tests
    validate_physics(offline_result, ers_config, report)
    validate_performance(offline_result, greedy_result, report)  # Use greedy for main comparison
    validate_strategy_intelligence(offline_result, track, report)
    validate_different_objectives(vehicle_model, track, ers_config, initial_soc, report)

    # Additional test: Compare baselines
    print("\n" + "="*70)
    print("5. BASELINE COMPARISON")
    print("="*70)

    smart_improvement = greedy_result.lap_time - baseline_result.lap_time
    optimal_vs_smart = baseline_result.lap_time - offline_result.lap_time
    optimal_vs_greedy = greedy_result.lap_time - offline_result.lap_time
    optimal_vs_greedy_pct = (optimal_vs_greedy / greedy_result.lap_time) * 100

    print(f"\nLap Time Hierarchy:")
    print(f"  Offline Optimal:   {offline_result.lap_time:.3f}s (fastest)")
    print(f"  Smart Heuristic:   {baseline_result.lap_time:.3f}s (+{optimal_vs_smart:.3f}s)")
    print(f"  Pure Greedy:       {greedy_result.lap_time:.3f}s (+{optimal_vs_greedy:.3f}s)")

    report.add_test(
        "Smart Heuristic beats Pure Greedy",
        baseline_result.lap_time < greedy_result.lap_time,
        f"Smart: {baseline_result.lap_time:.2f}s vs Greedy: {greedy_result.lap_time:.2f}s (validates lookahead)",
        f"{smart_improvement:+.3f}s improvement"
    )

    report.add_test(
        "Optimal improvement vs Literature Baseline",
        1.0 < optimal_vs_greedy_pct < 10.0,  # Literature expectation: 2-7%
        f"Offline optimal is {optimal_vs_greedy_pct:.1f}% faster than pure greedy (literature expects 2-7%)",
        f"{optimal_vs_greedy_pct:.1f}%"
    )

    # Create plots
    print("\n" + "="*70)
    print("GENERATING VALIDATION PLOTS")
    print("="*70)
    plot_validation_results(offline_result, baseline_result, greedy_result, track)

    # Print final report
    all_passed = report.print_summary()

    if all_passed:
        print("\n✓ ✓ ✓  ALL VALIDATION TESTS PASSED  ✓ ✓ ✓")
        print("\nYour F1 ERS optimization project makes sense!")
        print("The approach is physically plausible, shows performance gains,")
        print("and demonstrates intelligent strategy decisions.")
    else:
        print("\n⚠  SOME VALIDATION TESTS FAILED  ⚠")
        print("\nReview the failed tests above to identify issues.")

    print("\n" + "="*70)

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
