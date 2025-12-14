"""
Run all ERS strategies and compare their performance.

This script runs:
1. Offline NLP (optimal baseline)
2. Online MPC tracking
3. Simple Rule-Based
4. Greedy Baseline
5. Always Deploy Greedy

And generates comparison plots and statistics.
"""

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent))

from solvers import (
    ForwardBackwardSolver,
    SpatialNLPSolver,
    OptimalTrajectory,
)
from strategies import (
    OnlineMPController,
    SimpleRuleBasedStrategy,
    PureGreedyStrategy,
    AlwaysDeployGreedy,
)
from models import F1TrackModel, VehicleDynamicsModel
from config import ERSConfig, VehicleConfig
from utils import RunManager
from simulator import LapSimulator


def run_offline_optimal(vehicle_model, track, ers_config, args):
    """Run offline NLP optimization"""
    print("\n" + "="*70)
    print("OFFLINE OPTIMAL (NLP)")
    print("="*70)
    
    # Get velocity profile
    fb_solver = ForwardBackwardSolver(vehicle_model, track, use_ers_power=True)
    velocity_profile = fb_solver.solve()
    
    # Optimize ERS
    nlp_solver = SpatialNLPSolver(vehicle_model, track, ers_config, ds=5.0)
    optimal_trajectory = nlp_solver.solve(
        v_limit_profile=velocity_profile.v,
        initial_soc=args.initial_soc,
        final_soc_min=args.final_soc_min,
        energy_limit=ers_config.deployment_limit_per_lap,
    )
    
    print(f"  Lap Time: {optimal_trajectory.lap_time:.3f}s")
    print(f"  Status: {optimal_trajectory.solver_status}")
    
    return optimal_trajectory, velocity_profile


def run_mpc_strategy(vehicle_model, track, ers_config, optimal_trajectory, args):
    """Run online MPC controller"""
    print("\n" + "="*70)
    print("ONLINE MPC")
    print("="*70)
    
    # Create MPC controller
    mpc_controller = OnlineMPController(
        vehicle_model=vehicle_model,
        track_model=track,
        horizon_distance=200.0,
        dt=0.1
    )
    
    # Set reference from optimal
    mpc_controller.set_reference(optimal_trajectory)
    
    # Create simulator
    simulator = LapSimulator(
        vehicle_model=vehicle_model,
        track_model=track,
        dt=0.05
    )
    
    # Initial state
    initial_state = np.array([
        0.0,                    # position
        optimal_trajectory.v_opt[0],  # velocity
        args.initial_soc        # SOC
    ])
    
    # Run simulation
    start_time = time.time()
    results = simulator.simulate_lap(
        controller=mpc_controller,
        initial_state=initial_state,
        max_time=200.0
    )
    sim_time = time.time() - start_time
    
    # Print results
    if results['completed']:
        print(f"  Lap Time: {results['lap_time']:.3f}s")
        print(f"  Gap to Optimal: +{results['lap_time'] - optimal_trajectory.lap_time:.3f}s")
    else:
        print(f"  FAILED: {results.get('error', 'Unknown error')}")
    
    print(f"  Simulation Time: {sim_time:.2f}s")
    print(f"  MPC Stats: {mpc_controller.get_statistics()}")
    
    return results, mpc_controller


def run_simple_rule_strategy(vehicle_model, track, ers_config, vehicle_config, args):
    """Run simple rule-based strategy"""
    print("\n" + "="*70)
    print("SIMPLE RULE-BASED")
    print("="*70)
    
    # Create controller
    controller = SimpleRuleBasedStrategy(
        track_model=track,
        vehicle_config=vehicle_config,
        ers_config=ers_config
    )
    
    # Create simulator
    simulator = LapSimulator(
        vehicle_model=vehicle_model,
        track_model=track,
        dt=0.05
    )
    
    # Initial state
    initial_state = np.array([0.0, 50.0, args.initial_soc])
    
    # Run simulation
    start_time = time.time()
    results = simulator.simulate_lap(
        controller=controller,
        initial_state=initial_state,
        max_time=200.0
    )
    sim_time = time.time() - start_time
    
    # Print results
    if results['completed']:
        print(f"  Lap Time: {results['lap_time']:.3f}s")
    else:
        print(f"  FAILED")
    
    print(f"  Simulation Time: {sim_time:.2f}s")
    
    return results, controller


def run_greedy_strategy(vehicle_model, track, ers_config, vehicle_config, args):
    """Run greedy baseline strategy"""
    print("\n" + "="*70)
    print("GREEDY BASELINE")
    print("="*70)
    
    controller = PureGreedyStrategy(
        track_model=track,
        vehicle_config=vehicle_config,
        ers_config=ers_config
    )
    
    simulator = LapSimulator(vehicle_model, track, dt=0.05)
    initial_state = np.array([0.0, 50.0, args.initial_soc])
    
    start_time = time.time()
    results = simulator.simulate_lap(controller, initial_state, max_time=200.0)
    sim_time = time.time() - start_time
    
    if results['completed']:
        print(f"  Lap Time: {results['lap_time']:.3f}s")
    else:
        print(f"  FAILED")
    
    print(f"  Simulation Time: {sim_time:.2f}s")
    
    return results, controller


def run_always_deploy_strategy(vehicle_model, track, ers_config, vehicle_config, args):
    """Run always-deploy greedy strategy"""
    print("\n" + "="*70)
    print("ALWAYS DEPLOY GREEDY")
    print("="*70)
    
    controller = AlwaysDeployGreedy(
        track_model=track,
        vehicle_config=vehicle_config,
        ers_config=ers_config
    )
    
    simulator = LapSimulator(vehicle_model, track, dt=0.05)
    initial_state = np.array([0.0, 50.0, args.initial_soc])
    
    start_time = time.time()
    results = simulator.simulate_lap(controller, initial_state, max_time=200.0)
    sim_time = time.time() - start_time
    
    if results['completed']:
        print(f"  Lap Time: {results['lap_time']:.3f}s")
    else:
        print(f"  FAILED")
    
    print(f"  Simulation Time: {sim_time:.2f}s")
    
    return results, controller


def create_comparison_plots(all_results, run_manager):
    """Create comprehensive comparison plots"""
    print("\n" + "="*70)
    print("GENERATING COMPARISON PLOTS")
    print("="*70)
    
    # Extract completed results
    completed = {name: res for name, res in all_results.items() if res['results']['completed']}
    
    if len(completed) == 0:
        print("  No completed laps to compare!")
        return
    
    # 1. Lap time comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Strategy Comparison', fontsize=16, fontweight='bold')
    
    # Lap times
    ax = axes[0, 0]
    names = list(completed.keys())
    times = [completed[name]['results']['lap_time'] for name in names]
    colors = ['green', 'blue', 'orange', 'red', 'purple'][:len(names)]
    
    bars = ax.bar(names, times, color=colors, alpha=0.7)
    ax.set_ylabel('Lap Time (s)')
    ax.set_title('Lap Time Comparison')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, time in zip(bars, times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{time:.2f}s',
                ha='center', va='bottom', fontsize=9)
    
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Gap to optimal
    ax = axes[0, 1]
    if 'Offline Optimal' in completed:
        optimal_time = completed['Offline Optimal']['results']['lap_time']
        gaps = [times[i] - optimal_time for i in range(len(times))]
        bars = ax.bar(names, gaps, color=colors, alpha=0.7)
        ax.set_ylabel('Gap to Optimal (s)')
        ax.set_title('Performance Gap')
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.grid(axis='y', alpha=0.3)
        
        for bar, gap in zip(bars, gaps):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'+{gap:.2f}s' if gap > 0 else f'{gap:.2f}s',
                    ha='center', va='bottom' if gap > 0 else 'top', fontsize=9)
        
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Velocity profiles
    ax = axes[1, 0]
    for name, data in completed.items():
        states = data['results']['states']
        s = states[:, 0]
        v = states[:, 1] * 3.6  # Convert to km/h
        ax.plot(s, v, label=name, alpha=0.7)
    
    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Velocity (km/h)')
    ax.set_title('Velocity Profiles')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # SOC profiles
    ax = axes[1, 1]
    for name, data in completed.items():
        states = data['results']['states']
        s = states[:, 0]
        soc = states[:, 2] * 100  # Convert to %
        ax.plot(s, soc, label=name, alpha=0.7)
    
    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('SOC (%)')
    ax.set_title('Battery SOC')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    run_manager.save_plot(fig, 'strategy_comparison')
    plt.close(fig)
    
    print("  ✓ Comparison plots saved")


def print_summary_table(all_results, optimal_trajectory):
    """Print a summary table of all strategies"""
    print("\n" + "="*70)
    print("STRATEGY COMPARISON SUMMARY")
    print("="*70)
    
    print(f"\n{'Strategy':<20} {'Lap Time':<12} {'Gap':<10} {'Status':<10}")
    print("-" * 70)
    
    for name, data in all_results.items():
        results = data['results']
        
        if results['completed']:
            lap_time = results['lap_time']
            gap = lap_time - optimal_trajectory.lap_time
            status = "✓"
            print(f"{name:<20} {lap_time:>8.3f}s   {gap:>+6.3f}s   {status:<10}")
        else:
            print(f"{name:<20} {'FAILED':<12} {'-':<10} {'✗':<10}")
    
    print("="*70)


def main(args):
    """Main execution function"""
    
    print("="*70)
    print("  F1 ERS STRATEGY COMPARISON")
    print("="*70)
    
    # Setup
    run_manager = RunManager(args.track, base_dir="results/strategy_comparison")
    ers_config = ERSConfig()
    
    # Vehicle config
    track_configs = {
        'monaco': VehicleConfig.for_monaco,
        'monza': VehicleConfig.for_monza,
        'montreal': VehicleConfig.for_montreal,
        'spa': VehicleConfig.for_spa,
        'silverstone': VehicleConfig.for_silverstone,
    }
    
    vehicle_config = track_configs.get(
        args.track.lower(), 
        lambda: VehicleConfig()
    )()
    
    print(f"\nTrack: {args.track}")
    print(f"Running {len(args.strategies)} strategies")
    
    # Load track
    print("\n" + "="*70)
    print("LOADING TRACK")
    print("="*70)
    
    track = F1TrackModel(year=args.year, gp=args.track, ds=5.0)
    
    tumftm_path = Path(f'data/racelines/{args.track.lower()}.csv')
    if tumftm_path.exists() and args.use_tumftm:
        print(f"   Loading TUMFTM raceline: {tumftm_path}")
        track.load_from_tumftm_raceline(str(tumftm_path))
    else:
        print(f"   Loading from FastF1 ({args.year} {args.track})...")
        track.load_from_fastf1(driver=args.driver)
    
    print(f"   Track loaded: {track.total_length:.0f}m")
    
    # Create vehicle model
    vehicle_model = VehicleDynamicsModel(vehicle_config, ers_config)
    
    # Storage for all results
    all_results = {}
    
    # Run strategies
    if 'optimal' in args.strategies or 'all' in args.strategies:
        optimal_trajectory, velocity_profile = run_offline_optimal(
            vehicle_model, track, ers_config, args
        )
        all_results['Offline Optimal'] = {
            'results': {
                'completed': True,
                'lap_time': optimal_trajectory.lap_time,
                'states': np.column_stack([
                    optimal_trajectory.s,
                    optimal_trajectory.v_opt,
                    optimal_trajectory.soc_opt
                ]),
                'controls': np.column_stack([
                    optimal_trajectory.P_ers_opt,
                    optimal_trajectory.throttle_opt,
                    optimal_trajectory.brake_opt
                ])
            },
            'controller': None
        }
    else:
        # Need optimal for reference
        print("\nComputing offline optimal (needed for reference)...")
        optimal_trajectory, velocity_profile = run_offline_optimal(
            vehicle_model, track, ers_config, args
        )
    
    if 'mpc' in args.strategies or 'all' in args.strategies:
        results, controller = run_mpc_strategy(
            vehicle_model, track, ers_config, optimal_trajectory, args
        )
        all_results['Online MPC'] = {'results': results, 'controller': controller}
    
    if 'rule' in args.strategies or 'all' in args.strategies:
        results, controller = run_simple_rule_strategy(
            vehicle_model, track, ers_config, vehicle_config, args
        )
        all_results['Simple Rule'] = {'results': results, 'controller': controller}
    
    if 'greedy' in args.strategies or 'all' in args.strategies:
        results, controller = run_greedy_strategy(
            vehicle_model, track, ers_config, vehicle_config, args
        )
        all_results['Pure Greedy'] = {'results': results, 'controller': controller}
    
    if 'always' in args.strategies or 'all' in args.strategies:
        results, controller = run_always_deploy_strategy(
            vehicle_model, track, ers_config, vehicle_config, args
        )
        all_results['Always Deploy'] = {'results': results, 'controller': controller}
    
    # Generate comparison
    print_summary_table(all_results, optimal_trajectory)
    
    if args.plot:
        create_comparison_plots(all_results, run_manager)
    
    print("\n" + "="*70)
    print("  COMPLETE")
    print("="*70)
    print(f"\n📁 Results saved to: {run_manager.run_dir}")
    
    return all_results, optimal_trajectory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Compare all F1 ERS strategies',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all strategies
  python run_all_strategies.py --track Monaco --strategies all
  
  # Run specific strategies
  python run_all_strategies.py --track Monza --strategies optimal mpc rule
  
  # Quick comparison (no plots)
  python run_all_strategies.py --track Spa --strategies all --no-plot
        """
    )
    
    parser.add_argument('--track', type=str, default='Monaco',
                        help='Track name')
    parser.add_argument('--year', type=int, default=2024,
                        help='Season year')
    parser.add_argument('--driver', type=str, default=None,
                        help='Driver code for FastF1')
    parser.add_argument('--initial-soc', type=float, default=0.5,
                        help='Initial battery SOC')
    parser.add_argument('--final-soc-min', type=float, default=0.3,
                        help='Minimum final SOC')
    parser.add_argument('--use-tumftm', action='store_true',
                        help='Prefer TUMFTM raceline')
    parser.add_argument('--plot', dest='plot', action='store_true', default=True,
                        help='Generate plots')
    parser.add_argument('--no-plot', dest='plot', action='store_false',
                        help='Skip plot generation')
    parser.add_argument('--strategies', nargs='+', 
                        choices=['all', 'optimal', 'mpc', 'rule', 'greedy', 'always'],
                        default=['all'],
                        help='Strategies to run')
    
    args = parser.parse_args()
    
    all_results, optimal_trajectory = main(args)