"""
Multi-Lap MPC vs Open-Loop Comparison Script

Demonstrates the value of MPC by simulating:
1. Offline NLP reference (computed once before race)
2. Open-loop execution (blindly follows reference, no adaptation)
3. MPC execution (re-optimizes each lap, adapts to degradation)

Shows that MPC handles model mismatch better than open-loop.
"""
import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from solvers import ForwardBackwardSolver, SpatialNLPSolver
from models import F1TrackModel, VehicleDynamicsModel
from config import ERSConfig, VehicleConfig, TireParameters
from controllers import ERSModelPredictiveController # TODO rename tracking to controllers
from simulation.multi_lap import (
    MultiLapSimulator,
    OpenLoopController,
    RaceResult
)


def main(args):
    """Main execution"""
    
    print("="*70)
    print("  MULTI-LAP MPC DEMONSTRATION")
    print("  Comparing Open-Loop vs MPC with Tire Degradation")
    print("="*70)
    
    # =========================================================================
    print("\n" + "="*70)
    print("SETUP")
    print("="*70)
    
    # Configuration
    ers_config = ERSConfig()
    vehicle_config = VehicleConfig()
    
    # Override tire parameters to show degradation effect
    tire_params = TireParameters()
    
    print(f"\nTrack: {args.track}")
    print(f"Number of laps: {args.n_laps}")
    print(f"Tire degradation: {args.tire_deg_rate*100:.1f}% per lap")
    print(f"Fuel consumption: {args.fuel_per_lap:.1f} kg/lap")
    
    # =========================================================================
    print("\n" + "="*70)
    print("LOAD TRACK")
    print("="*70)
    
    track = F1TrackModel(year=args.year, gp=args.track, ds=5.0)
    
    tumftm_path = Path(f'data/racelines/{args.track.lower()}.csv')
    
    if tumftm_path.exists():
        print(f"   Loading TUMFTM raceline: {tumftm_path}")
        track.load_from_tumftm_raceline(str(tumftm_path))
    else:
        print(f"   Loading from FastF1...")
        try:
            track.load_from_fastf1(driver=args.driver)
        except Exception as e:
            print(f"   ⚠ FastF1 failed: {e}")
            return
    
    print(f"   Track loaded: {track.total_length:.0f}m")
    
    # =========================================================================
    print("\n" + "="*70)
    print("CREATE MODELS")
    print("="*70)
    
    vehicle_model = VehicleDynamicsModel(vehicle_config, ers_config, tire_params)
    print("   ✓ Vehicle dynamics model ready")
    
    # Compute speed limits (with fresh tires)
    v_max = track.compute_speed_limits(vehicle_config)
    
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 1: COMPUTE OFFLINE REFERENCE (NLP)")
    print("="*70)
    print("\nThis is the 'perfect' strategy computed before the race")
    print("Assumes: Fresh tires, full fuel, perfect model")
    
    # Get velocity profile with ERS
    fb_solver = ForwardBackwardSolver(vehicle_model, track, use_ers_power=True)
    velocity_profile = fb_solver.solve()
    
    # Optimize ERS strategy
    nlp_solver = SpatialNLPSolver(vehicle_model, track, ers_config, ds=5.0)
    reference_trajectory = nlp_solver.solve(
        v_limit_profile=velocity_profile.v,
        initial_soc=args.initial_soc,
        final_soc_min=0.4,
        energy_limit=ers_config.deployment_limit_per_lap,
    )
    
    print(f"\n   Reference lap time: {reference_trajectory.lap_time:.3f}s")
    print(f"   (This is with fresh tires and full fuel)")
    
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 2: SIMULATE OPEN-LOOP (Follows Reference Blindly)")
    print("="*70)
    print("\nExecutes the pre-computed strategy without adaptation")
    print("As tires degrade → strategy becomes suboptimal")
    
    open_loop_controller = OpenLoopController(reference_trajectory)
    
    simulator = MultiLapSimulator(vehicle_model, track, dt=0.1)
    
    open_loop_result = simulator.simulate_race(
        controller=open_loop_controller,
        n_laps=args.n_laps,
        initial_soc=args.initial_soc,
        tire_deg_rate=args.tire_deg_rate,
        fuel_consumption_per_lap=args.fuel_per_lap,
        strategy_name="Open-Loop (No Adaptation)"
    )
    
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 3: SIMULATE MPC (Re-optimizes Each Lap)")
    print("="*70)
    print("\nMPC detects tire degradation and adapts strategy:")
    print("  - Early laps: Aggressive (tires still good)")
    print("  - Late laps: Conservative (tires degraded)")
    
    mpc_controller = ERSModelPredictiveController(
        vehicle_model=vehicle_model,
        track_model=track,
        ers_config=ers_config,
        horizon_segments=args.mpc_horizon,
        verbose=args.verbose
    )
    
    mpc_result = simulator.simulate_race(
        controller=mpc_controller,
        n_laps=args.n_laps,
        initial_soc=args.initial_soc,
        tire_deg_rate=args.tire_deg_rate,
        fuel_consumption_per_lap=args.fuel_per_lap,
        reference_trajectory=reference_trajectory,
        strategy_name="MPC (Adaptive)"
    )
    
    # =========================================================================
    print("\n" + "="*70)
    print("RESULTS COMPARISON")
    print("="*70)
    
    open_loop_result.print_summary()
    mpc_result.print_summary()
    
    # Comparison
    time_diff = open_loop_result.total_time - mpc_result.total_time
    improvement_pct = (time_diff / open_loop_result.total_time) * 100
    
    print(f"\n{'='*70}")
    print("FINAL COMPARISON")
    print(f"{'='*70}")
    print(f"Open-Loop Total: {open_loop_result.total_time:.3f}s")
    print(f"MPC Total:       {mpc_result.total_time:.3f}s")
    print(f"\nMPC Advantage:   {time_diff:.3f}s ({improvement_pct:.2f}%)")
    
    if time_diff > 0:
        print(f"\n✓ MPC is FASTER (adapts to degradation)")
    else:
        print(f"\n⚠ Open-loop is faster (degradation effect too small)")
    
    print(f"{'='*70}\n")
    
    # =========================================================================
    if args.plot:
        print("\n" + "="*70)
        print("GENERATING PLOTS")
        print("="*70)
        
        plot_race_comparison(open_loop_result, mpc_result, args.track)
    
    return reference_trajectory, open_loop_result, mpc_result


def plot_race_comparison(open_loop: RaceResult, mpc: RaceResult, track_name: str):
    """Plot comparison of strategies"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{track_name} - Multi-Lap Comparison: Open-Loop vs MPC', 
                 fontsize=14, fontweight='bold')
    
    laps = [r.lap_number for r in open_loop.lap_results]
    
    # Lap times
    ax = axes[0, 0]
    ax.plot(laps, [r.lap_time for r in open_loop.lap_results], 
            'o-', label='Open-Loop', linewidth=2, markersize=8)
    ax.plot(laps, [r.lap_time for r in mpc.lap_results], 
            's-', label='MPC', linewidth=2, markersize=8)
    ax.set_xlabel('Lap Number')
    ax.set_ylabel('Lap Time (s)')
    ax.set_title('Lap Time Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # SOC
    ax = axes[0, 1]
    ax.plot(laps, [r.final_soc * 100 for r in open_loop.lap_results],
            'o-', label='Open-Loop', linewidth=2, markersize=8)
    ax.plot(laps, [r.final_soc * 100 for r in mpc.lap_results],
            's-', label='MPC', linewidth=2, markersize=8)
    ax.set_xlabel('Lap Number')
    ax.set_ylabel('End-of-Lap SOC (%)')
    ax.set_title('Battery Management')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=20, color='r', linestyle='--', alpha=0.5, label='Min SOC')
    
    # Tire degradation
    ax = axes[1, 0]
    tire_deg = [r.tire_degradation * 100 for r in open_loop.lap_results]
    ax.plot(laps, tire_deg, 'o-', linewidth=2, markersize=8, color='brown')
    ax.set_xlabel('Lap Number')
    ax.set_ylabel('Tire Grip (%)')
    ax.set_title('Tire Degradation (Same for Both)')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=100, color='g', linestyle='--', alpha=0.5, label='Fresh')
    
    # Cumulative time difference
    ax = axes[1, 1]
    cumulative_diff = []
    cumsum_open = 0
    cumsum_mpc = 0
    
    for ol, mp in zip(open_loop.lap_results, mpc.lap_results):
        cumsum_open += ol.lap_time
        cumsum_mpc += mp.lap_time
        cumulative_diff.append(cumsum_open - cumsum_mpc)
    
    ax.plot(laps, cumulative_diff, 'o-', linewidth=2, markersize=8, color='purple')
    ax.set_xlabel('Lap Number')
    ax.set_ylabel('Time Difference (s)')
    ax.set_title('Cumulative MPC Advantage')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    # Add annotation
    final_diff = cumulative_diff[-1]
    color = 'green' if final_diff > 0 else 'red'
    ax.text(0.05, 0.95, f'Final: {final_diff:+.3f}s',
            transform=ax.transAxes, fontsize=12, fontweight='bold',
            verticalalignment='top', color=color,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save
    output_dir = Path('results') / track_name.lower() / 'mpc_comparison'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filepath = output_dir / 'multi_lap_comparison.png'
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"\n   ✓ Saved plot: {filepath}")
    
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Multi-Lap MPC Demonstration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Example:
            python run_mpc_multilap.py --track Monaco --n-laps 5 --plot
            python run_mpc_multilap.py --track Monza --n-laps 10 --tire-deg-rate 0.03
        """
    )
    
    parser.add_argument('--track', type=str, default='Monaco',
                        help='Track name')
    parser.add_argument('--year', type=int, default=2024,
                        help='Season year')
    parser.add_argument('--driver', type=str, default=None,
                        help='Driver code for FastF1')
    
    # Race parameters
    parser.add_argument('--n-laps', type=int, default=5,
                        help='Number of laps to simulate')
    parser.add_argument('--initial-soc', type=float, default=0.5,
                        help='Initial battery SOC')
    parser.add_argument('--tire-deg-rate', type=float, default=0.02,
                        help='Tire degradation per lap (0.02 = 2%%)')
    parser.add_argument('--fuel-per-lap', type=float, default=1.5,
                        help='Fuel consumption per lap (kg)')
    
    # MPC parameters
    parser.add_argument('--mpc-horizon', type=int, default=50,
                        help='MPC horizon in segments (~250m for 50)')
    
    # Output
    parser.add_argument('--plot', action='store_true', default=True,
                        help='Generate plots')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose MPC output')
    
    args = parser.parse_args()
    
    reference_trajectory, open_loop_result, mpc_result = main(args)
