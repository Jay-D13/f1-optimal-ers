"""
Comparison script: ECMS vs NLP for F1 ERS optimization

This script demonstrates how to:
1. Run the existing NLP solver (global optimal)
2. Run ECMS variants (local/instantaneous optimal)
3. Compare results and analyze the optimality gap
"""

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from solvers import (
    ForwardBackwardSolver,
    SpatialNLPSolver,
    ECMSSolver,
    AdaptiveECMSSolver,
    TelemetryECMSSolver,
    ECMSConfig,
    OptimalTrajectory,
)
from models import F1TrackModel, VehicleDynamicsModel
from config import ERSConfig, VehicleConfig
from utils import RunManager


def compare_solvers(args):
    """Compare ECMS variants against NLP optimal."""
    
    print("="*70)
    print("  ECMS vs NLP COMPARISON")
    print("="*70)
    
    # =========================================================================
    # SETUP
    # =========================================================================
    
    run_manager = RunManager(args.track, base_dir="results")
    
    ers_config = ERSConfig()
    
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
    
    # =========================================================================
    # LOAD TRACK
    # =========================================================================
    
    print("\n" + "="*70)
    print("LOADING TRACK")
    print("="*70)
    
    track = F1TrackModel(year=args.year, gp=args.track, ds=5.0)
    
    tumftm_path = Path(f'data/racelines/{args.track.lower()}.csv')
    
    if tumftm_path.exists():
        print(f"   Loading TUMFTM raceline: {tumftm_path}")
        track.load_from_tumftm_raceline(str(tumftm_path))
    else:
        print(f"   Loading from FastF1 ({args.year} {args.track})...")
        track.load_from_fastf1(driver=args.driver)
    
    print(f"   Track loaded: {track.total_length:.0f}m, {len(track.segments)} segments")
    
    # =========================================================================
    # CREATE VEHICLE MODEL
    # =========================================================================
    
    vehicle_model = VehicleDynamicsModel(vehicle_config, ers_config)
    
    # =========================================================================
    # PHASE 1: Velocity Profile
    # =========================================================================
    
    print("\n" + "="*70)
    print("PHASE 1: VELOCITY PROFILE")
    print("="*70)
    
    fb_solver = ForwardBackwardSolver(vehicle_model, track, use_ers_power=True)
    velocity_profile = fb_solver.solve()
    
    print(f"   Theoretical lap time (with ERS): {velocity_profile.lap_time:.3f}s")
    
    # =========================================================================
    # PHASE 2: Solve with multiple methods
    # =========================================================================
    
    results = {}
    
    # --- NLP (Reference) ---
    if args.run_nlp:
        print("\n" + "="*70)
        print("SOLVER: NLP (Reference)")
        print("="*70)
        
        nlp_solver = SpatialNLPSolver(vehicle_model, track, ers_config, ds=5.0)
        nlp_trajectory = nlp_solver.solve(
            v_limit_profile=velocity_profile.v,
            initial_soc=args.initial_soc,
            final_soc_min=args.final_soc_min,
            energy_limit=ers_config.deployment_limit_per_lap,
        )
        results['NLP'] = nlp_trajectory
    
    # --- Basic ECMS ---
    print("\n" + "="*70)
    print("SOLVER: ECMS (Basic)")
    print("="*70)
    
    ecms_config = ECMSConfig(
        lambda_base=args.lambda_base,
        use_adaptive_lambda=True,
        soc_target_profile=nlp_trajectory.soc_opt,
        K_p=3.0,
        K_i=0.5,
        # soc_target=args.initial_soc,
    )
    
    ecms_solver = ECMSSolver(vehicle_model, track, ers_config, ecms_config, ds=5.0)
    ecms_trajectory = ecms_solver.solve(
        v_limit_profile=velocity_profile.v,
        initial_soc=args.initial_soc,
        final_soc_min=args.final_soc_min,
        energy_limit=ers_config.deployment_limit_per_lap,
    )
    results['ECMS'] = ecms_trajectory
    
    # --- Adaptive ECMS (with lookahead) ---
    print("\n" + "="*70)
    print("SOLVER: A-ECMS (Adaptive with Lookahead)")
    print("="*70)
    
    aecms_config = ECMSConfig(
        lambda_base=args.lambda_base,
        use_adaptive_lambda=True,
        use_predictive=True,
        lookahead_distance=500.0,
        K_p=3.0,
        K_i=0.5,
        soc_target_profile=nlp_trajectory.soc_opt,
    )
    
    aecms_solver = AdaptiveECMSSolver(vehicle_model, track, ers_config, aecms_config, ds=5.0)
    aecms_trajectory = aecms_solver.solve(
        v_limit_profile=velocity_profile.v,
        initial_soc=args.initial_soc,
        final_soc_min=args.final_soc_min,
        energy_limit=ers_config.deployment_limit_per_lap,
    )
    results['A-ECMS'] = aecms_trajectory
    
    # --- Telemetry-calibrated ECMS (if NLP available) ---
    if args.run_nlp and 'NLP' in results:
        print("\n" + "="*70)
        print("SOLVER: T-ECMS (Calibrated from NLP)")
        print("="*70)
        
        tecms_solver = TelemetryECMSSolver(vehicle_model, track, ers_config, ecms_config, ds=5.0)
        tecms_solver.calibrate_from_nlp(results['NLP'])
        tecms_trajectory = tecms_solver.solve(
            v_limit_profile=velocity_profile.v,
            initial_soc=args.initial_soc,
            final_soc_min=args.final_soc_min,
            energy_limit=ers_config.deployment_limit_per_lap,
        )
        results['T-ECMS'] = tecms_trajectory
    
    # =========================================================================
    # RESULTS COMPARISON
    # =========================================================================
    
    print("\n" + "="*70)
    print("RESULTS COMPARISON")
    print("="*70)
    
    print(f"\n{'Solver':<12} {'Lap Time':<12} {'Gap to Best':<12} {'Final SOC':<12} {'Energy (MJ)':<12} {'Solve Time':<12}")
    print("-"*72)
    
    best_time = min(r.lap_time for r in results.values())
    
    for name, traj in sorted(results.items(), key=lambda x: x[1].lap_time):
        gap = traj.lap_time - best_time
        energy_stats = traj.compute_energy_stats()
        print(f"{name:<12} {traj.lap_time:<12.3f} {gap:+<12.3f} "
              f"{energy_stats['final_soc']*100:<12.1f}% "
              f"{energy_stats['total_deployed_MJ']:<12.3f} "
              f"{traj.solve_time:<12.3f}")
    
    # =========================================================================
    # PLOTTING
    # =========================================================================
    
    if args.plot:
        print("\n" + "="*70)
        print("GENERATING COMPARISON PLOTS")
        print("="*70)
        
        fig = create_comparison_plot(results, velocity_profile, track, args.track)
        run_manager.save_plot(fig, 'ecms_vs_nlp_comparison')
        plt.close(fig)
        
        print("   ✓ Plots saved")
    
    print("\n" + "="*70)
    print("  COMPARISON COMPLETE")
    print("="*70)
    
    return results


def create_comparison_plot(results: dict, 
                          velocity_profile,
                          track,
                          track_name: str) -> plt.Figure:
    """Create comprehensive comparison plot."""
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    
    colors = {
        'NLP': 'blue',
        'ECMS': 'green', 
        'A-ECMS': 'orange',
        'T-ECMS': 'red',
    }
    
    # Plot velocity limit
    s_ref = velocity_profile.s
    axes[0].plot(s_ref, velocity_profile.v * 3.6, 'k--', alpha=0.5, 
                 label='Grip Limit', linewidth=1)
    
    for name, traj in results.items():
        color = colors.get(name, 'gray')
        
        # Velocity
        axes[0].plot(traj.s, traj.v_opt * 3.6, color=color, 
                    label=f'{name} ({traj.lap_time:.2f}s)', linewidth=1.5)
        
        # SOC
        axes[1].plot(traj.s, traj.soc_opt * 100, color=color, 
                    label=name, linewidth=1.5)
        
        # ERS Power
        axes[2].plot(traj.s[:-1], traj.P_ers_opt / 1000, color=color,
                    label=name, linewidth=1.5, alpha=0.8)
        
        # Cumulative energy deployed
        energy_deployed = np.zeros(len(traj.P_ers_opt))
        for k in range(len(traj.P_ers_opt)):
            v_avg = max(traj.v_opt[k], 5.0)
            dt = traj.ds / v_avg
            if traj.P_ers_opt[k] > 0:
                energy_deployed[k] = traj.P_ers_opt[k] * dt
        cumulative_energy = np.cumsum(energy_deployed) / 1e6
        axes[3].plot(traj.s[:-1], cumulative_energy, color=color,
                    label=name, linewidth=1.5)
    
    # Add 4 MJ limit line
    axes[3].axhline(y=4.0, color='red', linestyle='--', alpha=0.5, label='4 MJ Limit')
    
    # Formatting
    axes[0].set_ylabel('Velocity (km/h)')
    axes[0].set_title(f'{track_name} - ECMS vs NLP Comparison')
    axes[0].legend(loc='upper right', fontsize=8)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_ylabel('SOC (%)')
    axes[1].axhline(y=20, color='red', linestyle='--', alpha=0.5)
    axes[1].axhline(y=90, color='red', linestyle='--', alpha=0.5)
    axes[1].legend(loc='upper right', fontsize=8)
    axes[1].grid(True, alpha=0.3)
    
    axes[2].set_ylabel('ERS Power (kW)')
    axes[2].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[2].axhline(y=120, color='red', linestyle='--', alpha=0.3)
    axes[2].axhline(y=-120, color='green', linestyle='--', alpha=0.3)
    axes[2].legend(loc='upper right', fontsize=8)
    axes[2].grid(True, alpha=0.3)
    
    axes[3].set_ylabel('Cumulative Energy (MJ)')
    axes[3].set_xlabel('Distance (m)')
    axes[3].legend(loc='upper left', fontsize=8)
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def lambda_sensitivity_analysis(args):
    """
    Analyze sensitivity of ECMS to λ parameter.
    
    Useful for understanding:
    - How to tune λ for different tracks
    - Trade-off between lap time and SOC management
    """
    
    print("="*70)
    print("  LAMBDA SENSITIVITY ANALYSIS")
    print("="*70)
    
    # Setup (abbreviated)
    ers_config = ERSConfig()
    vehicle_config = VehicleConfig()
    
    track = F1TrackModel(year=args.year, gp=args.track, ds=5.0)
    tumftm_path = Path(f'data/racelines/{args.track.lower()}.csv')
    if tumftm_path.exists():
        track.load_from_tumftm_raceline(str(tumftm_path))
    else:
        track.load_from_fastf1()
    
    vehicle_model = VehicleDynamicsModel(vehicle_config, ers_config)
    
    fb_solver = ForwardBackwardSolver(vehicle_model, track, use_ers_power=True)
    velocity_profile = fb_solver.solve()
    
    # Test range of λ values
    lambda_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.5, 10.0]
    results = []
    
    print(f"\n{'Lambda':<10} {'Lap Time':<12} {'Final SOC':<12} {'Energy (MJ)':<12}")
    print("-"*50)
    
    for lambda_val in lambda_values:
        ecms_config = ECMSConfig(
            lambda_base=lambda_val,
            use_adaptive_lambda=False,  # Pure λ test
        )
        
        solver = ECMSSolver(vehicle_model, track, ers_config, ecms_config, ds=5.0)
        solver.verbose = False
        
        traj = solver.solve(
            v_limit_profile=velocity_profile.v,
            initial_soc=args.initial_soc,
            final_soc_min=args.final_soc_min,
            energy_limit=ers_config.deployment_limit_per_lap,
        )
        
        energy_stats = traj.compute_energy_stats()
        results.append({
            'lambda': lambda_val,
            'lap_time': traj.lap_time,
            'final_soc': energy_stats['final_soc'],
            'energy_deployed': energy_stats['total_deployed_MJ'],
        })
        
        print(f"{lambda_val:<10.1f} {traj.lap_time:<12.3f} "
              f"{energy_stats['final_soc']*100:<12.1f}% "
              f"{energy_stats['total_deployed_MJ']:<12.3f}")
    
    # Plot sensitivity
    if args.plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        lambdas = [r['lambda'] for r in results]
        lap_times = [r['lap_time'] for r in results]
        final_socs = [r['final_soc'] * 100 for r in results]
        energies = [r['energy_deployed'] for r in results]
        
        ax1.plot(lambdas, lap_times, 'b-o', linewidth=2, markersize=8)
        ax1.set_xlabel('Equivalence Factor (λ)')
        ax1.set_ylabel('Lap Time (s)', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.grid(True, alpha=0.3)
        ax1.set_title('Lap Time vs λ')
        
        ax1_twin = ax1.twinx()
        ax1_twin.plot(lambdas, final_socs, 'r--s', linewidth=2, markersize=8)
        ax1_twin.set_ylabel('Final SOC (%)', color='red')
        ax1_twin.tick_params(axis='y', labelcolor='red')
        
        ax2.plot(energies, lap_times, 'g-o', linewidth=2, markersize=8)
        for i, (e, t, l) in enumerate(zip(energies, lap_times, lambdas)):
            ax2.annotate(f'λ={l}', (e, t), textcoords="offset points", 
                        xytext=(5, 5), fontsize=8)
        ax2.set_xlabel('Energy Deployed (MJ)')
        ax2.set_ylabel('Lap Time (s)')
        ax2.set_title('Pareto Front: Lap Time vs Energy')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'results/lambda_sensitivity_{args.track.lower()}.png', dpi=150)
        plt.close()
        
        print(f"\n   ✓ Sensitivity plot saved")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ECMS vs NLP Comparison')
    
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
    parser.add_argument('--lambda-base', type=float, default=2.5,
                        help='Base equivalence factor for ECMS')
    parser.add_argument('--run-nlp', action='store_true', default=True,
                        help='Also run NLP for comparison')
    parser.add_argument('--plot', action='store_true', default=True,
                        help='Generate plots')
    parser.add_argument('--sensitivity', action='store_true',
                        help='Run lambda sensitivity analysis')
    
    args = parser.parse_args()
    
    if args.sensitivity:
        lambda_sensitivity_analysis(args)
    else:
        compare_solvers(args)
