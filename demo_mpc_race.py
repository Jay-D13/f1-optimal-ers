#!/usr/bin/env python3
"""
Demo: MPC Controller and Multi-Lap Race Simulation

This script demonstrates the full workflow:
1. Offline optimization (single lap)
2. MPC controller setup with co-state extraction
3. Single lap simulation with MPC
4. Multi-lap race with shrinking-horizon strategy

Usage:
    python demo_mpc_race.py --track Monaco --laps 10
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    VehicleConfig, ERSConfig, 
    get_vehicle_config, get_ers_config
)
from models import F1TrackModel, VehicleDynamicsModel
from solvers import (
    ForwardBackwardSolver, 
    SpatialNLPSolver,
    LapTimeMapGenerator,
    LapTimeMapConfig,
)
from controllers import ERSMPCController, MPCConfig
from simulation import (
    LapSimulator, 
    MultiLapRaceSimulator,
    RaceConfig,
    compare_strategies,
)


def run_offline_optimization(vehicle_model, track, ers_config):
    """Run offline optimization to get reference trajectory"""
    
    print("\n" + "="*60)
    print("PHASE 1: OFFLINE OPTIMIZATION")
    print("="*60)
    
    # Forward-backward for velocity limits
    print("\n   Computing grip-limited velocity profile...")
    fb_solver = ForwardBackwardSolver(vehicle_model, track, use_ers_power=True)
    velocity_profile = fb_solver.solve(flying_lap=True)
    
    print(f"   ✓ Theoretical lap time: {velocity_profile.lap_time:.3f}s")
    
    # Full NLP optimization
    print("\n   Running spatial NLP optimization...")
    nlp_solver = SpatialNLPSolver(vehicle_model, track, ers_config, ds=5.0)
    
    optimal_trajectory = nlp_solver.solve(
        v_limit_profile=velocity_profile.v,
        initial_soc=0.5,
        final_soc_min=0.3,
        is_flying_lap=True
    )
    
    print(f"   ✓ Optimal lap time: {optimal_trajectory.lap_time:.3f}s")
    print(f"   ✓ Solve time: {optimal_trajectory.solve_time:.2f}s")
    
    return optimal_trajectory, velocity_profile


def demo_mpc_single_lap(vehicle_model, track, ers_config, optimal_trajectory):
    """Demonstrate MPC controller on a single lap"""
    
    print("\n" + "="*60)
    print("PHASE 2: MPC SINGLE LAP SIMULATION")
    print("="*60)
    
    # Create MPC controller with better tracking weights
    mpc_config = MPCConfig(
        horizon_distance=200.0,   # Shorter horizon for speed
        horizon_segments=40,      # Fewer segments for speed
        w_laptime=1.0,
        w_velocity_track=0.1,     # Increased from 0.001
        w_soc_track=1.0,          # Increased from 0.01
        w_terminal_soc=5.0,       # Increased
        max_iter=50,              # Reduced for speed
        tol=1e-3,                 # Relaxed for speed
        verbose=False,
        debug=True,               # Enable debug output
    )
    
    mpc_controller = ERSMPCController(
        vehicle_model=vehicle_model,
        track_model=track,
        ers_config=ers_config,
        config=mpc_config,
    )
    
    # Set reference trajectory
    mpc_controller.set_reference(optimal_trajectory)
    
    # Create lap simulator with smaller time step
    simulator = LapSimulator(
        vehicle_model=vehicle_model,
        track_model=track,
        controller=mpc_controller,
        dt=0.1  # 100ms steps
    )
    
    # Simulate with fresh tires
    print("\n   Simulating lap with fresh tires...")
    print("   (Debug output every 50 steps)")
    
    result_fresh = simulator.simulate_lap(
        initial_soc=0.5,
        initial_velocity=optimal_trajectory.v_opt[0],
        reference=optimal_trajectory
    )
    
    print(f"\n   ✓ Fresh tires lap time: {result_fresh.lap_time:.3f}s")
    print(f"   ✓ Final SOC: {result_fresh.final_soc*100:.1f}%")
    print(f"   ✓ Reference lap time: {optimal_trajectory.lap_time:.3f}s")
    print(f"   ✓ Gap to reference: {result_fresh.lap_time - optimal_trajectory.lap_time:.3f}s")
    
    # Simulate with degraded tires
    print("\n   Simulating lap with 15% tire degradation...")
    mpc_controller.reset()
    
    result_degraded = simulator.simulate_lap(
        initial_soc=0.5,
        initial_velocity=optimal_trajectory.v_opt[0] * 0.95,
        reference=optimal_trajectory
    )
    
    print(f"\n   ✓ Degraded tires lap time: {result_degraded.lap_time:.3f}s")
    print(f"   ✓ Final SOC: {result_degraded.final_soc*100:.1f}%")
    
    # Controller statistics
    stats = mpc_controller.get_statistics()
    print(f"\n   MPC Statistics:")
    print(f"     Success rate: {stats['success_rate']:.1f}%")
    print(f"     Avg solve time: {stats['avg_solve_time']*1000:.1f}ms")
    
    return result_fresh, mpc_controller


def demo_multi_lap_race(vehicle_model, track, ers_config, optimal_trajectory, n_laps=10):
    """Demonstrate multi-lap race simulation"""
    
    print("\n" + "="*60)
    print(f"PHASE 3: MULTI-LAP RACE SIMULATION ({n_laps} laps)")
    print("="*60)
    
    # Create MPC controller
    mpc_config = MPCConfig(
        horizon_distance=200.0,  # Shorter for race (faster computation)
        horizon_segments=40,
        verbose=False,
    )
    
    mpc_controller = ERSMPCController(
        vehicle_model=vehicle_model,
        track_model=track,
        ers_config=ers_config,
        config=mpc_config,
    )
    mpc_controller.set_reference(optimal_trajectory)
    
    # Race configuration
    race_config = RaceConfig(
        total_laps=n_laps,
        initial_soc=0.5,
        initial_fuel_kg=100.0,  # Reduced for short race
        fuel_consumption_per_lap=2.0,
        tire_degradation_per_lap=0.01,
        reoptimize_every_n_laps=3,
        safety_car_probability=0.0,  # Disable for demo
    )
    
    # Create race simulator
    race_sim = MultiLapRaceSimulator(
        vehicle_model=vehicle_model,
        track_model=track,
        ers_config=ers_config,
        offline_trajectory=optimal_trajectory,
        controller=mpc_controller,
        race_config=race_config,
    )
    
    # Run quick race (using lap time map)
    print("\n   Running quick race simulation (lap time map)...")
    quick_result = race_sim.run_quick_race(n_laps)
    
    print(f"\n   Quick race results:")
    summary = quick_result.get_summary()
    print(f"     Total time: {summary['total_time']:.2f}s")
    print(f"     Best lap: {summary['best_lap_time']:.3f}s")
    print(f"     Avg lap: {summary['avg_lap_time']:.3f}s")
    
    return quick_result


def plot_results(optimal_trajectory, mpc_result, race_result, track_name):
    """Generate visualization plots"""
    
    print("\n" + "="*60)
    print("GENERATING PLOTS")
    print("="*60)
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle(f'F1 ERS MPC Demo - {track_name}', fontsize=14, fontweight='bold')
    
    # 1. Velocity comparison
    ax = axes[0, 0]
    ax.plot(optimal_trajectory.s / 1000, optimal_trajectory.v_opt * 3.6, 
            'b-', label='Offline Optimal', linewidth=2)
    ax.plot(mpc_result.positions / 1000, mpc_result.velocities * 3.6,
            'r--', label='MPC', linewidth=1.5, alpha=0.8)
    ax.set_xlabel('Distance [km]')
    ax.set_ylabel('Velocity [km/h]')
    ax.set_title('Velocity Profile Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. SOC comparison
    ax = axes[0, 1]
    ax.plot(optimal_trajectory.s / 1000, optimal_trajectory.soc_opt * 100,
            'b-', label='Offline Optimal', linewidth=2)
    ax.plot(mpc_result.positions / 1000, mpc_result.socs * 100,
            'r--', label='MPC', linewidth=1.5, alpha=0.8)
    ax.set_xlabel('Distance [km]')
    ax.set_ylabel('State of Charge [%]')
    ax.set_title('Battery SOC Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. ERS Power
    ax = axes[1, 0]
    ax.plot(optimal_trajectory.s[:-1] / 1000, optimal_trajectory.P_ers_opt / 1000,
            'b-', label='Offline Optimal', linewidth=1.5)
    ax.plot(mpc_result.positions[:-1] / 1000, mpc_result.P_ers_history / 1000,
            'r--', label='MPC', linewidth=1, alpha=0.8)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axhline(y=120, color='g', linestyle=':', label='Max Deploy')
    ax.axhline(y=-120, color='orange', linestyle=':', label='Max Harvest')
    ax.set_xlabel('Distance [km]')
    ax.set_ylabel('ERS Power [kW]')
    ax.set_title('ERS Power Profile')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # 4. Controls
    ax = axes[1, 1]
    ax.plot(mpc_result.positions[:-1] / 1000, mpc_result.throttle_history * 100,
            'g-', label='Throttle', linewidth=1)
    ax.plot(mpc_result.positions[:-1] / 1000, mpc_result.brake_history * 100,
            'r-', label='Brake', linewidth=1)
    ax.set_xlabel('Distance [km]')
    ax.set_ylabel('Control Input [%]')
    ax.set_title('Control Inputs (MPC)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 105])
    
    # 5. Race lap times
    ax = axes[2, 0]
    laps = np.arange(1, len(race_result.lap_times) + 1)
    ax.bar(laps - 0.2, race_result.lap_times, width=0.4, label='Lap Time', color='steelblue')
    ax.plot(laps, race_result.lap_times, 'ko-', markersize=4)
    ax.axhline(y=optimal_trajectory.lap_time, color='r', linestyle='--', 
               label=f'Offline Optimal ({optimal_trajectory.lap_time:.2f}s)')
    ax.set_xlabel('Lap')
    ax.set_ylabel('Lap Time [s]')
    ax.set_title('Race Lap Times')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 6. Race SOC evolution
    ax = axes[2, 1]
    ax.plot(laps, np.array(race_result.lap_soc_start) * 100, 'b-o', 
            label='SOC Start', markersize=5)
    ax.plot(laps, np.array(race_result.lap_soc_end) * 100, 'r-s', 
            label='SOC End', markersize=5)
    ax.fill_between(laps, 
                    np.array(race_result.lap_soc_end) * 100,
                    np.array(race_result.lap_soc_start) * 100,
                    alpha=0.3, label='Energy Used')
    ax.set_xlabel('Lap')
    ax.set_ylabel('State of Charge [%]')
    ax.set_title('Race Energy Management')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    output_path = Path('results') / 'mpc_demo_results.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   ✓ Saved plot to {output_path}")
    
    return fig


def main(args):
    """Main demo execution"""
    
    print("="*60)
    print("  F1 ERS MPC AND MULTI-LAP RACE DEMO")
    print("="*60)
    
    # Configuration
    ers_config = get_ers_config(args.regulations)
    vehicle_config = get_vehicle_config(args.regulations)
    
    print(f"\n   Track: {args.track}")
    print(f"   Regulations: {args.regulations}")
    print(f"   Race laps: {args.laps}")
    
    # Load track
    print("\n   Loading track...")
    track = F1TrackModel(year=2024, gp=args.track, ds=5.0)
    
    tumftm_path = Path(f'data/racelines/{args.track.lower()}.csv')
    if tumftm_path.exists():
        track.load_from_tumftm_raceline(str(tumftm_path))
    else:
        try:
            track.load_from_fastf1()
        except Exception as e:
            print(f"   ⚠ Could not load track: {e}")
            print("   Using synthetic track data...")
            # Create minimal track for demo
            _create_demo_track(track)
    
    # Compute speed limits
    track.compute_speed_limits(vehicle_config)
    
    # Create vehicle model
    vehicle_model = VehicleDynamicsModel(vehicle_config, ers_config)
    
    # Run phases
    optimal_trajectory, velocity_profile = run_offline_optimization(
        vehicle_model, track, ers_config
    )
    
    mpc_result, mpc_controller = demo_mpc_single_lap(
        vehicle_model, track, ers_config, optimal_trajectory
    )
    
    race_result = demo_multi_lap_race(
        vehicle_model, track, ers_config, optimal_trajectory, n_laps=args.laps
    )
    
    # Visualization
    if args.plot:
        fig = plot_results(optimal_trajectory, mpc_result, race_result, args.track)
        plt.show()
    
    print("\n" + "="*60)
    print("  DEMO COMPLETE")
    print("="*60)
    
    return optimal_trajectory, mpc_result, race_result


def _create_demo_track(track):
    """Create minimal synthetic track for demo when no data available"""
    from models.track import TrackSegment, TrackData
    import numpy as np
    
    # Simple oval-ish track: 5km
    total_length = 5000.0
    ds = 5.0
    n_points = int(total_length / ds)
    
    s = np.linspace(0, total_length, n_points)
    
    # Create varying radius (straights and corners)
    radius = np.ones(n_points) * 1000  # Default: straight
    
    # Add corners
    corner_positions = [500, 1500, 2500, 3500, 4500]
    corner_radii = [80, 120, 60, 150, 100]
    corner_widths = [200, 300, 150, 250, 200]
    
    for pos, r, w in zip(corner_positions, corner_radii, corner_widths):
        mask = np.abs(s - pos) < w / 2
        radius[mask] = r
    
    # Create segments
    track.segments = []
    for i in range(n_points):
        segment = TrackSegment(
            distance=s[i],
            length=ds,
            radius=radius[i],
            curvature=1.0 / radius[i],
            gradient=0.0,
            x=s[i],  # Simple linear track
            y=0.0,
            sector=min(int(s[i] / (total_length / 3)) + 1, 3),
        )
        track.segments.append(segment)
    
    track.total_length = total_length
    track.ds = ds
    
    # Create track data
    track.track_data = TrackData(
        s=s,
        ds=ds,
        n_points=n_points,
        total_length=total_length,
        radius=radius,
        curvature=1.0 / radius,
        gradient=np.zeros(n_points),
        x=s,
        y=np.zeros(n_points),
        v_max_corner=np.zeros(n_points),
        sector=np.array([seg.sector for seg in track.segments]),
        is_braking_zone=np.zeros(n_points, dtype=bool),
        is_acceleration_zone=np.zeros(n_points, dtype=bool),
    )
    
    print(f"   ✓ Created synthetic track: {total_length:.0f}m")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='F1 ERS MPC and Multi-Lap Race Demo'
    )
    
    parser.add_argument('--track', type=str, default='Monaco',
                        help='Track name')
    parser.add_argument('--regulations', type=str, default='2025',
                        choices=['2025', '2026'],
                        help='Regulation year')
    parser.add_argument('--laps', type=int, default=10,
                        help='Number of race laps')
    parser.add_argument('--plot', action='store_true', default=True,
                        help='Generate plots')
    
    args = parser.parse_args()
    
    main(args)