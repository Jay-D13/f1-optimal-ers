from config import ERSConfig, VehicleConfig
from models import F1TrackModel, VehicleDynamicsModel
from controllers import ERSOptimalController, SimpleRuleBasedStrategy
from simulation import LapSimulator
from visualization.results_viz import plot_results
from visualization.animation import visualize_lap_animated
import numpy as np
import matplotlib.pyplot as plt

def main():
    """Main execution function"""
    print("=" * 60)
    print("F1 ERS Optimal Control Project - Improved Robustness")
    print("=" * 60)
    
    # Initialize configurations
    ers_config = ERSConfig()
    vehicle_config = VehicleConfig()
    
    print("\n1. Loading track data...")
    track = F1TrackModel(2023, 'Monaco', 'Q')
    
    try:
        track.load_from_fastf1('VER')
        print("   ✓ Loaded real track data from FastF1")
        
        print("\n   Generating track visualization...")
        track.visualize_track('track_analysis.png')
        
    except Exception as e:
        print(f"   ⚠ Error: {e}")
        track.create_synthetic_track()
        print("   ✓ Created synthetic Monaco track")
    
    print(f"   Track length: {track.total_length:.0f}m")
    print(f"   Number of segments: {len(track.segments)}")
    
    print("\n=== TRACK DIAGNOSTICS ===")
    radii = [seg.radius for seg in track.segments]
    print(f"Min radius: {min(radii):.1f}m")
    print(f"Max radius: {max(radii):.1f}m")
    print(f"Avg radius: {np.mean(radii):.1f}m")
    corner_count = sum(1 for r in radii if r < 500)
    print(f"Corner segments: {corner_count}/{len(track.segments)}")
    print(f"Percentage corners: {100*corner_count/len(track.segments):.1f}%")

    # Monaco should have ~30-40% corner segments
    if corner_count < 20:
        print("⚠️  WARNING: Track has too few corners!")
        print("    FastF1 data processing may have failed")
    
    print("\n2. Creating vehicle dynamics model...")
    vehicle_model = VehicleDynamicsModel(vehicle_config, ers_config)
    print("   ✓ Vehicle model initialized")
    
    print("\n3. Setting up optimal controller...")
    try:
        mpc_controller = ERSOptimalController(vehicle_model, track, horizon_time=3.0)
        mpc_controller.setup_optimization()
        print("   ✓ MPC controller ready")
        mpc_available = True
    except Exception as e:
        print(f"   ⚠ MPC setup failed: {e}")
        print("   Will use rule-based strategy only")
        mpc_available = False
    
    print("\n4. Creating baseline strategy...")
    rule_based = SimpleRuleBasedStrategy(ers_config)
    print("   ✓ Rule-based strategy ready")
    
    print("\n5. Running simulations...")
    simulator = LapSimulator(vehicle_model, track, rule_based)
    
    # Simulate with rule-based strategy
    print("\n   Running rule-based strategy simulation...")
    baseline_results = simulator.simulate_lap(initial_soc=0.5)
    print(f"   ✓ Completed: Lap time = {baseline_results['lap_time']:.2f}s")
    print(f"                Final SOC = {baseline_results['final_soc']*100:.1f}%")
    
    # Simulate with MPC if available
    if mpc_available:
        print("\n   Running MPC strategy simulation...")
        mpc_simulator = LapSimulator(vehicle_model, track, mpc_controller)
        mpc_results = mpc_simulator.simulate_lap(initial_soc=0.5)
        print(f"   ✓ Completed: Lap time = {mpc_results['lap_time']:.2f}s")
        print(f"                Final SOC = {mpc_results['final_soc']*100:.1f}%")
        
        # Compare results
        print("\n" + "=" * 60)
        print("RESULTS COMPARISON")
        print("=" * 60)
        print(f"{'Strategy':<20} {'Lap Time (s)':<15} {'Final SOC (%)':<15} {'Status'}")
        print("-" * 60)
        print(f"{'Rule-Based':<20} {baseline_results['lap_time']:<15.2f} "
              f"{baseline_results['final_soc']*100:<15.1f} "
              f"{'✓' if baseline_results['completed'] else '✗'}")
        print(f"{'MPC':<20} {mpc_results['lap_time']:<15.2f} "
              f"{mpc_results['final_soc']*100:<15.1f} "
              f"{'✓' if mpc_results['completed'] else '✗'}")
        
        improvement = (baseline_results['lap_time'] - mpc_results['lap_time'])
        print(f"\nMPC Improvement: {improvement:.2f}s ({improvement/baseline_results['lap_time']*100:.1f}%)")
    
    # Plot results
    print("\n6. Generating plots...")
    fig_baseline = plot_results(baseline_results, "Rule-Based Strategy")
    
    if mpc_available:
        fig_mpc = plot_results(mpc_results, "MPC Strategy")
        
        # Animated visualization
        print("\n7. Creating animated visualization...")
        try:
            fig_anim, anim = visualize_lap_animated(
                track, mpc_results, "MPC", 'mpc_lap_animation.gif'
            )
            print("   ✓ Animation created")
        except Exception as e:
            print(f"   ⚠ Could not create animation: {e}")
    
    # Save plots
    try:
        fig_baseline.savefig('baseline_results.png', dpi=150, bbox_inches='tight')
        print("   ✓ Saved baseline_results.png")
        
        if mpc_available:
            fig_mpc.savefig('mpc_results.png', dpi=150, bbox_inches='tight')
            print("   ✓ Saved mpc_results.png")
    except:
        print("   ⚠ Could not save plots")
    
    plt.show()
    
    print("\n" + "=" * 60)
    print("PROJECT COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    # import fastf1

    # session = fastf1.get_session(2023, 'Monaco', 'Q')
    # session.load()

    # lap = session.laps.pick_fastest()
    # telemetry = lap.get_telemetry()

    # What's in the telemetry DataFrame:
    # print(telemetry.columns)
    # Output: ['Time', 'DriverAhead', 'DistanceToDriverAhead', 'Date', 
    #          'RPM', 'Speed', 'nGear', 'Throttle', 'Brake', 'DRS',
    #          'X', 'Y', 'Z', 'Status', 'Distance', ...]
    
    # x,y,z for track shape
    
    main()