import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import ERSConfig, VehicleConfig
from models import F1TrackModel, VehicleDynamicsModel
from controllers import GlobalOfflineOptimizer, OnlineMPController, SimpleRuleBasedStrategy, PureGreedyStrategy, ApexVelocitySolver
from simulation import LapSimulator, compare_strategies
from utils import (
    plot_lap_results, 
    plot_strategy_comparison,
    plot_track_with_ers,
    plot_offline_solution,
    visualize_track,
    visualize_lap_animated,
    plot_results
)


# def main():
#     """Main execution function for hierarchical ERS optimization"""
    
#     print("="*70)
#     print("  F1 ERS OPTIMAL CONTROL - HIERARCHICAL APPROACH")
#     print("="*70)
#     print("\nThis implementation follows the literature approach:")
#     print("  1. Offline Global Optimization (Spatial-domain NLP)")
#     print("  2. Online MPC Tracking (Time-domain NMPC)")
#     print("="*70)
    
#     # =========================================================================
#     # STEP 1: Configuration
#     # =========================================================================
#     print("\n" + "="*70)
#     print("STEP 1: CONFIGURATION")
#     print("="*70)
    
#     ers_config = ERSConfig()
#     vehicle_config = VehicleConfig.for_monaco()
    
#     print(f"\nERS Configuration:")
#     print(f"  Max deployment power: {ers_config.max_deployment_power/1000:.0f} kW")
#     print(f"  Max recovery power:   {ers_config.max_recovery_power/1000:.0f} kW")
#     print(f"  Battery capacity:     {ers_config.battery_capacity/1e6:.1f} MJ")
#     print(f"  Usable energy/lap:    {ers_config.battery_usable_energy/1e6:.1f} MJ")
    
#     print(f"\nVehicle Configuration (Monaco):")
#     print(f"  Mass:       {vehicle_config.mass:.0f} kg")
#     print(f"  Cd:         {vehicle_config.cd:.2f}")
#     print(f"  Cl:         {vehicle_config.cl:.2f}")
#     print(f"  ICE Power:  {vehicle_config.max_ice_power/1000:.0f} kW")
    
#     # =========================================================================
#     # STEP 2: Load Track Data
#     # =========================================================================
#     print("\n" + "="*70)
#     print("STEP 2: TRACK DATA LOADING")
#     print("="*70)
    
#     track_name = 'Monaco'
#     year = 2025
    
#     print(f"\nLoading {year} {track_name} GP track data...")
    
#     track = F1TrackModel(year, track_name, 'Q', ds=5.0)
    
#     try:
#         track.load_from_fastf1('NOR')  
#         # track.load_from_fastf1('VER') # DU DU DU DUHUU MAX VERSTAPPEN
#         print("   ✓ Loaded real track data from FastF1")
#     except Exception as e:
#         print(f"   ⚠ FastF1 load failed: {e}")
#         print("   Make sure FastF1 is installed and data is cached.")
#         return

#     print("\n   Generating track visualization...")
#     visualize_track(track, track_name=track_name, driver_name='Norris', save_path=f'visualization/{track_name.lower()}_analysis.png')
#     # return 

#     # Compute speed limits with vehicle config
#     print("\nComputing aerodynamic speed limits...")
#     v_max = track.compute_speed_limits(vehicle_config)
#     print(f"   Speed range: {v_max.min()*3.6:.0f} - {v_max.max()*3.6:.0f} km/h")
    
#     # =========================================================================
#     # STEP 3: Create Models
#     # =========================================================================
#     print("\n" + "="*70)
#     print("STEP 3: MODEL CREATION")
#     print("="*70)
    
#     print("\nCreating vehicle dynamics model...")
#     vehicle_model = VehicleDynamicsModel(vehicle_config, ers_config)
#     print("   ✓ Vehicle dynamics model ready")
    
#     # Verify dynamics sign convention
#     print("\nVerifying battery dynamics sign convention...")
#     dynamics = vehicle_model.create_time_domain_dynamics()
    
#     # Test: Deployment should decrease SOC
#     test_state = np.array([0, 50, 0.5])  # s=0, v=50m/s, soc=50%
#     test_deploy = np.array([100000, 1.0, 0])  # 100kW deployment
#     test_harvest = np.array([-100000, 0, 0.5])  # 100kW harvest
    
#     deploy_result = dynamics(test_state, test_deploy, np.array([0, 1000])).full().flatten()
#     harvest_result = dynamics(test_state, test_harvest, np.array([0, 1000])).full().flatten()
    
#     print(f"   Deploy (100kW):  dsoc/dt = {deploy_result[2]:.6f} (should be < 0)")
#     print(f"   Harvest (100kW): dsoc/dt = {harvest_result[2]:.6f} (should be > 0)")
    
#     if deploy_result[2] >= 0 or harvest_result[2] <= 0:
#         print("   ⚠ WARNING: Battery dynamics may have incorrect sign!")
#     else:
#         print("   ✓ Battery dynamics sign convention verified")
    
#     # =========================================================================
#     # STEP 4: Offline Global Optimization
#     # =========================================================================
#     print("\n" + "="*70)
#     print("STEP 4: OFFLINE GLOBAL OPTIMIZATION (Phase 1)")
#     print("="*70)
    
#     print("\nThis phase solves the minimum-time problem:")
#     print("  minimize  T = ∫(1/v)ds  over the entire lap")
#     print("  subject to: vehicle dynamics, energy limits, track constraints")
    
#     initial_soc = 0.5  # Start at 50% SOC
    
#     offline_optimizer = GlobalOfflineOptimizer(
#         vehicle_model=vehicle_model,
#         track_model=track,
#         ds=5.0  # 5m spatial discretization
#     )
    
#     print("\nSetting up NLP...")
#     offline_optimizer.setup_nlp(
#         initial_soc=initial_soc,
#         final_soc_min=0.3,  # End with at least 30% SOC
#         energy_limit=4e6    # 4MJ regulatory limit
#     )
    
#     print("\nSolving global optimization (this may take 1-3 minutes)...")
#     optimal_trajectory = offline_optimizer.solve()
    
#     # Plot offline solution
#     print("\nGenerating offline solution plots...")
#     fig_offline = plot_offline_solution(
#         optimal_trajectory,
#         title=f"Offline Optimal Solution - {track_name}",
#         save_path="offline_solution.png"
#     )
    
#     # =========================================================================
#     # STEP 5: Online MPC Controller
#     # =========================================================================
#     print("\n" + "="*70)
#     print("STEP 5: ONLINE MPC CONTROLLER (Phase 2)")
#     print("="*70)
    
#     print("\nThis phase tracks the offline reference in real-time:")
#     print("  Cost: J = Σ[(v - v_ref)² + (soc - soc_ref)² + Δu²]")
#     print("  Horizon: ~200m look-ahead")
    
#     mpc_controller = OnlineMPController(
#         vehicle_model=vehicle_model,
#         track_model=track,
#         horizon_distance=200.0,  # 200m prediction horizon
#         dt=0.1                   # 100ms time step
#     )
#     mpc_controller.set_reference(optimal_trajectory)
#     print("   ✓ MPC controller initialized with offline reference")
    
#     # =========================================================================
#     # STEP 6: Baseline Strategy
#     # =========================================================================
#     print("\n" + "="*70)
#     print("STEP 6: BASELINE STRATEGY")
#     print("="*70)

#     # Pass the models to the baseline so it can do lookahead math
#     baseline_strategy = PureGreedyStrategy(track, vehicle_config, ers_config)
#     print("   ✓ Rule-based baseline strategy ready")
    
#     # =========================================================================
#     # STEP 7: Run Simulations
#     # =========================================================================
#     print("\n" + "="*70)
#     print("STEP 7: LAP SIMULATIONS")
#     print("="*70)
    
#     # Compare all strategies
#     results = compare_strategies(
#         vehicle_model=vehicle_model,
#         track_model=track,
#         offline_trajectory=optimal_trajectory,
#         mpc_controller=mpc_controller,
#         baseline_controller=baseline_strategy,
#         initial_soc=initial_soc
#     )
    
#     # =========================================================================
#     # STEP 8: Visualization
#     # =========================================================================
#     print("\n" + "="*70)
#     print("STEP 8: GENERATING VISUALIZATIONS")
#     print("="*70)
    
#     # Plot results
#     for name, result in results.items():
#         # Convert to dict format for plot_results
#         result_dict = {
#             'times': result.times,
#             'states': result.states,
#             'controls': result.controls,
#             'lap_time': result.lap_time,
#             'completed': result.completed,
#         }
#         fig = plot_results(result_dict, f"{name.upper()} Strategy - {track_name}")
#         fig.savefig(f'{name}_results.png', dpi=150, bbox_inches='tight')
#         print(f"   ✓ Saved {name}_results.png")
    
#     # Animation for MPC
#     if 'mpc' in results:
#         print("\n   Creating animation...")
#         result_dict = {
#             'times': results['mpc'].times,
#             'states': results['mpc'].states,
#         }
#         try:
#             fig_anim, anim = visualize_lap_animated(
#                 track, result_dict, "MPC", 'mpc_lap_animation.gif'
#             )
#         except Exception as e:
#             print(f"   Could not create animation: {e}")
    
#     # Individual result plots
#     for name, result in results.items():
#         fig = plot_lap_results(
#             result,
#             title=f"{name.upper()} Strategy Results - {track_name}",
#             save_path=f"{name}_results.png"
#         )
    
#     # Comparison plot
#     fig_comparison = plot_strategy_comparison(
#         results,
#         title=f"Strategy Comparison - {track_name}",
#         save_path="strategy_comparison.png"
#     )
    
#     # Track visualization with ERS
#     if results.get('mpc'):
#         fig_track = plot_track_with_ers(
#             track,
#             results['mpc'],
#             title=f"Track Analysis with MPC ERS Strategy - {track_name}",
#             save_path="track_ers_analysis.png"
#         )
    
#     # =========================================================================
#     # STEP 9: Summary
#     # =========================================================================
#     print("\n" + "="*70)
#     print("PROJECT COMPLETE - SUMMARY")
#     print("="*70)
    
#     print(f"\n{'Strategy':<15} {'Lap Time (s)':<15} {'Final SOC (%)':<15} {'Energy (MJ)':<15}")
#     print("-"*60)
    
#     for name, result in results.items():
#         net_energy = (result.energy_deployed - result.energy_recovered) / 1e6
#         print(f"{name:<15} {result.lap_time:<15.3f} {result.final_soc*100:<15.1f} {net_energy:<15.2f}")
    
#     print("\nKey Insights:")
#     if 'mpc' in results and 'baseline' in results:
#         improvement = results['baseline'].lap_time - results['mpc'].lap_time
#         print(f"  • MPC improvement over baseline: {improvement:.3f}s ({improvement/results['baseline'].lap_time*100:.2f}%)")
    
#     if 'offline' in results and 'mpc' in results:
#         tracking_gap = results['mpc'].lap_time - results['offline'].lap_time
#         print(f"  • MPC tracking gap from optimal: {tracking_gap:.3f}s")
    
#     print(f"\nOutput files saved:")
#     print("  • offline_solution.png    - Global optimal trajectory")
#     print("  • mpc_results.png         - MPC simulation results")
#     print("  • baseline_results.png    - Rule-based baseline results")
#     print("  • strategy_comparison.png - Side-by-side comparison")
#     print("  • track_ers_analysis.png  - ERS on track map")
    
#     # Show plots
#     plt.show()
    
#     print("\n" + "="*70)
#     print("  END OF F1 ERS OPTIMIZATION")
#     print("="*70)
    
#     return results


if __name__ == "__main__":
    track = F1TrackModel(2025, 'Monaco')
    track.load_from_fastf1()
    ers_config = ERSConfig()
    vehicle_config = VehicleConfig.for_monaco()
    vehicle = VehicleDynamicsModel(vehicle_config, ers_config)
    initial_soc = 0.5
    track_name = 'Monaco'

    # Phase 1 (NumPy - Fast)
    velocity_solver = ApexVelocitySolver(vehicle, track)
    phase1_profile = velocity_solver.solve() # Returns v_max array

    # Phase 2 (CasADi - Stable)
    optimizer = GlobalOfflineOptimizer(vehicle, track) # TODO still doesn't converge but hey gets results
    optimizer.setup_nlp(
        v_limit_profile=phase1_profile.v,
        initial_soc=0.5,
        energy_limit=4e6
    )
    optimal_trajectory = optimizer.solve()
    
    print("\n" + "="*70)
    print("STEP 5: ONLINE MPC CONTROLLER (Phase 2)")
    print("="*70)
    
    print("\nThis phase tracks the offline reference in real-time:")
    print("  Cost: J = Σ[(v - v_ref)² + (soc - soc_ref)² + Δu²]")
    print("  Horizon: ~200m look-ahead")
    
    mpc_controller = OnlineMPController( # TODO MPC still fails a lot (uses fallback A LOT) need to tune
        vehicle_model=vehicle,
        track_model=track,
        horizon_distance=200.0,  # 200m prediction horizon
        dt=0.1                   # 100ms time step
    )
    mpc_controller.set_reference(optimal_trajectory)
    print("   ✓ MPC controller initialized with offline reference")
    
    # =========================================================================
    # STEP 7: Run Simulations
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 7: LAP SIMULATIONS")
    print("="*70)
    
    # Compare all strategies
    results = compare_strategies(
        vehicle_model=vehicle,
        track_model=track,
        offline_trajectory=optimal_trajectory,
        mpc_controller=mpc_controller,
        baseline_controller=PureGreedyStrategy(track, vehicle_config, ers_config),
        initial_soc=initial_soc
    )
    
    # =========================================================================
    # STEP 8: Visualization
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 8: GENERATING VISUALIZATIONS")
    print("="*70)
    
    # Plot results
    for name, result in results.items():
        # Convert to dict format for plot_results
        result_dict = {
            'times': result.times,
            'states': result.states,
            'controls': result.controls,
            'lap_time': result.lap_time,
            'completed': result.completed,
        }
        fig = plot_results(result_dict, f"{name.upper()} Strategy - {track_name}")
        fig.savefig(f'{name}_results.png', dpi=150, bbox_inches='tight')
        print(f"   ✓ Saved {name}_results.png")
    
    # Animation for MPC
    if 'mpc' in results:
        print("\n   Creating animation...")
        result_dict = {
            'times': results['mpc'].times,
            'states': results['mpc'].states,
        }
        try:
            fig_anim, anim = visualize_lap_animated(
                track, result_dict, "MPC", 'mpc_lap_animation.gif'
            )
        except Exception as e:
            print(f"   Could not create animation: {e}")
    
    # Individual result plots
    for name, result in results.items():
        fig = plot_lap_results(
            result,
            title=f"{name.upper()} Strategy Results - {track_name}",
            save_path=f"{name}_results.png"
        )
    
    # Comparison plot
    fig_comparison = plot_strategy_comparison(
        results,
        title=f"Strategy Comparison - {track_name}",
        save_path="strategy_comparison.png"
    )
    
    # Track visualization with ERS
    if results.get('mpc'):
        fig_track = plot_track_with_ers(
            track,
            results['mpc'],
            title=f"Track Analysis with MPC ERS Strategy - {track_name}",
            save_path="track_ers_analysis.png"
        )
    
    # =========================================================================
    # STEP 9: Summary
    # =========================================================================
    print("\n" + "="*70)
    print("PROJECT COMPLETE - SUMMARY")
    print("="*70)
    
    print(f"\n{'Strategy':<15} {'Lap Time (s)':<15} {'Final SOC (%)':<15} {'Energy (MJ)':<15}")
    print("-"*60)
    
    for name, result in results.items():
        net_energy = (result.energy_deployed - result.energy_recovered) / 1e6
        print(f"{name:<15} {result.lap_time:<15.3f} {result.final_soc*100:<15.1f} {net_energy:<15.2f}")
    
    print("\nKey Insights:")
    if 'mpc' in results and 'baseline' in results:
        improvement = results['baseline'].lap_time - results['mpc'].lap_time
        print(f"  • MPC improvement over baseline: {improvement:.3f}s ({improvement/results['baseline'].lap_time*100:.2f}%)")
    
    if 'offline' in results and 'mpc' in results:
        tracking_gap = results['mpc'].lap_time - results['offline'].lap_time
        print(f"  • MPC tracking gap from optimal: {tracking_gap:.3f}s")
    
    print(f"\nOutput files saved:")
    print("  • offline_solution.png    - Global optimal trajectory")
    print("  • mpc_results.png         - MPC simulation results")
    print("  • baseline_results.png    - Rule-based baseline results")
    print("  • strategy_comparison.png - Side-by-side comparison")
    print("  • track_ers_analysis.png  - ERS on track map")
    
    # Show plots
    plt.show()
    
    print("\n" + "="*70)
    print("  END OF F1 ERS OPTIMIZATION")
    print("="*70)