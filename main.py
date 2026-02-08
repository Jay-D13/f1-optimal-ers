import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from solvers import (
    ForwardBackwardSolver,
    SpatialNLPSolver,
)
from models import F1TrackModel, VehicleDynamicsModel
from config import VehicleConfig, get_vehicle_config, get_ers_config
from simulation import simulate_multiple_laps
from strategies import OptimalTrackingStrategy
from utils import RunManager, export_results, export_multilap_results
from visualization import (
    visualize_track,
    plot_offline_solution,
    create_comparison_plot,
    visualize_lap_animated,
    plot_simple_results,
    plot_multilap_overview,
    plot_multilap_distance_heatmap,
    plot_multilap_speed_overlay,
)


def main(args):
    """Main execution function."""
    
    print("="*70)
    print("  F1 ERS OPTIMAL CONTROL")
    print("="*70)

    run_manager = RunManager(args.track, base_dir="results")
    
    # =========================================================================
    print("\n" + "="*70)
    print("CONFIGURATION")
    print("="*70)
    
    ers_config = get_ers_config(args.regulations)
    
    # Vehicle config (track-specific)
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
    vehicle_config = get_vehicle_config(args.regulations, base=vehicle_config)
    
    print(f"\nTrack: {args.track}")
    print(f"ERS Config: {ers_config.max_deployment_power/1000:.0f}kW deploy, "
          f"{ers_config.battery_usable_energy/1e6:.1f}MJ/lap limit")
    print(f"Vehicle: Cd={vehicle_config.cd:.2f}, Cl={vehicle_config.cl:.2f}")
    print(f"Collocation: {args.collocation}")
    
    # =========================================================================
    print("\n" + "="*70)
    print("LOAD TRACK")
    print("="*70)
    
    track = F1TrackModel(year=args.year, gp=args.track, ds=5.0)
    driver = args.driver #if args.driver else 'VER' # DU DU DU DUUU MAX VERSTAPPEN
    
    # Try TUMFTM raceline first, fallback to FastF1
    tumftm_path = Path(f'data/racelines/{args.track.lower()}.csv')
    
    if tumftm_path.exists() and args.use_tumftm:
        print(f"   Loading TUMFTM raceline: {tumftm_path}")
        track.load_from_tumftm_raceline(str(tumftm_path))
    else:
        print(f"   Loading from FastF1 ({args.year} {args.track})...")
        try:
            _, driver = track.load_from_fastf1(driver=args.driver)
        except Exception as e:
            print(f"   ⚠ FastF1 failed: {e}")
            print("   Please provide a TUMFTM raceline or check FastF1 cache.")
            return
    
    print(f"   Track loaded: {track.total_length:.0f}m, {len(track.segments)} segments")
    print(f"   Driver for telemetry: {driver}")
    
    v_max = track.compute_speed_limits(vehicle_config)

    # =========================================================================
    print("\n" + "="*70)
    print("CREATE MODELS")
    print("="*70)
    
    vehicle_model = VehicleDynamicsModel(vehicle_config, ers_config)
    print("   ✓ Vehicle dynamics model ready")
    

    # =========================================================================
    print("\n" + "="*70)
    print("PHASE 1 - VELOCITY PROFILE (Forward-Backward)")
    print("="*70)
    
    # Enable Flying Lap! -> Flying lap means it's a continuous lap without start from startline
    USE_FLYING_LAP = True # TODO make argument
    
    print(f"\n   Computing theoretical profile WITHOUT ERS (Flying: {USE_FLYING_LAP})...")
    fb_solver = ForwardBackwardSolver(vehicle_model, track, use_ers_power=False)
    velocity_profile_no_ers = fb_solver.solve(flying_lap=USE_FLYING_LAP)
    
    print(f"\n   Computing theoretical profile WITH ERS (Flying: {USE_FLYING_LAP})...")
    fb_solver.use_ers_power = True
    velocity_profile_with_ers = fb_solver.solve(flying_lap=USE_FLYING_LAP)
    
    print(f"\n   Results:")
    print(f"     No ERS:   {velocity_profile_no_ers.lap_time:.3f}s "
          f"(v: {velocity_profile_no_ers.v.min()*3.6:.0f}-{velocity_profile_no_ers.v.max()*3.6:.0f} km/h)")
    print(f"     With ERS: {velocity_profile_with_ers.lap_time:.3f}s "
          f"(v: {velocity_profile_with_ers.v.min()*3.6:.0f}-{velocity_profile_with_ers.v.max()*3.6:.0f} km/h)")
    print(f"     Theoretical improvement: {velocity_profile_no_ers.lap_time - velocity_profile_with_ers.lap_time:.3f}s")

    # =========================================================================
    print("\n" + "="*70)
    print(f"PHASE 2 - ERS OPTIMIZATION (Spatial NLP - {args.collocation.upper()})")
    print("="*70)
    
    nlp_solver = SpatialNLPSolver(
        vehicle_model, 
        track, 
        ers_config, 
        ds=5.0,
        collocation_method=args.collocation
    )
    
    # Use the WITH-ERS velocity limit for optimization
    optimal_trajectory = nlp_solver.solve(
        v_limit_profile=velocity_profile_with_ers.v,
        initial_soc=args.initial_soc,
        final_soc_min=args.final_soc_min,
        is_flying_lap=USE_FLYING_LAP
    )

    # =========================================================================
    multi_lap_result = None
    if args.multi_lap > 1:
        print("\n" + "="*70)
        print(f"PHASE 3 - MULTI-LAP DIAGNOSTICS ({args.multi_lap} laps)")
        print("="*70)
        strategy = OptimalTrackingStrategy(
            vehicle_config,
            ers_config,
            track,
            reference_profile=optimal_trajectory,
        )
        initial_velocity_multilap = float(optimal_trajectory.v_opt[0])
        multi_lap_result = simulate_multiple_laps(
            vehicle_model=vehicle_model,
            track_model=track,
            controller=strategy,
            num_laps=args.multi_lap,
            initial_soc=args.initial_soc,
            initial_velocity=initial_velocity_multilap,
            reference=optimal_trajectory,
            dt=0.1,
        )
        print(f"   Completed laps: {multi_lap_result.completed_laps}/{args.multi_lap}")
        lap_times = [f"{s['lap_time']:.3f}s" for s in multi_lap_result.lap_summaries]
        print(f"   Lap times: {', '.join(lap_times)}")
        print(f"   Final SOC after multi-lap: {multi_lap_result.final_soc * 100:.2f}%")
    
    # =========================================================================
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    energy_stats = optimal_trajectory.compute_energy_stats()
    
    summary_text = f"""
        {'='*70}
        F1 ERS OPTIMIZATION RESULTS - {args.track.upper()}
        {'='*70}
        Regulations Year:      {args.regulations}
        Collocation Method:    {args.collocation}

        TRACK INFORMATION:
        Track:                 {args.track}
        Year of track data:    {args.year}
        Total Length:          {track.total_length:.0f} m
        Number of Segments:    {len(track.segments)}

        LAP TIME PERFORMANCE:
        Lap Time (No ERS):      {velocity_profile_no_ers.lap_time:.3f} s
        Lap Time (With ERS):    {velocity_profile_with_ers.lap_time:.3f} s
        Lap Time (Optimal):     {optimal_trajectory.lap_time:.3f} s
        
        Improvement vs No ERS:  {velocity_profile_no_ers.lap_time - optimal_trajectory.lap_time:.3f} s ({((velocity_profile_no_ers.lap_time - optimal_trajectory.lap_time) / velocity_profile_no_ers.lap_time * 100):.2f}%)
        Gap to Theoretical:     {optimal_trajectory.lap_time - velocity_profile_with_ers.lap_time:.3f} s

        SOLVER INFORMATION:
        Status:                 {optimal_trajectory.solver_status}
        Solve Time:             {optimal_trajectory.solve_time:.2f} s
        Solver Type:            {optimal_trajectory.solver_name}

        ENERGY MANAGEMENT:
        Initial SOC:            {energy_stats['initial_soc']*100:.1f}%
        Final SOC:              {energy_stats['final_soc']*100:.1f}%
        Energy Deployed:        {energy_stats['total_deployed_MJ']:.3f} MJ
        Energy Recovered:       {energy_stats['total_recovered_MJ']:.3f} MJ
        Net Energy Used:        {energy_stats['net_energy_MJ']:.3f} MJ
        Recovery Efficiency:    {(energy_stats['total_recovered_MJ'] / max(energy_stats['total_deployed_MJ'], 1e-6) * 100):.1f}%

        VELOCITY STATISTICS:
        No ERS Profile:         {velocity_profile_no_ers.v.min()*3.6:.0f} - {velocity_profile_no_ers.v.max()*3.6:.0f} km/h
        With ERS Profile:       {velocity_profile_with_ers.v.min()*3.6:.0f} - {velocity_profile_with_ers.v.max()*3.6:.0f} km/h
        Optimal Strategy:       {optimal_trajectory.v_opt.min()*3.6:.0f} - {optimal_trajectory.v_opt.max()*3.6:.0f} km/h (avg: {optimal_trajectory.v_opt.mean()*3.6:.0f} km/h)

        {'='*70}
    """

    if multi_lap_result is not None:
        lap_summary_lines = [
            (
                f"Lap {item['lap']}: {item['lap_time']:.3f}s | "
                f"SOC {item['soc_start']*100:.1f}% -> {item['soc_end']*100:.1f}% | "
                f"Net {item['net_energy_MJ']:.3f} MJ"
            )
            for item in multi_lap_result.lap_summaries
        ]
        summary_text += (
            "\nMULTI-LAP DIAGNOSTICS:\n"
            f"Requested Laps:         {args.multi_lap}\n"
            f"Completed Laps:         {multi_lap_result.completed_laps}\n"
            + "\n".join(lap_summary_lines)
            + "\n"
        )
    
    print(summary_text)
    run_manager.save_summary(summary_text)
    
    # =========================================================================
    print("\n" + "="*70)
    print("SAVING DATA")
    print("="*70)
    
    results_dict = export_results(
        optimal_trajectory, velocity_profile_no_ers, track, args
    )
    run_manager.save_json(results_dict, 'results_summary')
    
    # Save numpy arrays for detailed analysis
    run_manager.save_numpy(optimal_trajectory.s, 'distance')
    run_manager.save_numpy(optimal_trajectory.v_opt, 'velocity_optimal')
    run_manager.save_numpy(velocity_profile_no_ers.v, 'velocity_no_ers')
    run_manager.save_numpy(velocity_profile_with_ers.v, 'velocity_with_ers')
    run_manager.save_numpy(optimal_trajectory.soc_opt, 'soc_optimal')
    run_manager.save_numpy(optimal_trajectory.P_ers_opt, 'ers_power')
    run_manager.save_numpy(optimal_trajectory.throttle_opt, 'throttle')
    run_manager.save_numpy(optimal_trajectory.brake_opt, 'brake')

    if multi_lap_result is not None:
        run_manager.save_json(export_multilap_results(multi_lap_result), "multi_lap_summary")
        run_manager.save_numpy(multi_lap_result.times, "multi_lap_times")
        run_manager.save_numpy(multi_lap_result.lap_index_states, "multi_lap_index_states")
        run_manager.save_numpy(multi_lap_result.positions, "multi_lap_positions")
        run_manager.save_numpy(multi_lap_result.positions_absolute, "multi_lap_positions_absolute")
        run_manager.save_numpy(multi_lap_result.velocities, "multi_lap_velocities")
        run_manager.save_numpy(multi_lap_result.socs, "multi_lap_socs")
        run_manager.save_numpy(multi_lap_result.controls_time, "multi_lap_controls_time")
        run_manager.save_numpy(multi_lap_result.lap_index_controls, "multi_lap_index_controls")
        run_manager.save_numpy(multi_lap_result.P_ers_history, "multi_lap_ers_power")
    
    # =========================================================================
    if args.plot:
        print("\n" + "="*70)
        print("GENERATING PLOTS")
        print("="*70)
        
        # Track visualization
        print("\n Track Visualization...")
        fig_track = visualize_track( # TODO fix track curves scale
            track, 
            track_name=args.track,
            driver_name=driver
        )
        if fig_track is not None:
            run_manager.save_plot(fig_track, '01_track_analysis')
            plt.close(fig_track)
        
        # Offline solution (reusing existing function)
        print("\n Offline Solution...")
        fig_offline = plot_offline_solution(
            optimal_trajectory,
            title=f"{args.track} - Offline Optimal Solution ({args.collocation})",
            ers_config=ers_config,
        )
        run_manager.save_plot(fig_offline, '02_offline_solution')
        plt.close(fig_offline)
        
        # Comprehensive comparison (with/without ERS)
        print("\n ERS Comparison Analysis...")
        fig_comparison = create_comparison_plot(
            track,
            velocity_profile_no_ers.v,
            velocity_profile_with_ers.v,
            optimal_trajectory,
            args.track,
            ers_config=ers_config,
        )
        run_manager.save_plot(fig_comparison, '03_ers_comparison')
        plt.close(fig_comparison)
        
        # Simple results plot
        print("\n Simple Results...")
        fig_simple = plot_simple_results(
            optimal_trajectory, 
            velocity_profile_no_ers, 
            track, 
            args.track,
            ers_config=ers_config,
        )
        run_manager.save_plot(fig_simple, '04_simple_results')
        plt.close(fig_simple)
        
        # Animation (optional, can be slow)
        if args.save_animation:
            print("\n Creating Animation (this may take a while)...")
            results_dict_for_anim = {
                'times': optimal_trajectory.t_opt,
                'states': np.column_stack([
                    optimal_trajectory.s,
                    optimal_trajectory.v_opt,
                    optimal_trajectory.soc_opt
                ]),
                'controls': np.column_stack([
                    optimal_trajectory.P_ers_opt,
                    optimal_trajectory.throttle_opt,
                    optimal_trajectory.brake_opt
                ]),
                'lap_time': optimal_trajectory.lap_time,
                'completed': True
            }
            
            animation_path = run_manager.plots_dir / "05_lap_animation.gif"
            fig_anim, anim = visualize_lap_animated(
                track,
                results_dict_for_anim,
                strategy_name="Optimal ERS",
                save_path=str(animation_path),
                timing_mode=args.animation_timing_mode,
                trail_length_m=args.animation_trail_m,
                ers_config=ers_config,
            )
            if fig_anim is not None:
                print(f"   ✓ Saved 05_lap_animation.gif")
                plt.close(fig_anim)

        if multi_lap_result is not None:
            print("\n Multi-Lap Overview...")
            fig_multilap_overview = plot_multilap_overview(
                multi_lap_result,
                track_name=args.track,
                ers_config=ers_config,
            )
            run_manager.save_plot(fig_multilap_overview, "06_multilap_overview")
            plt.close(fig_multilap_overview)

            print("\n Multi-Lap Distance Heatmap...")
            fig_multilap_heatmap = plot_multilap_distance_heatmap(
                multi_lap_result,
                track_name=args.track,
                track_length_m=track.total_length,
                ds_m=track.ds,
            )
            run_manager.save_plot(fig_multilap_heatmap, "07_multilap_ers_heatmap")
            plt.close(fig_multilap_heatmap)

            print("\n Multi-Lap Speed Overlay...")
            fig_multilap_speed = plot_multilap_speed_overlay(
                multi_lap_result,
                track_name=args.track,
                track_length_m=track.total_length,
                ds_m=track.ds,
            )
            run_manager.save_plot(fig_multilap_speed, "08_multilap_speed_overlay")
            plt.close(fig_multilap_speed)
        
        print("\n✓ All plots saved!")
    
    print("\n" + "="*70)
    print("  COMPLETE")
    print("="*70)
    print(f"\n📁 All results saved to: {run_manager.run_dir}")
    
    return optimal_trajectory, run_manager


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='F1 ERS Optimal Control',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        python main.py --track Monaco --plot
        python main.py --track Monza --initial-soc 0.6 --plot --save-animation
        python main.py --track Spa --year 2023 --driver VER --plot
        python main.py --track Monaco --plot --multi-lap 5
        python main.py --track Monaco --plot --save-animation --animation-timing-mode physical --animation-trail-m 150
        
        # Compare collocation methods:
        python main.py --track Monaco --collocation euler --plot
        python main.py --track Monaco --collocation trapezoidal --plot
        python main.py --track Monaco --collocation hermite_simpson --plot

        Collocation Methods:
        euler            - 1st order explicit Euler (fastest, least accurate)
        trapezoidal      - 2nd order implicit trapezoidal (good balance)
        hermite_simpson  - 4th order Hermite-Simpson (most accurate, slower)
        """
    )
    
    parser.add_argument('--track', type=str, default='Monaco',
                        help='Track name (e.g., Monaco, Monza, Spa)')
    parser.add_argument('--year', type=int, default=2024,
                        help='Season year for FastF1 data')
    parser.add_argument('--driver', type=str, default=None,
                        help='Driver code for FastF1 (e.g., VER, HAM)')
    parser.add_argument('--initial-soc', type=float, default=0.5,
                        help='Initial battery SOC (0.0-1.0)')
    parser.add_argument('--final-soc-min', type=float, default=0.3,
                        help='Minimum final SOC (0.0-1.0)')
    parser.add_argument('--use-tumftm', action='store_true',
                        help='Prefer TUMFTM raceline if available')
    parser.add_argument('--plot', action='store_true', default=True,
                        help='Generate plots')
    parser.add_argument('--save-animation', action='store_true',
                        help='Save lap animation (slow, optional)')
    parser.add_argument('--animation-timing-mode', type=str, default='physical',
                        choices=['physical', 'smooth'],
                        help='Animation timing mode: physical (time-accurate) or smooth')
    parser.add_argument('--animation-trail-m', type=float, default=120.0,
                        help='Animated trail length in meters')
    parser.add_argument('--multi-lap', type=int, default=1,
                        help='Run multi-lap diagnostics using consecutive laps with SOC carry-over')
    parser.add_argument('--solver', type=str, default='nlp',
                        choices=['nlp'],
                        help='Solver to use (nlp is fully implemented)')
    parser.add_argument('--collocation', type=str, default='euler',
                        choices=['euler', 'trapezoidal', 'hermite_simpson'],
                        help='Collocation method: euler (1st order), trapezoidal (2nd order), hermite_simpson (4th order)')
    parser.add_argument('--regulations', type=str, default='2025',
                        choices=['2025', '2026'],
                        help='Choose "2025" for the V6 Turbo Hybrid era rules (2014-2025) or "2026" for new upcoming engine specs')
    
    args = parser.parse_args()
    
    optimal_trajectory, run_manager = main(args)
