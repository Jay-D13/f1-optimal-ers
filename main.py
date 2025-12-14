import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from solvers import (
    ForwardBackwardSolver,
    SpatialNLPSolver,
    OptimalTrajectory
)
from models import F1TrackModel, VehicleDynamicsModel
from config import ERSConfig, VehicleConfig, get_vehicle_config, get_ers_config
from utils import RunManager, export_results
from visualization import (
    visualize_track,
    plot_offline_solution,
    create_comparison_plot,
    visualize_lap_animated,
    plot_simple_results
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
    print(f"Vehicle: Cd={vehicle_config.cd}, Cl={vehicle_config.cl}")
    
    # =========================================================================
    print("\n" + "="*70)
    print("LOAD TRACK")
    print("="*70)
    
    track = F1TrackModel(year=args.year, gp=args.track, ds=5.0)
    
    # Try TUMFTM raceline first, fallback to FastF1
    tumftm_path = Path(f'data/racelines/{args.track.lower()}.csv')
    
    if tumftm_path.exists() and args.use_tumftm:
        print(f"   Loading TUMFTM raceline: {tumftm_path}")
        track.load_from_tumftm_raceline(str(tumftm_path))
    else:
        print(f"   Loading from FastF1 ({args.year} {args.track})...")
        try:
            track.load_from_fastf1(driver=args.driver)
        except Exception as e:
            print(f"   ⚠ FastF1 failed: {e}")
            print("   Please provide a TUMFTM raceline or check FastF1 cache.")
            return
    
    print(f"   Track loaded: {track.total_length:.0f}m, {len(track.segments)} segments")
    
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
    
    # Profile WITHOUT ERS (ICE power only)
    print("\n   Computing theoretical profile WITHOUT ERS...")
    fb_solver = ForwardBackwardSolver(vehicle_model, track, use_ers_power=False)
    velocity_profile_no_ers = fb_solver.solve()
    
    # Profile WITH ERS (ICE + ERS power)
    print("\n   Computing theoretical profile WITH ERS...")
    fb_solver.use_ers_power = True
    velocity_profile_with_ers = fb_solver.solve()
    
    print(f"\n   Results:")
    print(f"     No ERS:   {velocity_profile_no_ers.lap_time:.3f}s "
          f"(v: {velocity_profile_no_ers.v.min()*3.6:.0f}-{velocity_profile_no_ers.v.max()*3.6:.0f} km/h)")
    print(f"     With ERS: {velocity_profile_with_ers.lap_time:.3f}s "
          f"(v: {velocity_profile_with_ers.v.min()*3.6:.0f}-{velocity_profile_with_ers.v.max()*3.6:.0f} km/h)")
    print(f"     Theoretical improvement: {velocity_profile_no_ers.lap_time - velocity_profile_with_ers.lap_time:.3f}s")

    # =========================================================================
    print("\n" + "="*70)
    print("PHASE 2 - ERS OPTIMIZATION (Spatial NLP)")
    print("="*70)
    
    nlp_solver = SpatialNLPSolver(vehicle_model, track, ers_config, ds=5.0)
    
    # Use the WITH-ERS velocity limit for optimization
    optimal_trajectory = nlp_solver.solve(
        v_limit_profile=velocity_profile_with_ers.v,#v_max,
        initial_soc=args.initial_soc,
        final_soc_min=args.final_soc_min,
        energy_limit=ers_config.deployment_limit_per_lap,
    )
    
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
        Solver Type:            {args.solver}

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
    
    # =========================================================================
    if args.plot:
        print("\n" + "="*70)
        print("GENERATING PLOTS")
        print("="*70)
        
        # Track visualization
        print("\n Track Visualization...")
        fig_track = visualize_track( # TODO put driver name on plot and fix curves
            track, 
            track_name=args.track,
            driver_name=args.driver
        )
        run_manager.save_plot(fig_track, '01_track_analysis')
        plt.close(fig_track)
        
        # Offline solution (reusing existing function)
        print("\n Offline Solution...")
        fig_offline = plot_offline_solution(
            optimal_trajectory,
            title=f"{args.track} - Offline Optimal Solution"
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
            args.track
        )
        run_manager.save_plot(fig_comparison, '03_ers_comparison')
        plt.close(fig_comparison)
        
        # Simple results plot
        print("\n Simple Results...")
        fig_simple = plot_simple_results(
            optimal_trajectory, 
            velocity_profile_no_ers, 
            track, 
            args.track
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
                save_path=str(animation_path)
            )
            print(f"   ✓ Saved 05_lap_animation.gif")
            plt.close(fig_anim)
        
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
            
            python main.py --track Monaco --solver nlp --plot --save-animation
            TODO: for the presentation -> different start and end SOCs
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
    parser.add_argument('--solver', type=str, default='nlp',
                        choices=['nlp', 'ecms', 'pmp', 'mpcc'],
                        help='Solver to use (nlp is fully implemented)')
    parser.add_argument('--regulations', type=str, default='2025',
                        choices=['2025', '2026'],
                        help='Choose "2025" for the V6 Turbo Hybrid era rules (2014-2025) or "2026" for new upcoming engine specs')
    
    args = parser.parse_args()
    
    optimal_trajectory, run_manager = main(args)