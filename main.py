import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))

from solvers import (
    ForwardBackwardSolver,
    SpatialNLPSolver,
    MultiLapSpatialNLPSolver,
)
from models import F1TrackModel, VehicleDynamicsModel
from config import ERSConfig, VehicleConfig, get_vehicle_config, get_ers_config
from config.app_config import AppConfig, build_parser
from utils import RunManager, export_results
from visualization import (
    visualize_track,
    plot_offline_solution,
    create_comparison_plot,
    visualize_lap_animated,
    plot_simple_results
)


def _format_multi_lap_breakdown(trajectory) -> str:
    """Format per-lap metrics for text summary."""
    if trajectory.lap_times is None:
        return ""

    lines = ["        PER-LAP BREAKDOWN:"]
    for i, lap_time in enumerate(trajectory.lap_times):
        deployed = trajectory.lap_energy_deployed[i] / 1e6
        recovered = trajectory.lap_energy_recovered[i] / 1e6
        soc_start = trajectory.lap_start_soc[i] * 100
        soc_end = trajectory.lap_end_soc[i] * 100
        lines.append(
            f"        Lap {i + 1:<2}: {lap_time:>7.3f} s | SOC {soc_start:>5.1f}% -> "
            f"{soc_end:>5.1f}% | Deploy {deployed:>5.3f} MJ | Recover {recovered:>5.3f} MJ"
        )

    return "\n".join(lines)


def main(args):
    """Main execution function."""
    
    if args.laps < 1:
        raise ValueError("--laps must be >= 1")

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
    print(f"Spatial step ds: {args.ds} m")
    print(f"Laps in NLP horizon: {args.laps}")
    print(f"NLP Backend: {args.nlp_solver}")
    if args.nlp_solver == "auto":
        print("NLP Backend Auto: fatrop on Apple Silicon, ipopt otherwise")
    if args.nlp_solver == "ipopt":
        print(f"Ipopt linear solver: {args.ipopt_linear_solver}")
        print(f"Ipopt Hessian: {args.ipopt_hessian}")
    if args.per_lap_final_soc_min is not None:
        print(f"Per-lap SOC floor: {args.per_lap_final_soc_min:.2f}")
    
    # =========================================================================
    print("\n" + "="*70)
    print("LOAD TRACK")
    print("="*70)
    
    track = F1TrackModel(year=args.year, gp=args.track, ds=args.ds)
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

    if args.laps <= 1:
        nlp_solver = SpatialNLPSolver(
            vehicle_model,
            track,
            ers_config,
            ds=args.ds,
            collocation_method=args.collocation,
            nlp_solver=args.nlp_solver,
            ipopt_linear_solver=args.ipopt_linear_solver,
            ipopt_hessian_approximation=args.ipopt_hessian,
        )
        optimal_trajectory = nlp_solver.solve(
            v_limit_profile=velocity_profile_with_ers.v,
            initial_soc=args.initial_soc,
            final_soc_min=args.final_soc_min,
            is_flying_lap=USE_FLYING_LAP,
        )
    else:
        nlp_solver = MultiLapSpatialNLPSolver(
            vehicle_model,
            track,
            ers_config,
            ds=args.ds,
            collocation_method=args.collocation,
            nlp_solver=args.nlp_solver,
            ipopt_linear_solver=args.ipopt_linear_solver,
            ipopt_hessian_approximation=args.ipopt_hessian,
        )
        optimal_trajectory = nlp_solver.solve(
            v_limit_profile=velocity_profile_with_ers.v,
            n_laps=args.laps,
            initial_soc=args.initial_soc,
            final_soc_min=args.final_soc_min,
            is_flying_lap=USE_FLYING_LAP,
            per_lap_final_soc_min=args.per_lap_final_soc_min,
        )
    
    # =========================================================================
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    energy_stats = optimal_trajectory.compute_energy_stats()
    n_laps = max(1, int(getattr(optimal_trajectory, "n_laps", 1)))
    total_time_no_ers = velocity_profile_no_ers.lap_time * n_laps
    total_time_with_ers = velocity_profile_with_ers.lap_time * n_laps
    total_time_optimal = optimal_trajectory.lap_time
    improvement = total_time_no_ers - total_time_optimal
    improvement_pct = (improvement / total_time_no_ers * 100.0) if total_time_no_ers > 1e-9 else 0.0
    lap_breakdown = _format_multi_lap_breakdown(optimal_trajectory)
    
    summary_text = f"""
        {'='*70}
        F1 ERS OPTIMIZATION RESULTS - {args.track.upper()}
        {'='*70}
        Regulations Year:      {args.regulations}
        Collocation Method:    {args.collocation}
        NLP Horizon Laps:      {n_laps}

        TRACK INFORMATION:
        Track:                 {args.track}
        Year of track data:    {args.year}
        Total Length:          {track.total_length:.0f} m
        Number of Segments:    {len(track.segments)}

        LAP TIME PERFORMANCE:
        Total Time (No ERS):    {total_time_no_ers:.3f} s
        Total Time (With ERS):  {total_time_with_ers:.3f} s
        Total Time (Optimal):   {total_time_optimal:.3f} s
        Avg Lap (Optimal):      {total_time_optimal / n_laps:.3f} s
        
        Improvement vs No ERS:  {improvement:.3f} s ({improvement_pct:.2f}%)
        Gap to Theoretical:     {total_time_optimal - total_time_with_ers:.3f} s

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

{lap_breakdown if lap_breakdown else ""}
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
    if optimal_trajectory.lap_times is not None:
        run_manager.save_numpy(optimal_trajectory.lap_times, 'lap_times')
        run_manager.save_numpy(optimal_trajectory.lap_start_soc, 'lap_start_soc')
        run_manager.save_numpy(optimal_trajectory.lap_end_soc, 'lap_end_soc')
        run_manager.save_numpy(optimal_trajectory.lap_energy_deployed, 'lap_energy_deployed')
        run_manager.save_numpy(optimal_trajectory.lap_energy_recovered, 'lap_energy_recovered')
    
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
            title=f"{args.track} - Offline Optimal Solution ({args.collocation}, {n_laps} lap(s))"
        )
        run_manager.save_plot(fig_offline, '02_offline_solution')
        plt.close(fig_offline)

        if n_laps == 1:
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
                fig_anim, anim = visualize_lap_animated( # TODO fix speed slow down and speed up of ego car
                    track,
                    results_dict_for_anim,
                    strategy_name="Optimal ERS",
                    save_path=str(animation_path)
                )
                print(f"   ✓ Saved 05_lap_animation.gif")
                plt.close(fig_anim)
        else:
            print("\n Multi-lap run detected: skipping single-lap comparison and animation plots.")
        
        print("\n✓ All plots saved!")
    
    print("\n" + "="*70)
    print("  COMPLETE")
    print("="*70)
    print(f"\n📁 All results saved to: {run_manager.run_dir}")
    
    return optimal_trajectory, run_manager


if __name__ == "__main__":
    parser = build_parser()
    namespace = parser.parse_args()
    args_dict = vars(namespace)
    args_dict.pop("config", None)
    args = AppConfig(**args_dict)

    optimal_trajectory, run_manager = main(args)
