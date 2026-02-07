import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import tomllib

sys.path.insert(0, str(Path(__file__).parent))

from solvers import (
    ForwardBackwardSolver,
    SpatialNLPSolver,
    MultiLapSpatialNLPSolver,
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


DEFAULTS = {
    "track": "Monaco",
    "year": 2024,
    "driver": None,
    "initial_soc": 0.5,
    "final_soc_min": 0.3,
    "ds": 5.0,
    "laps": 1,
    "per_lap_final_soc_min": None,
    "use_tumftm": False,
    "plot": True,
    "save_animation": False,
    "solver": "nlp",
    "collocation": "euler",
    "nlp_solver": "auto",
    "ipopt_linear_solver": "mumps",
    "ipopt_hessian": "limited-memory",
    "regulations": "2025",
}


PRESETS = {
    "quick": {
        "collocation": "euler",
        "ds": 10.0,
        "plot": False,
    },
    "balanced": {
        "collocation": "trapezoidal",
        "ds": 5.0,
    },
    "accurate": {
        "collocation": "hermite_simpson",
        "ds": 2.5,
    },
    "multi-lap": {
        "laps": 10,
        "initial_soc": 0.55,
        "final_soc_min": 0.45,
        "per_lap_final_soc_min": 0.40,
    },
}


CHOICES = {
    "solver": ("nlp",),
    "collocation": ("euler", "trapezoidal", "hermite_simpson"),
    "nlp_solver": ("auto", "ipopt", "fatrop", "sqpmethod"),
    "ipopt_hessian": ("limited-memory", "exact"),
    "regulations": ("2025", "2026"),
}


def _load_config(path: str | None) -> dict:
    if not path:
        return {}

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    suffix = config_path.suffix.lower()
    content = config_path.read_text()

    if suffix == ".toml":
        data = tomllib.loads(content)
    elif suffix == ".json":
        data = json.loads(content)
    else:
        raise ValueError("Unsupported config format. Use .toml or .json")

    if isinstance(data, dict) and isinstance(data.get("args"), dict):
        data = data["args"]

    if not isinstance(data, dict):
        raise ValueError("Config must be a JSON/TOML object")

    return data


def _normalize_overrides(source: str, overrides: dict) -> dict:
    unknown_keys = [key for key in overrides.keys() if key not in DEFAULTS]
    if unknown_keys:
        joined = ", ".join(sorted(unknown_keys))
        print(f"Warning: ignoring unknown {source} keys: {joined}")

    normalized = {key: value for key, value in overrides.items() if key in DEFAULTS}
    for key, value in normalized.items():
        if key in CHOICES and value is not None and value not in CHOICES[key]:
            allowed = ", ".join(CHOICES[key])
            raise ValueError(
                f"Invalid {source} value for '{key}': {value}. Allowed: {allowed}"
            )

    return normalized


def _resolve_args(cli_args, preset_overrides: dict, config_overrides: dict):
    resolved = DEFAULTS.copy()
    resolved.update(preset_overrides)
    resolved.update(config_overrides)

    for key, value in vars(cli_args).items():
        if value is not None:
            resolved[key] = value

    return argparse.Namespace(**resolved)


def _print_presets():
    print("Available presets:")
    for name, values in PRESETS.items():
        rendered = ", ".join(f"{k}={v}" for k, v in values.items())
        print(f"  {name}: {rendered}")


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

    if getattr(args, "preset", None):
        print(f"Preset: {args.preset}")
    if getattr(args, "config", None):
        print(f"Config file: {args.config}")
    
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
    parser = argparse.ArgumentParser(
        description='F1 ERS Optimal Control',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        python main.py --track Monaco --plot
        python main.py --track Monza --initial-soc 0.6 --plot --save-animation
        python main.py --track Spa --year 2023 --driver VER --plot
        python main.py --track Monza --laps 10 --final-soc-min 0.45

        # Compare collocation methods:
        python main.py --track Monaco --collocation euler --plot
        python main.py --track Monaco --collocation trapezoidal --plot
        python main.py --track Monaco --collocation hermite_simpson --plot

        # Use a preset or config
        python main.py --preset quick --track Monaco
        python main.py --config configs/monza.toml

        Collocation Methods:
        euler            - 1st order explicit Euler (fastest, least accurate)
        trapezoidal      - 2nd order implicit trapezoidal (good balance)
        hermite_simpson  - 4th order Hermite-Simpson (most accurate, slower)
        """
    )

    parser.add_argument('--config', type=str, default=None,
                        help='Path to JSON/TOML config file with default args')
    parser.add_argument('--preset', type=str, default=None,
                        choices=sorted(PRESETS.keys()),
                        help='Apply a preset argument bundle')
    parser.add_argument('--list-presets', action='store_true',
                        help='List available presets and exit')

    parser.add_argument('--track', type=str, default=None,
                        help='Track name (e.g., Monaco, Monza, Spa)')
    parser.add_argument('--year', type=int, default=None,
                        help='Season year for FastF1 data')
    parser.add_argument('--driver', type=str, default=None,
                        help='Driver code for FastF1 (e.g., VER, HAM)')
    parser.add_argument('--initial-soc', type=float, default=None,
                        help='Initial battery SOC (0.0-1.0)')
    parser.add_argument('--final-soc-min', type=float, default=None,
                        help='Minimum final SOC (0.0-1.0)')
    parser.add_argument('--ds', type=float, default=None,
                        help='Spatial discretization step in meters')
    parser.add_argument('--laps', type=int, default=None,
                        help='Number of consecutive laps in NLP horizon (1 = single lap)')
    parser.add_argument('--per-lap-final-soc-min', type=float, default=None,
                        help='Optional SOC floor enforced at the end of every lap')
    parser.add_argument('--use-tumftm', action=argparse.BooleanOptionalAction,
                        default=None,
                        help='Prefer TUMFTM raceline if available')
    parser.add_argument('--plot', action=argparse.BooleanOptionalAction,
                        default=None,
                        help='Generate plots')
    parser.add_argument('--save-animation', action=argparse.BooleanOptionalAction,
                        default=None,
                        help='Save lap animation (slow, optional)')
    parser.add_argument('--solver', type=str, default=None,
                        choices=['nlp'],
                        help='Solver to use (nlp is fully implemented)')
    parser.add_argument('--collocation', type=str, default=None,
                        choices=['euler', 'trapezoidal', 'hermite_simpson'],
                        help='Collocation method: euler (1st order), trapezoidal (2nd order), hermite_simpson (4th order)')
    parser.add_argument('--nlp-solver', type=str, default=None,
                        choices=['auto', 'ipopt', 'fatrop', 'sqpmethod'],
                        help='NLP backend solver (auto: fatrop on Apple Silicon, ipopt otherwise)')
    parser.add_argument('--ipopt-linear-solver', type=str, default=None,
                        help='Ipopt linear solver backend (e.g. mumps)')
    parser.add_argument('--ipopt-hessian', type=str, default=None,
                        choices=['limited-memory', 'exact'],
                        help='Ipopt Hessian approximation mode')
    parser.add_argument('--regulations', type=str, default=None,
                        choices=['2025', '2026'],
                        help='Choose "2025" for the V6 Turbo Hybrid era rules (2014-2025) or "2026" for new upcoming engine specs')

    args = parser.parse_args()

    if args.list_presets:
        _print_presets()
        raise SystemExit(0)

    try:
        config_overrides = _load_config(args.config)
        config_overrides = _normalize_overrides("config", config_overrides)
    except (ValueError, FileNotFoundError, json.JSONDecodeError) as exc:
        parser.error(str(exc))

    preset_overrides = PRESETS.get(args.preset, {}) if args.preset else {}
    try:
        preset_overrides = _normalize_overrides(f"preset '{args.preset}'", preset_overrides)
    except ValueError as exc:
        parser.error(str(exc))

    resolved_args = _resolve_args(args, preset_overrides, config_overrides)

    optimal_trajectory, run_manager = main(resolved_args)
