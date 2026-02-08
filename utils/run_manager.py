from pathlib import Path
from datetime import datetime
import json
from typing import Dict
import numpy as np
import matplotlib.pyplot as plt

from models import F1TrackModel
from solvers import OptimalTrajectory

class RunManager:
    """Manages results directory structure and file saving"""
    
    def __init__(self, track_name: str, base_dir: str = "results"):
        self.track_name = track_name
        self.base_dir = Path(base_dir)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create directory structure: results/track_name/YYYYMMDD_HHMMSS/
        self.run_dir = self.base_dir / track_name.lower() / self.timestamp
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        self.plots_dir = self.run_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)
        
        self.data_dir = self.run_dir / "data"
        self.data_dir.mkdir(exist_ok=True)
        
        print(f"\n📁 Results directory: {self.run_dir}")
    
    def save_plot(self, fig: plt.Figure, name: str):
        """Save a matplotlib figure"""
        path = self.plots_dir / f"{name}.png"
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"   ✓ Saved {path.name}")
        return path
    
    def save_json(self, data: Dict, name: str):
        """Save data as JSON"""
        path = self.data_dir / f"{name}.json"
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        with open(path, 'w') as f:
            json.dump(convert_numpy(data), f, indent=2)
        print(f"   ✓ Saved {path.name}")
        return path
    
    def save_numpy(self, array: np.ndarray, name: str):
        """Save numpy array"""
        path = self.data_dir / f"{name}.npy"
        np.save(path, array)
        print(f"   ✓ Saved {path.name}")
        return path
    
    def save_summary(self, summary: str):
        """Save text summary"""
        path = self.run_dir / "summary.txt"
        path.touch(exist_ok=True)
        with open(path, 'w') as f:
            f.write(summary)
        print(f"   ✓ Saved summary.txt")
        return path
    
def export_results(optimal_trajectory: OptimalTrajectory,
                                velocity_profile,
                                track: F1TrackModel,
                                args) -> Dict:

    energy_stats = optimal_trajectory.compute_energy_stats()
    n_laps = max(1, int(getattr(optimal_trajectory, "n_laps", 1)))
    baseline_total = float(velocity_profile.lap_time) * n_laps
    
    results = {
        'metadata': {
            'track': args.track,
            'year': args.year,
            'driver': args.driver,
            'timestamp': datetime.now().isoformat(),
            'solver': args.solver,
            'nlp_solver': getattr(args, 'nlp_solver', None),
            'ipopt_linear_solver': getattr(args, 'ipopt_linear_solver', None),
            'ipopt_hessian': getattr(args, 'ipopt_hessian', None),
            'initial_soc': args.initial_soc,
            'final_soc_min': args.final_soc_min,
            'n_laps': n_laps,
            'per_lap_final_soc_min': getattr(args, 'per_lap_final_soc_min', None),
            'ds': getattr(args, 'ds', None),
        },
        'track_info': {
            'total_length': float(track.total_length),
            'n_segments': len(track.segments),
            'ds': float(track.ds),
        },
        'performance': {
            'lap_time': float(optimal_trajectory.lap_time),
            'lap_time_no_ers': baseline_total,
            'time_improvement': float(baseline_total - optimal_trajectory.lap_time),
            'solver_status': optimal_trajectory.solver_status,
            'solve_time': float(optimal_trajectory.solve_time),
            'lap_times': (
                optimal_trajectory.lap_times
                if optimal_trajectory.lap_times is not None
                else np.array([optimal_trajectory.lap_time])
            ),
        },
        'energy': {
            'initial_soc': float(energy_stats['initial_soc']),
            'final_soc': float(energy_stats['final_soc']),
            'total_deployed_MJ': float(energy_stats['total_deployed_MJ']),
            'total_recovered_MJ': float(energy_stats['total_recovered_MJ']),
            'net_energy_MJ': float(energy_stats['net_energy_MJ']),
            'energy_efficiency': float(energy_stats['total_recovered_MJ'] / 
                                      max(energy_stats['total_deployed_MJ'], 1e-6)),
            'lap_deployed_MJ': energy_stats.get('lap_deployed_MJ'),
            'lap_recovered_MJ': energy_stats.get('lap_recovered_MJ'),
            'lap_start_soc': energy_stats.get('lap_start_soc'),
            'lap_end_soc': energy_stats.get('lap_end_soc'),
        },
        'velocity_stats': {
            'max_speed_kmh': float(optimal_trajectory.v_opt.max() * 3.6),
            'min_speed_kmh': float(optimal_trajectory.v_opt.min() * 3.6),
            'avg_speed_kmh': float(optimal_trajectory.v_opt.mean() * 3.6),
        },
    }
    
    return results
