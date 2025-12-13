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
    
    results = {
        'metadata': {
            'track': args.track,
            'year': args.year,
            'driver': args.driver,
            'timestamp': datetime.now().isoformat(),
            'solver': args.solver,
            'initial_soc': args.initial_soc,
            'final_soc_min': args.final_soc_min,
        },
        'track_info': {
            'total_length': float(track.total_length),
            'n_segments': len(track.segments),
            'ds': float(track.ds),
        },
        'performance': {
            'lap_time': float(optimal_trajectory.lap_time),
            'lap_time_no_ers': float(velocity_profile.lap_time),
            'time_improvement': float(velocity_profile.lap_time - optimal_trajectory.lap_time),
            'solver_status': optimal_trajectory.solver_status,
            'solve_time': float(optimal_trajectory.solve_time),
        },
        'energy': {
            'initial_soc': float(energy_stats['initial_soc']),
            'final_soc': float(energy_stats['final_soc']),
            'total_deployed_MJ': float(energy_stats['total_deployed_MJ']),
            'total_recovered_MJ': float(energy_stats['total_recovered_MJ']),
            'net_energy_MJ': float(energy_stats['net_energy_MJ']),
            'energy_efficiency': float(energy_stats['total_recovered_MJ'] / 
                                      max(energy_stats['total_deployed_MJ'], 1e-6)),
        },
        'velocity_stats': {
            'max_speed_kmh': float(optimal_trajectory.v_opt.max() * 3.6),
            'min_speed_kmh': float(optimal_trajectory.v_opt.min() * 3.6),
            'avg_speed_kmh': float(optimal_trajectory.v_opt.mean() * 3.6),
        },
    }
    
    return results