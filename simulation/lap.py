import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import time

from models import VehicleDynamicsModel, F1TrackModel
from solvers import OptimalTrajectory

@dataclass
class LapResult:
    """Container for lap simulation results"""
    
    # Time series data
    times: np.ndarray          # Time stamps (s)
    positions: np.ndarray      # Track position (m)
    velocities: np.ndarray     # Velocity (m/s)
    socs: np.ndarray           # State of charge (0-1)
    
    # Control history
    P_ers_history: np.ndarray      # ERS power (W)
    throttle_history: np.ndarray   # Throttle (0-1)
    brake_history: np.ndarray      # Brake (0-1)
    
    # Summary statistics
    lap_time: float
    final_soc: float
    energy_deployed: float     # Total deployment (J)
    energy_recovered: float    # Total recovery (J)
    completed: bool
    
    # Reference tracking (if available)
    v_ref_history: Optional[np.ndarray] = None
    soc_ref_history: Optional[np.ndarray] = None
    
    # Solver performance
    solve_times: Optional[np.ndarray] = None
    solve_success: Optional[np.ndarray] = None
    
    @property
    def states(self) -> np.ndarray:
        """Combined states array for compatibility"""
        return np.column_stack([self.positions, self.velocities, self.socs])
    
    @property
    def controls(self) -> np.ndarray:
        """Combined controls array for compatibility"""
        return np.column_stack([
            self.P_ers_history, 
            self.throttle_history, 
            self.brake_history
        ])

class LapSimulator:
    """Simulate a complete lap with given controller"""
    
    def __init__(self,
                 vehicle_model: VehicleDynamicsModel,
                 track_model: F1TrackModel,
                 controller,
                 dt: float = 0.1):
        self.vehicle = vehicle_model
        self.track = track_model
        self.controller = controller
        self.dt = dt
        
        self.dynamics_func = vehicle_model.create_time_domain_dynamics()
        
    def simulate_lap(self,
                     initial_soc: float = 0.5,
                     initial_velocity: float = 30.0,
                     max_time: float = 200.0,
                     reference: Optional[OptimalTrajectory] = None
                     ) -> LapResult:
        """Simulate one complete lap
        
        Args:
            initial_soc: Starting battery SOC
            initial_velocity: Starting velocity (m/s)
            max_time: Maximum simulation time (s)
            reference: Optional reference trajectory for tracking
        
        Returns:
            LapResult with complete telemetry
        """
        # Initialize state: [position, velocity, soc]
        state = np.array([0.0, initial_velocity, initial_soc])
        
        # Storage lists
        times = [0.0]
        positions = [state[0]]
        velocities = [state[1]]
        socs = [state[2]]
        P_ers_history = []
        throttle_history = []
        brake_history = []
        v_ref_history = []
        soc_ref_history = []
        solve_times = []
        solve_success = []
        
        # Simulation loop
        t = 0.0
        lap_complete = False
        constraints = self.vehicle.get_constraints()
        
        while not lap_complete and t < max_time:
            position = state[0] % self.track.total_length
            segment = self.track.get_segment_at_distance(position)
            
            # Get reference if available
            if reference is not None:
                ref = reference.get_reference_at_distance(position)
                v_ref_history.append(ref['v_ref'])
                soc_ref_history.append(ref['soc_ref'])
            
            # Get control action
            start_solve = time.time()
            
            if hasattr(self.controller, 'solve_mpc_step'):
                control, info = self.controller.solve_mpc_step(state, position)
                solve_times.append(time.time() - start_solve)
                solve_success.append(info.get('success', False))
            else:
                control = self.controller.get_control(state, {
                    'gradient': segment.gradient,
                    'radius': segment.radius,
                })
                solve_times.append(0.0)
                solve_success.append(True)
            
            # Store control
            P_ers_history.append(control[0])
            throttle_history.append(control[1])
            brake_history.append(control[2])
            
            # Get track parameters
            track_params = np.array([segment.gradient, segment.radius])
            
            # Integrate dynamics (RK4)
            state = self._integrate_rk4(state, control, track_params)
            
            # Apply state constraints
            state[1] = np.clip(state[1], constraints['v_min'], constraints['v_max'])
            state[2] = np.clip(state[2], constraints['soc_min'], constraints['soc_max'])
            
            # Update time
            t += self.dt
            
            # Store state
            times.append(t)
            positions.append(state[0])
            velocities.append(state[1])
            socs.append(state[2])
            
            # Check lap completion
            if state[0] >= self.track.total_length:
                lap_complete = True
        
        # Convert to arrays
        times = np.array(times)
        positions = np.array(positions)
        velocities = np.array(velocities)
        socs = np.array(socs)
        P_ers_history = np.array(P_ers_history)
        throttle_history = np.array(throttle_history)
        brake_history = np.array(brake_history)
        
        # Compute energy
        energy_deployed = 0.0
        energy_recovered = 0.0
        for P_ers in P_ers_history:
            if P_ers > 0:
                energy_deployed += P_ers * self.dt
            else:
                energy_recovered += -P_ers * self.dt
        
        return LapResult(
            times=times,
            positions=positions,
            velocities=velocities,
            socs=socs,
            P_ers_history=P_ers_history,
            throttle_history=throttle_history,
            brake_history=brake_history,
            lap_time=t,
            final_soc=state[2],
            energy_deployed=energy_deployed,
            energy_recovered=energy_recovered,
            completed=lap_complete,
            v_ref_history=np.array(v_ref_history) if v_ref_history else None,
            soc_ref_history=np.array(soc_ref_history) if soc_ref_history else None,
            solve_times=np.array(solve_times),
            solve_success=np.array(solve_success),
        )
    
    def _integrate_rk4(self, state: np.ndarray, control: np.ndarray,
                       track_params: np.ndarray) -> np.ndarray:
        """RK4 integration step"""
        
        k1 = self.dynamics_func(state, control, track_params).full().flatten()
        k2 = self.dynamics_func(state + self.dt/2 * k1, control, track_params).full().flatten()
        k3 = self.dynamics_func(state + self.dt/2 * k2, control, track_params).full().flatten()
        k4 = self.dynamics_func(state + self.dt * k3, control, track_params).full().flatten()
        
        return state + self.dt / 6 * (k1 + 2*k2 + 2*k3 + k4)
    
    def replay_trajectory(self, trajectory: OptimalTrajectory) -> LapResult:
        """Replay an offline trajectory without simulation"""
        return LapResult(
            times=trajectory.t_opt,
            positions=trajectory.s,
            velocities=trajectory.v_opt,
            socs=trajectory.soc_opt,
            P_ers_history=trajectory.P_ers_opt,
            throttle_history=trajectory.throttle_opt,
            brake_history=trajectory.brake_opt,
            lap_time=trajectory.lap_time,
            final_soc=trajectory.soc_opt[-1],
            energy_deployed=trajectory.energy_deployed,
            energy_recovered=trajectory.energy_recovered,
            completed=True,
        )

def compare_strategies(vehicle_model,
                       track_model,
                       offline_trajectory: OptimalTrajectory,
                       controllers: Dict,
                       initial_soc: float = 0.5) -> Dict[str, LapResult]:
    """
    Compare multiple control strategies.
    
    Args:
        vehicle_model: VehicleDynamicsModel
        track_model: F1TrackModel
        offline_trajectory: Reference trajectory
        controllers: Dict of name -> controller
        initial_soc: Starting SOC
        
    Returns:
        Dict of name -> LapResult
    """
    print("\n" + "="*60)
    print("STRATEGY COMPARISON")
    print("="*60)
    
    results = {}
    
    # Offline optimal (replay)
    print("\n1. Evaluating offline optimal...")
    simulator = LapSimulator(vehicle_model, track_model, None)
    results['offline'] = simulator.replay_trajectory(offline_trajectory)
    print(f"   Lap time: {results['offline'].lap_time:.3f}s")
    
    # Each controller
    for i, (name, controller) in enumerate(controllers.items(), 2):
        print(f"\n{i}. Running {name}...")
        
        sim = LapSimulator(vehicle_model, track_model, controller)
        results[name] = sim.simulate_lap(
            initial_soc=initial_soc,
            reference=offline_trajectory
        )
        
        print(f"   Lap time: {results[name].lap_time:.3f}s")
        if results[name].solve_success is not None:
            success_rate = np.mean(results[name].solve_success) * 100
            print(f"   Success rate: {success_rate:.1f}%")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Strategy':<15} {'Lap Time':<12} {'Final SOC':<12} {'Status'}")
    print("-"*50)
    
    for name, result in results.items():
        status = "✓" if result.completed else "✗"
        print(f"{name:<15} {result.lap_time:<12.3f} {result.final_soc*100:<12.1f}% {status}")
    
    return results
