"""
Multi-lap race simulator with tire degradation and fuel effects

Key improvements:
1. MPC called less frequently (every N meters instead of every timestep)
2. Control held between MPC calls for efficiency
3. Better fallback handling
"""
import numpy as np
from typing import Dict, List
from dataclasses import dataclass

from models import VehicleDynamicsModel, F1TrackModel
from solvers import OptimalTrajectory


@dataclass
class SingleMultiLapResult:
    """Results for a single lap"""
    lap_number: int
    lap_time: float
    final_soc: float
    energy_deployed: float
    energy_recovered: float
    tire_degradation: float  # Grip factor at end of lap
    fuel_mass: float  # Mass at end of lap
    avg_velocity: float
    

@dataclass  
class RaceResult:
    """Results for entire race stint"""
    lap_results: List[SingleMultiLapResult]
    total_time: float
    strategy_name: str
    
    @property
    def avg_lap_time(self) -> float:
        return self.total_time / len(self.lap_results)
    
    def print_summary(self):
        """Print race summary"""
        print(f"\n{'='*60}")
        print(f"RACE RESULTS: {self.strategy_name}")
        print(f"{'='*60}")
        print(f"{'Lap':<6} {'Time':<8} {'SOC':<8} {'Tire':<8} {'Fuel':<8}")
        print(f"{'-'*60}")
        
        for lap in self.lap_results:
            print(f"{lap.lap_number:<6} "
                  f"{lap.lap_time:<8.3f} "
                  f"{lap.final_soc*100:<7.1f}% "
                  f"{lap.tire_degradation*100:<7.1f}% "
                  f"{lap.fuel_mass:<8.1f}")
        
        print(f"{'-'*60}")
        print(f"Total Time: {self.total_time:.3f}s")
        print(f"Avg Lap:    {self.avg_lap_time:.3f}s")
        print(f"{'='*60}\n")


class MultiLapSimulator:
    """
    Simulates multiple laps with tire degradation and fuel effects
    """
    
    def __init__(self,
                 vehicle_model: VehicleDynamicsModel,
                 track_model: F1TrackModel,
                 dt: float = 0.1,
                 mpc_call_interval: float = None):
        """
        Args:
            vehicle_model: Vehicle dynamics model
            track_model: Track model
            dt: Simulation timestep (s)
            mpc_call_interval: Distance between MPC calls (m). 
                              If None, calls every timestep (original behavior)
        """
        
        self.vehicle = vehicle_model
        self.track = track_model
        self.dt = dt
        self.mpc_call_interval = mpc_call_interval  # meters between MPC calls
        
        self.dynamics_func = vehicle_model.create_time_domain_dynamics()
        
    def simulate_race(self,
                      controller,
                      n_laps: int,
                      initial_soc: float = 0.5,
                      initial_velocity: float = 30.0,
                      tire_deg_rate: float = 0.02,  # 2% per lap
                      fuel_consumption_per_lap: float = 1.5,  # kg/lap
                      reference_trajectory: OptimalTrajectory = None,
                      strategy_name: str = "Unknown") -> RaceResult:
        """
        Simulate multi-lap race
        """
        
        print(f"\n{'='*60}")
        print(f"SIMULATING {n_laps}-LAP RACE: {strategy_name}")
        print(f"{'='*60}")
        
        # Set reference if controller supports it
        if hasattr(controller, 'set_reference') and reference_trajectory is not None:
            controller.set_reference(reference_trajectory)
        
        lap_results = []
        
        # Initial conditions
        tire_grip = 1.0  # Fresh tires
        fuel_mass = self.vehicle.vehicle.mass  # Start with full fuel
        soc = initial_soc
        
        total_time = 0.0
        
        for lap_num in range(1, n_laps + 1):
            print(f"\n  Lap {lap_num}/{n_laps} "
                  f"(Tire: {tire_grip*100:.1f}%, Fuel: {fuel_mass:.1f}kg)")
            
            # Simulate one lap
            lap_result = self._simulate_single_lap(
                controller=controller,
                lap_number=lap_num,
                initial_soc=soc,
                initial_velocity=initial_velocity,
                tire_degradation=tire_grip,
                fuel_mass=fuel_mass
            )
            
            # Update conditions for next lap
            soc = lap_result.final_soc
            tire_grip = max(0.70, tire_grip - tire_deg_rate)  # Min 70% grip
            fuel_mass = max(798, fuel_mass - fuel_consumption_per_lap)  # Min mass
            
            total_time += lap_result.lap_time
            lap_results.append(lap_result)
            
            print(f"    ✓ Lap time: {lap_result.lap_time:.3f}s, "
                  f"SOC: {lap_result.final_soc*100:.1f}%")
        
        return RaceResult(
            lap_results=lap_results,
            total_time=total_time,
            strategy_name=strategy_name
        )
    
    def _simulate_single_lap(self,
                            controller,
                            lap_number: int,
                            initial_soc: float,
                            initial_velocity: float,
                            tire_degradation: float,
                            fuel_mass: float) -> SingleMultiLapResult:
        """Simulate a single lap"""
        
        # State: [position, velocity, soc]
        state = np.array([0.0, initial_velocity, initial_soc])
        
        t = 0.0
        lap_complete = False
        max_time = 300.0  # Safety timeout
        
        energy_deployed = 0.0
        energy_recovered = 0.0
        
        constraints = self.vehicle.get_constraints()
        
        # For MPC: track when we last called
        is_mpc = hasattr(controller, 'solve_step')
        last_mpc_position = -1000.0  # Force first call
        current_control = np.array([0.0, 0.7, 0.0])  # Default: 70% throttle, no brake
        mpc_calls = 0
        
        while not lap_complete and t < max_time:
            position = state[0] % self.track.total_length
            velocity = state[1]
            soc = state[2]
            
            # Determine if we should call controller
            should_call = False
            if is_mpc:
                if self.mpc_call_interval is not None:
                    distance_since_mpc = position - last_mpc_position
                    if distance_since_mpc < 0:
                        distance_since_mpc += self.track.total_length
                    should_call = distance_since_mpc >= self.mpc_call_interval
                else:
                    should_call = True  # Call every timestep (original behavior)
            else:
                should_call = True  # Non-MPC always called
            
            # Get control from controller
            if is_mpc:
                if should_call:
                    control, info = controller.solve_step(
                        position=position,
                        velocity=velocity,
                        soc=soc,
                        tire_degradation=tire_degradation,
                        fuel_mass=fuel_mass
                    )
                    current_control = control
                    last_mpc_position = position
                    mpc_calls += 1
                # else: use current_control (hold between MPC calls)
                control = current_control
            else:
                # Simple controller (e.g., reference following)
                segment = self.track.get_segment_at_distance(position)
                control = controller.get_control(state, {
                    'gradient': segment.gradient,
                    'radius': segment.radius,
                })
            
            # Track energy
            P_ers = control[0]
            if P_ers > 0:
                energy_deployed += P_ers * self.dt
            else:
                energy_recovered += -P_ers * self.dt
            
            # Get track parameters
            segment = self.track.get_segment_at_distance(position)
            track_params = np.array([segment.gradient, segment.radius])
            
            # Integrate dynamics
            state = self._integrate_rk4(state, control, track_params)
            
            # Apply constraints
            state[1] = np.clip(state[1], constraints['v_min'], constraints['v_max'])
            state[2] = np.clip(state[2], constraints['soc_min'], constraints['soc_max'])
            
            t += self.dt
            
            # Check lap completion
            if state[0] >= self.track.total_length:
                lap_complete = True
        
        if is_mpc and mpc_calls > 0:
            print(f"    MPC calls: {mpc_calls}")
        
        avg_velocity = self.track.total_length / t
        
        return SingleMultiLapResult(
            lap_number=lap_number,
            lap_time=t,
            final_soc=state[2],
            energy_deployed=energy_deployed,
            energy_recovered=energy_recovered,
            tire_degradation=tire_degradation,
            fuel_mass=fuel_mass,
            avg_velocity=avg_velocity
        )
    
    def _integrate_rk4(self, state: np.ndarray, control: np.ndarray,
                       track_params: np.ndarray) -> np.ndarray:
        """RK4 integration step"""
        
        k1 = self.dynamics_func(state, control, track_params).full().flatten()
        k2 = self.dynamics_func(state + self.dt/2 * k1, control, track_params).full().flatten()
        k3 = self.dynamics_func(state + self.dt/2 * k2, control, track_params).full().flatten()
        k4 = self.dynamics_func(state + self.dt * k3, control, track_params).full().flatten()
        
        return state + self.dt / 6 * (k1 + 2*k2 + 2*k3 + k4)


class OpenLoopController:
    """
    Simple controller that follows offline reference trajectory
    Doesn't adapt to degradation (for comparison)
    """
    
    def __init__(self, reference_trajectory: OptimalTrajectory):
        self.reference = reference_trajectory
        
    def get_control(self, state: np.ndarray, track_params: Dict) -> np.ndarray:
        """Get control from reference trajectory"""
        position = state[0] % self.reference.s[-1]
        ref = self.reference.get_reference_at_distance(position)
        
        return np.array([
            ref['P_ers_ref'],
            ref['throttle_ref'],
            ref['brake_ref']
        ])