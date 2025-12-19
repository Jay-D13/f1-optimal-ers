"""
Multi-Lap Race Simulation with Shrinking-Horizon MPC

Implements race strategy optimization as described in academic literature:
- ETH Zurich approach: Lap-time maps + NLP for energy allocation
- Shrinking-horizon MPC: Re-optimize remaining race at each lap
- Convex formulations for real-time computation

Key Features:
1. Lap-time map generation: T(fuel_mass, SOC, tire_condition)
2. Energy allocation optimization across multiple laps
3. Safety car and disturbance handling
4. Tire and fuel degradation modeling
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Callable
import time

from models import VehicleDynamicsModel, F1TrackModel
from config import ERSConfig
from solvers import OptimalTrajectory
from simulation.lap import LapSimulator, LapResult


@dataclass
class RaceConfig:
    """Configuration for race simulation"""
    total_laps: int = 50
    
    # Initial conditions
    initial_soc: float = 0.5
    initial_fuel_kg: float = 110.0      # Full tank
    
    # Degradation models
    fuel_consumption_per_lap: float = 2.0   # kg/lap
    tire_degradation_per_lap: float = 0.01  # 1% grip loss per lap
    
    # ERS constraints (per lap)
    target_soc_per_lap: float = 0.0     # Net SOC change target (0 = neutral)
    soc_tolerance: float = 0.05          # Allowed deviation
    
    # Strategy parameters
    reoptimize_every_n_laps: int = 5     # How often to re-solve strategy
    safety_car_probability: float = 0.02 # Per-lap probability
    
    # Pit stop (simplified)
    pit_window_start: int = 15
    pit_window_end: int = 35
    pit_stop_time: float = 25.0          # seconds


@dataclass
class LapTimeMap:
    """
    Pre-computed lap time as function of vehicle state.
    
    T = f(fuel_mass, initial_soc, final_soc, tire_condition)
    
    This allows rapid evaluation during race strategy optimization.
    """
    # Grid dimensions
    fuel_grid: np.ndarray           # [kg]
    soc_start_grid: np.ndarray      # [0-1]
    soc_end_grid: np.ndarray        # [0-1]
    tire_grid: np.ndarray           # [0-1] degradation factor
    
    # Lap time values (4D array)
    lap_times: np.ndarray           # Shape: (fuel, soc_start, soc_end, tire)
    
    # Energy deployment map
    energy_deployed: np.ndarray     # MJ deployed for each combination
    
    def get_lap_time(self, 
                     fuel_mass: float, 
                     soc_start: float, 
                     soc_end: float,
                     tire_factor: float) -> float:
        """
        Interpolate lap time from map.
        
        Uses multilinear interpolation for smooth gradients.
        """
        from scipy.interpolate import interpn
        
        point = np.array([[fuel_mass, soc_start, soc_end, tire_factor]])
        
        try:
            lap_time = interpn(
                (self.fuel_grid, self.soc_start_grid, 
                 self.soc_end_grid, self.tire_grid),
                self.lap_times,
                point,
                method='linear',
                bounds_error=False,
                fill_value=None
            )[0]
            
            # Handle out-of-bounds with nearest
            if np.isnan(lap_time):
                lap_time = interpn(
                    (self.fuel_grid, self.soc_start_grid, 
                     self.soc_end_grid, self.tire_grid),
                    self.lap_times,
                    point,
                    method='nearest',
                    bounds_error=False
                )[0]
                
        except Exception:
            # Fallback: use center value with scaling
            base_time = self.lap_times[
                len(self.fuel_grid)//2,
                len(self.soc_start_grid)//2,
                len(self.soc_end_grid)//2,
                len(self.tire_grid)//2
            ]
            # Simple scaling
            fuel_factor = 1 + 0.003 * (fuel_mass - 50)
            tire_factor_adj = 1 + 0.1 * (1 - tire_factor)
            lap_time = base_time * fuel_factor * tire_factor_adj
        
        return float(lap_time)


@dataclass
class RaceState:
    """Current state of the race"""
    lap: int = 0
    position_in_lap: float = 0.0
    
    # Vehicle state
    velocity: float = 30.0
    soc: float = 0.5
    fuel_mass: float = 110.0
    tire_condition: float = 1.0
    
    # Cumulative
    total_time: float = 0.0
    total_energy_deployed: float = 0.0
    total_energy_recovered: float = 0.0
    
    # Flags
    pit_completed: bool = False
    safety_car_active: bool = False
    finished: bool = False


@dataclass
class RaceResult:
    """Complete race results"""
    # Per-lap data
    lap_times: List[float] = field(default_factory=list)
    lap_soc_start: List[float] = field(default_factory=list)
    lap_soc_end: List[float] = field(default_factory=list)
    lap_fuel: List[float] = field(default_factory=list)
    lap_tire: List[float] = field(default_factory=list)
    lap_energy_deployed: List[float] = field(default_factory=list)
    lap_energy_recovered: List[float] = field(default_factory=list)
    
    # Summary
    total_time: float = 0.0
    total_laps: int = 0
    pit_stops: List[int] = field(default_factory=list)
    safety_car_laps: List[int] = field(default_factory=list)
    
    # Strategy effectiveness
    planned_times: List[float] = field(default_factory=list)
    actual_times: List[float] = field(default_factory=list)
    
    # Detailed telemetry (optional)
    full_telemetry: Optional[List[LapResult]] = None
    
    def get_summary(self) -> Dict:
        """Get race summary statistics"""
        return {
            'total_time': self.total_time,
            'total_laps': self.total_laps,
            'avg_lap_time': np.mean(self.lap_times) if self.lap_times else 0,
            'best_lap_time': np.min(self.lap_times) if self.lap_times else 0,
            'worst_lap_time': np.max(self.lap_times) if self.lap_times else 0,
            'pit_stops': len(self.pit_stops),
            'safety_car_laps': len(self.safety_car_laps),
            'total_energy_deployed_MJ': sum(self.lap_energy_deployed) / 1e6,
            'total_energy_recovered_MJ': sum(self.lap_energy_recovered) / 1e6,
            'strategy_accuracy': self._compute_strategy_accuracy(),
        }
    
    def _compute_strategy_accuracy(self) -> float:
        """How well did actual match planned?"""
        if not self.planned_times or not self.actual_times:
            return 0.0
        n = min(len(self.planned_times), len(self.actual_times))
        errors = [abs(p - a) for p, a in zip(self.planned_times[:n], self.actual_times[:n])]
        return 1.0 - np.mean(errors) / np.mean(self.actual_times[:n])


class RaceStrategyOptimizer:
    """
    Shrinking-horizon MPC for race energy strategy.
    
    At each lap, solves:
        min  Σ T_i(fuel_i, soc_i, soc_{i+1}, tire_i)
        s.t. soc_{i+1} - soc_i + E_deployed_i - E_recovered_i = 0  (energy balance)
             soc_min ≤ soc_i ≤ soc_max
             E_deployed_i ≤ E_max_deploy
             fuel_{i+1} = fuel_i - fuel_consumption
             tire_{i+1} = tire_i - degradation
    
    This gives optimal energy allocation for remaining laps.
    """
    
    def __init__(self,
                 lap_time_map: LapTimeMap,
                 ers_config: ERSConfig,
                 race_config: RaceConfig):
        
        self.lap_time_map = lap_time_map
        self.ers = ers_config
        self.config = race_config
        
        # Current strategy (SOC targets per lap)
        self.soc_targets: Optional[np.ndarray] = None
        self.energy_allocation: Optional[np.ndarray] = None
        
    def optimize_strategy(self,
                          current_lap: int,
                          current_state: RaceState,
                          remaining_laps: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Optimize energy strategy for remaining race.
        
        Returns:
            soc_targets: Target SOC at start of each remaining lap
            energy_plan: Energy to deploy each lap [J]
        """
        import casadi as ca
        
        N = remaining_laps
        opti = ca.Opti()
        
        # Decision variables
        SOC = opti.variable(N + 1)      # SOC at start of each lap
        E_DEPLOY = opti.variable(N)     # Energy deployed each lap
        E_RECOVER = opti.variable(N)    # Energy recovered each lap
        
        # Fixed parameters
        fuel = current_state.fuel_mass
        tire = current_state.tire_condition
        
        # Objective: minimize total race time
        total_time = 0
        
        for i in range(N):
            # Predict vehicle state for this lap
            fuel_i = fuel - i * self.config.fuel_consumption_per_lap
            tire_i = max(0.5, tire - i * self.config.tire_degradation_per_lap)
            
            # Lap time from map (approximated as quadratic for solver)
            base_time = self.lap_time_map.get_lap_time(
                fuel_i, 0.5, 0.5, tire_i
            )
            
            # Model lap time as function of energy
            # T ≈ T_base - k * (E_deploy - E_nominal) / P_max
            # More deployment = faster lap (up to a point)
            k_speed = 0.1 / self.ers.max_deployment_power
            E_nominal = 2e6  # 2 MJ nominal deployment
            
            lap_time_i = base_time - k_speed * (E_DEPLOY[i] - E_nominal)
            
            # Add penalty for SOC deviation from charge-sustaining
            soc_penalty = 10.0 * (SOC[i+1] - SOC[i])**2
            
            total_time += lap_time_i + soc_penalty
        
        opti.minimize(total_time)
        
        # Constraints
        
        # Initial SOC
        opti.subject_to(SOC[0] == current_state.soc)
        
        # SOC dynamics (each lap)
        for i in range(N):
            # Energy balance: SOC change = (recovered - deployed) / capacity
            dsoc = (E_RECOVER[i] * self.ers.recovery_efficiency - 
                   E_DEPLOY[i] / self.ers.deployment_efficiency) / self.ers.battery_capacity
            opti.subject_to(SOC[i+1] == SOC[i] + dsoc)
        
        # SOC bounds
        opti.subject_to(opti.bounded(self.ers.min_soc, SOC, self.ers.max_soc))
        
        # Energy bounds (per lap)
        opti.subject_to(opti.bounded(0, E_DEPLOY, self.ers.deployment_limit_per_lap))
        opti.subject_to(opti.bounded(0, E_RECOVER, self.ers.recovery_limit_per_lap))
        
        # Charge-sustaining constraint (final SOC ≈ initial)
        opti.subject_to(SOC[N] >= current_state.soc - 0.1)
        opti.subject_to(SOC[N] <= current_state.soc + 0.1)
        
        # Solve
        opts = {
            'ipopt.max_iter': 200,
            'ipopt.print_level': 0,
            'print_time': 0,
        }
        opti.solver('ipopt', opts)
        
        # Initial guess: uniform energy distribution
        opti.set_initial(SOC, np.ones(N + 1) * current_state.soc)
        opti.set_initial(E_DEPLOY, np.ones(N) * 2e6)
        opti.set_initial(E_RECOVER, np.ones(N) * 1.5e6)
        
        try:
            sol = opti.solve()
            soc_targets = sol.value(SOC)
            energy_plan = sol.value(E_DEPLOY)
        except Exception as e:
            print(f"   Strategy optimization failed: {e}")
            # Fallback: uniform strategy
            soc_targets = np.ones(N + 1) * current_state.soc
            energy_plan = np.ones(N) * 2e6
        
        self.soc_targets = soc_targets
        self.energy_allocation = energy_plan
        
        return soc_targets, energy_plan
    
    def get_lap_targets(self, lap_in_race: int) -> Dict:
        """Get targets for a specific lap"""
        if self.soc_targets is None or lap_in_race >= len(self.soc_targets) - 1:
            return {
                'soc_start': 0.5,
                'soc_end': 0.5,
                'energy_budget': 2e6,
            }
        
        return {
            'soc_start': self.soc_targets[lap_in_race],
            'soc_end': self.soc_targets[lap_in_race + 1],
            'energy_budget': self.energy_allocation[lap_in_race],
        }


class MultiLapRaceSimulator:
    """
    Full race simulation with integrated MPC strategy.
    
    Combines:
    1. Offline optimal trajectory (single lap)
    2. Race strategy optimizer (multi-lap)
    3. MPC controller (real-time)
    4. Physical simulation
    """
    
    def __init__(self,
                 vehicle_model: VehicleDynamicsModel,
                 track_model: F1TrackModel,
                 ers_config: ERSConfig,
                 offline_trajectory: OptimalTrajectory,
                 controller,  # ERSMPCController
                 race_config: Optional[RaceConfig] = None):
        
        self.vehicle = vehicle_model
        self.track = track_model
        self.ers = ers_config
        self.offline_trajectory = offline_trajectory
        self.controller = controller
        self.config = race_config or RaceConfig()
        
        # Create lap simulator
        self.lap_simulator = LapSimulator(
            vehicle_model, track_model, controller, dt=0.1
        )
        
        # Create or load lap time map
        self.lap_time_map = self._generate_lap_time_map()
        
        # Create strategy optimizer
        self.strategy_optimizer = RaceStrategyOptimizer(
            self.lap_time_map, ers_config, self.config
        )
        
        print(f"\n   Multi-Lap Race Simulator initialized:")
        print(f"     Total laps: {self.config.total_laps}")
        print(f"     Initial fuel: {self.config.initial_fuel_kg} kg")
        
    def _generate_lap_time_map(self) -> LapTimeMap:
        """
        Generate lap time map from offline trajectory.
        
        For production, this would involve running many offline optimizations.
        Here we approximate from the reference trajectory with scaling.
        """
        print("   Generating lap time map (approximation)...")
        
        # Grid definition
        fuel_grid = np.array([30, 50, 70, 90, 110])           # kg
        soc_start_grid = np.array([0.2, 0.35, 0.5, 0.65, 0.8])
        soc_end_grid = np.array([0.2, 0.35, 0.5, 0.65, 0.8])
        tire_grid = np.array([0.7, 0.8, 0.9, 1.0])
        
        # Reference lap time
        T_ref = self.offline_trajectory.lap_time
        
        # Build lap time array
        shape = (len(fuel_grid), len(soc_start_grid), 
                 len(soc_end_grid), len(tire_grid))
        lap_times = np.zeros(shape)
        energy_deployed = np.zeros(shape)
        
        for i, fuel in enumerate(fuel_grid):
            for j, soc_s in enumerate(soc_start_grid):
                for k, soc_e in enumerate(soc_end_grid):
                    for l, tire in enumerate(tire_grid):
                        # Approximate lap time with physics-based scaling
                        
                        # Fuel effect: heavier = slower
                        fuel_factor = 1 + 0.003 * (fuel - 70)  # ~0.3s per 10kg
                        
                        # Tire effect: degraded = slower
                        tire_factor = 1 + 0.15 * (1 - tire)  # ~1.5s at 90% grip
                        
                        # SOC effect: net deployment = faster
                        dsoc = soc_s - soc_e  # Positive = net deployment
                        energy_used = dsoc * self.ers.battery_capacity
                        # ~0.3s per MJ deployed (simplified)
                        soc_factor = 1 - 0.3 * energy_used / 1e6 / T_ref
                        
                        lap_times[i, j, k, l] = T_ref * fuel_factor * tire_factor * soc_factor
                        energy_deployed[i, j, k, l] = max(0, energy_used)
        
        print(f"   ✓ Lap time map generated: {shape}")
        print(f"     Time range: {lap_times.min():.2f}s - {lap_times.max():.2f}s")
        
        return LapTimeMap(
            fuel_grid=fuel_grid,
            soc_start_grid=soc_start_grid,
            soc_end_grid=soc_end_grid,
            tire_grid=tire_grid,
            lap_times=lap_times,
            energy_deployed=energy_deployed,
        )
    
    def run_race(self, 
                 store_telemetry: bool = False,
                 verbose: bool = True) -> RaceResult:
        """
        Run complete race simulation.
        
        Args:
            store_telemetry: Whether to store detailed per-lap telemetry
            verbose: Print progress updates
            
        Returns:
            RaceResult with complete race data
        """
        
        print("\n" + "="*60)
        print("RACE SIMULATION")
        print("="*60)
        
        # Initialize state
        state = RaceState(
            soc=self.config.initial_soc,
            fuel_mass=self.config.initial_fuel_kg,
        )
        
        result = RaceResult()
        if store_telemetry:
            result.full_telemetry = []
        
        # Initial strategy optimization
        self.strategy_optimizer.optimize_strategy(
            current_lap=0,
            current_state=state,
            remaining_laps=self.config.total_laps
        )
        
        # Main race loop
        for lap in range(self.config.total_laps):
            state.lap = lap
            
            # Check for safety car (random event)
            if np.random.random() < self.config.safety_car_probability:
                state.safety_car_active = True
                result.safety_car_laps.append(lap)
                if verbose:
                    print(f"   ⚠ Safety car deployed on lap {lap + 1}")
            else:
                state.safety_car_active = False
            
            # Re-optimize strategy periodically or after safety car
            if lap % self.config.reoptimize_every_n_laps == 0 or state.safety_car_active:
                remaining = self.config.total_laps - lap
                self.strategy_optimizer.optimize_strategy(
                    current_lap=lap,
                    current_state=state,
                    remaining_laps=remaining
                )
                if verbose:
                    print(f"   Strategy re-optimized (lap {lap + 1}, {remaining} remaining)")
            
            # Get lap targets from strategy
            lap_targets = self.strategy_optimizer.get_lap_targets(lap - state.lap)
            result.planned_times.append(
                self.lap_time_map.get_lap_time(
                    state.fuel_mass, lap_targets['soc_start'], 
                    lap_targets['soc_end'], state.tire_condition
                )
            )
            
            # Record pre-lap state
            result.lap_soc_start.append(state.soc)
            result.lap_fuel.append(state.fuel_mass)
            result.lap_tire.append(state.tire_condition)
            
            # Run lap simulation
            lap_start = time.time()
            lap_result = self._run_single_lap(state, lap_targets, state.safety_car_active)
            lap_time_actual = time.time() - lap_start
            
            # Update state from lap result
            state.velocity = lap_result.velocities[-1]
            state.soc = lap_result.socs[-1]
            state.fuel_mass -= self.config.fuel_consumption_per_lap
            state.tire_condition -= self.config.tire_degradation_per_lap
            state.tire_condition = max(0.5, state.tire_condition)  # Min 50% grip
            state.total_time += lap_result.lap_time
            state.total_energy_deployed += lap_result.energy_deployed
            state.total_energy_recovered += lap_result.energy_recovered
            
            # Record results
            result.lap_times.append(lap_result.lap_time)
            result.lap_soc_end.append(state.soc)
            result.lap_energy_deployed.append(lap_result.energy_deployed)
            result.lap_energy_recovered.append(lap_result.energy_recovered)
            result.actual_times.append(lap_result.lap_time)
            
            if store_telemetry:
                result.full_telemetry.append(lap_result)
            
            # Progress update
            if verbose and (lap + 1) % 5 == 0:
                print(f"   Lap {lap + 1}/{self.config.total_laps}: "
                      f"{lap_result.lap_time:.3f}s, SOC: {state.soc*100:.1f}%, "
                      f"Fuel: {state.fuel_mass:.1f}kg, Tire: {state.tire_condition*100:.0f}%")
        
        # Finalize
        result.total_time = state.total_time
        result.total_laps = self.config.total_laps
        
        print("\n" + "="*60)
        print("RACE COMPLETE")
        print("="*60)
        
        summary = result.get_summary()
        print(f"   Total time: {summary['total_time']:.3f}s ({summary['total_time']/60:.2f} min)")
        print(f"   Best lap: {summary['best_lap_time']:.3f}s")
        print(f"   Avg lap: {summary['avg_lap_time']:.3f}s")
        print(f"   Energy deployed: {summary['total_energy_deployed_MJ']:.2f} MJ")
        print(f"   Strategy accuracy: {summary['strategy_accuracy']*100:.1f}%")
        
        return result
    
    def _run_single_lap(self,
                        state: RaceState,
                        targets: Dict,
                        safety_car: bool = False) -> LapResult:
        """Run a single lap with MPC controller"""
        
        # Adjust controller targets based on strategy
        # (In production, this would modify the reference trajectory)
        
        # Handle safety car (reduced speed)
        if safety_car:
            # Simplified: just return a slower lap with minimal ERS
            sc_time = self.offline_trajectory.lap_time * 1.4  # 40% slower
            return LapResult(
                times=np.array([0, sc_time]),
                positions=np.array([0, self.track.total_length]),
                velocities=np.array([30, 30]),
                socs=np.array([state.soc, state.soc]),  # No ERS use
                P_ers_history=np.array([0]),
                throttle_history=np.array([0.3]),
                brake_history=np.array([0]),
                lap_time=sc_time,
                final_soc=state.soc,
                energy_deployed=0,
                energy_recovered=0,
                completed=True,
            )
        
        # Normal lap: use MPC controller
        return self.lap_simulator.simulate_lap(
            initial_soc=state.soc,
            initial_velocity=state.velocity,
            max_time=self.offline_trajectory.lap_time * 2,
            reference=self.offline_trajectory
        )
    
    def run_quick_race(self, n_laps: int = 10) -> RaceResult:
        """
        Run a quick race simulation without full physics.
        
        Uses lap time map directly for fast evaluation.
        Useful for strategy testing.
        """
        print(f"\n   Quick race simulation ({n_laps} laps)...")
        
        state = RaceState(
            soc=self.config.initial_soc,
            fuel_mass=self.config.initial_fuel_kg,
        )
        
        result = RaceResult()
        
        for lap in range(n_laps):
            # Simple energy management: alternate deploy/recover
            if lap % 2 == 0:
                soc_end = max(self.ers.min_soc, state.soc - 0.1)
            else:
                soc_end = min(self.ers.max_soc, state.soc + 0.05)
            
            # Get lap time from map
            lap_time = self.lap_time_map.get_lap_time(
                state.fuel_mass, state.soc, soc_end, state.tire_condition
            )
            
            # Update state
            dsoc = soc_end - state.soc
            energy = abs(dsoc) * self.ers.battery_capacity
            
            result.lap_times.append(lap_time)
            result.lap_soc_start.append(state.soc)
            result.lap_soc_end.append(soc_end)
            result.lap_fuel.append(state.fuel_mass)
            result.lap_tire.append(state.tire_condition)
            result.lap_energy_deployed.append(energy if dsoc < 0 else 0)
            result.lap_energy_recovered.append(energy if dsoc > 0 else 0)
            
            state.soc = soc_end
            state.fuel_mass -= self.config.fuel_consumption_per_lap
            state.tire_condition -= self.config.tire_degradation_per_lap
            state.tire_condition = max(0.5, state.tire_condition)
            state.total_time += lap_time
        
        result.total_time = state.total_time
        result.total_laps = n_laps
        
        print(f"   ✓ Quick race complete: {result.total_time:.2f}s "
              f"(avg: {np.mean(result.lap_times):.3f}s/lap)")
        
        return result


def compare_race_strategies(
    vehicle_model: VehicleDynamicsModel,
    track_model: F1TrackModel,
    ers_config: ERSConfig,
    offline_trajectory: OptimalTrajectory,
    strategies: Dict[str, Callable],
    n_laps: int = 10
) -> Dict[str, RaceResult]:
    """
    Compare different race strategies.
    
    Args:
        strategies: Dict of strategy_name -> controller_factory
        n_laps: Number of laps to simulate
        
    Returns:
        Dict of strategy_name -> RaceResult
    """
    print("\n" + "="*60)
    print("RACE STRATEGY COMPARISON")
    print("="*60)
    
    results = {}
    
    for name, controller_factory in strategies.items():
        print(f"\n   Testing strategy: {name}")
        
        controller = controller_factory()
        controller.set_reference(offline_trajectory)
        
        sim = MultiLapRaceSimulator(
            vehicle_model=vehicle_model,
            track_model=track_model,
            ers_config=ers_config,
            offline_trajectory=offline_trajectory,
            controller=controller,
            race_config=RaceConfig(total_laps=n_laps)
        )
        
        results[name] = sim.run_quick_race(n_laps)
    
    # Summary comparison
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    print(f"{'Strategy':<20} {'Total Time':>12} {'Best Lap':>12} {'Avg Lap':>12}")
    print("-" * 60)
    
    for name, result in results.items():
        summary = result.get_summary()
        print(f"{name:<20} {summary['total_time']:>12.2f}s {summary['best_lap_time']:>12.3f}s "
              f"{summary['avg_lap_time']:>12.3f}s")
    
    return results
