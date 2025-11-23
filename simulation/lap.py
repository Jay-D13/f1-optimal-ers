import numpy as np
from typing import Dict

from models import VehicleDynamicsModel, F1TrackModel


class LapSimulator:
    """Simulate a complete lap with given controller"""
    
    def __init__(self, vehicle_model: VehicleDynamicsModel,
                 track_model: F1TrackModel,
                 controller):
        self.vehicle_model = vehicle_model
        self.track_model = track_model
        self.controller = controller
        self.dynamics_func = vehicle_model.create_casadi_function()
        
    def simulate_lap(self, initial_soc: float = 0.5, max_time: float = 200) -> Dict:
        """Simulate one complete lap"""
        # Initial conditions
        initial_velocity = 30  # m/s
        state = np.array([0, initial_velocity, initial_soc])
        
        # Storage for results
        states = [state.copy()]
        controls = []
        times = [0]
        
        dt = 0.1
        lap_complete = False
        t = 0
        segment_idx = 0
        
        while not lap_complete and t < max_time:
            # Get current track segment
            current_segment = self.track_model.segments[segment_idx % len(self.track_model.segments)]
            
            # Get control action
            if hasattr(self.controller, 'solve_mpc_step'):
                control, info = self.controller.solve_mpc_step(state, state[0])
            else:
                control = self.controller.get_control(state, current_segment)
                
            controls.append(control.copy())
            
            # Get track parameters
            track_params = np.array([current_segment.gradient, current_segment.radius])
            
            # Integrate dynamics using the CasADi function
            x_dot = self.dynamics_func(state, control, track_params).full().flatten()
            state = state + x_dot * dt
            
            # Apply state constraints
            state[1] = np.clip(state[1], 5, 95)  # Velocity limits
            state[2] = np.clip(state[2], 0.1, 0.9)  # SOC limits (respect battery health)
            
            # Update segment index based on distance traveled
            if state[0] > (segment_idx + 1) * 50:
                segment_idx += 1
            
            states.append(state.copy())
            t += dt
            times.append(t)
            
            # Check lap completion
            if state[0] >= self.track_model.total_length:
                lap_complete = True
                
        return {
            'states': np.array(states),
            'controls': np.array(controls),
            'times': np.array(times),
            'lap_time': t,
            'final_soc': state[2],
            'completed': lap_complete
        }
        