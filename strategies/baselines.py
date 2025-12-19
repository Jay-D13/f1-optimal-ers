import numpy as np
from typing import Dict, Tuple
from scipy.interpolate import interp1d

from config import VehicleConfig, ERSConfig

class TrackingStrategy:

    def __init__(self, 
                 vehicle_config: VehicleConfig, 
                 ers_config: ERSConfig, 
                 track_model,
                 reference_profile=None):
        self.vehicle = vehicle_config
        self.ers = ers_config
        self.track = track_model
        self.profile = reference_profile
        
        if reference_profile is not None:

            if hasattr(reference_profile, 'v'):
                v_prof = reference_profile.v
            elif hasattr(reference_profile, 'v_opt'):
                v_prof = reference_profile.v_opt
            else:
                raise AttributeError(f"Reference profile {type(reference_profile)} must have 'v' or 'v_opt'")
            
            if hasattr(reference_profile, 'a_x'):
                a_prof = reference_profile.a_x
            else:
                s_arr = reference_profile.s
                dv_ds = np.gradient(v_prof, s_arr)
                a_prof = v_prof * dv_ds

            self.v_ref_interp = interp1d(
                reference_profile.s, 
                v_prof, 
                kind='linear', fill_value="extrapolate"
            )
            # Interpolate Acceleration
            self.a_ref_interp = interp1d(
                reference_profile.s, 
                a_prof, 
                kind='linear', fill_value="extrapolate"
            )
        else:
            self.v_ref_interp = None
            self.a_ref_interp = None
            
        # PD gains for velocity tracking
        self.kp_v = 2.0   # Proportional gain on velocity error

    @property
    def name(self) -> str:
        return "TrackingStrategy"

    def _compute_forces(self, v: float, gradient: float, radius: float = 1000.0) -> Dict[str, float]:
        """Compute all forces at current state, including grip limits."""
        mass = self.vehicle.mass
        g = self.vehicle.g
        rho = self.vehicle.rho_air
        c_w_a = self.vehicle.c_w_a
        c_z_a = self.vehicle.c_z_a_f + self.vehicle.c_z_a_r
        cr = self.vehicle.cr
        
        # Aerodynamic forces
        q = 0.5 * rho * v**2
        f_drag = q * c_w_a
        f_downforce = q * c_z_a
        
        # Normal force (weight + downforce)
        f_normal = mass * g * np.cos(gradient) + f_downforce
        
        # Rolling resistance
        f_roll = cr * f_normal
        
        # Gravity component
        f_grav = mass * g * np.sin(gradient)
        
        # Grip calculation (friction circle)
        mu_avg = 0.5 * (1.65 + 1.95)  # Average friction coefficient
        f_grip_total = mu_avg * f_normal
        
        # Lateral force from cornering
        safe_radius = max(radius, 15.0)
        a_lat = v**2 / safe_radius
        f_lat = mass * a_lat
        
        # Remaining grip for longitudinal
        f_grip_long_sq = f_grip_total**2 - f_lat**2
        f_grip_long = np.sqrt(max(f_grip_long_sq, 100.0))
        
        return {
            'drag': f_drag,
            'roll': f_roll,
            'gravity': f_grav,
            'normal': f_normal,
            'total_resistance': f_drag + f_roll + f_grav,
            'grip_total': f_grip_total,
            'grip_long': f_grip_long,
            'lat_force': f_lat,
            'is_grip_limited': f_lat > 0.7 * f_grip_total  # >70% grip used for cornering
        }

    def get_control(self, state: np.ndarray, track_info: Dict) -> np.ndarray:
        """
        Compute control [P_ers, throttle, brake] to track reference.
        """
        s, v, soc = state
        grad = track_info.get('gradient', 0.0)
        radius = track_info.get('radius', 1000.0)
        
        # Get reference
        v_ref = float(self.v_ref_interp(s)) if self.v_ref_interp else v
        a_ref = float(self.a_ref_interp(s)) if self.a_ref_interp else 0.0
        
        # Compute required force (feedforward + feedback)
        throttle, brake, P_ers = self._compute_tracking_control(
            v, v_ref, a_ref, grad, radius, soc
        )
        
        return np.array([P_ers, throttle, brake])
    
    def _compute_tracking_control(self, v: float, v_ref: float, a_ref: float, 
                                   gradient: float, radius: float, soc: float
                                   ) -> Tuple[float, float, float]:
        """
        Compute throttle/brake/ERS to track reference.
        
        Returns: (throttle, brake, P_ers)
        """
        mass = self.vehicle.mass
        v_safe = max(v, 5.0)
        
        # Compute forces including grip limits
        forces = self._compute_forces(v, gradient, radius)
        
        # Velocity error for feedback
        v_error = v_ref - v
        
        # Feedforward: force needed to achieve reference acceleration
        # Plus feedback correction based on velocity error
        a_cmd = a_ref + self.kp_v * v_error
        
        # Total force required
        f_required = mass * a_cmd + forces['total_resistance']
        
        # Default: no ERS
        P_ers = 0.0
        throttle = 0.0
        brake = 0.0
        
        if f_required > 0:
            # Need propulsion
            P_required = f_required * v_safe
            
            # Throttle assuming no ERS deployment
            throttle = P_required / self.vehicle.pow_max_ice
            throttle = np.clip(throttle, 0.0, 1.0)
            
        else:
            # Need braking
            brake = np.clip(-f_required / self.vehicle.max_brake_force, 0.0, 1.0)
        
        return throttle, brake, P_ers


class GreedyStrategy(TrackingStrategy):
    """
    Greedy ERS: Deploy when accelerating hard, harvest when braking.
    
    FIXED: When deploying ERS, reduce throttle so total power = required power.
    FIXED: Don't deploy ERS when grip-limited (would waste energy).
    """
    @property
    def name(self) -> str:
        return "Greedy (KERS)"
    
    def get_control(self, state: np.ndarray, track_info: Dict) -> np.ndarray:
        s, v, soc = state
        grad = track_info.get('gradient', 0.0)
        radius = track_info.get('radius', 1000.0)
        v_safe = max(v, 5.0)
        mass = self.vehicle.mass
        
        # Get reference
        v_ref = float(self.v_ref_interp(s)) if self.v_ref_interp else v
        a_ref = float(self.a_ref_interp(s)) if self.a_ref_interp else 0.0
        
        # Compute forces including grip
        forces = self._compute_forces(v, grad, radius)
        
        # Feedback control
        v_error = v_ref - v
        a_cmd = a_ref + self.kp_v * v_error
        f_required = mass * a_cmd + forces['total_resistance']
        
        P_ers = 0.0
        throttle = 0.0
        brake = 0.0
        
        if f_required > 0:
            # Need propulsion
            P_required = f_required * v_safe
            
            # Check if we're grip-limited (in a corner)
            # Don't deploy ERS if grip-limited - it would waste energy!
            is_grip_limited = forces['is_grip_limited']
            
            # Greedy logic: deploy ERS when accelerating hard AND not grip limited
            should_deploy = (
                a_cmd > 3.0 and                    # Significant acceleration
                v > 25.0 and                       # Not too slow
                soc > self.ers.min_soc + 0.05 and  # Have charge buffer
                not is_grip_limited                # NOT in corner (would waste energy)
            )
            
            if should_deploy:
                # Deploy full ERS, REDUCE THROTTLE to compensate
                P_ers = self.ers.max_deployment_power
                P_ice_needed = max(0, P_required - P_ers)
                throttle = P_ice_needed / self.vehicle.pow_max_ice
            else:
                # ICE only
                throttle = P_required / self.vehicle.pow_max_ice
            
            throttle = np.clip(throttle, 0.0, 1.0)
            
        else:
            # Need braking - harvest energy
            if soc < self.ers.max_soc - 0.02:
                P_ers = -self.ers.max_recovery_power
            
            brake = np.clip(-f_required / self.vehicle.max_brake_force, 0.0, 1.0)
        
        return np.array([P_ers, throttle, brake])


class AlwaysDeployStrategy(TrackingStrategy):
    """
    Always deploy ERS when accelerating (aggressive energy use).
    
    FIXED: Reduce throttle when deploying ERS.
    FIXED: Don't deploy when grip-limited.
    """
    @property
    def name(self) -> str:
        return "Always Deploy"
    
    def get_control(self, state: np.ndarray, track_info: Dict) -> np.ndarray:
        s, v, soc = state
        grad = track_info.get('gradient', 0.0)
        radius = track_info.get('radius', 1000.0)
        v_safe = max(v, 5.0)
        mass = self.vehicle.mass
        
        v_ref = float(self.v_ref_interp(s)) if self.v_ref_interp else v
        a_ref = float(self.a_ref_interp(s)) if self.a_ref_interp else 0.0
        
        forces = self._compute_forces(v, grad, radius)
        v_error = v_ref - v
        a_cmd = a_ref + self.kp_v * v_error
        f_required = mass * a_cmd + forces['total_resistance']
        
        P_ers = 0.0
        throttle = 0.0
        brake = 0.0
        
        if f_required > 0:
            P_required = f_required * v_safe
            
            # Always deploy when accelerating (if we have charge and not grip-limited)
            should_deploy = (
                soc > self.ers.min_soc and
                not forces['is_grip_limited']  # Don't waste energy in corners
            )
            
            if should_deploy:
                P_ers = self.ers.max_deployment_power
                P_ice_needed = max(0, P_required - P_ers)
                throttle = P_ice_needed / self.vehicle.pow_max_ice
            else:
                throttle = P_required / self.vehicle.pow_max_ice
            
            throttle = np.clip(throttle, 0.0, 1.0)
        else:
            if soc < self.ers.max_soc - 0.02:
                P_ers = -self.ers.max_recovery_power
            brake = np.clip(-f_required / self.vehicle.max_brake_force, 0.0, 1.0)
        
        return np.array([P_ers, throttle, brake])


class TargetSOCStrategy(TrackingStrategy):
    """
    Maintain target SOC - deploy when above target, harvest when below.
    
    FIXED: Reduce throttle when deploying ERS.
    FIXED: Don't deploy when grip-limited.
    """
    def __init__(self, vehicle_config, ers_config, track_model, reference_profile, target_soc=0.5):
        super().__init__(vehicle_config, ers_config, track_model, reference_profile)
        self.target_soc = target_soc
        self.soc_deadband = 0.05

    @property
    def name(self) -> str:
        return f"Target SOC ({self.target_soc*100:.0f}%)"

    def get_control(self, state: np.ndarray, track_info: Dict) -> np.ndarray:
        s, v, soc = state
        grad = track_info.get('gradient', 0.0)
        radius = track_info.get('radius', 1000.0)
        v_safe = max(v, 5.0)
        mass = self.vehicle.mass
        
        v_ref = float(self.v_ref_interp(s)) if self.v_ref_interp else v
        a_ref = float(self.a_ref_interp(s)) if self.a_ref_interp else 0.0
        
        forces = self._compute_forces(v, grad, radius)
        v_error = v_ref - v
        a_cmd = a_ref + self.kp_v * v_error
        f_required = mass * a_cmd + forces['total_resistance']
        
        P_ers = 0.0
        throttle = 0.0
        brake = 0.0
        
        # SOC-based decision
        soc_error = soc - self.target_soc
        
        if f_required > 0:
            P_required = f_required * v_safe
            
            # Deploy if above target SOC and not grip-limited
            should_deploy = (
                soc_error > self.soc_deadband and 
                soc > self.ers.min_soc and
                not forces['is_grip_limited']
            )
            
            if should_deploy:
                # Proportional deployment based on SOC error
                deploy_fraction = min(1.0, (soc_error - self.soc_deadband) / 0.2)
                P_ers = deploy_fraction * self.ers.max_deployment_power
                P_ice_needed = max(0, P_required - P_ers)
                throttle = P_ice_needed / self.vehicle.pow_max_ice
            else:
                throttle = P_required / self.vehicle.pow_max_ice
            
            throttle = np.clip(throttle, 0.0, 1.0)
        else:
            # Always harvest when braking (if below max SOC)
            if soc < self.ers.max_soc - 0.02:
                P_ers = -self.ers.max_recovery_power
            brake = np.clip(-f_required / self.vehicle.max_brake_force, 0.0, 1.0)
        
        return np.array([P_ers, throttle, brake])


class SmartRuleBasedStrategy(TrackingStrategy):
    """
    Smart heuristics: Deploy on straights when grip-available, harvest on braking.
    
    FIXED: Reduce throttle when deploying ERS.
    FIXED: Use actual grip calculations instead of radius heuristics.
    """
    @property
    def name(self) -> str:
        return "Smart Heuristic"
    
    def get_control(self, state: np.ndarray, track_info: Dict) -> np.ndarray:
        s, v, soc = state
        grad = track_info.get('gradient', 0.0)
        radius = track_info.get('radius', 1000.0)
        v_safe = max(v, 5.0)
        mass = self.vehicle.mass
        
        v_ref = float(self.v_ref_interp(s)) if self.v_ref_interp else v
        a_ref = float(self.a_ref_interp(s)) if self.a_ref_interp else 0.0
        
        forces = self._compute_forces(v, grad, radius)
        v_error = v_ref - v
        a_cmd = a_ref + self.kp_v * v_error
        f_required = mass * a_cmd + forces['total_resistance']
        
        P_ers = 0.0
        throttle = 0.0
        brake = 0.0
        
        # Determine conditions
        is_straight = radius > 300
        is_efficient_deploy = 20.0 < v < 90.0  # Sweet spot for ERS
        has_available_grip = not forces['is_grip_limited']  # Key check!
        
        if f_required > 0:
            P_required = f_required * v_safe
            
            # Smart deployment: only deploy when it will be useful
            should_deploy = (
                is_straight and                    # On straight
                is_efficient_deploy and           # In efficient speed range
                a_cmd > 2.0 and                   # Actually accelerating
                soc > self.ers.min_soc + 0.05 and # Have charge buffer
                has_available_grip                # Grip available to use the power!
            )
            
            if should_deploy:
                P_ers = self.ers.max_deployment_power
                P_ice_needed = max(0, P_required - P_ers)
                throttle = P_ice_needed / self.vehicle.pow_max_ice
            else:
                throttle = P_required / self.vehicle.pow_max_ice
            
            throttle = np.clip(throttle, 0.0, 1.0)
        else:
            # Harvest under braking
            if soc < self.ers.max_soc - 0.02:
                P_ers = -self.ers.max_recovery_power
            brake = np.clip(-f_required / self.vehicle.max_brake_force, 0.0, 1.0)
        
        return np.array([P_ers, throttle, brake])


class OptimalTrackingStrategy(TrackingStrategy):

    def __init__(self, vehicle_config, ers_config, track_model, reference_profile=None, optimal_trajectory=None):

        traj = optimal_trajectory if optimal_trajectory is not None else reference_profile
        
        super().__init__(vehicle_config, ers_config, track_model, traj)
        
        if traj is not None and hasattr(traj, 'P_ers_opt'):
            n_ers = len(traj.P_ers_opt)
            self.P_ers_interp = interp1d(
                traj.s[:n_ers],
                traj.P_ers_opt,
                kind='linear', fill_value="extrapolate" 
            )
        else:
            self.P_ers_interp = None
    
    @property
    def name(self) -> str:
        return "Optimal Tracking"
    
    def get_control(self, state: np.ndarray, track_info: Dict) -> np.ndarray:
        s, v, soc = state
        grad = track_info.get('gradient', 0.0)
        radius = track_info.get('radius', 1000.0)
        v_safe = max(v, 5.0)
        mass = self.vehicle.mass
        
        v_ref = float(self.v_ref_interp(s)) if self.v_ref_interp else v
        a_ref = float(self.a_ref_interp(s)) if self.a_ref_interp else 0.0
        P_ers_ref = float(self.P_ers_interp(s)) if self.P_ers_interp else 0.0
        
        forces = self._compute_forces(v, grad, radius)
        v_error = v_ref - v
        a_cmd = a_ref + self.kp_v * v_error
        f_required = mass * a_cmd + forces['total_resistance']
        
        # Use optimal ERS (clamped by SOC limits)
        P_ers = P_ers_ref
        if P_ers > 0 and soc <= self.ers.min_soc:
            P_ers = 0.0
        if P_ers < 0 and soc >= self.ers.max_soc:
            P_ers = 0.0
        
        throttle = 0.0
        brake = 0.0
        
        if f_required > 0:
            P_required = f_required * v_safe
            P_ice_needed = max(0, P_required - max(0, P_ers))
            throttle = np.clip(P_ice_needed / self.vehicle.pow_max_ice, 0.0, 1.0)
        else:
            brake = np.clip(-f_required / self.vehicle.max_brake_force, 0.0, 1.0)
        
        return np.array([P_ers, throttle, brake])