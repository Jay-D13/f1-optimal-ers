"""
Equivalent Consumption Minimization Strategy (ECMS) for F1 ERS

Mathematical Background:
========================

The ERS optimization problem can be written as:

    minimize    J = ∫ (1/v) ds                    (lap time)
    subject to  dv/ds = f(v, P_ers, ...)          (velocity dynamics)
                dSOC/ds = g(v, P_ers, ...)        (battery dynamics)  
                SOC(0) = SOC_0, SOC(L) ≥ SOC_f    (boundary conditions)
                ∫ P_ers⁺ ds ≤ E_max              (energy limit)

Using Pontryagin's Minimum Principle (PMP), we form the Hamiltonian:

    H = (1/v) + λ_v · f(v, P_ers) + λ_SOC · g(v, P_ers)

The optimal control satisfies:  u* = argmin H

For the battery dynamics specifically:
    λ_SOC represents the "marginal value" of stored energy
    
ECMS approximates PMP by:
1. Assuming λ_SOC is constant (or slowly varying) → call it λ
2. At each instant, choose P_ers to minimize instantaneous cost:
   
   J_instant = (performance loss from not using ERS) + λ · (battery energy used)

The equivalence factor λ can be interpreted as:
- High λ: "Battery energy is expensive" → conserve → less deployment
- Low λ: "Battery energy is cheap" → spend → more deployment

For charge-sustaining operation, the optimal λ satisfies:
    SOC(end) = SOC(start)

References:
-----------
- Paganelli et al., "Equivalent Consumption Minimization Strategy for HEVs"
- Onori & Serrao, "Adaptive ECMS for Plug-in HEVs" 
- Salazar et al., "Time-Optimal Control for Racing" (connects ECMS to PMP)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, List
import time

from solvers.base import BaseSolver, OptimalTrajectory


@dataclass
class ECMSConfig:
    """Configuration for ECMS solver"""
    
    # Base equivalence factor (if not using adaptive)
    lambda_base: float = 2.5
    
    # Adaptive lambda parameters (PI controller on SOC error)
    use_adaptive_lambda: bool = True
    K_p: float = 5.0            # Proportional gain for SOC error
    K_i: float = 1.0            # Integral gain for accumulated error
    
    # SOC target trajectory (for charge-sustaining)
    # If None, uses linear interpolation from initial to final_min
    soc_target_profile: Optional[np.ndarray] = None
    
    # Terminal SOC enforcement
    enforce_terminal_soc: bool = True
    terminal_soc_weight: float = 10.0  # How aggressively to enforce final SOC
    
    # Lookahead for predictive ECMS (A-ECMS)
    use_predictive: bool = False
    lookahead_distance: float = 500.0  # meters
    
    # Discretization for ERS power search
    n_ers_candidates: int = 41  # Odd number to include 0
    
    # Smoothing
    rate_limit: float = 50_000  # W/step max change in P_ers


class ECMSSolver(BaseSolver):
    """
    ECMS-based solver for F1 ERS optimization.
    
    Key differences from NLP:
    - NLP: Solves entire lap simultaneously (global optimal)
    - ECMS: Makes local decisions at each point (suboptimal but fast)
    
    Key differences from PMP:
    - PMP: Solves costate equations to find optimal λ(s)
    - ECMS: Uses heuristic/adaptive rules for λ
    
    When ECMS equals PMP:
    - For linear systems with quadratic cost, constant λ is optimal
    - For nonlinear systems, adaptive λ can approximate PMP solution
    """
    
    def __init__(self, 
                 vehicle_model, 
                 track_model, 
                 ers_config,
                 ecms_config: Optional[ECMSConfig] = None,
                 ds: float = 5.0):
        
        super().__init__(vehicle_model, track_model, ers_config)
        self.ds = ds
        self.ecms = ecms_config or ECMSConfig()
        
        # Discretization
        self.N = int(track_model.total_length / ds)
        self.s_grid = np.linspace(0, track_model.total_length, self.N + 1)
        
        # Adaptive state
        self._soc_error_integral = 0.0
        self._prev_P_ers = 0.0
        
    @property
    def name(self) -> str:
        return "ECMS"
    
    def solve(self,
              v_limit_profile: np.ndarray,
              initial_soc: float = 0.5,
              final_soc_min: float = 0.3,
              energy_limit: float = 4e6) -> OptimalTrajectory:
        """
        Solve ERS deployment using ECMS with terminal SOC enforcement.
        """
        
        self._log(f"Solving with ECMS (λ_base={self.ecms.lambda_base:.2f})...")
        start_time = time.time()
        
        # Reset adaptive state
        self._soc_error_integral = 0.0
        self._prev_P_ers = 0.0
        
        # Ensure v_limit is the right size
        if len(v_limit_profile) != self.N + 1:
            v_limit_profile = np.interp(
                self.s_grid,
                np.linspace(0, self.track.total_length, len(v_limit_profile)),
                v_limit_profile
            )
        
        # SOC target profile is not required; terminal SOC is enforced via λ shooting.
        soc_target = None
        
        # Get track data
        track_data = self.track.track_data
        gradient_arr = self._ensure_array_size(track_data.gradient, self.N)
        radius_arr = self._ensure_array_size(track_data.radius, self.N)
        
        # Initialize storage
        v_opt = np.zeros(self.N + 1)
        soc_opt = np.zeros(self.N + 1)
        P_ers_opt = np.zeros(self.N)
        throttle_opt = np.zeros(self.N)
        brake_opt = np.zeros(self.N)
        lambda_history = np.zeros(self.N)
        
        # Initial conditions
        v_opt[0] = min(v_limit_profile[0], 50.0)
        soc_opt[0] = initial_soc
        
        # Energy tracking
        total_energy_deployed = 0.0
        remaining_energy_budget = energy_limit
        
        # Get constraints
        cons = self.vehicle.get_constraints()
        
        
        def _run_once(lambda_base: float):
            """Run a single forward ECMS pass with a fixed λ (no SOC reference profile)."""
            # Backup and override λ settings for shooting
            _lambda_base_prev = self.ecms.lambda_base
            _adaptive_prev = self.ecms.use_adaptive_lambda
            self.ecms.lambda_base = float(lambda_base)
            # For terminal shooting, keep λ fixed (adaptive needs a reference to track)
            self.ecms.use_adaptive_lambda = False

            # Reset adaptive state
            self._soc_error_integral = 0.0
            self._prev_P_ers = 0.0

            # Initialize storage (fresh each run)
            v_opt = np.zeros(self.N + 1)
            soc_opt = np.zeros(self.N + 1)
            P_ers_opt = np.zeros(self.N)
            throttle_opt = np.zeros(self.N)
            brake_opt = np.zeros(self.N)
            lambda_history = np.zeros(self.N)

            # Initial conditions
            v_opt[0] = min(v_limit_profile[0], 50.0)
            soc_opt[0] = initial_soc

            # Energy tracking
            total_energy_deployed = 0.0
            remaining_energy_budget = energy_limit

            # Forward simulation
            for k in range(self.N):
                v_k = v_opt[k]
                soc_k = soc_opt[k]

                gradient_k = float(gradient_arr[min(k, len(gradient_arr)-1)])
                radius_k = float(radius_arr[min(k, len(radius_arr)-1)])
                radius_k = max(radius_k, 15.0)

                v_limit_k = v_limit_profile[k]
                v_limit_next = v_limit_profile[min(k+1, self.N)]

                progress = k / self.N

                # Fixed λ (no SOC reference)
                lambda_k = self._compute_lambda(
                    soc_k=soc_k,
                    soc_target_k=None,
                    progress=progress,
                    remaining_budget=remaining_energy_budget,
                    energy_limit=energy_limit,
                    final_soc_min=final_soc_min,
                    k=k
                )
                lambda_history[k] = lambda_k

                P_ers_k, throttle_k, brake_k = self._ecms_decision(
                    v_k=v_k,
                    soc_k=soc_k,
                    v_limit_k=v_limit_k,
                    v_limit_next=v_limit_next,
                    gradient_k=gradient_k,
                    radius_k=radius_k,
                    lambda_k=lambda_k,
                    remaining_budget=remaining_energy_budget,
                    soc_target_k=None,
                    progress=progress,
                    final_soc_min=final_soc_min
                )

                P_ers_opt[k] = P_ers_k
                throttle_opt[k] = throttle_k
                brake_opt[k] = brake_k

                # Simulate next step
                v_next, soc_next, energy_used, _ = self._simulate_step(
                    v_k=v_k,
                    soc_k=soc_k,
                    throttle=throttle_k,
                    brake=brake_k,
                    P_ers=P_ers_k,
                    gradient=gradient_k,
                    radius=radius_k
                )

                # Enforce velocity limit
                v_next = min(v_next, v_limit_next)

                v_opt[k+1] = v_next
                soc_opt[k+1] = soc_next

                if energy_used > 0:
                    total_energy_deployed += energy_used
                    remaining_energy_budget = max(0, remaining_energy_budget - energy_used)

            # Restore settings
            self.ecms.lambda_base = _lambda_base_prev
            self.ecms.use_adaptive_lambda = _adaptive_prev

            # Compute results
            _, lap_time = self._compute_lap_time(self.s_grid, v_opt)
            final_soc = soc_opt[-1]
            return v_opt, soc_opt, P_ers_opt, throttle_opt, brake_opt, lambda_history, lap_time, final_soc

        # === Terminal SOC shooting on λ (standalone ECMS) ===
        target_final_soc = float(final_soc_min)

        if self.ecms.enforce_terminal_soc:
            # Bracket λ so that soc_end crosses target_final_soc
            lam_lo = 0.0
            lam_hi = 10.0

            _, _, _, _, _, _, lap_lo, soc_lo = _run_once(lam_lo)
            _, _, _, _, _, _, lap_hi, soc_hi = _run_once(lam_hi)

            # Increase hi until we conserve enough SOC (or cap out)
            grow = 0
            while soc_hi < target_final_soc and lam_hi < 200.0 and grow < 20:
                lam_hi *= 2.0
                _, _, _, _, _, _, lap_hi, soc_hi = _run_once(lam_hi)
                grow += 1

            # If even huge λ cannot meet target, keep best feasible (max SOC)
            if soc_hi < target_final_soc:
                v_opt, soc_opt, P_ers_opt, throttle_opt, brake_opt, lambda_history, lap_time, final_soc = (
                    _run_once(lam_hi)
                )
            else:
                # Bisection to hit target final SOC
                best = None
                for _ in range(25):
                    lam_mid = 0.5 * (lam_lo + lam_hi)
                    v_m, soc_m, P_m, th_m, br_m, lam_hist_m, lap_m, soc_end_m = _run_once(lam_mid)

                    # We want soc_end >= target; push λ down as much as possible for lap time
                    if soc_end_m >= target_final_soc:
                        best = (v_m, soc_m, P_m, th_m, br_m, lam_hist_m, lap_m, soc_end_m, lam_mid)
                        lam_hi = lam_mid
                    else:
                        lam_lo = lam_mid

                if best is None:
                    v_opt, soc_opt, P_ers_opt, throttle_opt, brake_opt, lambda_history, lap_time, final_soc = _run_once(lam_hi)
                else:
                    v_opt, soc_opt, P_ers_opt, throttle_opt, brake_opt, lambda_history, lap_time, final_soc, lam_star = best
                    self._log(f"  λ* (shooting): {lam_star:.3f}")
        else:
            v_opt, soc_opt, P_ers_opt, throttle_opt, brake_opt, lambda_history, lap_time, final_soc = _run_once(self.ecms.lambda_base)

        # Forward sim done; arrays now filled for reporting below
# Compute results
        t_opt, lap_time = self._compute_lap_time(self.s_grid, v_opt)
        energy_deployed, energy_recovered = self._compute_energy_totals(P_ers_opt, v_opt)
        
        solve_time = time.time() - start_time
        
        # Check constraints
        final_soc = soc_opt[-1]
        status = 'optimal' if final_soc >= final_soc_min else 'suboptimal'
        
        self._log(f"✓ Solved in {solve_time:.3f}s")
        self._log(f"  Lap time: {lap_time:.3f}s")
        self._log(f"  Final SOC: {final_soc*100:.1f}% (min: {final_soc_min*100:.1f}%)")
        self._log(f"  Energy deployed: {energy_deployed/1e6:.3f} MJ")
        self._log(f"  λ range: {lambda_history.min():.2f} - {lambda_history.max():.2f}")
        
        return OptimalTrajectory(
            s=self.s_grid,
            ds=self.ds,
            n_points=self.N + 1,
            v_opt=v_opt,
            soc_opt=soc_opt,
            P_ers_opt=P_ers_opt,
            throttle_opt=throttle_opt,
            brake_opt=brake_opt,
            t_opt=t_opt,
            lap_time=lap_time,
            energy_deployed=energy_deployed,
            energy_recovered=energy_recovered,
            solve_time=solve_time,
            solver_status=status,
            solver_name=self.name,
        )
    
    def _ensure_array_size(self, arr: np.ndarray, min_size: int) -> np.ndarray:
        """Ensure array is at least min_size."""
        if len(arr) < min_size:
            return np.resize(arr, min_size)
        return arr
    
    def _compute_lambda(self,
                        soc_k: float,
                        soc_target_k: float,
                        progress: float,
                        remaining_budget: float,
                        energy_limit: float,
                        final_soc_min: float,
                        k: int) -> float:
        """
        Compute equivalence factor with adaptive adjustment.
        
        The λ parameter balances:
        - Low λ: Encourage ERS deployment (faster but depletes battery)
        - High λ: Conserve battery (slower but maintains SOC)
        
        Adaptation strategies:
        1. SOC tracking: Increase λ when below target, decrease when above
        2. Terminal penalty: Strongly increase λ near end if SOC is low
        3. Budget management: Increase λ when budget is nearly exhausted
        """
        lambda_base = self.ecms.lambda_base
        
        if not self.ecms.use_adaptive_lambda:
            return lambda_base
        
        # === 1. SOC Tracking (PI Controller) ===
        soc_error = soc_target_k - soc_k  # Positive if we're below target
        
        # Proportional term
        lambda_p = self.ecms.K_p * soc_error
        
        # Integral term
        lambda_i = self.ecms.K_i * self._soc_error_integral
        
        lambda_adaptive = lambda_base + lambda_p + lambda_i
        
        # === 2. Terminal SOC Enforcement ===
        if self.ecms.enforce_terminal_soc and progress > 0.7:
            # How much SOC do we need to preserve?
            remaining_distance_frac = 1.0 - progress
            
            # Project final SOC if we continue current trend
            # (simplified: assume linear depletion)
            soc_rate = (soc_k - self.ecms.K_p) if k > 10 else 0
            
            # If we're at risk of missing final SOC target
            margin_to_min = soc_k - final_soc_min
            if margin_to_min < 0.15:  # Less than 15% margin
                # Exponentially increase λ as we approach end with low SOC
                urgency = (1.0 - margin_to_min / 0.15) * (progress - 0.7) / 0.3
                lambda_adaptive += self.ecms.terminal_soc_weight * urgency
        
        # === 3. Budget Management ===
        if energy_limit > 0:
            budget_fraction = remaining_budget / energy_limit
            if budget_fraction < 0.1:
                # Nearly out of budget - heavily penalize deployment
                lambda_adaptive *= 3.0
        
        # === 4. Clamp to reasonable range ===
        return np.clip(lambda_adaptive, 0.5, 15.0)
    
    def _ecms_decision(self,
                       v_k: float,
                       soc_k: float,
                       v_limit_k: float,
                       v_limit_next: float,
                       gradient_k: float,
                       radius_k: float,
                       lambda_k: float,
                       remaining_budget: float,
                       soc_target_k: float,
                       progress: float,
                       final_soc_min: float) -> Tuple[float, float, float]:
        """
        Make instantaneous ERS decision using ECMS criterion.
        
        ECMS Hamiltonian (simplified for lap time):
        
            H ≈ -Δv/Δs + λ · P_battery / E_cap
            
        where:
        - Δv/Δs is the velocity improvement from ERS
        - P_battery is battery power flow (positive = depleting)
        - λ is the equivalence factor
        
        We discretize P_ers and find the value minimizing H.
        """
        veh = self.vehicle.vehicle
        ers = self.vehicle.ers
        cons = self.vehicle.get_constraints()
        
        v_safe = max(v_k, 5.0)
        dt = self.ds / v_safe
        
        # Compute forces at current state
        q = 0.5 * veh.rho_air * v_k**2
        F_drag = q * veh.c_w_a
        F_downforce = q * (veh.c_z_a_f + veh.c_z_a_r)
        F_normal = veh.mass * veh.g * np.cos(gradient_k) + F_downforce
        F_roll = veh.cr * F_normal
        F_gravity = veh.mass * veh.g * np.sin(gradient_k)
        F_resist = F_drag + F_roll + F_gravity
        
        # Grip limit
        F_grip_max = veh.mu_longitudinal * F_normal
        
        # Required acceleration to track velocity limit
        dv_desired = v_limit_next - v_k
        a_desired = dv_desired / dt
        F_desired = veh.mass * a_desired + F_resist
        
        # Determine mode
        if F_desired > 0:
            mode = 'accel'
        else:
            mode = 'brake'
        
        # Generate candidate ERS powers
        P_min = cons['P_ers_min']  # -120 kW (harvest)
        P_max = cons['P_ers_max']  # +120 kW (deploy)
        
        # Limit deployment by remaining budget
        max_deploy_this_step = remaining_budget / dt if dt > 0 else P_max
        P_max_budget = min(P_max, max_deploy_this_step)
        
        # Limit by SOC constraints
        if soc_k <= cons['soc_min'] + 0.02:
            P_max_budget = 0  # Can't deploy if SOC too low
        if soc_k >= cons['soc_max'] - 0.02:
            P_min = 0  # Can't harvest if SOC too high
        
        # Create candidate array
        n_candidates = self.ecms.n_ers_candidates
        P_candidates = np.linspace(P_min, P_max_budget, n_candidates)
        
        best_cost = float('inf')
        best_P_ers = 0.0
        best_throttle = 0.0
        best_brake = 0.0
        
        for P_ers in P_candidates:
            # Compute resulting acceleration and controls
            if mode == 'accel' or P_ers >= 0:
                # Propulsion case
                P_ers_effective = max(P_ers, 0) * ers.deployment_efficiency
                
                # ICE power needed
                P_total_needed = max(F_desired * v_safe, 0)
                P_ice_needed = max(0, P_total_needed - P_ers_effective)
                
                # Throttle
                throttle = np.clip(P_ice_needed / veh.pow_max_ice, 0, 1)
                brake = 0.0
                
                # Actual propulsion
                P_prop = throttle * veh.pow_max_ice + P_ers_effective
                F_prop = min(P_prop / v_safe, F_grip_max)
                
            else:
                # Braking/harvesting case
                throttle = 0.0
                
                F_brake_needed = -F_desired  # Positive braking force
                P_harvest = -P_ers  # Positive harvest power
                F_regen = P_harvest / v_safe
                
                F_mech_needed = max(0, F_brake_needed - F_regen)
                brake = np.clip(F_mech_needed / veh.max_brake_force, 0, 1)
                
                F_prop = 0
            
            # Compute resulting velocity change
            if P_ers >= 0:
                F_net = F_prop - F_resist
            else:
                F_brake_total = brake * veh.max_brake_force + (-P_ers) / v_safe
                F_net = -F_resist - F_brake_total
            
            a_actual = F_net / veh.mass
            v_next = v_k + a_actual * dt
            v_next = np.clip(v_next, cons['v_min'], v_limit_next)
            
            # === ECMS Cost Function ===
            # 
            # Cost = (time cost) + λ · (battery cost)
            #
            # Time cost: We want high velocity → low dt/ds
            # Approximate as: -(v_next - v_k) / v_limit_next  (velocity improvement)
            #
            # Battery cost: P_battery / E_capacity
            # P_battery > 0 when depleting (deployment)
            
            # Time benefit (negative cost = good)
            velocity_benefit = (v_next - v_k) / max(v_limit_next, 30.0)
            
            # Battery cost
            if P_ers >= 0:
                P_battery = P_ers / ers.deployment_efficiency
            else:
                P_battery = P_ers * ers.recovery_efficiency  # Negative (benefit)
            
            battery_cost = P_battery / ers.battery_capacity
            
            # Total ECMS cost
            cost = -velocity_benefit + lambda_k * battery_cost
            
            # Additional penalty for being far from SOC target
            soc_next = soc_k - P_battery * dt / ers.battery_capacity
            soc_penalty = 0.1 * (soc_next - soc_target_k)**2
            cost += soc_penalty
            
            if cost < best_cost:
                best_cost = cost
                best_P_ers = P_ers
                best_throttle = throttle
                best_brake = brake
        
        return best_P_ers, best_throttle, best_brake
    
    def _simulate_step(self,
                       v_k: float,
                       soc_k: float,
                       P_ers: float,
                       throttle: float,
                       brake: float,
                       gradient: float,
                       radius: float,
                       v_limit_next: float) -> Tuple[float, float, float]:
        """Simulate one step of vehicle dynamics."""
        veh = self.vehicle.vehicle
        ers = self.vehicle.ers
        cons = self.vehicle.get_constraints()
        
        v_safe = max(v_k, 5.0)
        dt = self.ds / v_safe
        
        # Forces
        q = 0.5 * veh.rho_air * v_k**2
        F_drag = q * veh.c_w_a
        F_downforce = q * (veh.c_z_a_f + veh.c_z_a_r)
        F_normal = veh.mass * veh.g * np.cos(gradient) + F_downforce
        F_roll = veh.cr * F_normal
        F_gravity = veh.mass * veh.g * np.sin(gradient)
        F_grip_max = veh.mu_longitudinal * F_normal
        
        # Propulsion
        P_ice = throttle * veh.pow_max_ice
        P_ers_deploy = max(P_ers, 0) * ers.deployment_efficiency
        F_prop = min((P_ice + P_ers_deploy) / v_safe, F_grip_max)
        
        # Braking
        F_brake_mech = brake * veh.max_brake_force
        F_brake_regen = max(-P_ers, 0) / v_safe
        F_brake_total = min(F_brake_mech + F_brake_regen, F_grip_max)
        
        # Dynamics
        F_net = F_prop - F_drag - F_roll - F_gravity - F_brake_total
        dv_dt = F_net / veh.mass
        
        v_next = v_k + dv_dt * dt
        v_next = np.clip(v_next, cons['v_min'], v_limit_next)
        
        # Battery dynamics
        if P_ers >= 0:
            P_battery = P_ers / ers.deployment_efficiency
        else:
            P_battery = P_ers * ers.recovery_efficiency
        
        dsoc = -P_battery * dt / ers.battery_capacity
        soc_next = np.clip(soc_k + dsoc, cons['soc_min'], cons['soc_max'])
        
        # Energy deployed
        energy_deployed = max(P_ers, 0) * dt
        
        return v_next, soc_next, energy_deployed


class AdaptiveECMSSolver(ECMSSolver):
    """
    A-ECMS: ECMS with predictive lookahead.
    
    Looks ahead at upcoming track geometry to anticipate:
    - Braking zones (don't deploy right before, will harvest there)
    - Straights (good place to deploy)
    - Tight corners (harvesting opportunity)
    """
    
    def __init__(self, 
                 vehicle_model, 
                 track_model, 
                 ers_config,
                 ecms_config: Optional[ECMSConfig] = None,
                 ds: float = 5.0):
        
        if ecms_config is None:
            ecms_config = ECMSConfig(use_predictive=True, lookahead_distance=500.0)
        ecms_config.use_predictive = True
            
        super().__init__(vehicle_model, track_model, ers_config, ecms_config, ds)
        
        # Precompute track characteristics for lookahead
        self._precompute_track_features()
        
    @property
    def name(self) -> str:
        return "A-ECMS"
    
    def _precompute_track_features(self):
        """Precompute track features for fast lookahead."""
        track_data = self.track.track_data
        n = len(track_data.radius)
        
        # Classify each point
        self._is_braking_zone = np.zeros(n, dtype=bool)
        self._is_straight = np.zeros(n, dtype=bool)
        self._is_corner = np.zeros(n, dtype=bool)
        
        lookahead_pts = int(self.ecms.lookahead_distance / self.ds)
        
        for i in range(n):
            r = track_data.radius[i]
            
            # Current classification
            self._is_straight[i] = r > 500
            self._is_corner[i] = r < 100
            
            # Look ahead for braking zones
            future_radii = []
            for j in range(1, min(lookahead_pts, n - i)):
                future_radii.append(track_data.radius[i + j])
            
            if future_radii:
                min_future_r = min(future_radii)
                # Braking zone: currently fast, tight corner coming
                if r > 200 and min_future_r < 80:
                    self._is_braking_zone[i] = True
    
    def _compute_lambda(self,
                        soc_k: float,
                        soc_target_k: float,
                        progress: float,
                        remaining_budget: float,
                        energy_limit: float,
                        final_soc_min: float,
                        k: int) -> float:
        """Compute λ with predictive adjustment."""
        
        # Get base lambda from parent
        lambda_base = super()._compute_lambda(
            soc_k, soc_target_k, progress, remaining_budget,
            energy_limit, final_soc_min, k
        )
        
        if not self.ecms.use_predictive:
            return lambda_base
        
        # Adjust based on upcoming track
        idx = min(k, len(self._is_braking_zone) - 1)
        
        if self._is_braking_zone[idx]:
            # Braking zone ahead - increase λ to save energy
            # (we'll harvest during braking anyway)
            lambda_base *= 1.4
        
        if self._is_straight[idx]:
            # On a straight - good place to deploy
            # Decrease λ to encourage deployment
            lambda_base *= 0.85
        
        if self._is_corner[idx]:
            # In a corner - likely low speed, less benefit from ERS
            # Slightly increase λ
            lambda_base *= 1.1
        
        return np.clip(lambda_base, 0.5, 15.0)


class TelemetryECMSSolver(ECMSSolver):
    """
    T-ECMS: ECMS calibrated from optimal NLP solution.
    
    Learns the λ profile that would reproduce NLP decisions.
    Useful for:
    - Understanding what λ schedule is optimal
    - Fast re-runs with learned parameters
    - Analyzing structure of optimal solution
    """
    
    def __init__(self, 
                 vehicle_model, 
                 track_model, 
                 ers_config,
                 ecms_config: Optional[ECMSConfig] = None,
                 ds: float = 5.0):
        
        super().__init__(vehicle_model, track_model, ers_config, ecms_config, ds)
        self._lambda_profile: Optional[np.ndarray] = None
        self._use_learned_profile = False
        
    @property
    def name(self) -> str:
        return "T-ECMS"
    
    def calibrate_from_nlp(self, nlp_trajectory: OptimalTrajectory):
        """
        Learn λ profile from NLP solution.
        
        For each point, find λ that would produce similar P_ers decision.
        """
        self._log("Calibrating λ profile from NLP solution...")
        
        n = len(nlp_trajectory.P_ers_opt)
        self._lambda_profile = np.zeros(n)
        
        # For each point, estimate what λ would give similar behavior
        for k in range(n):
            P_ers_nlp = nlp_trajectory.P_ers_opt[k]
            soc_k = nlp_trajectory.soc_opt[k]
            v_k = nlp_trajectory.v_opt[k]
            
            # Heuristic: λ relates to marginal value of energy
            # When NLP deploys heavily → low λ
            # When NLP harvests or doesn't deploy → high λ
            
            P_max = self.vehicle.get_constraints()['P_ers_max']
            
            if P_ers_nlp > 0.8 * P_max:
                # Heavy deployment - low λ
                self._lambda_profile[k] = 1.0
            elif P_ers_nlp > 0.3 * P_max:
                # Moderate deployment
                self._lambda_profile[k] = 2.0
            elif P_ers_nlp > 0:
                # Light deployment
                self._lambda_profile[k] = 3.0
            elif P_ers_nlp > -0.3 * P_max:
                # Light harvest or neutral
                self._lambda_profile[k] = 4.0
            else:
                # Heavy harvest
                self._lambda_profile[k] = 2.5  # Harvesting is good, moderate λ
        
        # Smooth the profile
        from scipy.ndimage import uniform_filter1d
        self._lambda_profile = uniform_filter1d(self._lambda_profile, size=10)
        
        self._use_learned_profile = True
        
        self._log(f"  Calibrated λ range: {self._lambda_profile.min():.2f} - {self._lambda_profile.max():.2f}")
    
    def _compute_lambda(self,
                        soc_k: float,
                        soc_target_k: float,
                        progress: float,
                        remaining_budget: float,
                        energy_limit: float,
                        final_soc_min: float,
                        k: int) -> float:
        """Use learned profile if available."""
        
        if self._use_learned_profile and self._lambda_profile is not None:
            idx = min(k, len(self._lambda_profile) - 1)
            lambda_learned = self._lambda_profile[idx]
            
            # Still apply SOC correction on top
            soc_error = soc_target_k - soc_k
            lambda_correction = self.ecms.K_p * soc_error * 0.5  # Reduced gain
            
            return np.clip(lambda_learned + lambda_correction, 0.5, 15.0)
        
        return super()._compute_lambda(
            soc_k, soc_target_k, progress, remaining_budget,
            energy_limit, final_soc_min, k
        )