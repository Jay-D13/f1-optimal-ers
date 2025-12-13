"""
Spatial-Domain NLP Solver for ERS Optimization
Offline optimizer using direct collocation in spatial domain

Problem formulation:
    minimize    T = ∑(ds / v[k])              (lap time)
    subject to: 
        v[k] ≤ v_limit[k]                     (grip limit from Forward-Backward)
        v dynamics from ERS + ICE power
        SOC dynamics from ERS power
        ∑(P_ers_deploy) ≤ 4 MJ                (regulatory limit)
        SOC_min ≤ SOC ≤ SOC_max               (battery limits)
"""

import numpy as np
import casadi as ca
import time

from solvers import BaseSolver, OptimalTrajectory


class SpatialNLPSolver(BaseSolver):
    """
    Finds the globally (offline) optimal ERS deployment strategy for a single lap
    """
    
    def __init__(self, vehicle_model, track_model, ers_config, ds: float = 5.0):
        super().__init__(vehicle_model, track_model, ers_config)
        self.ds = ds
        
        # Discretization
        self.N = int(track_model.total_length / ds)
        self.s_grid = np.linspace(0, track_model.total_length, self.N + 1)
        
        # CasADi problem storage
        self.opti = None
        self.solution = None
        
    @property
    def name(self) -> str:
        return "SpatialNLP"
    
    def solve(self,
              v_limit_profile: np.ndarray,
              initial_soc: float = 0.5,
              final_soc_min: float = 0.3,
              energy_limit: float = 4e6) -> OptimalTrajectory:
        """
        velocity is essentially FIXED by grip limits
        We're really just optimizing WHEN to deploy ERS for small speed gains
        """
        
        self._log(f"Setting up NLP with {self.N} nodes...")
        
        start_time = time.time()
        
        # Ensure v_limit is the right size
        if len(v_limit_profile) != self.N + 1:
            v_limit_profile = np.interp(
                self.s_grid, 
                np.linspace(0, self.track.total_length, len(v_limit_profile)),
                v_limit_profile
            )
        
        # Build and solve NLP
        trajectory = self._build_and_solve(
            v_limit_profile=v_limit_profile,
            initial_soc=initial_soc,
            final_soc_min=final_soc_min,
            energy_limit=energy_limit
        )
        
        trajectory.solve_time = time.time() - start_time
        
        self._log(f"✓ Solved in {trajectory.solve_time:.2f}s")
        self._log(f"  Lap time: {trajectory.lap_time:.3f}s")
        self._log(f"  Status: {trajectory.solver_status}")
        
        return trajectory
    
    def _build_and_solve(self,
                         v_limit_profile: np.ndarray,
                         initial_soc: float,
                         final_soc_min: float,
                         energy_limit: float) -> OptimalTrajectory:
        
        opti = ca.Opti()
        
        # Get constraints from vehicle model
        cons = self.vehicle.get_constraints()
        
        # Get track data
        track_data = self.track.track_data
        gradient_arr = track_data.gradient
        radius_arr = track_data.radius
        
        # Ensure arrays are right size
        if len(gradient_arr) < self.N:
            gradient_arr = np.resize(gradient_arr, self.N)
        if len(radius_arr) < self.N:
            radius_arr = np.resize(radius_arr, self.N)
        
        # =====================================================
        # DECISION VARIABLES
        # =====================================================
        
        # States at each node
        V = opti.variable(self.N + 1)       # Velocity (m/s)
        SOC = opti.variable(self.N + 1)     # State of charge (0-1)
        
        # Controls between nodes
        P_ERS = opti.variable(self.N)       # ERS power (W), + = deploy
        THROTTLE = opti.variable(self.N)    # Throttle (0-1)
        BRAKE = opti.variable(self.N)       # Brake (0-1)
        
        # =====================================================
        # OBJECTIVE: Minimize lap time
        # =====================================================
        
        lap_time = 0
        for k in range(self.N):
            v_avg = 0.5 * (V[k] + V[k+1])
            v_safe = ca.fmax(v_avg, 5.0)
            lap_time += self.ds / v_safe
        
        opti.minimize(lap_time)
        
        # =====================================================
        # DYNAMICS CONSTRAINTS
        # =====================================================
        
        # Get vehicle parameters
        veh = self.vehicle.vehicle
        ers = self.vehicle.ers
        
        for k in range(self.N):
            # Track parameters for this segment
            gradient_k = float(gradient_arr[min(k, len(gradient_arr)-1)])
            radius_k = float(radius_arr[min(k, len(radius_arr)-1)])
            radius_k = max(radius_k, 15.0)
            gradient_k = np.clip(gradient_k, -0.15, 0.15)
            
            # Current state
            v_k = V[k]
            soc_k = SOC[k]
            
            # Current control  
            P_ers_k = P_ERS[k]
            throttle_k = THROTTLE[k]
            brake_k = BRAKE[k]
            
            # --- Velocity dynamics ---
            # dv/ds = (1/v) * (F_net / m)
            
            # Aerodynamic forces
            q = 0.5 * veh.rho_air * v_k**2
            F_drag = q * veh.cd * veh.frontal_area
            F_downforce = q * veh.cl * veh.frontal_area
            
            # Normal force and friction
            F_normal = veh.mass * veh.g * ca.cos(gradient_k) + F_downforce
            F_roll = veh.cr * F_normal
            F_gravity = veh.mass * veh.g * ca.sin(gradient_k)
            
            # Propulsion: ICE + ERS deployment
            P_ice = throttle_k * veh.max_ice_power
            P_ers_deploy = ca.fmax(P_ers_k, 0)
            P_total = P_ice + P_ers_deploy
            
            v_safe = ca.fmax(v_k, 5.0)
            F_prop = P_total / v_safe
            
            # Grip limit on propulsion
            F_grip_max = veh.mu_longitudinal * F_normal
            F_prop = ca.fmin(F_prop, F_grip_max)
            
            # Braking: mechanical + ERS harvest
            F_brake_mech = brake_k * veh.max_brake_force
            P_ers_harvest = ca.fmin(P_ers_k, 0)  # Negative
            F_brake_regen = -P_ers_harvest / v_safe
            F_brake_total = ca.fmin(F_brake_mech + F_brake_regen, F_grip_max)
            
            # Net force and acceleration
            F_net = F_prop - F_drag - F_roll - F_gravity - F_brake_total
            dv_dt = F_net / veh.mass
            
            # Convert to spatial: dv/ds = dv/dt * dt/ds = dv/dt / v
            dv_ds = dv_dt / v_safe
            
            # Euler integration
            opti.subject_to(V[k+1] == V[k] + dv_ds * self.ds)
            
            # --- SOC dynamics ---
            # P_ers > 0: deploy -> SOC decreases
            # P_ers < 0: harvest -> SOC increases
            
            # Smooth efficiency transition
            sigma = 0.5 * (1 + ca.tanh(P_ers_k / 1000.0))
            eta_d = ers.deployment_efficiency
            eta_r = ers.recovery_efficiency
            
            P_battery = sigma * (P_ers_k / eta_d) + (1 - sigma) * (P_ers_k * eta_r)
            
            # dsoc/ds = (dsoc/dt) / v = (-P_battery / E_cap) / v
            dsoc_ds = -P_battery / (ers.battery_capacity * v_safe)
            
            opti.subject_to(SOC[k+1] == SOC[k] + dsoc_ds * self.ds)
        
        # =====================================================
        # STATE CONSTRAINTS
        # =====================================================
        
        for k in range(self.N + 1):
            # Velocity bounds
            opti.subject_to(V[k] >= cons['v_min'])
            opti.subject_to(V[k] <= v_limit_profile[k] * 1.005)  # Small margin
            
            # SOC bounds (hard - battery safety)
            opti.subject_to(SOC[k] >= cons['soc_min'])
            opti.subject_to(SOC[k] <= cons['soc_max'])
        
        # =====================================================
        # CONTROL CONSTRAINTS
        # =====================================================
        
        for k in range(self.N):
            # ERS power limits
            opti.subject_to(P_ERS[k] >= cons['P_ers_min'])
            opti.subject_to(P_ERS[k] <= cons['P_ers_max'])
            
            # Throttle/brake limits
            opti.subject_to(THROTTLE[k] >= 0)
            opti.subject_to(THROTTLE[k] <= 1)
            opti.subject_to(BRAKE[k] >= 0)
            opti.subject_to(BRAKE[k] <= 1)
            
            # No simultaneous throttle and heavy brake
            opti.subject_to(THROTTLE[k] * BRAKE[k] <= 0.1)
        
        # =====================================================
        # BOUNDARY CONDITIONS
        # =====================================================
        
        # Initial SOC
        opti.subject_to(SOC[0] == initial_soc)
        
        # Final SOC constraint
        opti.subject_to(SOC[self.N] >= final_soc_min)
        
        # Initial velocity (reasonable start)
        v_start = min(50.0, v_limit_profile[0])
        opti.subject_to(V[0] >= v_start * 0.9)
        opti.subject_to(V[0] <= v_start * 1.1)
        
        # =====================================================
        # ENERGY LIMIT CONSTRAINT
        # =====================================================
        
        total_deployment = 0
        for k in range(self.N):
            v_avg = ca.fmax(0.5 * (V[k] + V[k+1]), 5.0)
            dt_segment = self.ds / v_avg
            deployment_k = ca.fmax(P_ERS[k], 0) * dt_segment
            total_deployment += deployment_k
        
        opti.subject_to(total_deployment <= energy_limit)
        
        # =====================================================
        # SOLVER CONFIGURATION
        # =====================================================
        
        opts = {
            'ipopt.max_iter': 10000,
            'ipopt.print_level': 3,
            'print_time': 1,
            'ipopt.tol': 1e-5,
            'ipopt.acceptable_tol': 1e-4,
            'ipopt.acceptable_iter': 10,
            'ipopt.linear_solver': 'ma97',  # use 'mumps' if no license
            'ipopt.warm_start_init_point': 'yes',
            'ipopt.mu_strategy': 'adaptive',
            'ipopt.nlp_scaling_method': 'gradient-based',
        }
        
        opti.solver('ipopt', opts)
        
        # =====================================================
        # INITIAL GUESS
        # =====================================================
        
        self._set_initial_guess(opti, V, SOC, P_ERS, THROTTLE, BRAKE,
                                v_limit_profile, initial_soc)
        
        # =====================================================
        # SOLVE
        # =====================================================
        
        try:
            sol = opti.solve()
            status = 'optimal'
            
            v_opt = sol.value(V)
            soc_opt = sol.value(SOC)
            P_ers_opt = sol.value(P_ERS)
            throttle_opt = sol.value(THROTTLE)
            brake_opt = sol.value(BRAKE)
            
        except Exception as e:
            self._log(f"⚠ Solver warning: {e}")
            self._log("  Extracting best available solution...")
            status = 'suboptimal'
            
            try:
                v_opt = opti.debug.value(V)
                soc_opt = opti.debug.value(SOC)
                P_ers_opt = opti.debug.value(P_ERS)
                throttle_opt = opti.debug.value(THROTTLE)
                brake_opt = opti.debug.value(BRAKE)
            except:
                raise RuntimeError("Failed to extract any solution")
        
        # =====================================================
        # BUILD RESULT
        # =====================================================
        
        t_opt, lap_time = self._compute_lap_time(self.s_grid, v_opt)
        energy_deployed, energy_recovered = self._compute_energy_totals(P_ers_opt, v_opt)
        
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
            solve_time=0.0,  # Set by caller
            solver_status=status,
            solver_name=self.name,
        )
    
    def _set_initial_guess(self, opti, V, SOC, P_ERS, THROTTLE, BRAKE,
                           v_limit_profile, initial_soc):
        """initial guess for faster convergence."""
        
        # Velocity: 90% of limit (conservative but fast)
        v_init = 0.90 * v_limit_profile
        v_init = np.clip(v_init, 20, 90)
        opti.set_initial(V, v_init)
        
        # SOC: gradual depletion
        soc_init = np.linspace(initial_soc, initial_soc - 0.15, self.N + 1)
        soc_init = np.clip(soc_init, 0.25, 0.85)
        opti.set_initial(SOC, soc_init)
        
        # Controls: moderate throttle, no ERS initially
        opti.set_initial(P_ERS, np.zeros(self.N))
        opti.set_initial(THROTTLE, 0.7 * np.ones(self.N))
        opti.set_initial(BRAKE, np.zeros(self.N))
        
        self._log(f"Initial guess: v={v_init.min():.0f}-{v_init.max():.0f} m/s")

