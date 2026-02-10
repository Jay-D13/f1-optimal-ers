"""
Gymnasium environment for ERS policy training.

Action:
    a in [-1, 1]
    a > 0 -> deploy ERS power
    a < 0 -> harvest ERS power

Throttle and brake are supplied by a baseline tracking controller so this
minimal environment focuses on ERS decision quality first.
"""

from __future__ import annotations

import numpy as np

from models import F1TrackModel, VehicleDynamicsModel
from strategies.baselines import TrackingStrategy
from rl.common import (
    action_to_ers_power,
    build_observation,
    get_reference_arrays,
    sample_reference_speed,
)

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:
    gym = None
    spaces = None
    _GYM_IMPORT_ERROR = exc


if gym is None:

    class ERSLapEnv:  # pragma: no cover - runtime guard when optional dep missing
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "gymnasium is required for ERSLapEnv. "
                "Install RL dependencies first (gymnasium, torch)."
            ) from _GYM_IMPORT_ERROR


else:

    class ERSLapEnv(gym.Env):
        """Single-lap ERS training environment."""

        metadata = {"render_modes": []}

        def __init__(
            self,
            vehicle_model: VehicleDynamicsModel,
            track_model: F1TrackModel,
            reference_profile,
            dt: float = 0.1,
            initial_soc: float = 0.5,
            initial_velocity: float = 30.0,
            final_soc_target: float = 0.3,
            max_time: float = 220.0,
        ):
            super().__init__()

            self.vehicle_model = vehicle_model
            self.track = track_model
            self.ers = vehicle_model.ers
            self.dt = float(dt)
            self.initial_soc = float(initial_soc)
            self.initial_velocity = float(initial_velocity)
            self.final_soc_target = float(final_soc_target)
            self.max_time = float(max_time)
            self.max_steps = int(np.ceil(self.max_time / max(self.dt, 1e-4)))

            self.constraints = self.vehicle_model.get_constraints()
            self.dynamics_func = self.vehicle_model.create_time_domain_dynamics()
            self.tracking_controller = TrackingStrategy(
                vehicle_config=self.vehicle_model.vehicle,
                ers_config=self.ers,
                track_model=self.track,
                reference_profile=reference_profile,
            )

            self.s_ref, self.v_ref = get_reference_arrays(reference_profile)

            self.state = np.zeros(3, dtype=float)
            self.step_count = 0
            self.energy_deployed_j = 0.0
            self.energy_recovered_j = 0.0

            # ERS-only action for a minimal first RL problem
            self.action_space = spaces.Box(
                low=np.array([-1.0], dtype=np.float32),
                high=np.array([1.0], dtype=np.float32),
                dtype=np.float32,
            )

            self.observation_space = spaces.Box(
                low=np.array([0.0, 0.0, 0.0, -1.0, 0.0, 0.0, -1.5, 0.0, 0.0, -1.0], dtype=np.float32),
                high=np.array([1.0, 1.5, 1.0, 1.0, 1.0, 1.5, 1.5, 1.0, 1.0, 1.0], dtype=np.float32),
                dtype=np.float32,
            )

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)

            options = options or {}
            soc0 = float(options.get("initial_soc", self.initial_soc))
            v0 = float(options.get("initial_velocity", self.initial_velocity))

            self.state = np.array([0.0, v0, soc0], dtype=float)
            self.step_count = 0
            self.energy_deployed_j = 0.0
            self.energy_recovered_j = 0.0

            obs = self._get_observation(self.state)
            info = {
                "energy_deployed_j": self.energy_deployed_j,
                "energy_recovered_j": self.energy_recovered_j,
            }
            return obs, info

        def step(self, action):
            action_value = float(np.asarray(action, dtype=float).reshape(-1)[0])
            prev_state = self.state.copy()

            track_info = self._get_track_info(prev_state[0])
            base_control = self.tracking_controller.get_control(prev_state, track_info)
            throttle = float(np.clip(base_control[1], 0.0, 1.0))
            brake = float(np.clip(base_control[2], 0.0, 1.0))

            p_ers = action_to_ers_power(
                action_value=action_value,
                speed_mps=float(prev_state[1]),
                soc=float(prev_state[2]),
                ers=self.ers,
                dt=self.dt,
                deployed_energy_j=self.energy_deployed_j,
                recovered_energy_j=self.energy_recovered_j,
            )

            control = np.array([p_ers, throttle, brake], dtype=float)
            track_params = np.array([track_info["gradient"], track_info["radius"]], dtype=float)
            next_state = self._integrate_rk4(prev_state, control, track_params)

            next_state[1] = float(np.clip(next_state[1], self.constraints["v_min"], self.constraints["v_max"]))
            next_state[2] = float(np.clip(next_state[2], self.constraints["soc_min"], self.constraints["soc_max"]))

            self._update_energy_totals(p_ers)
            self.state = next_state
            self.step_count += 1

            progress_m = max(0.0, float(next_state[0] - prev_state[0]))
            terminated = bool(next_state[0] >= self.track.total_length)
            truncated = bool(self.step_count >= self.max_steps and not terminated)

            reward = self._compute_reward(
                progress_m=progress_m,
                p_ers=p_ers,
                soc=next_state[2],
                terminated=terminated,
                truncated=truncated,
            )

            obs = self._get_observation(next_state)
            info = {
                "energy_deployed_j": self.energy_deployed_j,
                "energy_recovered_j": self.energy_recovered_j,
                "lap_complete": terminated,
                "final_soc": float(next_state[2]),
                "time_s": self.step_count * self.dt,
            }

            return obs, float(reward), terminated, truncated, info

        def _compute_reward(
            self,
            progress_m: float,
            p_ers: float,
            soc: float,
            terminated: bool,
            truncated: bool,
        ) -> float:
            """
            Reward balances pace and strategic SOC outcome.
            """
            speed_reward = 0.05 * progress_m
            ers_penalty = 0.02 * (
                abs(p_ers) / max(float(self.ers.max_deployment_power), 1.0)
            )
            reward = speed_reward - ers_penalty

            if terminated:
                soc_error = abs(float(soc) - self.final_soc_target)
                reward += 100.0 - 60.0 * soc_error
                if soc < self.final_soc_target:
                    reward -= 40.0 * (self.final_soc_target - soc)
            elif truncated:
                reward -= 50.0

            return float(reward)

        def _get_observation(self, state: np.ndarray) -> np.ndarray:
            track_info = self._get_track_info(state[0])
            base_control = self.tracking_controller.get_control(state, track_info)

            deploy_remaining = max(
                float(self.ers.deployment_limit_per_lap) - self.energy_deployed_j,
                0.0,
            )
            recover_remaining = max(
                float(self.ers.recovery_limit_per_lap) - self.energy_recovered_j,
                0.0,
            )

            v_ref = sample_reference_speed(
                position_m=float(state[0]),
                total_length_m=float(self.track.total_length),
                s_ref=self.s_ref,
                v_ref=self.v_ref,
            )

            return build_observation(
                state=state,
                track_info=track_info,
                total_length_m=float(self.track.total_length),
                reference_speed_mps=v_ref,
                deploy_remaining_j=deploy_remaining,
                recover_remaining_j=recover_remaining,
                deployment_limit_j=float(self.ers.deployment_limit_per_lap),
                recovery_limit_j=float(self.ers.recovery_limit_per_lap),
                throttle_base=float(base_control[1]),
                brake_base=float(base_control[2]),
            )

        def _update_energy_totals(self, p_ers: float) -> None:
            if p_ers > 0.0:
                self.energy_deployed_j += p_ers * self.dt
            else:
                self.energy_recovered_j += -p_ers * self.dt

        def _get_track_info(self, position_m: float) -> dict[str, float]:
            segment = self.track.get_segment_at_distance(position_m % self.track.total_length)
            return {
                "gradient": float(segment.gradient),
                "radius": float(segment.radius),
                "curvature": float(segment.curvature),
                "sector": float(segment.sector),
            }

        def _integrate_rk4(
            self,
            state: np.ndarray,
            control: np.ndarray,
            track_params: np.ndarray,
        ) -> np.ndarray:
            k1 = self.dynamics_func(state, control, track_params).full().flatten()
            k2 = self.dynamics_func(state + self.dt / 2.0 * k1, control, track_params).full().flatten()
            k3 = self.dynamics_func(state + self.dt / 2.0 * k2, control, track_params).full().flatten()
            k4 = self.dynamics_func(state + self.dt * k3, control, track_params).full().flatten()
            return state + self.dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
