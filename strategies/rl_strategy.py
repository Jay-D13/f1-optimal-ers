from __future__ import annotations

from typing import Protocol

import numpy as np

from rl.common import (
    action_to_ers_power,
    build_observation,
    get_reference_arrays,
    sample_reference_speed,
)
from rl.policy import load_policy_checkpoint
from .base import BaseStrategy
from .baselines import TrackingStrategy


class _PredictPolicy(Protocol):
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        ...


class RLERSStrategy(BaseStrategy):
    """
    Inference-time strategy wrapper for a trained ERS RL policy.

    The policy decides only ERS command; throttle/brake come from the baseline
    tracking controller so behavior remains physically plausible.
    """

    def __init__(
        self,
        vehicle_config,
        ers_config,
        track_model,
        *,
        policy: _PredictPolicy | None = None,
        model_path: str | None = None,
        reference_profile=None,
        dt: float = 0.1,
        final_soc_target: float = 0.3,
    ):
        super().__init__(vehicle_config, ers_config, track_model)

        if policy is None and model_path is None:
            raise ValueError("Provide either 'policy' or 'model_path'")

        self.policy = policy if policy is not None else self._load_model(model_path)
        self.dt = float(dt)
        self.final_soc_target = float(final_soc_target)

        self.tracking_controller = TrackingStrategy(
            vehicle_config=vehicle_config,
            ers_config=ers_config,
            track_model=track_model,
            reference_profile=reference_profile,
        )

        self.s_ref, self.v_ref = get_reference_arrays(reference_profile)

        self.energy_deployed_j = 0.0
        self.energy_recovered_j = 0.0

    @property
    def name(self) -> str:
        return "RL ERS"

    def reset(self):
        self.energy_deployed_j = 0.0
        self.energy_recovered_j = 0.0
        if hasattr(self.tracking_controller, "reset"):
            self.tracking_controller.reset()

    def get_control(self, state: np.ndarray, track_info: dict) -> np.ndarray:
        base_control = self.tracking_controller.get_control(state, track_info)
        throttle = float(np.clip(base_control[1], 0.0, 1.0))
        brake = float(np.clip(base_control[2], 0.0, 1.0))

        deploy_remaining = max(self.ers.deployment_limit_per_lap - self.energy_deployed_j, 0.0)
        recover_remaining = max(self.ers.recovery_limit_per_lap - self.energy_recovered_j, 0.0)

        v_ref = sample_reference_speed(
            position_m=float(state[0]),
            total_length_m=float(self.track.total_length),
            s_ref=self.s_ref,
            v_ref=self.v_ref,
        )

        obs = build_observation(
            state=state,
            track_info=track_info,
            total_length_m=float(self.track.total_length),
            reference_speed_mps=v_ref,
            deploy_remaining_j=deploy_remaining,
            recover_remaining_j=recover_remaining,
            deployment_limit_j=float(self.ers.deployment_limit_per_lap),
            recovery_limit_j=float(self.ers.recovery_limit_per_lap),
            throttle_base=throttle,
            brake_base=brake,
        )

        action_value = self._predict_action(obs)
        p_ers = action_to_ers_power(
            action_value=action_value,
            speed_mps=float(state[1]),
            soc=float(state[2]),
            ers=self.ers,
            dt=self.dt,
            deployed_energy_j=self.energy_deployed_j,
            recovered_energy_j=self.energy_recovered_j,
        )

        if p_ers > 0:
            self.energy_deployed_j += p_ers * self.dt
        else:
            self.energy_recovered_j += -p_ers * self.dt

        return np.array([p_ers, throttle, brake], dtype=float)

    def _predict_action(self, observation: np.ndarray) -> float:
        action = self.policy.predict(observation, deterministic=True)
        return float(np.asarray(action, dtype=float).reshape(-1)[0])

    @staticmethod
    def _load_model(model_path: str):
        if model_path is None:
            raise ValueError("model_path must not be None")

        return load_policy_checkpoint(model_path, device="cpu")
