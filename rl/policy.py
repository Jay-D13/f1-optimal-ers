"""
CleanRL-style PPO policy module used for training and inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal


def _layer_init(layer: nn.Module, std: float = np.sqrt(2.0), bias_const: float = 0.0) -> nn.Module:
    torch.nn.init.orthogonal_(layer.weight, std)
    if layer.bias is not None:
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class PPOPolicy(nn.Module):
    """Continuous-action actor-critic policy."""

    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)

        self.critic = nn.Sequential(
            _layer_init(nn.Linear(self.obs_dim, 64)),
            nn.Tanh(),
            _layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            _layer_init(nn.Linear(64, 1), std=1.0),
        )

        self.actor_mean = nn.Sequential(
            _layer_init(nn.Linear(self.obs_dim, 64)),
            nn.Tanh(),
            _layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            _layer_init(nn.Linear(64, self.action_dim), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, self.action_dim))

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs)

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        action_mean = self.actor_mean(obs)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)

        if action is None:
            action = probs.sample()

        log_prob = probs.log_prob(action).sum(1)
        entropy = probs.entropy().sum(1)
        value = self.critic(obs).squeeze(-1)
        return action, log_prob, entropy, value

    @torch.no_grad()
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """
        Inference helper compatible with strategy wrappers.
        Returns numpy action in [-1, 1].
        """
        device = next(self.parameters()).device
        obs_t = torch.as_tensor(observation, dtype=torch.float32, device=device).reshape(1, -1)
        mean = self.actor_mean(obs_t)
        if deterministic:
            action_t = mean
        else:
            logstd = self.actor_logstd.expand_as(mean)
            std = torch.exp(logstd)
            action_t = Normal(mean, std).sample()
        action_t = torch.tanh(action_t)
        return action_t.squeeze(0).detach().cpu().numpy()


@dataclass
class PolicyCheckpoint:
    obs_dim: int
    action_dim: int
    state_dict: dict[str, Any]

    def save(self, path: str | Path) -> Path:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "obs_dim": self.obs_dim,
                "action_dim": self.action_dim,
                "state_dict": self.state_dict,
            },
            out_path,
        )
        return out_path


def save_policy_checkpoint(policy: PPOPolicy, path: str | Path) -> Path:
    payload = PolicyCheckpoint(
        obs_dim=policy.obs_dim,
        action_dim=policy.action_dim,
        state_dict=policy.state_dict(),
    )
    return payload.save(path)


def load_policy_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> PPOPolicy:
    ckpt = torch.load(Path(path), map_location=device)
    if not {"obs_dim", "action_dim", "state_dict"} <= set(ckpt.keys()):
        raise ValueError(
            "Invalid policy checkpoint. Expected keys: obs_dim, action_dim, state_dict"
        )

    policy = PPOPolicy(obs_dim=int(ckpt["obs_dim"]), action_dim=int(ckpt["action_dim"]))
    policy.load_state_dict(ckpt["state_dict"])
    policy.to(device)
    policy.eval()
    return policy
