"""Networks for the MAPPO within-environment baseline (R2-round2 #3).

Two modules, both parameter-shared across agents (matching the rest of the
codebase's single-actor / single-critic design):

- ``GaussianActor`` — stochastic continuous policy. Same trunk as
  ``SimpleActor`` (Linear->LayerNorm->ReLU->Linear->ReLU->Linear->Tanh) producing
  the action mean in ``[-max_action, max_action]``, plus a state-independent
  learnable ``log_std`` per action dimension. PPO needs log-probabilities, so the
  deterministic ``SimpleActor`` cannot be reused directly.
- ``CentralizedValue`` — CTDE value function conditioned on the **concatenation of
  all agents' per-agent observations in index order** (the same fix C used by
  ``MADDPGSharedActorCriticIndependentObs``: head ``i`` <-> agent ``i`` binding is
  exact). Outputs one value per agent (``[B, N]``).

Both keep the file-based ``save_checkpoint`` / ``load_checkpoint`` convention of
``SimpleActor`` / ``SharedCritic`` so ``train_seeded.py`` offload/resume works.
"""

import os
from typing import Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.optim as optim

from config import device as DEFAULT_DEVICE


# Standard PPO learning rate (Yu et al. 2021 use 5e-4; 3e-4 is the common
# continuous-control default). Disclosed as a baseline hyperparameter — we do NOT
# reuse the DDPG 1e-3 because PPO is on-policy and more sensitive to step size.
MAPPO_LR = 3e-4
# Initial policy std; matches the 0.3*max_action exploration scale used elsewhere.
INIT_STD = 0.3


class GaussianActor(nn.Module):
    """Shared stochastic Gaussian policy for continuous actions in [-1, 1]."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dim: Optional[int] = None,
        max_action: float = 1.0,
        lr: float = MAPPO_LR,
        init_std: float = INIT_STD,
        device: str = DEFAULT_DEVICE,
        chckpnt_file: str = "mappo_actor.pth",
    ):
        super().__init__()
        self.device = torch.device(device)
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.max_action = max_action

        if hidden_dim is None:
            hidden_dim = max(256, observation_dim)

        self.net = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )
        # State-independent log-std (one per action dim), standard for PPO.
        self.log_std = nn.Parameter(
            torch.full((action_dim,), float(torch.log(torch.tensor(init_std))))
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.to(self.device)
        self._init_weights()
        self.chckpnt_file = chckpnt_file

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (mean, std). mean in [-max_action, max_action]; std broadcast."""
        obs = obs.to(self.device)
        mean = self.net(obs) * self.max_action
        log_std = self.log_std.clamp(-5.0, 2.0)
        std = log_std.exp().expand_as(mean)
        return mean, std

    def distribution(self, obs: torch.Tensor) -> torch.distributions.Normal:
        mean, std = self.forward(obs)
        return torch.distributions.Normal(mean, std)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = False):
        """Sample a raw action + its log-prob. Caller clamps before env.step.

        Returns (raw_action [.., act_dim], logprob [..]). The log-prob is computed
        on the unclamped sample (standard PPO); the env receives the clamped value.
        """
        dist = self.distribution(obs)
        if deterministic:
            raw = dist.mean
        else:
            raw = dist.sample()
        logprob = dist.log_prob(raw).sum(-1)
        return raw, logprob

    def evaluate_actions(self, obs: torch.Tensor, action: torch.Tensor):
        """Log-prob + entropy of given actions under the current policy (graph live)."""
        dist = self.distribution(obs)
        logprob = dist.log_prob(action.to(self.device)).sum(-1)
        entropy = dist.entropy().sum(-1)
        return logprob, entropy

    def save_checkpoint(self, filename: str = None, file_prefix: str = None):
        if filename is None:
            filename = self.chckpnt_file
        if file_prefix is not None:
            filename = f"{file_prefix}_{filename}"
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            filename,
        )

    def load_checkpoint(self, filepath: str = None, raise_on_no_file: bool = False):
        if filepath is None:
            filepath = self.chckpnt_file
        if not os.path.isfile(filepath):
            if raise_on_no_file:
                raise FileNotFoundError(f"Checkpoint file '{filepath}' not found.")
            self.to(self.device)
            return
        ckpt = torch.load(filepath, map_location=self.device)
        self.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.to(self.device)


class CentralizedValue(nn.Module):
    """CTDE value V(concat_obs) -> per-agent values [B, N] (index-ordered)."""

    def __init__(
        self,
        input_dim: int,           # N * obs_dim
        output_dim: int,          # N
        hidden_dim: Optional[int] = None,
        activation: Type[nn.Module] = nn.ReLU,
        lr: float = MAPPO_LR,
        device: str = DEFAULT_DEVICE,
        chckpnt_file: str = "mappo_critic.pth",
    ):
        super().__init__()
        self.device = torch.device(device)
        if hidden_dim is None:
            hidden_dim = max(128, input_dim)

        self.v_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            activation(),
            nn.Linear(hidden_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.to(self.device)
        self._init_weights()
        self.chckpnt_file = chckpnt_file

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, joint_obs: torch.Tensor) -> torch.Tensor:
        """joint_obs: [B, N*obs_dim] -> values [B, N]."""
        return self.v_net(joint_obs.to(self.device))

    def save_checkpoint(self, filename: str = None, file_prefix: str = None):
        if filename is None:
            filename = self.chckpnt_file
        if file_prefix is not None:
            filename = f"{file_prefix}_{filename}"
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            filename,
        )

    def load_checkpoint(self, filepath: str = None, raise_on_no_file: bool = False):
        if filepath is None:
            filepath = self.chckpnt_file
        if not os.path.isfile(filepath):
            if raise_on_no_file:
                raise FileNotFoundError(f"Checkpoint file '{filepath}' not found.")
            self.to(self.device)
            return
        ckpt = torch.load(filepath, map_location=self.device)
        self.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.to(self.device)
