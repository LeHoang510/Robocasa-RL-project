"""
TD3+BC — Offline Reinforcement Learning with Behavioral Cloning regularization.

Reference: Fujimoto & Gu, "A Minimalist Approach to Offline Reinforcement Learning"
           NeurIPS 2021.  https://arxiv.org/abs/2106.06860

The key idea: augment the TD3 actor loss with a BC term so the policy maximizes
Q-value while staying close to the demonstrated actions:

    L_actor = -λ * Q(s, π(s))  +  ||π(s) − a_demo||²
    λ        = α / mean(|Q(s, π(s))|)          (normalises the two terms)

Training is entirely offline — no env interaction. Evaluation in env only.

Observation format: same 16D proprioceptive state as BCAgent.
Action format:      same 12D HDF5 ordering as BCAgent.

Usage
-----
agent = TD3BCAgent.load("checkpoints/exp7_td3bc_<ts>/td3bc_best.pt")
action = agent.predict_from_obs_dict(obs_dict)
"""

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .bc_policy import extract_bc_obs


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------

class _Actor(nn.Module):
    """Deterministic policy: obs_norm → action in [-1, 1] (tanh output)."""

    def __init__(self, obs_dim: int, action_dim: int, net_arch: list[int]):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in net_arch:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers += [nn.Linear(in_dim, action_dim), nn.Tanh()]
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class _Critic(nn.Module):
    """Q(s, a) estimator — takes normalised obs and normalised action."""

    def __init__(self, obs_dim: int, action_dim: int, net_arch: list[int]):
        super().__init__()
        layers = []
        in_dim = obs_dim + action_dim
        for h in net_arch:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, action], dim=-1))


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class TD3BCAgent:
    """
    TD3+BC offline RL agent.

    All networks operate in normalised spaces:
      • obs:    (obs - obs_mean) / obs_std
      • action: (action - action_shift) / action_scale  ∈ [-1, 1]

    predict() / predict_from_obs_dict() match the BCAgent interface.
    """

    def __init__(
        self,
        obs_dim:       int,
        action_dim:    int,
        actor_arch:    list[int],
        critic_arch:   list[int],
        obs_mean:      np.ndarray,
        obs_std:       np.ndarray,
        action_low:    np.ndarray,
        action_high:   np.ndarray,
        # TD3 hypers
        gamma:         float = 0.99,
        tau:           float = 0.005,
        policy_noise:  float = 0.2,
        noise_clip:    float = 0.5,
        policy_delay:  int   = 2,
        # BC weight
        alpha:         float = 2.5,
        device:        str   = "cpu",
    ):
        self.device        = torch.device(device)
        self.gamma         = gamma
        self.tau           = tau
        self.policy_noise  = policy_noise
        self.noise_clip    = noise_clip
        self.policy_delay  = policy_delay
        self.alpha         = alpha
        self.action_dim    = action_dim
        self._total_steps  = 0

        # Normalisation constants
        self.obs_mean   = torch.tensor(obs_mean,                          dtype=torch.float32, device=self.device)
        self.obs_std    = torch.tensor(obs_std,                           dtype=torch.float32, device=self.device)
        self._act_scale = torch.tensor((action_high - action_low) / 2.0, dtype=torch.float32, device=self.device)
        self._act_shift = torch.tensor((action_high + action_low) / 2.0, dtype=torch.float32, device=self.device)
        self.action_low  = action_low.astype(np.float32)
        self.action_high = action_high.astype(np.float32)

        # Networks
        self.actor   = _Actor( obs_dim, action_dim, actor_arch).to(self.device)
        self.critic1 = _Critic(obs_dim, action_dim, critic_arch).to(self.device)
        self.critic2 = _Critic(obs_dim, action_dim, critic_arch).to(self.device)

        self.actor_tgt   = copy.deepcopy(self.actor)
        self.critic1_tgt = copy.deepcopy(self.critic1)
        self.critic2_tgt = copy.deepcopy(self.critic2)

        for net in (self.actor_tgt, self.critic1_tgt, self.critic2_tgt):
            for p in net.parameters():
                p.requires_grad_(False)

    def _init_optimizers(self, actor_lr: float, critic_lr: float):
        self.actor_opt  = torch.optim.Adam(self.actor.parameters(),  lr=actor_lr)
        self.critic_opt = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=critic_lr,
        )

    # ------------------------------------------------------------------
    # Normalisation helpers
    # ------------------------------------------------------------------

    def _norm_obs(self, obs: torch.Tensor) -> torch.Tensor:
        return (obs - self.obs_mean) / (self.obs_std + 1e-8)

    def _norm_action(self, action: torch.Tensor) -> torch.Tensor:
        return (action - self._act_shift) / (self._act_scale + 1e-8)

    def _denorm_action(self, action_n: torch.Tensor) -> torch.Tensor:
        return action_n * self._act_scale + self._act_shift

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def update(
        self,
        obs:      torch.Tensor,
        action:   torch.Tensor,
        reward:   torch.Tensor,
        next_obs: torch.Tensor,
        done:     torch.Tensor,
    ) -> dict:
        """
        One gradient step.  All tensors are raw (un-normalised) on self.device.
        Returns dict of scalar metrics.
        """
        self._total_steps += 1

        obs_n      = self._norm_obs(obs)
        next_obs_n = self._norm_obs(next_obs)
        action_n   = self._norm_action(action)   # ∈ [-1, 1]

        # ---- Critic update -------------------------------------------
        with torch.no_grad():
            noise = (torch.randn_like(action_n) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_action_n = (self.actor_tgt(next_obs_n) + noise).clamp(-1.0, 1.0)
            next_action   = self._denorm_action(next_action_n)

            q1_tgt = self.critic1_tgt(next_obs_n, next_action)
            q2_tgt = self.critic2_tgt(next_obs_n, next_action)
            q_tgt  = reward + self.gamma * (1.0 - done) * torch.min(q1_tgt, q2_tgt)

        q1 = self.critic1(obs_n, action)
        q2 = self.critic2(obs_n, action)
        critic_loss = F.mse_loss(q1, q_tgt) + F.mse_loss(q2, q_tgt)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        metrics = {"critic_loss": critic_loss.item()}

        # ---- Delayed actor update ------------------------------------
        if self._total_steps % self.policy_delay == 0:
            pi_n   = self.actor(obs_n)
            pi     = self._denorm_action(pi_n)
            q_pi   = self.critic1(obs_n, pi)

            # λ normalises Q magnitude so BC and Q terms are comparable
            lam = self.alpha / (q_pi.abs().mean().detach() + 1e-8)

            actor_loss = -lam * q_pi.mean() + F.mse_loss(pi_n, action_n)

            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            # EMA target updates
            self._ema(self.actor,   self.actor_tgt)
            self._ema(self.critic1, self.critic1_tgt)
            self._ema(self.critic2, self.critic2_tgt)

            metrics["actor_loss"] = actor_loss.item()
            metrics["q_mean"]     = q_pi.mean().item()

        return metrics

    def _ema(self, src: nn.Module, tgt: nn.Module):
        for ps, pt in zip(src.parameters(), tgt.parameters()):
            pt.data.copy_(self.tau * ps.data + (1.0 - self.tau) * pt.data)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(
        self,
        obs:           np.ndarray,
        deterministic: bool = True,
    ) -> tuple[np.ndarray, None]:
        batched = obs.ndim == 1
        if batched:
            obs = obs[None]

        obs_t   = torch.tensor(obs, dtype=torch.float32, device=self.device)
        obs_n   = self._norm_obs(obs_t)
        action_n = self.actor(obs_n)
        action   = self._denorm_action(action_n).cpu().numpy()
        action   = np.clip(action, self.action_low, self.action_high)
        return (action[0] if batched else action), None

    def predict_from_obs_dict(self, obs_dict: dict) -> np.ndarray:
        obs_16d  = extract_bc_obs(obs_dict)
        action, _ = self.predict(obs_16d)
        return action

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def save(self, path: str):
        torch.save(
            {
                "actor_state":   self.actor.state_dict(),
                "critic1_state": self.critic1.state_dict(),
                "critic2_state": self.critic2.state_dict(),
                "obs_mean":      self.obs_mean.cpu().numpy(),
                "obs_std":       self.obs_std.cpu().numpy(),
                "action_low":    self.action_low,
                "action_high":   self.action_high,
                "obs_dim":       int(self.obs_mean.shape[0]),
                "action_dim":    self.action_dim,
                "actor_arch":    [m.out_features for m in self.actor.net
                                  if isinstance(m, nn.Linear)][:-1],
                "critic_arch":   [m.out_features for m in self.critic1.net
                                  if isinstance(m, nn.Linear)][:-1],
                "gamma":         self.gamma,
                "tau":           self.tau,
                "policy_noise":  self.policy_noise,
                "noise_clip":    self.noise_clip,
                "policy_delay":  self.policy_delay,
                "alpha":         self.alpha,
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "TD3BCAgent":
        ckpt  = torch.load(path, map_location=device, weights_only=False)
        agent = cls(
            obs_dim      = ckpt["obs_dim"],
            action_dim   = ckpt["action_dim"],
            actor_arch   = ckpt["actor_arch"],
            critic_arch  = ckpt["critic_arch"],
            obs_mean     = ckpt["obs_mean"],
            obs_std      = ckpt["obs_std"],
            action_low   = ckpt["action_low"],
            action_high  = ckpt["action_high"],
            gamma        = ckpt["gamma"],
            tau          = ckpt["tau"],
            policy_noise = ckpt["policy_noise"],
            noise_clip   = ckpt["noise_clip"],
            policy_delay = ckpt["policy_delay"],
            alpha        = ckpt["alpha"],
            device       = device,
        )
        agent.actor.load_state_dict(ckpt["actor_state"])
        agent.critic1.load_state_dict(ckpt["critic1_state"])
        agent.critic2.load_state_dict(ckpt["critic2_state"])
        agent.actor_tgt   = copy.deepcopy(agent.actor)
        agent.critic1_tgt = copy.deepcopy(agent.critic1)
        agent.critic2_tgt = copy.deepcopy(agent.critic2)
        for net in (agent.actor_tgt, agent.critic1_tgt, agent.critic2_tgt):
            for p in net.parameters():
                p.requires_grad_(False)
        agent.actor.eval()
        return agent
