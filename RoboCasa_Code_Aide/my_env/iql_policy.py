"""
IQL — Implicit Q-Learning for offline reinforcement learning.

Reference: Kostrikov et al., "Offline Reinforcement Learning with Implicit Q-Learning"
           ICLR 2022.  https://arxiv.org/abs/2110.06169

Key ideas vs TD3+BC:
  • Never evaluates actions outside the dataset (no OOD action problem).
  • V(s) is learned with expectile regression:  L_V = E[|τ - 1(A<0)| · A²]
    where A = Q(s,a) - V(s).  τ > 0.5 makes V approach max_a Q(s,a).
  • Q is backed up with V(s') instead of max_a Q(s',a):  Q_target = r + γ·V(s')
  • Actor trained with Advantage-Weighted Regression (AWR):
    L_actor = -E[exp(β·A) · log π(a_demo | s)]

Three networks: twin critics Q1/Q2, value network V, Gaussian actor π.
Only critics have target networks (EMA-updated).

Observation format: same 16D as BCAgent.
Action format:      same 12D HDF5 ordering.

Usage
-----
agent = IQLAgent.load("checkpoints/exp8_iql_<ts>/iql_best.pt")
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

class _GaussianActor(nn.Module):
    """
    Stochastic Gaussian policy with tanh squashing.
    Outputs actions in [-1, 1] (normalised action space).
    At evaluation, uses the deterministic mean.
    """

    LOG_STD_MIN = -5.0
    LOG_STD_MAX = 2.0

    def __init__(self, obs_dim: int, action_dim: int, net_arch: list[int]):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in net_arch:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        self.trunk         = nn.Sequential(*layers)
        self.mu_head       = nn.Linear(in_dim, action_dim)
        self.log_std_head  = nn.Linear(in_dim, action_dim)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h       = self.trunk(obs)
        mu      = self.mu_head(h)
        log_std = self.log_std_head(h).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mu, log_std.exp()

    def log_prob_of(self, obs: torch.Tensor, action_n: torch.Tensor) -> torch.Tensor:
        """
        Log-probability of a dataset action (already in [-1,1] space) under
        the current tanh-Gaussian policy.  Includes the tanh Jacobian correction.
        """
        mu, std = self(obs)
        # Clamp once — used for both atanh and the Jacobian to keep 1 - a² > 0
        action_c = action_n.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        pre_tanh = torch.atanh(action_c)
        dist     = torch.distributions.Normal(mu, std)
        log_p    = dist.log_prob(pre_tanh).sum(-1, keepdim=True)
        # Jacobian: log|∂tanh/∂x|⁻¹ = -log(1 - tanh(x)²)
        log_p   -= torch.log(1.0 - action_c.pow(2) + 1e-6).sum(-1, keepdim=True)
        return log_p

    def get_mean_action(self, obs: torch.Tensor) -> torch.Tensor:
        mu, _ = self(obs)
        return torch.tanh(mu)


class _Critic(nn.Module):
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


class _ValueNet(nn.Module):
    def __init__(self, obs_dim: int, net_arch: list[int]):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in net_arch:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _expectile_loss(u: torch.Tensor, tau: float) -> torch.Tensor:
    """Asymmetric L2 loss — weights underestimation by τ and overestimation by (1-τ)."""
    weight = torch.where(u < 0, torch.tensor(1.0 - tau, device=u.device),
                                torch.tensor(tau,        device=u.device))
    return (weight * u.pow(2)).mean()


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class IQLAgent:
    """
    IQL offline RL agent.

    Normalised spaces:
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
        value_arch:    list[int],
        obs_mean:      np.ndarray,
        obs_std:       np.ndarray,
        action_low:    np.ndarray,
        action_high:   np.ndarray,
        # IQL hypers
        gamma:         float = 0.99,
        tau:           float = 0.005,       # EMA rate for target critics
        tau_expectile: float = 0.7,         # expectile for V update
        beta:          float = 3.0,         # temperature for AWR actor
        adv_clip:      float = 100.0,       # clip exp(β·A) for stability
        device:        str   = "cpu",
    ):
        self.device        = torch.device(device)
        self.gamma         = gamma
        self.tau           = tau
        self.tau_expectile = tau_expectile
        self.beta          = beta
        self.adv_clip      = adv_clip
        self.action_dim    = action_dim
        self._total_steps  = 0

        # Normalisation
        self.obs_mean   = torch.tensor(obs_mean,                          dtype=torch.float32, device=self.device)
        self.obs_std    = torch.tensor(obs_std,                           dtype=torch.float32, device=self.device)
        self._act_scale = torch.tensor((action_high - action_low) / 2.0, dtype=torch.float32, device=self.device)
        self._act_shift = torch.tensor((action_high + action_low) / 2.0, dtype=torch.float32, device=self.device)
        self.action_low  = action_low.astype(np.float32)
        self.action_high = action_high.astype(np.float32)

        # Networks
        self.actor   = _GaussianActor(obs_dim, action_dim, actor_arch).to(self.device)
        self.critic1 = _Critic(obs_dim, action_dim, critic_arch).to(self.device)
        self.critic2 = _Critic(obs_dim, action_dim, critic_arch).to(self.device)
        self.value   = _ValueNet(obs_dim, value_arch).to(self.device)

        # Target critics only (IQL has no target V)
        self.critic1_tgt = copy.deepcopy(self.critic1)
        self.critic2_tgt = copy.deepcopy(self.critic2)
        for net in (self.critic1_tgt, self.critic2_tgt):
            for p in net.parameters():
                p.requires_grad_(False)

    def _init_optimizers(self, actor_lr: float, critic_lr: float, value_lr: float):
        self.actor_opt  = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=critic_lr,
        )
        self.value_opt  = torch.optim.Adam(self.value.parameters(), lr=value_lr)

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
        """One gradient step. All tensors are raw (un-normalised) on self.device."""
        self._total_steps += 1

        obs_n      = self._norm_obs(obs)
        next_obs_n = self._norm_obs(next_obs)
        action_n   = self._norm_action(action)

        # ---- 1. Q update: Q(s,a) ← r + γ·V(s') -----------------------
        with torch.no_grad():
            v_next   = self.value(next_obs_n)
            q_target = reward + self.gamma * (1.0 - done) * v_next

        q1 = self.critic1(obs_n, action)
        q2 = self.critic2(obs_n, action)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # ---- 2. V update: expectile regression on Q_tgt ---------------
        with torch.no_grad():
            q1_t = self.critic1_tgt(obs_n, action)
            q2_t = self.critic2_tgt(obs_n, action)
            q_min = torch.min(q1_t, q2_t)

        v         = self.value(obs_n)
        value_loss = _expectile_loss(q_min - v, self.tau_expectile)

        self.value_opt.zero_grad()
        value_loss.backward()
        self.value_opt.step()

        # ---- 3. Actor update: advantage-weighted BC (AWR) -------------
        with torch.no_grad():
            advantage = q_min - v.detach()
            weight    = torch.exp(self.beta * advantage).clamp(max=self.adv_clip)

        log_prob   = self.actor.log_prob_of(obs_n, action_n)
        actor_loss = -(weight * log_prob).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # ---- EMA target critic update ---------------------------------
        for ps, pt in zip(self.critic1.parameters(), self.critic1_tgt.parameters()):
            pt.data.copy_(self.tau * ps.data + (1.0 - self.tau) * pt.data)
        for ps, pt in zip(self.critic2.parameters(), self.critic2_tgt.parameters()):
            pt.data.copy_(self.tau * ps.data + (1.0 - self.tau) * pt.data)

        return {
            "critic_loss": critic_loss.item(),
            "value_loss":  value_loss.item(),
            "actor_loss":  actor_loss.item(),
            "q_mean":      q_min.mean().item(),
            "v_mean":      v.mean().item(),
            "adv_mean":    advantage.mean().item(),
        }

    # ------------------------------------------------------------------
    # Inference  (deterministic: use policy mean)
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

        obs_t    = torch.tensor(obs, dtype=torch.float32, device=self.device)
        obs_n    = self._norm_obs(obs_t)
        action_n = self.actor.get_mean_action(obs_n)
        action   = self._denorm_action(action_n).cpu().numpy()
        action   = np.clip(action, self.action_low, self.action_high)
        return (action[0] if batched else action), None

    def predict_from_obs_dict(self, obs_dict: dict) -> np.ndarray:
        obs_16d   = extract_bc_obs(obs_dict)
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
                "value_state":   self.value.state_dict(),
                "obs_mean":      self.obs_mean.cpu().numpy(),
                "obs_std":       self.obs_std.cpu().numpy(),
                "action_low":    self.action_low,
                "action_high":   self.action_high,
                "obs_dim":       int(self.obs_mean.shape[0]),
                "action_dim":    self.action_dim,
                "actor_arch":    [m.out_features for m in self.actor.trunk
                                  if isinstance(m, nn.Linear)],
                "critic_arch":   [m.out_features for m in self.critic1.net
                                  if isinstance(m, nn.Linear)][:-1],
                "value_arch":    [m.out_features for m in self.value.net
                                  if isinstance(m, nn.Linear)][:-1],
                "gamma":         self.gamma,
                "tau":           self.tau,
                "tau_expectile": self.tau_expectile,
                "beta":          self.beta,
                "adv_clip":      self.adv_clip,
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "IQLAgent":
        ckpt  = torch.load(path, map_location=device, weights_only=False)
        agent = cls(
            obs_dim       = ckpt["obs_dim"],
            action_dim    = ckpt["action_dim"],
            actor_arch    = ckpt["actor_arch"],
            critic_arch   = ckpt["critic_arch"],
            value_arch    = ckpt["value_arch"],
            obs_mean      = ckpt["obs_mean"],
            obs_std       = ckpt["obs_std"],
            action_low    = ckpt["action_low"],
            action_high   = ckpt["action_high"],
            gamma         = ckpt["gamma"],
            tau           = ckpt["tau"],
            tau_expectile = ckpt["tau_expectile"],
            beta          = ckpt["beta"],
            adv_clip      = ckpt["adv_clip"],
            device        = device,
        )
        agent.actor.load_state_dict(ckpt["actor_state"])
        agent.critic1.load_state_dict(ckpt["critic1_state"])
        agent.critic2.load_state_dict(ckpt["critic2_state"])
        agent.value.load_state_dict(ckpt["value_state"])
        agent.critic1_tgt = copy.deepcopy(agent.critic1)
        agent.critic2_tgt = copy.deepcopy(agent.critic2)
        for net in (agent.critic1_tgt, agent.critic2_tgt):
            for p in net.parameters():
                p.requires_grad_(False)
        agent.actor.eval()
        return agent
