"""
Diffusion Policy — DDPM training + DDIM inference for robot action prediction.

Architecture:
  • Denoising network: MLP conditioned on (noisy_action ‖ obs ‖ time_emb)
  • Training: DDPM noise-prediction loss (predict ε at a random timestep)
  • Inference: DDIM with n_inference_steps << n_diffusion_steps (fast, deterministic)

Observation format: same 16D proprioceptive state as BCAgent (bc_policy.py).
Action format: same 12D HDF5 ordering as BCAgent.

Usage
-----
agent = DiffusionAgent.load("checkpoints/exp6_diffusion_<ts>/diffusion_best.pt")
action = agent.predict_from_obs_dict(obs_dict)   # identical interface to BCAgent
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .bc_policy import extract_bc_obs


# ---------------------------------------------------------------------------
# Noise schedules
# ---------------------------------------------------------------------------

def cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    """Cosine schedule (Nichol & Dhariwal 2021) — smoother than linear."""
    t = torch.linspace(0, T, T + 1, dtype=torch.float64)
    alpha_bar = torch.cos(((t / T + s) / (1 + s)) * math.pi * 0.5) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1.0 - alpha_bar[1:] / alpha_bar[:-1]
    return betas.clamp(0.0, 0.999).float()


def linear_beta_schedule(T: int) -> torch.Tensor:
    return torch.linspace(1e-4, 0.02, T)


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

class _SinusoidalEmb(nn.Module):
    """Sinusoidal positional embedding for the diffusion timestep."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        inv_freq = torch.exp(
            -math.log(10000)
            * torch.arange(half, dtype=torch.float32, device=t.device)
            / max(half - 1, 1)
        )
        emb = t.float().unsqueeze(1) * inv_freq.unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class _DenoiseMLP(nn.Module):
    """
    Predicts noise ε(x_t, obs, t).
    Input: concat(x_t, obs_normalised, time_emb_projected)  →  predicted noise
    """

    def __init__(
        self,
        action_dim:   int,
        obs_dim:      int,
        net_arch:     list[int],
        time_emb_dim: int,
    ):
        super().__init__()
        self.time_emb = nn.Sequential(
            _SinusoidalEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
        )
        in_dim = action_dim + obs_dim + time_emb_dim
        layers = []
        for h in net_arch:
            layers += [nn.Linear(in_dim, h), nn.SiLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        x_t: torch.Tensor,
        obs: torch.Tensor,
        t:   torch.Tensor,
    ) -> torch.Tensor:
        return self.net(torch.cat([x_t, obs, self.time_emb(t)], dim=-1))


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class DiffusionAgent:
    """
    Diffusion Policy agent.

    Drop-in replacement for BCAgent — same predict() / predict_from_obs_dict()
    interface so eval scripts work unchanged.
    """

    def __init__(
        self,
        obs_dim:           int,
        action_dim:        int,
        net_arch:          list[int],
        time_emb_dim:      int,
        obs_mean:          np.ndarray,
        obs_std:           np.ndarray,
        action_low:        np.ndarray,
        action_high:       np.ndarray,
        n_diffusion_steps: int  = 100,
        n_inference_steps: int  = 10,
        beta_schedule:     str  = "cosine",
        device:            str  = "cpu",
    ):
        self.device       = torch.device(device)
        self.T            = n_diffusion_steps
        self.n_inf        = n_inference_steps
        self.action_dim   = action_dim
        self._beta_sched  = beta_schedule

        self.net = _DenoiseMLP(action_dim, obs_dim, net_arch, time_emb_dim).to(self.device)

        # Pre-compute DDPM schedule tensors
        betas     = (cosine_beta_schedule if beta_schedule == "cosine"
                     else linear_beta_schedule)(n_diffusion_steps)
        alphas    = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        def _buf(t: torch.Tensor) -> torch.Tensor:
            return t.to(self.device)

        self._alpha_bar          = _buf(alpha_bar)
        self._sqrt_ab            = _buf(alpha_bar.sqrt())
        self._sqrt_one_minus_ab  = _buf((1.0 - alpha_bar).sqrt())

        # Obs normalisation
        self.obs_mean = torch.tensor(obs_mean, dtype=torch.float32, device=self.device)
        self.obs_std  = torch.tensor(obs_std,  dtype=torch.float32, device=self.device)

        # Action normalisation: map [low, high] → [-1, 1] for diffusion
        self._act_scale = torch.tensor(
            (action_high - action_low) / 2.0, dtype=torch.float32, device=self.device
        )
        self._act_shift = torch.tensor(
            (action_high + action_low) / 2.0, dtype=torch.float32, device=self.device
        )
        self.action_low  = action_low.astype(np.float32)
        self.action_high = action_high.astype(np.float32)

    # ------------------------------------------------------------------
    # Forward process (used during training)
    # ------------------------------------------------------------------

    def q_sample(
        self,
        x0:    torch.Tensor,
        t:     torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """Add noise to x0 at diffusion step t: x_t = sqrt(ᾱ_t)*x0 + sqrt(1-ᾱ_t)*ε."""
        return (
            self._sqrt_ab[t].view(-1, 1) * x0
            + self._sqrt_one_minus_ab[t].view(-1, 1) * noise
        )

    def _norm_obs(self, obs: torch.Tensor) -> torch.Tensor:
        return (obs - self.obs_mean) / (self.obs_std + 1e-8)

    def _norm_action(self, action: torch.Tensor) -> torch.Tensor:
        return (action - self._act_shift) / (self._act_scale + 1e-8)

    def _denorm_action(self, action: torch.Tensor) -> torch.Tensor:
        return action * self._act_scale + self._act_shift

    def training_loss(
        self,
        obs:    torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """
        DDPM noise-prediction loss.
        obs and action are raw (un-normalised) tensors from the DataLoader.
        """
        B = obs.shape[0]
        obs_n    = self._norm_obs(obs)
        action_n = self._norm_action(action)

        t     = torch.randint(0, self.T, (B,), device=self.device)
        noise = torch.randn_like(action_n)
        x_t   = self.q_sample(action_n, t, noise)

        return F.mse_loss(self.net(x_t, obs_n, t), noise)

    # ------------------------------------------------------------------
    # Reverse process — DDIM inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(
        self,
        obs:           np.ndarray,
        deterministic: bool = True,
    ) -> tuple[np.ndarray, None]:
        """
        Denoise from Gaussian noise to action using DDIM.
        Returns (action, None) to match BCAgent.predict() signature.
        """
        batched = obs.ndim == 1
        if batched:
            obs = obs[None]
        B = obs.shape[0]

        obs_n = self._norm_obs(
            torch.tensor(obs, dtype=torch.float32, device=self.device)
        )

        # Evenly-spaced DDIM timesteps: [T-1, ..., 0]
        step      = max(1, (self.T - 1) // max(self.n_inf - 1, 1))
        timesteps = list(range(self.T - 1, 0, -step)) + [0]
        timesteps = timesteps[:self.n_inf]

        x = torch.randn(B, self.action_dim, device=self.device)

        for i, t_cur in enumerate(timesteps):
            t_tensor = torch.full((B,), t_cur, dtype=torch.long, device=self.device)
            eps_pred = self.net(x, obs_n, t_tensor)

            ab_t    = self._alpha_bar[t_cur]
            t_prev  = timesteps[i + 1] if i + 1 < len(timesteps) else -1
            ab_prev = self._alpha_bar[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=self.device)

            # Predict x0, then DDIM step (η = 0, deterministic)
            x0_pred = (x - (1.0 - ab_t).sqrt() * eps_pred) / ab_t.sqrt()
            x0_pred = x0_pred.clamp(-1.0, 1.0)
            x = ab_prev.sqrt() * x0_pred + (1.0 - ab_prev).sqrt() * eps_pred

        action = self._denorm_action(x).cpu().numpy()
        action = np.clip(action, self.action_low, self.action_high)
        return (action[0] if batched else action), None

    def predict_from_obs_dict(self, obs_dict: dict) -> np.ndarray:
        obs_16d = extract_bc_obs(obs_dict)
        action, _ = self.predict(obs_16d)
        return action

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def save(self, path: str):
        torch.save(
            {
                "state_dict":        self.net.state_dict(),
                "obs_mean":          self.obs_mean.cpu().numpy(),
                "obs_std":           self.obs_std.cpu().numpy(),
                "action_low":        self.action_low,
                "action_high":       self.action_high,
                "obs_dim":           int(self.obs_mean.shape[0]),
                "action_dim":        self.action_dim,
                "net_arch":          [m.out_features for m in self.net.net
                                      if isinstance(m, nn.Linear)][:-1],
                "time_emb_dim":      self.net.time_emb[1].in_features,
                "n_diffusion_steps": self.T,
                "n_inference_steps": self.n_inf,
                "beta_schedule":     self._beta_sched,
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "DiffusionAgent":
        ckpt  = torch.load(path, map_location=device, weights_only=False)
        agent = cls(
            obs_dim           = ckpt["obs_dim"],
            action_dim        = ckpt["action_dim"],
            net_arch          = ckpt["net_arch"],
            time_emb_dim      = ckpt["time_emb_dim"],
            obs_mean          = ckpt["obs_mean"],
            obs_std           = ckpt["obs_std"],
            action_low        = ckpt["action_low"],
            action_high       = ckpt["action_high"],
            n_diffusion_steps = ckpt["n_diffusion_steps"],
            n_inference_steps = ckpt["n_inference_steps"],
            beta_schedule     = ckpt.get("beta_schedule", "cosine"),
            device            = device,
        )
        agent.net.load_state_dict(ckpt["state_dict"])
        agent.net.eval()
        return agent
