"""
Diffusion Policy — MLP denoiser with DDPM training and DDIM inference.

Architecture
------------
  Conditioning  : obs_horizon frames of 16D state → flattened → linear projection
  Denoiser      : MLP with residual blocks, conditioned via FiLM on (obs, timestep)
  Training      : predict noise added to action chunk (standard DDPM objective)
  Inference     : DDIM (10 steps, ~10x faster than full DDPM)

Reference: Chi et al. "Diffusion Policy" RSS 2023.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Sinusoidal timestep embedding
# ---------------------------------------------------------------------------

def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """timesteps: (B,) int64 → (B, dim) float embeddings."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=timesteps.device) / (half - 1)
    ).float()
    args = timesteps.float()[:, None] * freqs[None]
    return torch.cat([args.sin(), args.cos()], dim=-1)  # (B, dim)


# ---------------------------------------------------------------------------
# FiLM-conditioned residual block
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    """Linear residual block conditioned by FiLM (scale + bias from cond)."""

    def __init__(self, dim: int, cond_dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1   = nn.Linear(dim, dim)
        self.fc2   = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.drop  = nn.Dropout(dropout)
        # FiLM: predict scale and bias from conditioning
        self.film  = nn.Linear(cond_dim, dim * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.film(cond).chunk(2, dim=-1)
        h = self.norm1(x)
        h = F.mish(self.fc1(h))
        h = gamma * h + beta           # FiLM modulation
        h = self.drop(h)
        h = self.norm2(h)
        h = self.fc2(h)
        return x + h


# ---------------------------------------------------------------------------
# MLP Denoiser
# ---------------------------------------------------------------------------

class MLPDenoiser(nn.Module):
    """
    Predicts noise added to action chunk, conditioned on (obs, timestep).

    Input  : noisy_actions (B, action_horizon * act_dim)
    Output : predicted_noise (B, action_horizon * act_dim)
    """

    def __init__(
        self,
        act_dim:        int,
        action_horizon: int,
        obs_dim:        int,
        obs_horizon:    int,
        hidden_dim:     int = 512,
        n_blocks:       int = 4,
        time_emb_dim:   int = 128,
        dropout:        float = 0.0,
    ):
        super().__init__()
        action_flat = act_dim * action_horizon
        obs_flat    = obs_dim * obs_horizon
        cond_dim    = time_emb_dim + obs_flat

        self.time_emb_dim   = time_emb_dim
        self.action_flat    = action_flat

        # Project flattened obs to same space
        self.input_proj = nn.Linear(action_flat, hidden_dim)
        self.blocks = nn.ModuleList([
            ResBlock(hidden_dim, cond_dim, dropout) for _ in range(n_blocks)
        ])
        self.out_proj = nn.Linear(hidden_dim, action_flat)

    def forward(
        self,
        noisy_actions: torch.Tensor,   # (B, action_horizon, act_dim)
        obs:           torch.Tensor,   # (B, obs_horizon, obs_dim)
        t:             torch.Tensor,   # (B,) int
    ) -> torch.Tensor:
        B = noisy_actions.shape[0]
        x    = noisy_actions.reshape(B, -1)        # (B, action_flat)
        obs_flat = obs.reshape(B, -1)              # (B, obs_flat)
        t_emb = sinusoidal_embedding(t, self.time_emb_dim)  # (B, time_emb_dim)
        cond  = torch.cat([t_emb, obs_flat], dim=-1)        # (B, cond_dim)

        h = F.mish(self.input_proj(x))
        for block in self.blocks:
            h = block(h, cond)
        noise_pred = self.out_proj(h)
        return noise_pred.reshape(B, -1, noisy_actions.shape[-1])


# ---------------------------------------------------------------------------
# DDPM noise schedule
# ---------------------------------------------------------------------------

class DDPMScheduler:
    """Linear beta schedule. Stores all precomputed tensors on CPU."""

    def __init__(self, T: int = 100, beta_start: float = 1e-4, beta_end: float = 0.02):
        self.T = T
        betas              = torch.linspace(beta_start, beta_end, T)
        alphas             = 1.0 - betas
        alphas_cumprod     = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.betas              = betas
        self.alphas_cumprod     = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev
        self.sqrt_alphas_cumprod      = alphas_cumprod.sqrt()
        self.sqrt_one_minus_alphas_cp = (1 - alphas_cumprod).sqrt()
        self.posterior_variance       = (
            betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)
        ).clamp(min=1e-20)

    def to(self, device):
        for attr in ["betas", "alphas_cumprod", "alphas_cumprod_prev",
                     "sqrt_alphas_cumprod", "sqrt_one_minus_alphas_cp",
                     "posterior_variance"]:
            setattr(self, attr, getattr(self, attr).to(device))
        return self

    def add_noise(self, x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor):
        """Forward diffusion: x_t = sqrt(ᾱ_t)·x0 + sqrt(1−ᾱ_t)·noise."""
        s = self.sqrt_alphas_cumprod[t][:, None, None]
        r = self.sqrt_one_minus_alphas_cp[t][:, None, None]
        return s * x0 + r * noise


# ---------------------------------------------------------------------------
# DDIM sampler (fast inference, ~10 steps)
# ---------------------------------------------------------------------------

class DDIMSampler:
    def __init__(self, scheduler: DDPMScheduler, n_steps: int = 10):
        self.scheduler = scheduler
        T = scheduler.T
        # Evenly spaced timesteps from T-1 down to 0
        self.timesteps = torch.linspace(T - 1, 0, n_steps).long()

    @torch.no_grad()
    def sample(
        self,
        model:  MLPDenoiser,
        obs:    torch.Tensor,   # (B, obs_horizon, obs_dim)
        shape:  tuple,          # (B, action_horizon, act_dim)
        device: torch.device,
    ) -> torch.Tensor:
        x = torch.randn(shape, device=device)
        sc = self.scheduler

        for i, t_val in enumerate(self.timesteps):
            t = t_val.expand(shape[0]).to(device)
            noise_pred = model(x, obs, t)

            alpha_bar = sc.alphas_cumprod[t_val].to(device)
            # Fix #2: keep alpha_bar_prev on the same device as x (CUDA bug)
            if i + 1 < len(self.timesteps):
                alpha_bar_p = sc.alphas_cumprod[self.timesteps[i + 1]].to(device)
            else:
                alpha_bar_p = torch.tensor(1.0, device=device)

            # DDIM update (deterministic)
            x0_pred = (x - (1 - alpha_bar).sqrt() * noise_pred) / alpha_bar.sqrt()
            x0_pred = x0_pred.clamp(-3, 3)
            dir_xt  = (1 - alpha_bar_p).sqrt() * noise_pred
            x = alpha_bar_p.sqrt() * x0_pred + dir_xt

        return x


# ---------------------------------------------------------------------------
# Full agent (wraps model + normalisation + inference)
# ---------------------------------------------------------------------------

class DiffusionAgent:
    """
    Wraps DiffusionPolicy for env interaction.

    predict(obs_dict_sequence) → action (12D numpy array)
    """

    def __init__(
        self,
        obs_dim:        int,
        act_dim:        int,
        obs_horizon:    int,
        action_horizon: int,
        obs_mean:       np.ndarray,
        obs_std:        np.ndarray,
        act_mean:       np.ndarray,
        act_std:        np.ndarray,
        device:         str = "cpu",
        # model kwargs
        hidden_dim:     int = 512,
        n_blocks:       int = 4,
        T_ddpm:         int = 100,
        T_ddim:         int = 10,
    ):
        self.obs_dim        = obs_dim
        self.act_dim        = act_dim
        self.obs_horizon    = obs_horizon
        self.action_horizon = action_horizon
        self.device         = torch.device(device)

        self.obs_mean = torch.tensor(obs_mean, dtype=torch.float32, device=self.device)
        self.obs_std  = torch.tensor(obs_std,  dtype=torch.float32, device=self.device)
        self.act_mean = torch.tensor(act_mean, dtype=torch.float32, device=self.device)
        self.act_std  = torch.tensor(act_std,  dtype=torch.float32, device=self.device)

        self.model = MLPDenoiser(
            act_dim=act_dim, action_horizon=action_horizon,
            obs_dim=obs_dim, obs_horizon=obs_horizon,
            hidden_dim=hidden_dim, n_blocks=n_blocks,
        ).to(self.device)

        self.scheduler = DDPMScheduler(T=T_ddpm).to(self.device)
        self.sampler   = DDIMSampler(self.scheduler, n_steps=T_ddim)

        # Ring buffer for obs history
        self._obs_buffer = np.zeros((obs_horizon, obs_dim), dtype=np.float32)
        # Pre-computed action queue (execute chunk, then re-plan)
        self._action_queue: list[np.ndarray] = []

    # ------------------------------------------------------------------

    def reset(self, first_obs: np.ndarray | None = None):
        # Fix #3: prime buffer with first_obs (matches training padding with first frame).
        # Zeros would mismatch the training-time padding strategy.
        if first_obs is not None:
            self._obs_buffer[:] = first_obs
        else:
            self._obs_buffer[:] = 0.0
        self._action_queue.clear()

    def observe(self, obs_16d: np.ndarray):
        """Push a new 16D obs into the ring buffer."""
        self._obs_buffer = np.roll(self._obs_buffer, -1, axis=0)
        self._obs_buffer[-1] = obs_16d

    def predict(self, obs_16d: np.ndarray) -> np.ndarray:
        """
        Given latest 16D obs, return next action (12D).
        Re-plans every action_horizon steps.
        """
        self.observe(obs_16d)

        if not self._action_queue:
            obs_norm = (self._obs_buffer - self.obs_mean.cpu().numpy()) / self.obs_std.cpu().numpy()
            obs_t = torch.tensor(obs_norm, dtype=torch.float32, device=self.device).unsqueeze(0)

            self.model.eval()
            actions_norm = self.sampler.sample(
                self.model, obs_t,
                shape=(1, self.action_horizon, self.act_dim),
                device=self.device,
            )  # (1, action_horizon, act_dim)

            actions = (actions_norm.squeeze(0) * self.act_std + self.act_mean).cpu().numpy()
            self._action_queue = list(actions)  # list of act_dim arrays

        return self._action_queue.pop(0).clip(-1, 1)

    # ------------------------------------------------------------------

    def save(self, path: str):
        torch.save({
            "model":        self.model.state_dict(),
            "obs_mean":     self.obs_mean.cpu().numpy(),
            "obs_std":      self.obs_std.cpu().numpy(),
            "act_mean":     self.act_mean.cpu().numpy(),
            "act_std":      self.act_std.cpu().numpy(),
            "obs_dim":      self.obs_dim,
            "act_dim":      self.act_dim,
            "obs_horizon":  self.obs_horizon,
            "action_horizon": self.action_horizon,
        }, path)

    @classmethod
    def load(cls, path: str, device: str = "cpu", **kwargs) -> "DiffusionAgent":
        ckpt = torch.load(path, map_location=device, weights_only=False)
        agent = cls(
            obs_dim        = ckpt["obs_dim"],
            act_dim        = ckpt["act_dim"],
            obs_horizon    = ckpt["obs_horizon"],
            action_horizon = ckpt["action_horizon"],
            obs_mean       = ckpt["obs_mean"],
            obs_std        = ckpt["obs_std"],
            act_mean       = ckpt["act_mean"],
            act_std        = ckpt["act_std"],
            device         = device,
            **kwargs,
        )
        agent.model.load_state_dict(ckpt["model"])
        return agent
