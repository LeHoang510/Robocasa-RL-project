"""
Demo dataset loader for Diffusion Policy training.

Loads (obs_history, action_chunk) pairs from LeRobot parquet datasets.
Supports combining multiple dataset directories (e.g. human + mimicgen).

Each sample:
  obs     : (obs_horizon, obs_dim)   — stacked consecutive observations
  actions : (action_horizon, act_dim) — future action chunk
"""

import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# LeRobot stores actions in a different column order than robosuite env.step() expects.
# Without this reorder, the policy outputs actions the env interprets completely wrong.
_LEROBOT_TO_HDF5 = np.array([5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4])


class DemoDataset(Dataset):
    """
    Parameters
    ----------
    dataset_dirs : list[str]
        Paths to LeRobot dataset roots (each must contain data/chunk-*/episode_*.parquet).
    obs_horizon  : int   — number of obs frames to stack (default 2, gives velocity info)
    action_horizon : int — number of future actions to predict (default 8)
    max_episodes : int | None — cap on episodes per dataset dir (None = all)
    """

    def __init__(
        self,
        dataset_dirs: list[str],
        obs_horizon: int = 2,
        action_horizon: int = 8,
        max_episodes: int | None = None,
        verbose: bool = True,
    ):
        self.obs_horizon    = obs_horizon
        self.action_horizon = action_horizon

        all_obs:     list[np.ndarray] = []  # (T, obs_dim) per episode
        all_actions: list[np.ndarray] = []  # (T, act_dim) per episode

        for ds_dir in dataset_dirs:
            parquets = sorted(glob.glob(
                os.path.join(ds_dir, "data", "chunk-*", "episode_*.parquet")
            ))
            if not parquets:
                raise FileNotFoundError(f"No parquet files found in {ds_dir}")

            if max_episodes is not None:
                parquets = parquets[:max_episodes]

            for p in parquets:
                df = pd.read_parquet(p)
                obs  = np.stack(df["observation.state"].values).astype(np.float32)
                acts = np.stack(df["action"].values).astype(np.float32)
                # Fix #1: reorder LeRobot actions to robosuite/HDF5 env order
                acts = acts[:, _LEROBOT_TO_HDF5]
                all_obs.append(obs)
                all_actions.append(acts)

            if verbose:
                print(f"  Loaded {len(parquets):>5} episodes from {ds_dir}")

        # Build flat index: (episode_idx, start_step)
        self._obs     = all_obs
        self._actions = all_actions
        self._index   = []  # list of (ep_idx, t)
        # Track which episode each window belongs to (for episode-level split)
        self._episode_of_index: list[int] = []

        for ep_idx, obs in enumerate(self._obs):
            T = len(obs)
            # need at least action_horizon future steps
            for t in range(T - action_horizon + 1):
                self._index.append((ep_idx, t))
                self._episode_of_index.append(ep_idx)

        # Compute normalisation stats over all obs
        all_obs_cat = np.concatenate(all_obs, axis=0)
        self.obs_mean = all_obs_cat.mean(0).astype(np.float32)
        self.obs_std  = (all_obs_cat.std(0) + 1e-8).astype(np.float32)

        all_act_cat = np.concatenate(all_actions, axis=0)
        self.act_mean = all_act_cat.mean(0).astype(np.float32)
        self.act_std  = (all_act_cat.std(0) + 1e-8).astype(np.float32)

        self.obs_dim = all_obs_cat.shape[-1]
        self.act_dim = all_act_cat.shape[-1]

        if verbose:
            total_ep = sum(len(o) for o in all_obs)
            print(f"  Total samples : {len(self._index):,}  "
                  f"(obs_dim={self.obs_dim}, act_dim={self.act_dim})")
            print(f"  Obs horizon   : {obs_horizon}  "
                  f"Action horizon : {action_horizon}")

    def episode_split(self, val_frac: float = 0.05, seed: int = 42):
        """
        Fix #4: split by episode, not by window, to prevent data leakage.
        Returns (train_indices, val_indices) as lists of dataset sample indices.
        """
        n_episodes = len(self._obs)
        rng = np.random.default_rng(seed)
        ep_ids = np.arange(n_episodes)
        rng.shuffle(ep_ids)
        n_val_ep = max(1, int(n_episodes * val_frac))
        val_episodes  = set(ep_ids[:n_val_ep].tolist())
        train_episodes = set(ep_ids[n_val_ep:].tolist())

        train_idx = [i for i, ep in enumerate(self._episode_of_index) if ep in train_episodes]
        val_idx   = [i for i, ep in enumerate(self._episode_of_index) if ep in val_episodes]
        return train_idx, val_idx

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int):
        ep_idx, t = self._index[idx]
        obs_ep  = self._obs[ep_idx]
        act_ep  = self._actions[ep_idx]

        # Obs history: pad with first frame if t < obs_horizon-1
        obs_frames = []
        for k in range(self.obs_horizon):
            src = max(0, t - (self.obs_horizon - 1 - k))
            obs_frames.append(obs_ep[src])
        obs_stack = np.stack(obs_frames, axis=0)  # (obs_horizon, obs_dim)

        # Normalise obs
        obs_norm = (obs_stack - self.obs_mean) / self.obs_std

        # Action chunk
        acts = act_ep[t : t + self.action_horizon]  # (action_horizon, act_dim)
        acts_norm = (acts - self.act_mean) / self.act_std

        return (
            torch.tensor(obs_norm,  dtype=torch.float32),
            torch.tensor(acts_norm, dtype=torch.float32),
        )
