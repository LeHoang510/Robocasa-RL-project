"""
Privileged demo dataset for Diffusion Policy training.

Re-simulates demo episodes through PrivilegedPnPEnv to collect
(25D privileged obs, 12D action) pairs, then builds sliding-window
(obs_history, action_chunk) samples — same interface as DemoDataset.

25D obs layout:
  robot0_base_to_eef_pos   (3)
  robot0_base_to_eef_quat  (4)
  robot0_gripper_qpos      (2)
  obj_pos                  (3)
  obj_quat                 (4)
  obj_to_robot0_eef_pos    (3)
  target_pos               (3)
  obj_to_target            (3)

Results are cached to disk so re-simulation only runs once.
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

# Action reorder: LeRobot column order → robosuite/HDF5 env order
_LEROBOT_TO_HDF5 = np.array([5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4])


def _make_raw_env(horizon: int, seed: int = 0):
    from env.pnp_env import make_env
    return make_env(horizon=horizon, seed=seed, privileged=False)


def collect_privileged_demos(
    dataset_dirs: list[str],
    n_episodes_per_dir: int,
    horizon: int,
    cache_path: str,
    verbose: bool = True,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Re-simulate demo episodes through PrivilegedPnPEnv.
    Returns per-episode obs list and actions list (not yet windowed).
    Results are cached to cache_path so this only runs once.
    """
    if os.path.exists(cache_path):
        if verbose:
            print(f"  [Cache] Loading from {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        all_obs     = list(data["obs"])
        all_actions = list(data["actions"])
        if verbose:
            total = sum(len(o) for o in all_obs)
            print(f"  [Cache] {len(all_obs)} episodes, {total:,} transitions")
        return all_obs, all_actions

    from env.pnp_env import PrivilegedPnPEnv, extract_obs as extract_privileged_obs

    raw_env = _make_raw_env(horizon, seed=0)
    env     = PrivilegedPnPEnv(raw_env)

    all_obs:     list[np.ndarray] = []
    all_actions: list[np.ndarray] = []

    for ds_dir in dataset_dirs:
        parquets = sorted(glob.glob(
            os.path.join(ds_dir, "data", "chunk-*", "episode_*.parquet")
        ))[:n_episodes_per_dir]

        if verbose:
            print(f"  Re-simulating {len(parquets)} episodes from {os.path.basename(ds_dir)} …")

        for p in tqdm(parquets, leave=False, disable=not verbose):
            df      = pd.read_parquet(p)
            actions = np.stack(df["action"].values).astype(np.float32)
            actions = actions[:, _LEROBOT_TO_HDF5]

            obs_dict = env.raw_env.reset()
            env._target_pos    = env._read_target_pos()
            env._initial_obj_z = float(np.array(obs_dict["obj_pos"])[2])

            ep_obs, ep_acts = [], []
            for t, action in enumerate(actions):
                obs_25d = extract_privileged_obs(obs_dict, env._target_pos)
                ep_obs.append(obs_25d)
                ep_acts.append(action)

                obs_dict, _, done, _ = env.raw_env.step(action)
                if done:
                    break

            if len(ep_obs) > 0:
                all_obs.append(np.array(ep_obs,  dtype=np.float32))
                all_actions.append(np.array(ep_acts, dtype=np.float32))

    env.close()

    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    np.savez_compressed(cache_path, obs=np.array(all_obs, dtype=object),
                        actions=np.array(all_actions, dtype=object))

    total = sum(len(o) for o in all_obs)
    if verbose:
        print(f"  [Cache] Saved {len(all_obs)} episodes, {total:,} transitions → {cache_path}")

    return all_obs, all_actions


class PrivilegedDemoDataset(Dataset):
    """
    (obs_history 25D, action_chunk 12D) pairs from re-simulated demos.
    Same interface as DemoDataset but with privileged 25D observations.
    """

    def __init__(
        self,
        dataset_dirs:       list[str],
        n_episodes_per_dir: int  = 500,
        obs_horizon:        int  = 2,
        action_horizon:     int  = 8,
        horizon:            int  = 300,
        cache_path:         str  = "src/checkpoints/privileged_cache.npz",
        verbose:            bool = True,
    ):
        self.obs_horizon    = obs_horizon
        self.action_horizon = action_horizon

        all_obs, all_actions = collect_privileged_demos(
            dataset_dirs       = dataset_dirs,
            n_episodes_per_dir = n_episodes_per_dir,
            horizon            = horizon,
            cache_path         = cache_path,
            verbose            = verbose,
        )

        self._obs     = all_obs
        self._actions = all_actions
        self._index:          list[tuple[int, int]] = []
        self._episode_of_index: list[int]           = []

        for ep_idx, obs in enumerate(self._obs):
            T = len(obs)
            for t in range(T - action_horizon + 1):
                self._index.append((ep_idx, t))
                self._episode_of_index.append(ep_idx)

        all_obs_cat = np.concatenate(all_obs, axis=0)
        all_act_cat = np.concatenate(all_actions, axis=0)

        self.obs_mean = all_obs_cat.mean(0).astype(np.float32)
        self.obs_std  = (all_obs_cat.std(0) + 1e-8).astype(np.float32)
        self.act_mean = all_act_cat.mean(0).astype(np.float32)
        self.act_std  = (all_act_cat.std(0) + 1e-8).astype(np.float32)
        self.obs_dim  = all_obs_cat.shape[-1]   # 25
        self.act_dim  = all_act_cat.shape[-1]   # 12

        if verbose:
            print(f"  Total samples : {len(self._index):,}  "
                  f"(obs_dim={self.obs_dim}, act_dim={self.act_dim})")

    # ------------------------------------------------------------------

    def episode_split(self, val_frac: float = 0.05, seed: int = 42):
        n_episodes = len(self._obs)
        rng = np.random.default_rng(seed)
        ep_ids = np.arange(n_episodes)
        rng.shuffle(ep_ids)
        n_val_ep = max(1, int(n_episodes * val_frac))
        val_eps   = set(ep_ids[:n_val_ep].tolist())
        train_eps = set(ep_ids[n_val_ep:].tolist())
        train_idx = [i for i, ep in enumerate(self._episode_of_index) if ep in train_eps]
        val_idx   = [i for i, ep in enumerate(self._episode_of_index) if ep in val_eps]
        return train_idx, val_idx

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int):
        ep_idx, t = self._index[idx]
        obs_ep  = self._obs[ep_idx]
        act_ep  = self._actions[ep_idx]

        obs_frames = []
        for k in range(self.obs_horizon):
            src = max(0, t - (self.obs_horizon - 1 - k))
            obs_frames.append(obs_ep[src])
        obs_stack = np.stack(obs_frames, axis=0)

        obs_norm  = (obs_stack - self.obs_mean) / self.obs_std
        acts      = act_ep[t : t + self.action_horizon]
        acts_norm = (acts - self.act_mean) / self.act_std

        return (
            torch.tensor(obs_norm,  dtype=torch.float32),
            torch.tensor(acts_norm, dtype=torch.float32),
        )
