"""
ACT dataset: re-simulates LeRobot parquet demos through PnPEnv with cameras.

Each sample:
  images  : (n_cameras, 3, H, W)  float32  [0, 1]
  proprio : (obs_dim,)             float32
  actions : (chunk_size, act_dim)  float32  — future action chunk starting at this step

Caching
-------
Re-simulation is slow (~2-3 s/episode). The collected data is saved to a .npz
file on first run; subsequent runs load instantly from cache.

Re-simulation limitation
------------------------
Demo actions are replayed in a fresh env. Physics diverges ~5-10 steps in,
so later observations won't match the original recording. This is a known
trade-off when the original HDF5 files with saved states are unavailable.
ACT is somewhat tolerant of this because action chunks span multiple steps
and the policy can re-plan at every step via temporal ensemble.
"""

from __future__ import annotations

import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

# Same action-column reorder as DemoDataset
_LEROBOT_TO_HDF5 = np.array([5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4])

# The parquet 'action' column stores a (12,) numpy array per row


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def collect_act_demos(
    dataset_dirs:     list[str],
    n_episodes_per_dir: int,
    horizon:          int,
    camera_names:     list[str],
    img_h:            int,
    img_w:            int,
    cache_path:       str,
) -> dict[str, np.ndarray]:
    """
    Re-simulate LeRobot demos through the env with camera observations.

    Returns a dict with keys:
      "images"  : (N, n_cameras, H, W, 3)  uint8
      "proprios": (N, obs_dim)              float32
      "actions" : (N, act_dim)              float32
      "episode_ids": (N,)                   int — which episode each step belongs to
    """
    if os.path.exists(cache_path):
        print(f"[ACTDataset] Loading cached demos from {cache_path}")
        data = dict(np.load(cache_path, allow_pickle=False))
        print(f"  {data['images'].shape[0]:,} steps  "
              f"{int(data['episode_ids'].max())+1} episodes")
        return data

    print(f"[ACTDataset] Collecting demos → {cache_path}")
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)

    from env.pnp_env import PnPEnv, extract_obs
    from robosuite.controllers import load_composite_controller_config

    # ------------------------------------------------------------------
    # Pass 1: count total steps so we can pre-allocate arrays.
    # This avoids accumulating images in a Python list then np.stack
    # (which would double peak memory for large image datasets).
    # ------------------------------------------------------------------
    all_parquets: list[tuple[str, np.ndarray]] = []
    for ds_dir in dataset_dirs:
        parquets = sorted(
            glob.glob(os.path.join(ds_dir, "data", "**", "episode_*.parquet"),
                      recursive=True)
        )
        parquets = parquets[:n_episodes_per_dir]
        print(f"  {ds_dir}: {len(parquets)} episodes")
        for ep_path in parquets:
            df = pd.read_parquet(ep_path)
            if "action" not in df.columns:
                continue
            acts = np.stack(df["action"].values).astype(np.float32)
            acts = acts[:, _LEROBOT_TO_HDF5]
            all_parquets.append((ep_path, acts))

    total_steps = sum(min(len(acts), horizon) for _, acts in all_parquets)
    n_cameras   = len(camera_names)
    obs_dim     = 3 + 4 + 2 + 3 + 4 + 3 + 3 + 3   # 25D privileged obs

    est_gb = total_steps * n_cameras * img_h * img_w * 3 / 1e9
    print(f"[ACTDataset] {len(all_parquets)} episodes  {total_steps:,} steps  "
          f"~{est_gb:.1f} GB images (uint8)")

    # Pre-allocate contiguous arrays — no list accumulation, no peak doubling
    images_arr   = np.empty((total_steps, n_cameras, img_h, img_w, 3), dtype=np.uint8)
    proprios_arr = np.empty((total_steps, obs_dim),  dtype=np.float32)
    actions_arr  = np.empty((total_steps, 12),        dtype=np.float32)
    ep_ids_arr   = np.empty((total_steps,),            dtype=np.int32)

    # ------------------------------------------------------------------
    # Pass 2: re-simulate and fill arrays in-place
    # ------------------------------------------------------------------
    ctrl = load_composite_controller_config(controller=None, robot="PandaOmron")
    raw_env = PnPEnv(
        robots="PandaOmron",
        controller_configs=ctrl,
        use_camera_obs=True,
        has_renderer=False,
        has_offscreen_renderer=True,
        camera_names=camera_names,
        camera_heights=img_h,
        camera_widths=img_w,
        reward_shaping=True,
        control_freq=20,
        ignore_done=False,
        horizon=horizon,
        seed=0,
    )
    raw_env.set_difficulty(0)

    ptr = 0   # write cursor into pre-allocated arrays
    for ep_global, (ep_path, demo_actions) in enumerate(
        tqdm(all_parquets, desc="  Collecting", unit="ep")
    ):
        obs_dict   = raw_env.reset()
        target_pos = _read_target_pos(raw_env)

        ep_start = ptr
        for act in demo_actions[:horizon]:
            imgs = np.stack([
                obs_dict[f"{cam}_image"] for cam in camera_names
            ], axis=0)                                 # (n_cam, H, W, 3)
            images_arr[ptr]   = imgs
            proprios_arr[ptr] = extract_obs(obs_dict, target_pos)
            actions_arr[ptr]  = act
            ep_ids_arr[ptr]   = ep_global
            ptr += 1

            obs_dict, _, done, _ = raw_env.step(act)
            if done:
                break

    raw_env.close()

    # Trim to actual steps collected (early dones may shorten some episodes)
    data = {
        "images":      images_arr[:ptr],
        "proprios":    proprios_arr[:ptr],
        "actions":     actions_arr[:ptr],
        "episode_ids": ep_ids_arr[:ptr],
    }
    np.savez_compressed(cache_path, **data)
    print(f"[ACTDataset] Saved {ptr:,} steps ({ep_global+1} episodes) → {cache_path}")
    return data


def _read_target_pos(raw_env) -> np.ndarray:
    try:
        cab    = raw_env.cab
        offset = cab.get_reset_regions(raw_env)["level0"]["offset"]
        return (np.array(cab.pos) + np.array(offset)).astype(np.float32)
    except Exception:
        return np.array([2.25, -0.2, 1.42], dtype=np.float32)


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class ACTDataset(Dataset):
    """
    Each sample is (images, proprio, action_chunk) where action_chunk is the
    next `chunk_size` actions from the current step (same episode, zero-padded
    at episode boundaries).

    Parameters
    ----------
    data       : dict returned by collect_act_demos
    chunk_size : number of future actions to predict (e.g. 100)
    indices    : optional subset of valid step indices (for train/val split)
    """

    def __init__(
        self,
        data:       dict[str, np.ndarray],
        chunk_size: int = 100,
        indices:    np.ndarray | None = None,
    ):
        self.chunk_size = chunk_size
        self.images      = data["images"]      # (N, n_cam, H, W, 3) uint8
        self.proprios    = data["proprios"]    # (N, obs_dim)
        self.actions     = data["actions"]     # (N, act_dim)
        self.episode_ids = data["episode_ids"] # (N,)

        # Build valid indices: must have at least chunk_size steps left in episode
        if indices is not None:
            self._idx = indices
        else:
            self._idx = self._build_indices()

    def _build_indices(self) -> np.ndarray:
        N = len(self.actions)
        valid = []
        for i in range(N):
            # Find how many steps remain in this episode
            eid   = self.episode_ids[i]
            ep_end = i
            while ep_end < N and self.episode_ids[ep_end] == eid:
                ep_end += 1
            remaining = ep_end - i
            if remaining >= 1:   # include all steps; pad short chunks at end
                valid.append(i)
        return np.array(valid, dtype=np.int64)

    def __len__(self):
        return len(self._idx)

    def __getitem__(self, item):
        i   = int(self._idx[item])
        eid = self.episode_ids[i]

        # Collect chunk, padding with last action if episode ends early
        N   = len(self.actions)
        chunk = []
        for k in range(self.chunk_size):
            j = i + k
            if j < N and self.episode_ids[j] == eid:
                chunk.append(self.actions[j])
            else:
                chunk.append(chunk[-1])   # repeat last action as padding
        actions = np.stack(chunk, axis=0).astype(np.float32)  # (chunk_size, act_dim)

        # Image: (n_cam, H, W, 3) uint8 → float32 [0,1], channel first
        img_np  = self.images[i].astype(np.float32) / 255.0   # (n_cam, H, W, 3)
        images  = torch.from_numpy(img_np).permute(0, 3, 1, 2)  # (n_cam, 3, H, W)

        proprio = torch.from_numpy(self.proprios[i])
        actions = torch.from_numpy(actions)

        return images, proprio, actions

    # ------------------------------------------------------------------
    # Episode-level train / val split (no cross-episode leakage)
    # ------------------------------------------------------------------

    def episode_split(
        self, val_frac: float = 0.1, seed: int = 0
    ) -> tuple["ACTDataset", "ACTDataset"]:
        rng      = np.random.default_rng(seed)
        all_eps  = np.unique(self.episode_ids)
        rng.shuffle(all_eps)
        n_val    = max(1, int(len(all_eps) * val_frac))
        val_eps  = set(all_eps[:n_val].tolist())
        train_eps = set(all_eps[n_val:].tolist())

        train_idx = self._idx[np.isin(self.episode_ids[self._idx], list(train_eps))]
        val_idx   = self._idx[np.isin(self.episode_ids[self._idx], list(val_eps))]

        data = {
            "images":      self.images,
            "proprios":    self.proprios,
            "actions":     self.actions,
            "episode_ids": self.episode_ids,
        }
        return (
            ACTDataset(data, self.chunk_size, train_idx),
            ACTDataset(data, self.chunk_size, val_idx),
        )
