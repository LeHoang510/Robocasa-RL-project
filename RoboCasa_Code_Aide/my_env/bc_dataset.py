"""
PyTorch Dataset for Behavioral Cloning from robocasa LeRobot-format datasets.

Dataset layout (produced by robocasa download_datasets.py):
  <dataset_dir>/
    data/
      chunk-000/episode_000000.parquet
      chunk-000/episode_000001.parquet
      ...
    meta/
      info.json        ← total_episodes, splits, fps, data_path template
      modality.json    ← obs/action key layouts

Each parquet file is one episode and has columns:
  observation.state  – 16D robot proprioceptive state (object positions NOT included):
                       [base_pos(3), base_quat(4), eef_pos_rel(3), eef_quat_rel(4), gripper(2)]
  action             – 12D in LeRobot ordering:
                       [base_motion(4), control_mode(1), eef_pos(3), eef_rot(3), gripper(1)]

The env's step() expects actions in HDF5 ordering:
  [eef_pos(3), eef_rot(3), gripper(1), base_motion(4), control_mode(1)]

This module reorders actions to HDF5 format at load time so the trained policy
can be passed directly to env.step() at evaluation.
"""

import json
from collections import OrderedDict
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


# LeRobot → HDF5 action reordering index
# LeRobot:  [base_motion(0:4), control_mode(4:5), eef_pos(5:8), eef_rot(8:11), gripper(11:12)]
# HDF5:     [eef_pos(0:3),     eef_rot(3:6),      gripper(6:7), base_motion(7:11), ctrl(11:12)]
_LEROBOT_TO_HDF5 = np.array([5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4])


def reorder_action_lerobot_to_hdf5(action: np.ndarray) -> np.ndarray:
    """Convert action(s) from LeRobot column ordering to robosuite/HDF5 ordering."""
    return action[..., _LEROBOT_TO_HDF5]


def _load_split_episodes(dataset_dir: str, split: str) -> list[int]:
    """
    Parse the split range from info.json and return a list of episode indices.
    Split format: "start:end"  (e.g. "0:100" means episodes 0..99).
    """
    info_path = Path(dataset_dir) / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)

    if split == "all":
        return list(range(int(info["total_episodes"])))

    split_str = info["splits"].get(split)
    if split_str is None:
        raise ValueError(
            f"Split '{split}' not found in {info_path}. "
            f"Available: {list(info['splits'].keys())}"
        )
    start, end = map(int, split_str.split(":"))
    return list(range(start, end))


def _episode_parquet_path(dataset_dir: str, info: dict, episode_index: int) -> Path:
    chunk_size = info["chunks_size"]
    chunk_id   = episode_index // chunk_size
    template   = info["data_path"]
    rel_path   = template.format(
        episode_chunk=chunk_id,
        episode_index=episode_index,
    )
    return Path(dataset_dir) / rel_path


def _episode_video_path(
    dataset_dir: str,
    info: dict,
    episode_index: int,
    video_key: str,
) -> Path:
    chunk_size = info["chunks_size"]
    chunk_id   = episode_index // chunk_size
    template   = info["video_path"]
    rel_path   = template.format(
        episode_chunk=chunk_id,
        episode_index=episode_index,
        video_key=video_key,
    )
    return Path(dataset_dir) / rel_path


def _resolve_video_key(info: dict, camera_key: str) -> str:
    """
    Map a RoboCasa camera name such as 'robot0_eye_in_hand' to the dataset's
    video folder key such as 'observation.images.robot0_eye_in_hand'.
    """
    features = info.get("features", {})
    direct_key = f"observation.images.{camera_key}"
    if direct_key in features:
        return direct_key
    if camera_key in features:
        return camera_key
    available = sorted(k for k in features if k.startswith("observation.images."))
    raise KeyError(
        f"Camera '{camera_key}' not found in dataset video features. "
        f"Available keys: {available}"
    )


class BCDemoDataset(Dataset):
    """
    Parameters
    ----------
    dataset_dir  : Root directory of the LeRobot dataset (contains data/, meta/).
    split        : Dataset split to use, e.g. "train".
    max_episodes : Cap on number of episodes to load (None = all in the split).
    reorder_action: If True (default), reorder actions from LeRobot to HDF5 format.
    """

    def __init__(
        self,
        dataset_dir:    str,
        split:          str = "train",
        max_episodes:   int | None = None,
        reorder_action: bool = True,
        verbose:        bool = True,
    ):
        super().__init__()
        self.dataset_dir = dataset_dir

        info_path = Path(dataset_dir) / "meta" / "info.json"
        with open(info_path) as f:
            info = json.load(f)

        episode_indices = _load_split_episodes(dataset_dir, split)
        if max_episodes is not None:
            episode_indices = episode_indices[:max_episodes]

        if verbose:
            print(f"[BCDataset] Loading {len(episode_indices)} episodes "
                  f"(split='{split}') from {dataset_dir}")

        all_obs, all_actions = [], []

        for ep_idx in episode_indices:
            path = _episode_parquet_path(dataset_dir, info, ep_idx)
            df   = pd.read_parquet(path)

            obs     = np.stack(df["observation.state"].to_numpy()).astype(np.float32)
            actions = np.stack(df["action"].to_numpy()).astype(np.float32)

            if reorder_action:
                actions = reorder_action_lerobot_to_hdf5(actions)

            all_obs.append(obs)
            all_actions.append(actions)

        self.obs     = torch.tensor(np.concatenate(all_obs),     dtype=torch.float32)
        self.actions = torch.tensor(np.concatenate(all_actions), dtype=torch.float32)

        if verbose:
            print(f"[BCDataset] {len(self.obs)} transitions  "
                  f"obs_dim={self.obs_dim}  action_dim={self.action_dim}")

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return self.obs[idx], self.actions[idx]

    @property
    def obs_dim(self) -> int:
        return self.obs.shape[1]

    @property
    def action_dim(self) -> int:
        return self.actions.shape[1]

    def compute_normalisation(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (mean, std) over observations for input normalisation."""
        obs_np = self.obs.numpy()
        return obs_np.mean(0).astype(np.float32), obs_np.std(0).astype(np.float32)


class OfflineRLDataset(Dataset):
    """
    Loads (obs, action, reward, next_obs, done) transition tuples for offline RL.

    Constructs next_obs by shifting obs by one timestep within each episode.
    Actions are reordered from LeRobot → HDF5 ordering (same as BCDemoDataset).
    """

    def __init__(
        self,
        dataset_dir:    str,
        split:          str = "train",
        max_episodes:   int | None = None,
        verbose:        bool = True,
    ):
        super().__init__()

        info_path = Path(dataset_dir) / "meta" / "info.json"
        with open(info_path) as f:
            info = json.load(f)

        episode_indices = _load_split_episodes(dataset_dir, split)
        if max_episodes is not None:
            episode_indices = episode_indices[:max_episodes]

        if verbose:
            print(f"[OfflineRLDataset] Loading {len(episode_indices)} episodes "
                  f"(split='{split}') from {dataset_dir}")

        all_obs, all_actions, all_rewards, all_next_obs, all_dones = [], [], [], [], []

        for ep_idx in episode_indices:
            path    = _episode_parquet_path(dataset_dir, info, ep_idx)
            df      = pd.read_parquet(path)

            obs     = np.stack(df["observation.state"].to_numpy()).astype(np.float32)
            actions = np.stack(df["action"].to_numpy()).astype(np.float32)
            actions = reorder_action_lerobot_to_hdf5(actions)
            rewards = df["next.reward"].to_numpy().astype(np.float32).reshape(-1, 1)
            dones   = df["next.done"].to_numpy().astype(np.float32).reshape(-1, 1)

            # next_obs[t] = obs[t+1]; for the terminal step use obs[t] itself
            # (the (1-done) mask in the Bellman backup zeroes it out anyway)
            next_obs = np.concatenate([obs[1:], obs[-1:]], axis=0)

            all_obs.append(obs)
            all_actions.append(actions)
            all_rewards.append(rewards)
            all_next_obs.append(next_obs)
            all_dones.append(dones)

        self.obs      = torch.tensor(np.concatenate(all_obs),      dtype=torch.float32)
        self.actions  = torch.tensor(np.concatenate(all_actions),  dtype=torch.float32)
        self.rewards  = torch.tensor(np.concatenate(all_rewards),  dtype=torch.float32)
        self.next_obs = torch.tensor(np.concatenate(all_next_obs), dtype=torch.float32)
        self.dones    = torch.tensor(np.concatenate(all_dones),    dtype=torch.float32)

        if verbose:
            print(f"[OfflineRLDataset] {len(self.obs)} transitions  "
                  f"obs_dim={self.obs.shape[1]}  action_dim={self.actions.shape[1]}  "
                  f"reward_range=[{self.rewards.min():.3f}, {self.rewards.max():.3f}]")

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return (
            self.obs[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_obs[idx],
            self.dones[idx],
        )

    @property
    def obs_dim(self) -> int:
        return self.obs.shape[1]

    @property
    def action_dim(self) -> int:
        return self.actions.shape[1]

    def compute_normalisation(self) -> tuple[np.ndarray, np.ndarray]:
        obs_np = self.obs.numpy()
        return obs_np.mean(0).astype(np.float32), obs_np.std(0).astype(np.float32)


class ImageBCDemoDataset(Dataset):
    """
    Image-conditioned BC dataset.

    Each sample contains:
      - 16D proprioceptive state
      - stacked RGB images from one or more camera videos
      - action in HDF5 ordering

    Images are loaded lazily per episode to keep memory usage manageable.
    """

    def __init__(
        self,
        dataset_dir: str,
        split: str = "all",
        max_episodes: int | None = None,
        camera_keys: tuple[str, ...] = ("robot0_agentview_left", "robot0_eye_in_hand"),
        image_size: int = 84,
        cache_num_episodes: int = 8,
        reorder_action: bool = True,
        verbose: bool = True,
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.camera_keys = tuple(camera_keys)
        self.image_size  = int(image_size)
        self.cache_num_episodes = int(cache_num_episodes)

        info_path = Path(dataset_dir) / "meta" / "info.json"
        with open(info_path) as f:
            self.info = json.load(f)

        episode_indices = _load_split_episodes(dataset_dir, split)
        if max_episodes is not None:
            episode_indices = episode_indices[:max_episodes]

        self._episode_meta = []
        self._index = []

        if verbose:
            print(
                f"[ImageBCDataset] Indexing {len(episode_indices)} episodes "
                f"(split='{split}') from {dataset_dir}"
            )

        for local_ep_idx, episode_index in enumerate(episode_indices):
            parquet_path = _episode_parquet_path(dataset_dir, self.info, episode_index)
            df = pd.read_parquet(parquet_path, columns=["observation.state", "action"])

            obs = np.stack(df["observation.state"].to_numpy()).astype(np.float32)
            actions = np.stack(df["action"].to_numpy()).astype(np.float32)
            if reorder_action:
                actions = reorder_action_lerobot_to_hdf5(actions)

            video_paths = {
                cam: _episode_video_path(
                    dataset_dir,
                    self.info,
                    episode_index,
                    _resolve_video_key(self.info, cam),
                )
                for cam in self.camera_keys
            }
            reader_lengths = []
            for path in video_paths.values():
                reader = imageio.get_reader(path)
                reader_lengths.append(len(reader))
                reader.close()
            length = min(len(obs), len(actions), *reader_lengths)
            self._episode_meta.append(
                {
                    "obs": obs[:length],
                    "actions": actions[:length],
                    "video_paths": video_paths,
                    "episode_index": episode_index,
                    "length": length,
                }
            )
            self._index.extend((local_ep_idx, t) for t in range(length))

        self._video_cache: OrderedDict[int, dict[str, list[np.ndarray]]] = OrderedDict()

        if verbose:
            print(
                f"[ImageBCDataset] {len(self._index)} transitions  "
                f"obs_dim={self.obs_dim}  action_dim={self.action_dim}  "
                f"cams={list(self.camera_keys)}  image_size={self.image_size}"
            )

    def _load_episode_frames(self, local_ep_idx: int) -> dict[str, list[np.ndarray]]:
        if local_ep_idx in self._video_cache:
            frames = self._video_cache.pop(local_ep_idx)
            self._video_cache[local_ep_idx] = frames
            return frames

        meta = self._episode_meta[local_ep_idx]
        frames_by_cam = {}
        for cam, path in meta["video_paths"].items():
            reader = imageio.get_reader(path)
            frames = [frame for frame in reader]
            reader.close()
            frames_by_cam[cam] = frames
        self._video_cache[local_ep_idx] = frames_by_cam
        while len(self._video_cache) > self.cache_num_episodes:
            self._video_cache.popitem(last=False)
        return frames_by_cam

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        frame_t = torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        resized = F.interpolate(
            frame_t,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        return (resized.squeeze(0).numpy() / 255.0).astype(np.float32)

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        local_ep_idx, step_idx = self._index[idx]
        meta = self._episode_meta[local_ep_idx]
        frames_by_cam = self._load_episode_frames(local_ep_idx)

        obs = torch.tensor(meta["obs"][step_idx], dtype=torch.float32)
        action = torch.tensor(meta["actions"][step_idx], dtype=torch.float32)

        imgs = []
        for cam in self.camera_keys:
            frame = frames_by_cam[cam][step_idx]
            imgs.append(self._preprocess_frame(frame))
        image = torch.tensor(np.concatenate(imgs, axis=0), dtype=torch.float32)

        return obs, image, action

    @property
    def obs_dim(self) -> int:
        return int(self._episode_meta[0]["obs"].shape[1])

    @property
    def action_dim(self) -> int:
        return int(self._episode_meta[0]["actions"].shape[1])

    @property
    def image_channels(self) -> int:
        return 3 * len(self.camera_keys)

    def compute_normalisation(self) -> tuple[np.ndarray, np.ndarray]:
        all_obs = np.concatenate([m["obs"] for m in self._episode_meta], axis=0)
        return all_obs.mean(0).astype(np.float32), all_obs.std(0).astype(np.float32)
