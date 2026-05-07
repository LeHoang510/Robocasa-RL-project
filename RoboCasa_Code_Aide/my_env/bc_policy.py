"""
Behavioral Cloning policy — MLP that predicts actions from 16D robot state.

Observation format (must match bc_dataset.py):
  [base_pos(3), base_quat(4), eef_pos_rel(3), eef_quat_rel(4), gripper_qpos(2)]  = 16D

Action format (HDF5 / robosuite ordering):
  [eef_pos(3), eef_rot(3), gripper(1), base_motion(4), control_mode(1)]  = 12D

The predict_from_env() helper extracts the correct 16D obs directly from a raw
robosuite environment, so no custom gym wrapper is needed at evaluation.
"""

import numpy as np
import torch
import torch.nn as nn


# Keys used to build the 16D observation from a robosuite obs dict, in order.
# These match the LeRobot modality.json state layout.
BC_OBS_KEYS = [
    "robot0_base_pos",          # 3
    "robot0_base_quat",         # 4
    "robot0_base_to_eef_pos",   # 3
    "robot0_base_to_eef_quat",  # 4
    "robot0_gripper_qpos",      # 2  → total 16
]


def extract_bc_obs(obs_dict: dict) -> np.ndarray:
    """
    Build the 16D BC observation vector from a robosuite observation dictionary.
    Works on the raw obs dict returned by env.reset() / env.step() / env._get_observations().
    """
    return np.concatenate(
        [np.array(obs_dict[k]).flatten() for k in BC_OBS_KEYS],
        dtype=np.float32,
    )


class _MLP(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, net_arch: list[int]):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for hidden in net_arch:
            layers += [nn.Linear(in_dim, hidden), nn.ReLU()]
            in_dim = hidden
        layers.append(nn.Linear(in_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BCAgent:
    """
    Trained BC policy.  Predicts HDF5-format actions from 16D robot state.

    Usage
    -----
    agent = BCAgent.load("checkpoints/exp5_bc_<ts>/bc_best.pt")

    # From raw obs dict (e.g. after env.reset() / env.step()):
    action = agent.predict_from_obs_dict(obs_dict)

    # From pre-extracted 16D array:
    action, _ = agent.predict(obs_16d)
    """

    def __init__(
        self,
        obs_dim:      int,
        action_dim:   int,
        net_arch:     list[int],
        obs_mean:     np.ndarray,
        obs_std:      np.ndarray,
        action_low:   np.ndarray,
        action_high:  np.ndarray,
        device:       str = "cpu",
    ):
        self.device = torch.device(device)
        self.net = _MLP(obs_dim, action_dim, net_arch).to(self.device)

        self.obs_mean    = torch.tensor(obs_mean,   dtype=torch.float32, device=self.device)
        self.obs_std     = torch.tensor(obs_std,    dtype=torch.float32, device=self.device)
        self.action_low  = action_low.astype(np.float32)
        self.action_high = action_high.astype(np.float32)

    def predict(
        self,
        obs: np.ndarray,
        deterministic: bool = True,
    ) -> tuple[np.ndarray, None]:
        """
        Predict from a 16D obs array (or batched (N,16) array).
        Returns (action, None) to match the SB3 model.predict() signature.
        """
        batched = obs.ndim == 1
        if batched:
            obs = obs[None]

        t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        t = (t - self.obs_mean) / (self.obs_std + 1e-8)

        with torch.no_grad():
            action = self.net(t).cpu().numpy()

        action = np.clip(action, self.action_low, self.action_high)
        return (action[0] if batched else action), None

    def predict_from_obs_dict(self, obs_dict: dict) -> np.ndarray:
        """Convenience: extract 16D obs from raw env obs dict and predict."""
        obs_16d = extract_bc_obs(obs_dict)
        action, _ = self.predict(obs_16d)
        return action

    def save(self, path: str):
        torch.save(
            {
                "state_dict":  self.net.state_dict(),
                "obs_mean":    self.obs_mean.cpu().numpy(),
                "obs_std":     self.obs_std.cpu().numpy(),
                "action_low":  self.action_low,
                "action_high": self.action_high,
                "obs_dim":     int(self.obs_mean.shape[0]),
                "action_dim":  int(self.action_low.shape[0]),
                "net_arch":    [m.out_features for m in self.net.net
                                if isinstance(m, nn.Linear)][:-1],
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "BCAgent":
        ckpt = torch.load(path, map_location=device, weights_only=False)
        agent = cls(
            obs_dim      = ckpt["obs_dim"],
            action_dim   = ckpt["action_dim"],
            net_arch     = ckpt["net_arch"],
            obs_mean     = ckpt["obs_mean"],
            obs_std      = ckpt["obs_std"],
            action_low   = ckpt["action_low"],
            action_high  = ckpt["action_high"],
            device       = device,
        )
        agent.net.load_state_dict(ckpt["state_dict"])
        agent.net.eval()
        return agent
