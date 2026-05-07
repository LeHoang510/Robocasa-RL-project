"""
Image-conditioned behavioral cloning policy for RoboCasa pick-and-place.

Uses stacked RGB observations from one or more cameras plus the 16D proprio state.
This is the smallest imitation-learning setup in this repo that can actually
see the apple and cabinet at inference time.
"""

import numpy as np
import torch
import torch.nn as nn

from .bc_policy import extract_bc_obs


class _ImageEncoder(nn.Module):
    def __init__(self, in_channels: int, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, out_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _ImageBCNet(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        image_channels: int,
        action_dim: int,
        mlp_arch: list[int],
        image_feat_dim: int = 256,
    ):
        super().__init__()
        self.image_encoder = _ImageEncoder(image_channels, out_dim=image_feat_dim)

        layers = []
        in_dim = obs_dim + image_feat_dim
        for hidden in mlp_arch:
            layers += [nn.Linear(in_dim, hidden), nn.ReLU()]
            in_dim = hidden
        layers.append(nn.Linear(in_dim, action_dim))
        self.policy_head = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        img_feat = self.image_encoder(image)
        return self.policy_head(torch.cat([obs, img_feat], dim=-1))


class ImageBCAgent:
    def __init__(
        self,
        obs_dim: int,
        image_channels: int,
        action_dim: int,
        mlp_arch: list[int],
        camera_names: list[str],
        image_size: int,
        obs_mean: np.ndarray,
        obs_std: np.ndarray,
        action_low: np.ndarray,
        action_high: np.ndarray,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.camera_names = list(camera_names)
        self.image_size = int(image_size)
        self.net = _ImageBCNet(
            obs_dim=obs_dim,
            image_channels=image_channels,
            action_dim=action_dim,
            mlp_arch=mlp_arch,
        ).to(self.device)

        self.obs_mean = torch.tensor(obs_mean, dtype=torch.float32, device=self.device)
        self.obs_std = torch.tensor(obs_std, dtype=torch.float32, device=self.device)
        self.action_low = action_low.astype(np.float32)
        self.action_high = action_high.astype(np.float32)

    def _camera_obs_to_tensor(self, obs_dict: dict) -> torch.Tensor:
        imgs = []
        for cam in self.camera_names:
            key = f"{cam}_image"
            frame = np.array(obs_dict[key], dtype=np.float32)
            if frame.shape[0] != self.image_size or frame.shape[1] != self.image_size:
                frame_t = torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
                frame_t = torch.nn.functional.interpolate(
                    frame_t,
                    size=(self.image_size, self.image_size),
                    mode="bilinear",
                    align_corners=False,
                )
                frame = frame_t.squeeze(0).permute(1, 2, 0).cpu().numpy()
            imgs.append((frame / 255.0).transpose(2, 0, 1))
        image = np.concatenate(imgs, axis=0)
        return torch.tensor(image, dtype=torch.float32, device=self.device).unsqueeze(0)

    @torch.no_grad()
    def predict_from_obs_dict(self, obs_dict: dict) -> np.ndarray:
        obs = extract_bc_obs(obs_dict)[None]
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        obs_t = (obs_t - self.obs_mean) / (self.obs_std + 1e-8)
        img_t = self._camera_obs_to_tensor(obs_dict)
        action = self.net(obs_t, img_t).cpu().numpy()[0]
        return np.clip(action, self.action_low, self.action_high)

    def save(self, path: str):
        torch.save(
            {
                "state_dict": self.net.state_dict(),
                "obs_mean": self.obs_mean.cpu().numpy(),
                "obs_std": self.obs_std.cpu().numpy(),
                "action_low": self.action_low,
                "action_high": self.action_high,
                "camera_names": self.camera_names,
                "image_size": self.image_size,
                "obs_dim": int(self.obs_mean.shape[0]),
                "image_channels": 3 * len(self.camera_names),
                "action_dim": int(self.action_low.shape[0]),
                "mlp_arch": [
                    m.out_features for m in self.net.policy_head if isinstance(m, nn.Linear)
                ][:-1],
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "ImageBCAgent":
        ckpt = torch.load(path, map_location=device, weights_only=False)
        agent = cls(
            obs_dim=ckpt["obs_dim"],
            image_channels=ckpt["image_channels"],
            action_dim=ckpt["action_dim"],
            mlp_arch=ckpt["mlp_arch"],
            camera_names=ckpt["camera_names"],
            image_size=ckpt["image_size"],
            obs_mean=ckpt["obs_mean"],
            obs_std=ckpt["obs_std"],
            action_low=ckpt["action_low"],
            action_high=ckpt["action_high"],
            device=device,
        )
        agent.net.load_state_dict(ckpt["state_dict"])
        agent.net.eval()
        return agent
