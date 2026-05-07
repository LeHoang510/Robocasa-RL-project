"""
BC Pre-training for SAC Privileged Actor — PnPCounterToCab
==========================================================
Re-simulates demo episodes through PrivilegedPnPEnv to collect
(25D privileged obs, 12D action) pairs, then trains an MLP that
matches the SAC actor architecture exactly.

The saved weights are loaded into SAC via --bc_weights to bootstrap
the actor with grasping behavior before online RL begins.

Usage
-----
# Step 1: BC pre-train
python RoboCasa_Code_Aide/my_rl_scripts/train_bc_privileged.py \\
    --config RoboCasa_Code_Aide/config/exp9_sac_privileged.yaml \\
    --n_demos 50 --epochs 200

# Step 2: SAC fine-tune from BC weights
python RoboCasa_Code_Aide/my_rl_scripts/train_sac_privileged.py \\
    --config RoboCasa_Code_Aide/config/exp9_sac_privileged.yaml \\
    --bc_weights checkpoints/bc_priv_<timestamp>.pt
"""

import argparse
import os
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, TensorDataset, random_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from my_env import MyPnPCounterToCab
from my_env.bc_dataset import BCDemoDataset
from my_env.privileged_env import PrivilegedPnPEnv, extract_privileged_obs
from robosuite.controllers import load_composite_controller_config


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def _make_raw_env(cfg: dict, seed: int = 0):
    controller_config = load_composite_controller_config(controller=None, robot="PandaOmron")
    return MyPnPCounterToCab(
        robots="PandaOmron",
        controller_configs=controller_config,
        use_camera_obs=False,
        has_renderer=False,
        has_offscreen_renderer=False,
        reward_shaping=True,
        control_freq=20,
        ignore_done=False,
        horizon=cfg["env"]["horizon"],
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Demo collection
# ---------------------------------------------------------------------------

def collect_privileged_demos(cfg: dict, n_episodes: int, verbose: bool = True):
    """
    Re-simulate demo episodes through PrivilegedPnPEnv.

    Actions come from the LeRobot dataset; the env state differs from the
    original demo (different random seed), but the motions encode realistic
    reaching and grasping behavior that SAC can bootstrap from.

    Returns obs (N, 25) and actions (N, 12) as float32 numpy arrays.
    """
    d_cfg   = cfg["dataset"]
    dataset = BCDemoDataset(
        dataset_dir  = d_cfg["dataset_dir"],
        split        = "train",
        max_episodes = n_episodes,
        verbose      = False,
    )

    raw = _make_raw_env(cfg, seed=0)
    env = PrivilegedPnPEnv(raw)

    all_obs, all_actions = [], []
    ep_len = max(1, len(dataset) // max(n_episodes, 1))

    for ep_idx in range(n_episodes):
        obs_dict = env.raw_env.reset()
        env._target_pos    = env._read_target_pos()
        env._initial_obj_z = float(np.array(obs_dict["obj_pos"])[2])
        obs = extract_privileged_obs(obs_dict, env._target_pos)

        for step in range(ep_len):
            act_idx = min(ep_idx * ep_len + step, len(dataset) - 1)
            action  = dataset.actions[act_idx].numpy()

            all_obs.append(obs.copy())
            all_actions.append(action.copy())

            obs_dict, _, done, _ = env.raw_env.step(action)
            obs = extract_privileged_obs(obs_dict, env._target_pos)
            if done:
                break

        if verbose:
            print(f"  [Demo] episode {ep_idx + 1}/{n_episodes}  "
                  f"total transitions so far: {len(all_obs)}", end="\r")

    env.close()
    if verbose:
        print(f"\n[BC] Collected {len(all_obs)} transitions from {n_episodes} episodes")

    return (
        np.array(all_obs,     dtype=np.float32),
        np.array(all_actions, dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# Network — must match SAC actor architecture exactly
# ---------------------------------------------------------------------------

class PrivBCNet(nn.Module):
    """
    MLP mirroring the SB3 SAC actor layout for net_arch=[256, 256, 256]:
      latent_pi : Linear(25,256) ReLU Linear(256,256) ReLU Linear(256,256) ReLU
      output    : Linear(256, 12)   ← maps to SAC actor.mu at load time
    """

    def __init__(self, obs_dim: int = 25, action_dim: int = 12,
                 net_arch: tuple = (256, 256, 256)):
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

    def linear_layers(self) -> list[nn.Linear]:
        return [m for m in self.net if isinstance(m, nn.Linear)]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="BC pre-training for privileged SAC actor")
    parser.add_argument("--config",     type=str,   required=True)
    parser.add_argument("--n_demos",    type=int,   default=50,
                        help="Number of demo episodes to re-simulate (default 50)")
    parser.add_argument("--epochs",     type=int,   default=200)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int,   default=256)
    parser.add_argument("--save_path",  type=str,   default=None,
                        help="Output .pt path (default: checkpoints/bc_priv_<ts>.pt)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = args.save_path or f"checkpoints/bc_priv_{timestamp}.pt"
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)

    # ------------------------------------------------------------------
    # Collect demonstrations
    # ------------------------------------------------------------------
    print(f"[BC] Collecting {args.n_demos} demo episodes through privileged env …")
    obs_np, act_np = collect_privileged_demos(cfg, n_episodes=args.n_demos, verbose=True)

    obs_mean = obs_np.mean(0).astype(np.float32)
    obs_std  = (obs_np.std(0) + 1e-8).astype(np.float32)

    obs_norm = (obs_np - obs_mean) / obs_std
    obs_t    = torch.tensor(obs_norm, dtype=torch.float32)
    act_t    = torch.tensor(act_np,   dtype=torch.float32)

    full_ds  = TensorDataset(obs_t, act_t)
    n_val    = max(1, int(len(full_ds) * 0.1))
    n_train  = len(full_ds) - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=512,             shuffle=False, num_workers=0)

    print(f"[BC] {n_train} train  {n_val} val  transitions | obs_dim=25  act_dim=12")

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    net     = PrivBCNet().to(device)
    opt     = torch.optim.Adam(net.parameters(), lr=args.lr)
    sch     = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.01)
    loss_fn = nn.MSELoss()
    best_val_loss = float("inf")

    print(f"[BC] Training {args.epochs} epochs on {device} …\n")

    for epoch in range(1, args.epochs + 1):
        net.train()
        train_loss = 0.0
        for obs_b, act_b in train_loader:
            obs_b, act_b = obs_b.to(device), act_b.to(device)
            loss = loss_fn(net(obs_b), act_b)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            train_loss += loss.item() * len(obs_b)
        train_loss /= n_train

        net.eval()
        val_loss = 0.0
        with torch.no_grad():
            for obs_b, act_b in val_loader:
                val_loss += loss_fn(net(obs_b.to(device)), act_b.to(device)).item() * len(obs_b)
        val_loss /= n_val
        sch.step()

        if epoch % 20 == 0 or epoch == args.epochs:
            print(f"  Epoch {epoch:4d}/{args.epochs}  "
                  f"train_loss={train_loss:.5f}  val_loss={val_loss:.5f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "state_dict": net.state_dict(),
                "obs_mean":   obs_mean,
                "obs_std":    obs_std,
            }, save_path)

    print(f"\n[BC] Best val loss : {best_val_loss:.5f}")
    print(f"[BC] Weights saved → {save_path}")
    print(f"\nNext step:")
    print(f"  python RoboCasa_Code_Aide/my_rl_scripts/train_sac_privileged.py \\")
    print(f"      --config RoboCasa_Code_Aide/config/exp9_sac_privileged.yaml \\")
    print(f"      --bc_weights {save_path}")


if __name__ == "__main__":
    main()
