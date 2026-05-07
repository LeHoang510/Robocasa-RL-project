"""
Diffusion Policy Training Script — PnPCounterToCab (RoboCasa)
=============================================================
Trains a diffusion policy (DDPM) on robocasa LeRobot-format demonstration data.

Observation space (16D, robot proprioception only — matches BCAgent):
  [base_pos(3), base_quat(4), eef_pos_rel(3), eef_quat_rel(4), gripper(2)]

Action space (12D, HDF5/env ordering):
  [eef_pos(3), eef_rot(3), gripper(1), base_motion(4), control_mode(1)]

Usage
-----
python RoboCasa_Code_Aide/my_rl_scripts/train_diffusion.py \
    --config RoboCasa_Code_Aide/config/exp6_diffusion.yaml

python RoboCasa_Code_Aide/my_rl_scripts/train_diffusion.py \
    --config RoboCasa_Code_Aide/config/exp6_diffusion.yaml \
    --dataset_dir /path/to/lerobot_dataset
"""

import argparse
import os
import sys
from datetime import datetime

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from my_env import MyPnPCounterToCab, DiffusionAgent
from my_env.bc_dataset import BCDemoDataset
from robosuite.controllers import load_composite_controller_config

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

def _build_raw_env(cfg: dict):
    controller_config = load_composite_controller_config(
        controller=None, robot="PandaOmron",
    )
    return MyPnPCounterToCab(
        robots="PandaOmron",
        controller_configs=controller_config,
        use_camera_obs=False,
        has_renderer=False,
        has_offscreen_renderer=False,
        reward_shaping=True,
        control_freq=20,
        renderer="mjviewer",
        ignore_done=False,
        horizon=cfg["env"]["horizon"],
        seed=cfg["env"]["seed"],
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(agent: DiffusionAgent, raw_env, n_episodes: int, horizon: int):
    successes, rewards = [], []
    for _ in range(n_episodes):
        obs_dict   = raw_env.reset()
        done       = False
        ep_reward  = 0.0
        step       = 0
        while not done and step < horizon:
            action = agent.predict_from_obs_dict(obs_dict)
            obs_dict, reward, done, _ = raw_env.step(action)
            ep_reward += reward
            step += 1
        successes.append(float(raw_env._check_success()))
        rewards.append(ep_reward)
    return float(np.mean(successes)), float(np.mean(rewards))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train Diffusion Policy on RoboCasa demos")
    parser.add_argument("--config",      type=str, required=True)
    parser.add_argument("--dataset_dir", type=str, default=None)
    parser.add_argument("--run_name",    type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.dataset_dir is not None:
        cfg["dataset"]["dataset_dir"] = args.dataset_dir

    t_cfg   = cfg["training"]
    d_cfg   = cfg["dataset"]
    p_cfg   = cfg["policy"]
    log_cfg = cfg["logging"]

    device_str = t_cfg.get("device", "auto")
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name  = args.run_name or f"{cfg['experiment']['name']}_{timestamp}"
    save_path = os.path.join(log_cfg["save_dir"], run_name)
    os.makedirs(save_path, exist_ok=True)

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    print(f"[INFO] Loading demos from {d_cfg['dataset_dir']}")
    dataset = BCDemoDataset(
        dataset_dir  = d_cfg["dataset_dir"],
        split        = d_cfg.get("split", "train"),
        max_episodes = d_cfg.get("max_episodes", None),
        verbose      = True,
    )
    if dataset.obs_dim == 16:
        print(
            "[WARNING] This diffusion policy only sees 16D robot proprioception. "
            "It does not observe apple/cabinet state, so performance on "
            "randomized pick-and-place may stay poor even with more demos."
        )

    val_frac = d_cfg.get("val_frac", 0.1)
    n_val    = max(1, int(len(dataset) * val_frac))
    n_train  = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(cfg["env"]["seed"]),
    )
    train_loader = DataLoader(
        train_ds, batch_size=t_cfg["batch_size"],
        shuffle=True, num_workers=0, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=0)
    print(f"[INFO] Train: {n_train}  Val: {n_val}  transitions")

    obs_mean, obs_std = dataset.compute_normalisation()

    # ------------------------------------------------------------------
    # Environment (only for evaluation)
    # ------------------------------------------------------------------
    print("[INFO] Building eval environment …")
    raw_env = _build_raw_env(cfg)
    raw_env.reset()
    low, high = raw_env.action_spec

    # ------------------------------------------------------------------
    # Policy
    # ------------------------------------------------------------------
    agent = DiffusionAgent(
        obs_dim           = dataset.obs_dim,
        action_dim        = dataset.action_dim,
        net_arch          = p_cfg["net_arch"],
        time_emb_dim      = p_cfg.get("time_emb_dim", 32),
        obs_mean          = obs_mean,
        obs_std           = obs_std,
        action_low        = low.astype(np.float32),
        action_high       = high.astype(np.float32),
        n_diffusion_steps = t_cfg.get("n_diffusion_steps", 100),
        n_inference_steps = t_cfg.get("n_inference_steps", 10),
        beta_schedule     = p_cfg.get("beta_schedule", "cosine"),
        device            = device_str,
    )

    optimizer = torch.optim.AdamW(
        agent.net.parameters(),
        lr           = t_cfg["lr"],
        weight_decay = t_cfg.get("weight_decay", 1e-4),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=t_cfg["epochs"], eta_min=t_cfg["lr"] * 0.1,
    )

    # ------------------------------------------------------------------
    # WandB
    # ------------------------------------------------------------------
    use_wandb = log_cfg.get("use_wandb") and WANDB_AVAILABLE
    if use_wandb:
        wandb.init(project=log_cfg["wandb_project"], name=run_name, config=cfg)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_success = -1.0
    print(f"\n[INFO] Training Diffusion Policy for {t_cfg['epochs']} epochs on {device}")
    print(f"[INFO] T={t_cfg.get('n_diffusion_steps', 100)} diffusion steps, "
          f"{t_cfg.get('n_inference_steps', 10)} DDIM inference steps")
    print(f"[INFO] Save path: {save_path}\n")

    for epoch in range(1, t_cfg["epochs"] + 1):

        # Train
        agent.net.train()
        train_losses = []
        for obs_b, act_b in train_loader:
            obs_b = obs_b.to(device)
            act_b = act_b.to(device)
            loss  = agent.training_loss(obs_b, act_b)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.net.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())
        scheduler.step()

        # Validation
        agent.net.eval()
        val_losses = []
        with torch.no_grad():
            for obs_b, act_b in val_loader:
                val_losses.append(
                    agent.training_loss(obs_b.to(device), act_b.to(device)).item()
                )

        train_loss = float(np.mean(train_losses))
        val_loss   = float(np.mean(val_losses))
        metrics    = {"diffusion/train_loss": train_loss, "diffusion/val_loss": val_loss}

        # Env evaluation
        if epoch % t_cfg["eval_every"] == 0 or epoch == t_cfg["epochs"]:
            success_rate, mean_reward = evaluate(
                agent, raw_env,
                n_episodes = t_cfg["eval_episodes"],
                horizon    = cfg["env"]["horizon"],
            )
            metrics["eval/success_rate"] = success_rate
            metrics["eval/mean_reward"]  = mean_reward
            print(
                f"  Epoch {epoch:4d}/{t_cfg['epochs']}  "
                f"train={train_loss:.4f}  val={val_loss:.4f}  "
                f"success={success_rate:.2f}  reward={mean_reward:.1f}"
            )
            if success_rate > best_success:
                best_success = success_rate
                agent.save(os.path.join(save_path, "diffusion_best.pt"))
                print(f"    → new best ({best_success:.2f})")
        elif epoch % 10 == 0:
            print(
                f"  Epoch {epoch:4d}/{t_cfg['epochs']}  "
                f"train={train_loss:.4f}  val={val_loss:.4f}"
            )

        if use_wandb:
            wandb.log({"epoch": epoch, **metrics})

    # ------------------------------------------------------------------
    # Final save
    # ------------------------------------------------------------------
    final_path = os.path.join(save_path, "diffusion_final.pt")
    agent.save(final_path)
    print(f"\n[INFO] Final model → {final_path}")
    print(f"[INFO] Best model (success={best_success:.2f}) → {save_path}/diffusion_best.pt")
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
