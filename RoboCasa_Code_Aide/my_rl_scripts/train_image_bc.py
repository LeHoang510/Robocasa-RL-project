"""
Image-conditioned behavioral cloning for RoboCasa pick-and-place.
"""

import argparse
import os
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from my_env import MyPnPCounterToCab, ImageBCAgent
from my_env.bc_dataset import ImageBCDemoDataset
from robosuite.controllers import load_composite_controller_config

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def _build_raw_env(cfg: dict):
    controller_config = load_composite_controller_config(
        controller=None, robot="PandaOmron",
    )
    cam_names = cfg["dataset"]["camera_names"]
    return MyPnPCounterToCab(
        robots="PandaOmron",
        controller_configs=controller_config,
        use_camera_obs=True,
        has_renderer=False,
        has_offscreen_renderer=True,
        reward_shaping=True,
        control_freq=20,
        renderer="mjviewer",
        ignore_done=False,
        horizon=cfg["env"]["horizon"],
        seed=cfg["env"]["seed"],
        camera_names=cam_names,
        camera_heights=cfg["dataset"]["image_size"],
        camera_widths=cfg["dataset"]["image_size"],
    )


def evaluate(agent: ImageBCAgent, raw_env, n_episodes: int, horizon: int):
    successes, rewards = [], []
    for _ in range(n_episodes):
        obs_dict = raw_env.reset()
        done = False
        ep_reward = 0.0
        step = 0
        while not done and step < horizon:
            action = agent.predict_from_obs_dict(obs_dict)
            obs_dict, reward, done, _ = raw_env.step(action)
            ep_reward += reward
            step += 1
        successes.append(float(raw_env._check_success()))
        rewards.append(ep_reward)
    return float(np.mean(successes)), float(np.mean(rewards))


def main():
    parser = argparse.ArgumentParser(description="Train image-conditioned BC on RoboCasa demos")
    parser.add_argument("--config",      type=str, required=True)
    parser.add_argument("--dataset_dir", type=str, default=None)
    parser.add_argument("--run_name",    type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.dataset_dir is not None:
        cfg["dataset"]["dataset_dir"] = args.dataset_dir

    t_cfg = cfg["training"]
    d_cfg = cfg["dataset"]
    log_cfg = cfg["logging"]

    device_str = t_cfg.get("device", "auto")
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"{cfg['experiment']['name']}_{timestamp}"
    save_path = os.path.join(log_cfg["save_dir"], run_name)
    os.makedirs(save_path, exist_ok=True)

    print(f"[INFO] Loading image demos from {d_cfg['dataset_dir']}")
    dataset = ImageBCDemoDataset(
        dataset_dir=d_cfg["dataset_dir"],
        split=d_cfg.get("split", "all"),
        max_episodes=d_cfg.get("max_episodes", None),
        camera_keys=tuple(d_cfg["camera_names"]),
        image_size=d_cfg["image_size"],
        verbose=True,
    )

    val_frac = d_cfg.get("val_frac", 0.1)
    n_val = max(1, int(len(dataset) * val_frac))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(cfg["env"]["seed"]),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=t_cfg["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(val_ds, batch_size=t_cfg["batch_size"], shuffle=False, num_workers=0)
    obs_mean, obs_std = dataset.compute_normalisation()

    print("[INFO] Building eval environment …")
    raw_env = _build_raw_env(cfg)
    raw_env.reset()
    low, high = raw_env.action_spec

    agent = ImageBCAgent(
        obs_dim=dataset.obs_dim,
        image_channels=dataset.image_channels,
        action_dim=dataset.action_dim,
        mlp_arch=cfg["policy"]["mlp_arch"],
        camera_names=d_cfg["camera_names"],
        image_size=d_cfg["image_size"],
        obs_mean=obs_mean,
        obs_std=obs_std,
        action_low=low.astype(np.float32),
        action_high=high.astype(np.float32),
        device=device_str,
    )

    optimizer = torch.optim.AdamW(
        agent.net.parameters(),
        lr=t_cfg["lr"],
        weight_decay=t_cfg.get("weight_decay", 1e-4),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=t_cfg["epochs"], eta_min=t_cfg["lr"] * 0.1,
    )
    loss_fn = nn.MSELoss()

    obs_mean_t = torch.tensor(obs_mean, dtype=torch.float32, device=device)
    obs_std_t = torch.tensor(obs_std, dtype=torch.float32, device=device)

    use_wandb = log_cfg.get("use_wandb") and WANDB_AVAILABLE
    if use_wandb:
        wandb.init(project=log_cfg["wandb_project"], name=run_name, config=cfg)

    best_success = -1.0
    print(f"\n[INFO] Training image BC for {t_cfg['epochs']} epochs on {device}")
    print(f"[INFO] Cameras: {d_cfg['camera_names']}  image_size={d_cfg['image_size']}")
    print(f"[INFO] Save path: {save_path}\n")

    for epoch in range(1, t_cfg["epochs"] + 1):
        print(f"[INFO] Starting epoch {epoch}/{t_cfg['epochs']} ...")
        agent.net.train()
        train_losses = []
        for obs_b, img_b, act_b in train_loader:
            obs_b = obs_b.to(device)
            img_b = img_b.to(device)
            act_b = act_b.to(device)
            obs_norm = (obs_b - obs_mean_t) / (obs_std_t + 1e-8)
            pred = agent.net(obs_norm, img_b)
            loss = loss_fn(pred, act_b)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.net.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())
        scheduler.step()

        agent.net.eval()
        val_losses = []
        with torch.no_grad():
            for obs_b, img_b, act_b in val_loader:
                obs_b = obs_b.to(device)
                img_b = img_b.to(device)
                act_b = act_b.to(device)
                obs_norm = (obs_b - obs_mean_t) / (obs_std_t + 1e-8)
                val_losses.append(loss_fn(agent.net(obs_norm, img_b), act_b).item())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        metrics = {"image_bc/train_loss": train_loss, "image_bc/val_loss": val_loss}

        if epoch % t_cfg["eval_every"] == 0 or epoch == t_cfg["epochs"]:
            success_rate, mean_reward = evaluate(
                agent, raw_env,
                n_episodes=t_cfg["eval_episodes"],
                horizon=cfg["env"]["horizon"],
            )
            metrics["eval/success_rate"] = success_rate
            metrics["eval/mean_reward"] = mean_reward
            print(
                f"  Epoch {epoch:4d}/{t_cfg['epochs']}  "
                f"train={train_loss:.4f}  val={val_loss:.4f}  "
                f"success={success_rate:.2f}  reward={mean_reward:.1f}"
            )
            if success_rate > best_success:
                best_success = success_rate
                agent.save(os.path.join(save_path, "image_bc_best.pt"))
                print(f"    -> new best ({best_success:.2f})")
        elif epoch % 10 == 0:
            print(
                f"  Epoch {epoch:4d}/{t_cfg['epochs']}  "
                f"train={train_loss:.4f}  val={val_loss:.4f}"
            )

        if use_wandb:
            wandb.log({"epoch": epoch, **metrics})

    final_path = os.path.join(save_path, "image_bc_final.pt")
    agent.save(final_path)
    print(f"\n[INFO] Final model -> {final_path}")
    print(f"[INFO] Best model (success={best_success:.2f}) -> {save_path}/image_bc_best.pt")
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
