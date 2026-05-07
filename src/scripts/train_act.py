"""
Train ACT (Action Chunking with Transformers) with ResNet18 visual encoder.

Usage
-----
python src/scripts/train_act.py --config src/config/act.yaml
"""

import argparse
import os
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.act_dataset import collect_act_demos, ACTDataset
from env.act_policy   import ACTPolicy, ACTAgent

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Evaluation in environment
# ---------------------------------------------------------------------------

def evaluate(agent: ACTAgent, camera_names: list[str], img_h: int, img_w: int,
             n_episodes: int, seed: int) -> float:
    from env.pnp_env import PnPEnv, extract_obs
    from robosuite.controllers import load_composite_controller_config

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
        horizon=300,
        seed=seed,
    )
    raw_env.set_difficulty(0)

    successes = []
    for ep in range(n_episodes):
        obs_dict  = raw_env.reset()
        target_pos = _read_target_pos(raw_env)
        agent.reset()

        done = False; step = 0
        while not done and step < 300:
            imgs   = np.stack([obs_dict[f"{cam}_image"] for cam in camera_names], axis=0)
            proprio = extract_obs(obs_dict, target_pos)
            action  = agent.predict(imgs, proprio)
            obs_dict, _, done, _ = raw_env.step(action)
            step += 1

        success = bool(raw_env._check_success())
        successes.append(success)

    raw_env.close()
    return float(np.mean(successes))


def _read_target_pos(raw_env):
    try:
        cab    = raw_env.cab
        offset = cab.get_reset_regions(raw_env)["level0"]["offset"]
        return (np.array(cab.pos) + np.array(offset)).astype(np.float32)
    except Exception:
        return np.array([2.25, -0.2, 1.42], dtype=np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    d_cfg  = cfg["data"]
    m_cfg  = cfg["model"]
    t_cfg  = cfg["training"]
    log_cfg = cfg["logging"]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name  = f"{cfg['experiment']['name']}_{timestamp}"
    save_path = os.path.join(log_cfg["save_dir"], run_name)
    os.makedirs(save_path, exist_ok=True)

    device_str = t_cfg.get("device", "auto")
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    print(f"\n[ACT] device={device_str}  run={run_name}")

    camera_names = d_cfg["camera_names"]
    img_h = d_cfg.get("img_h", 128)
    img_w = d_cfg.get("img_w", 128)
    chunk_size = m_cfg["chunk_size"]

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    data = collect_act_demos(
        dataset_dirs        = d_cfg["dataset_dirs"],
        n_episodes_per_dir  = d_cfg.get("n_episodes_per_dir", 200),
        horizon             = d_cfg.get("horizon", 300),
        camera_names        = camera_names,
        img_h               = img_h,
        img_w               = img_w,
        cache_path          = d_cfg["cache_path"],
    )
    full_ds   = ACTDataset(data, chunk_size=chunk_size)
    train_ds, val_ds = full_ds.episode_split(
        val_frac = t_cfg.get("val_frac", 0.1),
        seed     = t_cfg.get("seed",     0),
    )
    print(f"[ACT] train={len(train_ds):,} steps  val={len(val_ds):,} steps")

    train_loader = DataLoader(
        train_ds, batch_size=t_cfg["batch_size"], shuffle=True,
        num_workers=t_cfg.get("num_workers", 4), pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=t_cfg["batch_size"], shuffle=False,
        num_workers=t_cfg.get("num_workers", 4), pin_memory=(device.type == "cuda"),
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    policy = ACTPolicy(
        obs_dim    = 25,
        act_dim    = 12,
        chunk_size = chunk_size,
        hidden_dim = m_cfg.get("hidden_dim", 512),
        n_heads    = m_cfg.get("n_heads",    8),
        enc_layers = m_cfg.get("enc_layers", 4),
        dec_layers = m_cfg.get("dec_layers", 7),
        latent_dim = m_cfg.get("latent_dim", 32),
        n_cameras  = len(camera_names),
        img_h      = img_h,
        img_w      = img_w,
        kl_weight  = m_cfg.get("kl_weight",  10.0),
        dropout    = m_cfg.get("dropout",    0.1),
        pretrained_backbone = m_cfg.get("pretrained_backbone", True),
    ).to(device)

    n_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"[ACT] parameters: {n_params/1e6:.1f}M")

    optim = torch.optim.AdamW(
        policy.parameters(),
        lr           = t_cfg["lr"],
        weight_decay = t_cfg.get("weight_decay", 1e-4),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=t_cfg["epochs"],
    )

    # ------------------------------------------------------------------
    # W&B
    # ------------------------------------------------------------------
    use_wandb = log_cfg.get("use_wandb") and WANDB_AVAILABLE
    if use_wandb:
        wandb.init(project=log_cfg["wandb_project"], name=run_name,
                   config=cfg, save_code=True)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_val   = float("inf")
    log_every  = t_cfg.get("log_every",  5)
    eval_every = t_cfg.get("eval_every", 20)

    for epoch in range(1, t_cfg["epochs"] + 1):
        # --- Train ---
        policy.train()
        train_losses = []
        train_recons, train_kls = [], []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d}", leave=False)
        for images, proprio, actions in pbar:
            images  = images.to(device)
            proprio = proprio.to(device)
            actions = actions.to(device)

            loss, recon, kl = policy(images, proprio, actions)
            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optim.step()

            train_losses.append(loss.item())
            train_recons.append(recon.item())
            train_kls.append(kl.item())
            pbar.set_postfix(recon=f"{recon.item():.4f}", kl=f"{kl.item():.4f}")

        scheduler.step()
        mean_train = float(np.mean(train_losses))

        if epoch % log_every != 0 and epoch != t_cfg["epochs"]:
            continue

        # --- Validation ---
        policy.eval()
        val_losses = []
        with torch.no_grad():
            for images, proprio, actions in val_loader:
                images  = images.to(device)
                proprio = proprio.to(device)
                actions = actions.to(device)
                loss, _, _ = policy(images, proprio, actions)
                val_losses.append(loss.item())
        mean_val = float(np.mean(val_losses)) if val_losses else float("inf")

        # --- Env eval ---
        success_rate = 0.0
        if epoch % eval_every == 0:
            agent = ACTAgent(
                policy, device=device_str,
                temporal_agg_gamma=m_cfg.get("temporal_agg_gamma", 0.01),
            )
            n_eval = t_cfg.get("n_eval_episodes", 5)
            success_rate = evaluate(agent, camera_names, img_h, img_w,
                                    n_eval, seed=42)
            policy.train()
            print(f"  [Eval] epoch={epoch}  success={success_rate:.0%}  ({n_eval} eps)")

        # --- Logging ---
        mean_recon = float(np.mean(train_recons))
        mean_kl    = float(np.mean(train_kls))
        print(f"[Epoch {epoch:03d}/{t_cfg['epochs']}]  "
              f"train={mean_train:.4f}  recon={mean_recon:.4f}  kl={mean_kl:.4f}  "
              f"val={mean_val:.4f}  success={success_rate:.2%}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}")

        if use_wandb:
            wandb.log({
                "train/loss":  mean_train, "train/recon": mean_recon,
                "train/kl":    mean_kl,    "val/loss":    mean_val,
                "eval/success": success_rate,
                "lr": scheduler.get_last_lr()[0],
            }, step=epoch)

        # --- Checkpoint ---
        if mean_val < best_val:
            best_val = mean_val
            policy.save(os.path.join(save_path, "act_best.pt"))

    policy.save(os.path.join(save_path, "act_final.pt"))
    print(f"\n[Done] Best val loss: {best_val:.4f}")
    print(f"       Model saved → {save_path}/act_final.pt")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
