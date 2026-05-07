"""
Train Diffusion Policy on RoboCasa PnPCounterToCabinet demos.

Usage
-----
python src/scripts/train_diffusion.py --config src/config/diffusion.yaml

# Use fewer episodes for a quick sanity-check:
python src/scripts/train_diffusion.py --config src/config/diffusion.yaml --max_episodes 200
"""

import argparse
import os
import sys
from datetime import datetime

import time
import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# Make src/ importable when running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.dataset             import DemoDataset
from env.privileged_dataset  import PrivilegedDemoDataset
from env.diffusion_policy    import DiffusionAgent, DDPMScheduler

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Env factory (only used for eval)
# ---------------------------------------------------------------------------

def _make_env(cfg: dict, privileged: bool = False):
    from env.pnp_env import make_env
    return make_env(
        horizon    = cfg["env"]["horizon"],
        seed       = cfg["env"]["seed"],
        privileged = privileged,
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(agent: DiffusionAgent, env, n_episodes: int, horizon: int,
             privileged: bool = False) -> tuple[float, float]:
    successes, rewards = [], []

    for _ in range(n_episodes):
        if privileged:
            obs_nd, _ = env.reset()
        else:
            obs_nd = env.reset()
            if isinstance(obs_nd, tuple):
                obs_nd = obs_nd[0]

        agent.reset(first_obs=obs_nd)
        done      = False
        ep_reward = 0.0
        step      = 0

        while not done and step < horizon:
            action = agent.predict(obs_nd)
            if privileged:
                obs_nd, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
            else:
                obs_dict, reward, done, _ = env.step(action)
                obs_nd = _extract_state(obs_dict)
            ep_reward += reward
            step += 1

        raw_env = env.raw_env if privileged else env
        successes.append(float(raw_env._check_success()))
        rewards.append(ep_reward)

    return float(np.mean(successes)), float(np.mean(rewards))




# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(cfg: dict, max_episodes: int | None = None, run_name: str | None = None):
    d_cfg   = cfg["dataset"]
    t_cfg   = cfg["training"]
    m_cfg   = cfg["model"]
    log_cfg = cfg["logging"]

    device_str = t_cfg.get("device", "auto")
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name  = run_name or f"{cfg['experiment']['name']}_{timestamp}"
    save_dir  = os.path.join(log_cfg["save_dir"], run_name)
    os.makedirs(save_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    print("\n[Dataset] Loading demos …")
    privileged = "n_episodes_per_dir" in d_cfg  # privileged config has this key
    if privileged:
        dataset = PrivilegedDemoDataset(
            dataset_dirs       = d_cfg["dirs"],
            n_episodes_per_dir = max_episodes or d_cfg["n_episodes_per_dir"],
            obs_horizon        = d_cfg["obs_horizon"],
            action_horizon     = d_cfg["action_horizon"],
            horizon            = cfg["env"]["horizon"],
            cache_path         = d_cfg["cache_path"],
            verbose            = True,
        )
    else:
        dataset = DemoDataset(
            dataset_dirs   = d_cfg["dirs"],
            obs_horizon    = d_cfg["obs_horizon"],
            action_horizon = d_cfg["action_horizon"],
            max_episodes   = max_episodes or d_cfg.get("max_episodes"),
            verbose        = True,
        )

    # Fix #4: episode-level split (no data leakage across episodes)
    train_idx, val_idx = dataset.episode_split(
        val_frac=t_cfg["val_frac"], seed=cfg["env"]["seed"]
    )
    n_train, n_val = len(train_idx), len(val_idx)
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=t_cfg["batch_size"],
                              shuffle=True, num_workers=4, pin_memory=(device_str == "cuda"))
    val_loader   = DataLoader(Subset(dataset, val_idx),   batch_size=512,
                              shuffle=False, num_workers=2)
    print(f"  Train: {n_train:,}   Val: {n_val:,}  (episode-level split)")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    agent = DiffusionAgent(
        obs_dim        = dataset.obs_dim,
        act_dim        = dataset.act_dim,
        obs_horizon    = dataset.obs_horizon,
        action_horizon = dataset.action_horizon,
        obs_mean       = dataset.obs_mean,
        obs_std        = dataset.obs_std,
        act_mean       = dataset.act_mean,
        act_std        = dataset.act_std,
        device         = device_str,
        hidden_dim     = m_cfg["hidden_dim"],
        n_blocks       = m_cfg["n_blocks"],
        T_ddpm         = m_cfg["T_ddpm"],
        T_ddim         = m_cfg["T_ddim"],
    )

    n_params = sum(p.numel() for p in agent.model.parameters())
    print(f"\n[Model] Parameters: {n_params:,}")

    scheduler_ddpm = DDPMScheduler(T=m_cfg["T_ddpm"]).to(device)
    optimizer      = torch.optim.AdamW(
        agent.model.parameters(),
        lr=t_cfg["lr"],
        weight_decay=t_cfg["weight_decay"],
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=t_cfg["epochs"], eta_min=t_cfg["lr"] * 0.1,
    )
    loss_fn = nn.MSELoss()

    # ------------------------------------------------------------------
    # WandB
    # ------------------------------------------------------------------
    use_wandb = log_cfg.get("use_wandb") and WANDB_AVAILABLE
    if use_wandb:
        wandb.init(project=log_cfg["wandb_project"], name=run_name, config=cfg)

    # ------------------------------------------------------------------
    # Env for evaluation
    # ------------------------------------------------------------------
    print("\n[Env] Building eval environment …")
    env = _make_env(cfg, privileged=privileged)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    best_success = -1.0
    print(f"\n[Train] {t_cfg['epochs']} epochs | device={device_str} | save → {save_dir}\n")

    log_every  = t_cfg.get("log_every",  10)   # print train loss every N epochs
    eval_every = t_cfg.get("eval_every", 50)   # env eval every N epochs (expensive)

    for epoch in range(1, t_cfg["epochs"] + 1):
        t0 = time.time()

        # ---- train ----
        agent.model.train()
        train_losses = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:>3}/{t_cfg['epochs']}", leave=False)
        for obs_b, acts_b in pbar:
            obs_b  = obs_b.to(device)
            acts_b = acts_b.to(device)

            B = obs_b.shape[0]
            t  = torch.randint(0, scheduler_ddpm.T, (B,), device=device)
            noise = torch.randn_like(acts_b)
            noisy = scheduler_ddpm.add_noise(acts_b, noise, t)

            noise_pred = agent.model(noisy, obs_b, t)
            loss = loss_fn(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        lr_scheduler.step()

        train_loss = float(np.mean(train_losses))
        metrics    = {"diffusion/train_loss": train_loss}
        elapsed    = time.time() - t0

        # ---- val (only when logging) ----
        do_log  = (epoch % log_every == 0)
        do_eval = (epoch % eval_every == 0 or epoch == t_cfg["epochs"])

        if do_log or do_eval:
            agent.model.eval()
            val_losses = []
            with torch.no_grad():
                for obs_b, acts_b in val_loader:
                    obs_b  = obs_b.to(device)
                    acts_b = acts_b.to(device)
                    B = obs_b.shape[0]
                    t  = torch.randint(0, scheduler_ddpm.T, (B,), device=device)
                    noise = torch.randn_like(acts_b)
                    noisy = scheduler_ddpm.add_noise(acts_b, noise, t)
                    noise_pred = agent.model(noisy, obs_b, t)
                    val_losses.append(loss_fn(noise_pred, noise).item())
            val_loss = float(np.mean(val_losses))
            metrics["diffusion/val_loss"] = val_loss

        # ---- env eval (expensive — skip unless eval epoch) ----
        if do_eval:
            success_rate, mean_reward = evaluate(
                agent, env, t_cfg["eval_episodes"], cfg["env"]["horizon"],
                privileged=privileged,
            )
            metrics["eval/success_rate"] = success_rate
            metrics["eval/mean_reward"]  = mean_reward
            print(f"  Epoch {epoch:4d}/{t_cfg['epochs']}  "
                  f"train={train_loss:.4f}  val={val_loss:.4f}  "
                  f"success={success_rate:.2f}  reward={mean_reward:.1f}  "
                  f"({elapsed:.0f}s)")
            if success_rate > best_success:
                best_success = success_rate
                agent.save(os.path.join(save_dir, "diffusion_best.pt"))
                print(f"    → new best ({best_success:.2f})")
        elif do_log:
            print(f"  Epoch {epoch:4d}/{t_cfg['epochs']}  "
                  f"train={train_loss:.4f}  val={val_loss:.4f}  ({elapsed:.0f}s)")

        if use_wandb:
            wandb.log({"epoch": epoch, **metrics})

    # ------------------------------------------------------------------
    # Final save
    # ------------------------------------------------------------------
    final_path = os.path.join(save_dir, "diffusion_final.pt")
    agent.save(final_path)
    print(f"\n[Done] Final model → {final_path}")
    print(f"       Best  model → {save_dir}/diffusion_best.pt  (success={best_success:.2f})")
    if use_wandb:
        wandb.finish()
    env.close()

    return best_success


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",       type=str, required=True)
    parser.add_argument("--max_episodes", type=int, default=None,
                        help="Limit episodes for a quick sanity check")
    parser.add_argument("--run_name",     type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train(cfg, max_episodes=args.max_episodes, run_name=args.run_name)


if __name__ == "__main__":
    main()
