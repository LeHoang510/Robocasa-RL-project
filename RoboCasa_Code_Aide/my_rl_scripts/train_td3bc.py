"""
TD3+BC Offline RL Training Script — PnPCounterToCab (RoboCasa)
==============================================================
Trains TD3+BC on a fixed offline dataset (no env interaction during training).

Unlike PPO/SAC which run environment steps during training, TD3+BC samples
gradient steps from the fixed demonstration buffer and evaluates periodically.

Observation space (16D, robot proprioception only):
  [base_pos(3), base_quat(4), eef_pos_rel(3), eef_quat_rel(4), gripper(2)]

Action space (12D, HDF5/env ordering):
  [eef_pos(3), eef_rot(3), gripper(1), base_motion(4), control_mode(1)]

Usage
-----
python RoboCasa_Code_Aide/my_rl_scripts/train_td3bc.py \
    --config RoboCasa_Code_Aide/config/exp7_td3bc.yaml
"""

import argparse
import os
import sys
from datetime import datetime

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from my_env import MyPnPCounterToCab, TD3BCAgent
from my_env.bc_dataset import OfflineRLDataset
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

def evaluate(agent: TD3BCAgent, raw_env, n_episodes: int, horizon: int):
    successes, rewards = [], []
    for _ in range(n_episodes):
        obs_dict  = raw_env.reset()
        done      = False
        ep_reward = 0.0
        step      = 0
        while not done and step < horizon:
            action = agent.predict_from_obs_dict(obs_dict)
            obs_dict, reward, done, _ = raw_env.step(action)
            ep_reward += reward
            step += 1
        successes.append(float(raw_env._check_success()))
        rewards.append(ep_reward)
    return float(np.mean(successes)), float(np.mean(rewards))


# ---------------------------------------------------------------------------
# Infinite data sampler — cycles through dataset indefinitely
# ---------------------------------------------------------------------------

def _infinite_loader(dataset, batch_size: int, device: torch.device):
    loader = DataLoader(
        dataset, batch_size=batch_size,
        shuffle=True, num_workers=0, drop_last=True,
    )
    while True:
        for batch in loader:
            yield tuple(t.to(device) for t in batch)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train TD3+BC on RoboCasa offline demos")
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
    print(f"[INFO] Loading offline dataset from {d_cfg['dataset_dir']}")
    dataset = OfflineRLDataset(
        dataset_dir  = d_cfg["dataset_dir"],
        split        = d_cfg.get("split", "train"),
        max_episodes = d_cfg.get("max_episodes", None),
        verbose      = True,
    )
    if dataset.obs_dim == 16:
        print(
            "[WARNING] TD3+BC is training from 16D robot proprioception only. "
            "Without apple/cabinet observations, offline control on randomized "
            "pick-and-place is fundamentally under-informed."
        )
    obs_mean, obs_std = dataset.compute_normalisation()
    data_iter = _infinite_loader(dataset, t_cfg["batch_size"], device)

    # ------------------------------------------------------------------
    # Environment (evaluation only)
    # ------------------------------------------------------------------
    print("[INFO] Building eval environment …")
    raw_env = _build_raw_env(cfg)
    raw_env.reset()
    low, high = raw_env.action_spec

    # ------------------------------------------------------------------
    # Agent
    # ------------------------------------------------------------------
    agent = TD3BCAgent(
        obs_dim      = dataset.obs_dim,
        action_dim   = dataset.action_dim,
        actor_arch   = p_cfg["actor_arch"],
        critic_arch  = p_cfg["critic_arch"],
        obs_mean     = obs_mean,
        obs_std      = obs_std,
        action_low   = low.astype(np.float32),
        action_high  = high.astype(np.float32),
        gamma        = p_cfg.get("gamma",        0.99),
        tau          = p_cfg.get("tau",          0.005),
        policy_noise = p_cfg.get("policy_noise", 0.2),
        noise_clip   = p_cfg.get("noise_clip",   0.5),
        policy_delay = p_cfg.get("policy_delay", 2),
        alpha        = p_cfg.get("alpha",        2.5),
        device       = device_str,
    )
    agent._init_optimizers(
        actor_lr  = t_cfg["actor_lr"],
        critic_lr = t_cfg["critic_lr"],
    )

    # ------------------------------------------------------------------
    # WandB
    # ------------------------------------------------------------------
    use_wandb = log_cfg.get("use_wandb") and WANDB_AVAILABLE
    if use_wandb:
        wandb.init(project=log_cfg["wandb_project"], name=run_name, config=cfg)

    # ------------------------------------------------------------------
    # Training loop (gradient steps, not epochs)
    # ------------------------------------------------------------------
    n_steps    = t_cfg["n_gradient_steps"]
    eval_every = t_cfg["eval_every"]
    log_every  = t_cfg.get("log_every", 1000)

    best_success  = -1.0
    critic_losses = []
    actor_losses  = []
    q_means       = []

    print(f"\n[INFO] TD3+BC training for {n_steps:,} gradient steps on {device}")
    print(f"[INFO] alpha={p_cfg.get('alpha', 2.5)}  gamma={p_cfg.get('gamma', 0.99)}  "
          f"tau={p_cfg.get('tau', 0.005)}")
    print(f"[INFO] Save path: {save_path}\n")

    agent.actor.train()
    agent.critic1.train()
    agent.critic2.train()

    for step in range(1, n_steps + 1):
        obs_b, act_b, rew_b, next_obs_b, done_b = next(data_iter)

        metrics = agent.update(obs_b, act_b, rew_b, next_obs_b, done_b)
        critic_losses.append(metrics["critic_loss"])
        if "actor_loss" in metrics:
            actor_losses.append(metrics["actor_loss"])
            q_means.append(metrics["q_mean"])

        # Logging
        if step % log_every == 0:
            cl = float(np.mean(critic_losses))
            al = float(np.mean(actor_losses)) if actor_losses else float("nan")
            qm = float(np.mean(q_means))       if q_means       else float("nan")
            print(f"  Step {step:7d}/{n_steps}  critic={cl:.4f}  actor={al:.4f}  Q={qm:.4f}")
            if use_wandb:
                wandb.log({
                    "step": step,
                    "td3bc/critic_loss": cl,
                    "td3bc/actor_loss":  al,
                    "td3bc/q_mean":      qm,
                })
            critic_losses.clear()
            actor_losses.clear()
            q_means.clear()

        # Evaluation
        if step % eval_every == 0 or step == n_steps:
            agent.actor.eval()
            success_rate, mean_reward = evaluate(
                agent, raw_env,
                n_episodes = t_cfg["eval_episodes"],
                horizon    = cfg["env"]["horizon"],
            )
            agent.actor.train()

            print(f"  Step {step:7d}/{n_steps}  "
                  f"success={success_rate:.2f}  reward={mean_reward:.1f}")
            if use_wandb:
                wandb.log({
                    "step":              step,
                    "eval/success_rate": success_rate,
                    "eval/mean_reward":  mean_reward,
                })
            if success_rate > best_success:
                best_success = success_rate
                agent.save(os.path.join(save_path, "td3bc_best.pt"))
                print(f"    → new best ({best_success:.2f})")

    # ------------------------------------------------------------------
    # Final save
    # ------------------------------------------------------------------
    final_path = os.path.join(save_path, "td3bc_final.pt")
    agent.save(final_path)
    print(f"\n[INFO] Final model → {final_path}")
    print(f"[INFO] Best model (success={best_success:.2f}) → {save_path}/td3bc_best.pt")
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
