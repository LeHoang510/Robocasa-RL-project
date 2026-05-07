"""
SAC / SAC+HER Training Script - PnPCounterToCab (RoboCasa)
===========================================================
Supports two experiment methods via YAML config:
  exp3 (sac)     -  SAC + dense reward shaping (off-policy baseline)
  exp4 (sac_her) - SAC + HER + sparse reward  (goal-conditioned)

Key differences vs PPO (train_ppo.py):
  - Off-policy: experiences stored in a replay buffer and reused across updates.
  - SAC auto-tunes entropy (ent_coef="auto") — no manual tuning needed.
  - SAC+HER relabels failed episodes with achieved goals, turning sparse rewards
    into dense synthetic successes (requires GoalEnv observation format).

Usage
-----
python RoboCasa_Code_Aide/my_rl_scripts/train_sac.py \
    --config RoboCasa_Code_Aide/config/exp3_sac_baseline.yaml

python RoboCasa_Code_Aide/my_rl_scripts/train_sac.py \
    --config RoboCasa_Code_Aide/config/exp4_sac_her.yaml
"""

import argparse
import os
import sys
from collections import deque
from datetime import datetime

import numpy as np
import yaml
import gymnasium as gym

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from my_env import MyPnPCounterToCab, PnPGoalEnv
from robosuite.controllers import load_composite_controller_config
from robosuite.wrappers.gym_wrapper import GymWrapper
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import (
    BaseCallback, CallbackList, CheckpointCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer

try:
    import wandb
    from wandb.integration.sb3 import WandbCallback
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class SuccessInfoWrapper(gym.Wrapper):
    """Attach task success to info so callbacks can log real success rates."""

    def __init__(self, env, raw_env):
        super().__init__(env)
        self.raw_env = raw_env

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["success"] = float(self.raw_env._check_success())
        return obs, reward, terminated, truncated, info


class SuccessRateCallback(BaseCallback):
    """Tracks rolling episode success rate and logs it to TensorBoard / WandB."""

    def __init__(self, window: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self._successes = deque(maxlen=window)

    def _on_step(self) -> bool:
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done:
                # HER envs report is_success; fallback to 'success' for standard envs
                success = info.get("is_success", info.get("success", False))
                self._successes.append(float(success))
        if self._successes:
            self.logger.record("rollout/success_rate", float(np.mean(self._successes)))
        return True


def _build_raw_env(cfg: dict, rank: int) -> MyPnPCounterToCab:
    controller_config = load_composite_controller_config(
        controller=None, robot="PandaOmron",
    )
    env = MyPnPCounterToCab(
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
        seed=cfg["env"]["seed"] + rank,
    )
    env.reset()  # initialise robots before GymWrapper accesses robot_model
    return env


def make_env_sac(cfg: dict, rank: int):
    """Standard flat-obs env for plain SAC (exp3)."""
    def _init():
        raw_env = _build_raw_env(cfg, rank)
        env     = GymWrapper(raw_env, keys=None)
        env     = SuccessInfoWrapper(env, raw_env)
        log_dir = os.path.join(cfg["logging"]["log_dir"], str(rank))
        os.makedirs(log_dir, exist_ok=True)
        env = Monitor(env, log_dir)
        env.reset(seed=cfg["env"]["seed"] + rank)
        return env
    return _init


def make_env_her(cfg: dict, rank: int):
    """GoalEnv (Dict obs) for SAC+HER (exp4)."""
    def _init():
        raw_env  = _build_raw_env(cfg, rank)
        gym_env  = GymWrapper(raw_env, keys=None)
        log_dir  = os.path.join(cfg["logging"]["log_dir"], str(rank))
        os.makedirs(log_dir, exist_ok=True)
        gym_env  = Monitor(gym_env, log_dir)
        gym_env.reset(seed=cfg["env"]["seed"] + rank)
        goal_env = PnPGoalEnv(
            raw_env,
            gym_env,
            goal_threshold=cfg["her"]["goal_threshold"],
        )
        return goal_env
    return _init


def main():
    parser = argparse.ArgumentParser(description="Train SAC / SAC+HER on PnP task")
    parser.add_argument("--config",          type=str, required=True)
    parser.add_argument("--total_timesteps", type=int, default=None)
    parser.add_argument("--seed",            type=int, default=None)
    parser.add_argument("--run_name",        type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.total_timesteps is not None:
        cfg["training"]["total_timesteps"] = args.total_timesteps
    if args.seed is not None:
        cfg["env"]["seed"] = args.seed

    method  = cfg["experiment"]["method"]   # "sac" or "sac_her"
    t_cfg   = cfg["training"]
    log_cfg = cfg["logging"]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name  = args.run_name or f"{cfg['experiment']['name']}_{timestamp}"
    save_path = os.path.join(log_cfg["save_dir"], run_name)
    os.makedirs(save_path, exist_ok=True)

    # Build vectorised environment
    # SAC works with a single env (off-policy, no need for many parallel envs)
    if method == "sac_her":
        env = DummyVecEnv([make_env_her(cfg, 0)])
        policy       = "MultiInputPolicy"   # required for Dict obs spaces
        replay_buffer_class  = HerReplayBuffer
        replay_buffer_kwargs = dict(
            n_sampled_goal       = cfg["her"]["n_sampled_goal"],
            goal_selection_strategy = cfg["her"]["goal_selection_strategy"],
        )
    else:
        env = DummyVecEnv([make_env_sac(cfg, 0)])
        policy       = "MlpPolicy"
        replay_buffer_class  = ReplayBuffer
        replay_buffer_kwargs = {}

    # SAC model 
    model = SAC(
        policy=policy,
        env=env,
        learning_rate=t_cfg["lr"],
        buffer_size=t_cfg["buffer_size"],
        batch_size=t_cfg["batch_size"],
        tau=t_cfg["tau"],
        gamma=t_cfg["gamma"],
        ent_coef=t_cfg.get("ent_coef", "auto"),   # "auto" lets SAC self-tune
        learning_starts=t_cfg["learning_starts"],
        policy_kwargs=dict(net_arch=cfg["policy"]["net_arch"]),
        replay_buffer_class=replay_buffer_class,
        replay_buffer_kwargs=replay_buffer_kwargs,
        verbose=1,
        tensorboard_log=os.path.join(save_path, "tb_logs"),
        seed=cfg["env"]["seed"],
    )

    # Callbacks
    checkpoint_cb = CheckpointCallback(
        save_freq=max(t_cfg.get("checkpoint_freq", 50_000), 1),
        save_path=os.path.join(save_path, "ckpts"),
        name_prefix="sac_pnp",
        verbose=1,
    )
    callbacks = [checkpoint_cb, SuccessRateCallback(window=100)]

    if log_cfg.get("use_wandb"):
        if not WANDB_AVAILABLE:
            print("[WARNING] wandb not installed — skipping WandB logging.")
        else:
            wandb.init(
                project=log_cfg["wandb_project"],
                name=run_name,
                config=cfg,
                sync_tensorboard=True,
            )
            callbacks.append(
                WandbCallback(
                    gradient_save_freq=1000,
                    model_save_path=os.path.join(save_path, "wandb_model"),
                    verbose=1,
                )
            )

    # Train 
    print(f"\n[INFO] Experiment  : {cfg['experiment']['name']}")
    print(f"[INFO] Method      : {method}")
    print(f"[INFO] Steps       : {t_cfg['total_timesteps']:,}")
    print(f"[INFO] Save path   : {save_path}")
    if method == "sac_her":
        print(f"[INFO] HER goals   : {cfg['her']['n_sampled_goal']} "
              f"({cfg['her']['goal_selection_strategy']} strategy)")
    print()

    model.learn(
        total_timesteps=t_cfg["total_timesteps"],
        callback=CallbackList(callbacks),
        progress_bar=True,
    )

    final_path = os.path.join(save_path, "final_model")
    model.save(final_path)
    print(f"\n[INFO] Saved → {final_path}.zip")

    env.close()
    if log_cfg.get("use_wandb") and WANDB_AVAILABLE:
        wandb.finish()


if __name__ == "__main__":
    main()
