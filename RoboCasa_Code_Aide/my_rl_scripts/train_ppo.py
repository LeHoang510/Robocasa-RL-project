"""
PPO Training Script - PnPCounterToCab (RoboCasa)
=================================================
Supports exp1 and exp2 via YAML config:
  exp1 (ppo)        - PPO + reward shaping only
  exp2 (curriculum) - PPO + reward shaping + CurriculumCallback

For SAC / SAC+HER experiments, use train_sac.py instead.

Usage
-----
python RoboCasa_Code_Aide/my_rl_scripts/train_ppo.py \
    --config RoboCasa_Code_Aide/config/exp1_ppo_baseline.yaml

python RoboCasa_Code_Aide/my_rl_scripts/train_ppo.py \
    --config RoboCasa_Code_Aide/config/exp2_curriculum.yaml

# Override any YAML value from CLI:
python RoboCasa_Code_Aide/my_rl_scripts/train_ppo.py \
    --config RoboCasa_Code_Aide/config/exp1_ppo_baseline.yaml \
    --total_timesteps 2000000
"""

import argparse
import os
import sys
from collections import deque
from datetime import datetime

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from my_env import MyPnPCounterToCab
from robosuite.controllers import load_composite_controller_config
from robosuite.wrappers.gym_wrapper import GymWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback, CallbackList, CheckpointCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

try:
    import wandb
    from wandb.integration.sb3 import WandbCallback
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class SuccessRateCallback(BaseCallback):
    """
    Tracks rolling episode success rate and logs it to TensorBoard / WandB
    as `rollout/success_rate`. Used by all three experiments so the learning
    curve shows both reward and success rate side-by-side.
    """

    def __init__(self, window: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self._successes = deque(maxlen=window)

    def _on_step(self) -> bool:
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done:
                self._successes.append(float(info.get("success", False)))
        if self._successes:
            self.logger.record("rollout/success_rate", float(np.mean(self._successes)))
        return True



class CurriculumCallback(BaseCallback):
    """
    Monitors rolling episode success rate and advances placement difficulty
    whenever the rate exceeds `success_threshold`.

    Calls VecEnv.set_attr("difficulty", new_level) so all parallel envs
    switch simultaneously at their next episode reset.
    """

    def __init__(self, cfg: dict, verbose: int = 1):
        super().__init__(verbose)
        self.success_threshold  = cfg["success_threshold"]
        self.eval_window        = cfg["eval_window"]
        self.num_levels         = cfg["num_levels"]
        self.current_difficulty = cfg["initial_difficulty"]
        self._successes = deque(maxlen=self.eval_window)

    def _on_step(self) -> bool:
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done:
                self._successes.append(float(info.get("success", False)))

        if (len(self._successes) >= self.eval_window
                and self.current_difficulty < self.num_levels - 1):
            rate = float(np.mean(self._successes))
            if rate >= self.success_threshold:
                self.current_difficulty += 1
                self.training_env.set_attr("difficulty", self.current_difficulty)
                self._successes.clear()
                if self.verbose:
                    print(f"\n[CURRICULUM] success={rate:.2f} → "
                          f"difficulty {self.current_difficulty}\n")
                self.logger.record("curriculum/difficulty",        self.current_difficulty)
                self.logger.record("curriculum/success_at_advance", rate)

        if self._successes:
            self.logger.record("curriculum/success_rate", float(np.mean(self._successes)))
            self.logger.record("curriculum/difficulty",   self.current_difficulty)
        return True


def make_env(cfg: dict, rank: int):
    """Return a callable that creates one environment instance."""
    def _init():
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
        env = GymWrapper(env, keys=None)
        log_dir = os.path.join(cfg["logging"]["log_dir"], str(rank))
        os.makedirs(log_dir, exist_ok=True)
        env = Monitor(env, log_dir)
        env.reset(seed=cfg["env"]["seed"] + rank)
        return env
    return _init


def load_config(path: str, overrides: dict) -> dict:
    """Load YAML and apply optional CLI overrides into the training section."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for key, val in overrides.items():
        if val is not None:
            cfg["training"][key] = val
    return cfg


def main():
    parser = argparse.ArgumentParser(description="Train PPO on RoboCasa PnP task")
    parser.add_argument("--config",          type=str, required=True,
                        help="Path to YAML experiment config")
    # CLI overrides (take precedence over YAML when provided)
    parser.add_argument("--n_envs",          type=int, default=None)
    parser.add_argument("--total_timesteps", type=int, default=None)
    parser.add_argument("--seed",            type=int, default=None)
    parser.add_argument("--run_name",        type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config, {
        "n_envs":          args.n_envs,
        "total_timesteps": args.total_timesteps,
    })
    if args.seed is not None:
        cfg["env"]["seed"] = args.seed

    method  = cfg["experiment"]["method"]
    t_cfg   = cfg["training"]
    log_cfg = cfg["logging"]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name  = args.run_name or f"{cfg['experiment']['name']}_{timestamp}"
    save_path = os.path.join(log_cfg["save_dir"], run_name)
    os.makedirs(save_path, exist_ok=True)

    # NOTE: SubprocVecEnv causes MuJoCo/robosuite to fail in forked processes
    # (robot_model stays None). DummyVecEnv runs envs sequentially in the
    # same process and is stable. Use it unconditionally for now.
    n_envs  = t_cfg["n_envs"]
    env_fns = [make_env(cfg, i) for i in range(n_envs)]
    env     = DummyVecEnv(env_fns)

    # PPO model 
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=t_cfg["lr"],
        n_steps=t_cfg["n_steps"],
        batch_size=t_cfg["batch_size"],
        n_epochs=t_cfg["n_epochs"],
        gamma=t_cfg["gamma"],
        gae_lambda=t_cfg["gae_lambda"],
        clip_range=t_cfg["clip_range"],
        ent_coef=t_cfg["ent_coef"],
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=cfg["policy"]["net_arch"]),
        verbose=1,
        tensorboard_log=os.path.join(save_path, "tb_logs"),
        seed=cfg["env"]["seed"],
    )

    # Callbacks 
    checkpoint_cb = CheckpointCallback(
        save_freq=max(t_cfg.get("checkpoint_freq", 50_000) // n_envs, 1),
        save_path=os.path.join(save_path, "ckpts"),
        name_prefix="ppo_pnp",
        verbose=1,
    )
    callbacks = [checkpoint_cb, SuccessRateCallback(window=100)]

    if method == "curriculum":
        callbacks.append(CurriculumCallback(cfg["curriculum"], verbose=1))
        print("[INFO] Curriculum learning enabled - starting at difficulty 0")

    if log_cfg.get("use_wandb"):
        if not WANDB_AVAILABLE:
            print("[WARNING] wandb not installed - skipping WandB logging.")
        else:
            wandb.init(
                project=log_cfg["wandb_project"],
                name=run_name,
                config=cfg,             # full YAML saved to WandB
                sync_tensorboard=True,  # training curves synced automatically
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
    print(f"[INFO] Envs        : {n_envs}")
    print(f"[INFO] Save path   : {save_path}\n")

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
