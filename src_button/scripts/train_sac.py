"""
SAC training for StartCoffeeMachine (CoffeePressButton).

Usage
-----
python src_button/scripts/train_sac.py --config src_button/config/sac_button.yaml
"""

import argparse
import os
import sys
from datetime import datetime

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.button_env import make_env

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

try:
    import wandb
    from wandb.integration.sb3 import WandbCallback
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

class ProgressCallback(BaseCallback):
    def __init__(self, log_freq: int = 5_000):
        super().__init__()
        self.log_freq      = log_freq
        self._accum        = {"rc/reach": 0.0, "rc/press": 0.0, "rc/retreat": 0.0, "rc/dist": 0.0}
        self._count        = 0
        self._ep_successes = []
        self._last_log     = 0

    def _on_step(self) -> bool:
        info = self.locals["infos"][0]
        for k in self._accum:
            self._accum[k] += float(info.get(k, 0.0))
        self._count += 1

        done = bool(self.locals["dones"][0])
        if done:
            self._ep_successes.append(float(info.get("is_success", 0.0)))

        if self.num_timesteps - self._last_log >= self.log_freq:
            n    = max(self._count, 1)
            avg  = {k: v / n for k, v in self._accum.items()}
            sr   = float(np.mean(self._ep_successes)) if self._ep_successes else 0.0

            print(f"\n[Step {self.num_timesteps:>7,}]  "
                  f"success={sr:.2f}  "
                  f"reach={avg['rc/reach']:.3f}  "
                  f"press={avg['rc/press']:.3f}  "
                  f"retreat={avg['rc/retreat']:.3f}  "
                  f"dist={avg['rc/dist']:.3f}")

            for k, v in avg.items():
                self.logger.record(f"progress/{k.split('/')[-1]}", v)
            self.logger.record("progress/success_rate", sr)

            self._accum        = {k: 0.0 for k in self._accum}
            self._count        = 0
            self._ep_successes = []
            self._last_log     = self.num_timesteps

        return True


class CheckpointCallback(BaseCallback):
    def __init__(self, save_freq: int, save_path: str):
        super().__init__()
        self.save_freq = save_freq
        self.save_path = save_path

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq == 0:
            tag = f"step_{self.num_timesteps}"
            self.model.save(os.path.join(self.save_path, tag))
            print(f"[Checkpoint] saved → {tag}")
        return True


class EvalCallback(BaseCallback):
    def __init__(self, eval_env_fn, eval_freq: int, n_eval_episodes: int, save_path: str):
        super().__init__()
        self.eval_env_fn = eval_env_fn
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.save_path = save_path
        self.best_success = -1.0
        self._last_eval = 0

    def _run_eval(self):
        env = self.eval_env_fn()
        successes, rewards, lengths = [], [], []
        try:
            for ep in range(self.n_eval_episodes):
                obs, _ = env.reset(seed=10_000 + ep)
                done = False
                ep_reward = 0.0
                ep_len = 0
                info = {}
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, term, trunc, info = env.step(action)
                    done = term or trunc
                    ep_reward += reward
                    ep_len += 1
                successes.append(float(info.get("is_success", 0.0)))
                rewards.append(ep_reward)
                lengths.append(ep_len)
        finally:
            env.close()

        return (
            float(np.mean(successes)) if successes else 0.0,
            float(np.mean(rewards)) if rewards else 0.0,
            float(np.mean(lengths)) if lengths else 0.0,
        )

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval < self.eval_freq:
            return True

        success_rate, mean_reward, mean_len = self._run_eval()
        self._last_eval = self.num_timesteps

        print(f"\n[Eval {self.num_timesteps:>7,}]  "
              f"success={success_rate:.2f}  "
              f"reward={mean_reward:.2f}  "
              f"len={mean_len:.1f}")

        self.logger.record("eval/success_rate", success_rate)
        self.logger.record("eval/mean_reward", mean_reward)
        self.logger.record("eval/mean_ep_length", mean_len)

        if success_rate > self.best_success:
            self.best_success = success_rate
            best_path = os.path.join(self.save_path, "best_model")
            self.model.save(best_path)
            print(f"[Eval] new best success → {success_rate:.2f}  saved → best_model.zip")

        return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   type=str, required=True)
    parser.add_argument("--run_name", type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    t_cfg   = cfg["training"]
    p_cfg   = cfg["policy"]
    log_cfg = cfg["logging"]
    e_cfg   = cfg["env"]

    device_str = t_cfg.get("device", "auto")
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name  = args.run_name or f"{cfg['experiment']['name']}_{timestamp}"
    save_path = os.path.join(log_cfg["save_dir"], run_name)
    os.makedirs(save_path, exist_ok=True)
    checkpoint_dir = os.path.join(save_path, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    def _make():
        env = make_env(horizon=e_cfg["horizon"], seed=e_cfg["seed"])
        return Monitor(env)

    train_env = DummyVecEnv([_make])

    model = SAC(
        policy          = "MlpPolicy",
        env             = train_env,
        learning_rate   = t_cfg["lr"],
        buffer_size     = t_cfg["buffer_size"],
        learning_starts = t_cfg["learning_starts"],
        batch_size      = t_cfg["batch_size"],
        gamma           = p_cfg.get("gamma",    0.99),
        tau             = p_cfg.get("tau",      0.005),
        ent_coef        = p_cfg.get("ent_coef", "auto"),
        policy_kwargs   = dict(
            net_arch      = p_cfg["net_arch"],
            activation_fn = torch.nn.ReLU,
        ),
        replay_buffer_kwargs = {"handle_timeout_termination": True},
        tensorboard_log = os.path.join(save_path, "tb"),
        device          = device_str,
        verbose         = 1,
    )

    obs_dim = train_env.observation_space.shape[0]
    print(f"\n[SAC-Button] obs_dim={obs_dim}  act_dim={train_env.action_space.shape[0]}")
    print(f"[SAC-Button] device={device_str}  ent_coef={p_cfg.get('ent_coef','auto')}")

    callbacks = [
        ProgressCallback(log_freq=t_cfg.get("log_freq", 5_000)),
        CheckpointCallback(
            save_freq = t_cfg.get("checkpoint_freq", 50_000),
            save_path = checkpoint_dir,
        ),
        EvalCallback(
            eval_env_fn=lambda: make_env(
                horizon=e_cfg["horizon"],
                seed=e_cfg["seed"],
                has_renderer=False,
            ),
            eval_freq=t_cfg.get("eval_freq", 10_000),
            n_eval_episodes=t_cfg.get("eval_episodes", 20),
            save_path=save_path,
        ),
    ]

    use_wandb = log_cfg.get("use_wandb") and WANDB_AVAILABLE
    if use_wandb:
        wandb.init(project=log_cfg["wandb_project"], name=run_name,
                   config=cfg, sync_tensorboard=True)
        callbacks.append(WandbCallback(verbose=2))

    print(f"\n[SAC-Button] Training for {t_cfg['total_timesteps']:,} steps → {save_path}\n")
    model.learn(
        total_timesteps     = t_cfg["total_timesteps"],
        callback            = callbacks,
        reset_num_timesteps = True,
        tb_log_name         = run_name,
        progress_bar        = True,
    )

    model.save(os.path.join(save_path, "final"))
    print(f"\n[Done] Model saved → {save_path}/final.zip")
    train_env.close()

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
