"""Train PPO on the custom RoboCasa counter-to-cabinet task."""

import argparse
import csv
import os
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import MyPnPCounterToCab
from robosuite.controllers import load_composite_controller_config
from robosuite.wrappers.gym_wrapper import GymWrapper

try:
    import gymnasium as gym
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
except ImportError as exc:
    raise SystemExit(
        "stable-baselines3 and gymnasium are required. Install them in your env with:\n"
        "  pip install stable-baselines3 gymnasium"
    ) from exc


class SuccessInfoWrapper(gym.Wrapper):
    """Expose RoboCasa success and optionally terminate when the task is solved."""

    def __init__(self, env, terminate_on_success=True):
        super().__init__(env)
        self.terminate_on_success = terminate_on_success
        self.episode_success = False

    def reset(self, **kwargs):
        self.episode_success = False
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        success = bool(self.env.env._check_success())
        self.episode_success = self.episode_success or success
        info["is_success"] = self.episode_success
        if success and self.terminate_on_success:
            terminated = True
        return obs, reward, terminated, truncated, info


class TrainingCurveCallback(BaseCallback):
    """Write episode metrics to CSV and periodically render a reward curve."""

    def __init__(self, output_dir: Path, plot_every: int = 10):
        super().__init__()
        self.output_dir = output_dir
        self.plot_every = plot_every
        self.csv_path = output_dir / "training_curve.csv"
        self.png_path = output_dir / "training_curve.png"
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_training_start(self):
        with self.csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "timesteps", "reward", "length", "success"])

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" not in info:
                continue

            ep_reward = float(info["episode"]["r"])
            ep_length = int(info["episode"]["l"])
            success = bool(info.get("is_success", False))

            self.episode_rewards.append(ep_reward)
            self.episode_lengths.append(ep_length)
            episode = len(self.episode_rewards)

            with self.csv_path.open("a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([episode, self.num_timesteps, ep_reward, ep_length, int(success)])

            if episode % self.plot_every == 0:
                self._save_plot()

        return True

    def _on_training_end(self):
        if self.episode_rewards:
            self._save_plot()

    def _save_plot(self):
        rewards = np.array(self.episode_rewards, dtype=np.float32)
        window = min(20, len(rewards))
        if window > 1:
            kernel = np.ones(window, dtype=np.float32) / window
            smoothed = np.convolve(rewards, kernel, mode="valid")
            x_smooth = np.arange(window, len(rewards) + 1)
        else:
            smoothed = rewards
            x_smooth = np.arange(1, len(rewards) + 1)

        plt.figure(figsize=(8, 4.5))
        plt.plot(np.arange(1, len(rewards) + 1), rewards, alpha=0.35, label="episode")
        plt.plot(x_smooth, smoothed, linewidth=2, label=f"mean/{window}")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("PPO training curve: MyPnPCounterToCab")
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.png_path, dpi=160)
        plt.close()


def make_env(args, rank: int):
    def _init():
        robots = "PandaOmron"
        controller_config = load_composite_controller_config(controller=None, robot=robots)
        env = MyPnPCounterToCab(
            robots=robots,
            controller_configs=controller_config,
            use_camera_obs=False,
            has_renderer=not args.headless,
            has_offscreen_renderer=False,
            reward_shaping=True,
            control_freq=20,
            renderer="mjviewer",
            ignore_done=False,
            seed=args.seed + rank,
            horizon=args.horizon,
        )
        env.reset()
        env = GymWrapper(env, keys=None)
        env = SuccessInfoWrapper(env)
        env = Monitor(env, str(args.output_dir / "monitor" / f"env_{rank}"))
        env.reset(seed=args.seed + rank)
        return env

    return _init


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=str, default="PnPCounterToCab")
    parser.add_argument("--total_timesteps", type=int, default=200_000)
    parser.add_argument("--n_envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--n_steps", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--checkpoint_freq", type=int, default=25_000)
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device for PPO. CPU is recommended for the default MLP policy.",
    )
    parser.add_argument("--output_dir", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.task != "PnPCounterToCab":
        raise ValueError("This project trains the custom PnPCounterToCab task.")

    if args.device.startswith("cuda"):
        print(
            "Torch CUDA:",
            torch.__version__,
            torch.version.cuda,
            torch.cuda.is_available(),
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO CUDA",
        )
        if not torch.cuda.is_available():
            raise RuntimeError(
                "--device cuda was requested, but torch.cuda.is_available() is False "
                "inside this training process."
            )

    if args.output_dir is None:
        run_name = datetime.now().strftime("pnp_counter_to_cab_%Y%m%d_%H%M%S")
        args.output_dir = Path("runs") / run_name
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "checkpoints").mkdir(exist_ok=True)
    (args.output_dir / "monitor").mkdir(exist_ok=True)

    env_fns = [make_env(args, i) for i in range(args.n_envs)]
    env = SubprocVecEnv(env_fns) if args.n_envs > 1 else DummyVecEnv(env_fns)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=args.seed,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        ent_coef=args.ent_coef,
        device=args.device,
        tensorboard_log=str(args.output_dir / "tensorboard"),
    )

    callbacks = [
        CheckpointCallback(
            save_freq=max(args.checkpoint_freq // max(args.n_envs, 1), 1),
            save_path=str(args.output_dir / "checkpoints"),
            name_prefix="ppo_robocasa",
        ),
        TrainingCurveCallback(args.output_dir),
    ]

    model.learn(total_timesteps=args.total_timesteps, callback=callbacks)

    final_path = args.output_dir / "ppo_robocasa_final"
    model.save(str(final_path))
    env.close()

    print(f"Saved final model to {final_path}.zip")
    print(f"Saved training metrics to {args.output_dir / 'training_curve.csv'}")
    print(f"Saved training plot to {args.output_dir / 'training_curve.png'}")


if __name__ == "__main__":
    main()
