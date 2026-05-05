"""Evaluate a trained PPO microwave button-press policy."""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import imageio
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import MyMicrowavePressButton
from robosuite.controllers import load_composite_controller_config
from robosuite.wrappers.gym_wrapper import GymWrapper

try:
    from stable_baselines3 import PPO
except ImportError as exc:
    raise SystemExit(
        "stable-baselines3 and gymnasium are required. Install them in your env with:\n"
        "  pip install stable-baselines3 gymnasium"
    ) from exc


VIZ_CAMERAS = [
    "robot0_agentview_center",
    "robot0_agentview_left",
    "robot0_agentview_right",
    "robot0_eye_in_hand",
]


def render_tiled_frame(raw_env, camera_names=VIZ_CAMERAS, width=256, height=256):
    cols = 2
    rows = (len(camera_names) + cols - 1) // cols
    tile_rows = []

    for row in range(rows):
        row_frames = []
        for col in range(cols):
            idx = row * cols + col
            if idx < len(camera_names):
                frame = raw_env.sim.render(
                    camera_name=camera_names[idx],
                    width=width,
                    height=height,
                    depth=False,
                )
                frame = np.flipud(frame)
            else:
                frame = np.zeros((height, width, 3), dtype=np.uint8)
            row_frames.append(frame)
        tile_rows.append(np.concatenate(row_frames, axis=1))

    return np.concatenate(tile_rows, axis=0)


def make_eval_env(args):
    robots = "PandaOmron"
    controller_config = load_composite_controller_config(controller=None, robot=robots)
    raw_env = MyMicrowavePressButton(
        robots=robots,
        controller_configs=controller_config,
        use_camera_obs=False,
        has_renderer=False,
        has_offscreen_renderer=args.save_video,
        reward_shaping=True,
        control_freq=20,
        ignore_done=False,
        horizon=args.horizon,
        camera_names=VIZ_CAMERAS,
        camera_heights=256,
        camera_widths=256,
        seed=args.seed,
        render_camera=VIZ_CAMERAS[0],
        )
    raw_env.reset()
    return GymWrapper(raw_env, keys=None)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=str, default="MicrowavePressButton")
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=250)
    parser.add_argument("--save_video", action="store_true")
    parser.add_argument("--video_path", type=Path, default=Path("eval_videos"))
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device used when loading the PPO policy.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.task != "MicrowavePressButton":
        raise ValueError("This project evaluates the custom MicrowavePressButton task.")

    env = make_eval_env(args)
    model = PPO.load(str(args.model_path), env=env, device=args.device)

    video_folder = args.video_path / args.model_path.stem
    if args.save_video:
        video_folder.mkdir(parents=True, exist_ok=True)

    success_count = 0
    episode_rewards = []

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        terminated = False
        truncated = False
        episode_success = False
        frames = []
        episode_reward = 0.0

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _info = env.step(action)
            episode_reward += float(reward)
            episode_success = episode_success or bool(env.env._check_success())


            if args.save_video:
                frames.append(render_tiled_frame(env.env))
                frames.append(render_tiled_frame(env.env))

            if episode_success:
                break
            

        is_success = episode_success
        success_count += int(is_success)
        episode_rewards.append(episode_reward)

        print(
            f"Episode {ep + 1:03d}: reward={episode_reward:.2f}, "
            f"success={is_success}"
        )

        if args.save_video and frames:
            video_file = video_folder / f"eval_ep_{ep + 1:03d}.mp4"
            imageio.mimsave(video_file, frames, fps=20)
            print(f"Saved video to {video_file}")

    success_rate = success_count / args.episodes
    mean_reward = float(np.mean(episode_rewards)) if episode_rewards else 0.0
    print(f"Success Rate: {success_count}/{args.episodes} ({100.0 * success_rate:.2f}%)")
    print(f"Mean Reward: {mean_reward:.2f}")
    env.close()


if __name__ == "__main__":
    main()
