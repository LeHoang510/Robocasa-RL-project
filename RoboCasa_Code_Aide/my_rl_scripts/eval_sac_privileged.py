"""
Evaluate a SAC model trained with privileged 25D observations.
"""

import argparse
import os
import sys

import imageio
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from my_env import MyPnPCounterToCab
from my_env.privileged_env import PrivilegedPnPEnv
from robosuite.controllers import load_composite_controller_config
from stable_baselines3 import SAC


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
    for r in range(rows):
        row_frames = []
        for c in range(cols):
            idx = r * cols + c
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


def main():
    parser = argparse.ArgumentParser(description="Evaluate privileged SAC model")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--config",     type=str, default=None,
                        help="Training config yaml — used to read horizon automatically")
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--episodes",   type=int, default=10)
    parser.add_argument("--horizon",    type=int, default=None,
                        help="Override episode horizon (default: read from --config, else 300)")
    parser.add_argument("--save_video", action="store_true")
    parser.add_argument("--video_dir",  type=str, default="eval_videos")
    args = parser.parse_args()

    # Resolve horizon: CLI > config file > safe default matching training
    if args.horizon is not None:
        horizon = args.horizon
    elif args.config is not None:
        with open(args.config) as f:
            _cfg = yaml.safe_load(f)
        horizon = _cfg["env"]["horizon"]
        print(f"[INFO] horizon={horizon} (from {args.config})")
    else:
        horizon = 300
        print(f"[INFO] horizon={horizon} (default — pass --config to read from yaml)")

    controller_config = load_composite_controller_config(
        controller=None, robot="PandaOmron",
    )
    raw_env = MyPnPCounterToCab(
        robots="PandaOmron",
        controller_configs=controller_config,
        use_camera_obs=False,
        has_renderer=False,
        has_offscreen_renderer=args.save_video,
        reward_shaping=True,
        control_freq=20,
        ignore_done=False,
        horizon=horizon,
        camera_names=VIZ_CAMERAS if args.save_video else [],
        camera_heights=256,
        camera_widths=256,
        seed=args.seed,
    )
    env = PrivilegedPnPEnv(raw_env)

    print(f"[INFO] Loading model: {args.model_path}")
    model = SAC.load(args.model_path, env=env)

    if args.save_video:
        run_label = os.path.basename(args.model_path).replace(".zip", "")
        video_dir = os.path.join(args.video_dir, run_label)
        os.makedirs(video_dir, exist_ok=True)
        print(f"[INFO] Videos -> {video_dir}")

    success_count = 0
    episode_rewards = []

    for ep in range(args.episodes):
        obs, _ = env.reset()
        done = truncated = False
        ep_reward = 0.0
        frames = []

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = env.step(action)
            ep_reward += reward
            if args.save_video:
                frames.append(render_tiled_frame(raw_env))

        success = raw_env._check_success()
        success_count += int(success)
        episode_rewards.append(ep_reward)

        status = "SUCCESS" if success else "fail   "
        print(f"  Episode {ep + 1:3d}/{args.episodes}  [{status}]  reward = {ep_reward:8.2f}")

        if args.save_video and frames:
            vid_path = os.path.join(video_dir, f"ep_{ep:03d}.mp4")
            imageio.mimsave(vid_path, frames, fps=20)
            print(f"           -> saved {vid_path}")

    print(f"\n{'=' * 50}")
    print(f"  Model       : {args.model_path}")
    print(f"  Episodes    : {args.episodes}")
    print(f"  Success rate: {success_count}/{args.episodes} ({100 * success_count / args.episodes:.1f} %)")
    print(f"  Mean reward : {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"{'=' * 50}")

    env.close()


if __name__ == "__main__":
    main()
