"""
PPO Evaluation Script - PnPCounterToCab (RoboCasa)
===================================================
Load a trained PPO checkpoint and run evaluation episodes.
Optionally saves a tiled multi-camera video for each episode.

Usage
-----
# Basic evaluation (5 episodes, no video)
python my_rl_scripts/eval.py --model_path checkpoints/run_name/final_model.zip

# 10 episodes with video saved
python my_rl_scripts/eval.py \
    --model_path checkpoints/run_name/ckpts/ppo_pnp_500000_steps.zip \
    --episodes 10 --save_video
"""

import argparse
import os
import sys
import numpy as np
import imageio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from my_env import MyPnPCounterToCab
from robosuite.controllers import load_composite_controller_config
from robosuite.wrappers.gym_wrapper import GymWrapper
from stable_baselines3 import PPO, SAC


def _load_model(model_path: str, env):
    path = model_path.replace(".zip", "")
    if "sac" in os.path.basename(model_path).lower():
        return SAC.load(path, env=env)
    return PPO.load(path, env=env)


VIZ_CAMERAS = [
    "robot0_agentview_center",
    "robot0_agentview_left",
    "robot0_agentview_right",
    "robot0_eye_in_hand",
]



def render_tiled_frame(raw_env, camera_names=VIZ_CAMERAS, width=256, height=256):
    """
    Render one frame per camera and stitch into a 2-column tiled image.

    Returns
    -------
    np.ndarray  shape (rows*height, cols*width, 3)  uint8
    """
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
                frame = np.flipud(frame)          # MuJoCo renders upside-down
            else:
                frame = np.zeros((height, width, 3), dtype=np.uint8)
            row_frames.append(frame)
        tile_rows.append(np.concatenate(row_frames, axis=1))
    return np.concatenate(tile_rows, axis=0)

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained PPO model on PnPCounterToCab"
    )
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained model .zip (with or without extension)")
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--episodes",   type=int, default=5)
    parser.add_argument("--horizon",    type=int, default=500)
    parser.add_argument("--save_video", action="store_true",
                        help="Render and save tiled 4-camera video per episode")
    parser.add_argument("--video_dir",  type=str, default="eval_videos")
    args = parser.parse_args()

    # Build environment 
    controller_config = load_composite_controller_config(
        controller=None,
        robot="PandaOmron",
    )
    raw_env = MyPnPCounterToCab(
        robots="PandaOmron",
        controller_configs=controller_config,
        use_camera_obs=False,
        has_renderer=False,
        has_offscreen_renderer=args.save_video,   # only needed for video
        reward_shaping=True,
        control_freq=20,
        ignore_done=False,
        horizon=args.horizon,
        camera_names=VIZ_CAMERAS,
        camera_heights=256,
        camera_widths=256,
        seed=args.seed,
    )
    raw_env.reset()  # initialise robots before GymWrapper accesses robot_model
    env = GymWrapper(raw_env, keys=None)

    # Load trained model
    print(f"[INFO] Loading model: {args.model_path}")
    model = _load_model(args.model_path, env=env)

    # Video output directory
    if args.save_video:
        run_label = os.path.basename(model_path)
        video_dir = os.path.join(args.video_dir, run_label)
        os.makedirs(video_dir, exist_ok=True)
        print(f"[INFO] Videos will be saved to: {video_dir}")

    # Evaluation loop
    success_count  = 0
    episode_rewards = []

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        done, truncated = False, False
        ep_reward = 0.0
        frames = []

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = env.step(action)
            ep_reward += reward

            if args.save_video:
                frames.append(
                    render_tiled_frame(raw_env, VIZ_CAMERAS, width=256, height=256)
                )

        success = raw_env._check_success()
        success_count += int(success)
        episode_rewards.append(ep_reward)

        status = "SUCCESS" if success else "fail   "
        print(f"  Episode {ep + 1:3d}/{args.episodes}  [{status}]  "
              f"reward = {ep_reward:8.2f}")

        if args.save_video and frames:
            vid_path = os.path.join(video_dir, f"ep_{ep:03d}.mp4")
            imageio.mimsave(vid_path, frames, fps=20)
            print(f"           -> saved {vid_path}")

    # Summary 
    print(f"\n{'='*50}")
    print(f"  Model       : {args.model_path}")
    print(f"  Episodes    : {args.episodes}")
    print(f"  Success rate: {success_count}/{args.episodes} "
          f"({100 * success_count / args.episodes:.1f} %)")
    print(f"  Mean reward : {np.mean(episode_rewards):.2f} "
          f"± {np.std(episode_rewards):.2f}")
    print(f"{'='*50}")

    env.close()


if __name__ == "__main__":
    main()
