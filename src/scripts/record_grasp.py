"""
Record 4-camera tiled video of the SAC grasp policy.

Usage
-----
python src/scripts/record_grasp.py \
    --model src/checkpoints/sac_grasp_20260506_003957/final \
    --episodes 5 \
    --video_path eval_videos
"""

import argparse
import os
import sys
import numpy as np
import imageio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.pnp_env import PnPEnv, GraspEnv
from stable_baselines3 import SAC
from robosuite.controllers import load_composite_controller_config

VIZ_CAMERAS = [
    "robot0_agentview_center",
    "robot0_agentview_left",
    "robot0_agentview_right",
    "robot0_eye_in_hand",
]


def render_tiled_frame(raw_env, camera_names=VIZ_CAMERAS, width=256, height=256):
    """
    Render one frame from each camera and stitch into a single
    tiled image arranged in a 2-column grid.

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
                frame = np.flipud(frame)
            else:
                # Pad with a black tile if camera count is odd
                frame = np.zeros((height, width, 3), dtype=np.uint8)
            row_frames.append(frame)
        tile_rows.append(np.concatenate(row_frames, axis=1))
    return np.concatenate(tile_rows, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      type=str, required=True,
                        help="Path to SAC checkpoint (without .zip)")
    parser.add_argument("--episodes",   type=int, default=5)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--video_path", type=str, default="eval_videos",
                        help="Directory to save videos")
    args = parser.parse_args()

    print(f"\n[Record] Loading SAC grasp model from {args.model}")
    model = SAC.load(args.model, device="cpu")

    ctrl = load_composite_controller_config(controller=None, robot="PandaOmron")
    raw_env = PnPEnv(
        robots="PandaOmron",
        controller_configs=ctrl,
        use_camera_obs=False,
        has_renderer=False,
        has_offscreen_renderer=True,
        reward_shaping=True,
        control_freq=20,
        ignore_done=False,
        horizon=150,
        seed=args.seed,
        camera_names=VIZ_CAMERAS,
        camera_heights=256,
        camera_widths=256,
    )
    raw_env.set_difficulty(0)
    env = GraspEnv(raw_env)

    run_name = os.path.basename(os.path.dirname(args.model))
    video_dir = os.path.join(args.video_path, run_name)
    os.makedirs(video_dir, exist_ok=True)

    successes = []

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        done = False
        truncated = False
        ep_reward = 0.0
        info = {}
        frames = []

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward
            frames.append(render_tiled_frame(raw_env, VIZ_CAMERAS))

        is_success = bool(info.get("is_success", 0))
        if is_success:
            successes.append(True)
        else:
            successes.append(False)

        print(f"Episode {ep+1}/{args.episodes}  steps={len(frames):3d}  "
              f"reward={ep_reward:6.1f}  success={is_success}  "
              f"grasp={info.get('rc/grasp', 0):.2f}  "
              f"dist={info.get('rc/dist', 0):.3f}")

        if frames:
            vid_path = os.path.join(
                video_dir,
                f"eval_ep_{ep+1:02d}_{'ok' if is_success else 'fail'}.mp4",
            )
            imageio.mimsave(vid_path, frames, fps=20)
            print(f"  Saved multi-camera video ({len(VIZ_CAMERAS)} views) → {vid_path}")

    env.close()
    print(f"\nSuccess Rate: {sum(successes)}/{args.episodes} "
          f"({sum(successes)/args.episodes*100:.2f}%)")


if __name__ == "__main__":
    main()
