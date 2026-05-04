"""
Comparative Evaluation - All 3 Experiments
===========================================
Evaluates multiple trained models side-by-side and produces:
  • Console table: success rate and reward for each model
  • Bar chart:     success rate comparison saved as PNG
  • Optional video for each model (tiled 4-camera view)

Usage
-----
python RoboCasa_Code_Aide/my_rl_scripts/eval_all.py \
    --models \
        exp1_ppo_baseline/final_model.zip \
        exp2_curriculum/final_model.zip \
        exp3_bc_ppo/final_model.zip \
    --labels "PPO Baseline" "Curriculum" "BC+PPO" \
    --episodes 10

# With videos
python RoboCasa_Code_Aide/my_rl_scripts/eval_all.py \
    --models exp1/final_model.zip exp2/final_model.zip exp3/final_model.zip \
    --labels "PPO" "Curriculum" "BC+PPO" \
    --save_video --video_dir eval_videos
"""

import argparse
import os
import sys
import numpy as np
import imageio
import matplotlib.pyplot as plt

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


#  Helpers                                                                     #

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
                    width=width, height=height, depth=False,
                )
                frame = np.flipud(frame)
            else:
                frame = np.zeros((height, width, 3), dtype=np.uint8)
            row_frames.append(frame)
        tile_rows.append(np.concatenate(row_frames, axis=1))
    return np.concatenate(tile_rows, axis=0)


def evaluate_one(model_path: str, n_episodes: int, seed: int,
                 horizon: int, save_video: bool, video_dir: str, label: str):
    """Evaluate one model and return (success_rate, mean_reward, std_reward)."""
    controller_config = load_composite_controller_config(
        controller=None, robot="PandaOmron",
    )
    raw_env = MyPnPCounterToCab(
        robots="PandaOmron",
        controller_configs=controller_config,
        use_camera_obs=False,
        has_renderer=False,
        has_offscreen_renderer=save_video,
        reward_shaping=True,
        control_freq=20,
        ignore_done=False,
        horizon=horizon,
        camera_names=VIZ_CAMERAS if save_video else [],
        camera_heights=256,
        camera_widths=256,
        seed=seed,
    )
    env   = GymWrapper(raw_env, keys=None)
    model = _load_model(model_path, env=env)

    if save_video:
        os.makedirs(video_dir, exist_ok=True)

    successes, rewards = [], []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = truncated = False
        ep_reward = 0.0
        frames = []

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = env.step(action)
            ep_reward += reward
            if save_video:
                frames.append(render_tiled_frame(raw_env))

        success = raw_env._check_success()
        successes.append(float(success))
        rewards.append(ep_reward)

        status = "SUCCESS" if success else "fail   "
        print(f"    ep {ep+1:2d}/{n_episodes} [{status}] reward={ep_reward:7.2f}")

        if save_video and frames:
            vid_path = os.path.join(video_dir, f"{label}_ep{ep:02d}.mp4")
            imageio.mimsave(vid_path, frames, fps=20)

    env.close()
    return float(np.mean(successes)), float(np.mean(rewards)), float(np.std(rewards))


#  Visualisation                                                               #

def plot_comparison(labels, success_rates, mean_rewards, save_dir: str):
    """Save a bar chart comparing success rates and mean rewards."""
    os.makedirs(save_dir, exist_ok=True)
    x = np.arange(len(labels))

    _, axes = plt.subplots(1, 2, figsize=(10, 4))
    colors = plt.cm.tab10.colors[:len(labels)]

    # Success rate
    axes[0].bar(x, [r * 100 for r in success_rates], color=colors)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=11)
    axes[0].set_ylabel("Success Rate (%)")
    axes[0].set_title("Success Rate Comparison")
    axes[0].set_ylim(0, 105)
    for i, v in enumerate(success_rates):
        axes[0].text(i, v * 100 + 1.5, f"{v*100:.1f}%", ha="center", fontsize=10)

    # Mean reward
    axes[1].bar(x, mean_rewards, color=colors)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=11)
    axes[1].set_ylabel("Mean Episode Reward")
    axes[1].set_title("Mean Reward Comparison")
    for i, v in enumerate(mean_rewards):
        axes[1].text(i, v + 0.5, f"{v:.1f}", ha="center", fontsize=10)

    plt.tight_layout()
    out_path = os.path.join(save_dir, "comparison.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\n[INFO] Comparison chart saved → {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare all trained models")
    parser.add_argument("--models",     nargs="+", required=True,
                        help="Paths to model .zip files (one per experiment)")
    parser.add_argument("--labels",     nargs="+", default=None,
                        help="Display names for each model (same order as --models)")
    parser.add_argument("--episodes",   type=int, default=10)
    parser.add_argument("--horizon",    type=int, default=500)
    parser.add_argument("--seed",       type=int, default=100)
    parser.add_argument("--save_video", action="store_true")
    parser.add_argument("--video_dir",  type=str, default="eval_videos/comparison")
    parser.add_argument("--chart_dir",  type=str, default="eval_videos")
    args = parser.parse_args()

    labels = args.labels or [os.path.basename(os.path.dirname(p)) for p in args.models]
    if len(labels) != len(args.models):
        parser.error("--labels must have the same length as --models")

    success_rates, mean_rewards, std_rewards = [], [], []

    for model_path, label in zip(args.models, labels):
        print(f"\n{'─'*55}")
        print(f"  Evaluating: {label}")
        print(f"  Model     : {model_path}")
        print(f"{'─'*55}")

        sr, mr, std = evaluate_one(
            model_path=model_path,
            n_episodes=args.episodes,
            seed=args.seed,
            horizon=args.horizon,
            save_video=args.save_video,
            video_dir=os.path.join(args.video_dir, label.replace(" ", "_")),
            label=label.replace(" ", "_"),
        )
        success_rates.append(sr)
        mean_rewards.append(mr)
        std_rewards.append(std)

    #Summary table
    print(f"\n{'='*60}")
    print(f"  {'Method':<20} {'Success':>10} {'Mean Reward':>14} {'Std':>8}")
    print(f"  {'─'*20} {'─'*10} {'─'*14} {'─'*8}")
    for label, sr, mr, std in zip(labels, success_rates, mean_rewards, std_rewards):
        print(f"  {label:<20} {sr*100:>9.1f}%  {mr:>12.2f}  {std:>7.2f}")
    print(f"{'='*60}")

    #Bar chart
    plot_comparison(labels, success_rates, mean_rewards, args.chart_dir)


if __name__ == "__main__":
    main()
