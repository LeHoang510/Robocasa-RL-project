"""
Visualize trained policies in the PnP environment.

Usage
-----
# ACT
python src/scripts/visualize.py --type act \
    --model src/checkpoints/act_resnet18_<timestamp>/act_best.pt --episodes 5

# Diffusion
python src/scripts/visualize.py --type diffusion \
    --model src/checkpoints/diffusion_privileged_<timestamp>/diffusion_best.pt

# SAC
python src/scripts/visualize.py --type sac \
    --model src/checkpoints/sac_full_<timestamp>/final
"""

import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.pnp_env          import make_env
from env.diffusion_policy import DiffusionAgent  # noqa: F401


def run_act(model_path: str, n_episodes: int, seed: int, difficulty: int,
            camera_names: list[str] | None = None, img_h: int = 128, img_w: int = 128):
    from env.act_policy import ACTAgent
    from env.pnp_env    import PnPEnv, extract_obs
    from robosuite.controllers import load_composite_controller_config

    if camera_names is None:
        camera_names = ["robot0_agentview_right", "robot0_eye_in_hand"]

    print(f"\n[Visualize] Loading ACT model from {model_path}")
    agent = ACTAgent.load(model_path, device="cpu")
    print(f"  cameras={camera_names}  chunk_size={agent.chunk_size}")

    ctrl = load_composite_controller_config(controller=None, robot="PandaOmron")
    raw_env = PnPEnv(
        robots="PandaOmron", controller_configs=ctrl,
        use_camera_obs=True, has_renderer=True, has_offscreen_renderer=False,
        camera_names=camera_names, camera_heights=img_h, camera_widths=img_w,
        reward_shaping=True, control_freq=20, horizon=300, seed=seed,
    )
    raw_env.set_difficulty(difficulty)

    def _target_pos():
        try:
            cab    = raw_env.cab
            offset = cab.get_reset_regions(raw_env)["level0"]["offset"]
            return (np.array(cab.pos) + np.array(offset)).astype(np.float32)
        except Exception:
            return np.array([2.25, -0.2, 1.42], dtype=np.float32)

    successes = []
    for ep in range(n_episodes):
        obs_dict = raw_env.reset()
        tgt_pos  = _target_pos()
        agent.reset()
        done = False; ep_reward = 0.0; step = 0
        while not done and step < 300:
            raw_env.render()
            imgs    = np.stack([obs_dict[f"{cam}_image"] for cam in camera_names], axis=0)
            proprio = extract_obs(obs_dict, tgt_pos)
            action  = agent.predict(imgs, proprio)
            obs_dict, reward, done, info = raw_env.step(action)
            ep_reward += reward; step += 1
        success = bool(raw_env._check_success())
        successes.append(success)
        print(f"  Episode {ep+1}/{n_episodes}  steps={step:3d}  "
              f"reward={ep_reward:6.1f}  success={success}")
    raw_env.close()
    print(f"\nSuccess rate: {np.mean(successes):.0%}  ({sum(successes)}/{n_episodes})")


def run_diffusion(model_path: str, n_episodes: int, seed: int, difficulty: int):
    print(f"\n[Visualize] Loading diffusion model from {model_path}")
    agent = DiffusionAgent.load(model_path, device="cpu")
    print(f"  obs_dim={agent.obs_dim}  act_dim={agent.act_dim}")

    env = make_env(horizon=300, seed=seed, privileged=True,
                   has_renderer=True, difficulty=difficulty)
    successes = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        agent.reset(first_obs=obs)
        done = False; ep_reward = 0.0; step = 0
        while not done and step < 300:
            env.raw_env.render()
            action = agent.predict(obs)
            obs, reward, term, trunc, info = env.step(action)
            done = term or trunc; ep_reward += reward; step += 1
        success = bool(env.raw_env._check_success())
        successes.append(success)
        print(f"  Episode {ep+1}/{n_episodes}  steps={step:3d}  "
              f"reward={ep_reward:6.1f}  success={success}  "
              f"stage={info.get('rc/stage', 0):.1f}")
    env.close()
    print(f"\nSuccess rate: {np.mean(successes):.0%}  ({sum(successes)}/{n_episodes})")


def run_sac(model_path: str, n_episodes: int, seed: int, difficulty: int,
            grasp_only: bool):
    from stable_baselines3 import SAC

    print(f"\n[Visualize] Loading SAC model from {model_path}")
    horizon = 150 if grasp_only else 300
    env = make_env(horizon=horizon, seed=seed, privileged=not grasp_only,
                   has_renderer=True, difficulty=difficulty, grasp_only=grasp_only)

    model = SAC.load(model_path, device="cpu")
    print(f"  grasp_only={grasp_only}  horizon={horizon}")

    successes = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False; ep_reward = 0.0; step = 0
        while not done and step < horizon:
            env.raw_env.render()
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = env.step(action)
            done = term or trunc; ep_reward += reward; step += 1
        success = bool(info.get("is_success", 0))
        successes.append(success)
        print(f"  Episode {ep+1}/{n_episodes}  steps={step:3d}  "
              f"reward={ep_reward:6.1f}  success={success}  "
              f"grasp={info.get('rc/grasp', 0):.2f}")
    env.close()
    print(f"\nSuccess rate: {np.mean(successes):.0%}  ({sum(successes)}/{n_episodes})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      type=str, required=True,
                        help="Path to .pt (diffusion) or .zip (SAC) checkpoint")
    parser.add_argument("--type",       type=str, default="sac",
                        choices=["act", "diffusion", "sac"],
                        help="Model type (default: sac)")
    parser.add_argument("--episodes",   type=int, default=5)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--difficulty", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument("--grasp_only", action="store_true",
                        help="Use GraspEnv (stage-1 SAC checkpoint)")
    args = parser.parse_args()

    if args.type == "act":
        run_act(args.model, args.episodes, args.seed, args.difficulty)
    elif args.type == "diffusion":
        run_diffusion(args.model, args.episodes, args.seed, args.difficulty)
    else:
        run_sac(args.model, args.episodes, args.seed, args.difficulty, args.grasp_only)


if __name__ == "__main__":
    main()
