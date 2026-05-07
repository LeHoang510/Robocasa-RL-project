"""
Visualize a trained SAC policy on StartCoffeeMachine.

Usage
-----
python src_button/scripts/visualize.py \
    --model src_button/checkpoints/sac_button_<timestamp>/final \
    --episodes 5
"""

import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.button_env import make_env
from stable_baselines3 import SAC


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    type=str, required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed",     type=int, default=42)
    args = parser.parse_args()

    print(f"\n[Visualize] Loading SAC model from {args.model}")
    model = SAC.load(args.model, device="cpu")

    env = make_env(horizon=200, seed=args.seed, has_renderer=True)

    successes = []
    for ep in range(args.episodes):
        obs, _ = env.reset()
        done = False; ep_reward = 0.0; step = 0
        while not done and step < 200:
            env.raw_env.render()
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = env.step(action)
            done = term or trunc; ep_reward += reward; step += 1

        success = bool(info.get("is_success", 0))
        successes.append(success)
        print(f"  Episode {ep+1}/{args.episodes}  steps={step:3d}  "
              f"reward={ep_reward:6.1f}  success={success}  "
              f"dist={info.get('rc/dist', 0):.3f}")

    env.close()
    print(f"\nSuccess rate: {np.mean(successes):.0%}  ({sum(successes)}/{args.episodes})")


if __name__ == "__main__":
    main()
