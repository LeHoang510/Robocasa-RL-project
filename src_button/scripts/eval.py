"""
Headless evaluation of a trained SAC policy on StartCoffeeMachine.

Usage
-----
python src_button/scripts/eval.py \
    --model src_button/checkpoints/<run>/best_model \
    --episodes 100
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
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--horizon",  type=int, default=200)
    args = parser.parse_args()

    print(f"\n[Eval] Loading SAC model from {args.model}")
    model = SAC.load(args.model, device="cpu")

    successes, rewards, lengths = [], [], []

    for ep in range(args.episodes):
        env = make_env(horizon=args.horizon, seed=args.seed + ep, has_renderer=False)
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        step = 0
        info = {}
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = env.step(action)
            done = term or trunc
            ep_reward += reward
            step += 1
        env.close()

        success = bool(info.get("is_success", 0))
        successes.append(success)
        rewards.append(ep_reward)
        lengths.append(step)

        if (ep + 1) % 10 == 0:
            sr = np.mean(successes)
            print(f"  [{ep+1:>3}/{args.episodes}]  "
                  f"success_so_far={sr:.2f}  "
                  f"last_reward={ep_reward:6.1f}  "
                  f"last_steps={step:3d}  "
                  f"last_dist={info.get('rc/dist', 0):.3f}")

    print(f"\n{'='*50}")
    print(f"Episodes  : {args.episodes}")
    print(f"Success   : {np.mean(successes):.2%}  ({sum(successes)}/{args.episodes})")
    print(f"Avg reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"Avg steps : {np.mean(lengths):.1f} ± {np.std(lengths):.1f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
