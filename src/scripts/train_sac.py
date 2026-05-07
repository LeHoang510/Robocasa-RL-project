"""
SAC training for PnPCounterToCabinet — two-stage curriculum:

  Stage 1 (grasp_only=true):
    Train until robot reliably grasps the apple (success > 50%).
    Short horizon (150 steps), simple reward (reach + grasp).

  Stage 2 (grasp_only=false, --init_from stage1_checkpoint):
    Full pick-and-place with FSM reward.
    Initialize actor/critic from stage 1 weights → skips grasp rediscovery.

Usage
-----
# Stage 1 — grasp only
python src/scripts/train_sac.py --config src/config/sac_grasp.yaml

# Stage 2 — full task, init from stage 1
python src/scripts/train_sac.py --config src/config/sac_full.yaml \
    --init_from src/checkpoints/sac_grasp_<timestamp>/final
"""

import argparse
import os
import sys
from datetime import datetime

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.pnp_env import make_env

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
# Progress callback
# ---------------------------------------------------------------------------

class ProgressCallback(BaseCallback):
    _KEYS = ["rc/reach", "rc/grip", "rc/contact", "rc/grasp",
             "rc/lift", "rc/transport", "rc/stage"]

    def __init__(self, log_freq: int = 10_000):
        super().__init__()
        self.log_freq  = log_freq
        self._accum    = {k: 0.0 for k in self._KEYS}
        self._count    = 0
        self._last_log = 0
        self._successes = []

    def _on_step(self) -> bool:
        info = self.locals["infos"][0]
        for k in self._KEYS:
            self._accum[k] += float(info.get(k, 0.0))
        self._count += 1

        if "is_success" in info:
            self._successes.append(float(info["is_success"]))

        if self.num_timesteps - self._last_log >= self.log_freq:
            n = max(self._count, 1)
            avg = {k: self._accum[k] / n for k in self._KEYS}
            success_rate = float(np.mean(self._successes)) if self._successes else 0.0

            print(
                f"\n[Progress] step={self.num_timesteps:>7,}  "
                f"success={success_rate:.2f}  "
                f"stage={avg['rc/stage']:.2f}  "
                f"reach={avg['rc/reach']:.3f}  "
                f"grip={avg['rc/grip']:.3f}  "
                f"grasp={avg['rc/grasp']:.3f}  "
                f"lift={avg.get('rc/lift', 0):.3f}  "
                f"transport={avg.get('rc/transport', 0):.3f}"
            )

            for k, v in avg.items():
                self.logger.record(f"progress/{k.split('/')[-1]}", v)
            self.logger.record("progress/success_rate", success_rate)

            self._accum    = {k: 0.0 for k in self._KEYS}
            self._count    = 0
            self._successes = []
            self._last_log = self.num_timesteps

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


# ---------------------------------------------------------------------------
# Weight transfer (stage 1 → stage 2)
# ---------------------------------------------------------------------------

def _transfer_weights(model: SAC, src_path: str, device: str):
    """Copy actor + critic weights from stage-1 checkpoint into stage-2 model."""
    src = SAC.load(src_path, device=device)

    # Actor
    model.policy.actor.load_state_dict(src.policy.actor.state_dict())

    # Critic (both Q-networks)
    model.policy.critic.load_state_dict(src.policy.critic.state_dict())
    model.policy.critic_target.load_state_dict(src.policy.critic_target.state_dict())

    print(f"[Init] Transferred actor + critic weights from {src_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",    type=str, required=True)
    parser.add_argument("--init_from", type=str, default=None,
                        help="Stage-1 checkpoint path to init actor/critic")
    parser.add_argument("--run_name",  type=str, default=None)
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

    grasp_only = e_cfg.get("grasp_only", False)
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name   = args.run_name or f"{cfg['experiment']['name']}_{timestamp}"
    save_path  = os.path.join(log_cfg["save_dir"], run_name)
    os.makedirs(save_path, exist_ok=True)

    # ------------------------------------------------------------------
    # Environment
    # ------------------------------------------------------------------
    def _make():
        env = make_env(
            horizon    = e_cfg["horizon"],
            seed       = e_cfg["seed"],
            privileged = not grasp_only,
            grasp_only = grasp_only,
            difficulty = e_cfg.get("difficulty", 0),
        )
        return Monitor(env)

    train_env = DummyVecEnv([_make])

    # ------------------------------------------------------------------
    # SAC model
    # ------------------------------------------------------------------
    model = SAC(
        policy          = "MlpPolicy",
        env             = train_env,
        learning_rate   = t_cfg["lr"],
        buffer_size     = t_cfg["buffer_size"],
        learning_starts = t_cfg["learning_starts"],
        batch_size      = t_cfg["batch_size"],
        gamma           = p_cfg.get("gamma",    0.99),
        tau             = p_cfg.get("tau",      0.005),
        ent_coef        = p_cfg.get("ent_coef", 0.3),
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
    print(f"\n[SAC] obs_dim={obs_dim}  act_dim={train_env.action_space.shape[0]}")
    print(f"[SAC] grasp_only={grasp_only}  device={device_str}")
    print(f"[SAC] ent_coef={p_cfg.get('ent_coef', 0.3)}")

    # Transfer stage-1 weights if provided
    if args.init_from:
        _transfer_weights(model, args.init_from, device_str)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    callbacks = [
        ProgressCallback(log_freq=t_cfg.get("log_freq", 10_000)),
        CheckpointCallback(
            save_freq = t_cfg.get("checkpoint_freq", 50_000),
            save_path = os.path.join(save_path, "checkpoints"),
        ),
    ]

    use_wandb = log_cfg.get("use_wandb") and WANDB_AVAILABLE
    if use_wandb:
        wandb.init(project=log_cfg["wandb_project"], name=run_name, config=cfg,
                   sync_tensorboard=True)
        callbacks.append(WandbCallback(verbose=2))

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    print(f"\n[SAC] Training for {t_cfg['total_timesteps']:,} steps → {save_path}\n")
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
