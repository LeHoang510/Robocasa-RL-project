"""
SAC with Privileged Observations — PnPCounterToCab (RoboCasa)
=============================================================
Industry-standard pipeline for pick-and-place manipulation:

  Privileged state obs (25D, includes object pose + target pos + relative vectors)
  + Dense staged reward (reach → grasp → lift → transport)
  + SAC with automatic entropy tuning
  + 3-phase curriculum (reward weight schedule)
  + Demo warmstart (optional: preload replay buffer with BC demonstrations)

The privileged obs gives the policy full access to object state, unlike the
16D proprioception-only obs used in the BC/Diffusion/IQL experiments.

Usage
-----
python RoboCasa_Code_Aide/my_rl_scripts/train_sac_privileged.py \
    --config RoboCasa_Code_Aide/config/exp9_sac_privileged.yaml

# With demo warmstart:
python RoboCasa_Code_Aide/my_rl_scripts/train_sac_privileged.py \
    --config RoboCasa_Code_Aide/config/exp9_sac_privileged.yaml \
    --warmstart_demos 50
"""

import argparse
import os
import sys
from datetime import datetime

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from my_env import MyPnPCounterToCab
from my_env.privileged_env import PrivilegedPnPEnv
from robosuite.controllers import load_composite_controller_config

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

try:
    import wandb
    from wandb.integration.sb3 import WandbCallback
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def _make_raw_env(cfg: dict, seed: int = 0, has_renderer: bool = False):
    controller_config = load_composite_controller_config(
        controller=None, robot="PandaOmron",
    )
    return MyPnPCounterToCab(
        robots="PandaOmron",
        controller_configs=controller_config,
        use_camera_obs=False,
        has_renderer=has_renderer,
        has_offscreen_renderer=False,
        reward_shaping=True,
        control_freq=20,
        renderer="mjviewer",
        ignore_done=False,
        horizon=cfg["env"]["horizon"],
        seed=seed,
    )


def _make_priv_env(cfg: dict, seed: int = 0) -> PrivilegedPnPEnv:
    raw = _make_raw_env(cfg, seed=seed)
    return PrivilegedPnPEnv(raw)


def _make_monitored_priv_env(cfg: dict, seed: int = 0) -> Monitor:
    return Monitor(_make_priv_env(cfg, seed=seed))


# ---------------------------------------------------------------------------
# Progress callback — logs per-component reward breakdown
# ---------------------------------------------------------------------------

class BufferCheckpointCallback(BaseCallback):
    """Saves model + replay buffer together so training can resume from any checkpoint."""

    def __init__(self, save_freq: int, save_path: str, name_prefix: str = "sac_priv"):
        super().__init__()
        self.save_freq   = save_freq
        self.save_path   = save_path
        self.name_prefix = name_prefix

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq == 0:
            tag  = f"{self.name_prefix}_{self.num_timesteps}_steps"
            self.model.save(os.path.join(self.save_path, tag))
            self.model.save_replay_buffer(os.path.join(self.save_path, f"{tag}_buffer"))
            print(f"[Checkpoint] saved model + buffer → {tag}")
        return True


class ProgressCallback(BaseCallback):
    """
    Prints a per-component reward breakdown every `log_freq` steps so you
    can see exactly which sub-task the robot has learned without waiting for
    success_rate to become non-zero.

    Output columns:
      reach  — eef approaching object        (phase 0+)
      grip   — gripper closing near object   (phase 0+, fixed weight)
      grasp  — object lifted + near eef      (phase 1+)
      lift   — object height above table     (phase 1+)
      transport — object approaching cabinet (phase 0 tiny, phase 2 full)
    """

    _KEYS = ["rc/reach", "rc/grip", "rc/contact", "rc/grasp", "rc/lift", "rc/transport", "rc/stage"]

    def __init__(self, log_freq: int = 10_000, verbose: int = 1):
        super().__init__(verbose)
        self.log_freq = log_freq
        self._accum   = {k: 0.0 for k in self._KEYS}
        self._count   = 0
        self._last_log = 0

    def _on_step(self) -> bool:
        info = self.locals["infos"][0]
        for k in self._KEYS:
            self._accum[k] += float(info.get(k, 0.0))
        self._count += 1

        if self.num_timesteps - self._last_log >= self.log_freq:
            n = max(self._count, 1)
            reach     = self._accum["rc/reach"]     / n
            grip      = self._accum["rc/grip"]      / n
            contact   = self._accum["rc/contact"]   / n
            grasp     = self._accum["rc/grasp"]     / n
            lift      = self._accum["rc/lift"]      / n
            transport = self._accum["rc/transport"] / n
            stage     = self._accum["rc/stage"]     / n

            print(
                f"\n[Progress] step={self.num_timesteps:>7,}  "
                f"stage={stage:.2f}"
                f"  reach={reach:.3f}"
                f"  contact={contact:.3f}"
                f"  grip={grip:.3f}"
                f"  grasp={grasp:.3f}"
                f"  lift={lift:.3f}"
                f"  transport={transport:.3f}"
            )

            self.logger.record("progress/stage",     stage)
            self.logger.record("progress/reach",     reach)
            self.logger.record("progress/contact",   contact)
            self.logger.record("progress/grip",      grip)
            self.logger.record("progress/grasp",     grasp)
            self.logger.record("progress/lift",      lift)
            self.logger.record("progress/transport", transport)

            self._accum   = {k: 0.0 for k in self._KEYS}
            self._count   = 0
            self._last_log = self.num_timesteps

        return True


# CurriculumCallback removed: reward weights are now driven by FSM task state
# (_compute_reward in privileged_env.py), not by training timestep.


# ---------------------------------------------------------------------------
# Demo warmstart — preload replay buffer with offline demo transitions
# ---------------------------------------------------------------------------

def _warmstart_replay_buffer(model: SAC, cfg: dict, n_episodes: int, device: str):
    """
    Load offline demo transitions into the SAC replay buffer.

    Transitions are re-simulated so that the privileged obs (including object
    state) is properly populated — the LeRobot parquet only has 16D obs, but
    here we collect 19D privileged obs by stepping through the env.

    Note: re-simulating with demo actions in a fresh env gives different object
    positions than the original demo, so these are approximate demonstrations.
    We use them purely to warm-start the replay buffer with "reasonable motions"
    rather than exact replays.
    """
    from my_env.bc_dataset import BCDemoDataset
    from my_env.privileged_env import extract_privileged_obs

    d_cfg = cfg["dataset"]
    print(f"[Warmstart] Loading {n_episodes} demo episodes …")
    dataset = BCDemoDataset(
        dataset_dir  = d_cfg["dataset_dir"],
        split        = "train",
        max_episodes = n_episodes,
        verbose      = False,
    )

    env = _make_priv_env(cfg, seed=0)
    env.reset()

    added = 0
    for ep_idx in range(min(n_episodes, 20)):   # cap at 20 to avoid long setup
        obs_dict = env.raw_env.reset()
        env._target_pos = env._read_target_pos()
        obs = extract_privileged_obs(obs_dict, env._target_pos)
        done = False
        step = 0

        # Use episode length from dataset as cap
        ep_len = len(dataset) // n_episodes

        while not done and step < ep_len:
            # Use dataset actions (already HDF5-ordered)
            act_idx = min(ep_idx * ep_len + step, len(dataset) - 1)
            action  = dataset.actions[act_idx].numpy()

            obs_dict, _, done, info = env.raw_env.step(action)
            next_obs = extract_privileged_obs(obs_dict, env._target_pos)
            success  = bool(env.raw_env._check_success())
            reward   = env._compute_reward(obs_dict, success)
            terminated = success
            truncated  = done and not success

            model.replay_buffer.add(
                obs, next_obs, action, reward, terminated or truncated, [info]
            )
            obs = next_obs
            step += 1
            added += 1

    env.close()
    print(f"[Warmstart] Added {added} transitions to replay buffer.")


# ---------------------------------------------------------------------------
# BC → SAC weight transfer
# ---------------------------------------------------------------------------

def _load_bc_weights_into_sac(model: SAC, bc_path: str, device_str: str):
    """
    Transfer BC pre-trained weights into the SAC actor.

    BC MLP layout  (train_bc_privileged.py):
      Linear(25,256) ReLU  Linear(256,256) ReLU  Linear(256,256) ReLU  Linear(256,12)

    SAC actor layout (SB3, net_arch=[256,256,256]):
      latent_pi : Linear(25,256) ReLU  Linear(256,256) ReLU  Linear(256,256) ReLU
      mu        : Linear(256, 12)
      log_std   : Linear(256, 12)   ← untouched; SAC learns entropy from scratch

    Obs normalisation (mean/std recorded during BC training) is fused into the
    first linear layer so the SAC actor can accept raw 25D obs without a wrapper.
    """
    ckpt     = torch.load(bc_path, map_location=device_str, weights_only=False)
    obs_mean = torch.tensor(ckpt["obs_mean"], dtype=torch.float32)
    obs_std  = torch.tensor(ckpt["obs_std"],  dtype=torch.float32)

    # Rebuild BC net and load weights (state_dict keys are under "net.*")
    import torch.nn as nn
    from my_rl_scripts.train_bc_privileged import PrivBCNet
    bc_net = PrivBCNet()
    bc_net.load_state_dict(ckpt["state_dict"])

    bc_linears  = [m for m in bc_net.net if isinstance(m, nn.Linear)]
    sac_linears = [m for m in model.policy.actor.latent_pi if isinstance(m, nn.Linear)]

    # Fuse normalisation into first layer so raw obs work correctly:
    #   net(obs) with normalised input  ≡  net_fused(obs) with raw input
    W0 = bc_linears[0].weight.data          # (256, 25)
    b0 = bc_linears[0].bias.data            # (256,)
    W0_fused = W0 / obs_std.unsqueeze(0)    # (256, 25)
    b0_fused = b0 - W0_fused @ obs_mean
    sac_linears[0].weight.data.copy_(W0_fused)
    sac_linears[0].bias.data.copy_(b0_fused)

    # Remaining hidden layers: copy directly (no normalisation involved)
    for bc_l, sac_l in zip(bc_linears[1:-1], sac_linears[1:]):
        sac_l.weight.data.copy_(bc_l.weight.data)
        sac_l.bias.data.copy_(bc_l.bias.data)

    # Output layer → mu (mean action head)
    model.policy.actor.mu.weight.data.copy_(bc_linears[-1].weight.data)
    model.policy.actor.mu.bias.data.copy_(bc_linears[-1].bias.data)

    # Initialize log_std to small value so BC signal isn't drowned by noise.
    # SB3 log_std is a nn.Linear; bias = -2.0 → std≈0.13 (dominates the default 0.0).
    with torch.no_grad():
        model.policy.actor.log_std.weight.data.fill_(0.0)
        model.policy.actor.log_std.bias.data.fill_(-2.0)

    print(f"[BC→SAC] Loaded BC weights from {bc_path}")
    print(f"[BC→SAC] Transferred: latent_pi + mu (BC) | log_std init to -2.0 (std≈0.13)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SAC with privileged obs for PnP")
    parser.add_argument("--config",           type=str, required=True)
    parser.add_argument("--run_name",         type=str, default=None)
    parser.add_argument("--warmstart_demos",  type=int, default=0,
                        help="Number of demo episodes to preload into replay buffer (0=disabled)")
    parser.add_argument("--resume",           type=str, default=None,
                        help="Path to checkpoint zip (without .zip) to resume from. "
                             "Expects a matching _buffer.pkl in the same directory.")
    parser.add_argument("--bc_weights",      type=str, default=None,
                        help="Path to BC pre-trained weights .pt from train_bc_privileged.py. "
                             "Initialises the SAC actor with grasping behavior before RL starts.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    t_cfg   = cfg["training"]
    log_cfg = cfg["logging"]

    device_str = t_cfg.get("device", "auto")
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name  = args.run_name or f"{cfg['experiment']['name']}_{timestamp}"
    save_path = os.path.join(log_cfg["save_dir"], run_name)
    os.makedirs(save_path, exist_ok=True)

    print("[INFO] Building training environment …")
    train_env = _make_monitored_priv_env(cfg, seed=cfg["env"]["seed"])

    # print("[INFO] Building eval environment …")
    # eval_env  = _make_monitored_priv_env(cfg, seed=cfg["env"]["seed"] + 999)

    # ------------------------------------------------------------------
    # SAC model
    # ------------------------------------------------------------------
    p_cfg = cfg["policy"]
    model = SAC(
        policy              = "MlpPolicy",
        env                 = DummyVecEnv([lambda: train_env]),
        learning_rate       = t_cfg["lr"],
        buffer_size         = t_cfg["buffer_size"],
        learning_starts     = t_cfg["learning_starts"],
        batch_size          = t_cfg["batch_size"],
        gamma               = p_cfg.get("gamma",    0.99),
        tau                 = p_cfg.get("tau",      0.005),
        ent_coef            = p_cfg.get("ent_coef", "auto"),  # auto entropy tuning
        target_entropy      = p_cfg.get("target_entropy", "auto"),
        policy_kwargs       = dict(
            net_arch        = p_cfg["net_arch"],
            activation_fn   = torch.nn.ReLU,
        ),
        # Correctly handle horizon-truncated episodes: bootstrap V(s') instead
        # of treating timeout as a terminal state (avoids value underestimation).
        replay_buffer_kwargs = {"handle_timeout_termination": True},
        tensorboard_log     = os.path.join(save_path, "tb"),
        device              = device_str,
        verbose             = 1,
    )

    print(f"[INFO] SAC  obs_dim={train_env.observation_space.shape[0]}  "
          f"action_dim={train_env.action_space.shape[0]}  device={device_str}")
    print(f"[INFO] ent_coef={p_cfg.get('ent_coef','auto')}  device={device_str}")
    print(f"[INFO] Reward: FSM-gated (stage advances on actual task progress)")

    # ------------------------------------------------------------------
    # Resume from checkpoint (model + replay buffer)
    # ------------------------------------------------------------------
    if args.resume:
        print(f"[INFO] Resuming from checkpoint: {args.resume}")
        model = SAC.load(args.resume, env=DummyVecEnv([lambda: train_env]),
                         device=device_str)
        buf_path = args.resume + "_buffer"
        model.load_replay_buffer(buf_path)
        print(f"[INFO] Replay buffer loaded: {buf_path}.pkl  "
              f"({model.replay_buffer.size()} transitions)")

    # ------------------------------------------------------------------
    # BC weight initialisation — transfer pre-trained actor weights into SAC
    # ------------------------------------------------------------------
    if args.bc_weights:
        _load_bc_weights_into_sac(model, args.bc_weights, device_str)

    # ------------------------------------------------------------------
    # Demo warmstart
    # ------------------------------------------------------------------
    if args.warmstart_demos > 0 and "dataset" in cfg:
        _warmstart_replay_buffer(model, cfg, args.warmstart_demos, device_str)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    callbacks = []

    # Per-component reward progress
    callbacks.append(ProgressCallback(
        log_freq = t_cfg.get("eval_freq", 10_000),
        verbose  = 1,
    ))

    # Checkpoint — saves model + replay buffer so training can resume from any point
    callbacks.append(BufferCheckpointCallback(
        save_freq  = t_cfg.get("checkpoint_freq", 50_000),
        save_path  = os.path.join(save_path, "checkpoints"),
        name_prefix = "sac_priv",
    ))

    # Eval (saves best model automatically) — comment out during debug to save time
    # callbacks.append(EvalCallback(
    #     eval_env          = DummyVecEnv([lambda: eval_env]),
    #     best_model_save_path = save_path,
    #     log_path          = os.path.join(save_path, "eval_logs"),
    #     eval_freq         = t_cfg.get("eval_freq",     10_000),
    #     n_eval_episodes   = t_cfg.get("eval_episodes", 20),
    #     deterministic     = True,
    #     verbose           = 1,
    # ))

    # WandB
    use_wandb = log_cfg.get("use_wandb") and WANDB_AVAILABLE
    if use_wandb:
        wandb.init(project=log_cfg["wandb_project"], name=run_name, config=cfg,
                   sync_tensorboard=True)
        callbacks.append(WandbCallback(verbose=2))

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    print(f"\n[INFO] Training for {t_cfg['total_timesteps']:,} steps")
    print(f"[INFO] Save path: {save_path}\n")

    model.learn(
        total_timesteps     = t_cfg["total_timesteps"],
        callback            = callbacks,
        reset_num_timesteps = args.resume is None,  # False when resuming
        tb_log_name         = run_name,
        progress_bar      = True,
    )

    # ------------------------------------------------------------------
    # Final save
    # ------------------------------------------------------------------
    model.save(os.path.join(save_path, "sac_priv_final"))
    print(f"\n[INFO] Final model saved → {save_path}/sac_priv_final.zip")
    print(f"[INFO] Best model (by eval) → {save_path}/best_model.zip")

    train_env.close()
    # eval_env.close()
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
