# RoboCasa Code Aide — Implementation Notes

## Overview

This folder contains two parallel implementations:

| Folder | Status | Purpose |
|--------|--------|---------|
| `env/` + `rl_scripts/` | Original (skeleton) | Provided as starter code |
| `my_env/` + `my_rl_scripts/` | **Complete implementation** | Our working code |

The task is to train a **PandaOmron** robot to pick an apple from a kitchen counter and place it inside a cabinet (`PnPCounterToCab`), comparing two training methods:

| Experiment | Method | Config |
|-----------|--------|--------|
| exp1 | PPO + reward shaping (baseline) | `config/exp1_ppo_baseline.yaml` |
| exp2 | PPO + reward shaping + curriculum learning | `config/exp2_curriculum.yaml` |

---

## Quick Start

```bash
# Exp 1 — PPO baseline
python RoboCasa_Code_Aide/my_rl_scripts/train.py \
    --config RoboCasa_Code_Aide/config/exp1_ppo_baseline.yaml

# Exp 2 — Curriculum learning
python RoboCasa_Code_Aide/my_rl_scripts/train.py \
    --config RoboCasa_Code_Aide/config/exp2_curriculum.yaml

# Evaluate a trained model (10 episodes, save video)
python RoboCasa_Code_Aide/my_rl_scripts/eval.py \
    --model_path checkpoints/<run>/final_model.zip \
    --episodes 10 --save_video

# Compare both methods
python RoboCasa_Code_Aide/my_rl_scripts/eval_all.py \
    --models checkpoints/exp1_.../final_model.zip \
             checkpoints/exp2_.../final_model.zip \
    --labels "PPO Baseline" "Curriculum" \
    --n_episodes 20 \
    --output results/comparison.png
```

Any YAML value can be overridden from the CLI:

```bash
python RoboCasa_Code_Aide/my_rl_scripts/train.py \
    --config RoboCasa_Code_Aide/config/exp1_ppo_baseline.yaml \
    --total_timesteps 2000000 --seed 123
```

---

## `my_env/pnp_env.py` — Custom Environment

### What was added to the original

The original `env/custom_pnp_counter_to_cab.py` had a placeholder reward that always returned 0:

```python
def reward(self, action=None):
    r = 0
    return r
```

Two additions were made, each clearly marked with `# ── NEW ──` comments:

1. **Dense shaped reward** — 5-stage additive reward signal.
2. **Curriculum difficulty system** — `DIFFICULTY_PLACEMENTS`, `self.difficulty`, `set_difficulty()`.

All other original code (layout fix, fixed objects, deterministic fixture placement) is unchanged.

### Reward Design

The reward breaks the task into 5 sequential sub-goals. All stages are **additive** — the agent always receives a gradient signal toward the next goal even before it masters the current one.

```
Stage 1 – Reach      : gripper moves toward the apple       max ~0.15 / step
Stage 2 – Grasp      : gripper closes on the apple          +0.25 / step
Stage 3 – Transport  : carry the apple toward the cabinet   max ~0.30 / step
Stage 4 – Place      : apple lands inside the cabinet       +1.00
Stage 5 – Release    : gripper backs away after placing     +0.50
                                                  total max ~  2.20
```

#### Stage 1 — Reach

```python
dist_to_obj = np.linalg.norm(eef_pos - obj_pos)
r += 0.15 * (1.0 - np.tanh(5.0 * dist_to_obj))
```

Smoothly decays with distance; active only while the apple is not yet grasped.

#### Stage 2 — Grasp bonus

```python
is_grasped = OU.check_obj_grasped(self, "obj")
if is_grasped:
    r += 0.25
```

Awarded every step the apple is held, sustaining motivation to keep holding it.

#### Stage 3 — Transport

```python
dist_to_cab = OU.obj_fixture_bbox_min_dist(self, "obj", self.cab)
r += 0.30 * (1.0 - np.tanh(3.0 * dist_to_cab))
```

Active while the apple is grasped.

#### Stage 4 — Placement

```python
obj_in_cab = OU.obj_inside_of(self, "obj", self.cab)
if obj_in_cab:
    r += 1.00
```

Large reward makes placement the dominant objective.

#### Stage 5 — Release

```python
released = OU.gripper_obj_far(self, "obj", th=0.15)
if obj_in_cab and released:
    r += 0.50
```

Prevents the robot from holding the apple inside the cabinet indefinitely.

### Curriculum Difficulty Levels

| Level | Description | Placement size | Apple position |
|-------|-------------|----------------|----------------|
| 0 (easy) | Apple always close to cabinet | (0.15, 0.10) | (0.0, −0.9) |
| 1 (medium) | Medium region | (0.35, 0.20) | (0.0, −0.5) |
| 2 (hard) | Full original region | (0.60, 0.30) | (0.0, −1.0) |

`CurriculumCallback` in `train.py` calls `env.set_attr("difficulty", level)` when the rolling success rate exceeds `success_threshold`. The change takes effect at the next episode reset.

---

## `my_rl_scripts/train.py` — Training Script

Loads a YAML config and trains PPO, optionally with curriculum learning.

### PPO Hyperparameters

| Parameter | Value | Reason |
|-----------|-------|--------|
| `learning_rate` | 3e-4 | Standard Adam default for PPO |
| `n_steps` | 1024 | ~2 full episodes per env before each update |
| `batch_size` | 64 | Mini-batch for gradient steps |
| `n_epochs` | 10 | Gradient steps per rollout |
| `gamma` | 0.99 | Long-horizon discount |
| `gae_lambda` | 0.95 | GAE smoothing |
| `clip_range` | 0.2 | PPO clip parameter |
| `ent_coef` | 0.005 | Small entropy bonus to discourage premature convergence |
| `net_arch` | [256, 256] | 2 hidden layers sufficient for proprioceptive obs |

### Callbacks

| Callback | Purpose |
|----------|---------|
| `CheckpointCallback` | Saves model every 10 000 steps to `checkpoints/<run>/ckpts/` |
| `SuccessRateCallback` | Logs rolling `rollout/success_rate` (window=100) to TensorBoard/WandB |
| `CurriculumCallback` | (exp2 only) Advances difficulty when success rate ≥ threshold |
| `WandbCallback` | (optional) Syncs gradients and model weights to Weights & Biases |

### Logging

TensorBoard logs are always written to `checkpoints/<run>/tb_logs/`.

```bash
tensorboard --logdir checkpoints/
```

WandB is enabled by setting `use_wandb: true` in the YAML config. Training curves are synced via `sync_tensorboard=True` and appear in the **Charts** tab of the WandB run.

### Known issues / design notes

- `SubprocVecEnv` causes `robot_model = None` in MuJoCo subprocesses → always use `DummyVecEnv`.
- `env.reset()` must be called before `GymWrapper(env)` to initialise `robot_model`.
- `n_envs: 1` in all configs for stability.

---

## `my_rl_scripts/eval.py` — Evaluation Script

```bash
# Evaluate 10 episodes, print success rate + mean reward
python RoboCasa_Code_Aide/my_rl_scripts/eval.py \
    --model_path checkpoints/<run>/final_model.zip \
    --episodes 10

# Same but also save tiled 4-camera video
python RoboCasa_Code_Aide/my_rl_scripts/eval.py \
    --model_path checkpoints/<run>/final_model.zip \
    --episodes 10 --save_video
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_path` | required | Path to `.zip` checkpoint |
| `--episodes` | 5 | Number of evaluation episodes |
| `--horizon` | 500 | Max steps per episode |
| `--seed` | 42 | Evaluation seed |
| `--save_video` | False | Render and save tiled 4-camera MP4 |
| `--video_dir` | `eval_videos/` | Output directory for videos |

Videos are saved as `eval_videos/<model_name>/ep_000.mp4`, etc., with 4 camera views tiled 2×2.

---

## `my_rl_scripts/eval_all.py` — Multi-model Comparison

Runs evaluation for multiple trained models and produces a bar chart comparing success rate and mean reward.

```bash
python RoboCasa_Code_Aide/my_rl_scripts/eval_all.py \
    --models checkpoints/exp1_.../final_model.zip \
             checkpoints/exp2_.../final_model.zip \
    --labels "PPO Baseline" "Curriculum" \
    --n_episodes 20 \
    --output results/comparison.png
```

Output: a printed table + `comparison.png` with side-by-side bars for each method.

---

## Typical Full Workflow

```
1. Train exp1 (baseline PPO)
   python RoboCasa_Code_Aide/my_rl_scripts/train.py \
       --config RoboCasa_Code_Aide/config/exp1_ppo_baseline.yaml

2. Train exp2 (curriculum)
   python RoboCasa_Code_Aide/my_rl_scripts/train.py \
       --config RoboCasa_Code_Aide/config/exp2_curriculum.yaml

3. Monitor training curves
   tensorboard --logdir checkpoints/
   # or view in WandB Charts tab (if use_wandb: true in config)

4. Evaluate each model
   python RoboCasa_Code_Aide/my_rl_scripts/eval.py \
       --model_path checkpoints/<run>/final_model.zip \
       --episodes 20 --save_video

5. Compare both methods
   python RoboCasa_Code_Aide/my_rl_scripts/eval_all.py \
       --models checkpoints/exp1_.../final_model.zip \
                checkpoints/exp2_.../final_model.zip \
       --labels "PPO Baseline" "Curriculum" \
       --output results/comparison.png
```
