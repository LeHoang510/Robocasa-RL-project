# Report: PandaOmron Counter-To-Cabinet Pick And Place

## Task

The selected RoboCasa atomic task is a pick-and-place task with the `PandaOmron`
robot: pick an apple from the counter and place it inside an open cabinet. The
custom environment is `MyPnPCounterToCab`, derived from RoboCasa's
`PickPlaceCounterToCabinet`.

To reduce unnecessary variance for reinforcement learning, the kitchen layout
and style are fixed, the target object is always `apple_1`, and the distractor
object is always `bowl_1`. Object placement still varies inside the configured
sampling regions, so the policy must learn the manipulation behavior rather than
memorizing a single exact state.

## Method

The method is reinforcement learning with PPO from Stable-Baselines3 and dense
reward shaping.

The original RoboCasa task uses a sparse success reward, which makes exploration
hard because the robot must accidentally complete a long sequence before seeing
positive feedback. The custom reward decomposes the task into shaped terms:

- Reach reward: distance between end effector and apple.
- Grasp reward: contact/proximity-based grasp detection.
- Lift reward: encourages moving the apple upward after grasping.
- Place reward: distance between apple and the cabinet interior.
- Inside reward: apple center inside the cabinet.
- Release reward: apple inside the cabinet and gripper far from the object.
- Success reward: `10.0` when the original full success condition is satisfied.

The PPO policy uses proprioceptive and object-state observations through
robosuite's `GymWrapper`. Camera observations are used for visualization and
evaluation videos, not for the default PPO policy, which keeps training lighter.

## Reproducibility

Training command:

```bash
python RoboCasa_Code/rl_scripts/train_robocasa.py \
  --total_timesteps 200000 \
  --n_envs 1 \
  --seed 42 \
  --device cuda \
  --output_dir RoboCasa_Code/runs/pnp_counter_to_cab_final
```

Evaluation command:

```bash
python RoboCasa_Code/rl_scripts/eval_robocasa.py \
  --model_path RoboCasa_Code/runs/pnp_counter_to_cab_final/ppo_robocasa_final.zip \
  --episodes 20 \
  --seed 100 \
  --device cpu \
  --save_video \
  --video_path RoboCasa_Code/eval_videos
```

The training script saves:

- final model: `ppo_robocasa_final.zip`
- checkpoints: `checkpoints/`
- training metrics: `training_curve.csv`
- plot: `training_curve.png`
- TensorBoard logs: `tensorboard/`

## Difficulties And Solutions

Sparse rewards were the main difficulty. A random policy almost never places the
object in the cabinet and releases it correctly, so PPO receives little useful
signal. Reward shaping solves this by rewarding intermediate progress while
preserving the original task success condition.

Scene variability was another difficulty. Full RoboCasa layout/style diversity
is useful for generalization but expensive for a small course project. The
solution is to fix layout/style and object identities, while still retaining
randomized object placement.

Rendering and dependency setup can also be fragile. The scripts separate
training from visualization: training runs headless with state observations,
while evaluation can enable offscreen rendering only when videos are requested.

## Training Curve

The curve is generated automatically by `train_robocasa.py` at:

```text
RoboCasa_Code/runs/pnp_counter_to_cab_final/training_curve.png
```

The CSV source is:

```text
RoboCasa_Code/runs/pnp_counter_to_cab_final/training_curve.csv
```

## Visualization

Task visualization can be produced in two ways:

1. `visualize_custom_env.py` for interactive random-policy inspection.
2. `eval_robocasa.py --save_video` for multi-camera videos of the trained policy.

The evaluation videos are saved as MP4 files under:

```text
RoboCasa_Code/eval_videos/<model_name>/
```
