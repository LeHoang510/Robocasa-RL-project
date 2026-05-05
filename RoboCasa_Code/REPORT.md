# Report: PandaOmron Microwave Button Press

## Task

The selected RoboCasa atomic task is `MicrowavePressButton` with the
`PandaOmron` robot. The goal is to press the microwave start button, turn the
microwave on, and move the gripper away from the button.

The custom environment is `MyMicrowavePressButton`, derived from RoboCasa's
`MicrowavePressButton`. The layout and style are fixed to reduce variance for a
small reinforcement-learning project.

## Method

The method is reinforcement learning with PPO from Stable-Baselines3 and dense
reward shaping.

The original task success condition is sparse: the microwave must be turned on
and the gripper must be far from the start button. The custom reward decomposes
this into easier subgoals:

- Reach reward: distance between end effector and the microwave start button.
- Contact reward: gripper touches the start button.
- Press reward: microwave state becomes `turned_on`.
- Release reward: microwave is on and the gripper moves away from the button.
- Success reward: `10.0` when the original success condition is satisfied.

The PPO policy uses state observations through robosuite's `GymWrapper`. Camera
observations are used only for visualization and optional evaluation videos.

## Reproducibility

Training command:

```bash
python RoboCasa_Code/rl_scripts/train_robocasa.py \
  --total_timesteps 100000 \
  --n_envs 1 \
  --seed 42 \
  --device cpu \
  --output_dir RoboCasa_Code/runs/microwave_press_button
```

Evaluation command:

```bash
python RoboCasa_Code/rl_scripts/eval_robocasa.py \
  --model_path RoboCasa_Code/runs/microwave_press_button/ppo_robocasa_final.zip \
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

The previous counter-to-cabinet pick-and-place task was too difficult for short
PPO runs because it required grasping, lifting, placing, and releasing. The
microwave button task has a shorter action sequence and much denser feedback,
making it more appropriate for fast experimentation.

The main remaining challenge is exploration: the gripper must still discover
contact with a small button. Dense reach and contact rewards make that behavior
much easier to learn than a sparse success-only signal.

## Training Curve

The curve is generated automatically by `train_robocasa.py` at:

```text
RoboCasa_Code/runs/microwave_press_button/training_curve.png
```

The CSV source is:

```text
RoboCasa_Code/runs/microwave_press_button/training_curve.csv
```
