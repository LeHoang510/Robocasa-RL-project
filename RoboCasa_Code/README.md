# RoboCasa PandaOmron Microwave Button Project

This project solves the RoboCasa atomic task `MicrowavePressButton` with the
`PandaOmron` robot. The submitted task variant is `MyMicrowavePressButton`, a
custom environment with fixed layout/style and dense reward shaping for PPO.

## Files

- `env/custom_microwave_press_button.py`: custom RoboCasa microwave button task.
- `env/custom_pnp_counter_to_cab.py`: older pick-and-place attempt, kept for reference.
- `rl_scripts/train_robocasa.py`: PPO training script that saves checkpoints, final model, CSV logs, and a training curve.
- `rl_scripts/eval_robocasa.py`: evaluation script with success-rate reporting and optional multi-camera videos.
- `visualize_custom_env.py`: random-policy visualizer for checking the environment and cameras.
- `REPORT.md`: method, difficulties, and expected results discussion.

## Environment

Use the Python environment where RoboCasa and robosuite are installed:

```bash
cd "/d/Work/Telecom Paris/IA705/Project"
export PYTHONPATH="$PWD/robocasa:$PWD/robosuite:$PWD/RoboCasa_Code:$PYTHONPATH"
pip install stable-baselines3 gymnasium matplotlib imageio termcolor
```

The default `--device cpu` is usually best for PPO with an MLP policy because
MuJoCo simulation is CPU-bound.

## Train

Quick smoke run:

```bash
python RoboCasa_Code/rl_scripts/train_robocasa.py \
  --total_timesteps 1000 \
  --n_envs 1 \
  --seed 42 \
  --device cpu \
  --output_dir RoboCasa_Code/runs/microwave_smoke
```

Main run:

```bash
python RoboCasa_Code/rl_scripts/train_robocasa.py \
  --total_timesteps 100000 \
  --n_envs 1 \
  --seed 42 \
  --device cpu \
  --output_dir RoboCasa_Code/runs/microwave_press_button
```

Outputs:

- `RoboCasa_Code/runs/microwave_press_button/ppo_robocasa_final.zip`
- `RoboCasa_Code/runs/microwave_press_button/checkpoints/*.zip`
- `RoboCasa_Code/runs/microwave_press_button/training_curve.csv`
- `RoboCasa_Code/runs/microwave_press_button/training_curve.png`
- `RoboCasa_Code/runs/microwave_press_button/tensorboard/`

## Evaluate

```bash
python RoboCasa_Code/rl_scripts/eval_robocasa.py \
  --model_path RoboCasa_Code/runs/microwave_press_button/ppo_robocasa_final.zip \
  --episodes 20 \
  --device cpu \
  --seed 100
```

To save visualization videos:

```bash
python RoboCasa_Code/rl_scripts/eval_robocasa.py \
  --model_path RoboCasa_Code/runs/microwave_press_button/ppo_robocasa_final.zip \
  --episodes 5 \
  --seed 100 \
  --device cpu \
  --save_video \
  --video_path RoboCasa_Code/eval_videos
```

## Visualize The Task

```bash
python RoboCasa_Code/visualize_custom_env.py
```

This opens the simulator plus a matplotlib camera grid and runs random actions.
