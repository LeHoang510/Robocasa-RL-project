# RoboCasa PandaOmron Atomic Task Project

This project solves the RoboCasa atomic task `PickPlaceCounterToCabinet` with the
`PandaOmron` robot. The submitted task variant is `MyPnPCounterToCab`, a custom
environment with fixed layout/style, fixed target object (`apple_1`), a fixed
distractor (`bowl_1`), and dense reward shaping for PPO.

## Files

- `env/custom_pnp_counter_to_cab.py`: custom RoboCasa environment and shaped reward.
- `rl_scripts/train_robocasa.py`: PPO training script that saves checkpoints, final model, CSV logs, and a training curve.
- `rl_scripts/eval_robocasa.py`: evaluation script with success-rate reporting and optional multi-camera videos.
- `visualize_custom_env.py`: random-policy visualizer for checking the environment and cameras.
- `REPORT.md`: method, difficulties, and expected results discussion.

## Environment

Use the Python 3.12 environment where RoboCasa and robosuite were installed:

```bash
cd "/media/vlod08/Windows_Data/Work/Telecom Paris/IA705/Project"
export PYTHONPATH="$PWD/robocasa:$PWD/robosuite:$PWD/RoboCasa_Code:$PYTHONPATH"
pip install stable-baselines3 gymnasium matplotlib imageio termcolor
```

The assignment versions are already present in this workspace:

- RoboCasa commit: `0f59111531118148b3d9383ab2e3e28f66d324fd`
- robosuite commit: `aaa8b9b214ce8e77e82926d677b4d61d55e577ab`

## Train

The default `--device cpu` is the safest option for PPO with an MLP policy. To
use an RTX 50-series GPU, install a PyTorch wheel with CUDA 12.8 support:

```bash
python -m pip install --upgrade torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
  --index-url https://download.pytorch.org/whl/cu128
```

Then verify CUDA from the same terminal:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Quick smoke run:

```bash
python RoboCasa_Code/rl_scripts/train_robocasa.py \
  --total_timesteps 1000 \
  --n_envs 1 \
  --seed 42 \
  --device cpu \
  --output_dir RoboCasa_Code/runs/smoke
```

Main run:

```bash
python RoboCasa_Code/rl_scripts/train_robocasa.py \
  --total_timesteps 200000 \
  --n_envs 1 \
  --seed 42 \
  --device cuda \
  --output_dir RoboCasa_Code/runs/pnp_counter_to_cab_final
```

Outputs:

- `RoboCasa_Code/runs/pnp_counter_to_cab_final/ppo_robocasa_final.zip`
- `RoboCasa_Code/runs/pnp_counter_to_cab_final/checkpoints/*.zip`
- `RoboCasa_Code/runs/pnp_counter_to_cab_final/training_curve.csv`
- `RoboCasa_Code/runs/pnp_counter_to_cab_final/training_curve.png`
- `RoboCasa_Code/runs/pnp_counter_to_cab_final/tensorboard/`

## Evaluate

```bash
python RoboCasa_Code/rl_scripts/eval_robocasa.py \
  --model_path RoboCasa_Code/runs/pnp_counter_to_cab_final/ppo_robocasa_final.zip \
  --episodes 20 \
  --device cpu \
  --seed 100
```

To save visualization videos:

```bash
python RoboCasa_Code/rl_scripts/eval_robocasa.py \
  --model_path RoboCasa_Code/runs/pnp_counter_to_cab_final/ppo_robocasa_final.zip \
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

## Package For Submission

After training, create the zip from the project root:

```bash
zip -r robocasa_submission.zip RoboCasa_Code \
  -x "RoboCasa_Code/**/__pycache__/*" \
  -x "RoboCasa_Code/**/.DS_Store"
```
