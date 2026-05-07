# Robot Manipulation Project

GitHub: [LeHoang510/Robocasa-RL-project](https://github.com/LeHoang510/Robocasa-RL-project)


Project Drive: <https://drive.google.com/drive/folders/1HE1ZfvGqZCAy_0qYc3lcxlwaDj3IEOvQ?usp=drive_link>

This repository contains our Robotique project on manipulation learning in RoboCasa.

The work is split into three main parts:
- `RoboCasa_Code_Aide/`: Phase 1 broad exploration on `PnPCounterToCabinet`
- `src/`: Phase 2 rewrite with more focused experiments on `PnPCounterToCabinet`
- `src_button/`: Phase 3 simpler task on `CoffeePressButton`

Useful documents:
- [Project Report](REPORT.md)
- [Phase 1 Implementation Notes](RoboCasa_Code_Aide/README.md)
- [Results README](results/README.md)

## Project Structure

```text
Project/
├── REPORT.md
├── README.md
├── RoboCasa_Code_Aide/
│   ├── config/
│   ├── my_env/
│   ├── my_rl_scripts/
│   └── README.md
├── src/
│   ├── config/
│   ├── env/
│   └── scripts/
├── src_button/
│   ├── config/
│   ├── env/
│   └── scripts/
├── robocasa/
└── robosuite/
```

## Installation

These steps assume you want everything in one local `uv` environment.

### 1. Create the environment

```bash
uv venv
source .venv/bin/activate
uv sync
```

### 2. Clone `robosuite`

If `robosuite/` is not already present:

```bash
git clone https://github.com/ARISE-Initiative/robosuite.git
```

Install it in editable mode with `uv`:

```bash
uv pip install -e ./robosuite --config-settings editable_mode=compat
```

### 3. Clone `robocasa`

If `robocasa/` is not already present:

```bash
git clone https://github.com/robocasa/robocasa.git
```

Install it in editable mode with `uv`:

```bash
uv pip install -e ./robocasa --config-settings editable_mode=compat
```

### 4. Notes

- Most training scripts assume you run commands from the project root.
- Several configs expect datasets under `robocasa/datasets/...`.

## How To Run Experiments

All commands below are meant to be launched from the project root.

### Phase 1: `RoboCasa_Code_Aide/`

#### Experiments

```bash
uv run python RoboCasa_Code_Aide/my_rl_scripts/train_ppo.py \
  --config RoboCasa_Code_Aide/config/exp1_ppo_baseline.yaml

uv run python RoboCasa_Code_Aide/my_rl_scripts/train_ppo.py \
  --config RoboCasa_Code_Aide/config/exp2_curriculum.yaml

uv run python RoboCasa_Code_Aide/my_rl_scripts/train_sac.py \
  --config RoboCasa_Code_Aide/config/exp3_sac_baseline.yaml

uv run python RoboCasa_Code_Aide/my_rl_scripts/train_sac.py \
  --config RoboCasa_Code_Aide/config/exp4_sac_her.yaml

uv run python RoboCasa_Code_Aide/my_rl_scripts/train_bc.py \
  --config RoboCasa_Code_Aide/config/exp5_bc.yaml

uv run python RoboCasa_Code_Aide/my_rl_scripts/train_diffusion.py \
  --config RoboCasa_Code_Aide/config/exp6_diffusion.yaml

uv run python RoboCasa_Code_Aide/my_rl_scripts/train_td3bc.py \
  --config RoboCasa_Code_Aide/config/exp7_td3bc.yaml

uv run python RoboCasa_Code_Aide/my_rl_scripts/train_iql.py \
  --config RoboCasa_Code_Aide/config/exp8_iql.yaml

uv run python RoboCasa_Code_Aide/my_rl_scripts/train_image_bc.py \
  --config RoboCasa_Code_Aide/config/exp9_image_bc.yaml

uv run python RoboCasa_Code_Aide/my_rl_scripts/train_sac_privileged.py \
  --config RoboCasa_Code_Aide/config/exp9_sac_privileged.yaml
```

#### Evaluation

```bash
uv run python RoboCasa_Code_Aide/my_rl_scripts/eval.py \
  --model_path checkpoints/<run>/final_model.zip \
  --episodes 10
```

For more Phase 1 details, see [RoboCasa_Code_Aide/README.md](RoboCasa_Code_Aide/README.md).

### Phase 2: `src/`

#### Diffusion 16D

```bash
uv run python src/scripts/train_diffusion.py \
  --config src/config/diffusion.yaml
```

#### Diffusion 25D privileged

```bash
uv run python src/scripts/train_diffusion.py \
  --config src/config/diffusion_privileged.yaml
```

#### SAC grasp stage

```bash
uv run python src/scripts/train_sac.py \
  --config src/config/sac_grasp.yaml
```

#### SAC full task

```bash
uv run python src/scripts/train_sac.py \
  --config src/config/sac_full.yaml \
  --init_from src/checkpoints/<grasp_run>/final
```

#### ACT

```bash
uv run python src/scripts/train_act.py \
  --config src/config/act.yaml
```

#### Phase 2 visualization

```bash
uv run python src/scripts/visualize.py \
  --type diffusion \
  --model src/checkpoints/<run>/diffusion_best.pt

uv run python src/scripts/visualize.py \
  --type sac \
  --model src/checkpoints/<run>/final

uv run python src/scripts/visualize.py \
  --type act \
  --model src/checkpoints/<run>/act_best.pt
```

### Phase 3: `src_button/`

#### Train SAC on CoffeePressButton

```bash
uv run python src_button/scripts/train_sac.py \
  --config src_button/config/sac_button.yaml
```

#### Evaluate

```bash
uv run python src_button/scripts/eval.py \
  --model src_button/checkpoints/<run>/best_model \
  --episodes 100
```

#### Visualize

```bash
uv run python src_button/scripts/visualize.py \
  --model src_button/checkpoints/<run>/best_model \
  --episodes 5
```

## Checkpoints and Outputs

- Phase 1 checkpoints are usually saved in `checkpoints/`
- Phase 2 checkpoints are usually saved in `src/checkpoints/`
- Phase 3 checkpoints are usually saved in `src_button/checkpoints/`
- Videos are saved in `eval_videos/`
- Only the grasp-apple demo and the CoffeePressButton demo were generated, because the other experiments all failed and making demos for them was unnecessary.

## Documentation Links

- Main report: [REPORT.md](REPORT.md)
- Phase 1 implementation details: [RoboCasa_Code_Aide/README.md](RoboCasa_Code_Aide/README.md)
- Results page: [results/README.md](results/README.md)
