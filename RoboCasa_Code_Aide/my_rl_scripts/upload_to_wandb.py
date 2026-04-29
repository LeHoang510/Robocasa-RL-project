"""
Upload existing TensorBoard logs to WandB.
==========================================
Use this when a training run finished without WandB enabled.
Reads the TB event file and replays all logged metrics into a new WandB run.

Usage
-----
python RoboCasa_Code_Aide/my_rl_scripts/upload_to_wandb.py \
    --tb_dir  checkpoints/exp1_ppo_baseline_20260413_204039/tb_logs/PPO_1 \
    --project robocasa-pnp \
    --name    exp1_ppo_baseline
"""

import argparse
import wandb
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def upload(tb_dir: str, project: str, name: str):
    print(f"[INFO] Reading TensorBoard logs from: {tb_dir}")
    ea = EventAccumulator(tb_dir)
    ea.Reload()

    scalar_tags = ea.Tags().get("scalars", [])
    if not scalar_tags:
        print("[ERROR] No scalar metrics found in the event file.")
        return

    print(f"[INFO] Found metrics: {scalar_tags}")

    # Collect all (step, tag, value) triples
    all_events: dict[int, dict] = {}
    for tag in scalar_tags:
        for event in ea.Scalars(tag):
            step = event.step
            if step not in all_events:
                all_events[step] = {}
            all_events[step][tag] = event.value

    print(f"[INFO] Uploading {len(all_events)} steps to WandB …")

    run = wandb.init(project=project, name=name, resume="never")

    for step in sorted(all_events.keys()):
        wandb.log(all_events[step], step=step)

    run.finish()
    print(f"[INFO] Done — view at {run.url}")


def main():
    parser = argparse.ArgumentParser(description="Upload TB logs to WandB")
    parser.add_argument("--tb_dir",  type=str, required=True,
                        help="Path to the TB event directory (contains events.out.tfevents.*)")
    parser.add_argument("--project", type=str, default="robocasa-pnp")
    parser.add_argument("--name",    type=str, required=True,
                        help="Run name to display in WandB")
    args = parser.parse_args()

    upload(args.tb_dir, args.project, args.name)


if __name__ == "__main__":
    main()
