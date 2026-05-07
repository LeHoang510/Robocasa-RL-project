# Robot Manipulation Learning - Project Report

Project Drive: <https://drive.google.com/drive/folders/1HE1ZfvGqZCAy_0qYc3lcxlwaDj3IEOvQ?usp=drive_link>

**Course:** Robotique - Telecom Paris S3  
**Task:** PnPCounterToCabinet (Pick apple from counter, place in cabinet) and CoffeePressButton  
**Robot:** PandaOmron (7-DOF arm + mobile base, 12D action space)  
**Simulator:** RoboCasa

---

## Overview

This project attempts to train a robot manipulator to perform an atomic task in the RoboCasa simulation environment. The primary task is **PnPCounterToCabinet**: pick up an apple from a kitchen counter and place it inside a cabinet - a two-stage manipulation problem requiring precision grasping and long-horizon planning. This task is challenging due to sparse rewards, the need for sequential subtask mastery (grasp then place), and the complexity of 3D object manipulation and didn't yield positive results so a secondary task, **CoffeePressButton**, was introduced in Phase 3 to explore a simpler manipulation problem.

The work progressed through three major phases:
1. **Phase 1 (RoboCasa_Code_Aide/)** - Broad exploration of 9 different algorithms (Experiments 1 -> 9)
2. **Phase 2 (src/)** - Focused implementation of the most promising approaches with cleaner code: Diffusion Policy, SAC with curriculum, and ACT with visual encoder
3. **Phase 3 (src_button/)** - choose another task (CoffeePressButton) 

**Overall result:** No method reliably solved the PnPCounterToCabinet task. The SAC grasp stage produced the first signs of success - occasional successful grasps in visualization (approximately 1/5 episodes). The CoffeePressButton task has a better result though, with a best checkpoint reaching 70% success during training and 54% success over 100 deterministic evaluation episodes.

---

## Task Description

### PnPCounterToCabinet
- **Robot:** PandaOmron (Panda arm on Omron mobile base)
- **Action space:** 12D continuous (7 arm joints + 1 gripper + 4 base movement)
- **Observation:** Proprioceptive + object state (varies per experiment: 16D, 25D)
- **Success:** Apple placed inside closed cabinet
- **Challenge:** Two sequential subtasks (grasp -> place), long horizon, 3D object manipulation, cabinet door interaction

###  CoffeePressButton
- **Action space:** Same 12D
- **Observation:** 16D (EEF pose + gripper + button position + relative vector + on/off flag)
- **Success:** Coffee machine turned on (button pressed) AND robot backed away
- **Why simpler:** No grasping required, button is fixed in space, single-stage task

---

## Phase 1: Broad Exploration - RoboCasa_Code_Aide/ (Experiments 1 -> 9)

At the start of the project, it was not clear which learning method would be the most suitable for this task. So the first phase was used as a broad exploration stage: several RL, imitation learning, and offline RL methods were tested on the same task. All experiments targeted PnPCounterToCabinet with a fixed kitchen layout so that comparisons would be easier.

### Experiment 1 - PPO Baseline

**Config:** `RoboCasa_Code_Aide/config/exp1_ppo_baseline.yaml`

| Parameter | Value |
|-----------|-------|
| Algorithm | PPO (Proximal Policy Optimization) |
| Obs | Default robocasa proprioceptive obs |
| Horizon | 500 steps |
| Total steps | 3,000,000 |
| Network | [256, 256] MLP |
| ent_coef | 0.005 |

**What was tried:** PPO with the first custom reward used in the project. This reward was still very basic: it gave a small reward for moving toward the apple, then larger rewards for grasping, moving to the cabinet, placing, and releasing. However, these rewards were mostly added independently, without strong conditions to make sure the robot learned the subtasks in the right order.

The reward used at this stage was:

```
Stage 1 - Reach:     0.02 x (1 - tanh(5d))
Stage 2 - Grasp:     +0.25 per step while grasped
Stage 3 - Transport: +0.20 x (1 - tanh(3d_cab))
Stage 4 - Place:     +2.00 per step while in cabinet
Stage 5 - Release:   +1.00 per step after release
```

This shows the main weakness of the first reward design: the robot got only a very small signal for approaching the apple, and there was no strong condition forcing it to learn the stages in the correct order.

**Why it failed:** The main issue was that the reward still did not give the robot the right guidance for grasping. The reach reward was very small, while later rewards were only useful after the robot had already discovered a good grasp by itself. So PPO spent many steps exploring but did not find a successful grasp often enough to build on it. Success rate stayed at 0.00.

**Why we moved on:** This suggested that the first reward design was too weak for this task, and that either a stronger RL setup or learning from demonstrations would be needed next.

---

### Experiment 2 - PPO + Curriculum Learning

**Config:** `RoboCasa_Code_Aide/config/exp2_curriculum.yaml`

| Parameter | Value |
|-----------|-------|
| Algorithm | PPO with 3-level curriculum |
| Curriculum | Difficulty 0-1-2 (apple starts closer, advances when success >20%) |
| Horizon | 500 steps |
| Total steps | 3,000,000 |

**What was tried:** The idea was to make the task easier at the beginning by placing the apple closer to the cabinet, then increase the difficulty once the robot showed some success. If this worked, PPO could first learn a simpler version and then transfer to the full task.

**Why it failed:** The curriculum never advanced past difficulty 0 because the robot still did not learn the first grasping stage reliably enough. So even though the setting was easier, the same weak-guidance problem from exp1 remained.

**Success rate:** 0.00

---

### Experiment 3 - SAC Baseline

**Config:** `RoboCasa_Code_Aide/config/exp3_sac_baseline.yaml`

| Parameter | Value |
|-----------|-------|
| Algorithm | SAC (Soft Actor-Critic) |
| Horizon | 500 steps |
| Total steps | 3,000,000 |
| Replay buffer | 1,000,000 |
| Network | [256, 256] MLP |
| ent_coef | auto |

**What was tried:** SAC is an off-policy algorithm, so it can reuse old experience from a replay buffer. This makes it more sample-efficient than PPO and usually a better choice for robotic control. The hope was that SAC would explore more effectively and learn grasping from the same task setup.

**Why it failed:** Even with SAC, the robot still had to learn reaching, grasping, moving, and placing all at once. The reward was denser than a pure sparse reward, but it still did not focus enough on the first hard step: grasping the apple. As a result, the robot did not reach stable success.

**Success rate:** 0.00

---

### Experiment 4 - SAC + HER

**Config:** `RoboCasa_Code_Aide/config/exp4_sac_her.yaml`

| Parameter | Value |
|-----------|-------|
| Algorithm | SAC + HER (Hindsight Experience Replay) |
| HER goals | n_sampled_goal=4, strategy="future" |
| Goal threshold | 0.10 meters |
| Total steps | 3,000,000 |

**What was tried:** HER was added to help with sparse success signals. The idea was that even failed episodes could still become useful training examples by changing the goal after the episode and treating the achieved state as a temporary target.

**Why it failed:** HER can help when the robot already produces somewhat useful trajectories. Here, the policy still struggled with the earlier part of the task, especially grasping. So the relabeled goals did not help much, because most episodes still did not contain useful object movement.

---

### Experiment 5 - Behavioral Cloning

**Config:** `RoboCasa_Code_Aide/config/exp5_bc.yaml`

| Parameter | Value |
|-----------|-------|
| Algorithm | BC (Behavioral Cloning) |
| Dataset | MimicGen demonstrations (LeRobot format) |
| Network | [64, 64] MLP (to avoid overfitting ~23k samples) |
| Epochs | 200 |
| Obs | 16D proprioceptive |

**What was tried:** Learn directly from demonstrations instead of relying only on trial and error. Behavioral cloning trains a policy to copy the action from the dataset for each observation.

**Why it failed:** The dataset only provided a limited 16D robot state, without direct object information. So during training, the policy learned from observations that did not fully describe where the apple and target were. At test time, this mismatch likely made the policy hard to use reliably on the real task setup.

Another possible problem was a mismatch between the demonstrations and the evaluation environment. Even if the action is correct for the original demo state, it may not match perfectly once the robot is rolled out in a fresh episode. This was only a suspicion, not something fully confirmed. Evaluation success was 0.00.

---

### Experiment 6 - Diffusion Policy

**Config:** `RoboCasa_Code_Aide/config/exp6_diffusion.yaml`

| Parameter | Value |
|-----------|-------|
| Algorithm | Diffusion Policy (DDPM + DDIM) |
| Dataset | MimicGen demos, LeRobot format |
| Network | [256, 256, 256] noise predictor |
| Diffusion steps | T=100 training, 10 DDIM steps inference |
| Epochs | 200 |

**What was tried:** Diffusion Policy is a stronger imitation learning method than plain BC. Instead of directly predicting one action, it learns a more flexible action distribution through a denoising process.

**Why it failed:** Although the model was more advanced than BC, it still learned from the same kind of demonstration data. So the possible problems were also similar: limited observations, and maybe a mismatch between what the demo represents and what the policy sees during rollout in a new episode. This was not fully verified. Success: 0.00.

---

### Experiment 7 — TD3+BC

**Config:** `RoboCasa_Code_Aide/config/exp7_td3bc.yaml`

**What was tried:** TD3+BC is an offline RL method that mixes reinforcement learning with behavioral cloning. The idea was that it might use the fixed dataset more effectively than pure imitation learning.

**Why it failed:** In practice, it still depended on the quality of the offline dataset. Since the observations were limited and may not match the evaluation setup well, the method still could not learn a useful policy. Success: 0.00.

---

### Experiment 8 - IQL

**Config:** `RoboCasa_Code_Aide/config/exp8_iql.yaml`

**What was tried:** IQL is a strong offline RL method that learns only from logged data and avoids some common offline RL instability problems. It was tested to see whether a better offline algorithm could extract more from the demonstrations.

**Why it failed:** IQL still depends on the dataset covering the important situations of the task. Here, the policy only had partial observations and the training data still did not transfer well to evaluation episodes. A possible reason was again some mismatch between the demo setting and the rollout setting. So even a stronger offline RL method was not enough. Success: 0.00.

---

### Experiment 9 - Image-Based BC / SAC-Privileged

**Config:** `RoboCasa_Code_Aide/config/exp9_image_bc.yaml`, `exp9_sac_privileged.yaml`

**What was tried:** Two final variants were explored in Phase 1:  
(1) image-based BC, to give the policy more visual information than proprioception alone, and  
(2) SAC with privileged observations, to give the RL agent direct access to the object and target state from the simulator.

**Why they failed:** Image BC did not solve the main transfer problem from the demonstration setting to evaluation. Even with images, the policy still had trouble turning the dataset into a working control policy.  
SAC-Privileged was more promising because it removed part of the observation problem, but it still did not solve grasping reliably within the available budget. This suggested that observations were only one part of the problem, and that reward design was also very important. Success: 0.00.

---

### Phase 1 Summary

After these 9 experiments, every method still had 0.00 success rate. Two main lessons came out of Phase 1:
1. **The first RL reward design was not good enough** - it was too basic and did not guide grasping clearly enough.
2. **The demonstration-based methods may have had a transfer problem** - the dataset setup and the evaluation setup may not have matched well enough, so copying the demonstrations did not lead to a working policy.

**Decision:** Instead of continuing to add more experiments on top of the same setup, the project moved to a cleaner second codebase (`src/`). The goal of Phase 2 was to rebuild the environment and training pipeline with better observations, better reward design, and more careful handling of demonstrations.

---

## Phase 2: Focused Experiments - src/

The `src/` folder was a clean rewrite focused on the main problems found in Phase 1:
- **Better observations** (including 25D privileged state) so the policy can see more of the task
- **Better reward shaping** with clearer conditions for grasp -> lift -> transport

### Experiment A - Diffusion Policy with 16D Obs

**Config:** `src/config/diffusion.yaml`

| Parameter | Value |
|-----------|-------|
| Algorithm | Diffusion Policy (DDPM + DDIM) |
| Obs | 16D proprioceptive (EEF pose 7D + gripper 2D + object relative 4D + goal 3D) |
| Dataset | Human demos (LeRobot) + MimicGen |
| obs_horizon | 2 (stack 2 frames for implicit velocity) |
| action_horizon | 8 |
| Network | 512 hidden dim, 4 blocks |
| Epochs | 200 (stopped at 100) |

**What was tried:** A cleaner re-implementation of Diffusion Policy with a better dataset loader. One bug from Phase 1 was fixed: the action format in the LeRobot files was read more carefully, and the action order was corrected before training.

**Training results (Wandb run `run-20260505_210256`):**

| Epoch | Train Loss | Val Loss | Eval Success |
|-------|-----------|----------|--------------|
| 10    | 0.198     | 0.201    | 0.00         |
| 50    | 0.193     | 0.196    | 0.00         |
| 100   | 0.190     | 0.193    | 0.00         |

Loss decreased steadily, which means the model was fitting the training data better. However, evaluation success stayed at 0.00 for all 100 epochs that were run.

**Why it still failed:** The cleaner implementation removed some obvious dataset-handling issues, but the policy still did not transfer to evaluation. One possible reason is that replaying demonstrations inside a fresh environment does not perfectly reproduce the original demo trajectory. After a few steps, the state seen during training may drift away from the state for which the original action was intended. This was not fully confirmed, but it remained a possible explanation for why the loss improved while success stayed at 0.00.

**Why 25D was next:** At this point there were two reasonable guesses: either the 16D observation was too weak, or there was some mismatch between the demonstrations and the rollout states. Since this was not clear, the next step was to try privileged 25D observations and see whether giving the model more exact state information would help.

---

### Experiment B - Diffusion Policy with 25D Privileged Obs

**Config:** `src/config/diffusion_privileged.yaml`

| Parameter | Value |
|-----------|-------|
| Algorithm | Diffusion Policy (DDPM + DDIM) |
| Obs | 25D: EEF pos(3)+quat(4)+gripper(2)+obj pos(3)+obj quat(4)+obj_to_eef(3)+target(3)+obj_to_target(3) |
| Dataset | 500 human + 500 MimicGen demos re-simulated through PrivilegedPnPEnv |
| Epochs | 300 |

**What was tried:** Re-simulate the whole dataset through the custom `PrivilegedPnPEnv` to extract more complete object information at every step. The idea was to give the model much more state information than the 16D setup.

**Training results (Wandb run `run-20260505_235920`):**

| Epoch | Train Loss | Val Loss | Eval Success |
|-------|-----------|----------|--------------|
| 10    | 0.247     | 0.249    | 0.00         |
| 100   | 0.198     | 0.200    | 0.00         |
| 200   | 0.192     | 0.194    | 0.00         |
| 300   | 0.185     | 0.187    | 0.00         |

The model trained for the full 300 epochs, but evaluation success still stayed at 0.00.

**Why it still failed:** The extra state information did not help. One possible explanation is that it made the suspected mismatch easier to notice: the observation could show the apple in one place, while the action came from a demo that was meant for a slightly different situation. But this remained a hypothesis, not a fully proven root cause.

**Conclusion on imitation learning:** Better architectures and better observations were not enough. A possible explanation was that the demonstration pipeline still had some mismatch between training data and evaluation rollout, but this was not fully confirmed. Because of time limits, this issue was not fully resolved in either Phase 1 or Phase 2, and the project returned to RL with a stronger environment design.

---

### Experiment C - SAC with Curriculum (Grasp Stage)

**Config:** `src/config/sac_grasp.yaml`

| Parameter | Value |
|-----------|-------|
| Algorithm | SAC (Soft Actor-Critic) |
| Environment | GraspEnv (stage 1 only) |
| Obs | 16D: EEF pos(3)+quat(4)+gripper(2)+obj pos(3)+obj_to_eef(3) |
| Horizon | 150 steps |
| Total steps | 500,000 |
| Network | [256, 256, 256] |
| ent_coef | "auto" |
| Reward | r_reach + r_grip + r_grasp (detailed below) |

**Design rationale:** At this point, imitation learning did not look promising anymore. So the next step was to return to pure RL, but with a more focused design:
- **Stage 1 (GraspEnv):** Learn only to grasp the apple. Short horizon (150 steps), dense shaping.
- **Stage 2 (PrivilegedPnPEnv):** Learn full pick-and-place, initialized from stage 1 weights.

The reward for stage 1 was designed iteratively through several bug-fixing cycles:

```
r_reach = 1.0 - tanh(3.0 × ||eef_to_apple||)    → dense approach signal
r_grip  = r_reach × grip_closed × 2.0            → reward closing fingers near apple  
r_grasp = 10.0 × is_grasped                      → strong sparse bonus on real contact
```

**Key issues discovered and fixed during this experiment:**

**Issue 1 - reward for being near the apple was misleading:** An earlier version included `r_contact = exp(-d)` to reward being close to the apple. But this value was already quite high even when the robot was still far away. This likely pushed the robot to stay at medium distance instead of really approaching the apple. The fix was to remove this term and rely on `r_reach`, which gave a clearer signal.

**Issue 2 - the first grasp check was too approximate:** The initial grasp detection used a simple distance and gripper-threshold rule. This did not always match real contact in the simulator. So the robot could sometimes get grasp reward without actually grasping the apple. This was fixed by using a real contact-based grasp check.

**Issue 3 - exploration stayed too random for too long:** The original config used `ent_coef=0.3` as a fixed value. In practice, this seemed to keep the policy too random for too long in a 12D action space. It was changed to `ent_coef="auto"` so SAC could tune it automatically.

**Training results (first run, `run-20260506_003958`, before all fixes):**

| Step | Eval Success | r_reach (avg) | r_grasp (avg) | Dist to Apple (avg) |
|------|-------------|---------------|----------------|---------------------|
| 10k  | 0.00        | 0.15          | 0.001          | 0.32m               |
| 50k  | 0.00        | 0.18          | 0.002          | 0.28m               |
| 100k | 0.00        | 0.21          | 0.003          | 0.25m               |
| 200k | 0.00        | 0.25          | 0.005          | 0.22m               |
| 300k | 0.00        | 0.24          | 0.004          | 0.23m               |

The reach reward improved slightly, which means the robot was getting closer to the apple, but progress stopped around 0.22m. The grasp reward never became strong enough. So this first version was stopped at 300k steps.

**After applying all three fixes** (ent_coef=auto, check_obj_grasped, removed r_contact, extended to 500k steps):

During visualization of the saved checkpoint, **the robot successfully grasped the apple in 1 out of 5 evaluation episodes.** This was not fully consistent, but it was still an important result: for the first time, the robot showed that it had learned part of the real behavior needed for the task.

**Why stage 2 was not reached:** The grasp success rate was still too low to justify moving to the full pick-and-place stage. Stage 2 would require lifting, transporting, and placing, so it only made sense if grasping had first become much more reliable.

---

### Experiment D - ACT with ResNet18 Visual Encoder

**Config:** `src/config/act.yaml`

| Parameter | Value |
|-----------|-------|
| Algorithm | ACT (Action Chunking with Transformers) |
| Visual encoder | ResNet18 (pretrained ImageNet, layer4 features) |
| Cameras | robot0_agentview_right + robot0_eye_in_hand |
| Image size | 84×84 |
| Obs | 25D proprioceptive + spatial image tokens |
| Chunk size | 40 steps |
| KL weight | 1e-3 |
| Dataset | 150 human + 150 MimicGen demos |
| Epochs | 300 (stopped at 50) |

**What was tried:** ACT is a vision-based imitation learning method designed for manipulation tasks. It predicts a chunk of future actions at once instead of one action at a time. The hope was that this would make the behavior more stable and better suited to long manipulation sequences.

Key design choices:
- **Keep spatial image information:** Instead of reducing the image to one global feature vector, the model keeps a small spatial grid of features so it can preserve where objects appear.
- **Predict action chunks:** The model predicts several future actions together, which may help produce smoother behavior.
- **Use chunk_size=40:** This was chosen as a balance between short-term precision and longer motion planning.
- **Use a small KL weight:** This keeps the latent part of the model useful without letting it dominate training.

**Implementation challenge — feature map size:**  
The first implementation assumed the ResNet output would be a 2×2 grid, but in practice it was 3×3. This caused a tensor size mismatch at runtime. The fix was to run a dummy forward pass at initialization and measure the real output shape directly.

**Memory challenge — image dataset too large:**  
Loading all images in a naive way used too much memory. This was fixed with a two-pass method: first count how much space is needed, then pre-allocate the full array and fill it directly. This reduced memory use a lot.

**Training results (Wandb run `run-20260506_100612`):**

| Epoch | Recon Loss | KL Loss | Val Loss | Eval Success |
|-------|-----------|---------|----------|--------------|
| 5     | 0.052     | 2.61    | 0.055    | —            |
| 10    | 0.041     | 2.58    | 0.044    | —            |
| 25    | 0.028     | 2.55    | 0.032    | —            |
| 50    | 0.017     | 2.52    | 0.021    | 0.00         |

Training loss decreased cleanly and the latent part of the model stayed active. But at epoch 50, evaluation success was still 0.00.

**Why it still failed:** ACT was a stronger imitation learning method, but it still depended on the same demonstration pipeline. So one possible problem was again a mismatch between the demonstration data and the rollout states seen at evaluation time. This was not fully proven, but it remained one reasonable explanation. Even with images, the model still could not turn the dataset into a reliable policy.

**Stopped at epoch 50:** With 0% success after 50 epochs, it did not seem reasonable to continue all the way to 300 epochs. The model was learning the dataset better, but not solving the task.

---

### Phase 2 Summary

| Experiment | Algorithm | Obs | Steps/Epochs | Best Eval Success | Stopped Early? |
|-----------|-----------|-----|--------------|-------------------|----------------|
| A | Diffusion Policy (16D) | 16D prop | 100 epochs | 0.00 | Yes (plateau) |
| B | Diffusion Policy (25D) | 25D privileged | 300 epochs | 0.00 | No (ran to end) |
| C | SAC Grasp Curriculum | 16D prop | 500k steps | ~0.20 (visualization)* | No |
| D | ACT + ResNet18 | 25D + images | 50 epochs | 0.00 | Yes (plateau) |


The SAC grasp stage was the only method to show any positive signal, with the robot successfully grasping the apple in approximately 1 out of 5 visualization episodes.

---

## Why Nothing Reliably Worked

### Root Cause 1: Possible Demo-to-Rollout Mismatch (Imitation Learning)

The imitation learning experiments looked promising at first because training loss often went down. But lower training loss did not translate into task success.

One possible reason is a **mismatch between the demonstrations and the environment used during training or evaluation**. In some parts of the pipeline, the actions from the demos had to be replayed or reinterpreted inside a fresh environment. After a few steps, the state seen by the model may no longer match the exact state for which the original demo action was produced.

This would mean the model can fit the dataset numerically while still failing to act correctly in a new episode. In simple words, the policy may be learning from examples that are close to correct, but not correct enough for precise manipulation. However, this remained a hypothesis in this project, not a fully proven explanation.

### Root Cause 2: Task Complexity and Reward Design for RL

PnPCounterToCabinet is a difficult task because the robot must first grasp the apple, then lift it, move it, and place it correctly. In the first phase, the reward was still too basic and did not guide these stages clearly enough. In the second phase, the reward became better, and this is why SAC finally started to show occasional grasp success.

Even then, the robot still struggled to reach a reliable grasping policy. This suggests that the task may require even more careful reward shaping, or more training time, to reach consistent success. The gap between the best training success (~20% in visualization) and the final evaluation (0%) also suggests that the policy was still not robust enough.

---

## Phase 3: CoffeePressButton — src_button/

After all PnP approaches failed to give reliable success, the project moved to a simpler task: pressing the button on a coffee machine. This task removes grasping completely, so the robot only needs to reach, press, and move away.

**Task design:**
- **Obs (16D):** EEF pos(3) + EEF quat(4) + gripper(2) + button_pos(3) + eef_to_button(3) + turned_on(1)
- **Reward:**
  ```
  r_reach   = 1.0 - tanh(3.0 × ||eef_to_button||)   dense approach
  r_press   = 8.0 × pressed_this_step                one-time bonus on button activation
  r_retreat = turned_on × tanh(5.0 × max(d - 0.05, 0))   encourages backing away after press
  r_success = 30.0 × success                         terminal bonus
  ```
- **Why this works better:** no grasping is required, the button position is fixed, and the task is much more direct

**Training results (1M steps, EvalCallback every 10k steps, 20 deterministic episodes per eval):**

| Step | Eval Success | Notes |
|------|-------------|-------|
| 10k  | 0.05 (1/20) | First positive result in the entire project |
| 50k  | ~0.20       | Robot consistently approaching button |
| ~peak | **0.70**   | Best checkpoint saved as `best_model.zip` |

**Final evaluation of best model (100 episodes, deterministic policy, headless):**

| Metric | Value |
|--------|-------|
| Success rate | **54% (54/100)** |
| Avg reward | 130.90 ± 47.11 |
| Avg steps to success | 139.3 ± 64.9 |

The gap between the training-time best (70%) and the final evaluation (54%) is expected: the training evaluation used only 20 episodes, while the final 100-episode evaluation gives a more reliable estimate. Even so, this was the first task in the project with clearly positive and repeatable results.

---

## Results Summary Table

### PnPCounterToCabinet - All Experiments

| Phase | Exp | Algorithm | Obs Dim | Training Budget | Eval Success | 
|-------|-----|-----------|---------|-----------------|--------------|
| 1 | 1 | PPO | Default | 3M steps | 0.00 |
| 1 | 2 | PPO + Curriculum | Default | 3M steps | 0.00 | 
| 1 | 3 | SAC | Default | 3M steps | 0.00 |
| 1 | 4 | SAC + HER | Default | 3M steps | 0.00 |
| 1 | 5 | BC | 16D | 200 epochs | 0.00 |
| 1 | 6 | Diffusion Policy | 16D | 200 epochs | 0.00 |
| 1 | 7 | TD3+BC | 16D | — | 0.00 |
| 1 | 8 | IQL | 16D | — | 0.00 |
| 1 | 9a | Image BC | Image + prop | — | 0.00 |
| 1 | 9b | SAC Privileged | 25D | — | 0.00 |
| 2 | A | Diffusion Policy | 16D | 100 epochs | 0.00 |
| 2 | B | Diffusion Policy | 25D | 300 epochs | 0.00 |
| **2** | **C** | **SAC Grasp** | **16D** | **500k steps** | **~0.20** |
| 2 | D | ACT + ResNet18 | 25D + img | 50 epochs | 0.00 |

### CoffeePressButton

| Phase | Algorithm | Obs Dim | Budget | Best Training Eval | Final Eval (100 ep) | Avg Steps |
|-------|-----------|---------|--------|--------------------|---------------------|-----------|
| **3** | **SAC** | **16D** | **1M steps** | **70%** | **54% (54/100)** | **139 ± 65** |

**SAC Grasp note:** Deterministic evaluation (greedy policy) showed 0.00 throughout training. However, during visualization with exploration noise enabled, the robot successfully grasped the apple in approximately 1 out of 5 episodes. A video recording of this is available. This indicates the policy had learned directional approach behavior but lacked the precision to consistently close the gripper on the apple without the noise for exploration assistance.

--- 

## Conclusion

This project explored a wide range of approaches for robot manipulation learning in simulation. The core finding is that **the PnPCounterToCabinet task is genuinely difficult**, and several bugs and design mistakes compounded to prevent progress:

- All imitation learning methods failed, and one possible explanation was a mismatch between the demo data and the rollout setting. This suspicion appeared in both Phase 1 and Phase 2, but it was not fully confirmed.
- RL methods suffered from poor reward shaping, incorrect grasp detection, and insufficient exploration in the early phase.
- When train grasp independently, the robot began to discover grasping behavior - showing approximately 20% success in visualization evaluation.

The project did not produce a fully working policy for the primary task. However, it produced:
- A clearer understanding of the possible reasons why imitation learning failed, especially the suspicion that replayed or re-used demo data may not match rollout states well enough
- A working reward design for SAC grasping with proper contact detection
- A simpler task (CoffeePressButton) showing the first positive evaluation results in the project (5% at step 10k)

Only two demos were kept: the grasp-apple demo from the partial SAC grasp success, and the CoffeePressButton demo from the final working task. The other experiments all failed, so generating demo videos for them was unnecessary.

The button-pressing task is the only experiment in this project to produce a genuinely working policy: **54% success rate over 100 deterministic evaluation episodes**, with the best checkpoint reaching 70% during training. This confirms that the SAC reward design and privileged observation approach are effective for learning this simpler manipulation task, even if the more complex pick-and-place task remains unsolved.
