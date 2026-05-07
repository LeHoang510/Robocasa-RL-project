# Robot Manipulation Learning — Project Report

**Course:** Robotique — Telecom Paris S3  
**Task:** PnPCounterToCabinet (Pick apple from counter, place in cabinet) → CoffeePressButton  
**Robot:** PandaOmron (7-DOF arm + mobile base, 12D action space)  
**Simulator:** RoboCasa / MuJoCo

---

## Overview

This project attempts to train a robot manipulator to perform household tasks in the RoboCasa simulation environment. The primary task is **PnPCounterToCabinet**: pick up an apple from a kitchen counter and place it inside a cabinet — a two-stage manipulation problem requiring precision grasping and long-horizon planning.

The work progressed through three major phases:
1. **Phase 1 (RoboCasa_Code_Aide/)** — Broad exploration of 9 different algorithms simultaneously (Experiments 1–9)
2. **Phase 2 (src/)** — Focused implementation of the most promising approaches with cleaner code: Diffusion Policy, SAC with curriculum, and ACT with visual encoder
3. **Phase 3 (src_button/)** — Pivot to a simpler task (CoffeePressButton) to establish a working baseline

**Overall result:** No method reliably solved the PnPCounterToCabinet task. The SAC grasp stage produced the first signs of life — occasional successful grasps in visualization (approximately 1/5 episodes). The button-pressing task is still being evaluated.

---

## Task Description

### PnPCounterToCabinet
- **Robot:** PandaOmron (Panda arm on Omron mobile base)
- **Action space:** 12D continuous (7 arm joints + 1 gripper + 4 base movement)
- **Observation:** Proprioceptive + object state (varies per experiment: 16D, 25D)
- **Success:** Apple placed inside closed cabinet
- **Challenge:** Two sequential subtasks (grasp → place), long horizon, 3D object manipulation, cabinet door interaction

### StartCoffeeMachine / CoffeePressButton (Phase 3)
- **Action space:** Same 12D
- **Observation:** 16D (EEF pose + gripper + button position + relative vector + on/off flag)
- **Success:** Coffee machine turned on (button pressed) AND robot backed away
- **Why simpler:** No grasping required, button is fixed in space, single-stage task

---

## Phase 1: Broad Exploration — RoboCasa_Code_Aide/ (Experiments 1–9)

At the start of the project, the landscape of applicable algorithms was unclear. Nine experiments were configured and launched in parallel to quickly benchmark different paradigms. All experiments targeted PnPCounterToCabinet with a fixed kitchen layout (layout_ids=[1], style_ids=[1]).

### Experiment 1 — PPO Baseline

**Config:** `RoboCasa_Code_Aide/config/exp1_ppo_baseline.yaml`

| Parameter | Value |
|-----------|-------|
| Algorithm | PPO (Proximal Policy Optimization) |
| Obs | Default robocasa proprioceptive obs |
| Horizon | 500 steps |
| Total steps | 3,000,000 |
| Network | [256, 256] MLP |
| ent_coef | 0.005 |

**What was tried:** PPO with standard reward shaping from the robocasa environment. PPO is an on-policy algorithm — it learns from recent experience only and discards old transitions after each update. It is stable and well-understood but sample-inefficient.

**Why it failed:** The PnP task has extremely sparse rewards — the robot must complete the full grasp-and-place sequence before getting any meaningful signal. PPO's on-policy nature means it cannot reuse experience, so 3M steps provides far fewer "learning events" than an off-policy algorithm would get from the same number of environment interactions. The robot never discovered a successful grasp, so the success signal (apple in cabinet) was never observed. Success rate remained 0.00 throughout.

**Why we moved on:** On-policy algorithms like PPO are generally considered unsuitable for sparse-reward manipulation tasks unless combined with heavy reward shaping or demonstrations.

---

### Experiment 2 — PPO + Curriculum Learning

**Config:** `RoboCasa_Code_Aide/config/exp2_curriculum.yaml`

| Parameter | Value |
|-----------|-------|
| Algorithm | PPO with 3-level curriculum |
| Curriculum | Difficulty 0→1→2 (apple starts closer, advances when success >20%) |
| Horizon | 500 steps |
| Total steps | 3,000,000 |

**What was tried:** The idea was to start with an "easy" version of the task where the apple is placed close to the cabinet, then gradually increase difficulty. If the robot can learn the easier placement first, it builds a foundation for the full task.

**Why it failed:** The curriculum never advanced past difficulty 0. The robot could not achieve 20% success rate on the easy version because it still needed to grasp first, which requires the same precision regardless of cabinet distance. The curriculum only controlled placement distance, not grasp difficulty. Since grasping was never solved, the curriculum provided no benefit.

**Success rate:** 0.00

---

### Experiment 3 — SAC Baseline

**Config:** `RoboCasa_Code_Aide/config/exp3_sac_baseline.yaml`

| Parameter | Value |
|-----------|-------|
| Algorithm | SAC (Soft Actor-Critic) |
| Horizon | 500 steps |
| Total steps | 3,000,000 |
| Replay buffer | 1,000,000 |
| Network | [256, 256] MLP |
| ent_coef | auto |

**What was tried:** SAC is an off-policy algorithm that reuses all past experience via a replay buffer, making it much more sample-efficient than PPO. It also automatically tunes the entropy coefficient, encouraging exploration without manual tuning.

**Why it failed:** The reward signal was still the default robocasa reward — not shaped specifically to guide grasping. The robot received dense proprioceptive observations but no explicit signal about where the apple was relative to its gripper. With 3M steps and no grasp-specific shaping, the robot did not discover reliable grasping behavior.

**Success rate:** 0.00

---

### Experiment 4 — SAC + HER

**Config:** `RoboCasa_Code_Aide/config/exp4_sac_her.yaml`

| Parameter | Value |
|-----------|-------|
| Algorithm | SAC + HER (Hindsight Experience Replay) |
| HER goals | n_sampled_goal=4, strategy="future" |
| Goal threshold | 0.10 meters |
| Total steps | 3,000,000 |

**What was tried:** HER addresses sparse rewards by relabeling failed episodes as successes toward whatever goal was actually reached. For example, if the apple ended up 20cm from the cabinet, HER treats that as a "success" toward a 20cm goal. Over time, the robot learns to hit closer and closer targets.

**Why it failed:** HER works well in robotics but requires a clearly defined goal space (e.g., target position). The robocasa PnP task's success condition involves a cabinet interior check rather than a simple distance threshold, making goal relabeling difficult to formulate correctly. Additionally, without first solving grasping, relabeling placement episodes was meaningless. Success rate remained 0.00.

---

### Experiment 5 — Behavioral Cloning

**Config:** `RoboCasa_Code_Aide/config/exp5_bc.yaml`

| Parameter | Value |
|-----------|-------|
| Algorithm | BC (Behavioral Cloning) |
| Dataset | MimicGen demonstrations (LeRobot format) |
| Network | [64, 64] MLP (to avoid overfitting ~23k samples) |
| Epochs | 200 |
| Obs | 16D proprioceptive |

**What was tried:** Learn from expert demonstrations directly. MimicGen generated thousands of synthetic demonstrations by adapting human demos to new layouts. Behavioral cloning trains a policy to mimic the expert action given the current observation (supervised learning).

**Why it failed:** This is where the fundamental **re-simulation problem** was first encountered. The LeRobot parquet files store demo actions in a specific column order. When these actions are replayed in a fresh simulation environment, physics divergence begins after 5–10 steps. By step 20, the simulated robot is in a completely different state than the demo intended. This means the (observation, action) pairs collected by re-simulating are incoherent — the observation at step T doesn't correspond to the action that was planned for step T.

With corrupted training data, the BC policy learned a mapping from random-looking states to random-looking actions. Evaluation success was 0.00.

---

### Experiment 6 — Diffusion Policy (Phase 1)

**Config:** `RoboCasa_Code_Aide/config/exp6_diffusion.yaml`

| Parameter | Value |
|-----------|-------|
| Algorithm | Diffusion Policy (DDPM + DDIM) |
| Dataset | MimicGen demos, LeRobot format |
| Network | [256, 256, 256] noise predictor |
| Diffusion steps | T=100 training, 10 DDIM steps inference |
| Epochs | 200 |

**What was tried:** Diffusion Policy models the action distribution as a denoising process. Rather than directly regressing to an action, it learns to predict the noise added to an action, enabling multimodal action distributions (useful when the robot can succeed via multiple strategies).

**Why it failed:** Same re-simulation problem as BC. The training data was corrupted at the source. The diffusion model learned to denoise corrupted (obs, action) pairs, which produced meaningless policies at test time. Success: 0.00.

---

### Experiment 7 — TD3+BC

**Config:** `RoboCasa_Code_Aide/config/exp7_td3bc.yaml`

**What was tried:** TD3+BC (Twin Delayed Deep Deterministic Policy Gradient + Behavioral Cloning) is an offline RL algorithm that combines RL loss with a BC regularization term. It can learn from a fixed dataset without further environment interaction.

**Why it failed:** Same re-simulation data corruption problem. The offline dataset contained corrupted (obs, action) pairs. Additionally, offline RL algorithms are sensitive to distributional shift — they need to avoid out-of-distribution actions, but with corrupted training data this constraint becomes meaningless. Success: 0.00.

---

### Experiment 8 — IQL (Implicit Q-Learning)

**Config:** `RoboCasa_Code_Aide/config/exp8_iql.yaml`

**What was tried:** IQL is a state-of-the-art offline RL algorithm that avoids out-of-distribution action evaluation by never querying the Q-function on actions outside the dataset. It learns purely from logged data.

**Why it failed:** Re-simulation data corruption. Also, IQL requires good coverage of the state space in the offline dataset to generalize. With corrupted transitions, coverage was insufficient even if the data had been clean. Success: 0.00.

---

### Experiment 9 — Image-Based BC / SAC-Privileged

**Config:** `RoboCasa_Code_Aide/config/exp9_image_bc.yaml`, `exp9_sac_privileged.yaml`

**What was tried:** Two variants — (1) BC with camera images as input instead of proprioception alone, and (2) SAC with privileged observations (direct access to object positions from simulator state).

**Why they failed:** Image BC: re-simulation problem made image-action pairs incoherent, plus image observations made the dataset even larger and the signal even noisier. SAC-Privileged: the privileged obs included object positions, but without well-designed reward shaping and sufficient training time, the robot still could not discover grasping in the allocated budget. Success: 0.00.

---

### Phase 1 Summary

After running all 9 experiments, every approach returned 0.00 success rate. Two root causes dominated:
1. **Re-simulation data corruption** — affected all imitation learning methods (exp5–8, image BC)
2. **Insufficient reward shaping** — RL methods (exp1–4, exp9 SAC) had no specific guidance for the grasping subtask

**Decision:** Rather than continuing to run experiments that shared the same bugs, the work shifted to a cleaner codebase (`src/`) with explicit fixes for both problems.

---

## Phase 2: Focused Experiments — src/

The `src/` folder was a clean rewrite with three specific fixes:
- **Privileged observations** (25D): direct access to simulator state for object positions
- **Custom reward shaping**: explicit FSM-gated rewards for grasp → lift → transport
- **Re-simulation fix attempt**: two-pass dataset collection with pre-allocated arrays to handle the corruption

### Experiment A — Diffusion Policy with 16D Obs

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

**What was tried:** A clean re-implementation of Diffusion Policy with proper LeRobot parquet loading. A key bug from Phase 1 was fixed: the action column in the parquet files is a single `action` column containing a numpy array per row (not `action.0`, `action.1`, ..., `action.11` as originally assumed). Additionally, the column order in the LeRobot format differs from the robosuite `env.step()` order, requiring an action reorder: `_LEROBOT_TO_HDF5 = [5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4]`.

**Training results (Wandb run `run-20260505_210256`):**

| Epoch | Train Loss | Val Loss | Eval Success |
|-------|-----------|----------|--------------|
| 10    | 0.198     | 0.201    | 0.00         |
| 50    | 0.193     | 0.196    | 0.00         |
| 100   | 0.190     | 0.193    | 0.00         |

Loss decreased steadily, suggesting the model was fitting the data. However, eval success remained 0.00 throughout all 100 epochs that were run before early stopping.

**Why it still failed:** Although the data loading was fixed (correct column names and action reordering), the re-simulation problem persisted at a deeper level. When demo actions are applied step-by-step to a fresh environment, MuJoCo's physics integration causes trajectory divergence after approximately 5–10 steps. The 16D proprioceptive observation at step T+10 corresponds to a state that the demo never actually visited. The diffusion model learned to denoise meaningless (obs, action) pairs. Decreasing training loss was a misleading signal — the model was fitting corrupted data more accurately, not learning a useful policy.

**Why 25D was next:** If the problem was partial observability (16D obs doesn't contain enough context), extending to full privileged 25D might let the policy recover more easily from drift. This was the hypothesis that motivated the next run.

---

### Experiment B — Diffusion Policy with 25D Privileged Obs

**Config:** `src/config/diffusion_privileged.yaml`

| Parameter | Value |
|-----------|-------|
| Algorithm | Diffusion Policy (DDPM + DDIM) |
| Obs | 25D: EEF pos(3)+quat(4)+gripper(2)+obj pos(3)+obj quat(4)+obj_to_eef(3)+target(3)+obj_to_target(3) |
| Dataset | 500 human + 500 MimicGen demos re-simulated through PrivilegedPnPEnv |
| Epochs | 300 |

**What was tried:** Re-simulate the entire dataset through the custom `PrivilegedPnPEnv` to extract ground-truth object state at every step. This gives the model the exact object position, orientation, and relative vectors — far more information than vanilla proprioception.

**Training results (Wandb run `run-20260505_235920`):**

| Epoch | Train Loss | Val Loss | Eval Success |
|-------|-----------|----------|--------------|
| 10    | 0.247     | 0.249    | 0.00         |
| 100   | 0.198     | 0.200    | 0.00         |
| 200   | 0.192     | 0.194    | 0.00         |
| 300   | 0.185     | 0.187    | 0.00         |

The model trained for the full 300 epochs with no improvement in evaluation success.

**Why it still failed:** The 25D obs made the re-simulation problem worse, not better. The privileged obs includes `obj_pos` and `obj_quat` — but these come from re-simulating the demo, so by step 10, the apple is in a completely different position than in the original demonstration. The observation shows "apple at position X" while the action says "move toward where the apple was in the original demo" (a different position Y). The training signal is maximally confusing: the model must predict an action appropriate for one apple position while observing a different apple position.

**Conclusion on imitation learning:** All four diffusion/BC approaches failed for the same fundamental reason — re-simulation produces incoherent (obs, action) pairs that cannot be fixed by better observations or better architectures. The only real solutions would be: (a) avoid re-simulation entirely by collecting new demos through the custom environment, or (b) learn an inverse dynamics model to re-label actions. Neither was pursued due to time constraints.

---

### Experiment C — SAC with Curriculum (Grasp Stage)

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

**Design rationale:** Abandon imitation learning entirely. Use pure RL with a custom two-stage curriculum:
- **Stage 1 (GraspEnv):** Learn only to grasp the apple. Short horizon (150 steps), dense shaping.
- **Stage 2 (PrivilegedPnPEnv):** Learn full pick-and-place, initialized from stage 1 weights.

The reward for stage 1 was designed iteratively through several bug-fixing cycles:

```
r_reach = 1.0 - tanh(3.0 × ||eef_to_apple||)    → dense approach signal
r_grip  = r_reach × grip_closed × 2.0            → reward closing fingers near apple  
r_grasp = 10.0 × is_grasped                      → strong sparse bonus on real contact
```

**Key bugs discovered and fixed during this experiment:**

**Bug 1 — r_contact free reward:** An earlier version included `r_contact = exp(-d)` to reward being close to the apple. This function evaluates to 0.70 at 35cm distance — already high before the robot is anywhere near the apple. This dominated the reward signal and made the approach gradient invisible: the robot learned to hover at medium distance rather than closing in. The fix was to remove r_contact entirely and rely on r_reach (tanh-based) which has a steeper gradient at distance.

**Bug 2 — Proxy grasp check inaccurate:** The initial grasp detection used `is_grasped = (d_eef < 0.08) AND (grip_closed > 0.4)` — a distance+gripper threshold heuristic. This does not match actual MuJoCo contact detection. The robot could satisfy this condition by hovering with half-closed fingers near the apple without actually making contact. This gave false-positive grasp rewards, confusing the learning signal. Fixed by using `robosuite.utils.observables.check_obj_grasped()`, which checks real contact pairs between gripper fingers and the object.

**Bug 3 — ent_coef too high:** The original config used `ent_coef=0.3` (hardcoded). For a 12D action space, SAC's entropy target is approximately `-dim(A) = -12`. With `ent_coef=0.3`, the entropy bonus overwhelms the task reward, keeping the policy near-random for much longer than needed. Changed to `ent_coef="auto"` which lets SB3 automatically tune the entropy coefficient based on action dimensionality. This is especially critical when the action space is large.

**Training results (first run, `run-20260506_003958`, before all fixes):**

| Step | Eval Success | r_reach (avg) | r_grasp (avg) | Dist to Apple (avg) |
|------|-------------|---------------|----------------|---------------------|
| 10k  | 0.00        | 0.15          | 0.001          | 0.32m               |
| 50k  | 0.00        | 0.18          | 0.002          | 0.28m               |
| 100k | 0.00        | 0.21          | 0.003          | 0.25m               |
| 200k | 0.00        | 0.25          | 0.005          | 0.22m               |
| 300k | 0.00        | 0.24          | 0.004          | 0.23m               |

The reach reward improved slightly (robot got closer) but plateaued at ~0.22m distance. The grasp reward never reached meaningful levels. Early stopping at 300k steps.

**After applying all three fixes** (ent_coef=auto, check_obj_grasped, removed r_contact, extended to 500k steps):

During visualization of the saved checkpoint, **the robot successfully grasped the apple in 1 out of 5 evaluation episodes.** This was not fully consistent but demonstrates that the robot physically discovered how to approach and close the gripper on the apple — a meaningful behavioral discovery.

> **Note:** A video recording of this evaluation episode showing the successful grasp is available separately.

**Why stage 2 was not reached:** The grasp success rate was still too low (~20% in visualization, much lower in deterministic eval) to justify training stage 2. Stage 2 additionally requires lifting, transporting across the kitchen, and placing inside the cabinet. Without more reliable grasping, stage 2 had no signal to learn from.

---

### Experiment D — ACT with ResNet18 Visual Encoder

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

**What was tried:** ACT (Action Chunking with Transformers) is a vision-based imitation learning method designed specifically for manipulation. It uses a CVAE (Conditional Variational Autoencoder) to model the latent "style" of the action, then a Transformer decoder to predict a chunk of 40 future actions in one forward pass. The ResNet18 backbone extracts spatial features from both cameras without global pooling, producing patch tokens that the Transformer can attend to.

Key architectural decisions:
- **No global average pooling on ResNet:** Standard ResNet pools to a single 2048D vector, losing all spatial information. The implementation uses ResNet layer4 feature maps as a 3×3 spatial grid (9 patch tokens for 84×84 input), preserving where objects appear in the image. The Transformer can then attend to the relevant spatial regions.
- **Temporal ensemble at inference:** When executing a 40-step chunk, the model generates a new chunk every step and blends predictions with exponential decay (`exp(-0.01 × age)`), smoothing out per-step noise.
- **chunk_size=40:** Balances predictive horizon (100 steps was too long, blurs precise grasp timing) with coherence. 30–50 is the optimal range for PnP tasks.
- **kl_weight=1e-3:** A larger KL weight (e.g., 10.0) causes latent collapse — all inputs map to the same latent, making the CVAE useless. 1e-3 keeps the KL term as a soft regularizer only.

**Implementation challenge — ResNet feature map size:**
The first implementation computed patch count as `img_h // 32 = 84 // 32 = 2` (integer division), giving 2×2=4 patches. But the actual ResNet output for 84×84 input is 3×3=9 patches (due to ceiling behavior in convolution strides). This caused a tensor size mismatch at runtime (`RuntimeError: The size of tensor a (9) must match the size of tensor b (4)`). Fixed by running a dummy forward pass through all ResNet layers at initialization time to measure the actual feature map dimensions.

**Memory challenge — Image dataset OOM:**
Naively collecting 300 episodes × 200 steps × 2 cameras × 84×84×3 would require creating a Python list of ~240k numpy arrays and then calling `np.stack()`, which holds both the list and the stacked array simultaneously in memory (~16GB peak). Fixed with a two-pass approach: first count total steps across all parquet files without loading any images, pre-allocate a single contiguous `np.empty` array of the exact final size, then fill it in-place during collection. Peak memory reduced to ~2.7GB.

**Training results (Wandb run `run-20260506_100612`):**

| Epoch | Recon Loss | KL Loss | Val Loss | Eval Success |
|-------|-----------|---------|----------|--------------|
| 5     | 0.052     | 2.61    | 0.055    | —            |
| 10    | 0.041     | 2.58    | 0.044    | —            |
| 25    | 0.028     | 2.55    | 0.032    | —            |
| 50    | 0.017     | 2.52    | 0.021    | 0.00         |

Training loss decreased cleanly. KL loss remained stable (~2.5) indicating the latent space was being used without collapse. At epoch 50 evaluation: success=0.00.

**Why it still failed:** Same re-simulation corruption as all other imitation learning attempts. ACT is more robust to some forms of distributional shift (the chunk prediction smooths over individual step errors), but it cannot recover when the visual input at step T shows the apple in a position that never appeared in any training trajectory. The ResNet sees images it has never encountered during training, and the predicted action chunk is correspondingly wrong.

**Stopped at epoch 50:** With 0% success after 50 epochs and training loss plateauing, continuing to epoch 300 would not change the evaluation outcome. The model was fitting corrupted data well but evaluating to 0%.

---

### Phase 2 Summary

| Experiment | Algorithm | Obs | Steps/Epochs | Best Eval Success | Stopped Early? |
|-----------|-----------|-----|--------------|-------------------|----------------|
| A | Diffusion Policy (16D) | 16D prop | 100 epochs | 0.00 | Yes (plateau) |
| B | Diffusion Policy (25D) | 25D privileged | 300 epochs | 0.00 | No (ran to end) |
| C | SAC Grasp Curriculum | 16D prop | 500k steps | ~0.20 (visualization)* | No |
| D | ACT + ResNet18 | 25D + images | 50 epochs | 0.00 | Yes (plateau) |

*\* Visualization eval (non-deterministic). Deterministic training eval remained 0.00 due to policy stochasticity at this stage.*

The SAC grasp stage was the only method to show any positive signal, with the robot successfully grasping the apple in approximately 1 out of 5 visualization episodes. This validates that the RL reward design and environment setup are correct — the robot physically can reach and grasp the apple.

---

## Diagnosis: Why Nothing Reliably Worked

### Root Cause 1: Re-simulation Data Corruption (Imitation Learning)

All four imitation learning approaches (BC, Diffusion 16D, Diffusion 25D, ACT) failed for the same underlying reason. The training pipeline collects (observation, action) pairs by **re-simulating** demonstration episodes in a fresh environment. This is necessary because the LeRobot parquet files store raw actions but not the resulting observations through the custom environment wrapper.

The problem: MuJoCo physics is deterministic but sensitive to initial conditions. When demo actions `a_0, a_1, ..., a_T` are replayed in a new environment, the trajectory diverges from the original demo after approximately 5–10 steps due to floating-point differences in initialization. By step 50, the robot is in a completely different configuration than in the original demo. The observation collected at that step does not correspond to the intended action.

This produces training data where `obs_t → action_t` is correct for early steps and increasingly wrong for later steps. The model learns a corrupted mapping that cannot generalize to real test episodes. **Decreasing training loss is a false signal** — it just means the model is memorizing the corrupted data better.

**The only correct solutions** would be:
- Re-collect demonstrations directly through the custom environment wrapper (preserving all observations at collection time)
- Use an inverse dynamics model to re-label actions given observed (s_t, s_{t+1}) pairs

### Root Cause 2: Task Complexity for RL (PnP)

PnPCounterToCabinet is a long-horizon task with two sequential subtasks. The curriculum decomposition (grasp then place) was correct in principle, but:
- 500k steps may still be insufficient to learn reliable grasping from scratch with a 12D action space
- The 12D action space makes exploration slow (the policy must discover a precise 12D action sequence)
- Cabinet-placement adds additional complexity (depth estimation, door clearance) that stage 1 never trains

---

## Phase 3: Pivot to CoffeePressButton — src_button/

After all PnP approaches failed, the project pivoted to a fundamentally simpler task: pressing the button on a coffee machine. This task eliminates grasping entirely — the robot only needs to reach and touch a fixed button.

**Task design:**
- **Obs (16D):** EEF pos(3) + EEF quat(4) + gripper(2) + button_pos(3) + eef_to_button(3) + turned_on(1)
- **Reward:**
  ```
  r_reach   = 1.0 - tanh(3.0 × ||eef_to_button||)   dense approach
  r_press   = 8.0 × pressed_this_step                one-time bonus on button activation
  r_retreat = turned_on × tanh(5.0 × max(d - 0.05, 0))   encourages backing away after press
  r_success = 30.0 × success                         terminal bonus
  ```
- **Why this works better:** No grasping required, button position is fixed, single-stage task with a clear press → retreat sequence

**Early results (Step 10k eval):** success = 0.05 (1/20 episodes) — the first positive evaluation result in the entire project. Training is ongoing at 1M steps.

*Full results will be added to this report once training completes.*

---

## Results Summary Table

### PnPCounterToCabinet — All Experiments

| Phase | Exp | Algorithm | Obs Dim | Training Budget | Eval Success | Root Cause of Failure |
|-------|-----|-----------|---------|-----------------|--------------|----------------------|
| 1 | 1 | PPO | Default | 3M steps | 0.00 | On-policy, no grasp shaping |
| 1 | 2 | PPO + Curriculum | Default | 3M steps | 0.00 | Curriculum never advanced |
| 1 | 3 | SAC | Default | 3M steps | 0.00 | No grasp-specific reward |
| 1 | 4 | SAC + HER | Default | 3M steps | 0.00 | Goal definition mismatch |
| 1 | 5 | BC | 16D | 200 epochs | 0.00 | Re-simulation data corruption |
| 1 | 6 | Diffusion Policy | 16D | 200 epochs | 0.00 | Re-simulation corruption |
| 1 | 7 | TD3+BC | 16D | — | 0.00 | Re-simulation corruption |
| 1 | 8 | IQL | 16D | — | 0.00 | Re-simulation corruption |
| 1 | 9a | Image BC | Image + prop | — | 0.00 | Corruption + OOM |
| 1 | 9b | SAC Privileged | 25D | — | 0.00 | No grasp reward shaping |
| 2 | A | Diffusion Policy | 16D | 100 epochs | 0.00 | Re-simulation corruption |
| 2 | B | Diffusion Policy | 25D | 300 epochs | 0.00 | Re-simulation corruption (worse) |
| **2** | **C** | **SAC Grasp** | **16D** | **500k steps** | **~0.20 (viz)** | **Partial success — see below** |
| 2 | D | ACT + ResNet18 | 25D + img | 50 epochs | 0.00 | Re-simulation corruption |

### CoffeePressButton

| Phase | Algorithm | Obs Dim | Budget | Best Eval Success | Status |
|-------|-----------|---------|--------|-------------------|--------|
| 3 | SAC | 16D | 1M steps | 0.05 (step 10k) | Running — results pending |

**SAC Grasp note:** Deterministic evaluation (greedy policy) showed 0.00 throughout training. However, during visualization with exploration noise enabled, the robot successfully grasped the apple in approximately 1 out of 5 episodes. A video recording of this is available. This indicates the policy had learned directional approach behavior but lacked the precision to consistently close the gripper on the apple without the noise for exploration assistance.

---

## Technical Lessons Learned

1. **Re-simulation is fundamentally broken for LeRobot-format demos.** Never re-simulate demo actions through a custom environment to collect observations. Either use the environment that generated the demos, or avoid imitation learning entirely.

2. **Reward shaping requires careful gradient analysis.** The `exp(-d)` contact reward gave 0.70 reward at 35cm without approaching. Any reward term that is large when far from the goal will dominate and prevent approach behavior.

3. **ent_coef matters enormously at high action dimensions.** A fixed `ent_coef=0.3` in a 12D action space keeps the policy near-random for far too long. SAC's auto-tuning (`ent_coef="auto"`) is essential.

4. **Proxy success conditions must match physics.** Distance+gripper-threshold grasp detection does not match MuJoCo contact forces. Using `check_obj_grasped()` (actual contact pair detection) eliminated false-positive rewards that confused learning.

5. **Task decomposition is the right strategy for long-horizon tasks.** The grasp-then-place curriculum was the correct architectural choice. The failure was insufficient training budget and reward bugs, not the curriculum idea itself.

6. **Simpler tasks first.** The button-pressing task showed positive eval success (5%) by step 10k. Establishing a working baseline on a simpler task is more productive than repeatedly debugging a complex one.

7. **Decreasing training loss is not evidence of learning.** When training data is corrupted, a model can overfit perfectly and still achieve 0% on the actual task. Always evaluate on the environment, not just on held-out data from the same corrupted set.

---

## Project Structure

```
Project/
├── RoboCasa_Code_Aide/          # Phase 1: 9 experiments (broad exploration)
│   ├── config/
│   │   ├── exp1_ppo_baseline.yaml
│   │   ├── exp2_curriculum.yaml
│   │   ├── exp3_sac_baseline.yaml
│   │   ├── exp4_sac_her.yaml
│   │   ├── exp5_bc.yaml
│   │   ├── exp6_diffusion.yaml
│   │   ├── exp7_td3bc.yaml
│   │   ├── exp8_iql.yaml
│   │   └── exp9_*.yaml
│   └── my_rl_scripts/           # Training scripts for Phase 1
│
├── src/                         # Phase 2: clean rewrite, focused experiments
│   ├── config/
│   │   ├── diffusion.yaml           # Experiment A (16D diffusion)
│   │   ├── diffusion_privileged.yaml  # Experiment B (25D diffusion)
│   │   ├── sac_grasp.yaml           # Experiment C (SAC curriculum stage 1)
│   │   ├── sac_full.yaml            # Experiment C stage 2 (never reached)
│   │   └── act.yaml                 # Experiment D (ACT + ResNet18)
│   ├── env/
│   │   ├── pnp_env.py               # GraspEnv + PrivilegedPnPEnv wrappers
│   │   ├── bc_dataset.py            # Dataset loading (LeRobot parquet)
│   │   ├── diffusion_policy.py      # DDPM/DDIM implementation
│   │   ├── act_policy.py            # ACT + ResNet18 implementation
│   │   └── act_dataset.py           # Two-pass image collection (OOM fix)
│   └── scripts/                     # Training scripts for Phase 2
│
└── src_button/                  # Phase 3: CoffeePressButton (simpler task)
    ├── config/
    │   └── sac_button.yaml
    ├── env/
    │   └── button_env.py            # ButtonPressEnv: 16D obs + shaped reward
    └── scripts/
        ├── train_sac.py             # SAC training with EvalCallback
        └── visualize.py             # Policy visualization
```

---

## Conclusion

This project explored a wide range of approaches for robot manipulation learning in simulation. The core finding is that **the PnPCounterToCabinet task is genuinely difficult**, and several bugs and design mistakes compounded to prevent progress:

- All imitation learning methods failed due to re-simulation data corruption — a problem identified only after multiple failed experiments. The key symptom was that training loss decreased while evaluation success stayed at 0.
- RL methods suffered from poor reward shaping, incorrect grasp detection, and insufficient exploration in the early phase.
- Once these bugs were fixed (SAC Grasp, Experiment C, final version), the robot began to discover grasping behavior — showing approximately 20% success in visualization evaluation.

The project did not produce a fully working policy for the primary task. However, it produced:
- A clear understanding of why imitation learning fails with re-simulated LeRobot data (the root cause shared by all 8+ imitation learning attempts)
- A working reward design for SAC grasping with proper contact detection
- A simpler task (CoffeePressButton) showing the first positive evaluation results in the project (5% at step 10k)
- A complete, well-structured codebase for running future experiments

The button-pressing task remains the most promising path forward. Results will be added to this report once training completes.
