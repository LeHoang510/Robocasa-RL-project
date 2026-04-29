"""
GoalEnv wrapper for PnPCounterToCab — required for SAC + HER (exp4).

HER (Hindsight Experience Replay) needs a Dict observation space with three keys:
  'observation'   - flat proprioceptive obs (same as standard GymWrapper output)
  'achieved_goal' - current apple position  (x, y, z)
  'desired_goal'  - target position inside cabinet (x, y, z)

At the end of each episode, HER relabels the achieved_goal of past transitions
as the desired_goal, turning failed episodes into synthetic successes. This
dramatically reduces the sparsity of the placement reward signal.

The compute_reward() method defines what counts as "success" for relabelled
transitions: apple within `goal_threshold` metres of the cabinet target.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class PnPGoalEnv(gym.Env):
    """
    GoalEnv wrapper around MyPnPCounterToCab + GymWrapper.

    Parameters
    ----------
    raw_env : MyPnPCounterToCab
        The unwrapped robosuite environment (used to read simulation state).
    gym_env : GymWrapper-wrapped + Monitor env
        The standard gym-compatible environment (used for step/reset/obs).
    goal_threshold : float
        Distance (metres) below which the apple is considered "at the goal".
    """

    metadata = {"render_modes": []}

    def __init__(self, raw_env, gym_env, goal_threshold: float = 0.10):
        super().__init__()
        self.raw_env        = raw_env
        self.gym_env        = gym_env
        self.goal_threshold = goal_threshold

        flat_obs_dim  = gym_env.observation_space.shape[0]
        goal_dim      = 3   # (x, y, z)

        goal_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(goal_dim,), dtype=np.float32
        )
        self.observation_space = spaces.Dict({
            "observation":   gym_env.observation_space,
            "achieved_goal": goal_space,
            "desired_goal":  goal_space,
        })
        self.action_space  = gym_env.action_space
        self._desired_goal = np.zeros(goal_dim, dtype=np.float32)

    # Internal helpers 

    def _apple_pos(self) -> np.ndarray:
        return self.raw_env.sim.data.body_xpos[
            self.raw_env.obj_body_id["obj"]
        ].copy().astype(np.float32)

    def _cabinet_target(self) -> np.ndarray:
        """Centre of the cabinet opening — the target placement position."""
        return np.array(self.raw_env.cab.pos, dtype=np.float32)

    def _wrap_obs(self, flat_obs: np.ndarray) -> dict:
        return {
            "observation":   flat_obs.astype(np.float32),
            "achieved_goal": self._apple_pos(),
            "desired_goal":  self._desired_goal.copy(),
        }

    # GoalEnv interface

    def compute_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: dict,
    ) -> np.ndarray:
        """
        Sparse binary reward used by HER when relabelling transitions.
          0.0  - apple is within goal_threshold of the cabinet target
         -1.0  - apple is further away

        Vectorised: achieved_goal / desired_goal may be (N, 3) batches.
        """
        dist = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        return -(dist > self.goal_threshold).astype(np.float32)

    # Standard Gym interface
    def reset(self, **kwargs):
        flat_obs, info          = self.gym_env.reset(**kwargs)
        self._desired_goal      = self._cabinet_target()
        return self._wrap_obs(flat_obs), info

    def step(self, action):
        flat_obs, reward, terminated, truncated, info = self.gym_env.step(action)
        goal_obs = self._wrap_obs(flat_obs)
        # HER requires 'is_success' in info for logging
        info["is_success"] = float(self.raw_env._check_success())
        return goal_obs, reward, terminated, truncated, info

    def render(self):
        return self.gym_env.render()

    def close(self):
        return self.gym_env.close()
