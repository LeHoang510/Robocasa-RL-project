"""
Privileged-observation Gymnasium wrapper for PnP pick-and-place.

Observation (25D):
  robot0_base_to_eef_pos   (3) — eef position relative to mobile base
  robot0_base_to_eef_quat  (4) — eef orientation relative to mobile base
  robot0_gripper_qpos      (2) — finger positions (grip width proxy)
  obj_pos                  (3) — object absolute world position
  obj_quat                 (4) — object orientation
  obj_to_robot0_eef_pos    (3) — vector from object centre to eef (reaching signal)
  target_pos               (3) — cabinet target position (world frame)
  obj_to_target            (3) — vector from object to target (transport signal)

FSM-gated reward (stage advances on real task state, not timestep):
  Stage 1 — not grasping:  1.0·r_reach + 1.0·r_grip
  Stage 2 — in hand:       0.5·r_reach + 1.0·r_grip + 1.0·r_grasp + 1.5·r_lift
  Stage 3 — lifted:        0.2·r_reach + 0.3·r_grip + 0.5·r_grasp + 1.0·r_lift + 2.0·r_transport
  Always:                  + 30·success

  r_reach     = 1 − tanh(10 · d(eef, obj))
  r_grip      = float(eef_near) · grip_closed_ratio
  r_grasp     = float(is_in_hand)
  r_lift      = float(is_in_hand) · clip((obj_z − table_z) / 0.15, 0, 1)
  r_transport = float(is_lifted)  · (1 − tanh(5 · d(obj, cabinet)))
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym


# ---------------------------------------------------------------------------
# Observation keys
# ---------------------------------------------------------------------------

PRIV_OBS_KEYS: list[str] = [
    "robot0_base_to_eef_pos",    # 3
    "robot0_base_to_eef_quat",   # 4
    "robot0_gripper_qpos",       # 2
    "obj_pos",                   # 3
    "obj_quat",                  # 4
    "obj_to_robot0_eef_pos",     # 3
]
# target_pos (3) and obj_to_target (3) are appended at runtime — not obs_dict keys
PRIV_OBS_DIM: int = 3 + 4 + 2 + 3 + 4 + 3 + 3 + 3   # = 25


TABLE_Z: float = 0.95           # approximate counter surface height (world frame)
GRASP_THRESHOLD: float = 0.04  # metres above table to count as "grasped"
GRASP_REACH_EPS: float = 0.08  # max obj–eef distance to count as a real grasp (not a fling)
LIFT_SCALE: float = 0.15        # denominator for lift reward normalisation
GRIPPER_OPEN_MAX: float = 0.08  # fully open Panda gripper: sum of both finger qpos ≈ 0.08 m


def extract_privileged_obs(
    obs_dict: dict,
    target_pos: np.ndarray | None = None,
) -> np.ndarray:
    """
    Build 25D privileged obs from raw robosuite obs dict.

    target_pos : cabinet target in world frame (3D). When provided, appends
                 target_pos and obj_to_target to the base 19D obs → 25D total.
                 Pass None only for legacy/testing; the env always provides it.
    """
    base = np.concatenate(
        [np.array(obs_dict[k]).flatten() for k in PRIV_OBS_KEYS],
        dtype=np.float32,
    )
    if target_pos is None:
        # Fallback: zero-fill so obs shape stays consistent
        return np.concatenate([base, np.zeros(6, dtype=np.float32)])
    obj_pos      = np.array(obs_dict["obj_pos"], dtype=np.float32)
    obj_to_target = target_pos - obj_pos
    return np.concatenate([base, target_pos.astype(np.float32), obj_to_target])


class PrivilegedPnPEnv(gym.Env):
    """
    Gymnasium wrapper around MyPnPCounterToCab with:
      • Privileged 25D state observations (see module docstring)
      • FSM-gated dense reward (stage advances on actual task state, not timestep)

    Parameters
    ----------
    raw_env : MyPnPCounterToCab instance (already constructed).
    """

    metadata = {"render_modes": []}

    def __init__(self, raw_env):
        super().__init__()
        self.raw_env = raw_env
        self._target_pos: np.ndarray = np.zeros(3, dtype=np.float32)
        self._table_z: float = TABLE_Z          # updated at each reset()
        self._initial_obj_z: float = TABLE_Z    # updated at each reset() from actual obj pos
        self._last_reward_components: dict = {}

        # Action space: PandaOmron always uses [-1, 1]^12
        # (robosuite action_spec requires a reset to work; hardcode the known value)
        self.action_space = gym.spaces.Box(
            low=-np.ones(12, dtype=np.float32),
            high=np.ones(12, dtype=np.float32),
            dtype=np.float32,
        )

        # Observation space: 25D (19D base + target_pos 3D + obj_to_target 3D)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(PRIV_OBS_DIM,),
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        obs_dict = self.raw_env.reset()
        self._target_pos   = self._read_target_pos()
        self._table_z      = self._read_table_z(obs_dict)
        # Ground-truth rest height: apple center at reset — used as lift baseline.
        # More reliable than fixture geometry which may over/underestimate.
        self._initial_obj_z = float(np.array(obs_dict["obj_pos"])[2])
        return extract_privileged_obs(obs_dict, self._target_pos), {}

    def step(self, action: np.ndarray):
        obs_dict, _, done, info = self.raw_env.step(action)

        success   = bool(self.raw_env._check_success())
        reward    = self._compute_reward(obs_dict, success)

        # robosuite done = horizon exceeded OR success
        terminated = success
        truncated  = done and not success

        info["is_success"] = float(success)
        info.update(self._last_reward_components)
        return extract_privileged_obs(obs_dict, self._target_pos), reward, terminated, truncated, info

    def close(self):
        self.raw_env.close()

    # ------------------------------------------------------------------
    # Reward computation
    # ------------------------------------------------------------------

    def _compute_reward(self, obs_dict: dict, success: bool) -> float:
        obj_to_eef = np.array(obs_dict["obj_to_robot0_eef_pos"], dtype=np.float64)
        obj_pos    = np.array(obs_dict["obj_pos"],               dtype=np.float64)

        d_eef = np.linalg.norm(obj_to_eef)
        obj_z = float(obj_pos[2])

        # Gripper openness: 0 = fully closed, 1 = fully open
        gripper_qpos = np.array(obs_dict["robot0_gripper_qpos"], dtype=np.float64)
        gripper_open = float(np.sum(np.abs(gripper_qpos)))
        grip_closed  = max(0.0, 1.0 - gripper_open / GRIPPER_OPEN_MAX)

        # ------------------------------------------------------------------
        # FSM stage detection — drives which rewards are active.
        # Each stage requires the previous one to be true.
        # ------------------------------------------------------------------
        eef_near   = d_eef < GRASP_REACH_EPS
        # is_in_hand: proximity + closed gripper (no height gate — height is what stage 2 teaches)
        is_in_hand = eef_near and grip_closed > 0.4
        is_lifted  = is_in_hand and obj_z > self._initial_obj_z + LIFT_SCALE * 0.5

        # ------------------------------------------------------------------
        # Reward components (each gated by its physical pre-condition)
        # ------------------------------------------------------------------
        r_reach     = 1.0 - np.tanh(3.0 * d_eef)
        # soft grip gate: reward closing gripper proportional to proximity (not binary)
        r_grip      = r_reach * grip_closed * 2.0
        r_contact   = float(np.exp(-d_eef))                       # dense contact signal
        r_grasp     = float(is_in_hand)
        # lift measured from initial rest position — avoids table_z estimation errors
        r_lift      = float(is_in_hand) * float(
            np.clip((obj_z - self._initial_obj_z) / LIFT_SCALE, 0.0, 1.0))
        d_transport = np.linalg.norm(obj_pos - self._target_pos)
        r_transport = float(is_lifted) * (1.0 - np.tanh(5.0 * d_transport))
        r_success   = 30.0 * float(success)

        # ------------------------------------------------------------------
        # FSM weight selection — weight set advances with actual task progress,
        # not with time.  This replaces the time-based curriculum.
        # ------------------------------------------------------------------
        if is_lifted:
            # Stage 3: object in hand and lifted → guide to cabinet
            total = (0.2 * r_reach + 0.3 * r_grip
                     + 0.5 * r_grasp + 1.0 * r_lift + 2.0 * r_transport)
        elif is_in_hand:
            # Stage 2: gripping → lift up. r_grasp=1.0 makes stage 2 slightly better than
            # stage 1 so SAC has incentive to stay in stage 2 and discover lifting.
            total = (0.5 * r_reach + 1.0 * r_grip
                     + 1.0 * r_grasp + 5.0 * r_lift)
        else:
            # Stage 1: not grasping yet → reach, close gripper, make contact
            total = 1.0 * r_reach + 1.0 * r_grip + 0.5 * r_contact

        total += r_success

        stage = 3 if is_lifted else (2 if is_in_hand else 1)
        self._last_reward_components = {
            "rc/reach":     r_reach,
            "rc/grip":      r_grip,
            "rc/contact":   r_contact,
            "rc/grasp":     r_grasp,
            "rc/lift":      r_lift,
            "rc/transport": r_transport,
            "rc/stage":     float(stage),
        }
        return float(total)

    # ------------------------------------------------------------------
    # Target position (cabinet interior)
    # ------------------------------------------------------------------

    def _read_table_z(self, obs_dict: dict) -> float:
        """
        Estimate counter surface height — three methods, most reliable first.

        1. Counter fixture geometry (robocasa Counter fixture pos + half-height).
        2. Initial object resting position minus a small object-radius offset.
        3. Hardcoded fallback TABLE_Z.
        """
        # Method 1: fixture geometry
        try:
            counter   = self.raw_env.counter
            geom_size = np.array(counter.size, dtype=np.float64).flatten()
            return float(counter.pos[2]) + float(geom_size[2])
        except Exception:
            pass
        # Method 2: object resting position at reset (obj bottom ≈ table surface)
        try:
            obj_z = float(np.array(obs_dict["obj_pos"])[2])
            return obj_z - 0.03   # approximate apple half-height
        except Exception:
            pass
        return TABLE_Z

    def _read_target_pos(self) -> np.ndarray:
        """
        Read the cabinet target position from env internals.
        Uses the 'level0' region offset from the cabinet fixture.
        Falls back to a hard-coded position if the env layout changed.
        """
        try:
            cab    = self.raw_env.cab
            offset = cab.get_reset_regions(self.raw_env)["level0"]["offset"]
            target = np.array(cab.pos, dtype=np.float64) + np.array(offset)
            return target.astype(np.float32)
        except Exception:
            return np.array([2.25, -0.2, 1.42], dtype=np.float32)
