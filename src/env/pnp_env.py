"""
Self-contained PnPCounterToCabinet environment for the src/ pipeline.

Two classes:
  PnPEnv           — raw robocasa env (fixed layout, curriculum difficulty,
                     custom dense reward)
  PrivilegedPnPEnv — Gymnasium wrapper with 25D privileged observations and
                     FSM-gated reward (stage advances on actual task state)

Observation (25D):
  robot0_base_to_eef_pos   (3) — eef position relative to mobile base
  robot0_base_to_eef_quat  (4) — eef orientation relative to mobile base
  robot0_gripper_qpos      (2) — finger positions
  obj_pos                  (3) — object world position
  obj_quat                 (4) — object orientation
  obj_to_robot0_eef_pos    (3) — vector from object to eef (reaching signal)
  target_pos               (3) — cabinet target in world frame
  obj_to_target            (3) — vector from object to target (transport signal)

FSM reward (stage driven by actual task state):
  Stage 1 — reaching  : r_reach + r_grip + r_contact
  Stage 2 — grasping  : r_reach + r_grip + r_grasp + r_lift
  Stage 3 — lifting   : r_reach + r_grip + r_grasp + r_lift + r_transport
  Always              : + 30 × success
"""

from __future__ import annotations

import os
import numpy as np
import gymnasium as gym
import robocasa
from robocasa.environments.kitchen.atomic.kitchen_pick_place import PickPlaceCounterToCabinet
import robocasa.utils.object_utils as OU
from robosuite.controllers import load_composite_controller_config


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OBS_KEYS: list[str] = [
    "robot0_base_to_eef_pos",    # 3
    "robot0_base_to_eef_quat",   # 4
    "robot0_gripper_qpos",       # 2
    "obj_pos",                   # 3
    "obj_quat",                  # 4
    "obj_to_robot0_eef_pos",     # 3
]
# target_pos (3) + obj_to_target (3) appended at runtime → total = 25
OBS_DIM: int = 3 + 4 + 2 + 3 + 4 + 3 + 3 + 3

GRASP_REACH_EPS:  float = 0.08   # max obj–eef distance to count as near
LIFT_SCALE:       float = 0.15   # lift height normalisation denominator
GRIPPER_OPEN_MAX: float = 0.08   # fully open Panda gripper finger sum (m)

# Object placement regions per curriculum level
_DIFFICULTY_PLACEMENTS = [
    ((0.15, 0.10), (0.0, -0.9)),   # easy   — apple close to cabinet
    ((0.35, 0.20), (0.0, -0.5)),   # medium
    ((0.60, 0.30), (0.0, -1.0)),   # hard   — original full region
]


# ---------------------------------------------------------------------------
# 25D observation extractor
# ---------------------------------------------------------------------------

def extract_obs(obs_dict: dict, target_pos: np.ndarray | None = None) -> np.ndarray:
    """Build 25D privileged obs from raw robosuite obs_dict."""
    base = np.concatenate(
        [np.array(obs_dict[k]).flatten() for k in OBS_KEYS],
        dtype=np.float32,
    )
    if target_pos is None:
        return np.concatenate([base, np.zeros(6, dtype=np.float32)])
    obj_to_target = target_pos.astype(np.float32) - np.array(obs_dict["obj_pos"], dtype=np.float32)
    return np.concatenate([base, target_pos.astype(np.float32), obj_to_target])


# ---------------------------------------------------------------------------
# Raw robocasa environment
# ---------------------------------------------------------------------------

class PnPEnv(PickPlaceCounterToCabinet):
    """
    PnPCounterToCabinet with:
      • Fixed layout/style (layout_ids=[1], style_ids=[1])
      • Always apple_1 object, bowl_1 distractor
      • Curriculum difficulty (0=easy, 1=medium, 2=hard object placement)
      • Dense shaped reward (reach → grasp → transport → place)
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("layout_ids", [1])
        kwargs.setdefault("style_ids",  [1])
        self.custom_seed = kwargs.get("seed", 0)
        self.difficulty  = 0
        super().__init__(*args, **kwargs)

    def set_difficulty(self, level: int):
        self.difficulty = int(np.clip(level, 0, len(_DIFFICULTY_PLACEMENTS) - 1))

    def _get_obj_cfgs(self):
        base  = os.path.join(robocasa.models.assets_root, "objects", "objaverse")
        size, pos = _DIFFICULTY_PLACEMENTS[self.difficulty]
        return [
            dict(
                name="obj",
                obj_groups=os.path.join(base, "apple", "apple_1", "model.xml"),
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(ref=self.cab),
                    size=size, pos=pos, offset=(0.0, 0.10),
                ),
            ),
            dict(
                name="distr_counter",
                obj_groups=os.path.join(base, "bowl", "bowl_1", "model.xml"),
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(ref=self.cab),
                    size=(1.0, 0.30), pos=(0.0, 1.0), offset=(0.0, -0.05),
                ),
            ),
        ]

    def _get_placement_initializer(self, cfg_list, z_offset=0.01):
        sampler = super()._get_placement_initializer(cfg_list, z_offset)
        is_fixture = cfg_list and cfg_list[0].get("type") == "fixture"
        if is_fixture:
            rng = np.random.default_rng(seed=getattr(self, "custom_seed", 0) or 0)
            sampler.rng = rng
            if hasattr(sampler, "samplers"):
                for s in sampler.samplers.values():
                    s.rng = rng
        return sampler

    def reward(self, action=None):
        """Dense reward used when PrivilegedPnPEnv is NOT wrapping this env."""
        obj_pos = self.sim.data.body_xpos[self.obj_body_id["obj"]].copy()
        try:
            eef_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id["right"]].copy()
        except (KeyError, TypeError):
            eef_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id].copy()

        is_grasped = OU.check_obj_grasped(self, "obj")
        obj_in_cab = OU.obj_inside_of(self, "obj", self.cab)
        released   = OU.gripper_obj_far(self, "obj", th=0.15)

        r = 0.0
        if not is_grasped and not obj_in_cab:
            r += 0.02 * (1.0 - np.tanh(5.0 * np.linalg.norm(eef_pos - obj_pos)))
        if is_grasped:
            r += 0.25
            dist_cab = OU.obj_fixture_bbox_min_dist(self, "obj", self.cab)
            r += 0.20 * (1.0 - np.tanh(3.0 * dist_cab))
        if obj_in_cab:
            r += 2.00
            if released:
                r += 1.00
        return r


# ---------------------------------------------------------------------------
# Gymnasium wrapper with 25D obs + FSM reward
# ---------------------------------------------------------------------------

class PrivilegedPnPEnv(gym.Env):
    """
    Gymnasium wrapper around PnPEnv.
    Provides 25D privileged observations and FSM-gated dense reward.
    """

    metadata = {"render_modes": []}

    def __init__(self, raw_env: PnPEnv):
        super().__init__()
        self.raw_env = raw_env
        self._target_pos:    np.ndarray = np.zeros(3, dtype=np.float32)
        self._initial_obj_z: float      = 0.95
        self._reward_info:   dict       = {}

        self.action_space = gym.spaces.Box(
            low=-np.ones(12, dtype=np.float32),
            high=np.ones(12, dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        obs_dict = self.raw_env.reset()
        self._target_pos    = self._read_target_pos()
        self._initial_obj_z = float(np.array(obs_dict["obj_pos"])[2])
        return extract_obs(obs_dict, self._target_pos), {}

    def step(self, action: np.ndarray):
        obs_dict, _, done, info = self.raw_env.step(action)
        success    = bool(self.raw_env._check_success())
        reward     = self._compute_reward(obs_dict, success)
        terminated = success
        truncated  = done and not success
        info["is_success"] = float(success)
        info.update(self._reward_info)
        return extract_obs(obs_dict, self._target_pos), reward, terminated, truncated, info

    def close(self):
        self.raw_env.close()

    # ------------------------------------------------------------------
    # FSM reward
    # ------------------------------------------------------------------

    def _compute_reward(self, obs_dict: dict, success: bool) -> float:
        obj_to_eef   = np.array(obs_dict["obj_to_robot0_eef_pos"], dtype=np.float64)
        obj_pos      = np.array(obs_dict["obj_pos"],               dtype=np.float64)
        gripper_qpos = np.array(obs_dict["robot0_gripper_qpos"],   dtype=np.float64)

        d_eef        = np.linalg.norm(obj_to_eef)
        obj_z        = float(obj_pos[2])
        grip_closed  = max(0.0, 1.0 - float(np.sum(np.abs(gripper_qpos))) / GRIPPER_OPEN_MAX)

        # FSM state
        eef_near   = d_eef < GRASP_REACH_EPS
        is_in_hand = eef_near and grip_closed > 0.4
        is_lifted  = is_in_hand and obj_z > self._initial_obj_z + LIFT_SCALE * 0.5

        # Components
        r_reach     = 1.0 - np.tanh(3.0 * d_eef)
        r_grip      = r_reach * grip_closed * 2.0
        r_contact   = float(np.exp(-d_eef))
        r_grasp     = float(is_in_hand)
        r_lift      = float(is_in_hand) * float(
            np.clip((obj_z - self._initial_obj_z) / LIFT_SCALE, 0.0, 1.0))
        d_transport = np.linalg.norm(obj_pos - self._target_pos)
        r_transport = float(is_lifted) * (1.0 - np.tanh(5.0 * d_transport))
        r_success   = 30.0 * float(success)

        # FSM weight selection
        if is_lifted:
            total = 0.2*r_reach + 0.3*r_grip + 0.5*r_grasp + 1.0*r_lift + 2.0*r_transport
        elif is_in_hand:
            total = 0.5*r_reach + 1.0*r_grip + 1.0*r_grasp + 5.0*r_lift
        else:
            total = 1.0*r_reach + 1.0*r_grip + 0.5*r_contact
        total += r_success

        stage = 3 if is_lifted else (2 if is_in_hand else 1)
        self._reward_info = {
            "rc/stage":     float(stage),
            "rc/reach":     r_reach,
            "rc/grip":      r_grip,
            "rc/contact":   r_contact,
            "rc/grasp":     r_grasp,
            "rc/lift":      r_lift,
            "rc/transport": r_transport,
        }
        return float(total)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _read_target_pos(self) -> np.ndarray:
        try:
            cab    = self.raw_env.cab
            offset = cab.get_reset_regions(self.raw_env)["level0"]["offset"]
            return (np.array(cab.pos, dtype=np.float64) + np.array(offset)).astype(np.float32)
        except Exception:
            return np.array([2.25, -0.2, 1.42], dtype=np.float32)


# ---------------------------------------------------------------------------
# Grasp-only environment (curriculum stage 1)
# ---------------------------------------------------------------------------

class GraspEnv(gym.Env):
    """
    Simplified environment for learning to grasp only.

    Success = gripper holds the apple for GRASP_HOLD_STEPS consecutive steps.
    Shorter horizon + dense reach/grasp reward makes grasping much easier to
    discover than in the full 300-step task.

    Once a grasp policy reaches >50% success, transfer its weights to FullSAC.
    """

    GRASP_HOLD_STEPS = 5   # must hold grasp this many steps to count as success

    metadata = {"render_modes": []}

    def __init__(self, raw_env: PnPEnv):
        super().__init__()
        self.raw_env         = raw_env
        self._target_pos     = np.zeros(3, dtype=np.float32)
        self._initial_obj_z  = 0.95
        self._grasp_counter  = 0
        self._reward_info    = {}

        self.action_space = gym.spaces.Box(
            low=-np.ones(12, dtype=np.float32),
            high=np.ones(12, dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        obs_dict = self.raw_env.reset()
        self._target_pos    = self._read_target_pos()
        self._initial_obj_z = float(np.array(obs_dict["obj_pos"])[2])
        self._grasp_counter = 0
        return extract_obs(obs_dict, self._target_pos), {}

    def step(self, action: np.ndarray):
        obs_dict, _, done, info = self.raw_env.step(action)

        reward, is_grasped = self._compute_reward(obs_dict)

        # Count consecutive grasp steps
        if is_grasped:
            self._grasp_counter += 1
        else:
            self._grasp_counter = 0

        success    = self._grasp_counter >= self.GRASP_HOLD_STEPS
        terminated = success
        truncated  = done and not success

        info["is_success"]    = float(success)
        info["grasp_counter"] = self._grasp_counter
        info.update(self._reward_info)
        return extract_obs(obs_dict, self._target_pos), reward, terminated, truncated, info

    def close(self):
        self.raw_env.close()

    def _compute_reward(self, obs_dict: dict) -> tuple[float, bool]:
        obj_to_eef   = np.array(obs_dict["obj_to_robot0_eef_pos"], dtype=np.float64)
        gripper_qpos = np.array(obs_dict["robot0_gripper_qpos"],   dtype=np.float64)

        d_eef       = np.linalg.norm(obj_to_eef)
        grip_closed = max(0.0, 1.0 - float(np.sum(np.abs(gripper_qpos))) / GRIPPER_OPEN_MAX)

        # Use robocasa contact detection — more reliable than proxy distance check
        is_grasped = bool(OU.check_obj_grasped(self.raw_env, "obj"))

        r_reach = 1.0 - np.tanh(3.0 * d_eef)
        r_grip  = r_reach * grip_closed * 2.0   # reward closing gripper when near
        r_grasp = 10.0 * float(is_grasped)      # large bonus for actual grasp

        # No r_contact: exp(-d) gives 0.70 free reward even at 35cm, kills approach gradient
        total = r_reach + r_grip + r_grasp

        self._reward_info = {
            "rc/reach":   r_reach,
            "rc/grip":    r_grip,
            "rc/contact": 0.0,
            "rc/grasp":   float(is_grasped),
        }
        return float(total), is_grasped

    def _read_target_pos(self) -> np.ndarray:
        try:
            cab    = self.raw_env.cab
            offset = cab.get_reset_regions(self.raw_env)["level0"]["offset"]
            return (np.array(cab.pos) + np.array(offset)).astype(np.float32)
        except Exception:
            return np.array([2.25, -0.2, 1.42], dtype=np.float32)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_env(
    horizon:      int  = 300,
    seed:         int  = 0,
    privileged:   bool = True,
    has_renderer: bool = False,
    difficulty:   int  = 0,
    grasp_only:   bool = False,
) -> PrivilegedPnPEnv | GraspEnv | PnPEnv:
    """
    Build and return a PnP environment.

    privileged=True  → PrivilegedPnPEnv (25D obs, FSM reward) — use for training
    privileged=False → raw PnPEnv (robosuite obs dict)        — use for recording
    """
    ctrl = load_composite_controller_config(controller=None, robot="PandaOmron")
    raw = PnPEnv(
        robots="PandaOmron",
        controller_configs=ctrl,
        use_camera_obs=False,
        has_renderer=has_renderer,
        has_offscreen_renderer=False,
        reward_shaping=True,
        control_freq=20,
        ignore_done=False,
        horizon=horizon,
        seed=seed,
    )
    raw.set_difficulty(difficulty)
    if grasp_only:
        return GraspEnv(raw)
    if privileged:
        return PrivilegedPnPEnv(raw)
    return raw
