"""
Gymnasium wrapper for the StartCoffeeMachine (CoffeePressButton) task.

Observation (16D):
  robot0_base_to_eef_pos   (3) — EEF position relative to robot base
  robot0_base_to_eef_quat  (4) — EEF orientation relative to robot base
  robot0_gripper_qpos      (2) — finger positions
  button_pos               (3) — button world position (fixed per episode)
  eef_to_button            (3) — vector EEF → button (key reaching signal)
  turned_on                (1) — whether the machine has already been activated

Reward:
  r_reach    = 1 - tanh(3 * ||eef_to_button||)      dense approach signal
  r_press    = +8 on the first button press         sparse subgoal bonus
  r_retreat  = f(distance) after press              encourages backing away
  r_success  = 30 * float(turned_on & far)          terminal bonus

Success: machine turned_on=True AND gripper moved away from button.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
import robocasa
from robocasa.environments.kitchen.atomic.kitchen_coffee import StartCoffeeMachine
from robosuite.controllers import load_composite_controller_config

OBS_DIM = 3 + 4 + 2 + 3 + 3 + 1   # 16


# ---------------------------------------------------------------------------
# Raw robocasa environment (fixed layout/style)
# ---------------------------------------------------------------------------

class ButtonEnv(StartCoffeeMachine):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("layout_ids", [1])
        kwargs.setdefault("style_ids",  [1])
        super().__init__(*args, **kwargs)


# ---------------------------------------------------------------------------
# Gymnasium wrapper
# ---------------------------------------------------------------------------

class ButtonPressEnv(gym.Env):
    """
    Gymnasium wrapper around ButtonEnv.
    Provides 15D privileged observations and shaped reward for SAC.
    """

    metadata = {"render_modes": []}

    def __init__(self, raw_env: ButtonEnv):
        super().__init__()
        self.raw_env      = raw_env
        self._button_pos  = np.zeros(3, dtype=np.float32)
        self._turned_on   = False
        self._reward_info: dict = {}

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
        super().reset(seed=seed)
        if seed is not None and hasattr(self.raw_env, "rng"):
            self.raw_env.rng = np.random.default_rng(seed)

        obs_dict = self.raw_env.reset()
        self._button_pos = self._read_button_pos()
        self._turned_on = bool(self.raw_env.coffee_machine.get_state()["turned_on"])
        return self._extract_obs(obs_dict), {}

    def step(self, action: np.ndarray):
        was_on = self._turned_on
        obs_dict, _, done, info = self.raw_env.step(action)
        self._turned_on = bool(self.raw_env.coffee_machine.get_state()["turned_on"])

        pressed_now = self._turned_on and not was_on
        success = bool(self.raw_env._check_success())
        reward = self._compute_reward(
            obs_dict,
            turned_on=self._turned_on,
            pressed_now=pressed_now,
            success=success,
        )
        info["is_success"] = float(success)
        info.update(self._reward_info)
        return self._extract_obs(obs_dict), reward, success, done and not success, info

    def render(self):
        self.raw_env.render()

    def close(self):
        self.raw_env.close()

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _extract_obs(self, obs_dict: dict) -> np.ndarray:
        eef_pos  = np.array(obs_dict["robot0_base_to_eef_pos"],  dtype=np.float32)
        eef_quat = np.array(obs_dict["robot0_base_to_eef_quat"], dtype=np.float32)
        gripper  = np.array(obs_dict["robot0_gripper_qpos"],     dtype=np.float32)

        # EEF world position for distance computation
        eef_world = self.raw_env.sim.data.site_xpos[
            self.raw_env.robots[0].eef_site_id["right"]
        ].astype(np.float32)
        eef_to_btn = self._button_pos - eef_world   # pointing toward button
        turned_on = np.array([float(self._turned_on)], dtype=np.float32)

        return np.concatenate([eef_pos, eef_quat, gripper,
                               self._button_pos, eef_to_btn, turned_on])

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _compute_reward(
        self,
        obs_dict: dict,
        turned_on: bool,
        pressed_now: bool,
        success: bool,
    ) -> float:
        eef_world = self.raw_env.sim.data.site_xpos[
            self.raw_env.robots[0].eef_site_id["right"]
        ].astype(np.float64)

        d_btn = float(np.linalg.norm(self._button_pos.astype(np.float64) - eef_world))
        r_reach = 1.0 - np.tanh(3.0 * d_btn)
        r_press = 8.0 * float(pressed_now)
        # After the machine turns on, encourage backing away so the official
        # RoboCasa success condition can trigger.
        r_retreat = float(turned_on) * np.tanh(5.0 * max(d_btn - 0.05, 0.0))
        r_success = 30.0 * float(success)

        self._reward_info = {
            "rc/reach":   r_reach,
            "rc/press":   r_press,
            "rc/retreat": r_retreat,
            "rc/dist":    d_btn,
            "rc/turned_on": float(turned_on),
            "rc/success": float(success),
        }
        return float(r_reach + r_press + r_retreat + r_success)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _read_button_pos(self) -> np.ndarray:
        """Average position of all start button geoms."""
        cm   = self.raw_env.coffee_machine
        positions = []
        for btn_name in cm._start_button_names:
            try:
                gid = self.raw_env.sim.model.geom_name2id(
                    f"{cm.naming_prefix}{btn_name}")
                positions.append(
                    self.raw_env.sim.data.geom_xpos[gid].copy())
            except Exception:
                pass
        if positions:
            return np.mean(positions, axis=0).astype(np.float32)
        return np.array(cm.pos, dtype=np.float32)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_env(
    horizon:      int  = 200,
    seed:         int  = 0,
    has_renderer: bool = False,
) -> ButtonPressEnv:
    ctrl = load_composite_controller_config(controller=None, robot="PandaOmron")
    raw = ButtonEnv(
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
    return ButtonPressEnv(raw)
