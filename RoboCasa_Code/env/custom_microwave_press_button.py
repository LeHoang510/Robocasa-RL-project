"""Custom RoboCasa microwave button-press task.

This task is intentionally simpler than pick-and-place: the robot only needs to
reach the microwave start button, press it, and move away after activation.
"""

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for repo_dir in (PROJECT_ROOT / "robocasa", PROJECT_ROOT / "robosuite"):
    if repo_dir.exists() and str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))

from robocasa.environments.kitchen.atomic.kitchen_microwave import MicrowavePressButton


class MyMicrowavePressButton(MicrowavePressButton):
    """Turn on the microwave by pressing the start button, with dense shaping."""

    def __init__(self, *args, **kwargs):
        if "layout_ids" not in kwargs:
            kwargs["layout_ids"] = [1]
        if "style_ids" not in kwargs:
            kwargs["style_ids"] = [1]

        super().__init__(behavior="turn_on", *args, **kwargs)

    def _get_obj_cfgs(self):
        """No loose object is needed for a pure button-press task."""
        return []

    def _button_name(self):
        return "start_button"

    def _button_geom_name(self):
        return f"{self.microwave.naming_prefix}{self._button_name()}"

    def _button_pos(self):
        button_id = self.sim.model.geom_name2id(self._button_geom_name())
        return np.array(self.sim.data.geom_xpos[button_id])

    def _eef_pos(self):
        return np.array(self.sim.data.site_xpos[self.robots[0].eef_site_id["right"]])

    def _button_pressed(self):
        return self.check_contact(
            self.robots[0].gripper["right"],
            self._button_geom_name(),
        )

    def reward(self, action=None):
        # Success bonus
        if self._check_success():
            return 25.0
    
        # Positions
        eef_pos = self._eef_pos()
        button_pos = self._button_pos()
        reach_dist = np.linalg.norm(eef_pos - button_pos)
    
        # Shaped reaching reward (smooth, dense)
        reach_reward = 1.0 - np.tanh(8.0 * reach_dist)
    
        # Binary signals (kept simple, scaled later)
        contact_reward = 1.0 if self._button_pressed() else 0.0
    
        turned_on = self.microwave.get_state()["turned_on"]
        press_reward = 1.0 if turned_on else 0.0
    
        release_reward = 1.0 if (
            turned_on and self.microwave.gripper_button_far(
                self, button=self._button_name()
            )
        ) else 0.0
    
        # Small action penalty (stabilizes motion)
        action_penalty = 0.0
        if action is not None:
            action_penalty = 0.005 * float(np.square(action).mean())
    
        # Final weighted reward
        return (
            3.0 * reach_reward        # learn to go to button
            + 3.0 * contact_reward   # encourage touching
            + 10.0 * press_reward    # strong signal for activation
            + 5.0 * release_reward   # move away after press
            - action_penalty
        )

def register_custom_env():
    """Register this task so robosuite.make('MyMicrowavePressButton', ...) works."""
    from robosuite.environments.base import register_env

    register_env(MyMicrowavePressButton)
