"""Custom RoboCasa counter-to-cabinet pick-and-place task.

The base RoboCasa atomic task only gives a sparse success reward. This subclass
keeps the task deterministic enough for a course project and adds a dense reward
that guides PPO through useful subgoals: reach, grasp, lift, move toward the
cabinet, place inside, and release.
"""

import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for repo_dir in (PROJECT_ROOT / "robocasa", PROJECT_ROOT / "robosuite"):
    if repo_dir.exists() and str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))

import robocasa
import robocasa.utils.object_utils as OU
from robocasa.environments.kitchen.atomic.kitchen_pick_place import (
    PickPlaceCounterToCabinet,
)


class MyPnPCounterToCab(PickPlaceCounterToCabinet):
    """Counter-to-cabinet atomic task with fixed objects and shaped reward."""

    def __init__(self, *args, **kwargs):
        if "layout_ids" not in kwargs:
            kwargs["layout_ids"] = [1]
        if "style_ids" not in kwargs:
            kwargs["style_ids"] = [1]

        self.custom_seed = kwargs.get("seed", 0)
        super().__init__(*args, **kwargs)

    def _get_obj_cfgs(self):
        """Use fixed object categories for reproducibility."""

        return [
            dict(
                name="obj",
                obj_groups="apple",
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(ref=self.cab),
                    size=(0.60, 0.30),
                    pos=(0.0, -1.0),
                    offset=(0.0, 0.10),
                ),
            ),
            dict(
                name="distr_counter",
                obj_groups="bowl",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(ref=self.cab),
                    size=(1.0, 0.30),
                    pos=(0.0, 1.0),
                    offset=(0.0, -0.05),
                ),
            ),
        ]

    def _get_placement_initializer(self, cfg_list, z_offset=0.01):
        """Keep fixture placement fixed while object placement may vary."""
        sampler = super()._get_placement_initializer(cfg_list, z_offset)

        is_fixture_placement = bool(cfg_list and cfg_list[0].get("type") == "fixture")
        if is_fixture_placement:
            seed_val = getattr(self, "custom_seed", 0) or 0
            fixed_rng = np.random.default_rng(seed=seed_val)
            sampler.rng = fixed_rng

            if hasattr(sampler, "samplers"):
                for sub_sampler in sampler.samplers.values():
                    sub_sampler.rng = fixed_rng

        return sampler

    def _obj_pos(self, obj_name="obj"):
        return np.array(self.sim.data.body_xpos[self.obj_body_id[obj_name]])

    def _eef_pos(self):
        return np.array(self.sim.data.site_xpos[self.robots[0].eef_site_id["right"]])

    def _cabinet_goal_pos(self):
        """Return the center of the first cabinet interior region."""
        regions = self.cab.get_int_sites(relative=False)
        if not regions:
            return np.array(self.cab.pos)

        p0, px, py, pz = [np.array(p) for p in next(iter(regions.values()))]
        return p0 + 0.5 * ((px - p0) + (py - p0) + (pz - p0))

    def _is_grasped(self):
        try:
            return OU.check_obj_grasped(self, "obj")
        except Exception:
            return False

    def reward(self, action=None):
        if self._check_success():
            return 10.0

        obj_pos = self._obj_pos("obj")
        eef_pos = self._eef_pos()
        goal_pos = self._cabinet_goal_pos()

        reach_dist = np.linalg.norm(eef_pos - obj_pos)
        place_dist = np.linalg.norm(obj_pos - goal_pos)

        reach_reward = 1.0 - np.tanh(6.0 * reach_dist)
        is_grasped = self._is_grasped()
        grasp_reward = 1.0 if is_grasped else 0.0
        lift_reward = np.clip((obj_pos[2] - goal_pos[2] + 0.15) / 0.20, 0.0, 1.0)
        lifted = lift_reward > 0.25

        place_reward = 0.0
        if is_grasped or lifted:
            place_reward = 1.0 - np.tanh(4.0 * place_dist)

        partial_inside = OU.obj_inside_of(self, "obj", self.cab, partial_check=True)
        full_inside = OU.obj_inside_of(self, "obj", self.cab)
        inside_reward = 0.5 * partial_inside + 1.0 * full_inside
        release_reward = (
            1.0 if full_inside and OU.gripper_obj_far(self, "obj", th=0.12) else 0.0
        )

        action_penalty = 0.0
        if action is not None:
            action_penalty = 0.01 * float(np.square(action).mean())

        return (
            0.5 * reach_reward
            + 1.0 * grasp_reward
            + 0.7 * lift_reward
            + 2.0 * place_reward
            + 3.0 * inside_reward
            + 2.0 * release_reward
            - action_penalty
        )

    def _check_success(self):
        return OU.obj_inside_of(self, "obj", self.cab) and OU.gripper_obj_far(
            self, "obj"
        )


def register_custom_env():
    """Register this task so robosuite.make('MyPnPCounterToCab', ...) works."""
    from robosuite.environments.base import register_env

    register_env(MyPnPCounterToCab)
