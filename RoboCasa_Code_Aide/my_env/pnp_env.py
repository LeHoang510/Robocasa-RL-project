"""
Custom PnPCounterToCab environment with modified reward function.

This class inherits from the original RoboCasa PnPCounterToCab environment
and allows you to customize the reward function without modifying the
original robocasa or skrl packages.

Usage:
    from my_env import MyPnPCounterToCab

    env = MyPnPCounterToCab(
        robots="PandaOmron",
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=["robot0_agentview_center"],
        camera_heights=128,
        camera_widths=128,
        control_freq=20,
        reward_shaping=True,
    )
"""

import os
import numpy as np
import robocasa
from robocasa.environments.kitchen.atomic.kitchen_pick_place import PickPlaceCounterToCabinet
import robocasa.utils.object_utils as OU


class MyPnPCounterToCab(PickPlaceCounterToCabinet):
    """
    PnPCounterToCab environment with modified reward function.

    This class inherits from the original PickPlaceCounterToCabinet and overrides
    the reward() method to implement a custom reward function.

    """

    # Placement region per difficulty level: (size, pos)
    # difficulty 0 (easy)  : apple always close to cabinet
    # difficulty 1 (medium): medium region
    # difficulty 2 (hard)  : full original region (same as baseline exp1)
    DIFFICULTY_PLACEMENTS = [
        ((0.15, 0.10), (0.0, -0.9)),   # easy
        ((0.35, 0.20), (0.0, -0.5)),   # medium
        ((0.60, 0.30), (0.0, -1.0)),   # hard  - same as original
    ]

    def __init__(self, *args, **kwargs):
        """
        Initialize the custom environment.
        All arguments are passed to the parent PickPlaceCounterToCabinet class.

        This environment fixes the kitchen layout to a single configuration
        while allowing object positions to vary.
        """
        # Fix the kitchen layout to layout 0, style 0 (you can change these values)
        # This prevents the kitchen configuration from changing between episodes
        if 'layout_ids' not in kwargs:
            kwargs['layout_ids'] = [1]  # Use layout 1
        if 'style_ids' not in kwargs:
            kwargs['style_ids'] = [1]   # Use style 1

        # Capture the seed if provided
        self.custom_seed = kwargs.get('seed', 0)

        self.difficulty = 0

        super().__init__(*args, **kwargs)

    def set_difficulty(self, level: int):
        """
        Advance the curriculum difficulty. Called by CurriculumCallback in train.py.
        Takes effect at the next episode reset (when _get_obj_cfgs is re-run).
        """
        self.difficulty = int(np.clip(level, 0, len(self.DIFFICULTY_PLACEMENTS) - 1))

    def _get_obj_cfgs(self):
        """
        Override to set specific objects:
        - Sample object (obj): always apple_1
        - Distractor objects: always bowl_1

        MODIFIED: apple placement size/pos now depends on self.difficulty
        (difficulty=2 reproduces the original hard placement exactly).
        """
        cfgs = []

        # Get the base path for robocasa objects
        base_path = os.path.join(robocasa.models.assets_root, "objects", "objaverse")

        # read placement parameters from difficulty level
        size, pos = self.DIFFICULTY_PLACEMENTS[self.difficulty]
        # (when difficulty=2, size=(0.60,0.30) and pos=(0.0,-1.0) - original values)

        # Sample object: always apple_1 (using full path to model.xml)
        apple_1_path = os.path.join(base_path, "apple", "apple_1", "model.xml")
        cfgs.append(
            dict(
                name="obj",
                obj_groups=apple_1_path,  # Force apple_1 as the sample object
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.cab,
                    ),
                    size=size,           # ← was (0.60, 0.30), now from difficulty
                    pos=pos,             # ← was (0.0, -1.0), now from difficulty
                    offset=(0.0, 0.10),
                ),
            )
        )

        # Distractor on counter: always bowl_1 (using full path to model.xml)
        bowl_1_path = os.path.join(base_path, "bowl", "bowl_1", "model.xml")
        cfgs.append(
            dict(
                name="distr_counter",
                obj_groups=bowl_1_path,  # Force bowl_1 as distractor
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.cab,
                    ),
                    size=(1.0, 0.30),
                    pos=(0.0, 1.0),
                    offset=(0.0, -0.05),
                ),
            )
        )

        return cfgs

    def _get_placement_initializer(self, cfg_list, z_offset=0.01):
        """
        Override to enforce deterministic placement for fixtures (appliances),
        while allowing random placement for objects.
        """
        sampler = super()._get_placement_initializer(cfg_list, z_offset)

        # Check if this sampler is for fixtures (appliances)
        # Fixture configs usually have type="fixture"
        is_fixture_placement = False
        if cfg_list and len(cfg_list) > 0:
            if cfg_list[0].get("type") == "fixture":
                is_fixture_placement = True

        if is_fixture_placement:
            # Use the environment seed for fixture placement to ensure deterministic behavior
            # appliances will stay in place throughout the run (assuming constant seed for env)
            # but will change if you change the training run seed.

            # Retrieve the seed captured in __init__
            seed_val = getattr(self, "custom_seed", 0)
            if seed_val is None:
                seed_val = 0

            fixed_rng = np.random.default_rng(seed=seed_val)

            # Set the RNG for the main sampler
            sampler.rng = fixed_rng

            # Set the RNG for all sub-samplers
            if hasattr(sampler, "samplers"):
                for sub_sampler in sampler.samplers.values():
                    sub_sampler.rng = fixed_rng

        return sampler

    def reward(self, action=None):
        """
        Dense shaped reward guiding the robot through 5 sub-goals.

        Reward stages (cumulative per step):
          Stage 1 - Reach     : gripper approaches the apple       max ~0.15
          Stage 2 - Grasp     : gripper closes on the apple        +0.25
          Stage 3 - Transport : carry the apple toward the cabinet max ~0.30
          Stage 4 - Place     : apple is inside the cabinet        +1.00
          Stage 5 - Release   : gripper moves away after placing   +0.50
                                                      total max  ~  2.20

        All stages are additive so the agent always receives a gradient
        signal toward the next goal even before it masters the current one.
        """
        # Object and end-effector positions
        obj_pos = self.sim.data.body_xpos[self.obj_body_id["obj"]].copy()
        try:
            eef_pos = self.sim.data.site_xpos[
                self.robots[0].eef_site_id["right"]
            ].copy()
        except (KeyError, TypeError):
            eef_pos = self.sim.data.site_xpos[
                self.robots[0].eef_site_id
            ].copy()

        # Boolean state flags 
        is_grasped = OU.check_obj_grasped(self, "obj")
        obj_in_cab = OU.obj_inside_of(self, "obj", self.cab)
        released   = OU.gripper_obj_far(self, "obj", th=0.15)

        r = 0.0

        # Stage 1 – Reach: very small so it never dominates later stages.
        # Max per episode: 0.02 × 500 = 10
        if not is_grasped and not obj_in_cab:
            dist_to_obj = np.linalg.norm(eef_pos - obj_pos)
            r += 0.02 * (1.0 - np.tanh(5.0 * dist_to_obj))

        # Stage 2 – Grasp (per step).
        # Max per episode: 0.25 × 500 = 125
        if is_grasped:
            r += 0.25
            # Stage 3 – Transport (per step, only while grasping).
            # Max per episode: 0.20 × 500 = 100  →  grasp+transport max = 225
            dist_to_cab = OU.obj_fixture_bbox_min_dist(self, "obj", self.cab)
            r += 0.20 * (1.0 - np.tanh(3.0 * dist_to_cab))

        # Stage 4 – Placement: PER-STEP reward, much larger than grasp+transport.
        # If placed at step 200 and held for 300 steps: 2.00 × 300 = 600 >> 225
        # This makes placing strictly better than endless grasping.
        if obj_in_cab:
            r += 2.00
            # Stage 5 – Release: additional per-step reward on top of placement.
            if released:
                r += 1.00

        return r
