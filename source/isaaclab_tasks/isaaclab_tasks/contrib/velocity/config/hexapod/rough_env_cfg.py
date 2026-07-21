# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils.configclass import configclass

from isaaclab_tasks.core.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

##
# Pre-defined configs
##
from isaaclab_assets import HEXAPOD_CFG  # isort: skip


@configclass
class HexapodRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # switch robot to hexapod
        self.scene.robot = HEXAPOD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # parent class's height scanner targets "base", which anymal/a1/go1 have but hexapod does not
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/CenterLink"

        # parent class's events/terminations also target "base"; hexapod's root body is CenterLink
        self.events.add_base_mass.params["asset_cfg"].body_names = "CenterLink"
        self.events.base_com.default.params["asset_cfg"].body_names = "CenterLink"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "CenterLink"
        self.terminations.base_contact.params["sensor_cfg"].body_names = "CenterLink"

        # parent class's reward sensors target generic quadruped naming (".*FOOT", ".*THIGH"),
        # which hexapod's six leg bodies and three body-segment links don't match
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = [
            "MiddleLeft",
            "MiddleRight",
            "BackLeft",
            "BackRight",
            "FrontLeft",
            "FrontRight",
        ]
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = ["CenterLink", "BackLink", "FrontLink"]


@configclass
class HexapodRoughEnvCfg_PLAY(HexapodRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None
