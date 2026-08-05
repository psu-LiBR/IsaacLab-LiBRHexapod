# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.sensors.imu import ImuCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.core.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg, MySceneCfg

##
# Pre-defined configs
##
from isaaclab_assets import HEXAPOD_CFG  # isort: skip


@configclass
class HexapodSceneCfg(MySceneCfg):
    """Adds the hexapod's IMU sensor slot to the shared locomotion scene."""

    imu: ImuCfg | None = None


@configclass
class HexapodRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    scene: HexapodSceneCfg = HexapodSceneCfg(num_envs=4096, env_spacing=2.5)

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # switch robot to hexapod
        self.scene.robot = HEXAPOD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # parent class's height scanner targets "base", which anymal/a1/go1 have but hexapod does not.
        # The link prims live under a "Geometry" Scope in this USD (verified against the actual stage:
        # CenterLink etc. are rigid bodies at Robot/Geometry/<name>, not directly under Robot).
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/Geometry/CenterLink"

        # IMU sensor at the physical mount location baked into the USD (rigid body, fixed-jointed to
        # CenterLink via IMUSensor_Joint). update_period=self.sim.dt matches contact_forces below --
        # freshest possible reading every physics step, same as the other ground-truth-adjacent sensors.
        self.scene.imu = ImuCfg(
            prim_path="{ENV_REGEX_NS}/Robot/Geometry/IMUSensor",
            update_period=self.sim.dt,
        )

        # parent class's contact sensor prim_path ("Robot/.*") matches every direct child of Robot,
        # including the "Physics" Scope prim that groups the hexapod's physics joints. PhysX then
        # recurses into that scope trying (and failing) to enable contact reporting on each joint
        # prim underneath, logging a "Failed to find contact report API at .../joints/<name>"
        # warning per environment per joint. Restrict the match to the actual rigid-body link names,
        # scoped under "Geometry" (where they actually live in this USD), so "Physics" is never matched.
        _body_link_names = (
            "CenterLink|BackLink|FrontLink|MiddleLeft|MiddleRight|BackLeft|BackRight|FrontLeft|FrontRight"
        )
        self.scene.contact_forces.default.prim_path = f"{{ENV_REGEX_NS}}/Robot/Geometry/({_body_link_names})"
        self.scene.contact_forces.physx.prim_path = f"{{ENV_REGEX_NS}}/Robot/Geometry/({_body_link_names})"

        # parent class's events/terminations also target "base"; hexapod's root body is CenterLink.
        # These are articulation body names (asset.data.body_names), matched via re.fullmatch against
        # the rigid-body link names -- NOT USD prim paths -- so no "Geometry" segment here.
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
