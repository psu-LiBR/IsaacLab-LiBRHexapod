# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.sensors.imu import ImuCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.core.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg, MySceneCfg

from isaaclab_assets import HEXAPOD_CFG  # isort: skip
from isaaclab_assets.robots.hexapod_actuators import with_fixed_actuator_delay  # isort: skip


@configclass
class HexapodSceneCfg(MySceneCfg):
    """Adds the hexapod's IMU sensor slot to the shared locomotion scene."""

    imu: ImuCfg | None = None


@configclass
class HexapodRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    scene: HexapodSceneCfg = HexapodSceneCfg(num_envs=4096, env_spacing=2.5)

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = HEXAPOD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # The link prims live under a "Geometry" scope in this USD (Robot/Geometry/<name>), not under "base".
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/Geometry/CenterLink"

        # IMU rigid body at the physical mount (fixed to CenterLink), read every physics step.
        self.scene.imu = ImuCfg(
            prim_path="{ENV_REGEX_NS}/Robot/Geometry/IMUSensor",
            update_period=self.sim.dt,
        )

        # Match only the link bodies: the parent's "Robot/.*" also matches the "Physics" joint scope, which
        # logs a contact-report warning per joint per env.
        _body_link_names = (
            "CenterLink|BackLink|FrontLink|MiddleLeft|MiddleRight|BackLeft|BackRight|FrontLeft|FrontRight"
        )
        self.scene.contact_forces.default.prim_path = f"{{ENV_REGEX_NS}}/Robot/Geometry/({_body_link_names})"
        self.scene.contact_forces.physx.prim_path = f"{{ENV_REGEX_NS}}/Robot/Geometry/({_body_link_names})"

        # The root body is CenterLink (articulation body names, so no "Geometry" segment).
        self.events.add_base_mass.params["asset_cfg"].body_names = "CenterLink"
        self.events.base_com.default.params["asset_cfg"].body_names = "CenterLink"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "CenterLink"
        self.terminations.base_contact.params["sensor_cfg"].body_names = "CenterLink"

        # The parent's quadruped body names (".*FOOT", ".*THIGH") do not match the hexapod.
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = [
            "MiddleLeft",
            "MiddleRight",
            "BackLeft",
            "BackRight",
            "FrontLeft",
            "FrontRight",
        ]
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = ["CenterLink", "BackLink", "FrontLink"]

        # Real-robot friction calibration (2026-09), inherited by every training cfg; PLAY cfgs pin 0.21.
        self.events.physics_material.params["static_friction_range"] = (0.18, 0.25)
        self.events.physics_material.params["dynamic_friction_range"] = (0.18, 0.25)


@configclass
class HexapodRoughEnvCfg_PLAY(HexapodRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spawn randomly in the grid instead of by terrain level, on a smaller terrain
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        # same servo lag in every rollout (the robot asset draws it randomly for training)
        self.scene.robot.actuators = with_fixed_actuator_delay(self.scene.robot.actuators)

        # eval pins friction to the best real-robot sweep row (mu = 0.21)
        self.events.physics_material.params["static_friction_range"] = (0.21, 0.21)
        self.events.physics_material.params["dynamic_friction_range"] = (0.21, 0.21)
