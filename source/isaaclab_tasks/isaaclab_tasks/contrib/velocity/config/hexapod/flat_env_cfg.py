# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hexapod flat-terrain velocity task: forward walking at a fixed command speed."""

import math

from isaaclab.utils.configclass import configclass

from isaaclab_assets.robots.hexapod_actuators import with_fixed_actuator_delay  # isort: skip

from .hexapod_obs_cfg import HexapodFlatObservationsCfg
from .hexapod_rewards import track_ang_vel_z_exp_deadzone
from .rough_env_cfg import HexapodRoughEnvCfg


@configclass
class HexapodFlatEnvCfg(HexapodRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # the whole body is the base for mass randomization (disabled: scale 1.0) and COM offsets
        self.events.add_base_mass.params["asset_cfg"].body_names = ["CenterLink", "BackLink", "FrontLink"]
        self.events.add_base_mass.params["mass_distribution_params"] = (1.0, 1.0)
        self.events.base_com.default.params["asset_cfg"].body_names = "CenterLink"
        self.events.base_com.default.params["com_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
        }
        # Real-robot friction calibration (2026-09): effective ground friction is the robot-side material, since
        # the terrain (mu = 1.0, multiply) leaves it unchanged.
        self.events.physics_material.params["static_friction_range"] = (0.18, 0.25)
        self.events.physics_material.params["dynamic_friction_range"] = (0.18, 0.25)

        self.rewards.feet_air_time.params["sensor_cfg"].body_names = [
            "MiddleLeft",
            "MiddleRight",
            "BackLeft",
            "BackRight",
            "FrontLeft",
            "FrontRight",
        ]

        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = ["CenterLink", "BackLink", "FrontLink"]

        self.terminations.base_contact.params["sensor_cfg"].body_names = "CenterLink"
        self.events.push_robot = None
        self.events.base_external_force_torque = None

        # deterministic reset pose
        self.events.reset_base.params = {
            "pose_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        # forward-only command; the six-legged gaits reach about 0.16 m/s
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_x = (0.2, 0.2)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.debug_vis = False

        self.rewards.track_lin_vel_xy_exp.weight = 1.0
        self.rewards.track_lin_vel_xy_exp.params["std"] = math.sqrt(0.25) * 0.2

        # Yaw tracking with a deadzone on the cumulative mean yaw rate: the spine undulation has zero net drift
        # and keeps full reward, sustained turning is penalized.
        self.rewards.track_ang_vel_z_exp.weight = 0.45
        self.rewards.track_ang_vel_z_exp.func = track_ang_vel_z_exp_deadzone
        self.rewards.track_ang_vel_z_exp.params = {
            "std": math.sqrt(0.25) * 0.15,
            "command_name": "base_velocity",
            "deadzone": 1.6,
        }

        self.rewards.dof_torques_l2.weight = -5.0e-8

        # Fixed-threshold air time. The per-leg duty-cycle version (hexapod_rewards.feet_air_time_per_leg) gave
        # too weak a signal to shape the gait.
        self.rewards.feet_air_time.weight = 0.25
        self.rewards.feet_air_time.params["threshold"] = 0.1  # s; shuffling air phases are typically < 0.02 s

        self.rewards.dof_pos_limits.weight = -1.0
        self.rewards.dof_acc_l2.weight = -8.5e-14
        self.rewards.ang_vel_xy_l2.weight = 0.0
        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.lin_vel_z_l2.weight = -0.00000001
        self.rewards.action_rate_l2.weight = -2.5e-6

        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None

        # Asymmetric actor-critic: the actor sees only hardware-observable quantities, the critic also sees
        # ground-truth base_lin_vel.
        self.observations = HexapodFlatObservationsCfg()  # type: ignore[assignment]

        self.curriculum.terrain_levels = None

        self.viewer.eye = (0.5, 0.0, 1.0)
        self.viewer.lookat = (0.5, 0.0, 0.0)
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"


class HexapodFlatEnvCfg_PLAY(HexapodFlatEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        # same servo lag in every rollout (the robot asset draws it randomly for training)
        self.scene.robot.actuators = with_fixed_actuator_delay(self.scene.robot.actuators)

        # play at the top speed
        self.commands.base_velocity.ranges.lin_vel_x = (0.16, 0.16)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)

        self.commands.base_velocity.resampling_time_range = (1000.0, 1000.0)

        # eval pins friction to the best real-robot sweep row (mu = 0.21)
        self.events.physics_material.params["static_friction_range"] = (0.21, 0.21)
        self.events.physics_material.params["dynamic_friction_range"] = (0.21, 0.21)

        # follow the robot from behind and above
        self.viewer.eye = (-1.0, 0.0, 0.5)
        self.viewer.lookat = (0.0, 0.0, 0.0)
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
