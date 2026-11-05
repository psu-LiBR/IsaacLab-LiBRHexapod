# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward-shaping configuration for the hexapod flat-locomotion task.

The baseline (`flat_env_cfg.py`) holds the commanded speed with short, rapid steps: its tight velocity
tracking leaves stride length unconstrained and its swing phases stay below the `feet_air_time` threshold.
This config makes long swing phases pay:

- looser velocity tracking (std 0.15), so a longer stride is not penalized,
- higher `feet_air_time` weight and threshold,
- a `feet_slide` penalty, so the reward cannot be collected by sliding a planted foot,
- higher ground friction, since the baseline range let the feet slip,
- a stronger `action_rate_l2` penalty against jitter.

`HexapodFlatRshapeSlide045/035/025EnvCfg` change only the `feet_slide` weight.
"""

import math

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

import isaaclab_tasks.core.velocity.mdp as mdp  # for feet_slide

from isaaclab_assets.robots.hexapod_actuators import with_fixed_actuator_delay  # isort: skip

from .hexapod_obs_cfg import HexapodFlatObservationsCfg
from .hexapod_rewards import track_ang_vel_z_exp_ema
from .rough_env_cfg import HexapodRoughEnvCfg


@configclass
class HexapodFlatRshapeEnvCfg(HexapodRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.events.add_base_mass.params["asset_cfg"].body_names = ["CenterLink", "BackLink", "FrontLink"]
        self.events.add_base_mass.params["mass_distribution_params"] = (1.0, 1.0)  # no randomization
        self.events.base_com.params["asset_cfg"].body_names = "CenterLink"
        self.events.base_com.params["com_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
        }
        # higher than the baseline, which let the feet slide instead of push
        self.events.physics_material.params["static_friction_range"] = (0.9, 1.0)
        self.events.physics_material.params["dynamic_friction_range"] = (0.7, 0.8)

        self.events.base_external_force_torque.params["asset_cfg"].body_names = "CenterLink"

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
        # std 0.1 -> 0.15: a sharper term is met by any gait at the commanded speed, with no gradient toward
        # longer strides.
        self.rewards.track_lin_vel_xy_exp.params["std"] = math.sqrt(0.25) * 0.3

        # EMA-smoothed yaw tracking: the spine undulation has zero net drift and keeps full reward, sustained
        # turning is penalized.
        self.rewards.track_ang_vel_z_exp.weight = 0.45
        self.rewards.track_ang_vel_z_exp.func = track_ang_vel_z_exp_ema
        self.rewards.track_ang_vel_z_exp.params = {
            "std": math.sqrt(0.25) * 0.15,
            "command_name": "base_velocity",
            "alpha": 0.98,  # ~33-step window (~0.66 s at 0.02 s dt)
        }

        self.rewards.dof_torques_l2.weight = -5.0e-5

        # Pay for airborne time per leg; friction and feet_slide keep it from being collected by lifting legs
        # without moving forward.
        self.rewards.feet_air_time.weight = 2.5
        self.rewards.feet_air_time.params["threshold"] = 0.16

        self.rewards.dof_pos_limits.weight = -1.0
        self.rewards.dof_acc_l2.weight = -8.5e-14
        self.rewards.ang_vel_xy_l2.weight = 0.0
        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.lin_vel_z_l2.weight = -0.00000001
        self.rewards.action_rate_l2.weight = -3.0e-2  # suppress high-frequency action jitter

        # Penalize horizontal foot motion while the foot is in contact.
        self.rewards.feet_slide = RewTerm(
            func=mdp.feet_slide,
            weight=-0.6,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=["MiddleLeft", "MiddleRight", "BackLeft", "BackRight", "FrontLeft", "FrontRight"],
                ),
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=["MiddleLeft", "MiddleRight", "BackLeft", "BackRight", "FrontLeft", "FrontRight"],
                ),
            },
        )

        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        self.scene.height_scanner = None

        # Asymmetric actor-critic: the critic also sees ground-truth base_lin_vel.
        self.observations = HexapodFlatObservationsCfg()  # type: ignore[assignment]

        self.curriculum.terrain_levels = None

        self.viewer.eye = (0.5, 0.0, 1.0)
        self.viewer.lookat = (0.5, 0.0, 0.0)
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"


class HexapodFlatRshapeEnvCfg_PLAY(HexapodFlatRshapeEnvCfg):
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

        self.events.physics_material.params["static_friction_range"] = (0.9, 1.0)
        self.events.physics_material.params["dynamic_friction_range"] = (0.7, 0.8)

        # follow the robot from behind and above
        self.viewer.eye = (-1.0, 0.0, 0.5)
        self.viewer.lookat = (0.0, 0.0, 0.0)
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"


##
# Foot-slide weight scan. Each arm changes exactly one number relative to
# HexapodFlatRshapeEnvCfg.
##


@configclass
class HexapodFlatRshapeSlide045EnvCfg(HexapodFlatRshapeEnvCfg):
    """Foot-slide penalty reduced from -0.6 to -0.45."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.feet_slide.weight = -0.45


@configclass
class HexapodFlatRshapeSlide035EnvCfg(HexapodFlatRshapeEnvCfg):
    """Foot-slide penalty reduced from -0.6 to -0.35."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.feet_slide.weight = -0.35


@configclass
class HexapodFlatRshapeSlide025EnvCfg(HexapodFlatRshapeEnvCfg):
    """Foot-slide penalty reduced from -0.6 to -0.25 (lightest arm)."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.feet_slide.weight = -0.25
