# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward-shaping configuration for the hexapod flat-locomotion task.

Baseline (`flat_env_cfg.py`) reaches the commanded velocity with short, rapid steps:
`track_lin_vel_xy_exp` at std 0.1 is near its maximum for any gait that holds the
commanded speed, so stride length is unconstrained, and swing phases stay below the
0.1 s `feet_air_time` threshold, which leaves that term slightly negative.

This config makes long swing phases worth more than cheap speed:

- looser velocity tracking (std 0.15) so a longer stride is not penalized,
- higher `feet_air_time` weight and threshold, paying for airborne time per leg,
- a `feet_slide` penalty so the reward cannot be collected by sliding a planted foot,
- higher ground friction, since the baseline range let the feet slip,
- a slightly stronger `action_rate_l2` penalty against high-frequency jitter.

`HexapodFlatRshapeSlide045/035/025EnvCfg` change only the `feet_slide` weight.
"""

import math

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp  # for feet_slide

from .hexapod_obs_cfg import HexapodFlatObservationsCfg
from .hexapod_rewards import track_ang_vel_z_exp_ema
from .rough_env_cfg import HexapodRoughEnvCfg


@configclass
class HexapodFlatRshapeEnvCfg(HexapodRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # os.environ["WANDB_DISABLE_SYMLINKS"] = "true"
        # override body values
        self.events.add_base_mass.params["asset_cfg"].body_names = ["CenterLink", "BackLink", "FrontLink"]
        self.events.add_base_mass.params["mass_distribution_params"] = (
            0.0,
            0.0,
        )  # just added this to see if (-5.0, 5.0) is the problem
        self.events.base_com.params["asset_cfg"].body_names = "CenterLink"
        self.events.base_com.params["com_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
        }
        # Baseline (0.5, 0.6) / (0.35, 0.45) let the feet slide instead of pushing.
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

        # reset position:
        self.events.reset_base.params = {
            "pose_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "yaw": (0.0, 0.0),  # ~±6°
                # include "z" here if your event supports it; e.g., ("z": (0.18, 0.22))
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (00.0, 0.0),
            },
        }

        # override velocity ranges -> based on actual velocities
        # max velocity of the physics gaits is 0.14 m/s - so these values should be in that range
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_x = (0.2, 0.2)  # 0.16 m/s for six legged gaits
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.debug_vis = False

        # override rewards
        self.rewards.track_lin_vel_xy_exp.weight = 1.0
        # std 0.1 -> 0.15: a sharper term is satisfied by any gait at the commanded
        # speed, which leaves no gradient toward longer strides.
        self.rewards.track_lin_vel_xy_exp.params["std"] = math.sqrt(0.25) * 0.3

        # Penalise the cumulative episode-average yaw rate rather than the instantaneous value.
        # No gait-cycle timescale is hardcoded: cumsum / steps_elapsed converges naturally.
        # Sinusoidal undulation (zero net drift) -> mean stays near zero -> full reward.
        # Sustained turning -> mean shifts away from command -> penalty.
        self.rewards.track_ang_vel_z_exp.weight = 0.45
        self.rewards.track_ang_vel_z_exp.func = track_ang_vel_z_exp_ema
        self.rewards.track_ang_vel_z_exp.params = {
            "std": math.sqrt(0.25) * 0.15,
            "command_name": "base_velocity",
            "alpha": 0.98,  # ~33-step effective window (~0.66 s at 0.02 s dt); tune if undulation still penalised
        }

        # Penalties

        # added these

        # standard
        self.rewards.dof_torques_l2.weight = -5.0e-5

        # Fixed-threshold air time: reward each leg for staying airborne >= threshold seconds.
        # Per-leg duty-cycle version produced only 0.027 sum (too weak to shape gait) -- reverted.
        # Pay for airborne time per leg. The friction and feet_slide settings above
        # keep this from being collected by lifting legs without moving forward.
        self.rewards.feet_air_time.weight = 2.5
        self.rewards.feet_air_time.params["threshold"] = 0.16

        self.rewards.dof_pos_limits.weight = -1.0
        self.rewards.dof_acc_l2.weight = -8.5e-14
        self.rewards.ang_vel_xy_l2.weight = 0.0  # -0.00000001
        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.lin_vel_z_l2.weight = -0.00000001  # -0.0001
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

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        # no height scan
        self.scene.height_scanner = None

        # Asymmetric actor-critic observations: actor uses only hardware-observable quantities,
        # critic additionally sees ground-truth base_lin_vel during training.
        self.observations = HexapodFlatObservationsCfg()  # type: ignore[assignment]

        # no terrain curriculum
        self.curriculum.terrain_levels = None

        # camera settings
        self.viewer.eye = (0.5, 0.0, 1.0)
        self.viewer.lookat = (0.5, 0.0, 0.0)
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"


class HexapodFlatRshapeEnvCfg_PLAY(HexapodFlatRshapeEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None

        # Play on the max velocity
        self.commands.base_velocity.ranges.lin_vel_x = (0.16, 0.16)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)

        self.commands.base_velocity.resampling_time_range = (1000.0, 1000.0)

        self.events.physics_material.params["static_friction_range"] = (0.9, 1.0)
        self.events.physics_material.params["dynamic_friction_range"] = (0.7, 0.8)

        # camera settings -- follow robot from behind and above
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
