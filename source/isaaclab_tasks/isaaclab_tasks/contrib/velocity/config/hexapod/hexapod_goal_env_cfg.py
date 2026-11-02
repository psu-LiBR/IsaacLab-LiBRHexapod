# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hexapod goal-reaching env: reach a fixed point straight ahead as fast as possible.

Inherits :class:`HexapodFlatEnvCfg` and overrides:

- commands: ``base_velocity`` -> ``pose_command`` at a curriculum distance (``GOAL_DISTANCES``)
- observations: ``velocity_commands`` -> ``pose_command``
- rewards: goal progress, time cost, time-decayed reach bonus, fall penalty
- terminations: adds ``reach_goal`` (radius ``REACH_RADIUS``)
"""

from isaaclab.envs.mdp.commands import UniformPose2dCommandCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

import isaaclab_tasks.core.velocity.mdp as mdp  # for feet_slide

from isaaclab_assets.robots.hexapod_actuators import with_fixed_actuator_delay  # isort: skip

from . import hexapod_goal_rewards as goal_rewards
from .flat_env_cfg import HexapodFlatEnvCfg
from .hexapod_goal_curriculum import goal_distance_curriculum
from .hexapod_goal_obs_cfg import HexapodGoalObservationsCfg

GOAL_DISTANCES = (0.5, 1.0, 1.5, 2.0)  # curriculum stages, m
REACH_RADIUS = 0.2  # m
EPISODE_LENGTH_S = 25.0
CURRICULUM_SUCCESS_THRESHOLD = 0.7  # windowed success rate needed to advance a stage
CURRICULUM_DEMOTION_THRESHOLD = 0.4  # below this rate the curriculum steps back one stage
CURRICULUM_WINDOW_FRACTIONS = (1.0, 1.0, 1.0, 1.0)  # per stage: fraction of scene.num_envs pooled per window


@configclass
class HexapodGoalEnvCfg(HexapodFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.episode_length_s = EPISODE_LENGTH_S

        # real-robot friction calibration (2026-09), restated from the flat base
        self.events.physics_material.params["static_friction_range"] = (0.18, 0.25)
        self.events.physics_material.params["dynamic_friction_range"] = (0.18, 0.25)

        # Command: a fixed pose at the active curriculum distance. Resampling == episode length, so it never
        # changes mid-episode; the curriculum rewrites pos_x.
        self.commands.base_velocity = None
        self.commands.pose_command = UniformPose2dCommandCfg(
            asset_name="robot",
            simple_heading=False,
            resampling_time_range=(EPISODE_LENGTH_S, EPISODE_LENGTH_S),
            debug_vis=False,
            ranges=UniformPose2dCommandCfg.Ranges(
                pos_x=(GOAL_DISTANCES[0], GOAL_DISTANCES[0]),
                pos_y=(0.0, 0.0),
                heading=(0.0, 0.0),
            ),
        )

        self.observations = HexapodGoalObservationsCfg()  # type: ignore[assignment]

        # Rewards. Velocity tracking is no longer the task.
        self.rewards.track_lin_vel_xy_exp = None
        self.rewards.track_ang_vel_z_exp = None

        # Anti-jump shaping, deliberately weaker than the flat-walking template so a fast gait can still emerge.
        self.rewards.lin_vel_z_l2.weight = -1.0e-4
        self.rewards.ang_vel_xy_l2.weight = -0.000001
        # Yaw rate would otherwise be unpenalized now that track_ang_vel_z_exp is gone.
        self.rewards.ang_vel_z_l2 = RewTerm(func=goal_rewards.ang_vel_z_l2, weight=-1.0e-4)
        # Planted-foot slide penalty. Kept far below the Rshape variant's -0.6: that weight was validated at
        # much higher friction, while the calibrated 0.18-0.25 band forces more slip. Raise only if play videos
        # show excessive sliding.
        self.rewards.feet_slide = RewTerm(
            func=mdp.feet_slide,
            weight=-0.1,
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
        # Body-level sideways drift, whatever its cause (feet_slide only covers planted feet).
        self.rewards.lin_vel_y_l2 = RewTerm(func=goal_rewards.lin_vel_y_l2, weight=-5.0e-4)
        self.rewards.dof_torques_l2.weight = -3.0e-4
        self.rewards.dof_acc_l2.weight = -2.5e-8
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.feet_air_time.weight = 0.15
        # The default feet_air_time gates on a velocity command; with pose_command it would become a position
        # gate that zeroes the reward near the goal. The goal task has no stand-still state, so drop the gate.
        self.rewards.feet_air_time.func = goal_rewards.feet_air_time_ungated
        del self.rewards.feet_air_time.params["command_name"]
        self.rewards.feet_air_time.params["threshold"] = 0.12  # s

        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.dof_pos_limits.weight = -1.0

        self.scene.height_scanner = None

        # Progress is a closing speed in m/s and Isaac Lab applies weight * dt, so the weight is the reward per
        # meter advanced. It is uncapped, so a small weight limits the payoff of a leap at the goal.
        self.rewards.progress = RewTerm(
            func=goal_rewards.progress_to_goal,
            weight=0.7,
            params={"command_name": "pose_command"},
        )

        # Fires once on arrival: full weight right after reset, decaying linearly to half by episode end.
        self.rewards.reach_bonus = RewTerm(
            func=goal_rewards.time_decayed_termination_signal,
            weight=2.0,
            params={"termination_name": "reach_goal", "episode_length_s": EPISODE_LENGTH_S, "min_fraction": 0.5},
        )
        self.rewards.fall_penalty = RewTerm(
            func=goal_rewards.termination_signal,
            weight=-1.0,
            params={"termination_name": "base_contact"},
        )

        # Small per-step cost that prefers faster arrival.
        self.rewards.time_penalty = RewTerm(
            func=goal_rewards.constant_per_step,
            weight=-0.005,
            params={},
        )

        # No proximity reward (e.g. position_command_error_tanh): lingering near the goal would pay, which
        # rewards a slow approach.

        self.terminations.reach_goal = DoneTerm(
            func=goal_rewards.reached_goal_done,
            params={"command_name": "pose_command", "radius": REACH_RADIUS},
        )

        # Non-overlapping windows of completed episodes; window sizes scale with num_envs.
        window_sizes = tuple(max(1, round(self.scene.num_envs * fraction)) for fraction in CURRICULUM_WINDOW_FRACTIONS)
        self.curriculum.goal_distance = CurrTerm(
            func=goal_distance_curriculum,
            params={
                "command_name": "pose_command",
                "distances": GOAL_DISTANCES,
                "success_threshold": CURRICULUM_SUCCESS_THRESHOLD,
                "window_sizes": window_sizes,
                "demotion_threshold": CURRICULUM_DEMOTION_THRESHOLD,
            },
        )


@configclass
class HexapodGoalEnvCfg_PLAY(HexapodGoalEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 16
        self.scene.env_spacing = 8.0
        self.commands.pose_command.ranges.pos_x = (GOAL_DISTANCES[-1], GOAL_DISTANCES[-1])
        self.curriculum.goal_distance = None
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        # same servo lag in every rollout (the robot asset draws it randomly for training)
        self.scene.robot.actuators = with_fixed_actuator_delay(self.scene.robot.actuators)

        # eval pins friction to the best real-robot sweep row (mu = 0.21)
        self.events.physics_material.params["static_friction_range"] = (0.21, 0.21)
        self.events.physics_material.params["dynamic_friction_range"] = (0.21, 0.21)

        # fixed world camera over the whole path
        self.viewer.eye = (-1.0, -3.0, 1.5)
        self.viewer.lookat = (2.5, 0.0, 0.2)
        self.viewer.origin_type = "world"
