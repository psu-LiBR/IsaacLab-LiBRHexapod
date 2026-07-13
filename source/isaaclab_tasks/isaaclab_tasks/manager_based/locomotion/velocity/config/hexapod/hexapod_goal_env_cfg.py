# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hexapod goal-reaching env: learn distance-proportional forward progress.

Inherits HexapodFlatEnvCfg (scene, robot, terrain, actions, and base events) and overrides:
- commands.base_velocity → commands.pose_command (curriculum from 1 m to 5 m)
- observations: swap velocity_commands → pose_command
- rewards: progress, time cost, terminal success bonus, and terminal fall penalty
- terminations: add reach_goal (radius 0.3m)
- episode_length_s = 45.0
"""

from isaaclab.envs.mdp.commands import UniformPose2dCommandCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from . import hexapod_goal_rewards as goal_rewards
from .flat_env_cfg import HexapodFlatEnvCfg
from .hexapod_goal_obs_cfg import HexapodGoalObservationsCfg


GOAL_DISTANCE_RANGE = (1.0, 10.0)
REACH_RADIUS = 0.3
EPISODE_LENGTH_S = 45.0


@configclass
class HexapodGoalEnvCfg(HexapodFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.episode_length_s = EPISODE_LENGTH_S

        # ===== Command: fixed pose at the active curriculum distance =====
        # UniformPose2dCommand samples (x, y, heading) per resampling interval; we pin
        # ranges to constants so each episode uses the active curriculum stage.
        # Resampling interval == episode length → never resamples mid-episode.
        self.commands.base_velocity = None
        self.commands.pose_command = UniformPose2dCommandCfg(
            asset_name="robot",
            simple_heading=False,
            resampling_time_range=(EPISODE_LENGTH_S, EPISODE_LENGTH_S),
            debug_vis=False,
            ranges=UniformPose2dCommandCfg.Ranges(
                pos_x=GOAL_DISTANCE_RANGE,
                pos_y=(0.0, 0.0),
                heading=(0.0, 0.0),
            ),
        )

        # ===== Observations: swap velocity_commands → pose_command =====
        self.observations = HexapodGoalObservationsCfg()  # type: ignore[assignment]

        # ===== Rewards =====
        # Drop velocity tracking (no longer the task)
        self.rewards.track_lin_vel_xy_exp = None
        self.rewards.track_ang_vel_z_exp = None

        # Anti-jump shaping: preserve the goal-reaching objective, but make
        # airborne/bounding solutions expensive.  These are intentionally weaker
        # than a full flat-walking template so the policy can still find a fast
        # locomotion style.
        self.rewards.lin_vel_z_l2.weight = -3.0
        self.rewards.ang_vel_xy_l2.weight = -0.3
        self.rewards.dof_torques_l2.weight = -2.0e-4
        self.rewards.dof_acc_l2.weight = -2.5e-7
        self.rewards.action_rate_l2.weight = -5.0e-2
        self.rewards.feet_air_time = None
        self.rewards.undesired_contacts = None
        self.rewards.flat_orientation_l2.weight = -2.0
        self.rewards.dof_pos_limits.weight = -1.0

        # Isaac Lab applies value * weight * dt. Progress is in m/s, so a weight
        # of 10 integrates to ten reward units per meter advanced.
        self.rewards.progress = RewTerm(
            func=goal_rewards.progress_to_goal,
            weight=10.0,
            params={"command_name": "pose_command"},
        )

        # Keep total return approximately proportional to forward distance.
        self.rewards.reach_bonus = None

        # At dt=0.02, this weight produces -25 fall reward.
        self.rewards.fall_penalty = RewTerm(
            func=goal_rewards.termination_signal,
            weight=-1250.0,
            params={"termination_name": "base_contact"},
        )

        # A modest time cost selects faster policies without making the known
        # approximately 0.14 m/s gait intrinsically negative.
        self.rewards.time_penalty = RewTerm(
            func=goal_rewards.constant_per_step,
            weight=-0.2,
            params={},
        )

        # NOTE: removed position_command_error_tanh terms.  Being close to the goal
        # for many steps accumulates large reward, which incentivises *slow* approach
        # (longer episode = more close-time = more reward).  Pure progress + sparse
        # bonus + time penalty avoids this pathology.

        # ===== Termination: success on reaching goal =====
        self.terminations.reach_goal = DoneTerm(
            func=goal_rewards.reached_goal_done,
            params={"command_name": "pose_command", "radius": REACH_RADIUS},
        )

        self.curriculum.goal_distance = None


@configclass
class HexapodGoalEnvCfg_PLAY(HexapodGoalEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 16
        self.scene.env_spacing = 12.0  # wide enough to see 10m straight-line walk per env
        self.commands.pose_command.ranges.pos_x = (GOAL_DISTANCE_RANGE[-1], GOAL_DISTANCE_RANGE[-1])
        self.curriculum.goal_distance = None
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None

        # Camera: fixed world view showing the full 10m goal-reaching path
        self.viewer.eye = (-1.0, -6.0, 3.0)
        self.viewer.lookat = (5.0, 0.0, 0.3)
        self.viewer.origin_type = "world"
