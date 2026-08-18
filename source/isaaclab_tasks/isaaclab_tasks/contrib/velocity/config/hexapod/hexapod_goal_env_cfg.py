# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hexapod goal-reaching env: learn progressively to reach 5 m as fast as possible.

Inherits HexapodFlatEnvCfg (scene, robot, terrain, actions, and base events) and overrides:
- commands.base_velocity → commands.pose_command (curriculum from 1 m to 5 m)
- observations: swap velocity_commands → pose_command
- rewards: progress, time cost, terminal success bonus, and terminal fall penalty
- terminations: add reach_goal (radius 0.3m)
- episode_length_s = 45.0
"""

from isaaclab.envs.mdp.commands import UniformPose2dCommandCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

from . import hexapod_goal_rewards as goal_rewards
from .flat_env_cfg import HexapodFlatEnvCfg
from .hexapod_goal_curriculum import goal_distance_curriculum
from .hexapod_goal_obs_cfg import HexapodGoalObservationsCfg

# GOAL_DISTANCES = (1.0, 2.0, 3.5, 5.0)
GOAL_DISTANCES = (0.5, 1.0, 1.5, 2.0)

REACH_RADIUS = 0.2
# EPISODE_LENGTH_S = 45.0
EPISODE_LENGTH_S = 25.0
CURRICULUM_SUCCESS_THRESHOLD = 0.7
# Below this windowed success rate, the curriculum steps back down one stage instead of
# holding -- catches post-advance regressions (e.g. catastrophic forgetting of the shorter
# gait) rather than leaving the policy stuck grinding at a stage it can no longer solve.
CURRICULUM_DEMOTION_THRESHOLD = 0.4
# Fraction of scene.num_envs pooled into one curriculum evaluation window, per stage. Easier
# stages advance on less data; the final stage keeps a full window's worth of confidence.
# Tune freely -- these are a starting guess, not a validated schedule.
CURRICULUM_WINDOW_FRACTIONS = (1.0, 1.0, 1.0, 1.0)


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
                pos_x=(GOAL_DISTANCES[0], GOAL_DISTANCES[0]),
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
        self.rewards.lin_vel_z_l2.weight = -1.0e-4
        self.rewards.ang_vel_xy_l2.weight = -0.000001
        # track_ang_vel_z_exp (above) is dropped since there's no yaw command to track, which
        # otherwise leaves yaw rate completely unpenalized -- add an explicit penalty so the
        # robot can still turn to face the goal without being free to spin wildly en route.
        self.rewards.ang_vel_z_l2 = RewTerm(func=goal_rewards.ang_vel_z_l2, weight=-1.0e-2)
        self.rewards.dof_torques_l2.weight = -3.0e-7
        self.rewards.dof_acc_l2.weight = -2.5e-7
        self.rewards.action_rate_l2.weight = -0.0001
        self.rewards.feet_air_time.weight = 0.15
        # The default mdp.feet_air_time gates on ||command[:, :2]|| > 0.1, assuming a velocity
        # command (m/s). Retargeting command_name to "pose_command" would silently turn that
        # into a position-error (m) gate instead, zeroing the reward once the robot got within
        # 10cm of the goal -- inside REACH_RADIUS's buffer, discouraging the final steps
        # needed to actually reach it. The goal task has no "stand still" command state to
        # begin with (reach_goal ends the episode on arrival), so drop the gate entirely
        # instead of retargeting it.
        self.rewards.feet_air_time.func = goal_rewards.feet_air_time_ungated
        del self.rewards.feet_air_time.params["command_name"]
        self.rewards.feet_air_time.params["threshold"] = 0.1  # your new value, in seconds

        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.dof_pos_limits.weight = -1.0

        self.scene.height_scanner = None

        # Isaac Lab applies value * weight * dt. Progress is in m/s, so a weight
        # of 10 integrates to ten reward units per meter advanced.
        # Cut further from 0.3 -- progress_to_goal is raw, uncapped closing velocity, so a
        # leap/dive burst earns reward proportional to however fast it momentarily closes the
        # distance. Lowering the weight doesn't stop that (a clamp on the velocity itself would
        # be the targeted fix), but it shrinks the exploit's payoff relative to the other reward
        # terms (anti-jump shaping, time_penalty, reach_bonus) so it's less worth the risk.
        self.rewards.progress = RewTerm(
            func=goal_rewards.progress_to_goal,
            weight=0.1,
            params={"command_name": "pose_command"},
        )

        # Time-decayed: fires at full weight immediately after reset, decaying linearly to
        # half weight by episode_length_s -- concentrates speed pressure into the single
        # high-salience terminal transition instead of relying only on the (much smaller)
        # per-step time_penalty differential.
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

        # Time cost selects faster policies without making the known approximately
        # 0.14 m/s gait intrinsically negative. Raised from -0.2 -- at -0.2 the full-episode
        # differential between a fast and a slow success was only ~6 reward units against a
        # ~97-unit successful episode (see the reach_bonus decay above for the other half of
        # the speed-pressure fix).
        self.rewards.time_penalty = RewTerm(
            func=goal_rewards.constant_per_step,
            weight=-0.005,
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

        # Evaluate non-overlapping windows of completed episodes. Curriculum
        # computation runs before command reset, so the next episodes sample the
        # newly selected fixed distance.
        # Window sizes scale off scene.num_envs (not a hardcoded constant) so changing
        # num_envs doesn't silently change how much data each curriculum decision pools.
        window_sizes = tuple(
            max(1, round(self.scene.num_envs * fraction)) for fraction in CURRICULUM_WINDOW_FRACTIONS
        )
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
        self.scene.env_spacing = 8.0  # wide enough to see 5m straight-line walk per env
        self.commands.pose_command.ranges.pos_x = (GOAL_DISTANCES[-1], GOAL_DISTANCES[-1])
        self.curriculum.goal_distance = None
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None

        # Camera: fixed world view showing the full 5m goal-reaching path
        self.viewer.eye = (-1.0, -3.0, 1.5)
        self.viewer.lookat = (2.5, 0.0, 0.2)
        self.viewer.origin_type = "world"
