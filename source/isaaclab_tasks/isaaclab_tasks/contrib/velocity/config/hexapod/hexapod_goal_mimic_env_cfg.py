# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hexapod goal-reaching env with an imitation warm-up phase.

Same two-phase scheme as :mod:`hexapod_mimic_env_cfg`, but layered on :class:`HexapodGoalEnvCfg` instead of the
flat velocity-tracking env: the pose command, goal observations, goal rewards/terminations and the distance
curriculum are inherited unchanged. The goal terms (progress, reach bonus, fall penalty, time penalty) stay at full
weight throughout; only the shaping terms in ``_SHAPING_REWARD_TERMS`` are scaled during imitation.

The reference gait, its period, sigma, weight and phase length are shared with the flat mimic env (the
``MIMIC_*`` constants in :mod:`hexapod_mimic_env_cfg`).
"""

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils.configclass import configclass

from isaaclab_assets.robots.hexapod_actuators import with_fixed_actuator_delay  # isort: skip

from .agents.rsl_rl_ppo_goal_cfg import HexapodGoalPPORunnerCfg
from .hexapod_goal_env_cfg import GOAL_DISTANCES, HexapodGoalEnvCfg
from .hexapod_mimic_curriculum import ramp_reward_weight
from .hexapod_mimic_env_cfg import (
    MIMIC_BLEND_ITERATIONS,
    MIMIC_CSV_PATH,
    MIMIC_GAIT_PERIOD,
    MIMIC_ITERATIONS,
    MIMIC_JOINT_SIGMA,
    MIMIC_RL_SCALE,
    MIMIC_SERVO_LAG_S,
    MIMIC_WEIGHT,
)
from .hexapod_mimic_rewards import joint_pos_imitation, reset_joints_to_mimic_reference

_STEPS_PER_ITER: int = HexapodGoalPPORunnerCfg().num_steps_per_env  # 96 for the goal task, not 48
GOAL_MIMIC_DECAY_STEPS: int = MIMIC_ITERATIONS * _STEPS_PER_ITER
"""Env steps after which the curriculum starts blending the imitation phase into the RL phase."""

GOAL_MIMIC_BLEND_END_STEPS: int = GOAL_MIMIC_DECAY_STEPS + MIMIC_BLEND_ITERATIONS * _STEPS_PER_ITER
"""Env steps at which the blend is complete."""

GOAL_MIMIC_ACTION_RATE_WEIGHT: float = -0.03
"""``action_rate_l2`` weight for the whole run (imitation and RL phases). The goal env's -5e-4 costs about 1e-4 per
step against a ~0.03 imitation reward, so it does not discourage per-step action chatter; -0.03 matches the
``HexapodGoalBigStepEnvCfg`` value. Not scaled or restored by the curriculum."""

# Shaping terms scaled during imitation. Goal terms and safety terms (dof_pos_limits, undesired_contacts) stay at
# full weight, as does action_rate_l2 (see GOAL_MIMIC_ACTION_RATE_WEIGHT). Terms that are None in the goal env
# (track_*) are skipped.
_SHAPING_REWARD_TERMS: list[str] = [
    "feet_air_time",
    "feet_slide",
    "dof_torques_l2",
    "dof_acc_l2",
    "ang_vel_xy_l2",
    "ang_vel_z_l2",
    "lin_vel_y_l2",
    "lin_vel_z_l2",
]


@configclass
class HexapodGoalMimicEnvCfg(HexapodGoalEnvCfg):
    """Goal-reaching hexapod environment with an imitation warm-up phase."""

    def __post_init__(self) -> None:
        super().__post_init__()

        self.rewards.action_rate_l2.weight = GOAL_MIMIC_ACTION_RATE_WEIGHT

        self.rewards.joint_pos_imitation = RewTerm(
            func=joint_pos_imitation,
            weight=MIMIC_WEIGHT,
            params={
                "csv_path": MIMIC_CSV_PATH,
                "gait_period": MIMIC_GAIT_PERIOD,
                "csv_col_order": None,  # the CSV and asset.data.joint_pos share the sim DOF order
                "joint_sigma": MIMIC_JOINT_SIGMA,
                "servo_lag_s": MIMIC_SERVO_LAG_S,
            },
        )

        # start every episode on the reference (the reward clock restarts at 0 on each reset)
        self.events.reset_robot_joints = EventTerm(
            func=reset_joints_to_mimic_reference,
            mode="reset",
            params={
                "csv_path": MIMIC_CSV_PATH,
                "gait_period": MIMIC_GAIT_PERIOD,
                "csv_col_order": None,
                "servo_lag_s": MIMIC_SERVO_LAG_S,
            },
        )

        # Long episodes: the first curriculum stage (0.5 m) ends within ~3 gait cycles, so the imitation phase would
        # be mostly start-up. Fix the goal at the final distance for the whole run.
        self.commands.pose_command.ranges.pos_x = (GOAL_DISTANCES[-1], GOAL_DISTANCES[-1])
        self.curriculum.goal_distance = None

        # Ramp imitation to 0 and the shaping terms to full weight over the blend window.
        self.curriculum.mimic_weight_decay = CurrTerm(
            func=ramp_reward_weight,
            params={
                "term_name": "joint_pos_imitation",
                "end_weight": 0.0,
                "start_step": GOAL_MIMIC_DECAY_STEPS,
                "end_step": GOAL_MIMIC_BLEND_END_STEPS,
            },
        )

        for term_name in _SHAPING_REWARD_TERMS:
            term = getattr(self.rewards, term_name, None)
            if term is None:
                continue
            full_weight = term.weight
            term.weight = full_weight * MIMIC_RL_SCALE
            setattr(
                self.curriculum,
                f"restore_{term_name}",
                CurrTerm(
                    func=ramp_reward_weight,
                    params={
                        "term_name": term_name,
                        "end_weight": full_weight,
                        "start_step": GOAL_MIMIC_DECAY_STEPS,
                        "end_step": GOAL_MIMIC_BLEND_END_STEPS,
                    },
                ),
            )


@configclass
class HexapodGoalMimicEnvCfg_PLAY(HexapodGoalMimicEnvCfg):
    """Play/eval variant: same setup as :class:`HexapodGoalEnvCfg_PLAY` (final goal distance, no randomisation)."""

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
