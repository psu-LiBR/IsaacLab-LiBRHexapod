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
``MIMIC_*`` constants in :mod:`hexapod_mimic_env_cfg`). The imitation term tracks the gait phase from the robot's
pose rather than a clock (see ``GOAL_MIMIC_PHASE_TRACKING``).
"""

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

from isaaclab_assets.robots.hexapod_actuators import with_fixed_actuator_delay, with_servo_model  # isort: skip

from . import hexapod_goal_rewards as goal_rewards
from .agents.rsl_rl_ppo_goal_mimic_cfg import HexapodGoalMimicPPORunnerCfg
from .hexapod_binary_env_cfg import (
    SERVO_LEG_LAG_PLAY_STEPS,
    SERVO_LEG_LAG_RANGE_STEPS,
    SERVO_STIFFNESS_SCALE_RANGE,
    SERVO_VOLTAGE_SCALE_PLAY,
    SERVO_VOLTAGE_SCALE_RANGE,
    _servo_group_overrides,
)
from .hexapod_goal_env_cfg import GOAL_DISTANCES, REACH_RADIUS, HexapodGoalEnvCfg
from .hexapod_mimic_curriculum import ramp_reward_weight
from .hexapod_mimic_env_cfg import (
    MIMIC_BLEND_ITERATIONS,
    MIMIC_CSV_COL_ORDER,
    MIMIC_CSV_PATH,
    MIMIC_GAIT_PERIOD,
    MIMIC_ITERATIONS,
    MIMIC_JOINT_SIGMA,
    MIMIC_RL_SCALE,
    MIMIC_SERVO_LAG_S,
    MIMIC_WEIGHT,
)
from .hexapod_mimic_rewards import joint_pos_imitation, joint_pos_imitation_phase, reset_joints_to_mimic_reference

_STEPS_PER_ITER: int = HexapodGoalMimicPPORunnerCfg().num_steps_per_env  # 48, the mimic runner's rollout length
GOAL_MIMIC_DECAY_STEPS: int = MIMIC_ITERATIONS * _STEPS_PER_ITER
"""Env steps after which the curriculum starts blending the imitation phase into the RL phase."""

GOAL_MIMIC_BLEND_END_STEPS: int = GOAL_MIMIC_DECAY_STEPS + MIMIC_BLEND_ITERATIONS * _STEPS_PER_ITER
"""Env steps at which the blend is complete."""

GOAL_MIMIC_ACTION_RATE_WEIGHT: float = -5.0e-3
"""``action_rate_l2`` weight for the whole run (no ramp). At -0.03 the penalty on std-1 exploration noise
(~0.0096/step) outweighs the imitation gain, so the policy collapses its action noise and stands still; at -5e-4
nothing discourages bang-bang chatter. -5e-3 sits between: the true gait's raw action-rate sum is ~0.137 per step, so
it costs the gait only ~1.4e-5 reward/step, while bang-bang chatter (per-step action change ~1.3 per joint) is clearly
penalized. It is no longer ramped up after the blend, so the RL phase is not pushed back toward the slow reference gait
(faster gaits have larger action changes)."""

GOAL_MIMIC_PHASE_TRACKING: bool = True
"""If True, the imitation reward tracks the gait phase from the robot's pose (:func:`joint_pos_imitation_phase`) so
the policy can choose its own gait frequency; False restores the time-locked :func:`joint_pos_imitation` (1 cycle per
``MIMIC_GAIT_PERIOD``)."""

GOAL_MIMIC_PHASE_MIN_STEP_ROWS: int = 2
"""Minimum rows of the 200-row reference table the phase advances per control step. The nominal 1 s gait at 50 Hz is
4 rows/step, so 2-6 rows (with :data:`GOAL_MIMIC_PHASE_MAX_STEP_ROWS`) is 0.5x-1.5x the reference speed."""

GOAL_MIMIC_PHASE_MAX_STEP_ROWS: int = 6
"""Maximum rows of the reference table the phase advances per control step (see
:data:`GOAL_MIMIC_PHASE_MIN_STEP_ROWS`)."""

# Shaping terms scaled during imitation. Goal terms and safety terms (dof_pos_limits, undesired_contacts) stay at
# full weight; action_rate_l2 is held constant (see GOAL_MIMIC_ACTION_RATE_WEIGHT). Terms that are None in
# the goal env (track_*) are skipped.
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

        if GOAL_MIMIC_PHASE_TRACKING:
            self.rewards.joint_pos_imitation = RewTerm(
                func=joint_pos_imitation_phase,
                weight=MIMIC_WEIGHT,
                params={
                    "csv_path": MIMIC_CSV_PATH,
                    "gait_period": MIMIC_GAIT_PERIOD,
                    "csv_col_order": MIMIC_CSV_COL_ORDER,
                    "joint_sigma": MIMIC_JOINT_SIGMA,
                    "min_step_rows": GOAL_MIMIC_PHASE_MIN_STEP_ROWS,
                    "max_step_rows": GOAL_MIMIC_PHASE_MAX_STEP_ROWS,
                },
            )
        else:
            self.rewards.joint_pos_imitation = RewTerm(
                func=joint_pos_imitation,
                weight=MIMIC_WEIGHT,
                params={
                    "csv_path": MIMIC_CSV_PATH,
                    "gait_period": MIMIC_GAIT_PERIOD,
                    "csv_col_order": MIMIC_CSV_COL_ORDER,
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
                "csv_col_order": MIMIC_CSV_COL_ORDER,
                "servo_lag_s": MIMIC_SERVO_LAG_S,
                "random_phase": True,  # random start phase per episode (PLAY pins it back to phase 0)
            },
        )

        # Long episodes: the first curriculum stage (0.5 m) ends within ~3 gait cycles, so the imitation phase would
        # be mostly start-up. Fix the goal at the final distance for the whole run.
        self.commands.pose_command.ranges.pos_x = (GOAL_DISTANCES[-1], GOAL_DISTANCES[-1])
        self.curriculum.goal_distance = None

        # reach_goal would end the episode and forfeit the remaining imitation reward (worth more than the bonus), so
        # it is off until the imitation weight has finished ramping to 0. reach_bonus reads this termination, so it
        # is off too; the robot just keeps walking past the goal until time_out.
        self.terminations.reach_goal = DoneTerm(
            func=goal_rewards.reached_goal_done_after,
            params={
                "command_name": "pose_command",
                "radius": REACH_RADIUS,
                "start_step": GOAL_MIMIC_BLEND_END_STEPS,
            },
        )

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
        # deterministic phase-0 start (training draws a random start phase)
        self.events.reset_robot_joints.params["random_phase"] = False
        # same servo lag in every rollout (the robot asset draws it randomly for training)
        self.scene.robot.actuators = with_fixed_actuator_delay(self.scene.robot.actuators)

        # eval pins friction to the best real-robot sweep row (mu = 0.21)
        self.events.physics_material.params["static_friction_range"] = (0.21, 0.21)
        self.events.physics_material.params["dynamic_friction_range"] = (0.21, 0.21)

        # fixed world camera over the whole path
        self.viewer.eye = (-0.75, -0.5, 0.5)
        self.viewer.lookat = (0.8, 0.0, 0.0)
        self.viewer.origin_type = "world"


# ---------------------------------------------------------------------------
# Servo-realistic variant: the same fitted XL430 leg model (firmware-default P gain, extra effective inertia, short
# delay) and per-env gain / battery-voltage randomisation as the Binary-Servo task, so a continuous policy cannot rely
# on the ~10x stiffer, ~30 Hz-bandwidth leg response of the default actuator. The SERVO_* values are shared with
# hexapod_binary_env_cfg so both variants model the same servo.
# ---------------------------------------------------------------------------


@configclass
class HexapodGoalMimicServoEnvCfg(HexapodGoalMimicEnvCfg):
    """Goal + mimic task with the randomised servo model of :class:`HexapodBinaryServoEnvCfg`."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.robot.actuators = with_servo_model(
            self.scene.robot.actuators,
            _servo_group_overrides(SERVO_LEG_LAG_RANGE_STEPS, SERVO_STIFFNESS_SCALE_RANGE, SERVO_VOLTAGE_SCALE_RANGE),
        )


@configclass
class HexapodGoalMimicServoEnvCfg_PLAY(HexapodGoalMimicEnvCfg_PLAY):
    """Play/eval variant of :class:`HexapodGoalMimicServoEnvCfg`: fixed nominal gain, delays and voltage."""

    def __post_init__(self) -> None:
        super().__post_init__()
        leg = (SERVO_LEG_LAG_PLAY_STEPS, SERVO_LEG_LAG_PLAY_STEPS)
        volt = (SERVO_VOLTAGE_SCALE_PLAY, SERVO_VOLTAGE_SCALE_PLAY)
        self.scene.robot.actuators = with_servo_model(
            self.scene.robot.actuators, _servo_group_overrides(leg, (1.0, 1.0), volt)
        )
