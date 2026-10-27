# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hexapod mimic + RL environment: imitation warm-up, then RL polish, in one run.

- Phase 1 (``MIMIC_ITERATIONS`` PPO iterations): the ``joint_pos_imitation`` reward is active and the RL
  rewards are scaled by ``MIMIC_RL_SCALE``.
- Phase 2: curriculum terms zero the imitation reward and restore the RL weights.

The checkpoint stays compatible with ``Isaac-Velocity-Flat-Hexapod-v0`` (same observation space), so it can be
fine-tuned there with ``isaaclab.bat train --rl_library rsl_rl --task Isaac-Velocity-Flat-Hexapod-v0
--checkpoint <path/to/model.pt>``.

``MIMIC_CSV_PATH`` points at a reference gait CSV (e.g. a ``HexapodRL_Rad_*.csv`` from play). If the file is
missing, training fails with ``FileNotFoundError`` (None selects the built-in tripod gait).
``MIMIC_JOINT_SIGMA`` sets the tracking tolerance (see ``hexapod_mimic_rewards.py``).
"""

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils.configclass import configclass

from isaaclab_assets.robots.hexapod_actuators import with_fixed_actuator_delay  # isort: skip

from .flat_env_cfg import HexapodFlatEnvCfg
from .hexapod_mimic_curriculum import ramp_reward_weight
from .hexapod_mimic_rewards import joint_pos_imitation, reset_joints_to_mimic_reference

MIMIC_CSV_PATH: str | None = (
    "hexapod-assets/Sim Gaits/SMOOTH_vertical_K3_NoNoise_rank1_lleg30-93_amp70_res50_hexapi_sim.csv"
)
"""Reference gait CSV in HexapI signs, columns in the ASSET joint order (see :data:`MIMIC_CSV_COL_ORDER`); None uses
the built-in tripod gait."""

MIMIC_CSV_COL_ORDER: list[int] | None = [1, 0, 2, 3, 6, 7, 4, 5]
"""Column permutation taking the CSV's columns to the sim DOF order (BackLink, FrontLink, MiddleLeft, MiddleRight,
BackLeft, BackRight, FrontLeft, FrontRight). The CSV columns are the asset order that ``playReal.py`` writes straight
to the action vector (FrontLink, BackLink, MiddleLeft, MiddleRight, FrontLeft, FrontRight, BackLeft, BackRight), so
front/back are swapped for the spine and each leg pair. Without it the reference replays backward in sim. Only applies
to ``MIMIC_CSV_PATH``; the built-in gait is already in sim DOF order."""

MIMIC_GAIT_PERIOD: float = 1.0
"""Duration of one reference gait cycle, s."""

MIMIC_ITERATIONS: int = 800
"""PPO iterations of the imitation phase."""

_STEPS_PER_ITER: int = 48  # must match RslRlOnPolicyRunnerCfg.num_steps_per_env
MIMIC_DECAY_STEPS: int = MIMIC_ITERATIONS * _STEPS_PER_ITER
"""Env steps after which the curriculum starts blending the imitation phase into the RL phase."""

MIMIC_BLEND_ITERATIONS: int = 200
"""PPO iterations over which the imitation weight ramps to 0 and the RL weights ramp to full."""

MIMIC_BLEND_END_STEPS: int = MIMIC_DECAY_STEPS + MIMIC_BLEND_ITERATIONS * _STEPS_PER_ITER
"""Env steps at which the blend is complete."""

MIMIC_WEIGHT: float = 3.0
"""Initial weight of the imitation reward."""

MIMIC_JOINT_SIGMA: float = 0.4
"""Per-joint Gaussian width, rad (the reward halves at about +-23 deg)."""

MIMIC_SERVO_LAG_S: list[float] = [0.10, 0.10, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08]
"""Per-joint servo delay compensation, s, in sim DOF order (BackLink, FrontLink, MiddleLeft, MiddleRight, BackLeft,
BackRight, FrontLeft, FrontRight): joint j's imitation reference is evaluated ``MIMIC_SERVO_LAG_S[j]`` in the past.
A plain list so it survives Hydra/OmegaConf serialization.

Derivation: the actuator command delay is 8-12 physics steps for the legs (mean 10 -> 50 ms) and 11-15 for the spine
(mean 13 -> 65 ms) at sim.dt = 5 ms, plus the first-order PD / DC-motor tracking lag. An offline single-joint servo
model (DCMotor torque-speed clip 1.4 N*m / 5.97 rad/s, Kp spine 10 / legs 50, damping 0.35, armature 1.3e-3 plus an
assumed link inertia, delay 13 / 10 steps) showed a joint commanded with ref(t) arriving ~105 ms late (spine) and
~80-95 ms late (legs). In Isaac Sim the open-loop oracle with the former 60 ms scalar lag left ~0.26 rad spine RMS
error, hence 0.10 s spine / 0.08 s legs."""

MIMIC_RANDOM_PHASE: bool = False
"""Default for the reset event's ``random_phase``: start every episode at gait phase 0. The goal mimic env turns it
on for training."""

MIMIC_RL_SCALE: float = 0.25
"""RL reward scale during imitation: 0 = pure imitation, 1 = unscaled."""

# RL terms scaled during imitation. The safety terms (dof_pos_limits, undesired_contacts) stay at full weight.
_RL_REWARD_TERMS: list[str] = [
    "track_lin_vel_xy_exp",
    "track_ang_vel_z_exp",
    "action_rate_l2",
    "dof_torques_l2",
    "dof_acc_l2",
    "ang_vel_xy_l2",
    "lin_vel_z_l2",
]
# "feet_air_time",

@configclass
class HexapodMimicEnvCfg(HexapodFlatEnvCfg):
    """Flat hexapod environment with an imitation warm-up phase."""

    def __post_init__(self) -> None:
        super().__post_init__()

        # real-robot friction calibration (2026-09), restated from the flat base
        self.events.physics_material.params["static_friction_range"] = (0.18, 0.25)
        self.events.physics_material.params["dynamic_friction_range"] = (0.18, 0.25)

        # Params are OmegaConf-compatible primitives; the MotionReference is built lazily on the first call.
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
                "random_phase": MIMIC_RANDOM_PHASE,
            },
        )

        # Ramp imitation to 0 and the RL rewards to full weight over the blend window.
        self.curriculum.mimic_weight_decay = CurrTerm(
            func=ramp_reward_weight,
            params={
                "term_name": "joint_pos_imitation",
                "end_weight": 0.0,
                "start_step": MIMIC_DECAY_STEPS,
                "end_step": MIMIC_BLEND_END_STEPS,
            },
        )

        # Scale the RL rewards down now and restore their full weight over the same window.
        for term_name in _RL_REWARD_TERMS:
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
                        "start_step": MIMIC_DECAY_STEPS,
                        "end_step": MIMIC_BLEND_END_STEPS,
                    },
                ),
            )


@configclass
class HexapodMimicEnvCfg_PLAY(HexapodMimicEnvCfg):
    """Play/eval variant: no randomisation, fixed velocity command, imitation reward kept for visualisation."""

    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        # same servo lag in every rollout (the robot asset draws it randomly for training)
        self.scene.robot.actuators = with_fixed_actuator_delay(self.scene.robot.actuators)

        self.commands.base_velocity.ranges.lin_vel_x = (0.16, 0.16)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.resampling_time_range = (1000.0, 1000.0)

        # eval pins friction to the best real-robot sweep row (mu = 0.21)
        self.events.physics_material.params["static_friction_range"] = (0.21, 0.21)
        self.events.physics_material.params["dynamic_friction_range"] = (0.21, 0.21)

        self.viewer.eye = (-1.0, 0.0, 0.5)
        self.viewer.lookat = (0.0, 0.0, 0.0)
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
