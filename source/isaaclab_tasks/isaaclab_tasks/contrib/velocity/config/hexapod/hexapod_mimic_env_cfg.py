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
missing, a built-in sinusoidal tripod gait is used. ``MIMIC_JOINT_SIGMA`` sets the tracking tolerance (see
``hexapod_mimic_rewards.py``).
"""

import isaaclab.envs.mdp as mdp
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils.configclass import configclass

from isaaclab_assets.robots.hexapod_actuators import with_fixed_actuator_delay  # isort: skip

from .flat_env_cfg import HexapodFlatEnvCfg
from .hexapod_mimic_rewards import joint_pos_imitation

MIMIC_CSV_PATH: str | None = "hexapod-assets/Sim Gaits/forward3_lleg35_amp65_sim.csv"
"""Reference gait CSV; None uses the built-in tripod gait."""

MIMIC_GAIT_PERIOD: float = 0.5
"""Duration of one reference gait cycle, s."""

MIMIC_ITERATIONS: int = 800
"""PPO iterations of the imitation phase."""

_STEPS_PER_ITER: int = 48  # must match RslRlOnPolicyRunnerCfg.num_steps_per_env
MIMIC_DECAY_STEPS: int = MIMIC_ITERATIONS * _STEPS_PER_ITER
"""Env steps after which the curriculum ends the imitation phase."""

MIMIC_WEIGHT: float = 3.0
"""Initial weight of the imitation reward."""

MIMIC_JOINT_SIGMA: float = 0.4
"""Per-joint Gaussian width, rad (the reward halves at about +-23 deg)."""

MIMIC_RL_SCALE: float = 0.5
"""RL reward scale during imitation: 0 = pure imitation, 1 = unscaled."""

# RL terms scaled during imitation. The safety terms (dof_pos_limits, undesired_contacts) stay at full weight.
_RL_REWARD_TERMS: list[str] = [
    "track_lin_vel_xy_exp",
    "track_ang_vel_z_exp",
    "feet_air_time",
    "action_rate_l2",
    "dof_torques_l2",
    "dof_acc_l2",
    "ang_vel_xy_l2",
    "lin_vel_z_l2",
]


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
                "csv_col_order": None,  # the CSV and asset.data.joint_pos share the sim DOF order
                "joint_sigma": MIMIC_JOINT_SIGMA,
            },
        )

        self.curriculum.mimic_weight_decay = CurrTerm(
            func=mdp.modify_reward_weight,
            params={
                "term_name": "joint_pos_imitation",
                "weight": 0.0,
                "num_steps": MIMIC_DECAY_STEPS,
            },
        )

        # Scale the RL rewards down now and restore their full weight when imitation ends.
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
                    func=mdp.modify_reward_weight,
                    params={
                        "term_name": term_name,
                        "weight": full_weight,
                        "num_steps": MIMIC_DECAY_STEPS,
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
