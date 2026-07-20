# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Hexapod mimic + RL environment configuration.

Two-phase training (automatic, single run):
  Phase 1 — Imitation:  joint_pos_imitation reward dominates for MIMIC_ITERATIONS.
  Phase 2 — RL polish:  modify_reward_weight curriculum zeroes the imitation reward
             after MIMIC_DECAY_STEPS, leaving only the standard flat-env RL rewards.

Usage (train):
    isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/train.py ^
        --task Isaac-Velocity-Flat-Hexapod-Mimic-v0 ^
        --num_envs 4096

After the full run the checkpoint is compatible with Isaac-Velocity-Flat-Hexapod-v0
(identical observation space), so you can keep fine-tuning under the flat env:
    isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/train.py ^
        --task Isaac-Velocity-Flat-Hexapod-v0 ^
        --checkpoint <path/to/mimic/model.pt>

Reference CSV:
    Point MIMIC_CSV_PATH at a CSV produced by play.py (HexapodRL_Rad_*.csv) or
    at a hand-crafted gait file.  If the file is absent the built-in sinusoidal
    tripod gait is used automatically — no CSV required to get started.

Tuning guidance:
    MIMIC_ITERATIONS    How many PPO iterations the imitation phase lasts.
                        800 is a reasonable starting point; increase if the gait
                        has not converged before the RL phase begins.
    MIMIC_WEIGHT        Initial weight of the imitation reward.  Higher values
                        (3–5) force closer tracking; lower values (1–2) let RL
                        rewards compete from the start.
    joint_sigma         Gaussian width for imitation reward (see mimic_rewards.py).
                        0.25 rad ≈ ±14° tolerance; tighten for stricter imitation.
"""

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils

from .flat_env_cfg import HexapodFlatEnvCfg, HexapodFlatEnvCfg_PLAY
from .hexapod_mimic_rewards import joint_pos_imitation


# ---------------------------------------------------------------------------
# User-tunable constants
# ---------------------------------------------------------------------------

# Path to a reference gait CSV.  Set to None to use the generated tripod gait.
MIMIC_CSV_PATH: str | None = (
    "hexapod-assets/Sim Gaits/forward3_lleg35_amp65_sim.csv"
)
# Duration of one gait cycle in the reference (seconds).  Overridden by CSV
# if the CSV contains a "time" column spanning exactly one cycle.
MIMIC_GAIT_PERIOD: float = 0.5   # seconds

# Number of PPO iterations devoted to imitation before RL takes over.
# With num_steps_per_env=48 per iteration:
#   common_step_counter = MIMIC_ITERATIONS * 48 after that many iterations.
MIMIC_ITERATIONS: int = 800

# Derived step count for modify_reward_weight curriculum.
# common_step_counter increments by 1 for each env.step() call.
_STEPS_PER_ITER: int = 48   # must match RslRlOnPolicyRunnerCfg.num_steps_per_env
MIMIC_DECAY_STEPS: int = MIMIC_ITERATIONS * _STEPS_PER_ITER  # = 38 400

# Initial weight of the imitation reward term.
MIMIC_WEIGHT: float = 3.0

# Gaussian width for joint position tracking (radians).
MIMIC_JOINT_SIGMA: float = 0.4   # per-joint sigma; ~±23° tolerance before reward halves

# RL reward scale during the imitation phase.
# 0.0 = pure imitation (no RL signal at all during phase 1).
# 0.1 = RL rewards at 10% — keeps a weak locomotion signal alongside imitation.
# 1.0 = no scaling (original behaviour).
MIMIC_RL_SCALE: float = 0.5

# Reward terms to scale down during the imitation phase.
# Safety/limit terms (dof_pos_limits, undesired_contacts) are intentionally
# excluded so the robot doesn't ignore joint limits while imitating.
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


# ---------------------------------------------------------------------------
# Environment configs
# ---------------------------------------------------------------------------

@configclass
class HexapodMimicEnvCfg(HexapodFlatEnvCfg):
    """Flat hexapod environment with an imitation learning warm-up phase.

    Inherits all rewards, observations, and settings from HexapodFlatEnvCfg.
    Adds:
      - joint_pos_imitation reward (weight = MIMIC_WEIGHT initially).
      - modify_reward_weight curriculum that sets the imitation weight to 0.0
        after MIMIC_DECAY_STEPS env steps, ending the imitation phase.
    """

    def __post_init__(self) -> None:
        super().__post_init__()

        # --- Imitation reward ---
        # Params contain only OmegaConf-compatible primitives (str, float, list[int]).
        # MotionReference is built lazily inside the reward function on first call.
        self.rewards.joint_pos_imitation = RewTerm(
            func=joint_pos_imitation,
            weight=MIMIC_WEIGHT,
            params={
                "csv_path": MIMIC_CSV_PATH,
                "gait_period": MIMIC_GAIT_PERIOD,
                "csv_col_order": None,  # CSV and asset.data.joint_pos share Sim DOF order
                "joint_sigma": MIMIC_JOINT_SIGMA,
            },
        )

        # --- Curriculum: decay imitation reward to zero after MIMIC_DECAY_STEPS ---
        self.curriculum.mimic_weight_decay = CurrTerm(
            func=mdp.modify_reward_weight,
            params={
                "term_name": "joint_pos_imitation",
                "weight": 0.0,
                "num_steps": MIMIC_DECAY_STEPS,
            },
        )

        # --- Scale down RL rewards during imitation phase, restore at transition ---
        # Capture each term's full weight, reduce it now, add a curriculum to restore
        # it at MIMIC_DECAY_STEPS so RL takes over cleanly when imitation ends.
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
    """Play/eval variant of the mimic environment.

    Disables randomisation and uses a fixed velocity command for evaluation.
    The imitation reward is kept active so you can visualise how closely the
    policy tracks the reference during playback.
    """

    def __post_init__(self) -> None:
        super().__post_init__()

        # Inherit all play-mode settings from the flat play config.
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None

        self.commands.base_velocity.ranges.lin_vel_x = (0.16, 0.16)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.resampling_time_range = (1000.0, 1000.0)

        self.events.physics_material.params["static_friction_range"] = (0.5, 0.6)
        self.events.physics_material.params["dynamic_friction_range"] = (0.35, 0.45)

        self.sim.render = sim_utils.RenderCfg(rendering_mode="performance")

        self.viewer.eye = (-1.0, 0.0, 0.5)
        self.viewer.lookat = (0.0, 0.0, 0.0)
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
