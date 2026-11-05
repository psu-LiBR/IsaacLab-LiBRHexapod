# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hexapod goal-reaching env with a binary (contact-bit) leg action space.

Inherits :class:`HexapodGoalEnvCfg` (observations, events, terminations, commands, curriculum) and swaps the
8-dim continuous joint action for:

- **6 leg bits**, one :class:`BinaryJointPositionActionCfg` term per leg. 1 / positive = stance (foot down,
  ``STANCE_POS``), 0 / negative = lift (foot up, ``LIFT_POS``). Both levels come from the reference tripod gait
  ``hexapod-assets/Sim Gaits/tripod_extendedquad_sim.csv``, sign-flipped into the HexapI joint convention
  (leg limits are [-0.0873, +1.9199] rad).
- **2 spine joints, scripted** by a zero-width :class:`SpineSineAction` that plays a fixed analytic traveling
  wave (see the spine-wave block below).

It also rebalances six inherited reward weights for the bit action space and adds a minimum-stance-legs penalty
(see :func:`_rebalance_binary_rewards`).

The action vector is ordered like the hardware bit numbering (bit n = action index n-1)::

    idx 0 = FrontRight   idx 1 = FrontLeft*   idx 2 = MiddleRight
    idx 3 = MiddleLeft*  idx 4 = BackRight    idx 5 = BackLeft

* Inferred from the right/left alternation; still to be confirmed on the hardware.
"""

from __future__ import annotations

from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils.configclass import configclass

from isaaclab_assets.robots.hexapod_actuators import with_servo_model  # isort: skip
from . import hexapod_binary_rewards as binary_rewards
from .hexapod_binary_actions import SpineSineActionCfg
from .hexapod_goal_env_cfg import HexapodGoalEnvCfg, HexapodGoalEnvCfg_PLAY

STANCE_POS = 0.460194236365692
"""Leg joint angle for the stance (foot-down) level, rad. Bit = 1 / positive action."""

LIFT_POS = 1.180398216278
"""Leg joint angle for the lift (foot-up) level, rad. Bit = 0 (bool) / negative action."""

GAIT_PERIOD_S = 1.0
"""Spine wave period, s (one gait cycle)."""

MIN_STANCE_LEGS = 4
"""Minimum number of stance (foot-down) legs per step before the ``min_stance_legs`` penalty applies.

May be 3 or 4 (current). At 4 the 3-legged reference tripods pay a deficit of 1 every step; at 3 they are free.
"""

MIN_STANCE_PENALTY_WEIGHT = -0.2
"""Weight of the ``min_stance_legs`` penalty (negative; a starting value, tune from a short training run)."""

# =========================================================================
# CANONICAL SPINE-WAVE DEFINITION (single source of truth)
# =========================================================================
# Two distinct spine waves exist and are deliberately not unified. Both are a truncated Fourier series with
# w = 2*pi / GAIT_PERIOD_S and t = episode_length_buf * step_dt:
#
#     q(t) = offset + SUM_k [ sin_coef[k]*sin((k+1)*w*t) + cos_coef[k]*cos((k+1)*w*t) ]
#
# WAVE 1 -- the RL env wave (training and -Play). An analytic traveling wave, not fitted to any CSV: same
# amplitude and offset on both joints, phase shifted by a quarter cycle (BackLink leads FrontLink):
#
#     FrontLink_Joint:  q(t) = OFFSET_SPINE - A_SPINE * sin(w*t)
#     BackLink_Joint :  q(t) = OFFSET_SPINE - A_SPINE * cos(w*t)
#
# The minus sign is the HexapI spine joint-sign flip: a positive coefficient walked every policy backward.
# The phase direction and sign are confirmed on the real robot (2026-09-15). A_SPINE is the exact amplitude of
# the open-loop gait generator, deg2rad(70) * 12/16. Must be mirrored verbatim in:
#   - scripts/reinforcement_learning/binary_rl/play_discrete_closeup.py  (_SPINE_* constants)
#   - scripts/sim2real_transfer/sim2real/binary_profile.py                (host-side binary.spine)
#   - scripts/sim2real_transfer/config/deployment.binary.example.yaml     (binary.spine block)
#
# WAVE 2 -- the tripod baseline wave, never played by the RL env. Used by eval_protocol.py (BASE_tripod_csv_bits)
# and play_discrete_closeup.py (--gait_npz tripod), swapped in with SpineSineAction.set_waveform() and then
# restored to Wave 1. Regenerated from the MATLAB gait generator with the same amplitude, the two joints in
# anti-phase (BackLink = -FrontLink) and phase origin -pi/4, which is also the origin of the leg schedule in
# tripod_bit_demos.npz:
#
#     FrontLink_Joint:  q(t) = _TRIPOD_FRONT_SIGN * A_SPINE * sin(w*t - pi/4)
#     BackLink_Joint :  q(t) = -_TRIPOD_FRONT_SIGN * A_SPINE * sin(w*t - pi/4)
#
# tripod_extendedquad_sim.csv has identical spine columns (an export artifact that folded the anti-phase pair
# onto one column), so it is deliberately not used for the spine. _TRIPOD_FRONT_SIGN = -1 was verified in sim
# (eval_protocol, 64 envs x 300 steps): -1 walks the tripod baseline forward (+0.83 m), +1 walks it backward.
# =========================================================================

A_SPINE = 0.9162978573
"""Spine traveling-wave amplitude, rad: ``deg2rad(70) * 12/16`` from the open-loop gait generator."""

OFFSET_SPINE = 0.0
"""Spine traveling-wave constant offset, rad."""

SPINE_SIN_COEF = {"FrontLink_Joint": [-A_SPINE], "BackLink_Joint": [0.0]}
"""Wave 1 sine coefficients ``[a1]``, rad. FrontLink is a pure sine, BackLink a pure cosine."""

SPINE_COS_COEF = {"FrontLink_Joint": [0.0], "BackLink_Joint": [-A_SPINE]}
"""Wave 1 cosine coefficients ``[b1]``, rad. See :data:`SPINE_SIN_COEF`."""

SPINE_OFFSET = {"FrontLink_Joint": OFFSET_SPINE, "BackLink_Joint": OFFSET_SPINE}
"""Wave 1 constant offset ``c0``, rad (same on both joints)."""

_TRIPOD_Q = A_SPINE * 0.7071067811865476  # A_SPINE * sin(pi/4); sin(theta - pi/4) = q * (sin theta - cos theta)
_TRIPOD_FRONT_SIGN = -1.0  # sim-verified global spine sign, see the block above
_TRIPOD_FRONT_SIN = _TRIPOD_FRONT_SIGN * _TRIPOD_Q
_TRIPOD_FRONT_COS = -_TRIPOD_FRONT_SIGN * _TRIPOD_Q

TRIPOD_SPINE_SIN_COEF = {"FrontLink_Joint": [_TRIPOD_FRONT_SIN], "BackLink_Joint": [-_TRIPOD_FRONT_SIN]}
"""Wave 2 (tripod baseline) sine coefficients ``[a1]``, rad; BackLink is the negation of FrontLink."""

TRIPOD_SPINE_COS_COEF = {"FrontLink_Joint": [_TRIPOD_FRONT_COS], "BackLink_Joint": [-_TRIPOD_FRONT_COS]}
"""Wave 2 cosine coefficients ``[b1]``, rad. See :data:`TRIPOD_SPINE_SIN_COEF`."""

TRIPOD_SPINE_OFFSET = {"BackLink_Joint": 0.0, "FrontLink_Joint": 0.0}
"""Wave 2 constant offset ``c0``, rad (zero on both joints)."""


def _leg_bit(joint_name: str) -> BinaryJointPositionActionCfg:
    """One contact bit for one leg joint: 1/positive -> stance, 0/negative -> lift."""
    return BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=[joint_name],
        open_command_expr={joint_name: STANCE_POS},
        close_command_expr={joint_name: LIFT_POS},
    )


@configclass
class HexapodBinaryActionsCfg:
    """6 leg contact bits (RL) + scripted spine sinusoid (no RL slot).

    Term declaration order fixes the policy action layout.  It matches the hardware
    bit numbering: bits 1..6 = FrontRight, FrontLeft, MiddleRight, MiddleLeft,
    BackRight, BackLeft (bits 2 and 4 inferred, pending hardware confirmation).
    """

    front_right = _leg_bit("FrontRight_Joint")
    front_left = _leg_bit("FrontLeft_Joint")
    middle_right = _leg_bit("MiddleRight_Joint")
    middle_left = _leg_bit("MiddleLeft_Joint")
    back_right = _leg_bit("BackRight_Joint")
    back_left = _leg_bit("BackLeft_Joint")

    spine_wave = SpineSineActionCfg(
        asset_name="robot",
        joint_names=["BackLink_Joint", "FrontLink_Joint"],
        sin_coef=SPINE_SIN_COEF,
        cos_coef=SPINE_COS_COEF,
        offset=SPINE_OFFSET,
        period=GAIT_PERIOD_S,
    )


def _rebalance_binary_rewards(rewards) -> None:
    """Rebalance six inherited ``HexapodGoalEnvCfg`` reward weights and add a minimum-stance-legs penalty.

    The six weights are corrected for the contact-bit action space; the ``min_stance_legs`` term is new.

    The goal-env weights were tuned for an 8-dim continuous action. With one stance/lift bit per leg, several
    shaping terms punish a clean walk and the goal progress is too weak to make it net-positive: the reference
    tripod walks +0.91 m with no falls yet scored -0.00116 reward per step (``eval_protocol.py``, 2026-09-09), and
    a 100k-iteration Double-DQN run never learned to walk. The same corrections are applied to the train and play
    variants so their reward definitions cannot drift apart. The evidence predates the DCMotor actuator swap
    (which shrank the torque and acceleration terms) and should be re-measured.

    Args:
        rewards: The already-initialised goal ``RewardsCfg`` instance to mutate in place.
    """
    # feet_slide: removed. The scripted spine wave sways the body while stance feet are planted, so the term
    # measures forced slip, not gait quality (-0.00106/step for the walking tripod and for a static pose alike),
    # and the 6-bit policy has no action that can reduce it.
    rewards.feet_slide = None

    # dof_acc_l2: -2.5e-7 -> -3.0e-8. Acceleration is dominated by the legs snapping between the two bit angles,
    # which the policy cannot soften. Kept slightly non-zero as insurance against wild chatter; action_rate_l2
    # is the main anti-chatter term.
    rewards.dof_acc_l2.weight = -3.0e-8

    # feet_air_time: 0.2 -> 0.0 (term kept for diagnostics). The lift pose barely clears the ground (~0.02 s of
    # air time vs the 0.12 s threshold), so the term punished stepping. Needs more swing clearance first.
    rewards.feet_air_time.weight = 0.0

    # progress: 0.2 -> 1.5. At 0.2 the whole 2 m course was worth +0.4, less than the per-step drag, so walking
    # was never net-positive. The earlier cut guarded against a "throw itself at the goal" exploit of the
    # continuous env, which two-level legs under the 1.4 N.m motor cap cannot produce. If a lunge appears,
    # clamp the closing speed in progress_to_goal rather than re-cutting this weight.
    rewards.progress.weight = 1.5

    # undesired_contacts: -1.0 -> -0.5. The scripted spine sways +-0.92 rad and the policy cannot counter it, so a
    # mistimed contact pattern drags a spine link and early exploration is over-taxed. Halved, not zeroed:
    # base_contact only watches CenterLink, so this is the only penalty on a belly-drag gait.
    rewards.undesired_contacts.weight = -0.5

    # time_penalty: -0.005 -> -0.003. The old value integrated to a -0.125 floor over a 25 s episode that a
    # slow early policy cannot offset. Still negative so faster arrival is preferred once a walk emerges.
    rewards.time_penalty.weight = -0.003

    # min_stance_legs: new term, weight -0.2. Deployed policies produced steps with all six legs commanded to
    # lift. The term charges max(0, MIN_STANCE_LEGS - n_stance) per step. MIN_STANCE_LEGS may be 3 or 4: at 4
    # (current) the two reference tripods (3 legs down) pay a deficit of 1, at 3 they cost zero. RewardManager
    # multiplies by step_dt (0.02 s) and an episode is 25 s = 1250 steps, so a sustained deficit of 1 costs
    # ~25 * 0.2 = 5 per episode, an all-lift episode ~ -20 (-15 at 3), and a one-step glitch
    # ~ -0.004 * deficit; total progress for the 2 m course is only ~ +3.0 (weight 1.5). This is a STARTING value:
    # tune it from a short training run, and do not let it swamp progress.
    rewards.min_stance_legs = RewTerm(
        func=binary_rewards.min_stance_legs_penalty,
        weight=MIN_STANCE_PENALTY_WEIGHT,
        params={"min_legs": MIN_STANCE_LEGS},
    )


@configclass
class HexapodBinaryEnvCfg(HexapodGoalEnvCfg):
    """Goal-reaching task, binary leg action space. Rewards rebalanced for contact bits."""

    def __post_init__(self):
        super().__post_init__()
        self.actions = HexapodBinaryActionsCfg()  # type: ignore[assignment]
        _rebalance_binary_rewards(self.rewards)


@configclass
class HexapodBinaryEnvCfg_PLAY(HexapodGoalEnvCfg_PLAY):
    """Play/eval variant: goal-play settings (16 envs, final distance, no push events)."""

    def __post_init__(self):
        super().__post_init__()
        self.actions = HexapodBinaryActionsCfg()  # type: ignore[assignment]
        _rebalance_binary_rewards(self.rewards)


# ---------------------------------------------------------------------------
# Opt-in "servo-realistic" variant. A fit to the real XL430 leg step response (2026-10-20) prefers the firmware
# P gain, a larger effective inertia and a short delay over the robot-wide long pure delay (leg RMS 3.5 vs 16
# counts). The extra inertia is an effective parameter of unknown origin. The spine fit cannot separate gain
# from delay, so the spine keeps the robot-wide delay. Both groups also draw a battery-voltage scale (real logs
# show ~10.0-10.4 V under load vs the 11.1 V the sim assumes). Not yet validated: leg stiffness 5 (vs 50) must
# still hold the body up and walk.
# ---------------------------------------------------------------------------

SERVO_LEG_STIFFNESS = 5.16
"""Nominal leg stiffness [N m/rad]: XL430 firmware-default P gain 640, as mapped in ``hexapod.py``."""

SERVO_LEG_ARMATURE = 8.6e-3
"""Leg armature [kg m^2]: current 1.3e-3 plus the fitted extra effective inertia (9.9e-3 - 2.6e-3)."""

SERVO_LEG_LAG_RANGE_STEPS = (3, 5)
"""Training-time ``(min, max)`` leg command delay, physics steps (fit: ~4)."""

SERVO_LEG_LAG_PLAY_STEPS = 4
"""Fixed play/eval leg command delay, physics steps (20 ms)."""

SERVO_STIFFNESS_SCALE_RANGE = (0.8, 1.75)
"""Per-env leg stiffness multiplier (about 4-9 N m/rad), covering the fitted 4.4-8.6 loaded range."""

SERVO_VOLTAGE_SCALE_RANGE = (0.85, 1.0)
"""Per-env supply-voltage multiplier on velocity limit, stall torque and effort limit (11.1 V nominal)."""

SERVO_VOLTAGE_SCALE_PLAY = 0.925
"""Fixed play/eval voltage multiplier: the middle of the training range (about 10.3 V)."""


def _servo_group_overrides(
    leg_delay: tuple[int, int],
    stiffness_scale: tuple[float, float],
    voltage_scale: tuple[float, float],
) -> dict[str, dict]:
    """Per-group :class:`ServoDCMotorCfg` overrides for the servo-realistic variant."""
    return {
        "leg_joints": {
            "stiffness": SERVO_LEG_STIFFNESS,
            "armature": SERVO_LEG_ARMATURE,
            "min_delay": leg_delay[0],
            "max_delay": leg_delay[1],
            "stiffness_scale_range": stiffness_scale,
            "voltage_scale_range": voltage_scale,
        },
        # the spine keeps the delay it inherits from the robot asset (random in training, pinned in play)
        "body_joints": {"voltage_scale_range": voltage_scale},
    }


@configclass
class HexapodBinaryServoEnvCfg(HexapodBinaryEnvCfg):
    """Binary-contact goal task with a randomised servo model (leg gain / inertia / delay, battery voltage)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot.actuators = with_servo_model(
            self.scene.robot.actuators,
            _servo_group_overrides(
                SERVO_LEG_LAG_RANGE_STEPS,
                SERVO_STIFFNESS_SCALE_RANGE,
                SERVO_VOLTAGE_SCALE_RANGE,
            ),
        )


@configclass
class HexapodBinaryServoEnvCfg_PLAY(HexapodBinaryEnvCfg_PLAY):
    """Play/eval variant of :class:`HexapodBinaryServoEnvCfg`: fixed nominal gain, delays and voltage."""

    def __post_init__(self):
        super().__post_init__()
        leg = (SERVO_LEG_LAG_PLAY_STEPS, SERVO_LEG_LAG_PLAY_STEPS)
        volt = (SERVO_VOLTAGE_SCALE_PLAY, SERVO_VOLTAGE_SCALE_PLAY)
        self.scene.robot.actuators = with_servo_model(
            self.scene.robot.actuators, _servo_group_overrides(leg, (1.0, 1.0), volt)
        )
