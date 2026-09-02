# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hexapod goal-reaching env with a binary (contact-bit) leg action space.

Action-space swap only: observations, rewards, events, terminations, commands, and the
distance curriculum are all inherited unchanged from :class:`HexapodGoalEnvCfg`.  The
8-dim continuous joint-position action is replaced by:

- **6 leg bits** — one :class:`BinaryJointPositionActionCfg` term per leg.  Convention
  (project contact-bit convention): **1 / positive float = stance (foot down),
  0 (bool) / negative float = lift (foot up)**.  The two target angles per leg come from
  the reference tripod gait CSV (``hexapod-assets/Sim Gaits/tripod_B11BL0_sim.csv``,
  legs snap between exactly two values), sign-flipped into the HexapI joint convention
  (leg limits are [-0.0873, +1.9199] rad; the old-USD CSV values are negative).  See
  ``preset_extraction.md`` (binary env work notes, 2026-08-30) for the derivation.
- **2 spine joints, not RL-controlled** — a zero-width scripted term
  (:class:`SpineSineAction`) plays the sinusoid fitted to the same CSV's spine columns
  (frequency pinned at exactly one cycle per gait period; residual rms ~0.02 rad).

Policy action vector is 6-dim, one bit per leg, ordered to match the hardware bit
numbering (bits 1..6 = action indices 0..5)::

    idx 0 = bit 1 = FrontRight
    idx 1 = bit 2 = FrontLeft      (inferred: right/left alternation)
    idx 2 = bit 3 = MiddleRight
    idx 3 = bit 4 = MiddleLeft     (inferred: right/left alternation)
    idx 4 = bit 5 = BackRight  (hardware "HR", hind-right)
    idx 5 = bit 6 = BackLeft   (hardware "HL", hind-left)

Bits 1/3/5/6 are stated on the hardware side (2026-08-28 slides); bits 2/4 are inferred
from the alternating right/left pattern and still need hardware-side confirmation.
"""

from __future__ import annotations

from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg
from isaaclab.utils.configclass import configclass

from .hexapod_binary_actions import SpineSineActionCfg
from .hexapod_goal_env_cfg import HexapodGoalEnvCfg, HexapodGoalEnvCfg_PLAY

# ---------------------------------------------------------------------------
# Preset values extracted from hexapod-assets/Sim Gaits/tripod_B11BL0_sim.csv
# (2026-08-30; see preset_extraction.md for evidence and uncertainties).
# CSV values are in the pre-HexapI joint convention (stance -0.4602 / lift -1.1804);
# the HexapI USD (commit 6c866ba3) flips the leg-joint sign, so the usable range is
# positive: verified against the live articulation limits [-0.0873, +1.9199] rad.
# ---------------------------------------------------------------------------

STANCE_POS = 0.460194236365692
"""Leg joint angle for the stance (foot-down) level, rad. Bit = 1 / positive action."""

LIFT_POS = 1.180398216278
"""Leg joint angle for the lift (foot-up) level, rad. Bit = 0 (bool) / negative action."""

GAIT_PERIOD_S = 1.0
"""Spine sine period, s (CSV native: 50 rows x 0.02 s = one cycle)."""

# --- 2026-08-31 spine-convention fix -------------------------------------------------
# The tripod CSV is written in the OLD robot's Sim DOF order, which the old USD confirms
# is [0 BackLink, 1 FrontLink, 2 MiddleLeft, 3 MiddleRight, 4 BackLeft, 5 BackRight,
# 6 FrontLeft, 7 FrontRight] (measured: Hexapod_Flattened.usd articulation DOF order).
# HexapI reorders the DOFs to [0 FrontLink, 1 BackLink, 2 ML, 3 MR, 4 FL, 5 FR, 6 BL,
# 7 BR] *and* flips the joint sign convention.  The first version of this file mapped the
# two spine columns by NAME and negated only the legs; that made the body-undulation wave
# travel the wrong way along the body, putting it in anti-phase with the tripod contact
# pattern -- the reference gait then fell over within 0.18 s (unified-eval baseline (4)).
# Measured, same protocol, 64 env x 300 steps: name-mapped spine = +0.41 m but fall_rate
# 1.00 at 0.20 s (a slide, not a walk); the mapping below = +0.68 m, fall_rate 0.00, which
# is 90% of the same CSV replayed natively on the ORIGINAL robot (+0.759 m, fall_rate 0.00).
# See 04_环境与训练/tripod基线诊断进度_20260831.md for the full variant table.
#
# Mapping actually used below: spine columns transfer by DOF INDEX (so the CSV's BackLink
# column drives FrontLink_Joint and vice-versa) and then take the global sign flip, i.e.
#   BackLink_Joint  <- -(CSV FrontLink column fit)
#   FrontLink_Joint <- -(CSV BackLink  column fit)
# implemented as amplitude kept positive with pi added to the phase.
# Written as (swapped fit) x (-1): amplitude and offset carry the sign flip, the phase is
# the unmodified fit of the *other* column.  This is bit-for-bit the arithmetic that the
# verified run used (diag_tripod.py --mode binary --bin_spine_swap --bin_spine_gain -1.0).
SPINE_AMPLITUDE = {"BackLink_Joint": -0.850309, "FrontLink_Joint": -0.844198}
"""Sine amplitude per spine joint, rad; negative = the HexapI sign flip (|A| ~48.7/48.4 deg)."""

SPINE_PHASE = {"BackLink_Joint": -0.732694, "FrontLink_Joint": 0.462744}
"""Sine phase per spine joint, rad (q = C + A sin(2*pi*t/T + phi)); fit of the swapped column."""

SPINE_OFFSET = {"BackLink_Joint": 0.012033, "FrontLink_Joint": -0.006512}
"""Constant offset per spine joint, rad (sign-flipped with the wave)."""

# Pre-fix values, kept for reference (do not use):
#   AMPLITUDE {"BackLink_Joint": 0.844198, "FrontLink_Joint": 0.850309}
#   PHASE     {"BackLink_Joint": 0.462744, "FrontLink_Joint": -0.732694}
#   OFFSET    {"BackLink_Joint": 0.006512, "FrontLink_Joint": -0.012033}


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
        amplitude=SPINE_AMPLITUDE,
        phase=SPINE_PHASE,
        offset=SPINE_OFFSET,
        period=GAIT_PERIOD_S,
    )


@configclass
class HexapodBinaryEnvCfg(HexapodGoalEnvCfg):
    """Goal-reaching task, binary leg action space. Everything else inherited."""

    def __post_init__(self):
        super().__post_init__()
        self.actions = HexapodBinaryActionsCfg()  # type: ignore[assignment]


@configclass
class HexapodBinaryEnvCfg_PLAY(HexapodGoalEnvCfg_PLAY):
    """Play/eval variant: goal-play settings (16 envs, final distance, no push events)."""

    def __post_init__(self):
        super().__post_init__()
        self.actions = HexapodBinaryActionsCfg()  # type: ignore[assignment]
