# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hexapod goal-reaching env with a binary (contact-bit) leg action space.

Action-space swap plus a small reward rebalance: observations, events, terminations,
commands, and the distance curriculum are inherited unchanged from
:class:`HexapodGoalEnvCfg`; the inherited reward weights get six corrections for the
contact-bit action space (see :func:`_rebalance_binary_rewards` for the per-term reason
and the measured evidence).  The 8-dim continuous joint-position action is
replaced by:

- **6 leg bits** — one :class:`BinaryJointPositionActionCfg` term per leg.  Convention
  (project contact-bit convention): **1 / positive float = stance (foot down),
  0 (bool) / negative float = lift (foot up)**.  The two target angles per leg come from
  the reference tripod gait CSV (``hexapod-assets/Sim Gaits/tripod_extendedquad_sim.csv``,
  legs snap between exactly two values -- the same ``STANCE_POS`` / ``LIFT_POS`` levels as
  the earlier ``tripod_B11BL0`` gait), sign-flipped into the HexapI joint convention
  (leg limits are [-0.0873, +1.9199] rad; the old-USD CSV values are negative).
- **2 spine joints, not RL-controlled** — a zero-width scripted term
  (:class:`SpineSineAction`) plays a *fixed analytic traveling body wave* (period = one
  gait cycle): ``FrontLink`` a pure sine, ``BackLink`` the same sine shifted +90 deg
  (i.e. a cosine), same amplitude/offset on both — a quarter-cycle wave travelling down
  the body.  This is **not** fitted to any CSV.  A separate *anti-phase* body wave
  (Wave 2: ``FrontLink = +-A_SPINE*sin(w*t - pi/4)``, ``BackLink`` the negation),
  regenerated analytically from the MATLAB gait generator, is used only by
  ``eval_protocol.py`` / ``play_discrete_closeup.py`` to score / visualise the reference
  tripod gait.  See the "CANONICAL SPINE-WAVE DEFINITION" block below.

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
# Leg contact-bit levels -- extracted from
# hexapod-assets/Sim Gaits/tripod_extendedquad_sim.csv (2026-09-10).  CSV: 50 rows
# x 0.02 s = one 1.0 s gait period, 8 columns in the OLD-USD Sim DOF order
# [0 BackLink, 1 FrontLink, 2 ML, 3 MR, 4 BL, 5 BR, 6 FL, 7 FR].  Leg columns snap
# between exactly STANCE_POS / LIFT_POS in the pre-HexapI joint convention (both
# negative); the HexapI USD (commit 6c866ba3) flips the leg-joint sign, so the usable
# range is positive -- verified against the live articulation limits [-0.0873, +1.9199]
# rad.  (These two leg levels are identical to the earlier tripod_B11BL0_sim.csv gait,
# so STANCE_POS / LIFT_POS are unchanged.)
# ---------------------------------------------------------------------------

STANCE_POS = 0.460194236365692
"""Leg joint angle for the stance (foot-down) level, rad. Bit = 1 / positive action."""

LIFT_POS = 1.180398216278
"""Leg joint angle for the lift (foot-up) level, rad. Bit = 0 (bool) / negative action."""

GAIT_PERIOD_S = 1.0
"""Spine wave period, s (CSV native: 50 rows x 0.02 s = one cycle)."""

# =========================================================================
# CANONICAL SPINE-WAVE DEFINITION  (single source of truth -- do not fork)
# =========================================================================
# There are TWO DISTINCT spine (body-bending) waves in this project.  They are
# deliberately NOT unified -- keep the distinction explicit everywhere.
#
# Common representation: truncated Fourier series, period = GAIT_PERIOD_S = 1.0 s,
# w = 2*pi / GAIT_PERIOD_S, t = episode_length_buf * step_dt, k = 0-based harmonic index:
#
#     q(t) = offset + SUM_k [ sin_coef[k]*sin((k+1)*w*t) + cos_coef[k]*cos((k+1)*w*t) ]
#
# -------------------------------------------------------------------------
# WAVE 1 -- the RL ENV wave   (SPINE_SIN_COEF / SPINE_COS_COEF / SPINE_OFFSET)
# -------------------------------------------------------------------------
# What HexapodBinaryActionsCfg.spine_wave / SpineSineAction plays during
# Isaac-Goal-Flat-Hexapod-Binary-v0 training AND its -Play-v0 variant.
#
# A FIXED ANALYTIC traveling body wave -- NOT fitted to any CSV.  Same amplitude, same
# offset, same period on both spine joints; ONLY the phase differs, by exactly pi/2 (a
# quarter cycle) -- a wave that travels down the body:
#
#     FrontLink_Joint:  q(t) = OFFSET_SPINE - A_SPINE * sin(w*t)
#     BackLink_Joint :  q(t) = OFFSET_SPINE - A_SPINE * sin(w*t + pi/2)
#                            = OFFSET_SPINE - A_SPINE * cos(w*t)
#
# In the sin_coef / cos_coef layout (single harmonic only):
#     FrontLink_Joint:  sin_coef = [-A_SPINE], cos_coef = [0.0]
#     BackLink_Joint :  sin_coef = [0.0],      cos_coef = [-A_SPINE]
#     both:             offset   = OFFSET_SPINE
#
# The -A_SPINE sign is the HexapI global spine-joint-sign flip -- see flag (b).
#
# UNVERIFIED -- FLAG FOR THE USER / an Isaac Sim check:
#   (a) Direction of the 90 deg shift: the user specified "sin + 90 deg", so BackLink
#       gets +pi/2 (implemented as a cosine).  Whether BackLink should LEAD or LAG
#       FrontLink for a forward-traveling wave is a one-line flip (swap which joint gets
#       sin vs cos).  Kept as-is per the user; the next thing to try if a retrained
#       policy still will not walk forward after the (b) sign fix.
#   (b) HexapI spine joint-sign convention for WAVE 1: CORRECTED 2026-09-10 from +A_SPINE
#       to -A_SPINE.  Three independent lines of evidence: the pre-Fourier committed wave
#       carried the flip as a negative fitted amplitude (SPINE_AMPLITUDE ~ -0.845 rad,
#       "negative = the HexapI sign flip"), dropped in the analytic rewrite; WAVE 2's
#       per-joint signs were checked in sim (FrontLink = -A*sin(w*t - pi/4) walks the
#       tripod baseline forward +0.83 m, the positive front-joint coefficient walks it
#       backward); and a re-eval of pipeline_20260909_230028 on the +A_SPINE wave walked
#       every learned policy AND the near-passive baselines backward, magnitudes mirrored.
#       WAVE 1 and WAVE 2 are still independent gaits -- this only aligns their global
#       joint-sign convention, not their phase structure.  Existing binary checkpoints
#       must be retrained (they predate this and the Fourier-form spine change alike);
#       confirm the fix with a short retrain that nets positive BL/cycle.
#   (c) OFFSET_SPINE numeric value -- chosen constant, unverified in sim.
#       A_SPINE is the EXACT analytic amplitude from the open-loop gait generator:
#       deg2rad(70) * 12/16 = deg2rad(52.5) = 0.9162978572970227 rad (amp=70 deg,
#       amp_desired=12) -- not a tunable guess.
#
# Downstream files that MUST mirror WAVE 1 verbatim:
#   - scripts/reinforcement_learning/binary_rl/play_discrete_closeup.py  (_SPINE_* consts)
#   - scripts/sim2real_transfer/sim2real/binary_profile.py               (binary.spine, host-side)
#   - scripts/sim2real_transfer/config/deployment.binary.example.yaml    (binary.spine block)
#
# -------------------------------------------------------------------------
# WAVE 2 -- the TRIPOD BASELINE wave  (TRIPOD_SPINE_SIN_COEF / _COS_COEF / _OFFSET)
# -------------------------------------------------------------------------
# NOT used by HexapodBinaryActionsCfg.  Used by eval_protocol.py's BASE_tripod_csv_bits
# anchor AND by play_discrete_closeup.py's --gait_npz tripod replay to score / visualise
# the reference extended-quadruped gait -- swapped into the running SpineSineAction via
# SpineSineAction.set_waveform(), then restored to WAVE 1.
#
# Same open-loop gait generator as WAVE 1, same amplitude (A_SPINE), but the
# extended-quadruped body parameters instead of the traveling-wave ones.  The two spine
# joints are ANTI-PHASE (BackLink = -FrontLink).  MATLAB gait generator (authoritative):
#
#     amp = 70;  amp_desired = 12;               % A = deg2rad(70)*12/16 = deg2rad(52.5) = A_SPINE
#     tt  = linspace(0, 2*pi, 50) - pi/4;        % gait phase, one 1.0 s cycle
#     xx  = A * sin(tt);                         % FRONT body joint -> FrontLink_Joint
#     yy  = A * sin(tt + body_phase);            % REAR  body joint -> BackLink_Joint
#     body_phase = (8*pi)/8 = pi   ->   yy = -xx   (genuinely anti-phase)
#
# With theta = w*t, w = 2*pi / GAIT_PERIOD_S, GAIT_PERIOD_S = 1.0:
#
#     FrontLink_Joint(t) = A_SPINE * sin(theta - pi/4)
#     BackLink_Joint (t) = A_SPINE * sin(theta - pi/4 + pi) = -A_SPINE * sin(theta - pi/4)
#
# The committed tripod_extendedquad_sim.csv has BYTE-IDENTICAL spine columns (an in-phase
# wave).  That collapse is a MATLAB->Sim body-correction EXPORT ARTIFACT (the export
# negates FrontLink and leaves BackLink, folding the anti-phase pair onto one column) --
# it was the BUG.  We do NOT reproduce it: WAVE 2 is regenerated analytically straight from
# the MATLAB spec above and KEPT anti-phase.  amp is taken as 70 deg (A_SPINE, matching
# WAVE 1) rather than the 65 deg the committed CSV was generated with, per the project
# decision that the baseline and the RL training wave share one amplitude.
#
# Single-harmonic sin/cos expansion (q = sqrt(2)/2 = 0.7071067811865476):
#     sin(theta - pi/4) = q * (sin theta - cos theta)
#   so  A_SPINE * sin(theta - pi/4)  ->  sin_coef = [ A_SPINE*q ], cos_coef = [ -A_SPINE*q ]
#   and the anti-phase joint (BackLink) negates BOTH coefficients.  offset = 0.0 on both.
#
# GLOBAL HexapI spine joint-sign: WAVE 2 changed from in-phase to anti-phase, so the
# per-joint signs were RE-VERIFIED in Isaac Sim, both assignments tried (2026-09-10):
#   command:  isaaclab.bat -p scripts/reinforcement_learning/binary_rl/eval_protocol.py
#             (default baselines, DCMotor actuator, friction 0.215, 64 envs x 300 steps,
#              no resets)   metric: BASE_tripod_csv_bits x_displacement_m
#   * Candidate 1  FrontLink = +A*sin(w*t - pi/4), BackLink = -A*sin(w*t - pi/4)
#         -> x_displacement_m = -0.8374 m  (BACKWARD, -0.4431 BL/cycle)
#   * Candidate 2  FrontLink = -A*sin(w*t - pi/4), BackLink = +A*sin(w*t - pi/4)  (global negation)
#         -> x_displacement_m = +0.8341 m  (FORWARD,  +0.4413 BL/cycle)
#   WINNER: Candidate 2 (positive x = forward), i.e. _TRIPOD_FRONT_SIGN = -1.0.  The
#   constants below are the winner.
#
# The -pi/4 phase origin is the SAME origin the leg-contact schedule in
# tripod_bit_demos.npz was generated from, so spine and legs stay aligned at reset
# (t = 0 -> CSV row 0 -> MATLAB tt = -pi/4).  eval_protocol.py's --tripod_spine_phase_deg
# can override the phase for calibration (it keeps the anti-phase BackLink = -FrontLink
# relation and the winning global sign).
#
# NOTE ON MAGNITUDE: the anti-phase WAVE 2 reaches ~0.44 BL/cycle forward (+0.8341 m over
# the 6-cycle window, straightness 0.98), right in line with the ~0.4 BL/cycle the
# extended-quad gait reaches on real hardware.  The earlier ~0.05-0.06 BL/cycle recorded
# here was the *in-phase* Wave 2 bug -- the two byte-identical CSV spine columns barely
# bent the body; the genuine anti-phase body wave does most of the propulsion (a
# `--spine_gain 0` run shows the leg bits alone net ~0).  step_dt (0.02 s), n_cycles (6.0)
# and body_length_m (0.315) in the BL/cycle conversion are all correct.
#
# Downstream files that MUST mirror WAVE 2 verbatim:
#   - scripts/reinforcement_learning/binary_rl/eval_protocol.py          (tripod baseline swap)
#   - scripts/reinforcement_learning/binary_rl/play_discrete_closeup.py  (--gait_npz tripod swap)
# =========================================================================

# --- WAVE 1: analytic traveling body wave (RL env) ---
A_SPINE = 0.9162978573
"""Spine traveling-wave amplitude, rad. EXACT analytic value from the open-loop gait
generator: ``deg2rad(70) * 12/16 = deg2rad(52.5) = 0.9162978572970227`` (amp=70 deg,
amp_desired=12). Not a guess."""

OFFSET_SPINE = 0.0
"""Spine traveling-wave constant offset, rad. FLAG: chosen constant, unverified in sim."""

SPINE_SIN_COEF = {"FrontLink_Joint": [-A_SPINE], "BackLink_Joint": [0.0]}
"""Per-spine-joint sine Fourier coefficients [a1], rad. FrontLink = pure sine, BackLink = pure cosine
(sin + 90 deg). Single harmonic -- analytic traveling wave, not a CSV fit. The ``-A_SPINE`` sign is the
HexapI global spine-joint-sign flip (see the CANONICAL block, flag (b)): the old fitted wave carried it
as a negative amplitude, WAVE 2's sim verification independently landed on the negative front-joint
sign, and a positive coefficient here walked every re-evaluated policy backward."""

SPINE_COS_COEF = {"FrontLink_Joint": [0.0], "BackLink_Joint": [-A_SPINE]}
"""Per-spine-joint cosine Fourier coefficients [b1], rad. See :data:`SPINE_SIN_COEF`."""

SPINE_OFFSET = {"FrontLink_Joint": OFFSET_SPINE, "BackLink_Joint": OFFSET_SPINE}
"""Per-spine-joint constant offset c0, rad (same on both joints)."""

# --- WAVE 2: extended-quadruped baseline spine wave (eval + closeup tripod replay) ---
# Analytic, single harmonic, SAME amplitude as WAVE 1 (A_SPINE), ANTI-PHASE on the two
# spine joints (BackLink = -FrontLink, MATLAB body_phase = pi), phase origin -pi/4:
#
#     FrontLink_Joint:  q(t) = _TRIPOD_FRONT_SIGN * A_SPINE * sin(w*t - pi/4)
#     BackLink_Joint :  q(t) = -_TRIPOD_FRONT_SIGN * A_SPINE * sin(w*t - pi/4)
#
# The committed tripod_extendedquad_sim.csv's two spine columns are byte-identical (an
# in-phase wave) -- a MATLAB->Sim body-correction EXPORT ARTIFACT that folded the
# anti-phase pair onto one column.  That was the bug; WAVE 2 is regenerated analytically
# from the MATLAB spec (see the CANONICAL block above) and KEPT anti-phase.
#
# _TRIPOD_FRONT_SIGN is the sim-verified global HexapI spine sign (2026-09-10, eval_protocol
# BASE_tripod_csv_bits, DCMotor actuator, friction 0.215, 64 envs x 300 steps, no resets):
#   * Candidate 1  FrontLink = +A*sin(w*t - pi/4)  ->  x_displacement_m = -0.8374 (backward)
#   * Candidate 2  FrontLink = -A*sin(w*t - pi/4)  ->  x_displacement_m = +0.8341 (forward)
# Winner: Candidate 2 (positive x = forward)  ->  _TRIPOD_FRONT_SIGN = -1.0.
#
# NOTE ON MAGNITUDE: the anti-phase wave nets ~0.44 BL/cycle forward (+0.8341 m / 6 cycles,
# straightness 0.98) -- in line with the ~0.4 BL/cycle the extended-quad gait reaches on
# real hardware.  The earlier ~0.05-0.06 BL/cycle here was the in-phase Wave 2 bug (the two
# byte-identical CSV spine columns barely bent the body); the genuine anti-phase body wave
# does most of the propulsion (`--spine_gain 0` shows the leg bits alone net ~0).  step_dt
# (0.02 s), n_cycles (6.0) and body_length_m (0.315) are all correct.
_TRIPOD_Q = A_SPINE * 0.7071067811865476  # A_SPINE*cos(pi/4) == A_SPINE*sin(pi/4) -> 0.6479204284835335
_TRIPOD_FRONT_SIGN = -1.0  # sim-verified global HexapI spine sign (Candidate 2); see the block above
_TRIPOD_FRONT_SIN = _TRIPOD_FRONT_SIGN * _TRIPOD_Q  # FrontLink sin_coef[0]
_TRIPOD_FRONT_COS = -_TRIPOD_FRONT_SIGN * _TRIPOD_Q  # FrontLink cos_coef[0]

TRIPOD_SPINE_SIN_COEF = {"FrontLink_Joint": [_TRIPOD_FRONT_SIN], "BackLink_Joint": [-_TRIPOD_FRONT_SIN]}
"""Tripod-baseline (extended-quad) sine coefficient ``[a1]``, rad; ANTI-PHASE -- ``BackLink = -FrontLink``
(MATLAB ``body_phase = pi``). ``FrontLink q(t) = _TRIPOD_FRONT_SIGN * A_SPINE * sin(w*t - pi/4)`` expanded to
sin/cos, with the sim-verified global HexapI spine sign. Swapped into :class:`SpineSineAction` via
``set_waveform()`` by ``eval_protocol.py`` and ``play_discrete_closeup.py`` -- NOT used by the RL env."""

TRIPOD_SPINE_COS_COEF = {"FrontLink_Joint": [_TRIPOD_FRONT_COS], "BackLink_Joint": [-_TRIPOD_FRONT_COS]}
"""Tripod-baseline cosine coefficient ``[b1]``, rad; ANTI-PHASE -- ``BackLink = -FrontLink``. See
:data:`TRIPOD_SPINE_SIN_COEF`."""

TRIPOD_SPINE_OFFSET = {"BackLink_Joint": 0.0, "FrontLink_Joint": 0.0}
"""Tripod-baseline constant offset ``c0``, rad (zero on both spine joints, same as WAVE 1)."""


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
    """Correct six inherited ``HexapodGoalEnvCfg`` reward weights for the contact-bit action space.

    The goal-env weights were tuned against an 8-dim *continuous* joint-position action.
    Several shaping terms misbehave once each leg is a single stance/lift bit, the
    goal-progress signal is too weak to make a clean forward walk net-positive, and two
    penalties stay expensive for an imperfect learning policy while costing the tuned
    tripod ~nothing. Applied identically to the train and play variants so their reward
    definitions cannot drift apart.

    Evidence (two sources; both pre-date the 2026-09 DCMotor actuator swap and should be
    re-measured):

    - ``eval_protocol.py`` on ``Isaac-Goal-Flat-Hexapod-Binary-v0``, baseline
      ``BASE_tripod_csv_bits`` (2026-09-09, 64 envs x 500 steps, no resets, goal pinned at
      2 m, old ``ImplicitActuatorCfg``). The reference tripod gait walks **+0.91 m** with
      **zero falls** yet scores ``reward_per_step -0.00116`` -- a straight forward walk is
      net-negative. Per-term (reward units per step): ``feet_slide -0.00106``,
      ``dof_acc_l2 -0.00030``, ``progress +0.00036`` (at weight 0.2),
      ``feet_air_time -4.9e-5``, ``time_penalty -1e-4``.
    - The last committed-config Double-DQN run (``runs_binary/ddqn_100k_s42``, 100k iters)
      logged mean episodic return **-2.98** and never learned to walk (episode ``progress``
      ~= 0). Dominant negatives were ``feet_slide`` (-1.16/episode), the old ``dof_acc_l2``
      (-1.23) and ``dof_torques_l2`` (-0.37), integrated over the ~1200-step episode
      against ~0 positive reward.

    The 2026-09 DCMotor swap (XL430 torque-speed curve, effort cap 4.5 -> 1.4 N.m, leg
    stiffness 20 -> 50, added armature) measurably collapses the torque/acceleration
    terms: ``compare_actuator_models.py`` (2026-09-09, same tripod gait) shows mean leg
    torque down ~23x, torque-cap saturation 0.29-0.50 -> 0.00, and joint-acceleration L2
    roughly 15-40x smaller. So ``dof_torques_l2`` (left at the inherited -3.0e-4) and
    ``dof_acc_l2`` are no longer the dominant negatives. After that swap the two
    asymmetric penalties still expensive for an imperfect policy are
    ``undesired_contacts`` (spine-link ground contact under the un-counterable scripted
    spine sinusoid) and the fixed ``time_penalty`` floor; both are trimmed below.

    Args:
        rewards: The already-initialised goal ``RewardsCfg`` instance to mutate in place.
    """
    # feet_slide -- REMOVED. The scripted spine sinusoid (SpineSineAction, not
    # RL-controlled) sways the body while the "stance" feet are planted, dragging them
    # across the inherited low-friction floor (calibrated 0.18-0.25 training band, from
    # the shared hexapod base). eval_protocol scores this penalty at -0.00106/step for the walking
    # tripod AND -0.00106/step for the all-six-feet-down baseline whose feet never move:
    # it measures forced slip from the waist wave, not gait quality, and the 6-bit policy
    # has no action channel that can reduce it. (The continuous goal env keeps feet_slide
    # because there the policy sets joint targets directly and can influence planted-foot
    # velocity.) Set to None rather than a small weight because there is no policy
    # behaviour for a non-zero weight to shape here -- it would only add a constant
    # negative offset and gradient noise.
    rewards.feet_slide = None

    # dof_acc_l2 -- inherited -2.5e-7 -> -3.0e-8 (~8x weaker). Joint acceleration in this
    # env is dominated by the legs snapping between the two fixed contact-bit angles
    # (STANCE 0.46 <-> LIFT 1.18 rad); the policy cannot soften that transition because
    # the action is a discrete bit, not a trajectory. eval (old actuator) showed the
    # inherited weight cost the stepping tripod ~4x what it cost a static pose, so it
    # directly taxed step frequency. The DCMotor swap now also shrinks the raw term
    # ~15-40x on its own (heavy reflected armature + no torque-cap saturation), so this
    # weight is near-inert either way. Kept slightly non-zero (not None) as cheap
    # insurance against a genuinely wild high-frequency solution; action_rate_l2 (left
    # unchanged, and acting on the actual 6-bit vector) is the primary anti-chatter term.
    rewards.dof_acc_l2.weight = -3.0e-8

    # feet_air_time -- weight 0.2 -> 0.0 (disabled, term kept for diagnostics). eval shows
    # the measured air time at foot-strike is ~0.02 s: the binary "lift" pose (1.18 rad)
    # barely lifts the foot clear of the ground, so contact chatters. That is far below
    # the inherited 0.12 s threshold, making feet_air_time_ungated net-NEGATIVE for a
    # clean walk (-4.9e-5/step) -- it punishes stepping instead of rewarding it. The
    # DCMotor swap makes the leg transitions slower, not higher, so the lift geometry is
    # unchanged. A useful stepping incentive needs the lift geometry to give real swing
    # clearance first, then a threshold recalibrated against measured air-times. Weight 0
    # (not None) leaves the term visible in the reward breakdown for that follow-up.
    rewards.feet_air_time.weight = 0.0

    # progress -- weight 0.2 -> 1.5. At 0.2 the whole 2 m course is worth only +0.4
    # reward, far less than the per-step drag it had to overcome, so locomotion never
    # became net-positive (the ddqn_100k_s42 run logged episode progress ~= 0 and a -2.98
    # return). progress had been cut hard from an earlier 10.0 (successive passes, see the
    # "hexapod goal reward scale-down" work) to suppress a "throw itself at the goal"
    # exploit specific to the *continuous* goal env -- the binary env's two-level legs
    # plus open-loop scripted spine cannot produce that explosive lunge (measured: under
    # the 1.4 N.m DCMotor cap the robot physically cannot), so the cut is unnecessary
    # here. 1.5 rather than 1.0: the DCMotor swap also made the robot weaker and slower
    # (peak leg speed ~1.45x lower), so it closes ground with less margin -- at 1.5 any
    # net forward walk clearly clears the -0.075 full-episode time_penalty floor plus the
    # (now small) per-step shaping costs, while the terminal reach_bonus still rewards
    # actually arriving. Still ~7x below the 10.0 that drove the continuous-env exploit.
    # If a lunge-style exploit ever does appear, the targeted lever is clamping the
    # closing speed inside progress_to_goal (a reward-function edit, out of scope here),
    # not re-cutting this.
    rewards.progress.weight = 1.5

    # undesired_contacts -- inherited -1.0 -> -0.5. This counts CenterLink / BackLink /
    # FrontLink ground contacts (>1 N) and contributes weight * count * step_dt =
    # -0.02 * count per step at the inherited weight. It is ~0 for the tuned tripod (which
    # keeps the body clear) but a large, asymmetric exploration tax on any early policy
    # that briefly lets the body sag: SpineSineAction sways the spine +-0.9163 rad every
    # cycle and the 6-bit policy has NO action channel to counter it, so a mistimed
    # contact pattern drags a spine link -- the same "policy cannot shape this" situation
    # that got feet_slide removed above. Halved rather than zeroed: base_contact
    # termination only watches CenterLink, so this stays the sole disincentive against a
    # BackLink / FrontLink belly-drag "gait" (at -0.5, dragging for even a third of an
    # episode still costs ~-4, swamping the progress gain over 2 m). The DCMotor swap
    # raises the resting base height (~0.055 m vs ~0.036 m measured), so genuine dragging
    # is already rarer -- the cut is mostly to de-tax exploration.
    rewards.undesired_contacts.weight = -0.5

    # time_penalty -- inherited -0.005 -> -0.003 (constant_per_step). At -0.005 this
    # integrates to weight * episode_seconds = -0.125 over a full 25 s timeout: a fixed
    # floor a slow-but-improving early policy cannot offset, forcing ~0.08 m of net
    # progress (at the new progress weight) just to break even on time alone before any
    # other term contributes. -0.003 drops that floor to -0.075 while keeping the term
    # negative, so it still selects for faster goal completion once a walk emerges. Kept
    # as a small non-zero speed pressure (not None): "as fast as possible" is still the
    # goal task's objective, just not what a policy that cannot yet walk should optimise.
    rewards.time_penalty.weight = -0.003


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
