# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unified evaluation protocol for every discrete (6-bit) hexapod policy family.

Why this exists
---------------
On this task "stand still with all six feet down" is a strong local optimum: it stops
collecting the per-step penalties, so *training reward can rise while the robot never
learns to walk*.  A comparison table built from training curves alone can therefore be
entirely misleading.  This script measures every policy on one fixed, deterministic
protocol and reports the metrics that actually separate "walking" from "standing".

Protocol (identical for every policy, baselines included)
--------------------------------------------------------
* fixed seed, fixed number of envs, fixed number of steps, fixed start state
* **no resets at all** during the window -- every termination is neutralised, so the
  measured displacement is one continuous trajectory and cannot be polluted by a
  reset teleport (project rule, 2026-07-21).
* all domain randomisation pinned to deterministic values (audited term by term over
  the whole config inheritance chain, project rule 2026-08-09) -- printed at startup.
* goal distance pinned, curriculum off, observation corruption off, pushes off.
* after env.reset() the command manager is recomputed (ManagerBasedEnv.reset() skips it,
  leaving the goal command from the previous rollout); ``--legacy_reset`` restores the
  old behaviour for reproducing pre-2026-09 tables. ``--warmup`` (>= 1, default 1) then
  flushes the IMU / contact-sensor buffers, which only refresh on a real step().
* per-policy reward-term breakdown (``reward_terms``) is reported for the baselines too,
  so the tripod gait and the learned policies are scored against identical reward terms.

Reported per policy
-------------------
    reward_per_step       the training curve's metric (per env-step, per env)
    net_displacement_m    straight-line distance travelled in the window
    x_displacement_m      component along the goal direction (+x)
    path_length_m         integrated travel (vibrating in place shows up here)
    straightness          net / path
    action_entropy_nats   did the policy collapse onto one pattern
    mean_stance_legs      mean popcount of the 6-bit action = feet on the ground
    stance_hist           distribution over 0..6 stance legs
    frac_5plus_stance     fraction of steps with >=5 feet down  <-- standing detector
    max_step_jump_m       teleport sentinel; must stay small
    disp_std_m            spread across envs
    x_disp_BL_per_cycle   x_displacement_m expressed in the group's standard unit, body
                          lengths per gait cycle, where "cycle" means the EXACT scripted-
                          spine period GAIT_PERIOD_S (see "BL/cycle" below); never null
    x_disp_BL_per_s       body lengths per second -- period-independent fallback, added
                          2026-09-14, always computable (see "BL/cycle" below)
    leg_toggle_hz_realized  DIAGNOSTIC ONLY, NOT used for x_disp_BL_per_cycle: measured
                          lift->stance rising-edge rate (Hz), averaged over every (env,
                          leg) pair in the window. Compare against the spine's fixed
                          1.0 Hz clock to gauge leg-bit chattering / gait pathology; added
                          2026-09-14 as a raw count, reworked into this Hz diagnostic
                          2026-09-15 after briefly (and wrongly) serving as the
                          x_disp_BL_per_cycle denominator -- see "Realized vs. assumed
                          cycle count" below

BL/cycle -- the group's standard unit (added 2026-08-31)
-------------------------------------------------------
Every gait number in the group's papers and slides is quoted in *body lengths per
cycle*, not metres per window.  This column is a pure unit change of
``x_displacement_m``; it is derived, it adds no new measurement, and it does not
enter any judgement in this script.

    x_disp_BL_per_cycle = x_displacement_m / BODY_LENGTH_M / n_spine_cycles
    n_spine_cycles      = steps * step_dt / GAIT_PERIOD_S -- the EXACT number of scripted-
                          spine cycles elapsed in the window.  Not an assumption:
                          GAIT_PERIOD_S is a hard-coded, non-learnable constant the
                          scripted spine (SpineSineAction, a zero-width action term the
                          6-bit leg policy cannot influence) runs on regardless of what
                          any policy does, so this count is exact for every policy in the
                          sweep, learned or scripted alike.

(a measured-leg-toggle-rate denominator was tried in its place on 2026-09-14 and reverted
2026-09-15 as wrong; see "Realized vs. assumed cycle count" below for why.  The measured
rate is still reported, as a diagnostic only, as ``leg_toggle_hz_realized``.)

with the two constants (both overridable on the command line):

  BODY_LENGTH_M  = 0.315  the robot's body length, m.  Source: the front-to-rear leg
      length of the updated HexapI USD, 0.315 m (per Jackson, 2026-09).  This is the
      quantity the BL/cycle metric should divide by (x_displacement_m / BODY_LENGTH_M /
      n_spine_cycles).
  GAIT_PERIOD_S  = 1.0    one gait cycle, s.  This is a fact about the SCRIPTED SPINE
      only (and, by construction, about the open-loop CSV / phase-table baselines): the
      env hard-codes it as ``GAIT_PERIOD_S`` in hexapod_binary_env_cfg.py, the scripted
      spine sinusoid runs at exactly ``sin(2*pi*t / 1.0 s)``, and the reference tripod CSV
      is 50 rows x 0.02 s (51st row duplicates the 1st) = one cycle.  step_dt is 0.02 s,
      so the default 300-step window is exactly 6.00 s = 6.00 cycles and the conversion is
      x / 1.89.  This is the sole denominator behind ``x_disp_BL_per_cycle`` -- for every
      policy in the sweep, learned or scripted -- see "Realized vs. assumed cycle count"
      below for why a per-policy measured leg-toggle rate is not used instead.


Caveat, on purpose in this docstring so it travels with the number: the paper values
(tripod/b11bl0 0.48, extquad 0.41, lleg30 0.61, lleg35 0.56 BL/cycle) are *real hardware*
replaying joint-angle trajectories, whereas this script measures *simulation* with a
6-bit contact action space.  The unit is shared; the experiment is not.  Use the
comparison for orientation, never as a replication claim.  Those anchors were computed
on a 26.5 cm body-length basis and are NOT directly comparable to this script's BL/cycle
column now that it divides by 0.315 m; this script's own previously-recorded BL/cycle
numbers must be re-measured.  The ``BASE_tripod_csv_bits`` anchor replays the
``tripod_extendedquad_sim.csv`` LEG timing (from ``tripod_bit_demos.npz``, paper value
0.41) AND, via ``SpineSineAction.set_waveform``, that gait's anti-phase SPINE wave (Wave 2:
``FrontLink = +-A_SPINE * sin(w*t - pi/4)``, ``BackLink`` the negation -- anti-phase,
regenerated analytically from the MATLAB gait generator, the global sign verified in sim
to walk the baseline forward; ``--tripod_spine_phase_deg`` overrides the phase).  Every
learned policy and the other baselines instead run on the analytic RL env spine wave
(Wave 1: FrontLink sine, BackLink sine + 90 deg).  These are two different spine waves by
design -- see the note above the tripod baseline run below.

With the anti-phase Wave 2 the tripod anchor comes out around 0.44 BL/cycle forward
(+0.83 m over the 6-cycle window, straightness ~0.98), in line with the ~0.4 of the
hardware.  The earlier ~0.05-0.09 BL/cycle recorded here was the *in-phase* Wave 2 bug --
the two byte-identical CSV spine columns barely bent the body.  The raw timing and
arithmetic are verified for this protocol: ``step_dt`` is 0.02 s, the window is 6.0 s,
and ``BODY_LENGTH_M`` is 0.315.  The value ``n_cycles=6`` still depends on treating the
fixed 1.0 s spine clock as the formal gait-cycle definition; that learned-policy
convention remains an explicit hypothesis until Jackson confirms it.  A
``--spine_gain 0`` run shows the leg bits alone net ~0 so almost all of the anchor's
travel is the body wave.

Two more things this number is NOT, kept here so they travel with it:
* It is measured over a NO-RESET window (every termination is neutralised -- see the
  protocol notes above).  The root is tracked as one continuous trajectory even if the
  robot lunges, scrabbles, or briefly noses down without tripping the CenterLink fall
  judge, so a marginally-stable "fast" gait reads higher here than a reset-enabled play
  video of the same checkpoint shows (the play env resets on base_contact / reach_goal
  and never accumulates the long slide).  ``progress`` in ``reward_terms`` is an
  independent cross-check: it telescopes to ``progress.weight * x_displacement_m`` over
  the window.  Sanity-check a high BL/cycle against fall_rate, straightness,
  action_entropy_nats and the play video before believing it -- there is no
  measurement-side inflation in this column (step_dt, n_cycles and the per-env mean net-x
  displacement were audited 2026-09), only the protocol difference just described.
* The 0.265 m -> 0.315 m body-length change (2026-09) divides every number by an extra
  1.189 vs. the pre-2026-09 tables: a checkpoint recorded at 0.95 BL/cycle on the old
  0.265 m basis is 0.80 on the current 0.315 m basis (0.95 * 0.265 / 0.315).  The
  displacement in metres is unchanged; only the unit basis moved.

Realized vs. assumed cycle count (found 2026-09-14)
-----------------------------------------------------
What was found: ``GAIT_PERIOD_S`` (and the ``n_cycles`` it fed) is a fact about the
SCRIPTED SPINE only -- ``SpineSineAction`` runs it on a fixed, uncontrollable 1.0 s clock,
and the open-loop baselines (``BASE_tripod_csv_bits`` and any ``'bits'``-type phase-table
policy, see ``make_bits_policy()``) are periodicity-locked to 1 Hz *by construction* of
their source table.  The six LEG bits are not: each is an independent
``BinaryJointPositionAction`` term the policy re-decides fresh every 50 Hz control step
(``discrete_action_wrapper.DiscreteBitsActionWrapper``), with no debounce, no
periodicity constraint, and no coupling to the spine's clock.  Dividing a trained
policy's displacement by an ASSUMED 1 Hz cycle count silently assumes its legs complete
one full stance/lift cycle every second -- true for the spine and the open-loop
baselines, never guaranteed for a policy that learned its own toggle cadence.

Evidence (measured live, RTX 4060, this script with a temporary rising-edge counter that
became the fix below): ``--num_envs 4 --steps 150`` (3.00 s window), seed 7, warmup 1.
``BASE_tripod_csv_bits`` self-check landed at ``n_cycles_realized = 3.00`` against
``n_cycles_assumed_1hz = 3.00`` -- an exact match, confirming the leg is genuinely 1 Hz
periodic by construction and that the rising-edge counter itself is correct.
``BASE_uniform_random`` realized ``36.92`` cycles over the same window (~12.3 Hz per leg,
in line with the ~50%-per-step flip probability of a uniform 6-bit action).  A trained
checkpoint, ``runs_binary/pipeline_20260911_090635/dqn/checkpoints/agent_100000.pt``
(predates the 2026-09-10 spine-wave sign fix and the DCMotor actuator swap, so its
absolute displacement/reward numbers are stale, but the leg-bit action space and the
``GAIT_PERIOD_S`` assumption it is being used to test are unaffected by either of those
changes), realized ``n_cycles_realized = 19.25`` over the same 3.00 s window -- roughly
6.4 Hz, i.e. **6.4x** the assumed 1.0 Hz.  Under the pre-fix formula this checkpoint would
report ``x_disp_BL_per_cycle`` ~= 0.714 (numerically equal to ``x_disp_BL_per_s`` here,
since ``gait_period_s = 1.0 s`` makes ``n_cycles_assumed_1hz`` and ``window_s`` the same
number); the realized-cycle-count formula reports ``0.111`` for the identical trajectory.
That is a larger inflation factor than the ~2.25x back-of-envelope estimate that first
flagged this bug, though this smoke run used a short 4-env/150-step window rather than the
standard 64-env/300-step protocol -- re-run the standard protocol for a production number.
Two other candidate explanations were checked in the same run and did not reproduce:
``step_dt`` printed as ``0.02`` (50 Hz control, not the 0.005 s physics dt), and every
policy in the run showed ``n_terminated = 0`` / ``n_truncated = 0`` with
``max_step_jump_m <= 0.0083`` m, i.e. no mid-window ``reach_goal`` firing or reset
teleport (goal pinned at 2.0 m, displacement well under the ~1.8 m needed to enter the
0.2 m reach radius).

What changed (2026-09-14, superseded the next day -- see below): ``x_disp_BL_per_cycle``
was made to divide by ``n_cycles_realized`` -- the measured mean lift->stance rising-edge
count per (env, leg) pair over the window, decoded every step via ``env.decode(a)`` and
compared against the previous measured step (no transition counted on the very first
measured step) -- instead of the assumed ``steps * step_dt / gait_period_s``. It was
``None`` when ``n_cycles_realized`` was ~0 (an all-stance-63 / all-lift-0 baseline that
never transitions); ``run_discrete_pipeline.py``'s ``rank()`` already null-checks that
field, so this never broke ranking.

Note on the checkpoint cited above as evidence: this section originally described
``runs_binary/pipeline_20260911_090635/dqn/checkpoints/agent_100000.pt`` as predating the
2026-09-10 spine-wave sign fix and the DCMotor actuator swap. That claim was not checked
against the checkpoint's actual filesystem mtime at the time; ``pipeline_20260911_090635``
is timestamped 2026-09-11, i.e. *after* both of those changes, and its checkpoint files
were confirmed (2026-09-15) to postdate both. The 6.4x realized-vs-assumed toggle-rate
finding itself is unaffected either way -- it is a property of the leg-bit action space,
not of the spine wave or actuator model -- but the "stale checkpoint" framing above was
unverified and should be read as such.

Reverted 2026-09-15 -- back to the exact spine-cycle denominator
------------------------------------------------------------------
The realized-cycle-count idea above was tried as the ``x_disp_BL_per_cycle`` denominator
and is wrong. Jackson (repo owner) caught it: ``GAIT_PERIOD_S`` is not an assumption to
begin with -- it is an exact, hard-coded, non-learnable constant. The two spine joints are
driven by ``SpineSineAction``, a zero-width action term (``action_dim == 0``) the 6-bit leg
policy cannot influence at all; its phase is a deterministic function of
``episode_length_buf`` that resets every episode. So
``steps * step_dt / GAIT_PERIOD_S`` is an *exact* count of real spine-wave cycles elapsed
in the window for every policy run by this script, not something that needs measuring.

Dividing by a measured leg-toggle rate instead was wrong for two reasons: (1) it breaks
cross-policy comparability -- two policies covering identical ground get different
denominators depending on how fast they happen to chatter their discrete leg bits, so a
policy that toggles faster (possibly a training pathology, not better locomotion) scores
*lower* BL/cycle for covering the *same* distance, backwards from the intent of the
metric; (2) it likely does not match the papers' convention -- "body lengths per cycle" in
the group's papers/slides almost certainly means per gait-generator/CPG period (a
controlled, designed quantity), not per raw footfall/bit-flip count. The
``BASE_tripod_csv_bits`` "exact match" (``n_cycles_realized = 3.00`` vs.
``n_cycles_assumed_1hz = 3.00``) cited above as validating evidence does not generalise --
it only confirms that ONE baseline's legs happen to toggle at 1 Hz by construction of its
source CSV; it says nothing about whether leg-toggle-counting is the right normalizer for
a policy running at some other rate.

``x_disp_BL_per_cycle`` is reverted to dividing by the exact spine-cycle count, renamed
``n_spine_cycles`` (was ``n_cycles_assumed_1hz`` -- "assumed" was itself a misnomer, since
the count is exact, not assumed) and reported once in the ``protocol`` block, since it is
identical for every policy in a run. It is never ``None`` (a 1.0 s ``gait_period_s``
denominator is never ~0 in practice, so the guard that existed for
``n_cycles_realized`` is not reinstated here). The leg-transition rising-edge measurement
above is kept -- it is a real, useful diagnostic about gait chattering -- but only as a
DIAGNOSTIC, never again as the BL/cycle denominator: renamed ``leg_toggle_hz_realized`` (a
rate, in Hz, rather than a raw per-window count so it is directly comparable to the
spine's fixed 1.0 Hz clock) and reported alongside ``x_disp_BL_per_cycle`` for
orientation, not folded into it.

Baselines run first, every time, as the anchors of the table:
    all-stance (63), all-lift (0), uniform random, tripod-CSV bit sequence.

The ``BASE_tripod_csv_bits`` anchor replays ``tripod_bit_demos.npz`` next to this script
(``--tripod_npz`` overrides) for the leg timing.  That NPZ is built by
``extract_bit_demos.py`` from ``tripod_extendedquad_sim.csv``; only its six leg columns
are used.  For that one baseline run the scripted spine term is additionally swapped from
the analytic RL env wave (Wave 1) to the anti-phase tripod body wave (Wave 2, regenerated
analytically from the MATLAB gait generator) via ``SpineSineAction.set_waveform``, then
restored so every subsequent policy in the sweep sees the RL env wave.  The two waves (constants
``SPINE_*`` vs ``TRIPOD_SPINE_*`` in ``hexapod_binary_env_cfg.py``) are intentionally
distinct.  ``--spine_gain != 1.0`` disables this swap (see the tripod run).

Run (from the repo root):
  isaaclab.bat -p scripts/reinforcement_learning/binary_rl/eval_protocol.py ^
      --policy net dqn_100k runs_binary/dqn_s42/checkpoints/agent_100000.pt ^
      --policy net ppo_masked_100k runs_binary/ppo_masked_s42/checkpoints/agent_100000.pt ^
      --out eval_results.json

The four baselines (all-stance 63, all-lift 0, uniform random, tripod-CSV bits) run
first every time as the anchors of the table. A masked-PPO checkpoint is recognised by
the ``run_meta.json`` written next to it (the greedy argmax is then restricted to the
recorded legal set). ``--legacy_reset`` reproduces the pre-2026-09 reset behaviour.
"""

import argparse
import os
import sys

# allow running from any CWD (e.g. the repo root, so relative asset paths resolve)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-Goal-Flat-Hexapod-Binary-v0")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--seed", type=int, default=7)
parser.add_argument(
    "--goal_distance",
    type=float,
    default=2.0,
    help="pinned goal distance (m); far enough that reach_goal cannot fire in the window",
)
parser.add_argument(
    "--policy",
    nargs=3,
    action="append",
    metavar=("TYPE", "NAME", "PATH"),
    default=[],
    help="TYPE in {net, bits}; PATH is a skrl checkpoint (.pt) or, for 'bits', a "
    "phase-table .npz. A 'net' checkpoint with a sibling run_meta.json declaring "
    "an action_mask is evaluated with the greedy argmax restricted to that set.",
)
parser.add_argument("--no_baselines", action="store_true")
parser.add_argument(
    "--legacy_reset",
    action="store_true",
    help="Reproduce the pre-fix reset behaviour bit-for-bit (for comparing against "
    "old result tables). By default this script now refreshes the scene and the "
    "command manager after env.reset() -- IsaacLab's ManagerBasedEnv.reset() runs "
    "_reset_idx() -> sim.forward() -> observation_manager.compute() but skips the "
    "per-step scene.update() and command_manager.compute() that step() does, so "
    "the first observation of a rollout otherwise carries stale IMU / "
    "projected-gravity buffers and the goal command left over from the previous "
    "rollout. Training is unaffected (its resets go through step()); only "
    "explicit-reset evaluation rollouts were. See run().",
)
parser.add_argument(
    "--tripod_npz",
    default="",
    help="tripod bit-demo npz for the tripod baseline (default: the tripod_bit_demos.npz "
    "committed next to this script)",
)
parser.add_argument("--v_min", type=float, default=-10.0, help="C51 support lower bound")
parser.add_argument("--v_max", type=float, default=10.0, help="C51 support upper bound")
parser.add_argument("--repeat", type=int, default=1, help="run each policy N times (determinism self-check)")
parser.add_argument(
    "--warmup",
    type=int,
    default=1,
    help="steps taken before measurement starts; discards the opening transient "
    "(first-step termination/reward pollution is a known quirk of this goal env family)",
)
parser.add_argument(
    "--fall_threshold",
    type=float,
    default=1.0,
    help="contact force on the base body counted as a fall (matches the env's own base_contact rule)",
)
parser.add_argument(
    "--body_length_m",
    type=float,
    default=0.315,
    help="robot body length used for the BL/cycle column; 0.315 m is the front-to-rear leg "
    "length of the updated HexapI USD (per Jackson, 2026-09)",
)
parser.add_argument(
    "--gait_period_s",
    type=float,
    default=1.0,
    help="EXACT scripted-spine gait-cycle period in s, used for x_disp_BL_per_cycle's "
    "denominator (n_spine_cycles = steps * step_dt / gait_period_s). Default matches "
    "GAIT_PERIOD_S in hexapod_binary_env_cfg.py (scripted spine sinusoid period, = tripod "
    "CSV 50 rows x 0.02 s)",
)
parser.add_argument(
    "--spine_gain",
    type=float,
    default=1.0,
    help="ABLATION, opt-in: multiply the scripted spine sinusoid amplitudes by this. "
    "1.0 (default) leaves the env exactly as trained/evaluated; 0.0 freezes the "
    "waist so only the legs can propel the robot",
)
parser.add_argument(
    "--spine_offset_gain",
    type=float,
    default=1.0,
    help="ABLATION, opt-in: same for the spine's constant offset (keep at 1.0 to hold "
    "the neutral posture while only the wave is removed)",
)
parser.add_argument(
    "--tripod_spine_phase_deg",
    type=float,
    default=None,
    help="CALIBRATION, opt-in: override the tripod-baseline (Wave 2) spine phase. When set, "
    "Wave 2 becomes s*A_SPINE * sin(w*t + radians(this)) on FrontLink_Joint and the "
    "negation on BackLink_Joint (anti-phase, single harmonic; s = the sim-verified global "
    "sign from the cfg default, A_SPINE from hexapod_binary_env_cfg). Default None = use "
    "the cfg's TRIPOD_SPINE_* (equivalent to --tripod_spine_phase_deg -45). Only affects "
    "BASE_tripod_csv_bits.",
)
parser.add_argument("--out", default="eval_protocol_results.json")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import json  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.contrib.velocity.config.hexapod.hexapod_binary_env_cfg import (  # noqa: E402
    A_SPINE,
    SPINE_COS_COEF,
    SPINE_OFFSET,
    SPINE_SIN_COEF,
    TRIPOD_SPINE_COS_COEF,
    TRIPOD_SPINE_OFFSET,
    TRIPOD_SPINE_SIN_COEF,
)

# Tripod-baseline (Wave 2) coefficients actually used for the swap: the cfg values, or a
# single-harmonic anti-phase override for phase calibration. Wave 2 is anti-phase
# (BackLink = -FrontLink); the phase override keeps that relation and the sim-verified
# global HexapI spine sign carried by the cfg default (FrontLink sin_coef sign).
if args.tripod_spine_phase_deg is None:
    TRIP_SIN, TRIP_COS, TRIP_OFF = TRIPOD_SPINE_SIN_COEF, TRIPOD_SPINE_COS_COEF, TRIPOD_SPINE_OFFSET
else:
    import math as _math

    _ph = _math.radians(args.tripod_spine_phase_deg)
    # FrontLink = _front_sign * A_SPINE * sin(w*t + ph)
    #           = (_front_sign*A*cos ph)*sin(w*t) + (_front_sign*A*sin ph)*cos(w*t)
    # BackLink negates both (anti-phase). _front_sign is taken from the cfg default so this
    # override can never disagree with the sim-verified global sign.
    _front_sign = 1.0 if TRIPOD_SPINE_SIN_COEF["FrontLink_Joint"][0] >= 0.0 else -1.0
    _s = _front_sign * A_SPINE * _math.cos(_ph)
    _c = _front_sign * A_SPINE * _math.sin(_ph)
    TRIP_SIN = {"FrontLink_Joint": [_s], "BackLink_Joint": [-_s]}
    TRIP_COS = {"FrontLink_Joint": [_c], "BackLink_Joint": [-_c]}
    TRIP_OFF = {"BackLink_Joint": 0.0, "FrontLink_Joint": 0.0}

try:
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
except ImportError:
    from isaaclab_tasks.utils import parse_env_cfg

from discrete_action_wrapper import DiscreteBitsActionWrapper

N_ACT = 64
torch.manual_seed(args.seed)
np.random.seed(args.seed)

AUDIT: list[str] = []


def _mid(rng):
    try:
        return (float(rng[0]) + float(rng[1])) / 2.0
    except Exception:
        return None


def pin_events(cfg) -> None:
    """Pin every randomisation term to a deterministic value; log what changed."""
    ev = getattr(cfg, "events", None)
    if ev is None:
        return
    for name in dir(ev):
        if name.startswith("_"):
            continue
        term = getattr(ev, name, None)
        if term is None or not hasattr(term, "params"):
            continue
        p = term.params
        if name == "reset_robot_joints":
            for k in ("position_range", "velocity_range"):
                if k in p:
                    old = p[k]
                    p[k] = (1.0, 1.0) if k == "position_range" else (0.0, 0.0)
                    AUDIT.append(f"events.{name}.{k}: {old} -> {p[k]}")
        elif name == "physics_material":
            for k in ("static_friction_range", "dynamic_friction_range", "restitution_range"):
                if k in p:
                    m = _mid(p[k])
                    old = p[k]
                    p[k] = (m, m)
                    AUDIT.append(f"events.{name}.{k}: {old} -> {p[k]}")
            if "num_buckets" in p:
                AUDIT.append(f"events.{name}.num_buckets: {p['num_buckets']} -> 1")
                p["num_buckets"] = 1
        elif name == "add_base_mass":
            if "mass_distribution_params" in p:
                old = p["mass_distribution_params"]
                p["mass_distribution_params"] = (1.0, 1.0) if p.get("operation") == "scale" else (0.0, 0.0)
                AUDIT.append(f"events.{name}.mass_distribution_params: {old} -> {p['mass_distribution_params']}")
        elif name == "base_com":
            if "com_range" in p:
                old = dict(p["com_range"])
                p["com_range"] = {k: (0.0, 0.0) for k in p["com_range"]}
                AUDIT.append(f"events.{name}.com_range: {old} -> zeros")
        elif name == "reset_base":
            for k in ("pose_range", "velocity_range"):
                if k in p and isinstance(p[k], dict):
                    old = dict(p[k])
                    p[k] = {kk: (0.0, 0.0) for kk in p[k]}
                    if any(abs(v[0]) + abs(v[1]) > 0 for v in old.values()):
                        AUDIT.append(f"events.{name}.{k}: {old} -> zeros")
    for kill in ("push_robot", "base_external_force_torque"):
        if getattr(ev, kill, None) is not None:
            setattr(ev, kill, None)
            AUDIT.append(f"events.{kill}: term -> None")


def build_env():
    cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=args.num_envs)
    cfg.seed = args.seed
    pin_events(cfg)

    # observation corruption off
    try:
        cfg.observations.policy.enable_corruption = False
        AUDIT.append("observations.policy.enable_corruption -> False")
    except Exception:
        pass

    # curriculum off (goal distance must not drift between policies)
    cur = getattr(cfg, "curriculum", None)
    if cur is not None:
        for name in dir(cur):
            if name.startswith("_"):
                continue
            term = getattr(cur, name, None)
            if term is not None and not callable(term):
                setattr(cur, name, None)
                AUDIT.append(f"curriculum.{name} -> None")

    # goal pinned, never resampled
    pc = getattr(cfg.commands, "pose_command", None)
    if pc is not None:
        pc.ranges.pos_x = (args.goal_distance, args.goal_distance)
        pc.ranges.pos_y = (0.0, 0.0)
        pc.ranges.heading = (0.0, 0.0)
        pc.resampling_time_range = (1.0e9, 1.0e9)
        AUDIT.append(f"commands.pose_command: pinned to ({args.goal_distance}, 0, 0), no resample")

    # --- NO RESETS: neutralise every termination without deleting the terms ---
    # (reward terms reference "base_contact"/"reach_goal" by name, so the terms must stay
    #  registered; they are made unreachable instead of removed.)
    old_len = cfg.episode_length_s
    cfg.episode_length_s = 1.0e6
    AUDIT.append(f"episode_length_s: {old_len} -> 1e6 (time_out can never fire)")
    tm = getattr(cfg, "terminations", None)
    if tm is not None and getattr(tm, "base_contact", None) is not None:
        old = tm.base_contact.params.get("threshold")
        tm.base_contact.params["threshold"] = 1.0e12
        AUDIT.append(f"terminations.base_contact.threshold: {old} -> 1e12 (fall never resets)")
    if tm is not None and getattr(tm, "reach_goal", None) is not None:
        AUDIT.append(
            f"terminations.reach_goal: kept (radius {tm.reach_goal.params.get('radius')}), "
            f"goal at {args.goal_distance} m -> unreachable inside the window; firings are reported"
        )

    # --- optional spine ablation (opt-in; default 1.0 = untouched behaviour) ---
    # The spine sinusoid is scripted and the RL policy cannot touch it, so "how much of the
    # forward motion is the waist rather than the legs" is only answerable by scaling it down.
    # Scaling the amplitudes is preferred over deleting the action term: the term stays
    # registered (same action-space width, same observation layout), only the wave shrinks.
    if args.spine_gain != 1.0:
        sw = getattr(getattr(cfg, "actions", None), "spine_wave", None)
        if sw is None:
            raise SystemExit("--spine_gain given but cfg.actions.spine_wave does not exist")
        # SpineSineActionCfg represents the wave as a truncated Fourier series: sin_coef /
        # cos_coef are dict[str, list[float]] (one coefficient list per spine joint).
        # Scaling every coefficient scales the wave amplitude linearly; the constant term
        # (offset, mean posture) is scaled separately via --spine_offset_gain. (Older
        # revisions carried a single amplitude/phase dict; guarded here so this ablation
        # still no-ops cleanly rather than crashing if the attrs are absent/renamed.)
        scaled: dict[str, tuple[dict, dict]] = {}
        for attr in ("sin_coef", "cos_coef"):
            table = getattr(sw, attr, None)
            if isinstance(table, dict):
                old = {k: list(v) for k, v in table.items()}
                setattr(sw, attr, {k: [c * args.spine_gain for c in v] for k, v in table.items()})
                scaled[attr] = (old, getattr(sw, attr))
        if not scaled:
            raise SystemExit("--spine_gain given but cfg.actions.spine_wave has no sin_coef/cos_coef dict to scale")
        off = getattr(sw, "offset", None)
        if args.spine_offset_gain != 1.0 and isinstance(off, dict):
            old_o = dict(off)
            sw.offset = {k: v * args.spine_offset_gain for k, v in off.items()}
            AUDIT.append(f"actions.spine_wave.offset: {old_o} -> x{args.spine_offset_gain}")
        for attr, (old, new) in scaled.items():
            AUDIT.append(f"actions.spine_wave.{attr}: {old} -> x{args.spine_gain} = {new}")

    e = gym.make(args.task, cfg=cfg)
    e = DiscreteBitsActionWrapper(e, n_bits=6)
    return e


def _spine_term(base_env):
    """Return the live ``SpineSineAction`` term from a built env, or None if absent.

    Robust to the action-manager API: tries ``ActionManager.get_term`` first, then the
    ``_terms`` / ``terms`` mapping. The term attr name in ``HexapodBinaryActionsCfg`` is
    ``spine_wave``.
    """
    am = base_env.action_manager
    for getter in ("get_term",):
        if hasattr(am, getter):
            try:
                return getattr(am, getter)("spine_wave")
            except Exception:
                pass
    terms = getattr(am, "_terms", None) or getattr(am, "terms", None) or {}
    return terms.get("spine_wave")


env = build_env()
base = env.base_env
device = env.device
N = env.num_envs
obs_dim = int(base.single_observation_space["policy"].shape[-1])
robot = base.scene[list(base.scene.articulations.keys())[0]]
POPCNT = torch.tensor([bin(i).count("1") for i in range(N_ACT)], device=device, dtype=torch.float32)

print("[protocol] ===== pinned configuration audit =====", flush=True)
for line in AUDIT:
    print("[protocol]   " + line, flush=True)
print(
    f"[protocol] task={args.task} num_envs={N} steps={args.steps} seed={args.seed} "
    f"obs_dim={obs_dim} step_dt={base.step_dt}",
    flush=True,
)


def root_rel():
    return (robot.data.root_pos_w - base.scene.env_origins).clone()


# --- "equivalent fall" judge -------------------------------------------------
# The protocol neutralises base_contact so a fall never resets the robot.  To keep the
# information the termination used to carry, its *criterion* is evaluated by hand every
# step (same sensor, same body, same threshold as mdp.illegal_contact).
_fall_sensor = None
_fall_body_ids = None
try:
    _bc = base.cfg.terminations.base_contact
    _fall_sensor = base.scene.sensors[_bc.params["sensor_cfg"].name]
    _names = _bc.params["sensor_cfg"].body_names
    _fall_body_ids = _fall_sensor.find_bodies(_names)[0]
    print(
        f"[protocol] fall judge: sensor='{_bc.params['sensor_cfg'].name}' bodies={_names} "
        f"ids={_fall_body_ids} threshold={args.fall_threshold} N",
        flush=True,
    )
except Exception as _exc:  # pragma: no cover
    print(f"[protocol] WARNING: fall judge unavailable ({_exc}); fall columns will be null", flush=True)


def fallen_now():
    """[N] bool: base body contact force above threshold -- the env's own fall criterion."""
    if _fall_sensor is None:
        return None
    f = _fall_sensor.data.net_forces_w_history
    f = f.torch if hasattr(f, "torch") else f
    return torch.any(
        torch.max(torch.linalg.norm(f[:, :, _fall_body_ids], dim=-1), dim=1)[0] > args.fall_threshold, dim=1
    )


@torch.no_grad()
def run(policy_fn, label, meta=None):
    """One fixed-window rollout. policy_fn(obs, t) -> int64 actions [N]."""
    torch.manual_seed(args.seed)
    obs, _ = env.reset(seed=args.seed)
    if not args.legacy_reset:
        # ManagerBasedEnv.reset() runs _reset_idx() -> sim.forward() -> observation
        # compute, but skips the command_manager.compute() that step() does, so the first
        # observation carries the base-frame goal command from the *previous* rollout.
        # Recompute it (dt=0.0: no integration, nothing moves) and recompute observations.
        base.command_manager.compute(dt=0.0)
        obs = base.observation_manager.compute()
    o = obs["policy"]
    # The IMU / contact sensors and the articulation acceleration buffers only refresh on
    # a real step() (they divide by dt, so a dt=0 refresh is not possible), and reset()
    # does not step. --warmup therefore MUST be >= 1: the discarded warmup steps also
    # flush those stale-after-reset sensor buffers. This is why the residual
    # projected_gravity / imu_ang_vel offset the RESULTS notes could not be explained by
    # the command manager alone.
    if args.warmup < 1:
        print(
            "[protocol] WARNING: --warmup 0 leaves stale IMU/contact buffers on the first "
            "measured step; >= 1 is strongly recommended",
            flush=True,
        )

    # opening transient: take --warmup steps before the measurement window opens
    for t in range(args.warmup):
        obs, _, _, _, _ = env.step(policy_fn(o, t))
        o = obs["policy"]

    start = root_rel()
    prev = start.clone()
    tot_r = 0.0
    # Per-term reward breakdown: accumulated for EVERY policy (baselines included) so the
    # tripod gait and the learned policies are scored against the identical reward terms
    # -- makes shaping penalties (action_rate_l2 etc.) visible and comparable instead of
    # only affecting what the RL policy was trained on.
    rterm_names = list(getattr(base, "reward_manager", None).active_terms) if hasattr(base, "reward_manager") else []
    rterm_sum = torch.zeros(len(rterm_names), device=device)
    path = torch.zeros(N, device=device)
    hist = torch.zeros(N_ACT, device=device)
    stance_hist = torch.zeros(7, device=device)
    max_jump = 0.0
    term_n = trunc_n = 0
    fall_steps = torch.zeros(N, device=device)
    first_fall = torch.full((N,), float("nan"), device=device)
    # Leg-toggle diagnostic tracking (added 2026-09-14; NOT used for x_disp_BL_per_cycle --
    # see the "BL/cycle" docstring section below and its "Reverted 2026-09-15" note): one
    # rising edge (lift -> stance) on a leg counts as one toggle for that leg, counted per
    # env. No transition is counted at the very first measured step (there is no prior
    # measured-step sample to compare against, and reaching back into the discarded
    # --warmup steps would require an extra, RNG-perturbing policy_fn() call).
    prev_stance = None
    leg_transitions = torch.zeros(N, 6, device=device)
    for t in range(args.steps):
        a = policy_fn(o, args.warmup + t)
        hist += torch.bincount(a, minlength=N_ACT).float()
        stance_hist += torch.bincount(POPCNT[a].long(), minlength=7).float()
        stance = env.decode(a) > 0  # [N, 6] bool: True = stance (foot down)
        if prev_stance is not None:
            leg_transitions += (stance & ~prev_stance).float()
        prev_stance = stance
        obs, rew, terminated, truncated, _ = env.step(a)
        o = obs["policy"]
        cur = root_rel()
        d = torch.linalg.norm((cur - prev)[:, :2], dim=1)
        max_jump = max(max_jump, float(d.max()))
        path += d
        prev = cur
        tot_r += float(rew.mean())
        if rterm_names:
            # _step_reward is the weight-applied per-term rate (reward / dt); * step_dt
            # puts it in the same "reward per step" units as reward_per_step.
            rterm_sum += base.reward_manager._step_reward.mean(dim=0) * base.step_dt
        term_n += int(terminated.sum())
        trunc_n += int(truncated.sum())
        fl = fallen_now()
        if fl is not None:
            fall_steps += fl.float()
            first_fall = torch.where(fl & torch.isnan(first_fall), torch.full_like(first_fall, float(t)), first_fall)
    end = root_rel()
    net_v = (end - start)[:, :2]
    net = torch.linalg.norm(net_v, dim=1)
    p = (hist / hist.sum()).clamp_min(1e-12)
    ent = float(-(p * p.log()).sum())
    sh = stance_hist / stance_hist.sum()
    top = torch.topk(hist, 4)
    x_disp_m = float(net_v[:, 0].mean())
    window_s = args.steps * base.step_dt
    # n_spine_cycles: the EXACT scripted-spine cycle count elapsed in the window. Not an
    # assumption -- GAIT_PERIOD_S is a hard-coded, non-learnable constant the zero-width
    # SpineSineAction term runs on regardless of what the 6-bit leg policy does -- see the
    # "BL/cycle" docstring section. This is the x_disp_BL_per_cycle denominator, identical
    # for every policy in the sweep (reported once, in the protocol block, not per result).
    n_spine_cycles = args.steps * base.step_dt / args.gait_period_s
    bl_per_cycle = round(x_disp_m / args.body_length_m / n_spine_cycles, 4)
    # DIAGNOSTIC ONLY, NOT used for x_disp_BL_per_cycle (see "Realized vs. assumed cycle
    # count" -> "Reverted 2026-09-15" in the docstring): how fast this policy's own leg
    # bits actually toggle between stance/lift, as a rate (Hz) so it is directly comparable
    # to the spine's fixed 1.0 Hz clock. Mean lift->stance rising-edge count, averaged over
    # every (env, leg) pair in the window, divided by the window duration.
    leg_toggle_hz_realized = float(leg_transitions.mean()) / window_s
    res = {
        "policy": label,
        "reward_per_step": round(tot_r / args.steps, 6),
        "net_displacement_m": round(float(net.mean()), 4),
        "x_displacement_m": round(x_disp_m, 4),
        # x_disp_BL_per_cycle (added 2026-08-31): same x-displacement measurement,
        # expressed in the group's standard unit (body lengths per gait cycle) -- see the
        # "BL/cycle" docstring section for the full history, including the 2026-09-14
        # realized-cycle-count detour and its 2026-09-15 revert. Divides by the EXACT
        # scripted-spine cycle count (n_spine_cycles, in the protocol block); never null.
        "x_disp_BL_per_cycle": bl_per_cycle,
        # Period-independent fallback (added 2026-09-14): body lengths per second. Makes no
        # assumption about cycle period at all, so it is always computable (no guard).
        "x_disp_BL_per_s": round(x_disp_m / args.body_length_m / window_s, 4),
        # DIAGNOSTIC ONLY -- NOT the x_disp_BL_per_cycle denominator (see above and the
        # docstring's "Reverted 2026-09-15" section). How fast this policy's own leg bits
        # toggle, in Hz; compare against the spine's fixed 1.0 Hz clock to gauge chattering.
        "leg_toggle_hz_realized": round(leg_toggle_hz_realized, 4),
        "path_length_m": round(float(path.mean()), 4),
        "straightness": round(float(net.mean() / max(float(path.mean()), 1e-9)), 3),
        "action_entropy_nats": round(ent, 3),
        "entropy_pct_of_uniform": round(100 * ent / np.log(N_ACT), 1),
        "mean_stance_legs": round(float((POPCNT * (hist / hist.sum())).sum()), 3),
        "stance_hist": [round(float(v), 3) for v in sh],
        "frac_5plus_stance": round(float(sh[5] + sh[6]), 3),
        "max_step_jump_m": round(max_jump, 4),
        "disp_std_m": round(float(net.std()), 4),
        "n_terminated": term_n,
        "n_truncated": trunc_n,
        "top_actions": [(int(i), round(float(v / hist.sum()), 3)) for i, v in zip(top.indices, top.values)],
    }
    if rterm_names:
        res["reward_terms"] = {n: round(float(v) / args.steps, 6) for n, v in zip(rterm_names, rterm_sum)}
    if _fall_sensor is not None:
        ever = ~torch.isnan(first_fall)
        surv = torch.where(ever, first_fall, torch.full_like(first_fall, float(args.steps))) * base.step_dt
        res.update(
            {
                "fall_rate": round(float(ever.float().mean()), 3),
                "survival_s": round(float(surv.mean()), 3),
                "survival_s_min": round(float(surv.min()), 3),
                "frac_steps_base_contact": round(float((fall_steps / args.steps).mean()), 3),
                "window_s": round(args.steps * base.step_dt, 2),
            }
        )
    if meta:
        res.update(meta)
    print("[protocol] " + json.dumps(res), flush=True)
    return res


# ---------------------------------------------------------------- policy builders
def build_mlp(state, has_ln):
    """Reconstruct a torch.nn.Sequential purely from a flat state dict (indices as keys)."""
    idxs = sorted({int(k.split(".")[0]) for k in state if k.split(".")[0].isdigit()})
    mods = []
    for i in range(max(idxs) + 1):
        w = state.get(f"{i}.weight")
        if w is None:
            mods.append(torch.nn.ReLU() if has_ln else torch.nn.ELU())
        elif w.dim() == 2:
            mods.append(torch.nn.Linear(w.shape[1], w.shape[0]))
        else:
            mods.append(torch.nn.LayerNorm(w.shape[0]))
    return torch.nn.Sequential(*mods)


def make_net_policy(path):
    ck = torch.load(path, map_location=device, weights_only=False)
    state = ck
    src = "raw"
    if isinstance(ck, dict):
        for key in ("q_network", "policy"):
            if key in ck:
                state, src = ck[key], key
                break
    state = {(k[4:] if k.startswith("net.") else k): v for k, v in state.items()}
    # Keep only the Sequential layer tensors ("<i>.weight" / "<i>.bias"); drop any mixin
    # buffers a policy model may carry (e.g. MaskedCategoricalMixin's "_action_mask").
    state = {k: v for k, v in state.items() if k.split(".")[0].isdigit()}
    # Linear weights are 2-D; a 1-D ".weight" can only be a LayerNorm (PQN-style net).
    has_ln = any(k.endswith(".weight") and v.dim() == 1 for k, v in state.items())
    net = build_mlp(state, has_ln).to(device)
    net.load_state_dict(state)
    net.eval()
    out_dim = net[-1].out_features
    atoms = out_dim // N_ACT if out_dim > N_ACT else 1
    support = torch.linspace(args.v_min, args.v_max, atoms, device=device) if atoms > 1 else None
    scaler = None
    if isinstance(ck, dict) and "observation_preprocessor" in ck:
        from skrl.resources.preprocessors.torch import RunningStandardScaler

        scaler = RunningStandardScaler(size=obs_dim, device=device)
        scaler.load_state_dict(ck["observation_preprocessor"])
        scaler.eval()
    arch = ("layernorm-relu" if has_ln else "elu") + (f"-c51x{atoms}" if atoms > 1 else "")
    meta = {
        "arch": arch,
        "state_key": src,
        "obs_scaler": scaler is not None,
        "hidden": [m.out_features for m in net if isinstance(m, torch.nn.Linear)][:-1],
    }

    # A masked-policy run records its legal action set in run_meta.json (written next to
    # the checkpoint's parent dir). Honour it here so the greedy argmax can never pick an
    # action the policy was never allowed to explore.
    legal = None
    meta_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(path))), "run_meta.json")
    if os.path.isfile(meta_path):
        with open(meta_path) as _mf:
            _rm = json.load(_mf)
        if isinstance(_rm.get("action_mask"), dict) and _rm["action_mask"].get("legal_actions"):
            legal = torch.zeros(N_ACT, dtype=torch.bool, device=device)
            legal[torch.tensor(_rm["action_mask"]["legal_actions"], device=device)] = True
            meta["action_mask_legal"] = int(legal.sum())

    def fn(o, t):
        x = scaler(o, train=False) if scaler is not None else o
        q = net(x)
        if support is not None:
            q = (torch.softmax(q.view(-1, N_ACT, atoms), dim=-1) * support).sum(-1)
        if legal is not None:
            q = q.masked_fill(~legal, float("-inf"))
        return torch.argmax(q, dim=1)

    return fn, meta


def make_bits_policy(path):
    d = np.load(path, allow_pickle=True)
    P = int(d["period_steps"])
    bits = torch.as_tensor(d["bits"][:P].astype(np.int64), device=device)  # [P, 6]
    shifts = torch.arange(6, device=device, dtype=torch.long)
    table = (bits * (1 << shifts)).sum(dim=1)  # [P] integer action per phase
    meta = {"arch": f"open-loop table (period {P})", "unique_actions": int(table.unique().numel())}

    def fn(o, t):
        phase = (base.episode_length_buf % P).long()
        return table[phase]

    return fn, meta


# ---------------------------------------------------------------- run everything
RESULTS = []


def do(fn, label, meta=None):
    for r in range(args.repeat):
        tag = label if args.repeat == 1 else f"{label}#rep{r}"
        try:
            RESULTS.append(run(fn, tag, meta))
        except Exception as exc:  # never let one policy kill the sweep
            import traceback

            print(f"[protocol] FAILED {tag}: {type(exc).__name__}: {exc}\n{traceback.format_exc()}", flush=True)
            RESULTS.append({"policy": tag, "error": f"{type(exc).__name__}: {exc}"})
        with open(args.out, "w") as f:
            json.dump(
                {
                    "protocol": {
                        "task": args.task,
                        "num_envs": N,
                        "steps": args.steps,
                        "seed": args.seed,
                        "goal_distance": args.goal_distance,
                        "warmup_steps_discarded": args.warmup,
                        "fall_threshold_N": args.fall_threshold,
                        "step_dt": base.step_dt,
                        "no_reset": True,
                        # BL/cycle conversion constants, recorded so any result file states
                        # its own unit basis. gait_period_s is exact -- the scripted
                        # spine's hard-coded period, not an assumption (see the "BL/cycle"
                        # docstring section). n_spine_cycles is identical for every policy
                        # in the sweep (a function of steps/step_dt/gait_period_s only), so
                        # it is recorded once here rather than per result.
                        "body_length_m": args.body_length_m,
                        "gait_period_s": args.gait_period_s,
                        "n_spine_cycles": round(args.steps * base.step_dt / args.gait_period_s, 4),
                        "audit": AUDIT,
                    },
                    "results": RESULTS,
                },
                f,
                indent=2,
            )


if not args.no_baselines:
    do(
        lambda o, t: torch.full((N,), 63, dtype=torch.long, device=device),
        "BASE_all_stance_63",
        {"arch": "fixed action 63 = all six feet down"},
    )
    do(
        lambda o, t: torch.full((N,), 0, dtype=torch.long, device=device),
        "BASE_all_lift_0",
        {"arch": "fixed action 0 = all six feet up"},
    )
    rand_gen = torch.Generator(device=device)

    def rand_pol(o, t):
        if t == 0:  # re-seed at the top of every rollout so repeats are identical
            rand_gen.manual_seed(args.seed)
        return torch.randint(0, N_ACT, (N,), device=device, generator=rand_gen)

    do(rand_pol, "BASE_uniform_random", {"arch": "uniform random 6-bit"})
    trip = args.tripod_npz or os.path.join(os.path.dirname(os.path.abspath(__file__)), "tripod_bit_demos.npz")
    if os.path.isfile(trip):
        f, m = make_bits_policy(trip)
        # The tripod anchor is scored on its OWN spine wave: the anti-phase tripod
        # body-bending wave (Wave 2, BackLink = -FrontLink), NOT the analytic RL env
        # traveling wave (Wave 1) that every learned policy and the other baselines run on.
        # Swap Wave 2 in for this one run via SpineSineAction.set_waveform, then restore
        # Wave 1 in a finally so a failure in the tripod run still leaves the correct env
        # wave for the rest of the sweep.
        #
        # --spine_gain != 1.0 already rescaled the Wave-1 cfg coefficients BEFORE the env
        # was built; calling set_waveform here would overwrite that ablation with the
        # (unscaled) tripod coefficients. Simpler correct choice: when the ablation is
        # active, SKIP the tripod-wave swap and leave the tripod baseline on the ablated
        # Wave 1, matching the other policies scored in that same run.
        term = _spine_term(base)
        swap_spine = term is not None and args.spine_gain == 1.0
        if swap_spine:
            term.set_waveform(TRIP_SIN, TRIP_COS, TRIP_OFF)
            _wave_desc = (
                "anti-phase tripod body wave (Wave 2: FrontLink +-A_SPINE*sin(w*t - pi/4), BackLink negated)"
                if args.tripod_spine_phase_deg is None
                else f"anti-phase A_SPINE*sin(w*t + {args.tripod_spine_phase_deg:g} deg) [phase override]"
            )
            AUDIT.append(
                f"spine_wave: tripod baseline uses {_wave_desc}; learned policies use the analytic env wave (Wave 1)"
            )
            print("[protocol]   " + AUDIT[-1], flush=True)
        elif term is not None and args.spine_gain != 1.0:
            AUDIT.append(
                "spine_wave: tripod baseline kept on the --spine_gain-ablated env wave "
                "(Wave 1); tripod Wave 2 swap skipped so the ablation is not overwritten"
            )
            print("[protocol]   " + AUDIT[-1], flush=True)
        try:
            do(f, "BASE_tripod_csv_bits", m)
        finally:
            if swap_spine:
                term.set_waveform(SPINE_SIN_COEF, SPINE_COS_COEF, SPINE_OFFSET)
    else:
        print(f"[protocol] tripod baseline skipped, not found: {trip}", flush=True)

for kind, name, path in args.policy:
    if kind != "bits" and not os.path.isfile(path):
        print(f"[protocol] MISSING {name}: {path}", flush=True)
        RESULTS.append({"policy": name, "error": "checkpoint missing", "path": path})
        continue
    try:
        if kind == "net":
            f, m = make_net_policy(path)
        elif kind == "bits":
            f, m = make_bits_policy(path)
        else:
            raise ValueError(f"unknown policy type {kind!r} (expected 'net' or 'bits')")
    except Exception as exc:
        print(f"[protocol] LOAD FAILED {name}: {type(exc).__name__}: {exc}", flush=True)
        RESULTS.append({"policy": name, "error": f"load: {type(exc).__name__}: {exc}", "path": path})
        continue
    m["checkpoint"] = path
    do(f, name, m)

print("[protocol] wrote " + args.out, flush=True)
simulation_app.close()
