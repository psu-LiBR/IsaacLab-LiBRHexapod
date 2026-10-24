# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unified evaluation protocol for every discrete (6-bit) hexapod policy family.

On this task "stand still with all six feet down" is a strong local optimum: it avoids the per-step penalties, so
training reward can rise while the robot never learns to walk. This script therefore measures every policy on one
fixed, deterministic protocol and reports metrics that separate walking from standing.

Protocol (identical for every policy, baselines included)
---------------------------------------------------------
* fixed seed, number of envs, number of steps and start state
* no resets: every termination is neutralised, so the displacement is one continuous trajectory that a reset
  teleport cannot pollute
* all domain randomisation pinned to deterministic values (audited term by term, printed at startup); goal
  distance pinned, curriculum, observation corruption and pushes off
* after ``env.reset()`` the command manager is recomputed (``ManagerBasedEnv.reset()`` skips it);
  ``--legacy_reset`` restores the old behaviour. ``--warmup`` (>= 1, default 1) then flushes the IMU / contact
  buffers, which only refresh on a real ``step()``
* the per-term reward breakdown (``reward_terms``) is reported for baselines too

Reported per policy
-------------------
    reward_per_step         the training curve's metric (per env-step, per env)
    net_displacement_m      straight-line distance travelled in the window
    x_displacement_m        component along the goal direction (+x)
    path_length_m           integrated travel (vibrating in place shows up here)
    straightness            net / path
    switch_metrics          per-leg contact-command switches per gait cycle, frozen legs, dwell times and a
                            pass/fail screen against ``--max_switches_per_leg_cycle`` (see ``switch_metrics.py``)
    action_entropy_nats     did the policy collapse onto one pattern
    mean_stance_legs        mean number of feet down (popcount of the 6-bit action)
    stance_hist             distribution over 0..6 stance legs
    frac_5plus_stance       fraction of steps with >= 5 feet down (standing detector)
    max_step_jump_m         teleport sentinel; must stay small
    disp_std_m              spread across envs
    x_disp_BL_per_cycle     x_displacement_m in body lengths per gait cycle (below)
    x_disp_BL_per_s         body lengths per second, independent of the cycle period
    leg_toggle_hz_realized  diagnostic only: measured lift->stance rate per leg (Hz), to compare against the
                            spine's fixed 1 Hz clock and spot leg-bit chattering

BL/cycle
--------
The group quotes gaits in body lengths per cycle. It is a pure unit change of ``x_displacement_m``:

    x_disp_BL_per_cycle = x_displacement_m / BODY_LENGTH_M / n_spine_cycles
    n_spine_cycles      = steps * step_dt / GAIT_PERIOD_S

``GAIT_PERIOD_S`` (1.0 s) is the period of the scripted spine wave (``SpineSineAction``), which no policy can
influence, so the cycle count is exact and identical for every policy (reported once in the ``protocol`` block).
It is deliberately not a measured leg-toggle rate: that would give a policy that chatters its bits a lower score
for the same distance. ``BODY_LENGTH_M`` is 0.315 m; both constants can be overridden on the command line. With
``step_dt`` = 0.02 s the default 300-step window is 6 s = 6 cycles.

Caveats:
* The paper values (tripod/b11bl0 0.48, extquad 0.41, lleg30 0.61, lleg35 0.56 BL/cycle) come from real hardware
  replaying joint trajectories and were computed on a 26.5 cm body length, whereas this script measures
  simulation with a 6-bit contact action space. Use them for orientation, not as a replication claim.
* The window has no resets, so a marginally stable "fast" gait that lunges or noses down without tripping the
  fall judge reads higher here than in a reset-enabled play video. ``progress`` in ``reward_terms`` telescopes to
  ``progress.weight * x_displacement_m`` and is an independent cross-check; also check ``fall_rate``,
  ``straightness`` and the play video before trusting a high value.

Baselines run first, every time, as the anchors of the table: all-stance (63), all-lift (0), uniform random and
``BASE_tripod_csv_bits``. The last replays the leg timing of ``tripod_bit_demos.npz`` (``--tripod_npz``
overrides; built by ``extract_bit_demos.py`` from ``tripod_extendedquad_sim.csv``, only its six leg columns are
used) and swaps the scripted spine from the RL env wave (Wave 1) to the anti-phase tripod wave (Wave 2) with
``SpineSineAction.set_waveform``, restoring it afterwards. Wave 2 reaches about 0.44 BL/cycle (+0.83 m over 6
cycles, straightness ~0.98), in line with the ~0.4 of the hardware; with ``--spine_gain 0`` the legs alone net
~0, so almost all of that travel is the body wave. The two waves (``SPINE_*`` vs ``TRIPOD_SPINE_*`` in
``hexapod_binary_env_cfg.py``) are intentionally distinct, and ``--spine_gain != 1.0`` disables the swap.
``--tripod_spine_phase_deg`` overrides the Wave 2 phase.

Run (from the repo root):
  isaaclab.bat -p scripts/reinforcement_learning/binary_rl/eval_protocol.py ^
      --policy net dqn_100k runs_binary/dqn_s42/checkpoints/agent_100000.pt ^
      --policy net ppo_masked_100k runs_binary/ppo_masked_s42/checkpoints/agent_100000.pt ^
      --out eval_results.json

A masked-PPO checkpoint is recognised by the ``run_meta.json`` next to it; the greedy argmax is then restricted
to the recorded legal set.
"""

import argparse
import os
import sys

# allow running from any CWD (e.g. the repo root, so relative asset paths resolve)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from switch_metrics import DEFAULT_MAX_SWITCHES_PER_LEG_CYCLE, SwitchRecorder, compact  # noqa: E402

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
    "--friction",
    type=float,
    default=None,
    help="optional explicit static/dynamic friction for a quantitative rollout. "
    "When omitted, retain the legacy deterministic midpoint of the task range; "
    "pass 0.21 for the real-robot-calibrated evaluation friction.",
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
    help="Reproduce the old reset behaviour (for comparing against old result tables). By default the "
    "command manager is recomputed after env.reset(), which otherwise leaves the previous rollout's goal "
    "command and stale IMU buffers in the first observation. Training is unaffected. See run().",
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
    help="steps taken before measurement starts; discards the opening transient",
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
    help="robot body length used for the BL/cycle column; 0.315 m is the front-to-rear leg length of the HexapI USD",
)
parser.add_argument(
    "--gait_period_s",
    type=float,
    default=1.0,
    help="scripted-spine gait-cycle period in s, the BL/cycle denominator (n_spine_cycles = steps * step_dt / "
    "gait_period_s). Matches GAIT_PERIOD_S in hexapod_binary_env_cfg.py",
)
parser.add_argument(
    "--spine_gain",
    type=float,
    default=1.0,
    help="ablation: multiply the scripted spine amplitudes by this (1.0 = unchanged, 0.0 = frozen waist, so "
    "only the legs can propel the robot)",
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
parser.add_argument(
    "--max_switches_per_leg_cycle",
    type=int,
    default=DEFAULT_MAX_SWITCHES_PER_LEG_CYCLE,
    help="switch_metrics screen: a complete gait cycle in which any leg's commanded contact bit flips more "
    "often than this fails basic_screen_pass (flips are counted on both edges)",
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

# Wave 2 (tripod baseline) coefficients used for the swap: the cfg values, or an anti-phase single-harmonic
# override of the phase that keeps the cfg's global spine sign.
if args.tripod_spine_phase_deg is None:
    TRIP_SIN, TRIP_COS, TRIP_OFF = TRIPOD_SPINE_SIN_COEF, TRIPOD_SPINE_COS_COEF, TRIPOD_SPINE_OFFSET
else:
    import math as _math

    _ph = _math.radians(args.tripod_spine_phase_deg)
    # FrontLink = _front_sign * A_SPINE * sin(w*t + ph), expanded into sin/cos terms; BackLink negates both
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
                    old = p[k]
                    m = (
                        args.friction
                        if args.friction is not None and k in ("static_friction_range", "dynamic_friction_range")
                        else _mid(p[k])
                    )
                    p[k] = (m, m)
                    source = (
                        "explicit --friction" if args.friction is not None and "friction" in k else "range midpoint"
                    )
                    AUDIT.append(f"events.{name}.{k}: {old} -> {p[k]} ({source})")
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

    try:
        cfg.observations.policy.enable_corruption = False
        AUDIT.append("observations.policy.enable_corruption -> False")
    except Exception:
        pass

    # curriculum off, so the goal distance cannot drift between policies
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

    # No resets: make every termination unreachable instead of removing it (reward terms reference
    # "base_contact" / "reach_goal" by name).
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

    # Optional spine ablation (default 1.0 = off): scale the scripted wave to see how much of the forward motion
    # is the waist rather than the legs. The action term stays registered, so the action and observation layouts
    # are unchanged.
    if args.spine_gain != 1.0:
        sw = getattr(getattr(cfg, "actions", None), "spine_wave", None)
        if sw is None:
            raise SystemExit("--spine_gain given but cfg.actions.spine_wave does not exist")
        # Scaling every sin/cos coefficient scales the amplitude linearly; the offset (mean posture) is scaled
        # separately via --spine_offset_gain.
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


# "Equivalent fall" judge: base_contact is neutralised, so its criterion is evaluated by hand every step (same
# sensor, body and threshold as mdp.illegal_contact).
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


# Torso contact for switch_metrics: the links the env's own ``undesired_contacts`` reward watches.
_torso_sensor = None
_torso_body_ids = None
_torso_threshold = args.fall_threshold
try:
    _uc = base.cfg.rewards.undesired_contacts
    _torso_sensor = base.scene.sensors[_uc.params["sensor_cfg"].name]
    _torso_names = _uc.params["sensor_cfg"].body_names
    _torso_body_ids = _torso_sensor.find_bodies(_torso_names)[0]
    _torso_threshold = float(_uc.params.get("threshold", args.fall_threshold))
    print(
        f"[protocol] torso judge: sensor='{_uc.params['sensor_cfg'].name}' bodies={_torso_names} "
        f"ids={_torso_body_ids} threshold={_torso_threshold} N",
        flush=True,
    )
except Exception as _exc:  # pragma: no cover
    print(f"[protocol] WARNING: torso judge unavailable ({_exc}); torso contact will not be screened", flush=True)


def torso_now():
    """[N] bool: any torso link above the undesired-contact threshold, or None if the judge is unavailable."""
    if _torso_sensor is None:
        return None
    f = _torso_sensor.data.net_forces_w_history
    f = f.torch if hasattr(f, "torch") else f
    return torch.any(torch.max(torch.linalg.norm(f[:, :, _torso_body_ids], dim=-1), dim=1)[0] > _torso_threshold, dim=1)


@torch.no_grad()
def run(policy_fn, label, meta=None):
    """One fixed-window rollout. policy_fn(obs, t) -> int64 actions [N]."""
    torch.manual_seed(args.seed)
    obs, _ = env.reset(seed=args.seed)
    if not args.legacy_reset:
        # reset() skips the command_manager.compute() that step() does, so the first observation would carry
        # the goal command of the previous rollout. Recompute it (dt=0: nothing moves) and the observations.
        base.command_manager.compute(dt=0.0)
        obs = base.observation_manager.compute()
    o = obs["policy"]
    # The IMU / contact sensors and acceleration buffers only refresh on a real step(), so --warmup must be >= 1
    # to flush their stale post-reset values.
    if args.warmup < 1:
        print(
            "[protocol] WARNING: --warmup 0 leaves stale IMU/contact buffers on the first "
            "measured step; >= 1 is strongly recommended",
            flush=True,
        )

    # discard the opening transient
    last_a = None
    for t in range(args.warmup):
        last_a = policy_fn(o, t)
        obs, _, _, _, _ = env.step(last_a)
        o = obs["policy"]

    start = root_rel()
    # Switch counting compares the first measured action with the last warm-up action.
    recorder = SwitchRecorder(start, robot.data.root_quat_w, previous_action=last_a)
    no_flag = torch.zeros(N, dtype=torch.bool, device=device)
    prev = start.clone()
    tot_r = 0.0
    # per-term reward breakdown, accumulated for every policy so all are scored on identical terms
    rterm_names = list(getattr(base, "reward_manager", None).active_terms) if hasattr(base, "reward_manager") else []
    rterm_sum = torch.zeros(len(rterm_names), device=device)
    path = torch.zeros(N, device=device)
    hist = torch.zeros(N_ACT, device=device)
    stance_hist = torch.zeros(7, device=device)
    max_jump = 0.0
    term_n = trunc_n = 0
    fall_steps = torch.zeros(N, device=device)
    first_fall = torch.full((N,), float("nan"), device=device)
    # Leg-toggle diagnostic: one lift -> stance rising edge counts as one toggle (none at the first measured step).
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
        phase = base.episode_length_buf.clone()  # the t the spine wave is driven from for this step
        obs, rew, terminated, truncated, _ = env.step(a)
        o = obs["policy"]
        cur = root_rel()
        d = torch.linalg.norm((cur - prev)[:, :2], dim=1)
        max_jump = max(max_jump, float(d.max()))
        path += d
        prev = cur
        tot_r += float(rew.mean())
        if rterm_names:
            # _step_reward is a rate (reward / dt); * step_dt gives reward per step
            rterm_sum += base.reward_manager._step_reward.mean(dim=0) * base.step_dt
        term_n += int(terminated.sum())
        trunc_n += int(truncated.sum())
        fl = fallen_now()
        if fl is not None:
            fall_steps += fl.float()
            first_fall = torch.where(fl & torch.isnan(first_fall), torch.full_like(first_fall, float(t)), first_fall)
        torso = torso_now()
        recorder.record(
            a,
            phase,
            cur,
            robot.data.root_quat_w,
            fl if fl is not None else no_flag,
            torso if torso is not None else no_flag,
        )
    end = root_rel()
    net_v = (end - start)[:, :2]
    net = torch.linalg.norm(net_v, dim=1)
    p = (hist / hist.sum()).clamp_min(1e-12)
    ent = float(-(p * p.log()).sum())
    sh = stance_hist / stance_hist.sum()
    top = torch.topk(hist, 4)
    x_disp_m = float(net_v[:, 0].mean())
    window_s = args.steps * base.step_dt
    # exact scripted-spine cycle count in the window (see the module docstring); identical for every policy
    n_spine_cycles = args.steps * base.step_dt / args.gait_period_s
    bl_per_cycle = round(x_disp_m / args.body_length_m / n_spine_cycles, 4)
    # diagnostic: mean lift -> stance rate per (env, leg) pair, Hz
    leg_toggle_hz_realized = float(leg_transitions.mean()) / window_s
    res = {
        "policy": label,
        "reward_per_step": round(tot_r / args.steps, 6),
        "net_displacement_m": round(float(net.mean()), 4),
        "x_displacement_m": round(x_disp_m, 4),
        "x_disp_BL_per_cycle": bl_per_cycle,
        "x_disp_BL_per_s": round(x_disp_m / args.body_length_m / window_s, 4),
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
    try:
        res["switch_metrics"] = {
            **compact(
                recorder.summarize(
                    dt=float(base.step_dt),
                    period=args.gait_period_s,
                    body_length=args.body_length_m,
                    max_switches_per_leg_cycle=args.max_switches_per_leg_cycle,
                )
            ),
            "fall_judge_available": _fall_sensor is not None,
            "torso_judge_available": _torso_sensor is not None,
        }
    except ValueError as exc:  # e.g. --gait_period_s is not a whole number of control steps
        res["switch_metrics"] = {"error": str(exc)}
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
    meta_path = os.path.join(os.path.dirname(os.path.abspath(path)), "run_meta.json")
    if not os.path.isfile(meta_path):
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
                        "requested_friction": args.friction,
                        "warmup_steps_discarded": args.warmup,
                        "fall_threshold_N": args.fall_threshold,
                        "step_dt": base.step_dt,
                        "no_reset": True,
                        # BL/cycle unit basis (identical for every policy, so recorded once)
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
