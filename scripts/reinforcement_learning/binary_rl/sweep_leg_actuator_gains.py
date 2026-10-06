# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sweep the hexapod LEG actuator PD gains against the open-loop tripod gait.

The reference tripod gait replayed through ``play_discrete_closeup.py`` under-tracks its
commanded leg-joint targets in simulation (the middle legs worst).  This harness replays
that exact gait open-loop while sweeping the six leg joints' ``stiffness`` x ``damping``
(plus a small ``velocity_limit_sim`` sweep at the best point) and reports, per leg joint,
how well the commanded target is tracked and how hard the actuator is working.

.. note::
   This harness was written for the ``ImplicitActuatorCfg`` leg actuator. After the
   migration of ``HEXAPOD_CFG`` to an explicit ``DCMotorCfg`` (XL430 torque-speed model),
   the ``velocity_limit_sim`` sweep is a **no-op**: an explicit actuator reads
   ``velocity_limit`` / ``effort_limit`` (not the ``*_sim`` fields), so
   ``write_joint_velocity_limit_to_sim_index`` no longer changes the model's speed cap.
   The ``stiffness`` x ``damping`` sweep still works. Use ``compare_actuator_models.py``
   (``hexapod_committed`` model) to evaluate the committed DCMotor config.

What it does
------------
* Builds ``Isaac-Goal-Flat-Hexapod-Binary-Play-v0`` and wraps it with
  ``DiscreteBitsActionWrapper(n_bits=6)`` (same as ``play_discrete_closeup.py``).
* Replays the committed ``tripod_bit_demos.npz`` phase table (``base.episode_length_buf %
  P``) open-loop -- no policy, identical action stream for every gain combo.
* For each gain combo: writes the leg-joint gains onto the LIVE articulation (all envs,
  global set -- every env is identical per combo, which keeps the replay simple and
  correct), resets, runs ``--periods_settle`` gait periods to reach steady cyclic motion,
  then records ``--periods_record`` periods of commanded-vs-actual joint angle and applied
  torque.
* Reports per-leg RMS / max tracking error, stance-phase sag, end-of-swing settling
  error, peak applied torque, torque-saturation fraction and a fall/instability flag;
  a pretty table to stdout (sorted by middle-leg RMS error) and a CSV.

Config resets that are disabled for clean data (documented per project rule, see
``eval_protocol.py``)
---------------------------------------------------------------------------------------
* ``terminations.base_contact`` / ``terminations.reach_goal`` are neutralised in place
  (threshold -> 1e12; goal pinned 50 m away, never resampled) rather than deleted, because
  the ``reach_bonus`` / ``fall_penalty`` reward terms look them up by name.  A bad-gain
  combo that falls or drifts therefore never resets mid-window.  ``time_out`` is left
  untouched (the record window is far shorter than one episode).
* ``events.push_robot`` / ``events.base_external_force_torque`` -> ``None`` (already off on
  the Play cfg; set defensively).
* ``events.reset_robot_joints`` pinned to ``position_range=(1.0, 1.0)`` /
  ``velocity_range=(0.0, 0.0)`` and ``events.reset_base`` pinned to zeros so every combo
  starts from the identical pose.
* ``events.physics_material`` pinned to its range midpoints with ``num_buckets=1`` so
  per-env friction variance does not confound the tracking error.
* ``curriculum.goal_distance`` -> ``None`` (already off on the Play cfg; set defensively).
* ``observations.policy.enable_corruption`` -> ``False`` (open-loop replay ignores obs
  anyway).

Run
---
Smoke::

  isaaclab.bat -p scripts/reinforcement_learning/binary_rl/sweep_leg_actuator_gains.py ^
      --num_envs 16 --stiffness 20 80 --damping 0.35 0.9 --periods_settle 2 --periods_record 2

Full default grid::

  isaaclab.bat -p scripts/reinforcement_learning/binary_rl/sweep_leg_actuator_gains.py --num_envs 32
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", default="Isaac-Goal-Flat-Hexapod-Binary-Play-v0", help="binary Play task id")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument(
    "--stiffness",
    type=float,
    nargs="+",
    default=[20.0, 30.0, 40.0, 60.0, 80.0, 120.0],
    help="leg-joint stiffness values to sweep",
)
parser.add_argument(
    "--damping",
    type=float,
    nargs="+",
    default=[0.2, 0.35, 0.6, 0.9, 1.5],
    help="leg-joint damping values to sweep",
)
parser.add_argument(
    "--vel_limit",
    type=float,
    nargs="+",
    default=[10.0],
    help="velocity_limit_sim value(s) for the main stiffness x damping grid",
)
parser.add_argument(
    "--vel_limit_sweep",
    type=float,
    nargs="+",
    default=[8.0, 10.0, 15.0],
    help="velocity_limit_sim values run only at the best (stiffness, damping) from the main grid",
)
parser.add_argument("--effort_limit", type=float, default=4.5, help="leg-joint effort_limit_sim (single value, N*m)")
parser.add_argument("--periods_settle", type=int, default=3, help="gait periods run unrecorded to reach steady motion")
parser.add_argument("--periods_record", type=int, default=3, help="gait periods recorded for the metrics")
parser.add_argument(
    "--gait_npz",
    default="tripod",
    help="phase table from extract_bit_demos.py; 'tripod' = tripod_bit_demos.npz next to this script",
)
parser.add_argument("--out_csv", default="", help="CSV output path (default: runs_binary/leg_actuator_sweep_<ts>.csv)")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--fall_height",
    type=float,
    default=-1.0,
    help="root world-z (above env origin) below which an env is flagged unstable; "
    "<=0 auto-picks 0.6 * the post-reset resting height",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import csv
import os
import sys
import time

import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401

try:
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
except ImportError:
    from isaaclab_tasks.utils import parse_env_cfg

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from discrete_action_wrapper import DiscreteBitsActionWrapper  # noqa: E402

# --- constants --------------------------------------------------------------------------
# Leg joint angles the BinaryJointPositionAction snaps between (hexapod_binary_actions.py).
STANCE_POS = 0.460194236365692
LIFT_POS = 1.180398216278

# Leg joints in the 6-bit action layout (DiscreteBitsActionWrapper / HexapodBinaryActionsCfg):
# action index i (== bit i, LSB first) drives this joint.
BIT_LEG_JOINTS = [
    "FrontRight_Joint",
    "FrontLeft_Joint",
    "MiddleRight_Joint",
    "MiddleLeft_Joint",
    "BackRight_Joint",
    "BackLeft_Joint",
]
LEG_LABELS = [n.replace("_Joint", "") for n in BIT_LEG_JOINTS]
LEG_BODY_NAMES = ["FrontRight", "FrontLeft", "MiddleRight", "MiddleLeft", "BackRight", "BackLeft"]
MID_IDS = [2, 3]  # MiddleRight, MiddleLeft within BIT_LEG_JOINTS
FALL_HEIGHT = 0.10  # resolved from the post-reset resting height (or --fall_height) before the sweep

torch.manual_seed(args.seed)
np.random.seed(args.seed)


def _t(x):
    """Return ``x`` as a torch tensor (Isaac Lab data buffers are ProxyArrays)."""
    return x.torch if hasattr(x, "torch") else x


def _mid(rng: tuple[float, float]) -> float:
    return 0.5 * (float(rng[0]) + float(rng[1]))


# --- env build -------------------------------------------------------------------------
env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=args.num_envs)
env_cfg.seed = args.seed

_disabled: list[str] = []

# --- neutralise resets that would interrupt a bad-gain combo (time_out left alone) ---
# The reach_bonus / fall_penalty reward terms reference these terminations by name, so the
# terms must stay registered; make them unreachable instead of deleting them.
tm = getattr(env_cfg, "terminations", None)
if tm is not None and getattr(tm, "base_contact", None) is not None:
    tm.base_contact.params["threshold"] = 1.0e12
    _disabled.append("terminations.base_contact -> threshold 1e12 (never fires)")
if tm is not None and getattr(tm, "reach_goal", None) is not None:
    _disabled.append("terminations.reach_goal -> goal pinned 50 m away (never fires)")

pc = getattr(getattr(env_cfg, "commands", None), "pose_command", None)
if pc is not None:
    pc.ranges.pos_x = (50.0, 50.0)
    pc.ranges.pos_y = (0.0, 0.0)
    pc.ranges.heading = (0.0, 0.0)
    pc.resampling_time_range = (1.0e9, 1.0e9)
    _disabled.append("commands.pose_command -> pinned (50, 0, 0), never resampled")

ev = getattr(env_cfg, "events", None)
for name in ("push_robot", "base_external_force_torque"):
    if ev is not None and getattr(ev, name, None) is not None:
        setattr(ev, name, None)
        _disabled.append(f"events.{name}")

if ev is not None and getattr(ev, "reset_robot_joints", None) is not None:
    ev.reset_robot_joints.params["position_range"] = (1.0, 1.0)
    ev.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
    _disabled.append("events.reset_robot_joints -> pinned (1,1)/(0,0)")

if ev is not None and getattr(ev, "reset_base", None) is not None:
    pr = ev.reset_base.params.get("pose_range", {})
    vr = ev.reset_base.params.get("velocity_range", {})
    ev.reset_base.params["pose_range"] = {k: (0.0, 0.0) for k in pr}
    ev.reset_base.params["velocity_range"] = {k: (0.0, 0.0) for k in vr}
    _disabled.append("events.reset_base -> pinned to zeros")

if ev is not None and getattr(ev, "physics_material", None) is not None:
    p = ev.physics_material.params
    for k in ("static_friction_range", "dynamic_friction_range", "restitution_range"):
        if k in p:
            m = _mid(p[k])
            p[k] = (m, m)
    if "num_buckets" in p:
        p["num_buckets"] = 1
    _disabled.append("events.physics_material -> pinned to range midpoints, num_buckets=1")

cur = getattr(env_cfg, "curriculum", None)
if cur is not None and getattr(cur, "goal_distance", None) is not None:
    cur.goal_distance = None
    _disabled.append("curriculum.goal_distance")

try:
    env_cfg.observations.policy.enable_corruption = False
    _disabled.append("observations.policy.enable_corruption -> False")
except AttributeError:
    pass

env = gym.make(args.task, cfg=env_cfg)
env = DiscreteBitsActionWrapper(env, n_bits=6)
base = env.base_env
device = env.device
num_envs = env.num_envs
robot = base.scene["robot"]
step_dt = float(base.step_dt)
max_episode_steps = int(getattr(base, "max_episode_length", 10**9))

print("[sweep] disabled / pinned for clean tracking data:", flush=True)
for line in _disabled:
    print(f"[sweep]   {line}", flush=True)

# --- resolve leg joints / bodies ------------------------------------------------------
leg_ids_list, leg_names_res = robot.find_joints(BIT_LEG_JOINTS, preserve_order=True)
leg_ids_list = [int(i) for i in leg_ids_list]  # passed to the write-to-sim API (wants a plain int sequence)
leg_ids = torch.tensor(leg_ids_list, device=device, dtype=torch.long)  # used for torch column indexing
print(f"[sweep] leg joint ids (bit order {LEG_LABELS}): {leg_ids_list} -> {leg_names_res}", flush=True)

body_ids_list, body_names_res = robot.find_bodies(LEG_BODY_NAMES, preserve_order=True)


# --- gait table ---------------------------------------------------------------------
npz_path = args.gait_npz
if npz_path == "tripod":
    npz_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tripod_bit_demos.npz")
_d = np.load(npz_path, allow_pickle=True)
P = int(_d["period_steps"]) if "period_steps" in _d else len(_d["bits"])
_bits_np = np.asarray(_d["bits"])[:P].astype(np.int64)  # [P, 6] in bit order (LSB = FrontRight)
if _bits_np.ndim != 2 or _bits_np.shape[1] != 6:
    raise SystemExit(f"{npz_path}: gait bits must be [P, 6], got {_bits_np.shape}")
bits_t = torch.as_tensor(_bits_np, device=device)  # [P, 6]
_shifts = torch.arange(6, device=device, dtype=torch.long)
gait_table = (bits_t * (1 << _shifts)).sum(dim=1)  # [P] int64 action per phase
_stance = torch.tensor(STANCE_POS, device=device)
_lift = torch.tensor(LIFT_POS, device=device)
gait_targets = torch.where(bits_t.bool(), _stance, _lift).to(torch.float32)  # [P, 6] commanded leg target per phase

settle_steps = args.periods_settle * P
record_steps = args.periods_record * P
print(
    f"[sweep] gait '{os.path.basename(npz_path)}': period {P} steps ({P * step_dt:.2f} s) | "
    f"settle {settle_steps} steps | record {record_steps} steps | step_dt {step_dt:.4f} s",
    flush=True,
)
if settle_steps + record_steps >= max_episode_steps:
    print(
        f"[sweep] WARNING: settle+record ({settle_steps + record_steps}) >= max_episode_length "
        f"({max_episode_steps}); time_out may reset mid-window. Lower --periods_*.",
        flush=True,
    )


# --- startup mass report ------------------------------------------------------------
def _print_mass_report() -> None:
    try:
        mass = _t(robot.data.default_mass)
    except Exception:  # noqa: BLE001 -- default_mass is deprecated in some builds
        mass = _t(robot.data.body_mass)
    mass0 = mass[0].detach().cpu()
    total = float(mass0.sum())
    per_leg = {LEG_BODY_NAMES[i]: round(float(mass0[body_ids_list[i]]), 5) for i in range(len(LEG_BODY_NAMES))}
    print(f"[sweep] leg link masses [kg]: {per_leg}", flush=True)
    print(f"[sweep] total robot mass [kg]: {total:.5f}", flush=True)
    try:
        mm = _t(robot.data.mass_matrix)[0].detach().cpu()
        diag = torch.diagonal(mm)
        n_j = int(robot.num_joints)
        off = mm.shape[-1] - n_j  # floating base contributes leading DoFs
        diag_legs = {LEG_LABELS[i]: round(float(diag[off + leg_ids_list[i]]), 6) for i in range(len(LEG_LABELS))}
        print(f"[sweep] mass-matrix diagonal for leg DOFs [kg*m^2] (base offset {off}): {diag_legs}", flush=True)
    except Exception as exc:  # noqa: BLE001 -- mass_matrix not implemented on every backend
        print(f"[sweep] mass-matrix diagonal unavailable: {type(exc).__name__}: {exc}", flush=True)


# --- gain application --------------------------------------------------------------
def set_leg_gains(stiffness: float, damping: float, vel_limit: float, effort_limit: float) -> None:
    """Write the leg-joint PD gains onto the live articulation across all envs.

    Sets both the physics-solver gains (drive the actual PD) and the implicit-actuator
    model buffers (drive ``robot.data.applied_torque`` and the effort clip).
    """
    robot.write_joint_stiffness_to_sim_index(stiffness=float(stiffness), joint_ids=leg_ids_list)
    robot.write_joint_damping_to_sim_index(damping=float(damping), joint_ids=leg_ids_list)
    robot.write_joint_velocity_limit_to_sim_index(limits=float(vel_limit), joint_ids=leg_ids_list)
    robot.write_joint_effort_limit_to_sim_index(limits=float(effort_limit), joint_ids=leg_ids_list)
    act = robot.actuators["leg_joints"]
    # leg_joints owns exactly the six leg joints, so a full-buffer assignment is correct.
    act.stiffness[:] = float(stiffness)
    act.damping[:] = float(damping)
    act.effort_limit[:] = float(effort_limit)
    if hasattr(act, "velocity_limit"):
        act.velocity_limit[:] = float(vel_limit)


# --- one combo -------------------------------------------------------------------
@torch.no_grad()
def run_combo(stiffness: float, damping: float, vel_limit: float, effort_limit: float) -> dict:
    """Replay the gait at one gain combo and return per-leg tracking metrics."""
    set_leg_gains(stiffness, damping, vel_limit, effort_limit)
    env.reset(seed=args.seed)

    def _phase_action() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ph = (base.episode_length_buf % P).long()  # [N]
        return gait_table[ph], gait_targets[ph], bits_t[ph]  # [N], [N,6], [N,6]

    for _ in range(settle_steps):
        a, _, _ = _phase_action()
        env.step(a)

    err_rec: list[torch.Tensor] = []  # [N, 6] actual - target
    trq_rec: list[torch.Tensor] = []  # [N, 6] applied torque
    stance_rec: list[torch.Tensor] = []  # [N, 6] bool: this leg in stance this step
    min_height = float("inf")
    nan_flag = False
    for _ in range(record_steps):
        a, tgt, stance = _phase_action()
        env.step(a)
        q = _t(robot.data.joint_pos)[:, leg_ids]
        trq = _t(robot.data.applied_torque)[:, leg_ids]
        if torch.isnan(q).any() or torch.isnan(trq).any():
            nan_flag = True
            q = torch.nan_to_num(q)
            trq = torch.nan_to_num(trq)
        err_rec.append((q - tgt).detach())
        trq_rec.append(trq.detach())
        stance_rec.append(stance.bool())
        root_z = _t(robot.data.root_pos_w)[:, 2] - _t(base.scene.env_origins)[:, 2]
        min_height = min(min_height, float(root_z.min()))

    err = torch.stack(err_rec)  # [S, N, 6]
    trq = torch.stack(trq_rec)  # [S, N, 6]
    stance = torch.stack(stance_rec)  # [S, N, 6]
    abs_err = err.abs()

    rms = torch.sqrt((err**2).mean(dim=(0, 1)))  # [6]
    max_abs = abs_err.amax(dim=(0, 1))  # [6]
    peak_trq = trq.abs().amax(dim=(0, 1))  # [6]
    sat_frac = (trq.abs() >= 0.99 * effort_limit).float().mean(dim=(0, 1))  # [6]

    # stance sag: signed mean (actual - target) over the steps this leg is planted.
    stance_sag = torch.zeros(6, device=device)
    for j in range(6):
        m = stance[:, :, j]
        stance_sag[j] = err[:, :, j][m].mean() if m.any() else torch.tensor(float("nan"), device=device)

    # end-of-swing settling error: mean |err| at the last swing step before a stance step.
    end_swing = torch.zeros(6, device=device)
    trans = (~stance[:-1]) & stance[1:]  # [S-1, N, 6] swing -> stance transition at step t
    for j in range(6):
        m = trans[:, :, j]
        end_swing[j] = abs_err[:-1, :, j][m].mean() if m.any() else torch.tensor(float("nan"), device=device)

    fall_flag = bool(min_height < FALL_HEIGHT) or nan_flag

    def agg(vals: torch.Tensor, ids: list[int]) -> float:
        return float(vals[ids].mean())

    mid_rms = float(torch.sqrt((err[:, :, MID_IDS] ** 2).mean()))

    return {
        "stiffness": stiffness,
        "damping": damping,
        "vel_limit": vel_limit,
        "effort_limit": effort_limit,
        "rms_rad": rms.detach().cpu().tolist(),
        "rms_deg": torch.rad2deg(rms).detach().cpu().tolist(),
        "max_abs_deg": torch.rad2deg(max_abs).detach().cpu().tolist(),
        "stance_sag_deg": torch.rad2deg(stance_sag).detach().cpu().tolist(),
        "end_swing_deg": torch.rad2deg(end_swing).detach().cpu().tolist(),
        "peak_torque": peak_trq.detach().cpu().tolist(),
        "sat_frac": sat_frac.detach().cpu().tolist(),
        "min_height_m": min_height,
        "nan_flag": nan_flag,
        "fall_flag": fall_flag,
        "mid_rms_deg": float(np.rad2deg(mid_rms)),
        "mid_max_deg": float(np.rad2deg(agg(max_abs, MID_IDS))),
        "mid_peak_torque": agg(peak_trq, MID_IDS),
        "mid_sat_frac": agg(sat_frac, MID_IDS),
    }


# --- run the grid -----------------------------------------------------------------
_print_mass_report()

# Resolve the instability threshold from the resting root height (settle a few steps at the
# nominal gains so contacts and the base pose are established).
set_leg_gains(args.stiffness[0], args.damping[0], args.vel_limit[0], args.effort_limit)
env.reset(seed=args.seed)
for _ in range(min(P, 40)):
    _ph = (base.episode_length_buf % P).long()
    env.step(gait_table[_ph])
_rest_h = float((_t(robot.data.root_pos_w)[:, 2] - _t(base.scene.env_origins)[:, 2]).mean())
FALL_HEIGHT = args.fall_height if args.fall_height > 0.0 else max(0.03, 0.6 * _rest_h)
print(f"[sweep] resting root height ~{_rest_h:.3f} m -> instability threshold {FALL_HEIGHT:.3f} m", flush=True)

combos: list[tuple[float, float, float, str]] = []
for s in args.stiffness:
    for d in args.damping:
        for v in args.vel_limit:
            combos.append((s, d, v, "grid"))

results: list[dict] = []
t0 = time.time()
for i, (s, d, v, phase) in enumerate(combos):
    r = run_combo(s, d, v, args.effort_limit)
    r["phase"] = phase
    results.append(r)
    print(
        f"[sweep] {i + 1}/{len(combos)} k={s:g} d={d:g} v={v:g} -> mid_rms {r['mid_rms_deg']:.2f} deg "
        f"mid_sat {r['mid_sat_frac'] * 100:.0f}% {'FALL' if r['fall_flag'] else 'ok'} "
        f"({time.time() - t0:.0f}s)",
        flush=True,
    )

# --- velocity_limit_sweep at the best (stiffness, damping) ---
grid_clear = [r for r in results if not r["fall_flag"]] or results
best_grid = min(grid_clear, key=lambda r: r["mid_rms_deg"])
bs, bd = best_grid["stiffness"], best_grid["damping"]
print(f"[sweep] best grid (stiffness, damping) = ({bs:g}, {bd:g}); vel_limit sweep {args.vel_limit_sweep}", flush=True)
_already = {r["vel_limit"] for r in results if r["stiffness"] == bs and r["damping"] == bd}
for v in args.vel_limit_sweep:
    if v in _already:
        continue
    r = run_combo(bs, bd, v, args.effort_limit)
    r["phase"] = "vel_sweep"
    results.append(r)
    print(
        f"[sweep] vel_sweep k={bs:g} d={bd:g} v={v:g} -> mid_rms {r['mid_rms_deg']:.2f} deg "
        f"{'FALL' if r['fall_flag'] else 'ok'}",
        flush=True,
    )


# --- output ---------------------------------------------------------------------
def _fmt6(vals: list[float]) -> str:
    return " ".join(f"{x:6.2f}" for x in vals)


results_sorted = sorted(results, key=lambda r: r["mid_rms_deg"])

print("\n" + "=" * 124, flush=True)
print(
    f"{'phase':>9} {'k':>5} {'d':>5} {'v':>5} | {'midRMS':>7} {'midMax':>7} {'midPkT':>7} {'midSat':>7} "
    f"{'minH':>6} | {'worst leg (RMS deg)':>22} | fall",
    flush=True,
)
print("-" * 124, flush=True)
for r in results_sorted:
    worst_i = int(np.argmax(r["rms_deg"]))
    print(
        f"{r['phase']:>9} {r['stiffness']:>5g} {r['damping']:>5g} {r['vel_limit']:>5g} | "
        f"{r['mid_rms_deg']:>7.2f} {r['mid_max_deg']:>7.2f} {r['mid_peak_torque']:>7.3f} "
        f"{r['mid_sat_frac'] * 100:>6.1f}% {r['min_height_m']:>6.3f} | "
        f"{LEG_LABELS[worst_i]:>13} {r['rms_deg'][worst_i]:>7.2f} | "
        f"{'YES' if r['fall_flag'] else ''}",
        flush=True,
    )
print("=" * 124, flush=True)
print(f"per-leg RMS error [deg], leg order {LEG_LABELS}:", flush=True)
for r in results_sorted[: min(len(results_sorted), 12)]:
    print(
        f"  k={r['stiffness']:>4g} d={r['damping']:>4g} v={r['vel_limit']:>4g}: {_fmt6(r['rms_deg'])}"
        f"   (mid sag {_fmt6([r['stance_sag_deg'][j] for j in MID_IDS])})",
        flush=True,
    )

# CSV: one row per (combo, leg) plus a MID_AGG row per combo.
out_csv = args.out_csv or os.path.join("runs_binary", f"leg_actuator_sweep_{time.strftime('%Y%m%d_%H%M%S')}.csv")
os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
with open(out_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(
        [
            "phase",
            "stiffness",
            "damping",
            "vel_limit",
            "effort_limit",
            "leg",
            "rms_rad",
            "rms_deg",
            "max_abs_err_deg",
            "stance_sag_deg",
            "end_swing_err_deg",
            "peak_abs_torque_Nm",
            "sat_frac",
            "min_height_m",
            "fall_flag",
            "nan_flag",
        ]
    )
    for r in results_sorted:
        for j, leg in enumerate(LEG_LABELS):
            w.writerow(
                [
                    r["phase"],
                    r["stiffness"],
                    r["damping"],
                    r["vel_limit"],
                    r["effort_limit"],
                    leg,
                    f"{r['rms_rad'][j]:.6f}",
                    f"{r['rms_deg'][j]:.4f}",
                    f"{r['max_abs_deg'][j]:.4f}",
                    f"{r['stance_sag_deg'][j]:.4f}",
                    f"{r['end_swing_deg'][j]:.4f}",
                    f"{r['peak_torque'][j]:.4f}",
                    f"{r['sat_frac'][j]:.4f}",
                    f"{r['min_height_m']:.4f}",
                    int(r["fall_flag"]),
                    int(r["nan_flag"]),
                ]
            )
        w.writerow(
            [
                r["phase"],
                r["stiffness"],
                r["damping"],
                r["vel_limit"],
                r["effort_limit"],
                "MID_AGG",
                "",
                f"{r['mid_rms_deg']:.4f}",
                f"{r['mid_max_deg']:.4f}",
                "",
                "",
                f"{r['mid_peak_torque']:.4f}",
                f"{r['mid_sat_frac']:.4f}",
                f"{r['min_height_m']:.4f}",
                int(r["fall_flag"]),
                int(r["nan_flag"]),
            ]
        )
print(f"\n[sweep] wrote {out_csv}", flush=True)

clear = [r for r in results if not r["fall_flag"] and not r["nan_flag"]]
pool = clear or results
rec = min(pool, key=lambda r: r["mid_rms_deg"])
note = "" if clear else " (NO fall-clear combo; global min)"
print(
    f"[sweep] RECOMMENDATION: leg_joints stiffness={rec['stiffness']:g} damping={rec['damping']:g} "
    f"velocity_limit_sim={rec['vel_limit']:g}  -> middle-leg RMS {rec['mid_rms_deg']:.2f} deg, "
    f"mid sat {rec['mid_sat_frac'] * 100:.0f}%{note}",
    flush=True,
)

env.close()
simulation_app.close()
