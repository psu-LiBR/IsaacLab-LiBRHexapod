# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
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
    x_disp_BL_per_cycle   x_displacement_m expressed in the group's standard unit,
                          body lengths per gait cycle (see "BL/cycle" below)

BL/cycle -- the group's standard unit (added 2026-08-31)
-------------------------------------------------------
Every gait number in the group's papers and slides is quoted in *body lengths per
cycle*, not metres per window.  This column is a pure unit change of
``x_displacement_m``; it is derived, it adds no new measurement, and it does not
enter any judgement in this script.

    x_disp_BL_per_cycle = x_displacement_m / BODY_LENGTH_M / n_cycles
    n_cycles            = steps * step_dt / GAIT_PERIOD_S

with the two constants (both overridable on the command line):

  BODY_LENGTH_M  = 0.265  the robot's body length, m.  Source: Jackson's own message
      with the reference gait CSVs -- "0.48 body lengths/cyc for b11bl0 and 0.41 BL/cyc
      (26.5cm body length)".  This is the same constant every earlier calibration in
      this project used, so the numbers stay comparable with our own history.
  GAIT_PERIOD_S  = 1.0    one gait cycle, s.  Not a choice: the env hard-codes it as
      ``GAIT_PERIOD_S`` in hexapod_binary_env_cfg.py, the scripted spine sinusoid runs at
      exactly ``sin(2*pi*t / 1.0 s)``, and the reference tripod CSV is 50 rows x 0.02 s
      (51st row duplicates the 1st) = one cycle.  step_dt is 0.02 s, so the default
      300-step window is exactly 6.00 s = 6.00 cycles and the conversion is x / 1.59.

Caveat, on purpose in this docstring so it travels with the number: the paper values
(tripod 0.48, extquad 0.41, lleg30 0.61, lleg35 0.56 BL/cycle) are *real hardware*
replaying joint-angle trajectories, whereas this script measures *simulation* with a
6-bit contact action space.  The unit is shared; the experiment is not.  Use the
comparison for orientation, never as a replication claim.

Baselines run first, every time, as the anchors of the table:
    all-stance (63), all-lift (0), uniform random, tripod-CSV bit sequence.

Run (from a worktree, .venv activated):
  CUDA_VISIBLE_DEVICES=<idle> python scripts/reinforcement_learning/eval_protocol.py \
      --policy net dqn_100k /path/agent_100000.pt \
      --policy muzero mz_final /path/ckpt_final.pt \
      --out runs_discrete/eval_protocol/results.json
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-Goal-Flat-Hexapod-Binary-v0")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--goal_distance", type=float, default=2.0,
                    help="pinned goal distance (m); far enough that reach_goal cannot fire in the window")
parser.add_argument("--policy", nargs=3, action="append", metavar=("TYPE", "NAME", "PATH"), default=[],
                    help="TYPE in {net,muzero,muzero_s,bits}; PATH is a checkpoint or a *_pattern.npz")
parser.add_argument("--no_baselines", action="store_true")
parser.add_argument("--fix_reset_command", action="store_true",
                    help="Recompute the command manager after env.reset(). IsaacLab's "
                         "ManagerBasedEnv.reset() calls _reset_idx() -> sim.forward() -> "
                         "observation_manager.compute() but never command_manager.compute(), so the "
                         "first observation of a rollout carries the base-frame goal left over from "
                         "the previous rollout. step() does call it, so training is unaffected; only "
                         "explicit-reset evaluation rollouts are. Off by default so existing numbers "
                         "reproduce bit-for-bit.")
parser.add_argument("--tripod_npz", default="",
                    help="tripod bit-demo npz for the third baseline (default: runs_discrete/demos_tripod_bits/tripod_bit_demos.npz)")
parser.add_argument("--v_min", type=float, default=-10.0, help="C51 support lower bound")
parser.add_argument("--v_max", type=float, default=10.0, help="C51 support upper bound")
parser.add_argument("--repeat", type=int, default=1, help="run each policy N times (determinism self-check)")
parser.add_argument("--warmup", type=int, default=1,
                    help="steps taken before measurement starts; discards the opening transient "
                         "(first-step termination/reward pollution is a known quirk of this goal env family)")
parser.add_argument("--fall_threshold", type=float, default=1.0,
                    help="contact force on the base body counted as a fall (matches the env's own base_contact rule)")
parser.add_argument("--body_length_m", type=float, default=0.265,
                    help="robot body length used for the BL/cycle column; 0.265 m is the value Jackson quoted "
                         "with the reference gait CSVs and the one every earlier calibration here used")
parser.add_argument("--gait_period_s", type=float, default=1.0,
                    help="one gait cycle in s for the BL/cycle column; must match GAIT_PERIOD_S in "
                         "hexapod_binary_env_cfg.py (scripted spine sinusoid period, = tripod CSV 50 rows x 0.02 s)")
parser.add_argument("--spine_gain", type=float, default=1.0,
                    help="ABLATION, opt-in: multiply the scripted spine sinusoid amplitudes by this. "
                         "1.0 (default) leaves the env exactly as trained/evaluated; 0.0 freezes the "
                         "waist so only the legs can propel the robot")
parser.add_argument("--spine_offset_gain", type=float, default=1.0,
                    help="ABLATION, opt-in: same for the spine's constant offset (keep at 1.0 to hold "
                         "the neutral posture while only the wave is removed)")
parser.add_argument("--out", default="eval_protocol_results.json")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import json
import os

import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401

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
        old_a = dict(sw.amplitude)
        sw.amplitude = {k: v * args.spine_gain for k, v in sw.amplitude.items()}
        old_o = dict(sw.offset)
        if args.spine_offset_gain != 1.0:
            sw.offset = {k: v * args.spine_offset_gain for k, v in sw.offset.items()}
            AUDIT.append(f"actions.spine_wave.offset: {old_o} -> x{args.spine_offset_gain}")
        AUDIT.append(f"actions.spine_wave.amplitude: {old_a} -> x{args.spine_gain} = {sw.amplitude}")

    e = gym.make(args.task, cfg=cfg)
    e = DiscreteBitsActionWrapper(e, n_bits=6)
    return e


env = build_env()
base = env.base_env
device = env.device
N = env.num_envs
obs_dim = int(base.single_observation_space["policy"].shape[-1])
robot = base.scene[list(base.scene.articulations.keys())[0]]
EYE = torch.eye(N_ACT, device=device)
POPCNT = torch.tensor([bin(i).count("1") for i in range(N_ACT)], device=device, dtype=torch.float32)

print("[protocol] ===== pinned configuration audit =====", flush=True)
for line in AUDIT:
    print("[protocol]   " + line, flush=True)
print(f"[protocol] task={args.task} num_envs={N} steps={args.steps} seed={args.seed} "
      f"obs_dim={obs_dim} step_dt={base.step_dt}", flush=True)


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
    print(f"[protocol] fall judge: sensor='{_bc.params['sensor_cfg'].name}' bodies={_names} "
          f"ids={_fall_body_ids} threshold={args.fall_threshold} N", flush=True)
except Exception as _exc:  # pragma: no cover
    print(f"[protocol] WARNING: fall judge unavailable ({_exc}); fall columns will be null", flush=True)


def fallen_now():
    """[N] bool: base body contact force above threshold -- the env's own fall criterion."""
    if _fall_sensor is None:
        return None
    f = _fall_sensor.data.net_forces_w_history
    f = f.torch if hasattr(f, "torch") else f
    return torch.any(torch.max(torch.linalg.norm(f[:, :, _fall_body_ids], dim=-1), dim=1)[0]
                     > args.fall_threshold, dim=1)


@torch.no_grad()
def run(policy_fn, label, meta=None):
    """One fixed-window rollout. policy_fn(obs, t) -> int64 actions [N]."""
    torch.manual_seed(args.seed)
    obs, _ = env.reset(seed=args.seed)
    if args.fix_reset_command:
        # See --fix_reset_command. Two lines, no physics touched.
        base.command_manager.compute(dt=0.0)
        obs = base.observation_manager.compute()
    o = obs["policy"]

    # opening transient: take --warmup steps before the measurement window opens
    for t in range(args.warmup):
        obs, _, _, _, _ = env.step(policy_fn(o, t))
        o = obs["policy"]

    start = root_rel()
    prev = start.clone()
    tot_r = 0.0
    path = torch.zeros(N, device=device)
    hist = torch.zeros(N_ACT, device=device)
    stance_hist = torch.zeros(7, device=device)
    max_jump = 0.0
    term_n = trunc_n = 0
    fall_steps = torch.zeros(N, device=device)
    first_fall = torch.full((N,), float("nan"), device=device)
    for t in range(args.steps):
        a = policy_fn(o, args.warmup + t)
        hist += torch.bincount(a, minlength=N_ACT).float()
        stance_hist += torch.bincount(POPCNT[a].long(), minlength=7).float()
        obs, rew, terminated, truncated, _ = env.step(a)
        o = obs["policy"]
        cur = root_rel()
        d = torch.linalg.norm((cur - prev)[:, :2], dim=1)
        max_jump = max(max_jump, float(d.max()))
        path += d
        prev = cur
        tot_r += float(rew.mean())
        term_n += int(terminated.sum())
        trunc_n += int(truncated.sum())
        fl = fallen_now()
        if fl is not None:
            fall_steps += fl.float()
            first_fall = torch.where(fl & torch.isnan(first_fall),
                                     torch.full_like(first_fall, float(t)), first_fall)
    end = root_rel()
    net_v = (end - start)[:, :2]
    net = torch.linalg.norm(net_v, dim=1)
    p = (hist / hist.sum()).clamp_min(1e-12)
    ent = float(-(p * p.log()).sum())
    sh = stance_hist / stance_hist.sum()
    top = torch.topk(hist, 4)
    res = {
        "policy": label,
        "reward_per_step": round(tot_r / args.steps, 6),
        "net_displacement_m": round(float(net.mean()), 4),
        "x_displacement_m": round(float(net_v[:, 0].mean()), 4),
        # Derived column, added 2026-08-31: same measurement, group-standard unit.
        # Nothing below reads it; it changes no existing column and no judgement.
        "x_disp_BL_per_cycle": round(
            float(net_v[:, 0].mean()) / args.body_length_m
            / (args.steps * base.step_dt / args.gait_period_s), 4),
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
    if _fall_sensor is not None:
        ever = ~torch.isnan(first_fall)
        surv = torch.where(ever, first_fall, torch.full_like(first_fall, float(args.steps))) * base.step_dt
        res.update({
            "fall_rate": round(float(ever.float().mean()), 3),
            "survival_s": round(float(surv.mean()), 3),
            "survival_s_min": round(float(surv.min()), 3),
            "frac_steps_base_contact": round(float((fall_steps / args.steps).mean()), 3),
            "window_s": round(args.steps * base.step_dt, 2),
        })
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
    meta = {"arch": arch, "state_key": src, "obs_scaler": scaler is not None,
            "hidden": [m.out_features for m in net if isinstance(m, torch.nn.Linear)][:-1]}

    def fn(o, t):
        x = scaler(o, train=False) if scaler is not None else o
        q = net(x)
        if support is not None:
            q = (torch.softmax(q.view(-1, N_ACT, atoms), dim=-1) * support).sum(-1)
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


def _mz_unscale(x):
    s = torch.sign(x)
    a = torch.abs(x)
    return s * (((torch.sqrt(1 + 4 * 1e-3 * (a + 1 + 1e-3)) - 1) / (2 * 1e-3)) ** 2 - 1)


def make_muzero_policy(path, stochastic=False):
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = ck["args"]
    L, H = cfg["latent"], cfg["hidden"]

    def mlp(i, o, h, layers=2, out_act=False):
        mods, d = [], i
        for _ in range(layers):
            mods += [torch.nn.Linear(d, h), torch.nn.ELU()]
            d = h
        mods += [torch.nn.Linear(d, o)]
        if out_act:
            mods += [torch.nn.ELU()]
        return torch.nn.Sequential(*mods)

    class Rep(torch.nn.Module):
        def __init__(s):
            super().__init__(); s.net = mlp(obs_dim, L, H); s.norm = torch.nn.LayerNorm(L)
        def forward(s, o):
            return s.norm(s.net(o))

    class Dyn(torch.nn.Module):
        def __init__(s):
            super().__init__(); s.trunk = mlp(L + N_ACT, H, H, layers=1, out_act=True)
            s.to_state = torch.nn.Linear(H, L); s.to_reward = torch.nn.Linear(H, 1)
            s.norm = torch.nn.LayerNorm(L)
        def forward(s, x, a):
            h = s.trunk(torch.cat([x, a], -1))
            return s.norm(s.to_state(h)), s.to_reward(h).squeeze(-1)

    class Pred(torch.nn.Module):
        def __init__(s):
            super().__init__(); s.trunk = mlp(L, H, H, layers=1, out_act=True)
            s.to_policy = torch.nn.Linear(H, N_ACT); s.to_value = torch.nn.Linear(H, 1)
        def forward(s, x):
            h = s.trunk(x)
            return s.to_policy(h), s.to_value(h).squeeze(-1)

    rep, dyn, pred = Rep().to(device), Dyn().to(device), Pred().to(device)
    rep.load_state_dict(ck["rep"]); dyn.load_state_dict(ck["dyn"]); pred.load_state_dict(ck["pred"])
    rep.eval(); dyn.eval(); pred.eval()
    rm, rv = ck["run_mean"].to(device), ck["run_var"].to(device)
    m, depth, sig, gam = cfg["n_cand"], cfg["search_depth"], cfg["sigma_scale"], cfg["discount"]
    meta = {"arch": f"muzero latent{L} hidden{H} cand{m} depth{depth}",
            "mode": "stochastic" if stochastic else "greedy"}

    def fn(o, t):
        x = torch.clamp((o - rm) / torch.sqrt(rv + 1e-8), -5.0, 5.0)
        s0 = rep(x)
        logits, _ = pred(s0)
        if stochastic:
            u = torch.rand_like(logits).clamp(1e-20, 1.0 - 1e-7)
            g = -torch.log(-torch.log(u))
            cand = torch.topk(g + logits, m, dim=1).indices
        else:
            g = torch.zeros_like(logits)
            cand = torch.topk(logits, m, dim=1).indices
        s = s0.unsqueeze(1).expand(-1, m, -1).reshape(N * m, -1)
        a = cand.reshape(N * m)
        q = torch.zeros(N * m, device=device)
        disc = 1.0
        for d in range(depth):
            s, r = dyn(s, EYE[a])
            q = q + disc * _mz_unscale(r)
            disc *= gam
            lg, v = pred(s)
            if d == depth - 1:
                q = q + disc * _mz_unscale(v)
            else:
                a = lg.argmax(1)
        q = q.reshape(N, m)
        qh = (q - q.min(1, keepdim=True).values) / (
            q.max(1, keepdim=True).values - q.min(1, keepdim=True).values).clamp_min(1e-8)
        pick = torch.argmax(torch.gather(g + logits, 1, cand) + sig * qh, dim=1)
        return torch.gather(cand, 1, pick.unsqueeze(1)).squeeze(1)

    return fn, meta


# ---------------------------------------------------------------- run everything
RESULTS = []


def do(fn, label, meta=None):
    for r in range(args.repeat):
        tag = label if args.repeat == 1 else f"{label}#rep{r}"
        try:
            RESULTS.append(run(fn, tag, meta))
        except Exception as exc:  # never let one policy kill the sweep
            print(f"[protocol] FAILED {tag}: {type(exc).__name__}: {exc}", flush=True)
            RESULTS.append({"policy": tag, "error": f"{type(exc).__name__}: {exc}"})
        with open(args.out, "w") as f:
            json.dump({"protocol": {"task": args.task, "num_envs": N, "steps": args.steps,
                                    "seed": args.seed, "goal_distance": args.goal_distance,
                                    "warmup_steps_discarded": args.warmup,
                                    "fall_threshold_N": args.fall_threshold,
                                    "step_dt": base.step_dt, "no_reset": True,
                                    # BL/cycle conversion constants, recorded so any
                                    # result file states its own unit basis.
                                    "body_length_m": args.body_length_m,
                                    "gait_period_s": args.gait_period_s,
                                    "n_cycles": round(args.steps * base.step_dt / args.gait_period_s, 4),
                                    "audit": AUDIT}, "results": RESULTS}, f, indent=2)


if not args.no_baselines:
    do(lambda o, t: torch.full((N,), 63, dtype=torch.long, device=device), "BASE_all_stance_63",
       {"arch": "fixed action 63 = all six feet down"})
    do(lambda o, t: torch.full((N,), 0, dtype=torch.long, device=device), "BASE_all_lift_0",
       {"arch": "fixed action 0 = all six feet up"})
    rand_gen = torch.Generator(device=device)

    def rand_pol(o, t):
        if t == 0:  # re-seed at the top of every rollout so repeats are identical
            rand_gen.manual_seed(args.seed)
        return torch.randint(0, N_ACT, (N,), device=device, generator=rand_gen)

    do(rand_pol, "BASE_uniform_random", {"arch": "uniform random 6-bit"})
    trip = args.tripod_npz or "runs_discrete/demos_tripod_bits/tripod_bit_demos.npz"
    if os.path.isfile(trip):
        f, m = make_bits_policy(trip)
        do(f, "BASE_tripod_csv_bits", m)
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
        elif kind == "muzero":
            f, m = make_muzero_policy(path, stochastic=False)
        elif kind == "muzero_s":
            f, m = make_muzero_policy(path, stochastic=True)
        else:
            raise ValueError(f"unknown policy type {kind}")
    except Exception as exc:
        print(f"[protocol] LOAD FAILED {name}: {type(exc).__name__}: {exc}", flush=True)
        RESULTS.append({"policy": name, "error": f"load: {type(exc).__name__}: {exc}", "path": path})
        continue
    m["checkpoint"] = path
    do(f, name, m)

print("[protocol] wrote " + args.out, flush=True)
simulation_app.close()
