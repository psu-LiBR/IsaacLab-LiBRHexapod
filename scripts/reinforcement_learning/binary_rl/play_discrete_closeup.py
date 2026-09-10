# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Replay a discrete policy OR an open-loop gait on the binary hexapod env and record a CLOSEUP video.

Identical to play_discrete.py except the recording camera is repositioned by hand every
step (play_openloop_closeup.py's ``_follow_cam``): headless rendering never fires Isaac's
own asset-root tracking callback, so ``viewer.origin_type = "asset_root"`` silently leaves
the camera fixed and the robot walks out of frame.  Same env, same render pipeline, same
RecordVideo wrapper -- only the camera placement differs.

Four action sources are supported (pass exactly one):

  * ``--checkpoint <agent_*.pt>`` -- a trained skrl checkpoint; runs the greedy policy
    (argmax Q / categorical logits, no exploration).
  * ``--gait_npz tripod`` -- the committed ``tripod_bit_demos.npz`` next to this script;
    ``--gait_npz <path.npz>`` for any table from ``extract_bit_demos.py``.
  * ``--gait_csv <path.csv>`` -- a Sim-DOF gait CSV whose leg columns are thresholded to
    contact *bits* on the fly (same rule as ``extract_bit_demos.py``).  Drives the binary
    env's 6-bit contact table.  Note: this path holds any *frozen* (constant) leg column
    at LIFT, so raw baseline gaits with intentionally-frozen stance legs shuffle in place
    rather than walk -- use ``--gait_csv_pos`` for those.
  * ``--gait_csv_pos <path.csv>`` -- replays a raw old-convention baseline gait CSV
    (e.g. ``hexapod-assets/Sim Gaits/forward3_lleg*_sim.csv``) as literal per-joint
    *position targets* that WALK, matching what ``playReal.py``'s replay does.  Because
    the binary env cannot accept 8 continuous joint targets, this path forces ``--task``
    to ``Isaac-Velocity-Flat-Hexapod-Play-v0`` (the same continuous-JointPositionAction
    env ``playReal.py`` uses) unless ``--task`` was given explicitly.

Raw-CSV convention and conversion (``--gait_csv_pos``)
----------------------------------------------------
Raw CSV: N rows x 8 cols, no header, radians, OLD pre-HexapI Sim-DOF column order
``[0 BackLink, 1 FrontLink, 2 MiddleLeft, 3 MiddleRight, 4 BackLeft, 5 BackRight,
6 FrontLeft, 7 FrontRight]`` in the old joint convention (leg stance ~= -0.4602,
lift ~= -1.1804; spine a +-~0.98 rad sinusoid).  Constants (``STANCE_POS``,
``LIFT_POS``, ``GAIT_PERIOD_S``, ``SPINE_AMPLITUDE/PHASE/OFFSET``) live in
``source/isaaclab_tasks/isaaclab_tasks/contrib/velocity/config/hexapod/hexapod_binary_env_cfg.py``
and are mirrored below.  Conversion to per-joint position targets:

  * Legs: ``v = -raw_value``; snap to the nearer of ``{STANCE_POS, LIFT_POS}``
    (threshold-free -- this is the fix; the ``--gait_csv`` midpoint threshold mishandles
    frozen columns).  Author leg columns are routed by joint NAME (col 2 -> MiddleLeft,
    3 -> MiddleRight, 4 -> BackLeft, 5 -> BackRight, 6 -> FrontLeft, 7 -> FrontRight).
  * Spine (BackLink, FrontLink): the raw CSV spine columns are DISCARDED and regenerated
    as ``q(t) = OFFSET[j] + AMPLITUDE[j] * sin(2*pi*t / GAIT_PERIOD_S + PHASE[j])`` with
    ``t = row_index * gait_dt`` -- identical to the binary env's ``SpineSineAction``.
  * Targets are emitted in the CURRENT runtime articulation DOF order, resolved by name
    from the live ``robot.data.joint_names`` (never a hardcoded permutation), then mapped
    to env actions via ``action = (target - default_joint_pos) / scale`` (scale ~= 0.5,
    read from the JointPositionAction term; mirrors ``playReal.py``).

The gait-bits path drives the exact same env and bit->angle decode that
``eval_protocol.py``'s ``BASE_tripod_csv_bits`` baseline uses, so the video is a faithful
visual check that the binary-env setup is wired correctly.  For the reference tripod
replay (``--gait_npz tripod`` or ``--gait_csv .../tripod_extendedquad_sim.csv``) the
scripted spine term is additionally swapped from the analytic RL traveling wave (WAVE 1)
to the anti-phase tripod body wave (WAVE 2: ``FrontLink = +-A_SPINE*sin(w*t - pi/4)``,
``BackLink`` the negation) -- the wave that gait's leg-contact schedule was designed
around, regenerated analytically from the MATLAB gait generator -- exactly as that eval
baseline does.

Receding-horizon goal (opt-in: ``--receding_horizon`` + ``--goal_ahead_m``, default OFF)
--------------------------------------------------------------------------------------
By default the goal env resets when the robot reaches the fixed ~2 m point ahead
(``reach_goal`` -> ``reached_goal_done``, radius ``REACH_RADIUS`` = 0.2 m), so a PLAY run
is a single dash-and-stop.  With ``--receding_horizon`` the ``pose_command`` target is
re-pinned ``--goal_ahead_m`` (default 2.0 m) ahead of the robot along world +x every
step, so the body-frame ``pose_command`` stays ~[goal_ahead, 0, 0, 0] and the robot
walks continuously -- turning the run into a *sustained* gait-quality read (BL/cycle,
straightness, fall behaviour over a long window) instead of one arrival.

Is this a good idea?  YES, but only as an opt-in for visual / long-horizon gait
inspection, kept OFF by default:

  * What it buys: play/eval then measures sustained locomotion instead of a single
    dash-and-stop -- the right thing for eyeballing a gait over many cycles.
  * The quantitative harness does NOT need it: ``eval_protocol.py`` already sidesteps
    the stop differently -- it pins ``--goal_distance 2.0`` (far enough that
    ``reach_goal`` cannot fire inside its 300-step / 6-cycle window), takes NO resets,
    and neutralises every termination (``tm.base_contact.params["threshold"] = 1e12``).
    Receding horizon is deliberately kept OUT of ``eval_protocol.py``.
  * ``progress_to_goal`` telescopes to ``weight*(initial_dist - final_dist)`` per
    episode; advancing the goal mid-episode would score a spurious large negative
    progress step unless its ``env._goal_prev_dist`` bookkeeping is re-seeded on the
    jump (mirroring the "first-post-reset step seeds prev_dist = curr_dist" logic in
    ``hexapod_goal_rewards.py::progress_to_goal``).  This script re-seeds it; in PLAY
    the reward is only printed, but it is kept clean.
  * Training is unaffected -- this is a live mutation of the running command /
    termination managers in this PLAY script only.  No env config class is touched.
  * Prior art: ``scripts/sim2real_transfer`` already ships an analogous
    ``goal.mode: receding`` + ``lookahead_m`` (``deployment.binary.example.yaml`` /
    ``command_source.py::RecedingGoalCommand``).  This flag matches that *semantics*
    (goal held a fixed distance ahead every step); the CLI name is ``--goal_ahead_m``
    rather than ``lookahead_m`` (sim2real is YAML, this is argparse) -- same meaning.

Caveats: the goal recedes exactly with the robot, so ``progress`` reads ~0 every step
(honest: the robot never closes distance to a moving goal) and ``reach_goal`` is
neutralised (radius -> -1) so it can never fire.  ``--receding_horizon`` is rejected
with ``--gait_csv_pos`` (that path forces the continuous velocity task, which has no
``pose_command``).

Runs on the Play variant of the task and records an MP4 through the regular render
pipeline (real robot mesh + ground plane), camera tracking the robot root.

Run (from IsaacLab root, .venv active):
  isaaclab.bat -p scripts/reinforcement_learning/binary_rl/play_discrete_closeup.py ^
      --gait_npz tripod --video_length 600
  isaaclab.bat -p scripts/reinforcement_learning/binary_rl/play_discrete_closeup.py ^
      --checkpoint runs_binary/<exp>/checkpoints/best_agent.pt --video_length 600
  isaaclab.bat -p scripts/reinforcement_learning/binary_rl/play_discrete_closeup.py ^
      --gait_csv_pos "hexapod-assets/Sim Gaits/forward3_lleg35_amp65_sim.csv" --video_length 600
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-Goal-Flat-Hexapod-Binary-Play-v0")
parser.add_argument("--checkpoint", default="", help="skrl checkpoint (.pt); omit to replay an open-loop gait")
parser.add_argument(
    "--gait_npz",
    default="",
    help="open-loop 6-bit gait table from extract_bit_demos.py; pass 'tripod' for the "
    "committed tripod_bit_demos.npz next to this script",
)
parser.add_argument(
    "--gait_csv",
    default="",
    help="reference gait CSV (Sim DOF order, 8 cols); leg columns are thresholded to "
    "contact bits on the fly, same rule as extract_bit_demos.py",
)
parser.add_argument(
    "--gait_csv_pos",
    default="",
    help="raw old-convention baseline gait CSV (Sim DOF order, 8 cols); replayed as "
    "literal per-joint position targets that WALK (legs snapped threshold-free to "
    "STANCE/LIFT, spine regenerated from the SpineSineAction formula). Forces "
    "--task Isaac-Velocity-Flat-Hexapod-Play-v0 unless --task is given explicitly. "
    "See the module docstring for the conversion and hexapod_binary_env_cfg.py for the "
    "constants.",
)
parser.add_argument("--gait_dt", type=float, default=0.02, help="seconds per CSV/NPZ row (period reporting only)")
parser.add_argument(
    "--receding_horizon",
    action="store_true",
    help="opt-in (default OFF): re-pin the pose_command goal --goal_ahead_m ahead of the "
    "robot (world +x) every step so it walks continuously instead of stopping at the "
    "~2 m goal. reach_goal is neutralised while on. PLAY/visual only; see the module "
    "docstring. Not supported with --gait_csv_pos.",
)
parser.add_argument(
    "--goal_ahead_m",
    type=float,
    default=2.0,
    help="receding-horizon look-ahead distance [m] (same semantics as sim2real "
    "commands.goal.lookahead_m). Only used with --receding_horizon.",
)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=600)
parser.add_argument("--video_length", type=int, default=600)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--out_dir",
    default="",
    help="video output dir (default: <checkpoint dir>/../videos_play, or <script dir>/videos_gait for a gait)",
)
parser.add_argument(
    "--eye",
    type=float,
    nargs=3,
    default=[0.9, 0.9, 0.45],
    help="camera offset from the robot root (same default as the approved closeup videos)",
)
parser.add_argument(
    "--lookat", type=float, nargs=3, default=[0.0, 0.0, 0.08], help="look-at offset from the robot root"
)
parser.add_argument("--v_min", type=float, default=-10.0, help="C51 checkpoints only: support lower bound")
parser.add_argument("--v_max", type=float, default=10.0, help="C51 checkpoints only: support upper bound")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

_sources = [bool(args.checkpoint), bool(args.gait_npz), bool(args.gait_csv), bool(args.gait_csv_pos)]
if sum(_sources) != 1:
    parser.error("pass exactly one of --checkpoint, --gait_npz, --gait_csv, --gait_csv_pos")

if args.receding_horizon and args.gait_csv_pos:
    parser.error(
        "--receding_horizon is not supported with --gait_csv_pos (that path forces the "
        "continuous velocity task, which has no pose_command)"
    )

# --gait_csv_pos needs a continuous-JointPositionAction env (the binary env's action space
# is 6 leg bits + a zero-width spine term). Default to the same env playReal.py uses unless
# the caller pinned --task explicitly.
if args.gait_csv_pos and args.task == parser.get_default("task"):
    args.task = "Isaac-Velocity-Flat-Hexapod-Play-v0"

args.headless = True
args.enable_cameras = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

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

from isaaclab_tasks.contrib.velocity.config.hexapod.hexapod_binary_env_cfg import (
    TRIPOD_SPINE_COS_COEF,
    TRIPOD_SPINE_OFFSET,
    TRIPOD_SPINE_SIN_COEF,
)

torch.manual_seed(args.seed)

env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=args.num_envs)
env_cfg.seed = args.seed
# closeup camera: driven manually every step (see _follow_cam below)
env_cfg.viewer.origin_type = "world"
env_cfg.viewer.asset_name = "robot"
env_cfg.viewer.env_index = 0
env_cfg.viewer.eye = tuple(args.eye)
env_cfg.viewer.lookat = tuple(args.lookat)

if args.out_dir:
    out_dir = args.out_dir
elif args.checkpoint:
    out_dir = os.path.join(os.path.dirname(os.path.abspath(args.checkpoint)), "..", "videos_play")
else:
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "videos_gait")

_name_prefix = "rl-video" if args.checkpoint else "gait-video"
env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array")
env = gym.wrappers.RecordVideo(
    env,
    video_folder=out_dir,
    step_trigger=lambda step: step == 0,
    video_length=args.video_length,
    name_prefix=_name_prefix,
    disable_logger=True,
)
# the raw-CSV position-replay path drives the env's native continuous JointPositionAction,
# so it must NOT go through the 6-bit discrete wrapper.
if not args.gait_csv_pos:
    env = DiscreteBitsActionWrapper(env, n_bits=6)
device = env.device


# architecture is auto-detected from the checkpoint's state-dict keys/shapes:
#   - plain MLP [128,128,128]+ELU -> 64      (DQN / DDQN / PPO of train_discrete*.py)
#   - Linear->LayerNorm->ReLU blocks -> 64   (PQN, train_discrete_pqn.py)
#   - plain MLP -> 64*atoms logits           (C51, train_discrete_c51.py; greedy uses expected Q)
def build_qnet(state: dict, obs_dim: int, hidden=(128, 128, 128)):
    has_layernorm = "1.weight" in state and state["1.weight"].dim() == 1
    out_dim = state[max((k for k in state if k.endswith(".weight")), key=lambda k: int(k.split(".")[0]))].shape[0]
    layers, in_dim = [], obs_dim
    for h in hidden:
        if has_layernorm:
            layers += [torch.nn.Linear(in_dim, h), torch.nn.LayerNorm(h), torch.nn.ReLU()]
        else:
            layers += [torch.nn.Linear(in_dim, h), torch.nn.ELU()]
        in_dim = h
    layers += [torch.nn.Linear(in_dim, out_dim)]
    arch = "layernorm-relu (PQN)" if has_layernorm else ("c51-distributional" if out_dim > 64 else "plain-elu")
    return torch.nn.Sequential(*layers), out_dim, arch


obs_space = env.single_observation_space["policy"]
obs_dim = int(obs_space.shape[-1])

# CSV leg columns in the action bit order (LSB = FrontRight); matches extract_bit_demos.py.
# CSV Sim-DOF layout: [0 BackLink, 1 FrontLink, 2 ML, 3 MR, 4 BL, 5 BR, 6 FL, 7 FR].
_LEG_COLS_BIT_ORDER = [7, 6, 3, 2, 5, 4]  # FR, FL, MR, ML, BR, BL

# Set true for the reference tripod replay: its LEG contact schedule is designed around the
# anti-phase tripod body wave (WAVE 2: BackLink = -FrontLink), NOT the analytic RL traveling
# wave (WAVE 1), so the scripted spine term is swapped to WAVE 2 after the env is built (see
# below), exactly as eval_protocol.py's BASE_tripod_csv_bits anchor does.
_tripod_spine_swap = False


def _gait_table_from_bits(bits_np: np.ndarray) -> torch.Tensor:
    """[P, 6] {0,1} contact bits (already exactly one period) -> [P] int64 action per phase."""
    if bits_np.ndim != 2 or bits_np.shape[1] != 6:
        raise SystemExit(f"gait bits must be [P, 6], got {bits_np.shape}")
    bt = torch.as_tensor(bits_np.astype(np.int64), device=device)
    shifts = torch.arange(6, device=device, dtype=torch.long)
    return (bt * (1 << shifts)).sum(dim=1)  # [P]


# --- raw-CSV -> per-joint position-target conversion (--gait_csv_pos) --------------------
# Constants mirror hexapod_binary_env_cfg.py. Leg levels: STANCE_POS / LIFT_POS /
# GAIT_PERIOD_S. Spine: the env's analytic "Wave 1" traveling body wave --
#   FrontLink_Joint: q(t) = 0.0 - 0.9162978573 * sin(2*pi*t / 1.0)
#   BackLink_Joint : q(t) = 0.0 - 0.9162978573 * sin(2*pi*t / 1.0 + pi/2)  (i.e. FrontLink + 90 deg)
# expressed below as OFFSET + AMPLITUDE * sin(2*pi*t / PERIOD + PHASE). This is NOT the old
# anti-phase tripod_B11BL0 CSV fit. Keep in sync with hexapod_binary_env_cfg.py (A_SPINE /
# OFFSET_SPINE / SPINE_SIN_COEF / SPINE_COS_COEF). The negative amplitude is the HexapI
# global spine-joint-sign flip (SPINE_SIN_COEF / SPINE_COS_COEF carry -A_SPINE); magnitude
# is the exact open-loop gait-generator value deg2rad(70) * 12/16 = deg2rad(52.5) = 0.9162978572970227 rad.
_STANCE_POS = 0.460194236365692  # leg joint angle, foot DOWN [rad]
_LIFT_POS = 1.180398216278  # leg joint angle, foot UP [rad]
_GAIT_PERIOD_S = 1.0
_SPINE_AMPLITUDE = {"BackLink_Joint": -0.9162978573, "FrontLink_Joint": -0.9162978573}
_SPINE_PHASE = {"BackLink_Joint": 1.5707963267948966, "FrontLink_Joint": 0.0}
_SPINE_OFFSET = {"BackLink_Joint": 0.0, "FrontLink_Joint": 0.0}
# author (old pre-HexapI) Sim-DOF CSV column -> runtime leg-joint name
_POS_LEG_COL_BY_NAME = {
    "MiddleLeft_Joint": 2,
    "MiddleRight_Joint": 3,
    "BackLeft_Joint": 4,
    "BackRight_Joint": 5,
    "FrontLeft_Joint": 6,
    "FrontRight_Joint": 7,
}


def _snap_leg(v: float) -> float:
    """Snap a leg angle to the nearer of STANCE_POS / LIFT_POS (threshold-free)."""
    return _STANCE_POS if abs(v - _STANCE_POS) <= abs(v - _LIFT_POS) else _LIFT_POS


def _build_gait_pos_targets(raw: np.ndarray, row_dt: float) -> dict[str, np.ndarray]:
    """Raw [P, 8] old-convention CSV -> {joint_name: [P] position targets [rad]}.

    Legs: ``-raw`` snapped to STANCE/LIFT. Spine: raw columns discarded, regenerated from
    the SpineSineAction formula with ``t = row_index * row_dt``.
    """
    p = raw.shape[0]
    t = np.arange(p, dtype=np.float64) * float(row_dt)
    out: dict[str, np.ndarray] = {}
    for name, col in _POS_LEG_COL_BY_NAME.items():
        out[name] = np.array([_snap_leg(-x) for x in raw[:, col]], dtype=np.float64)
    for name in ("BackLink_Joint", "FrontLink_Joint"):
        out[name] = _SPINE_OFFSET[name] + _SPINE_AMPLITUDE[name] * np.sin(
            2.0 * np.pi * t / _GAIT_PERIOD_S + _SPINE_PHASE[name]
        )
    return out


def _infer_joint_pos_offset_scale(sim_env, num_joints: int, dev: torch.device):
    """Infer ``joint_target = offset + scale * action`` for the env's JointPositionAction.

    Mirrors ``playReal.py``'s helper of the same name (credited); trimmed to the
    manager-based path and fixed to index env 0 for multi-env replay. Falls back to
    ``offset = default_joint_pos`` and ``scale = 0.5``.
    """
    offset_t: torch.Tensor | None = None
    scale_t: torch.Tensor | None = None

    am = getattr(sim_env, "action_manager", None)
    terms = None
    if am is not None:
        terms = getattr(am, "_terms", None)
        if terms is None:
            terms = getattr(am, "terms", None)
    if isinstance(terms, dict):
        for name, term in terms.items():
            lname = str(name).lower()
            if "joint" in lname and ("pos" in lname or "position" in lname):
                scale = getattr(term, "scale", None)
                if scale is None:
                    scale = getattr(term, "_scale", None)
                offset = getattr(term, "offset", None)
                if offset is None:
                    offset = getattr(term, "_offset", None)

                def _to_vec(x):
                    if isinstance(x, torch.Tensor):
                        v = x.detach().to(device=dev, dtype=torch.float32)
                        v = v[0] if v.dim() == 2 else v.flatten()
                        return v[:num_joints] if v.numel() >= num_joints else v.repeat(num_joints)[:num_joints]
                    if isinstance(x, (int, float)):
                        return torch.full((num_joints,), float(x), device=dev, dtype=torch.float32)
                    return None

                scale_t = _to_vec(scale)
                offset_t = _to_vec(offset)
                break

    if offset_t is None:
        robot = sim_env.scene["robot"]
        offset_t = robot.data.default_joint_pos[0, :num_joints].to(device=dev, dtype=torch.float32)
    if scale_t is None:
        scale_t = torch.full((num_joints,), 0.5, device=dev, dtype=torch.float32)
    return offset_t, scale_t


if args.checkpoint:
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    # DQN/DDQN checkpoints store the net under "q_network"; PPO under "policy".
    # Both are [128,128,128] MLPs to 64 outputs; greedy argmax works for either
    # (Q-values for DQN/DDQN, categorical logits for PPO).
    state = ckpt
    if isinstance(ckpt, dict):
        for key in ("q_network", "policy"):
            if key in ckpt:
                state = ckpt[key]
                break
    # skrl Model state dict prefixes layers with "net."
    state = {(k[4:] if k.startswith("net.") else k): v for k, v in state.items()}
    qnet, out_dim, arch = build_qnet(state, obs_dim)
    qnet = qnet.to(device)
    qnet.load_state_dict(state)
    qnet.eval()
    n_atoms = out_dim // 64 if out_dim > 64 else 1
    support = torch.linspace(args.v_min, args.v_max, n_atoms, device=device) if n_atoms > 1 else None
    print(f"[play_discrete] loaded checkpoint: {args.checkpoint} (arch: {arch}, out_dim {out_dim})")

    # PPO runs train with a running observation standardizer; replay must apply the same
    # transform (stored in the checkpoint) before the net. DQN/DDQN runs have no such key.
    obs_scaler = None
    if isinstance(ckpt, dict) and "observation_preprocessor" in ckpt:
        from skrl.resources.preprocessors.torch import RunningStandardScaler

        obs_scaler = RunningStandardScaler(size=obs_dim, device=device)
        obs_scaler.load_state_dict(ckpt["observation_preprocessor"])
        obs_scaler.eval()
        print("[play_discrete] applying stored observation standardizer")

    def policy_fn(obs):
        x = obs["policy"] if isinstance(obs, dict) else obs
        if obs_scaler is not None:
            x = obs_scaler(x, train=False)
        q = qnet(x)
        if support is not None:  # C51: greedy over expected Q of the distribution
            q = (torch.softmax(q.view(-1, 64, n_atoms), dim=-1) * support).sum(-1)
        return torch.argmax(q, dim=1)

elif args.gait_csv_pos:
    # ----- open-loop raw-CSV joint-position replay (walking), no network -----
    # See the module docstring for the conversion; constants mirror hexapod_binary_env_cfg.py.
    raw_pos = np.loadtxt(args.gait_csv_pos, delimiter=",")
    if raw_pos.ndim != 2 or raw_pos.shape[1] != 8:
        raise SystemExit(f"{args.gait_csv_pos}: expected a [T, 8] Sim-DOF CSV, got {raw_pos.shape}")
    P = int(raw_pos.shape[0])
    _pos_target_by_name = _build_gait_pos_targets(raw_pos, args.gait_dt)
    # env-dependent action mapping + runtime DOF order are resolved after `base` below.
    _POS_ACTIONS: torch.Tensor | None = None
    print(f"[play_discrete_closeup] open-loop raw-CSV position replay: {args.gait_csv_pos}")
    print(f"[play_discrete_closeup]   {P} rows ({P * args.gait_dt:.2f} s) | task {args.task}")

    def policy_fn(obs):  # noqa: ARG001 -- open-loop: action depends only on episode phase
        phase = (base.episode_length_buf % P).long()
        return _POS_ACTIONS[phase]

else:
    # ----- open-loop gait: phase-indexed 6-bit contact table, no network -----
    if args.gait_npz:
        npz_path = args.gait_npz
        if npz_path == "tripod":
            npz_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tripod_bit_demos.npz")
        d = np.load(npz_path, allow_pickle=True)
        period = int(d["period_steps"]) if "period_steps" in d else len(d["bits"])
        bits = np.asarray(d["bits"])[:period]
        gait_src = npz_path
        _tripod_spine_swap = os.path.basename(str(npz_path)).lower() == "tripod_bit_demos.npz"
    else:
        raw = np.loadtxt(args.gait_csv, delimiter=",")
        if raw.ndim != 2 or raw.shape[1] != 8:
            raise SystemExit(f"{args.gait_csv}: expected a [T, 8] Sim-DOF CSV, got {raw.shape}")
        # drop the wrap row if the CSV is written closed (row[-1] duplicates row[0]), so the
        # table is exactly one period -- same rule as extract_bit_demos.py (period = T - 1).
        if len(raw) > 1 and np.allclose(raw[0], raw[-1]):
            raw = raw[:-1]
        legs = raw[:, _LEG_COLS_BIT_ORDER]  # [T, 6]
        mid = 0.5 * (legs.max(axis=0) + legs.min(axis=0))  # per-leg two-level midpoint
        bits = (legs > mid).astype(np.int64)  # 1 = stance (foot down), matches extract_bit_demos.py
        gait_src = args.gait_csv
        _tripod_spine_swap = os.path.basename(str(args.gait_csv)).lower() == "tripod_extendedquad_sim.csv"

    gait_table = _gait_table_from_bits(bits)  # [P] int64
    P = int(gait_table.numel())
    stance_frac = np.asarray(bits, dtype=float)[:P].mean(axis=0)
    print(f"[play_discrete_closeup] open-loop gait: {gait_src}")
    print(
        f"[play_discrete_closeup]   period {P} steps ({P * args.gait_dt:.2f} s) | "
        f"distinct actions {sorted({int(v) for v in gait_table.tolist()})}"
    )
    print(f"[play_discrete_closeup]   per-leg stance fraction [FR,FL,MR,ML,BR,BL]: {np.round(stance_frac, 3).tolist()}")

    def policy_fn(obs):  # noqa: ARG001 -- open-loop: action depends only on episode phase
        phase = (base.episode_length_buf % P).long()
        return gait_table[phase]


_eye = np.asarray(args.eye, dtype=float)
_look = np.asarray(args.lookat, dtype=float)
# DiscreteBitsActionWrapper exposes the sim env as .base_env; .unwrapped does NOT
# reach the ManagerBasedRLEnv here (it stops at the wrapper), so ask for base_env first.
base = getattr(env, "base_env", None)
if base is None or not hasattr(base, "scene"):
    base = env
    while not hasattr(base, "scene") and hasattr(base, "env"):
        base = base.env
if not hasattr(base, "scene"):
    raise SystemExit("could not reach the sim env (no .scene); camera tracking impossible")

# --- tripod baseline: swap the scripted spine to WAVE 2 (anti-phase tripod body wave) ---
# The tripod LEG contact schedule (tripod_bit_demos.npz / tripod_extendedquad_sim.csv) is
# designed around the anti-phase tripod body wave (WAVE 2: BackLink = -FrontLink), NOT the
# analytic RL traveling wave (WAVE 1) the binary env plays by default. Swap it here so the
# video shows the same gait eval_protocol.py's BASE_tripod_csv_bits anchor scores. Single
# run, no restore needed.
if _tripod_spine_swap and not args.gait_csv_pos:
    _am = base.action_manager
    _sw = None
    try:
        _sw = _am.get_term("spine_wave")
    except Exception:  # noqa: BLE001 -- try the term mapping instead
        _terms = getattr(_am, "_terms", None) or getattr(_am, "terms", None) or {}
        _sw = _terms.get("spine_wave")
    if _sw is not None and hasattr(_sw, "set_waveform"):
        _sw.set_waveform(TRIPOD_SPINE_SIN_COEF, TRIPOD_SPINE_COS_COEF, TRIPOD_SPINE_OFFSET)
        print(
            "[play_discrete_closeup] tripod replay: spine swapped to WAVE 2 "
            "(anti-phase tripod body wave, BackLink = -FrontLink), matching eval_protocol BASE_tripod_csv_bits"
        )
    else:
        print("[play_discrete_closeup] WARNING: tripod replay but no spine_wave term to swap to WAVE 2")

if args.gait_csv_pos:
    # resolve targets into the live runtime DOF order (by name, never a hardcoded perm)
    # and map to env actions: action = (target - offset) / scale  (mirrors playReal.py).
    _jn = list(base.scene["robot"].data.joint_names)
    _missing = [n for n in _jn if n not in _pos_target_by_name]
    if _missing:
        raise SystemExit(f"--gait_csv_pos: no converted target for runtime joint(s) {_missing}")
    _off, _scl = _infer_joint_pos_offset_scale(base, len(_jn), device)
    _tbl = torch.zeros((P, len(_jn)), device=device, dtype=torch.float32)
    for _i, _nm in enumerate(_jn):
        _tbl[:, _i] = torch.as_tensor(_pos_target_by_name[_nm], device=device, dtype=torch.float32)
    _POS_ACTIONS = (_tbl - _off) / _scl
    print(f"[play_discrete_closeup]   runtime DOF order: {_jn}")
    print(f"[play_discrete_closeup]   offset[0:8]={_off.tolist()}  scale[0:8]={_scl.tolist()}")

# --- receding-horizon goal (opt-in; see the module docstring) --------------------------
# Assumed attribute names on the live UniformPose2dCommand term (verified against
# source/isaaclab/isaaclab/envs/mdp/commands/pose_2d_command.py): `pos_command_w`
# [num_envs, 3] world-frame target, `heading_command_w` [num_envs] world-frame heading.
# `_update_command()` rebuilds the base-frame `pos_command_b` / `heading_command_b` from
# these every step, so re-pinning the world buffers after env.step() is picked up on the
# next command_manager.compute(). NEEDS a live Isaac Sim run to confirm.
_pose_cmd = None
if args.receding_horizon:
    _pose_cmd = base.command_manager.get_term("pose_command")
    try:
        base.termination_manager.get_term_cfg("reach_goal").params["radius"] = -1.0
        print("[play_discrete_closeup] receding horizon ON: reach_goal neutralised (radius -> -1)")
    except (KeyError, ValueError):
        print("[play_discrete_closeup] receding horizon ON: no reach_goal termination to neutralise")
    # keep the cfg range consistent with the live buffers for any internal resample
    _pose_cmd.cfg.ranges.pos_x = (args.goal_ahead_m, args.goal_ahead_m)
    print(f"[play_discrete_closeup]   goal held {args.goal_ahead_m:.2f} m ahead (world +x) every step")


def _advance_receding_goal():
    """Re-pin pose_command --goal_ahead_m ahead of the robot along world +x.

    Mutates the live command term's world-frame buffers; ``command_manager.compute()`` on the
    next ``env.step()`` rebuilds the base-frame command (and the obs) from them. Re-seeds
    ``progress_to_goal``'s ``env._goal_prev_dist`` so the goal jump scores no spurious progress
    (mirrors the first-post-reset seeding in ``hexapod_goal_rewards.py``).
    """
    rp = base.scene["robot"].data.root_pos_w
    rp = rp.torch if hasattr(rp, "torch") else rp
    origins = base.scene.env_origins
    _pose_cmd.pos_command_w[:, 0] = rp[:, 0] + args.goal_ahead_m
    _pose_cmd.pos_command_w[:, 1] = origins[:, 1]  # hold the straight line to the goal (pos_y = 0)
    _pose_cmd.heading_command_w[:] = 0.0
    prev = getattr(base, "_goal_prev_dist", None)
    if prev is not None:
        new_dist = torch.norm(_pose_cmd.pos_command_w - rp[:, :3], dim=1)
        if prev.shape == new_dist.shape:
            base._goal_prev_dist = new_dist.clone()


try:
    from isaaclab_physx.renderers.kit_viewport_utils import set_kit_renderer_camera_view
except (ImportError, ModuleNotFoundError):
    set_kit_renderer_camera_view = None
    print("[play_discrete_closeup] WARNING: kit viewport utils unavailable -> camera will NOT track")


def _follow_cam():
    """Headless render never fires the tracking callback, so place the camera by hand each step."""
    rp = base.scene["robot"].data.root_pos_w
    rp = rp.torch if hasattr(rp, "torch") else rp
    o = rp[0].detach().cpu().numpy().astype(float)
    eye_w = o + _eye
    tgt_w = o + _look
    base.sim.set_camera_view(eye=tuple(eye_w), target=tuple(tgt_w))
    if set_kit_renderer_camera_view is not None:
        # this is the camera the video recorder actually reads from
        set_kit_renderer_camera_view(eye=eye_w, target=tgt_w, camera_prim_path="/OmniverseKit_Persp")


obs, _ = env.reset()
total_rew = torch.zeros(env.num_envs, device=device)
action_counts = torch.zeros(64, dtype=torch.long)
terminated_n = truncated_n = 0

for t in range(args.steps):
    _follow_cam()
    with torch.no_grad():
        a = policy_fn(obs)
    if not args.gait_csv_pos:  # discrete int actions only
        action_counts += torch.bincount(a.cpu(), minlength=64)
    obs, rew, terminated, truncated, _ = env.step(a)
    if args.receding_horizon:
        _advance_receding_goal()
        # rebuild base-frame command + obs from the new world target (dt=0.0: nothing
        # moves), mirroring eval_protocol.py's post-reset command/obs recompute.
        base.command_manager.compute(dt=0.0)
        obs = base.observation_manager.compute()
    total_rew += rew
    terminated_n += int(terminated.sum().item())
    truncated_n += int(truncated.sum().item())

    if t % 100 == 0:
        print(f"  step {t}: mean cum reward {total_rew.mean().item():+.3f}")

print(
    f"[play_discrete] {args.steps} steps x {env.num_envs} envs | mean total reward {total_rew.mean().item():+.3f} "
    f"| terminated {terminated_n} truncated {truncated_n}"
)
if not args.gait_csv_pos:
    top = torch.topk(action_counts, 8)
    print(
        "[play_discrete] top-8 greedy 6-bit patterns (int: count):",
        {f"{int(i):06b}": int(c) for i, c in zip(top.indices, top.values)},
    )
print(f"[play_discrete] video dir: {out_dir}")

env.close()
simulation_app.close()
