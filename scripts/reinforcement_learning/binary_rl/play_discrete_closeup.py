# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Replay a discrete policy OR an open-loop gait on the binary hexapod env and record a closeup video.

Same as play_discrete.py except the camera is repositioned by hand every step (``_follow_cam``): headless
rendering never fires Isaac's asset-root tracking callback, so the robot would otherwise walk out of frame.

Pass exactly one action source:

  * ``--checkpoint <agent_*.pt>``: a trained skrl checkpoint, run greedily (argmax Q / categorical logits).
  * ``--gait_npz tripod``: the ``tripod_bit_demos.npz`` next to this script, or ``--gait_npz <path.npz>`` for
    any table from ``extract_bit_demos.py``.
  * ``--gait_csv <path.csv>``: a sim-DOF gait CSV whose leg columns are thresholded to contact bits (same rule
    as ``extract_bit_demos.py``). A frozen (constant) leg column is held at LIFT, so gaits with intentionally
    frozen stance legs shuffle in place; use ``--gait_csv_pos`` for those.
  * ``--gait_csv_pos <path.csv>``: a raw baseline gait CSV (e.g. ``hexapod-assets/Sim Gaits/forward3_lleg*_sim.csv``)
    replayed as literal per-joint position targets, like ``playReal.py``. The binary env cannot take 8 continuous
    targets, so this forces ``--task Isaac-Velocity-Flat-Hexapod-Play-v0`` unless ``--task`` is given.

``--gait_csv_pos`` conversion
-----------------------------
The raw CSV has N rows x 8 columns, no header, radians, in the old pre-HexapI sim-DOF order ``[BackLink,
FrontLink, MiddleLeft, MiddleRight, BackLeft, BackRight, FrontLeft, FrontRight]`` and the old (negative) leg
convention. The constants mirror ``hexapod_binary_env_cfg.py``.

  * Legs: ``v = -raw``, snapped to the nearer of ``STANCE_POS`` / ``LIFT_POS``, routed by joint name.
  * Spine: the CSV columns are discarded and regenerated as the env's Wave 1 (``SpineSineAction``).
  * Targets are emitted in the live articulation DOF order (resolved by name from ``robot.data.joint_names``)
    and mapped to env actions via ``action = (target - default_joint_pos) / scale``.

The gait-bits path drives the same env and bit decode as the ``BASE_tripod_csv_bits`` baseline of
``eval_protocol.py``. For the reference tripod replay (``--gait_npz tripod`` or ``--gait_csv
.../tripod_extendedquad_sim.csv``) the scripted spine is also swapped from Wave 1 to the anti-phase tripod
wave (Wave 2), exactly as that baseline does.

Receding-horizon goal (opt-in: ``--receding_horizon`` + ``--goal_ahead_m``)
---------------------------------------------------------------------------
By default the episode ends when the robot reaches the goal ~2 m ahead, so a play run is one dash and stop. With
``--receding_horizon`` the ``pose_command`` target is re-pinned ``--goal_ahead_m`` (default 2.0) ahead of the
robot along world +x every step, so it walks continuously, which is useful for judging a gait over many cycles.
This mutates the running command / termination managers of this play script only; no env config changes.

  * ``eval_protocol.py`` does not need it: it pins a far goal, takes no resets and neutralises every termination.
  * ``progress_to_goal`` telescopes over an episode, so moving the goal mid-episode would score a spurious
    negative step; the script re-seeds ``env._goal_prev_dist`` on each jump.
  * ``progress`` reads ~0 every step (the goal recedes with the robot) and ``reach_goal`` is disabled (radius -1).
  * It matches ``goal.mode: receding`` of ``scripts/sim2real_transfer`` (``--goal_ahead_m`` = ``lookahead_m``).
  * It is rejected with ``--gait_csv_pos``, whose continuous velocity task has no ``pose_command``.

Sim-to-real diagnostic export (opt-in: ``--checkpoint`` + ``--export_gait_csv PATH``)
-------------------------------------------------------------------------------------
Writes a real-robot-units CSV of what the checkpoint commands for one gait cycle (50 steps at 50 Hz, i.e.
``GAIT_PERIOD_S`` = 1.0 s), by default the first 50 steps after ``env.reset()``, in the same rollout as the
video. Each row goes through the path a real deployment uses (``sim2real.binary_profile.BinaryActionAdapter``,
``sim2real.joint_mapping.JointMapping``), so the CSV can be replayed open-loop on the servos to tell a
policy/conversion bug from a closed-loop control, sensor or localization bug. Only valid with ``--checkpoint``.
``--deployment_config`` selects the sim2real YAML (default ``deployment.binary.example.yaml``).
``--gait_csv_start_step`` moves the window later, e.g. 50 records the second cycle, since every episode first
free-falls ~0.145 m onto its legs and cycle 1 may carry that landing transient. ``t_s`` is always the absolute
time since reset. The script warns if env 0 resets before the window closes.

A second file, ``<PATH minus .csv>_actual.csv``, holds the MEASURED joint positions of env 0 at the end of each
step (the response to that step's command). Its ``<joint>_real_rad`` columns use the same real-robot order and
conversion as the commanded CSV but are not soft-limit clipped, ``<joint>_sim_rad`` are the raw sim angles, and
joints are remapped by name from the live articulation order (the script refuses to run if the sets differ).
``root_x_m/root_y_m/root_z_m`` (env-local base position) help choose gait windows and ``root_quat_0..3`` (raw
``robot.data.root_quat_w`` order) allow comparing body tilt against a real IMU log.

Run (from the repo root):
  isaaclab.bat -p scripts/reinforcement_learning/binary_rl/play_discrete_closeup.py ^
      --gait_npz tripod --video_length 600
  isaaclab.bat -p scripts/reinforcement_learning/binary_rl/play_discrete_closeup.py ^
      --checkpoint runs_binary/<exp>/checkpoints/best_agent.pt --video_length 600
  isaaclab.bat -p scripts/reinforcement_learning/binary_rl/play_discrete_closeup.py ^
      --gait_csv_pos "hexapod-assets/Sim Gaits/forward3_lleg35_amp65_sim.csv" --video_length 600
  isaaclab.bat -p scripts/reinforcement_learning/binary_rl/play_discrete_closeup.py ^
      --checkpoint runs_binary/<exp>/checkpoints/best_agent.pt ^
      --export_gait_csv runs_binary/<exp>/onnx_bundle/BEST_gait.csv
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
parser.add_argument(
    "--export_gait_csv",
    default="",
    help="opt-in sim-to-real diagnostic: write a real-robot-units CSV of the loaded "
    "--checkpoint's own commanded gait for the first 50 steps after reset (one gait "
    "cycle). Only valid with --checkpoint; rejected with --gait_npz/--gait_csv/"
    "--gait_csv_pos (those are already literal CSVs/NPZs). See the module docstring.",
)
parser.add_argument(
    "--deployment_config",
    default="",
    help="sim2real deployment YAML used to convert --export_gait_csv rows to real-robot "
    "units (default: scripts/sim2real_transfer/config/deployment.binary.example.yaml, "
    "resolved relative to the repo root). Ignored unless --export_gait_csv is given.",
)
parser.add_argument(
    "--gait_csv_start_step",
    type=int,
    default=0,
    help="opt-in with --export_gait_csv: env-step index (0-based) at which to start "
    "recording, instead of the default 0 (immediately after reset). E.g. 50 records the "
    "SECOND gait cycle (steps 50-99) rather than the first. The CSV's t_s column stays "
    "absolute elapsed time since reset (step_index * step_dt) -- it is NOT renormalized to "
    "start at 0. Only valid together with --export_gait_csv.",
)
parser.add_argument(
    "--gait_csv_cycles",
    type=int,
    default=1,
    help="opt-in with --export_gait_csv: number of consecutive 50-step gait cycles to record "
    "(default 1). Use >1 to check whether the policy's contact pattern repeats cycle to cycle "
    "(a robot-ready loop needs it to). Needs --steps >= start_step + 50 * cycles. Only valid "
    "together with --export_gait_csv.",
)
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

if args.export_gait_csv:
    if not args.checkpoint:
        parser.error(
            "--export_gait_csv requires --checkpoint (it exports that checkpoint's own "
            "greedy-policy commands, not an open-loop gait replay)"
        )
    if args.gait_npz or args.gait_csv or args.gait_csv_pos:
        parser.error(
            "--export_gait_csv cannot be combined with --gait_npz/--gait_csv/--gait_csv_pos "
            "-- those flags already replay a literal CSV/NPZ gait you can read directly"
        )

if args.gait_csv_start_step != 0 and not args.export_gait_csv:
    parser.error("--gait_csv_start_step is only meaningful together with --export_gait_csv")
if args.gait_csv_start_step < 0:
    parser.error("--gait_csv_start_step must be >= 0")
if args.gait_csv_cycles < 1:
    parser.error("--gait_csv_cycles must be >= 1")
if args.gait_csv_cycles != 1 and not args.export_gait_csv:
    parser.error("--gait_csv_cycles is only meaningful together with --export_gait_csv")

# --gait_csv_pos needs a continuous JointPositionAction env, so default to the one playReal.py uses
if args.gait_csv_pos and args.task == parser.get_default("task"):
    args.task = "Isaac-Velocity-Flat-Hexapod-Play-v0"

args.headless = True
args.enable_cameras = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import csv
import os
import sys

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
    GAIT_PERIOD_S,
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


# The architecture is detected from the checkpoint's state-dict keys/shapes:
#   - plain MLP [128,128,128] + ELU -> 64      (DQN / DDQN / PPO)
#   - Linear -> LayerNorm -> ReLU blocks -> 64 (PQN)
#   - plain MLP -> 64 * atoms logits           (C51; greedy uses the expected Q)
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

# True for the reference tripod replay: its leg schedule is designed around the anti-phase Wave 2, so the spine
# is swapped to it after the env is built, as in eval_protocol.py's BASE_tripod_csv_bits.
_tripod_spine_swap = False


def _gait_table_from_bits(bits_np: np.ndarray) -> torch.Tensor:
    """[P, 6] {0,1} contact bits (already exactly one period) -> [P] int64 action per phase."""
    if bits_np.ndim != 2 or bits_np.shape[1] != 6:
        raise SystemExit(f"gait bits must be [P, 6], got {bits_np.shape}")
    bt = torch.as_tensor(bits_np.astype(np.int64), device=device)
    shifts = torch.arange(6, device=device, dtype=torch.long)
    return (bt * (1 << shifts)).sum(dim=1)  # [P]


# --- raw CSV -> per-joint position targets (--gait_csv_pos) ---
# Keep in sync with hexapod_binary_env_cfg.py (STANCE_POS, LIFT_POS, GAIT_PERIOD_S and Wave 1, the spine below):
#   q(t) = OFFSET + AMPLITUDE * sin(2*pi*t / PERIOD + PHASE), AMPLITUDE = -A_SPINE on both joints,
#   BackLink phase +pi/2 (a cosine), FrontLink phase 0.
_STANCE_POS = 0.460194236365692  # leg joint angle, foot DOWN [rad]
_LIFT_POS = 1.180398216278  # leg joint angle, foot UP [rad]
_GAIT_PERIOD_S = 1.0
_SPINE_AMPLITUDE = {"BackLink_Joint": -0.9162978573, "FrontLink_Joint": -0.9162978573}
_SPINE_PHASE = {"BackLink_Joint": 1.5707963267948966, "FrontLink_Joint": 0.0}
_SPINE_OFFSET = {"BackLink_Joint": 0.0, "FrontLink_Joint": 0.0}
# old pre-HexapI sim-DOF CSV column of each leg joint
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
    # Masked PPO stores its legal-action mask as a buffer ("_action_mask") beside the Sequential
    # layers. Pull it out so the net loads, and restrict the greedy argmax to it like eval_protocol.
    action_mask = state.pop("_action_mask", None)
    state = {k: v for k, v in state.items() if k.split(".")[0].isdigit()}
    qnet, out_dim, arch = build_qnet(state, obs_dim)
    qnet = qnet.to(device)
    qnet.load_state_dict(state)
    qnet.eval()
    legal_mask = action_mask.to(device=device, dtype=torch.bool) if action_mask is not None else None
    if legal_mask is not None:
        print(f"[play_discrete] masked policy: {int(legal_mask.sum())} legal actions")
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
        if legal_mask is not None:  # masked PPO: never pick an action outside the legal set
            q = q.masked_fill(~legal_mask, float("-inf"))
        return torch.argmax(q, dim=1)

    # --- opt-in: build the sim2real conversion path for --export_gait_csv ---------------
    # See the module docstring. Imported lazily (only when requested) since sim2real is a
    # separate plain-Python package with its own dependency footprint.
    _export_adapter = None
    _export_mapping = None
    if args.export_gait_csv:
        _sim2real_dir = os.path.abspath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "scripts", "sim2real_transfer")
        )
        if _sim2real_dir not in sys.path:
            sys.path.insert(0, _sim2real_dir)
        from sim2real.binary_profile import BinaryActionAdapter
        from sim2real.deployment_config import load_deployment_config
        from sim2real.joint_mapping import JointMapping

        _repo_root = os.path.abspath(os.path.join(_sim2real_dir, "..", ".."))
        _deploy_cfg_path = args.deployment_config or os.path.join(
            _repo_root, "scripts", "sim2real_transfer", "config", "deployment.binary.example.yaml"
        )
        _deploy_cfg = load_deployment_config(_deploy_cfg_path)
        if _deploy_cfg.binary is None:
            raise SystemExit(
                f"--export_gait_csv: {_deploy_cfg_path} has no 'binary' section -- pass a "
                "binary-profile deployment config (see deployment.binary.example.yaml)"
            )
        # The CSV leg-bit columns come straight from DiscreteBitsActionWrapper.decode(), so the deployment
        # config's leg_joint_order must match, or the columns would be mislabeled.
        _expected_leg_order = ["FrontRight", "FrontLeft", "MiddleRight", "MiddleLeft", "BackRight", "BackLeft"]
        if list(_deploy_cfg.binary.leg_joint_order) != _expected_leg_order:
            raise SystemExit(
                f"--export_gait_csv: {_deploy_cfg_path} binary.leg_joint_order "
                f"{_deploy_cfg.binary.leg_joint_order} != the DiscreteBitsActionWrapper bit order "
                f"{_expected_leg_order} -- the CSV leg-bit columns would be silently mislabeled"
            )
        _export_adapter = BinaryActionAdapter(_deploy_cfg.binary, _deploy_cfg.joints.sim_order)
        _export_mapping = JointMapping(_deploy_cfg.joints)
        _export_front_real_idx = _deploy_cfg.joints.real_order.index("FrontLink")
        _export_back_real_idx = _deploy_cfg.joints.real_order.index("BackLink")
        _export_leg_real_idx = {name: _deploy_cfg.joints.real_order.index(name) for name in _expected_leg_order}
        print(f"[play_discrete_closeup] --export_gait_csv: loaded deployment config {_deploy_cfg_path}")

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
        # drop the wrap row of a closed CSV (row[-1] == row[0]), as extract_bit_demos.py does
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
# the wrapper exposes the sim env as .base_env; .unwrapped stops at the wrapper
base = getattr(env, "base_env", None)
if base is None or not hasattr(base, "scene"):
    base = env
    while not hasattr(base, "scene") and hasattr(base, "env"):
        base = base.env
if not hasattr(base, "scene"):
    raise SystemExit("could not reach the sim env (no .scene); camera tracking impossible")

# Tripod baseline: swap the scripted spine to Wave 2 (anti-phase), so the video shows the gait that
# eval_protocol.py's BASE_tripod_csv_bits scores. Single run, so no restore.
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
    # keep the cfg range consistent with the live buffers
    _pose_cmd.cfg.ranges.pos_x = (args.goal_ahead_m, args.goal_ahead_m)
    print(f"[play_discrete_closeup]   goal held {args.goal_ahead_m:.2f} m ahead (world +x) every step")


def _advance_receding_goal():
    """Re-pin pose_command --goal_ahead_m ahead of the robot along world +x.

    Mutates the command term's world-frame buffers (the next ``env.step()`` rebuilds the base-frame command
    from them) and re-seeds ``env._goal_prev_dist`` so the jump scores no spurious progress.
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
        # the camera the video recorder reads from
        set_kit_renderer_camera_view(eye=eye_w, target=tgt_w, camera_prim_path="/OmniverseKit_Persp")


# --export_gait_csv bookkeeping (see the module docstring)
_export_rows: list[dict] | None = None
_export_actual_rows: list[dict] | None = None
_export_start_step = _export_end_step = _export_safety_end_step = None
_export_env0_reset_steps: list[int] = []
if args.export_gait_csv:
    # Remap the live articulation joints (named "FrontLink_Joint") by name into the deployment sim_order, and
    # refuse to run if the sets differ.
    _live_names = [str(n) for n in base.scene["robot"].data.joint_names]
    _live_key = [n.removesuffix("_Joint") for n in _live_names]
    _cfg_sim_order = list(_deploy_cfg.joints.sim_order)
    if sorted(_live_key) != sorted(_cfg_sim_order):
        raise SystemExit(
            f"--export_gait_csv: live sim joints {_live_names} do not match deployment sim_order "
            f"{_cfg_sim_order}; the actual-position CSV would be mislabeled"
        )
    _export_live_idx = np.array([_live_key.index(n) for n in _cfg_sim_order], dtype=np.int64)
    _export_real_cols = ["FrontLink", "BackLink"] + list(_expected_leg_order)
    print(f"[play_discrete_closeup] --export_gait_csv: live sim joint order     : {_live_names}")
    print(f"[play_discrete_closeup] --export_gait_csv: deployment sim_order    : {_cfg_sim_order}")
    print(
        "[play_discrete_closeup] --export_gait_csv: sim order "
        + ("matches" if _live_key == _cfg_sim_order else "DIFFERS from (remapped by name)")
        + " deployment sim_order"
    )
    print(
        f"[play_discrete_closeup] --export_gait_csv: CSV column order {_export_real_cols} "
        + ("matches" if _export_real_cols == list(_deploy_cfg.joints.real_order) else "DIFFERS from")
        + f" deployment real_order {list(_deploy_cfg.joints.real_order)}"
    )
    _export_real_order_idx = {n: list(_deploy_cfg.joints.real_order).index(n) for n in _export_real_cols}
    _export_actual_rows = []
    _export_step_dt = float(base.step_dt)
    _export_cycle_steps = round(GAIT_PERIOD_S / _export_step_dt)
    _export_num_steps = _export_cycle_steps * args.gait_csv_cycles
    if abs(_export_cycle_steps * _export_step_dt - GAIT_PERIOD_S) > 1e-6:
        print(
            f"[play_discrete_closeup] WARNING: --export_gait_csv: env step_dt={_export_step_dt:.6f}s does "
            f"not evenly divide GAIT_PERIOD_S={GAIT_PERIOD_S}s ({_export_cycle_steps} steps covers "
            f"{_export_cycle_steps * _export_step_dt:.4f}s)"
        )
    _export_start_step = args.gait_csv_start_step
    _export_end_step = _export_start_step + _export_num_steps  # exclusive
    if _export_end_step > args.steps:
        raise SystemExit(
            f"--export_gait_csv: recording window ends at step {_export_end_step} but --steps is "
            f"{args.steps}; raise --steps (and --video_length) to at least {_export_end_step}"
        )
    # Safety window covers both "the first 100 steps" (the usual landing-transient concern)
    # AND the actual export window itself, whichever is larger -- so a --gait_csv_start_step
    # past 100 is still fully covered.
    _export_safety_end_step = max(100, _export_end_step)
    _export_rows = []
    _cycle_note = "one gait cycle from reset" if _export_start_step == 0 else "a LATER cycle, not the first"
    print(
        f"[play_discrete_closeup] --export_gait_csv: recording {_export_num_steps} steps "
        f"(steps {_export_start_step}-{_export_end_step - 1}, "
        f"t={_export_start_step * _export_step_dt:.2f}-{(_export_end_step - 1) * _export_step_dt:.2f}s, "
        f"{_cycle_note}) from env 0"
    )

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
    if _export_rows is not None and _export_start_step <= t < _export_end_step:
        # env-0 bits as commanded this step (same policy_fn call as the video)
        bits_pm1 = env.decode(a)[0].detach().cpu().numpy()  # [6] +-1, DiscreteBitsActionWrapper order
        t_seconds = t * _export_step_dt  # absolute elapsed time since reset, not window-relative
        sim_target = _export_adapter.targets_sim(bits_pm1, t_seconds)
        real_rad = _export_mapping.sim_rad_to_real_rad(_export_mapping.clip_to_soft_limits(sim_target))
        raw_bits01 = (bits_pm1 > 0.0).astype(int)
        row = {"step": t, "t_s": t_seconds}
        for _name, _bit in zip(_expected_leg_order, raw_bits01):
            row[f"{_name}_bit"] = int(_bit)
        row["FrontLink_real_rad"] = float(real_rad[_export_front_real_idx])
        row["BackLink_real_rad"] = float(real_rad[_export_back_real_idx])
        for _name in _expected_leg_order:
            row[f"{_name}_real_rad"] = float(real_rad[_export_leg_real_idx[_name]])
        _export_rows.append(row)
    obs, rew, terminated, truncated, _ = env.step(a)
    if _export_actual_rows is not None and _export_start_step <= t < _export_end_step:
        # Measured joint positions at the END of this step, i.e. the sim's response to the
        # command recorded for the same step index in the commanded CSV.
        _q = base.scene["robot"].data.joint_pos
        _q = _q.torch if hasattr(_q, "torch") else _q
        _q_sim = _q[0].detach().cpu().numpy().astype(np.float64)[_export_live_idx]  # deployment sim_order
        _q_real = _export_mapping.sim_rad_to_real_rad(_q_sim)  # same conversion as the commands, unclipped
        _rp = base.scene["robot"].data.root_pos_w
        _rp = _rp.torch if hasattr(_rp, "torch") else _rp
        _rp = (_rp[0] - base.scene.env_origins[0]).detach().cpu().numpy().astype(np.float64)
        _arow = {
            "step": t,
            "t_s": t * _export_step_dt,
            "root_x_m": float(_rp[0]),  # env-local base position at the end of the step (progress + landing)
            "root_y_m": float(_rp[1]),
            "root_z_m": float(_rp[2]),
        }
        _rq = base.scene["robot"].data.root_quat_w
        _rq = _rq.torch if hasattr(_rq, "torch") else _rq
        # raw base orientation exactly as the sim returns it (component order is the sim's own; see module docstring)
        for _qi, _qv in enumerate(_rq[0].detach().cpu().numpy().astype(np.float64)):
            _arow[f"root_quat_{_qi}"] = float(_qv)
        for _name in _export_real_cols:
            _arow[f"{_name}_real_rad"] = float(_q_real[_export_real_order_idx[_name]])
        for _name in _export_real_cols:
            _arow[f"{_name}_sim_rad"] = float(_q_sim[_cfg_sim_order.index(_name)])
        _export_actual_rows.append(_arow)
    # A reset of env 0 inside the safety window means the export window is not one continuous rollout.
    if _export_safety_end_step is not None and t < _export_safety_end_step:
        if bool(terminated[0]) or bool(truncated[0]):
            _export_env0_reset_steps.append(t)
            print(
                f"[play_discrete_closeup] WARNING: env 0 terminated/truncated at step {t} "
                f"(within safety window steps 0-{_export_safety_end_step - 1}) -- the requested "
                "export window is NOT part of one continuous rollout; the CSV may be misleading"
            )
    if args.receding_horizon:
        _advance_receding_goal()
        # rebuild the base-frame command and obs from the new world target (dt=0: nothing moves)
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
if _export_safety_end_step is not None:
    if _export_env0_reset_steps:
        print(
            f"[play_discrete_closeup] WARNING: env 0 reset/terminated within steps "
            f"0-{_export_safety_end_step - 1} at step(s) {_export_env0_reset_steps}"
        )
    else:
        print(
            f"[play_discrete_closeup] env 0: no reset/termination detected within steps 0-{_export_safety_end_step - 1}"
        )
if not args.gait_csv_pos:
    top = torch.topk(action_counts, 8)
    print(
        "[play_discrete] top-8 greedy 6-bit patterns (int: count):",
        {f"{int(i):06b}": int(c) for i, c in zip(top.indices, top.values)},
    )
print(f"[play_discrete] video dir: {out_dir}")

if _export_rows is not None:
    _export_dir = os.path.dirname(os.path.abspath(args.export_gait_csv))
    if _export_dir:
        os.makedirs(_export_dir, exist_ok=True)
    _fieldnames = list(_export_rows[0].keys()) if _export_rows else []
    with open(args.export_gait_csv, "w", newline="") as _f:
        _writer = csv.DictWriter(_f, fieldnames=_fieldnames)
        _writer.writeheader()
        _writer.writerows(_export_rows)
    print(
        f"[play_discrete_closeup] --export_gait_csv: wrote {len(_export_rows)} rows -> "
        f"{os.path.abspath(args.export_gait_csv)}"
    )
    # companion file: <name>.csv -> <name>_actual.csv (run_discrete_pipeline.py relies on this name)
    _actual_path = os.path.splitext(os.path.abspath(args.export_gait_csv))[0] + "_actual.csv"
    _actual_fields = list(_export_actual_rows[0].keys()) if _export_actual_rows else []
    with open(_actual_path, "w", newline="") as _f:
        _writer = csv.DictWriter(_f, fieldnames=_actual_fields)
        _writer.writeheader()
        _writer.writerows(_export_actual_rows)
    print(
        f"[play_discrete_closeup] --export_gait_csv: wrote {len(_export_actual_rows)} actual-position rows -> "
        f"{_actual_path}"
    )

    # Robot-ready file for LiBR-hexapi/scripts/run_opt_gait.py / physics_gait_di.py, laid out like
    # analysis/export_optimized_gaits.py: no header row, '#' comment lines, columns body_1, body_2, FR, FL, MR, ML,
    # HR, HL. physics_gait_di.py negates body_1/body_2 before converting to encoder counts, so body_k = -real_rad
    # of the commanded FrontLink/BackLink; legs are stored as 0/1 contact bits.
    _robot_path = os.path.splitext(os.path.abspath(args.export_gait_csv))[0] + "_robot.csv"
    if _export_rows:
        _body = np.array([[-r["FrontLink_real_rad"], -r["BackLink_real_rad"]] for r in _export_rows])
        _bits = np.array([[r[f"{n}_bit"] for n in _expected_leg_order] for r in _export_rows], dtype=int)
        _hdr = [
            "play_discrete_closeup.py --export_gait_csv | RL policy gait (sim rollout, env 0)",
            f"checkpoint: {args.checkpoint}",
            f"steps {_export_start_step}-{_export_end_step - 1} at {1.0 / _export_step_dt:.0f} Hz "
            f"({args.gait_csv_cycles} cycle(s) of {_export_cycle_steps} rows)",
            "contact order (cyclic, [FR FL MR ML HR HL]): "
            + " -> ".join(dict.fromkeys("".join(str(int(b)) for b in row) for row in _bits)),
            "body: scripted RL spine wave (Wave 1), raw/unnegated like export_optimized_gaits.py",
            "columns: body_1, body_2, FR, FL, MR, ML, HR, HL",
        ]
        np.savetxt(
            _robot_path,
            np.column_stack([_body, _bits]),
            delimiter=",",
            fmt=["%.10f", "%.10f"] + ["%d"] * 6,
            header=chr(10).join(_hdr),
            comments="# ",
        )
        print(f"[play_discrete_closeup] --export_gait_csv: wrote robot-ready gait ({len(_bits)} rows) -> {_robot_path}")
        if args.gait_csv_cycles > 1:
            _per = _export_cycle_steps
            _same = [
                bool(np.array_equal(_bits[k * _per : (k + 1) * _per], _bits[:_per]))
                for k in range(1, args.gait_csv_cycles)
            ]
            _diff = [
                int((_bits[k * _per : (k + 1) * _per] != _bits[:_per]).any(axis=1).sum())
                for k in range(1, args.gait_csv_cycles)
            ]
            print(
                f"[play_discrete_closeup] periodicity: cycle k identical to cycle 1 (k=2..): {_same}; "
                f"rows differing from cycle 1: {_diff}"
            )

env.close()
simulation_app.close()
