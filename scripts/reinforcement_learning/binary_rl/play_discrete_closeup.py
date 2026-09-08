# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Replay a trained discrete checkpoint on the binary hexapod env and record a CLOSEUP video.

Identical to play_discrete.py except the recording camera is repositioned by hand every
step (play_openloop_closeup.py's ``_follow_cam``): headless rendering never fires Isaac's
own asset-root tracking callback, so ``viewer.origin_type = "asset_root"`` silently leaves
the camera fixed and the robot walks out of frame.  Same env, same render pipeline, same
RecordVideo wrapper, same greedy-argmax policy loading -- only the camera placement differs.

Loads the Q-network weights from a skrl checkpoint (``agent_*.pt`` / ``best_agent.pt``),
runs the greedy policy (argmax Q, no exploration) on the Play variant of the task, and
records an MP4 through the regular render pipeline (real robot mesh + ground plane),
with the camera tracking the robot root for a closeup view.

Run (from IsaacLab root, .venv active):
  CUDA_VISIBLE_DEVICES=<idle> ./isaaclab.sh -p scripts/reinforcement_learning/play_discrete.py \
      --checkpoint runs_discrete/<exp>/checkpoints/best_agent.pt --video_length 600
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-Goal-Flat-Hexapod-Binary-Play-v0")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=600)
parser.add_argument("--video_length", type=int, default=600)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--out_dir", default="", help="video output dir (default: <checkpoint dir>/../videos_play)")
parser.add_argument("--eye", type=float, nargs=3, default=[0.9, 0.9, 0.45],
                    help="camera offset from the robot root (same default as the approved closeup videos)")
parser.add_argument("--lookat", type=float, nargs=3, default=[0.0, 0.0, 0.08],
                    help="look-at offset from the robot root")
parser.add_argument("--v_min", type=float, default=-10.0, help="C51 checkpoints only: support lower bound")
parser.add_argument("--v_max", type=float, default=10.0, help="C51 checkpoints only: support upper bound")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
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

torch.manual_seed(args.seed)

env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=args.num_envs)
env_cfg.seed = args.seed
# closeup camera: driven manually every step (see _follow_cam below)
env_cfg.viewer.origin_type = "world"
env_cfg.viewer.asset_name = "robot"
env_cfg.viewer.env_index = 0
env_cfg.viewer.eye = tuple(args.eye)
env_cfg.viewer.lookat = tuple(args.lookat)

out_dir = args.out_dir or os.path.join(os.path.dirname(os.path.abspath(args.checkpoint)), "..", "videos_play")
env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array")
env = gym.wrappers.RecordVideo(
    env,
    video_folder=out_dir,
    step_trigger=lambda step: step == 0,
    video_length=args.video_length,
    disable_logger=True,
)
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
state = { (k[4:] if k.startswith("net.") else k): v for k, v in state.items() }
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
        x = obs["policy"] if isinstance(obs, dict) else obs
        if obs_scaler is not None:
            x = obs_scaler(x, train=False)
        q = qnet(x)
        if support is not None:  # C51: greedy over expected Q of the distribution
            q = (torch.softmax(q.view(-1, 64, n_atoms), dim=-1) * support).sum(-1)
        a = torch.argmax(q, dim=1)
    action_counts += torch.bincount(a.cpu(), minlength=64)
    obs, rew, terminated, truncated, _ = env.step(a)
    total_rew += rew
    terminated_n += int(terminated.sum().item())
    truncated_n += int(truncated.sum().item())
    if t % 100 == 0:
        print(f"  step {t}: mean cum reward {total_rew.mean().item():+.3f}")

print(f"[play_discrete] {args.steps} steps x {env.num_envs} envs | mean total reward {total_rew.mean().item():+.3f} "
      f"| terminated {terminated_n} truncated {truncated_n}")
top = torch.topk(action_counts, 8)
print("[play_discrete] top-8 greedy 6-bit patterns (int: count):",
      {f"{int(i):06b}": int(c) for i, c in zip(top.indices, top.values)})
print(f"[play_discrete] video dir: {out_dir}")

env.close()
simulation_app.close()
