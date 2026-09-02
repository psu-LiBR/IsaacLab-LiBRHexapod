# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Discrete Soft Actor-Critic on the hexapod binary contact-bit env (single file).

Hand-written implementation following the standard discrete-SAC structure
(arXiv:1910.07207; CleanRL ``sac_atari.py`` layout): categorical actor + twin
Q-networks + soft target updates + automatic temperature tuning against a target
entropy of ``0.89 * ln(64)`` (the CleanRL stabilization).  No new dependencies --
pure torch on the existing .venv.

Env stack matches the first-wave DQN/DDQN/PPO trials exactly (same task defaults,
curriculum active, no config overrides) so the curves are comparable:

    ManagerBased env (Box(6) bit floats) -> DiscreteBitsActionWrapper (Discrete(64))

All nets are [128,128,128] + ELU (project convention).  Replay and updates run
fully on GPU: one env step stores num_envs transitions, then one gradient step on a
uniformly sampled batch (same replay ratio as the first-wave DQN trial).
``done = terminated | truncated`` masks bootstrapping (auto-reset envs return the
post-reset obs as next_obs; masking on truncation avoids bootstrapping off a teleport
at the cost of slight horizon pessimism -- same convention as the skrl trials).

Checkpoints store the actor under the "policy" key with ``net.*`` layer names, so
``play_discrete.py`` replays them greedily (argmax over logits) without changes.

Run (from IsaacLab root, .venv active):
  CUDA_VISIBLE_DEVICES=<idle> ./isaaclab.sh -p scripts/reinforcement_learning/train_sac_d.py \
      --num_envs 4096 --timesteps 100000
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-Goal-Flat-Hexapod-Binary-v0")
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--timesteps", type=int, default=100000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--batch_size", type=int, default=4096)
parser.add_argument("--memory_slots", type=int, default=500, help="replay slots per env")
parser.add_argument("--q_lr", type=float, default=3e-4)
parser.add_argument("--policy_lr", type=float, default=3e-4)
parser.add_argument("--alpha_lr", type=float, default=3e-4)
parser.add_argument("--discount", type=float, default=0.99)
parser.add_argument("--polyak", type=float, default=0.005)
parser.add_argument("--target_entropy_scale", type=float, default=0.89)
parser.add_argument("--random_timesteps", type=int, default=20)
parser.add_argument("--learning_starts", type=int, default=50)
parser.add_argument("--log_interval", type=int, default=50)
parser.add_argument("--checkpoint_interval", type=int, default=2000)
parser.add_argument("--out_dir", default="runs_discrete/sacd_trial_20260830")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import csv
import math
import os
import time
from collections import deque

import gymnasium as gym
import torch
import torch.nn.functional as F

import isaaclab_tasks  # noqa: F401

try:
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
except ImportError:
    from isaaclab_tasks.utils import parse_env_cfg

from discrete_action_wrapper import DiscreteBitsActionWrapper

torch.manual_seed(args.seed)

env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=args.num_envs)
env_cfg.seed = args.seed
env = gym.make(args.task, cfg=env_cfg)
env = DiscreteBitsActionWrapper(env, n_bits=6)
device = env.device
N = env.num_envs
N_ACT = env.n_actions
obs_dim = int(env.single_observation_space["policy"].shape[-1])
print(f"[train_sac_d] task={args.task} num_envs={N} obs_dim={obs_dim} actions={N_ACT}", flush=True)


def mlp(out_units: int) -> torch.nn.Module:
    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            layers, in_dim = [], obs_dim
            for h in (128, 128, 128):
                layers += [torch.nn.Linear(in_dim, h), torch.nn.ELU()]
                in_dim = h
            layers += [torch.nn.Linear(in_dim, out_units)]
            self.net = torch.nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)

    return Net().to(device)


actor = mlp(N_ACT)
qf1, qf2 = mlp(N_ACT), mlp(N_ACT)
qf1_t, qf2_t = mlp(N_ACT), mlp(N_ACT)
qf1_t.load_state_dict(qf1.state_dict())
qf2_t.load_state_dict(qf2.state_dict())
q_opt = torch.optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
a_opt = torch.optim.Adam(actor.parameters(), lr=args.policy_lr)
target_entropy = args.target_entropy_scale * math.log(N_ACT)
log_alpha = torch.zeros(1, requires_grad=True, device=device)
alpha_opt = torch.optim.Adam([log_alpha], lr=args.alpha_lr)

# --- replay buffer (GPU tensors) ---
CAP = args.memory_slots * N
buf_obs = torch.zeros(CAP, obs_dim, device=device)
buf_next = torch.zeros(CAP, obs_dim, device=device)
buf_act = torch.zeros(CAP, dtype=torch.long, device=device)
buf_rew = torch.zeros(CAP, device=device)
buf_done = torch.zeros(CAP, device=device)
buf_ptr, buf_full = 0, False


def buf_store(o, a, r, no, d):
    global buf_ptr, buf_full
    n = o.shape[0]
    idx = (torch.arange(n, device=device) + buf_ptr) % CAP
    buf_obs[idx] = o
    buf_act[idx] = a
    buf_rew[idx] = r
    buf_next[idx] = no
    buf_done[idx] = d.float()
    buf_ptr = int((buf_ptr + n) % CAP)
    buf_full = buf_full or buf_ptr < n


os.makedirs(os.path.join(args.out_dir, "checkpoints"), exist_ok=True)
csv_path = os.path.join(args.out_dir, "train_log.csv")
csv_f = open(csv_path, "w", newline="")
csv_w = csv.writer(csv_f)
csv_w.writerow(["step", "ep_return_mean", "ep_len_mean", "qf_loss", "actor_loss", "alpha", "entropy", "sps"])

obs, _ = env.reset()
obs_t = obs["policy"]
cur_ret = torch.zeros(N, device=device)
cur_len = torch.zeros(N, device=device)
ep_returns, ep_lens = deque(maxlen=200), deque(maxlen=200)
qf_loss_v = actor_loss_v = entropy_v = 0.0
t0, steps_t0 = time.time(), 0

for step in range(args.timesteps):
    with torch.no_grad():
        if step < args.random_timesteps:
            act = torch.randint(0, N_ACT, (N,), device=device)
        else:
            logits = actor(obs_t)
            act = torch.distributions.Categorical(logits=logits).sample()
    nobs, rew, terminated, truncated, _ = env.step(act)
    nobs_t = nobs["policy"]
    done = (terminated | truncated).float()
    buf_store(obs_t, act, rew, nobs_t, done)
    cur_ret += rew
    cur_len += 1
    dmask = done.bool()
    if dmask.any():
        ep_returns.extend(cur_ret[dmask].tolist())
        ep_lens.extend(cur_len[dmask].tolist())
        cur_ret[dmask] = 0.0
        cur_len[dmask] = 0.0
    obs_t = nobs_t

    if step >= args.learning_starts:
        hi = CAP if buf_full else buf_ptr
        idx = torch.randint(0, hi, (args.batch_size,), device=device)
        b_o, b_a, b_r = buf_obs[idx], buf_act[idx], buf_rew[idx]
        b_no, b_d = buf_next[idx], buf_done[idx]
        alpha = log_alpha.exp().detach()
        with torch.no_grad():
            nlogits = actor(b_no)
            nlogp = F.log_softmax(nlogits, dim=1)
            nprobs = nlogp.exp()
            minq_t = torch.min(qf1_t(b_no), qf2_t(b_no))
            v_next = (nprobs * (minq_t - alpha * nlogp)).sum(1)
            q_target = b_r + args.discount * (1.0 - b_d) * v_next
        q1 = qf1(b_o).gather(1, b_a.unsqueeze(1)).squeeze(1)
        q2 = qf2(b_o).gather(1, b_a.unsqueeze(1)).squeeze(1)
        qf_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        q_opt.zero_grad()
        qf_loss.backward()
        q_opt.step()

        logits = actor(b_o)
        logp = F.log_softmax(logits, dim=1)
        probs = logp.exp()
        with torch.no_grad():
            minq = torch.min(qf1(b_o), qf2(b_o))
        actor_loss = (probs * (log_alpha.exp().detach() * logp - minq)).sum(1).mean()
        a_opt.zero_grad()
        actor_loss.backward()
        a_opt.step()

        with torch.no_grad():
            entropy = -(probs * logp).sum(1).mean()
        alpha_loss = (probs.detach() * (-log_alpha.exp() * (logp.detach() + target_entropy))).sum(1).mean()
        alpha_opt.zero_grad()
        alpha_loss.backward()
        alpha_opt.step()

        with torch.no_grad():
            for pt, p in zip(qf1_t.parameters(), qf1.parameters()):
                pt.mul_(1 - args.polyak).add_(args.polyak * p)
            for pt, p in zip(qf2_t.parameters(), qf2.parameters()):
                pt.mul_(1 - args.polyak).add_(args.polyak * p)
        qf_loss_v, actor_loss_v, entropy_v = float(qf_loss), float(actor_loss), float(entropy)

    if step % args.log_interval == 0:
        sps = (step - steps_t0) * N / max(1e-9, time.time() - t0)
        t0, steps_t0 = time.time(), step
        rmean = sum(ep_returns) / len(ep_returns) if ep_returns else float("nan")
        lmean = sum(ep_lens) / len(ep_lens) if ep_lens else float("nan")
        csv_w.writerow([step, f"{rmean:.4f}", f"{lmean:.1f}", f"{qf_loss_v:.5f}", f"{actor_loss_v:.5f}",
                        f"{float(log_alpha.exp()):.5f}", f"{entropy_v:.4f}", f"{sps:.0f}"])
        csv_f.flush()
        print(f"[sac_d {step}/{args.timesteps}] ep_ret={rmean:+.3f} ep_len={lmean:.0f} "
              f"qf={qf_loss_v:.4f} pi={actor_loss_v:+.4f} alpha={float(log_alpha.exp()):.4f} "
              f"H={entropy_v:.3f}/{target_entropy:.3f} sps={sps:.0f}", flush=True)

    if step > 0 and step % args.checkpoint_interval == 0:
        torch.save({"policy": actor.state_dict(), "qf1": qf1.state_dict(), "qf2": qf2.state_dict(),
                    "log_alpha": log_alpha.detach().cpu(), "step": step},
                   os.path.join(args.out_dir, "checkpoints", f"agent_{step}.pt"))

torch.save({"policy": actor.state_dict(), "qf1": qf1.state_dict(), "qf2": qf2.state_dict(),
            "log_alpha": log_alpha.detach().cpu(), "step": args.timesteps},
           os.path.join(args.out_dir, "checkpoints", "agent_final.pt"))
csv_f.close()
print(f"[train_sac_d] done. outputs in {args.out_dir}", flush=True)
env.close()
simulation_app.close()
