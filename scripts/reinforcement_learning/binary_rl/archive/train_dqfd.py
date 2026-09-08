# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""DQfD (Deep Q-learning from Demonstrations) on the hexapod binary env (single file).

Hand-written after "Deep Q-learning from Demonstrations" (arXiv:1704.03732, 2017-04,
AAAI 2018, Hester et al.). Core per task spec: DQN + demonstration buffer +
large-margin classification loss + pretraining phase.

Demonstrations: the tripod 6-bit sequence (E5 npz: actions only, 50-step period) is
executed OPEN-LOOP for --demo_steps env steps across all envs, with per-env phase =
episode_length_buf % period (same clock convention as the E3 open-loop scorer), while
recording (s, a, r, s', done) into a permanent demo buffer. Phase 2 pretrains on
demo-only batches; phase 3 runs epsilon-greedy DQN where every batch mixes a fixed
--demo_frac of demo samples (margin loss applied to demo samples only).

Hyperparameters inherited from the first-wave DQN trial (same values unless noted):
lr 1e-3, gamma 0.99, polyak 0.005 every 10 updates, batch 4096, replay 500 slots/env,
Q-net MLP [128,128,128]+ELU -> 64, random_timesteps 20, learning_starts 50.

Deviations from the first-wave DQN / paper (AI choices, recorded per project rules):
- Double-DQN targets (paper uses double DQN; first wave ran DQN and DDQN separately).
- No n-step return term and no prioritized replay (uniform + fixed demo fraction);
  task spec asks for the DQfD core only. L2 reg via Adam weight_decay=1e-5 (paper
  lambda3). Margin m=0.8, margin weight 1.0 (paper values).
- Exploration eps 0.2 -> 0.05 over the first 10% then constant (DQN trial went
  1.0 -> 0.05 @ 20%; demos supply structured exploration, paper uses eps=0.01 fixed;
  final 0.05 kept equal to the DQN trial for comparability).
- Comparison anchor: launched only after the first-wave DQN 2h checkpoint existed;
  that checkpoint is the no-demonstration baseline at equal budget. DQfD itself
  starts FROM SCRATCH (pretraining from demos is the point of the algorithm);
  "use the DQN checkpoint as起点" is interpreted as comparison reference, not init.

Checkpoints store the Q-net under "policy" with net.* names -> play_discrete.py
replays them greedily (argmax over Q) unchanged.

Smoke:  CUDA_VISIBLE_DEVICES=<idle> python scripts/reinforcement_learning/train_dqfd.py \
            --num_envs 16 --timesteps 200 --demo_steps 60 --pretrain_steps 50 \
            --batch_size 256 --memory_slots 200 --checkpoint_interval 100
Trial:  CUDA_VISIBLE_DEVICES=<idle> python scripts/reinforcement_learning/train_dqfd.py \
            --num_envs 4096 --timesteps 100000
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-Goal-Flat-Hexapod-Binary-v0")
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--timesteps", type=int, default=100000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--demo_npz",
                    default=os.path.expanduser(
                        "~/isaac6/IsaacLab/runs_discrete/demos_tripod_bits/tripod_bit_demos.npz"))
parser.add_argument("--demo_steps", type=int, default=150, help="env steps of open-loop demo rollout")
parser.add_argument("--pretrain_steps", type=int, default=2000, help="demo-only gradient steps")
parser.add_argument("--demo_frac", type=float, default=0.25, help="demo fraction per RL batch")
parser.add_argument("--margin", type=float, default=0.8)
parser.add_argument("--margin_weight", type=float, default=1.0)
parser.add_argument("--l2", type=float, default=1e-5, help="Adam weight_decay (paper lambda3)")
parser.add_argument("--batch_size", type=int, default=4096)
parser.add_argument("--memory_slots", type=int, default=500, help="agent replay slots per env")
parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument("--discount", type=float, default=0.99)
parser.add_argument("--polyak", type=float, default=0.005)
parser.add_argument("--target_update_interval", type=int, default=10, help="updates between soft target syncs")
parser.add_argument("--random_timesteps", type=int, default=20)
parser.add_argument("--learning_starts", type=int, default=50)
parser.add_argument("--eps_initial", type=float, default=0.2)
parser.add_argument("--eps_final", type=float, default=0.05)
parser.add_argument("--eps_fraction", type=float, default=0.1)
parser.add_argument("--log_interval", type=int, default=50)
parser.add_argument("--checkpoint_interval", type=int, default=2000)
parser.add_argument("--out_dir", default="runs_discrete/dqfd_trial_20260830")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import csv
import time
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

import isaaclab_tasks  # noqa: F401

try:
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
except ImportError:
    from isaaclab_tasks.utils import parse_env_cfg

# Archived script: add the parent binary_rl/ dir to sys.path so the shared
# discrete_action_wrapper module still resolves when run from archive/.
import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from discrete_action_wrapper import DiscreteBitsActionWrapper  # noqa: E402

torch.manual_seed(args.seed)

env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=args.num_envs)
env_cfg.seed = args.seed
env = gym.make(args.task, cfg=env_cfg)
env = DiscreteBitsActionWrapper(env, n_bits=6)
device = env.device
N = env.num_envs
N_ACT = env.n_actions
obs_dim = int(env.single_observation_space["policy"].shape[-1])
print(f"[train_dqfd] task={args.task} num_envs={N} obs_dim={obs_dim} actions={N_ACT}", flush=True)

# ---- demo action sequence (E5 npz, self-verified on load) ----
demo_npz = np.load(args.demo_npz)
period = int(demo_npz["period_steps"])
demo_seq = torch.as_tensor(demo_npz["action_idx"][:period].astype(np.int64), device=device)
assert demo_seq.min() >= 0 and demo_seq.max() < N_ACT and len(demo_seq) == period
print(f"[train_dqfd] demo: period={period} distinct={sorted(set(demo_seq.tolist()))} "
      f"source={demo_npz['source_csv']}", flush=True)


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


q_net = mlp(N_ACT)
q_tgt = mlp(N_ACT)
q_tgt.load_state_dict(q_net.state_dict())
opt = torch.optim.Adam(q_net.parameters(), lr=args.learning_rate, weight_decay=args.l2)
n_updates = 0


def soft_target_maybe():
    global n_updates
    n_updates += 1
    if n_updates % args.target_update_interval == 0:
        with torch.no_grad():
            for pt, p in zip(q_tgt.parameters(), q_net.parameters()):
                pt.mul_(1 - args.polyak).add_(args.polyak * p)


class Buffer:
    def __init__(self, cap: int):
        self.cap = cap
        self.obs = torch.zeros(cap, obs_dim, device=device)
        self.next = torch.zeros(cap, obs_dim, device=device)
        self.act = torch.zeros(cap, dtype=torch.long, device=device)
        self.rew = torch.zeros(cap, device=device)
        self.done = torch.zeros(cap, device=device)
        self.ptr, self.full = 0, False

    def store(self, o, a, r, no, d):
        n = o.shape[0]
        idx = (torch.arange(n, device=device) + self.ptr) % self.cap
        self.obs[idx] = o
        self.act[idx] = a
        self.rew[idx] = r
        self.next[idx] = no
        self.done[idx] = d.float()
        self.ptr = int((self.ptr + n) % self.cap)
        self.full = self.full or self.ptr < n

    @property
    def size(self):
        return self.cap if self.full else self.ptr

    def sample(self, n: int):
        idx = torch.randint(0, self.size, (n,), device=device)
        return self.obs[idx], self.act[idx], self.rew[idx], self.next[idx], self.done[idx]


demo_buf = Buffer(args.demo_steps * N)
agent_buf = Buffer(args.memory_slots * N)


def dqfd_loss(batches):
    """batches = list of (obs, act, rew, next, done, is_demo). Returns td, margin, qmean."""
    b_o = torch.cat([b[0] for b in batches])
    b_a = torch.cat([b[1] for b in batches])
    b_r = torch.cat([b[2] for b in batches])
    b_no = torch.cat([b[3] for b in batches])
    b_d = torch.cat([b[4] for b in batches])
    demo_mask = torch.cat([torch.full((b[0].shape[0],), b[5], dtype=torch.bool, device=device)
                           for b in batches])
    with torch.no_grad():
        a_star = q_net(b_no).argmax(1)                       # double-DQN action selection
        q_next = q_tgt(b_no).gather(1, a_star.unsqueeze(1)).squeeze(1)
        y = b_r + args.discount * (1.0 - b_d) * q_next
    q_all = q_net(b_o)
    q_taken = q_all.gather(1, b_a.unsqueeze(1)).squeeze(1)
    td = F.mse_loss(q_taken, y)
    if demo_mask.any():
        qd = q_all[demo_mask]
        ad = b_a[demo_mask]
        margins = torch.full_like(qd, args.margin)
        margins.scatter_(1, ad.unsqueeze(1), 0.0)            # l(a_E, a): 0 at expert action
        j_e = (qd + margins).max(1).values - qd.gather(1, ad.unsqueeze(1)).squeeze(1)
        margin = j_e.mean()
    else:
        margin = torch.zeros((), device=device)
    loss = td + args.margin_weight * margin
    opt.zero_grad()
    loss.backward()
    opt.step()
    soft_target_maybe()
    return float(td), float(margin), float(q_taken.mean())


os.makedirs(os.path.join(args.out_dir, "checkpoints"), exist_ok=True)
csv_path = os.path.join(args.out_dir, "train_log.csv")
csv_f = open(csv_path, "w", newline="")
csv_w = csv.writer(csv_f)
csv_w.writerow(["phase", "step", "ep_return_mean", "ep_len_mean", "td_loss", "margin_loss",
                "eps", "q_mean", "sps"])

# ================= phase 1: open-loop demo collection =================
obs, _ = env.reset()
obs_t = obs["policy"]
demo_rew_sum, demo_falls = 0.0, 0
t0 = time.time()
for t in range(args.demo_steps):
    # fetch the clock fresh each step (auto-reset zeroes it per env; same as E3 convention)
    act = demo_seq[env.base_env.episode_length_buf % period]
    nobs, rew, terminated, truncated, _ = env.step(act)
    nobs_t = nobs["policy"]
    done = (terminated | truncated)
    demo_buf.store(obs_t, act, rew, nobs_t, done.float())
    demo_rew_sum += float(rew.mean())
    demo_falls += int(terminated.sum())
    obs_t = nobs_t
print(f"[dqfd demo] {demo_buf.size} transitions in {time.time()-t0:.0f}s | "
      f"mean step reward {demo_rew_sum/args.demo_steps:+.4f} | terminations {demo_falls} "
      f"({demo_falls/(args.demo_steps*N)*100:.2f}% of steps)", flush=True)
csv_w.writerow(["demo", args.demo_steps, f"{demo_rew_sum/args.demo_steps:.4f}", "", "", "",
                "", "", ""])

# ================= phase 2: pretraining on demonstrations =================
t0 = time.time()
for i in range(args.pretrain_steps):
    td, mg, qm = dqfd_loss([(*demo_buf.sample(args.batch_size), True)])
    if i % 100 == 0 or i == args.pretrain_steps - 1:
        print(f"[dqfd pretrain {i}/{args.pretrain_steps}] td={td:.5f} margin={mg:.5f} "
              f"q_mean={qm:+.3f}", flush=True)
        csv_w.writerow(["pretrain", i, "", "", f"{td:.5f}", f"{mg:.5f}", "", f"{qm:.3f}", ""])
        csv_f.flush()
print(f"[dqfd pretrain] {args.pretrain_steps} steps in {time.time()-t0:.0f}s", flush=True)
torch.save({"policy": q_net.state_dict(), "step": 0},
           os.path.join(args.out_dir, "checkpoints", "agent_pretrained.pt"))

# ================= phase 3: epsilon-greedy RL with mixed batches =================
obs, _ = env.reset()
obs_t = obs["policy"]
cur_ret = torch.zeros(N, device=device)
cur_len = torch.zeros(N, device=device)
ep_returns, ep_lens = deque(maxlen=200), deque(maxlen=200)
td_v = mg_v = qm_v = 0.0
eps_steps = max(1, int(args.eps_fraction * args.timesteps))
t0, steps_t0 = time.time(), 0

for step in range(args.timesteps):
    eps = args.eps_final + max(0.0, (args.eps_initial - args.eps_final) * (1 - step / eps_steps))
    with torch.no_grad():
        if step < args.random_timesteps:
            act = torch.randint(0, N_ACT, (N,), device=device)
        else:
            act = q_net(obs_t).argmax(1)
            explore = torch.rand(N, device=device) < eps
            act = torch.where(explore, torch.randint(0, N_ACT, (N,), device=device), act)
    nobs, rew, terminated, truncated, _ = env.step(act)
    nobs_t = nobs["policy"]
    done = (terminated | truncated).float()
    agent_buf.store(obs_t, act, rew, nobs_t, done)
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
        n_demo = int(args.batch_size * args.demo_frac)
        td_v, mg_v, qm_v = dqfd_loss([
            (*demo_buf.sample(n_demo), True),
            (*agent_buf.sample(args.batch_size - n_demo), False),
        ])

    if step % args.log_interval == 0:
        sps = (step - steps_t0) * N / max(1e-9, time.time() - t0)
        t0, steps_t0 = time.time(), step
        rmean = sum(ep_returns) / len(ep_returns) if ep_returns else float("nan")
        lmean = sum(ep_lens) / len(ep_lens) if ep_lens else float("nan")
        csv_w.writerow(["rl", step, f"{rmean:.4f}", f"{lmean:.1f}", f"{td_v:.5f}",
                        f"{mg_v:.5f}", f"{eps:.4f}", f"{qm_v:.3f}", f"{sps:.0f}"])
        csv_f.flush()
        print(f"[dqfd {step}/{args.timesteps}] ep_ret={rmean:+.3f} ep_len={lmean:.0f} "
              f"td={td_v:.4f} margin={mg_v:.4f} eps={eps:.3f} q={qm_v:+.3f} sps={sps:.0f}",
              flush=True)

    if step > 0 and step % args.checkpoint_interval == 0:
        torch.save({"policy": q_net.state_dict(), "step": step},
                   os.path.join(args.out_dir, "checkpoints", f"agent_{step}.pt"))

torch.save({"policy": q_net.state_dict(), "step": args.timesteps},
           os.path.join(args.out_dir, "checkpoints", "agent_final.pt"))
csv_f.close()
print(f"[train_dqfd] done. outputs in {args.out_dir}", flush=True)
env.close()
simulation_app.close()
