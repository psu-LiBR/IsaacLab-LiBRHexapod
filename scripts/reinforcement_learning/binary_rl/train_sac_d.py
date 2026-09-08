# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Discrete Soft Actor-Critic (SAC-D) on the hexapod binary contact-bit env.

Categorical actor + twin Q-networks (one Q-value per 6-bit gait pattern) + soft target
updates + automatic temperature tuning. Single-file, pure torch on the existing env --
skrl ships no discrete SAC, and the discrete case only needs two model swaps and two
closed-form expectations vs. the continuous algorithm.

References
---------
* Christodoulou 2019, "Soft Actor-Critic for Discrete Action Settings" (arXiv:1910.07207)
  -- the exact-expectation critic target ``V(s) = sum_a pi(a|s) [min_i Q_i(s,a) -
  alpha log pi(a|s)]`` and actor loss ``sum_a pi(a|s) [alpha log pi(a|s) - min_i Q_i]``.
* CleanRL ``sac_atari.py`` -- layout and the ``0.89 * ln|A|`` target-entropy default.
* Sabatini, Li, Hutter 2026 (arXiv:2605.24975) -- the two off-policy fixes ported here:
  **timeout-aware critic targets** (bootstrap on truncation using the pre-reset
  observation, never zero the target at an artificial time limit) and **n-step returns**
  (``--n_step``, default 3; uncorrected, cut at the first episode boundary in the window).

The temperature loss differentiates ``log_alpha`` directly (the SpinningUp /
review-recommended form) rather than ``log_alpha.exp()``.

The actor is saved under the ``"policy"`` key as a plain ``Sequential`` state dict, so
``eval_protocol.py`` / ``play_discrete.py`` load it greedily (argmax over logits) with no
changes. A ``run_meta.json`` sidecar is written next to the checkpoints.

Run (from the repo root):
  isaaclab.bat -p scripts/reinforcement_learning/binary_rl/train_sac_d.py ^
      --num_envs 4096 --timesteps 100000 --seed 42 --experiment_name sacd_s42
"""

import argparse
import math
import os
import sys

# allow running from any CWD (e.g. the repo root, so relative asset paths resolve)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from binary_common import add_common_cli  # noqa: E402

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser()
add_common_cli(parser)
parser.add_argument("--replay_size", type=int, default=1_000_000, help="total replay transitions across all envs")
parser.add_argument("--batch_size", type=int, default=4096)
parser.add_argument("--n_step", type=int, default=3, help="n-step return horizon (1 = one-step)")
parser.add_argument("--q_lr", type=float, default=3e-4)
parser.add_argument("--policy_lr", type=float, default=3e-4)
parser.add_argument("--alpha_lr", type=float, default=3e-4)
parser.add_argument("--discount", type=float, default=0.99)
parser.add_argument("--polyak", type=float, default=0.005)
parser.add_argument("--target_entropy_scale", type=float, default=0.89, help="target entropy = scale * ln(n_actions)")
parser.add_argument("--random_timesteps", type=int, default=20)
parser.add_argument("--learning_starts", type=int, default=50)
parser.add_argument("--log_interval", type=int, default=50)
parser.add_argument("--out_dir", default="", help="alias for --directory + --experiment_name (full run path)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import csv  # noqa: E402
import time  # noqa: E402
from collections import deque  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from binary_common import build_env, mlp, nstep_return, write_run_meta  # noqa: E402

torch.manual_seed(args.seed)

if args.out_dir:
    args.directory, args.experiment_name = os.path.split(args.out_dir.rstrip("/\\"))
    args.directory = args.directory or "."
experiment_name = args.experiment_name or f"sacd_{args.task}"
run_dir = os.path.join(args.directory, experiment_name)
ckpt_dir = os.path.join(run_dir, "checkpoints")
os.makedirs(ckpt_dir, exist_ok=True)

env = build_env(args.task, args.num_envs, args.seed, args.device, compute_final_obs=True)
device = env.device
N = env.num_envs
N_ACT = env.n_actions
obs_dim = int(env.single_observation_space["policy"].shape[-1])
print(f"[sac_d] task={args.task} num_envs={N} obs_dim={obs_dim} actions={N_ACT} n_step={args.n_step}", flush=True)

# --- networks: categorical actor + twin Q (project convention [128,128,128] + ELU) ---
actor = mlp(obs_dim, N_ACT).to(device)
qf1, qf2 = mlp(obs_dim, N_ACT).to(device), mlp(obs_dim, N_ACT).to(device)
qf1_t, qf2_t = mlp(obs_dim, N_ACT).to(device), mlp(obs_dim, N_ACT).to(device)
qf1_t.load_state_dict(qf1.state_dict())
qf2_t.load_state_dict(qf2.state_dict())
q_opt = torch.optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
a_opt = torch.optim.Adam(actor.parameters(), lr=args.policy_lr)
target_entropy = args.target_entropy_scale * math.log(N_ACT)
log_alpha = torch.zeros(1, requires_grad=True, device=device)
alpha_opt = torch.optim.Adam([log_alpha], lr=args.alpha_lr)

# --- replay buffer (GPU) storing n-step transitions -----------------------------------
CAP = max(args.batch_size, (args.replay_size // N) * N)
buf_obs = torch.zeros(CAP, obs_dim, device=device)
buf_next = torch.zeros(CAP, obs_dim, device=device)  # obs to bootstrap from (pre-reset on a timeout)
buf_act = torch.zeros(CAP, dtype=torch.long, device=device)
buf_ret = torch.zeros(CAP, device=device)  # n-step discounted reward sum
buf_disc = torch.zeros(CAP, device=device)  # gamma ** (steps summed)
buf_boot = torch.zeros(CAP, device=device)  # 1.0 unless a real termination cut the window
buf_ptr = 0
buf_full = False


def buf_store(o, a, ret, no, disc, boot):
    global buf_ptr, buf_full
    m = o.shape[0]
    idx = (torch.arange(m, device=device) + buf_ptr) % CAP
    buf_obs[idx], buf_act[idx], buf_ret[idx] = o, a, ret
    buf_next[idx], buf_disc[idx], buf_boot[idx] = no, disc, boot
    buf_ptr = int((buf_ptr + m) % CAP)
    buf_full = buf_full or buf_ptr < m


# --- n-step staging: a rolling window of the last n steps ----------------------------
n = args.n_step
st_obs, st_act, st_rew = deque(maxlen=n), deque(maxlen=n), deque(maxlen=n)
st_term, st_trunc = deque(maxlen=n), deque(maxlen=n)
st_next, st_final = deque(maxlen=n), deque(maxlen=n)  # post-step obs; pre-reset obs this step


def flush_front():
    """Emit the n-step transition for the oldest step in a full staging window."""
    rew_win = torch.stack(list(st_rew))  # [n, N]
    term_win = torch.stack(list(st_term))
    trunc_win = torch.stack(list(st_trunc))
    ret, steps, boot = nstep_return(rew_win, term_win, trunc_win, args.discount)
    disc = args.discount ** steps.float()
    last = (steps - 1).clamp(min=0)  # index of the last summed step
    env_ix = torch.arange(N, device=device)
    next_stack = torch.stack(list(st_next))  # [n, N, obs]
    final_stack = torch.stack(list(st_final))
    took_timeout = trunc_win[last, env_ix].bool() & ~term_win[last, env_ix].bool()
    boot_obs = torch.where(took_timeout.unsqueeze(-1), final_stack[last, env_ix], next_stack[last, env_ix])
    buf_store(st_obs[0], st_act[0], ret, boot_obs, disc, boot)


csv_f = open(os.path.join(run_dir, "train_log.csv"), "w", newline="")  # noqa: SIM115
csv_w = csv.writer(csv_f)
csv_w.writerow(["step", "ep_return_mean", "ep_len_mean", "qf_loss", "actor_loss", "alpha", "entropy", "sps"])

write_run_meta(
    run_dir,
    algo="sac_d",
    obs_dim=obs_dim,
    n_actions=N_ACT,
    seed=args.seed,
    num_envs=args.num_envs,
    timesteps=args.timesteps,
    n_step=args.n_step,
)

obs, _ = env.reset()
obs_t = obs["policy"]
cur_ret = torch.zeros(N, device=device)
cur_len = torch.zeros(N, device=device)
ep_returns, ep_lens = deque(maxlen=200), deque(maxlen=200)
qf_loss_v = actor_loss_v = entropy_v = 0.0
t0, steps_t0 = time.time(), 0
next_ckpt = args.checkpoint_interval

for step in range(args.timesteps):
    with torch.no_grad():
        if step < args.random_timesteps:
            act = torch.randint(0, N_ACT, (N,), device=device)
        else:
            act = torch.distributions.Categorical(logits=actor(obs_t)).sample()
    nobs, rew, terminated, truncated, infos = env.step(act)
    nobs_t = nobs["policy"]
    done = (terminated | truncated).bool()
    # env.step auto-resets done envs, so nobs is post-reset for them. extras["final_obs"]
    # (present because build_env set compute_final_obs=True) holds the pre-reset obs, but
    # only for the envs that reset THIS step -- and the key persists stale on no-reset
    # steps, so only trust it where an env is actually done now.
    _fo = infos.get("final_obs")
    final_t = torch.where(done.unsqueeze(-1), _fo["policy"], nobs_t) if (_fo is not None and done.any()) else nobs_t

    st_obs.append(obs_t)
    st_act.append(act)
    st_rew.append(rew)
    st_term.append(terminated.float())
    st_trunc.append(truncated.float())
    st_next.append(nobs_t)
    st_final.append(final_t)
    if len(st_rew) == n:
        flush_front()

    cur_ret += rew
    cur_len += 1
    if done.any():
        ep_returns.extend(cur_ret[done].tolist())
        ep_lens.extend(cur_len[done].tolist())
        cur_ret[done] = 0.0
        cur_len[done] = 0.0
    obs_t = nobs_t

    if step >= args.learning_starts:
        hi = CAP if buf_full else buf_ptr
        if hi >= args.batch_size:
            idx = torch.randint(0, hi, (args.batch_size,), device=device)
            b_o, b_a, b_ret = buf_obs[idx], buf_act[idx], buf_ret[idx]
            b_no, b_disc, b_boot = buf_next[idx], buf_disc[idx], buf_boot[idx]
            alpha = log_alpha.exp().detach()

            with torch.no_grad():
                n_logp = F.log_softmax(actor(b_no), dim=1)
                n_probs = n_logp.exp()
                min_qt = torch.min(qf1_t(b_no), qf2_t(b_no))
                v_next = (n_probs * (min_qt - alpha * n_logp)).sum(1)
                target = b_ret + b_disc * b_boot * v_next
            q1 = qf1(b_o).gather(1, b_a.unsqueeze(1)).squeeze(1)
            q2 = qf2(b_o).gather(1, b_a.unsqueeze(1)).squeeze(1)
            qf_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
            q_opt.zero_grad()
            qf_loss.backward()
            q_opt.step()

            logits = actor(b_o)
            logp = F.log_softmax(logits, dim=1)
            probs = logp.exp()
            with torch.no_grad():
                min_q = torch.min(qf1(b_o), qf2(b_o))
            actor_loss = (probs * (alpha * logp - min_q)).sum(1).mean()
            a_opt.zero_grad()
            actor_loss.backward()
            a_opt.step()

            with torch.no_grad():
                entropy = -(probs * logp).sum(1)
            # differentiate log_alpha directly: pushes alpha up while entropy < target.
            alpha_loss = (log_alpha * (entropy.detach() - target_entropy)).mean()
            alpha_opt.zero_grad()
            alpha_loss.backward()
            alpha_opt.step()

            with torch.no_grad():
                for pt, p in zip(qf1_t.parameters(), qf1.parameters()):
                    pt.mul_(1 - args.polyak).add_(args.polyak * p)
                for pt, p in zip(qf2_t.parameters(), qf2.parameters()):
                    pt.mul_(1 - args.polyak).add_(args.polyak * p)
            qf_loss_v, actor_loss_v, entropy_v = float(qf_loss), float(actor_loss), float(entropy.mean())

    if step % args.log_interval == 0:
        sps = (step - steps_t0) * N / max(1e-9, time.time() - t0)
        t0, steps_t0 = time.time(), step
        rmean = sum(ep_returns) / len(ep_returns) if ep_returns else float("nan")
        lmean = sum(ep_lens) / len(ep_lens) if ep_lens else float("nan")
        csv_w.writerow(
            [
                step,
                f"{rmean:.4f}",
                f"{lmean:.1f}",
                f"{qf_loss_v:.5f}",
                f"{actor_loss_v:.5f}",
                f"{log_alpha.exp().item():.5f}",
                f"{entropy_v:.4f}",
                f"{sps:.0f}",
            ]
        )
        csv_f.flush()
        print(
            f"[sac_d {step}/{args.timesteps}] ep_ret={rmean:+.3f} ep_len={lmean:.0f} "
            f"qf={qf_loss_v:.4f} pi={actor_loss_v:+.4f} alpha={log_alpha.exp().item():.4f} "
            f"H={entropy_v:.3f}/{target_entropy:.3f} sps={sps:.0f}",
            flush=True,
        )

    if step + 1 >= next_ckpt:
        next_ckpt += args.checkpoint_interval
        torch.save(
            {
                "policy": actor.state_dict(),
                "qf1": qf1.state_dict(),
                "qf2": qf2.state_dict(),
                "log_alpha": log_alpha.detach().cpu(),
                "step": step + 1,
            },
            os.path.join(ckpt_dir, f"agent_{step + 1}.pt"),
        )

torch.save(
    {
        "policy": actor.state_dict(),
        "qf1": qf1.state_dict(),
        "qf2": qf2.state_dict(),
        "log_alpha": log_alpha.detach().cpu(),
        "step": args.timesteps,
    },
    os.path.join(ckpt_dir, "agent_final.pt"),
)
csv_f.close()
print(f"[sac_d] done. outputs in {run_dir}; checkpoints: {sorted(os.listdir(ckpt_dir))}", flush=True)
env.close()
simulation_app.close()
