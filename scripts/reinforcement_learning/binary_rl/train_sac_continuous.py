# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Continuous Soft Actor-Critic baseline for the hexapod (8-DOF joint-position control).

A comparison point *outside* the binary contact-bit action space: SAC on the standard
continuous goal-reaching task (``Isaac-Goal-Flat-Hexapod-v0``), so the discrete
contact-bit RL can be measured against a strong continuous learner on the same reward.

Implements the three contributions of Sabatini, Li & Hutter 2026, "Bridging the Gap:
Enabling Soft Actor Critic for High Performance Legged Locomotion" (arXiv:2605.24975;
reference code: github.com/leggedrobotics/rsl_rl_sac):

1. **Informed initialization** -- the mean head starts near zero (``N(0, 1e-3)`` weights,
   zero bias) so the initial policy sits at the default joint configuration; the log-std
   head starts state-independent at ``ln(sigma0)`` with ``sigma0 = 0.15`` (zeroed
   weights, ``bias = ln(sigma0)``); and the tanh-squashed action is rescaled per joint by
   the distance from the default pose to the soft joint limits divided by the action
   scale, so early exploration does not sit in tanh saturation.
2. **Timeout-aware critic targets** -- ``bootstrap = truncated``; the target is only
   zeroed on a real termination, and a truncation bootstraps off the *pre-reset*
   observation (``extras["final_obs"]``).
3. **n-step returns** -- ``--n_step`` (default 5), uncorrected, cut at the first episode
   boundary inside the window.

Twin Q-critics are asymmetric (they read the ``critic`` observation group, which adds
ground-truth ``base_lin_vel``), matching the repo's other hexapod tasks.

Run (from the repo root):
  isaaclab.bat -p scripts/reinforcement_learning/binary_rl/train_sac_continuous.py ^
      --task Isaac-Goal-Flat-Hexapod-v0 --num_envs 4096 --timesteps 60000 --experiment_name sac_cont_s42
"""

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from binary_common import add_common_cli  # noqa: E402

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser()
add_common_cli(parser)
parser.set_defaults(task="Isaac-Goal-Flat-Hexapod-v0", timesteps=60_000)
parser.add_argument("--replay_size", type=int, default=1_000_000, help="total replay transitions across all envs")
parser.add_argument("--batch_size", type=int, default=4096)
parser.add_argument("--n_step", type=int, default=5, help="n-step return horizon (paper: 5)")
parser.add_argument("--hidden", type=int, nargs="+", default=[256, 256, 256])
parser.add_argument("--q_lr", type=float, default=2e-4)
parser.add_argument("--policy_lr", type=float, default=2e-4)
parser.add_argument("--alpha_lr", type=float, default=2e-5)
parser.add_argument("--discount", type=float, default=0.99)
parser.add_argument("--polyak", type=float, default=0.003)
parser.add_argument("--init_noise_std", type=float, default=0.15, help="sigma0 (paper)")
parser.add_argument("--target_entropy_scale", type=float, default=0.167, help="target entropy = -scale * dim(A)")
parser.add_argument("--policy_frequency", type=int, default=2, help="critic updates per actor update")
parser.add_argument("--max_grad_norm", type=float, default=1.0)
parser.add_argument("--random_timesteps", type=int, default=20)
parser.add_argument("--learning_starts", type=int, default=50)
parser.add_argument("--log_interval", type=int, default=50)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import csv  # noqa: E402
import time  # noqa: E402
from collections import deque  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402

try:
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402
except ImportError:
    from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

from binary_common import mlp, nstep_return, write_run_meta  # noqa: E402

torch.manual_seed(args.seed)
LOG_STD_MIN, LOG_STD_MAX = -20.0, 2.0

experiment_name = args.experiment_name or f"sac_cont_{args.task}"
run_dir = os.path.join(args.directory, experiment_name)
ckpt_dir = os.path.join(run_dir, "checkpoints")
os.makedirs(ckpt_dir, exist_ok=True)

# --- env (continuous; asymmetric obs groups; expose the pre-reset observation) --------
env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
env_cfg.seed = args.seed
env_cfg.compute_final_obs = True
action_scale = float(getattr(env_cfg.actions.joint_pos, "scale", 0.5))
env = gym.make(args.task, cfg=env_cfg)
base = env.unwrapped
device = base.device
N = base.num_envs
obs_dim = int(base.single_observation_space["policy"].shape[-1])
has_critic = "critic" in base.single_observation_space.spaces
state_dim = int(base.single_observation_space["critic"].shape[-1]) if has_critic else obs_dim
act_dim = int(base.single_action_space.shape[-1])
print(
    f"[sac_cont] task={args.task} num_envs={N} obs_dim={obs_dim} state_dim={state_dim} act_dim={act_dim} "
    f"n_step={args.n_step} asymmetric_critic={has_critic}",
    flush=True,
)

# --- action-space calibration (paper): tanh(-1..1) -> [default - r-, default + r+] / scale
robot = base.scene["robot"]
q0 = robot.data.default_joint_pos[0].detach()
lim = robot.data.soft_joint_pos_limits[0].detach()  # [act_dim, 2]
a_lo = -(q0 - lim[:, 0]) / action_scale
a_hi = (lim[:, 1] - q0) / action_scale
act_bias = ((a_hi + a_lo) / 2.0).to(device)
act_range = ((a_hi - a_lo) / 2.0).clamp_min(1e-3).to(device)
print(f"[sac_cont] action range per joint: {[round(float(x), 3) for x in act_range.tolist()]}", flush=True)


class Actor(torch.nn.Module):
    """Gaussian policy with tanh squashing + per-joint rescale; paper-style informed init."""

    def __init__(self):
        super().__init__()
        self.trunk = mlp(obs_dim, 2 * act_dim, tuple(args.hidden))
        head = self.trunk[-1]
        with torch.no_grad():
            head.weight[:act_dim].normal_(0.0, 1e-3)  # mean head: start near default pose
            head.bias[:act_dim].zero_()
            head.weight[act_dim:].zero_()  # log-std head: start state-independent
            head.bias[act_dim:].fill_(math.log(args.init_noise_std))

    def forward(self, obs):
        mean, log_std = self.trunk(obs).chunk(2, dim=-1)
        return mean, log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)

    def sample(self, obs, deterministic=False):
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        if deterministic:
            x = mean
        else:
            x = mean + std * torch.randn_like(std)
        y = torch.tanh(x)
        action = act_bias + act_range * y
        # tanh + affine change-of-variables log-prob correction
        logp = (-0.5 * (((x - mean) / (std + 1e-8)) ** 2) - log_std - 0.5 * math.log(2 * math.pi)).sum(-1)
        logp = logp - (torch.log(act_range * (1 - y.pow(2)) + 1e-6)).sum(-1)
        return action, logp


def qnet():
    return mlp(state_dim + act_dim, 1, tuple(args.hidden)).to(device)


actor = Actor().to(device)
qf1, qf2, qf1_t, qf2_t = qnet(), qnet(), qnet(), qnet()
qf1_t.load_state_dict(qf1.state_dict())
qf2_t.load_state_dict(qf2.state_dict())
q_opt = torch.optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
a_opt = torch.optim.Adam(actor.parameters(), lr=args.policy_lr)
target_entropy = -args.target_entropy_scale * act_dim
log_alpha = torch.zeros(1, requires_grad=True, device=device)
alpha_opt = torch.optim.Adam([log_alpha], lr=args.alpha_lr)

# --- replay buffer (GPU) storing n-step transitions ----------------------------------
CAP = max(args.batch_size, (args.replay_size // N) * N)
buf_o = torch.zeros(CAP, obs_dim, device=device)
buf_s = torch.zeros(CAP, state_dim, device=device)
buf_a = torch.zeros(CAP, act_dim, device=device)
buf_no = torch.zeros(CAP, obs_dim, device=device)
buf_ns = torch.zeros(CAP, state_dim, device=device)
buf_ret = torch.zeros(CAP, device=device)
buf_disc = torch.zeros(CAP, device=device)
buf_boot = torch.zeros(CAP, device=device)
buf_ptr, buf_full = 0, False


def buf_store(o, s, a, no, ns, ret, disc, boot):
    global buf_ptr, buf_full
    m = o.shape[0]
    idx = (torch.arange(m, device=device) + buf_ptr) % CAP
    buf_o[idx], buf_s[idx], buf_a[idx] = o, s, a
    buf_no[idx], buf_ns[idx] = no, ns
    buf_ret[idx], buf_disc[idx], buf_boot[idx] = ret, disc, boot
    buf_ptr = int((buf_ptr + m) % CAP)
    buf_full = buf_full or buf_ptr < m


n = args.n_step
S = {k: deque(maxlen=n) for k in ("o", "s", "a", "rew", "term", "trunc", "no", "ns", "fo", "fs")}


def _critic_obs(obs_dict):
    return obs_dict["critic"] if has_critic else obs_dict["policy"]


def flush_front():
    rew_win = torch.stack(list(S["rew"]))
    term_win = torch.stack(list(S["term"]))
    trunc_win = torch.stack(list(S["trunc"]))
    ret, steps, boot = nstep_return(rew_win, term_win, trunc_win, args.discount)
    disc = args.discount ** steps.float()
    last = (steps - 1).clamp(min=0)
    ei = torch.arange(N, device=device)
    tt = (trunc_win[last, ei].bool() & ~term_win[last, ei].bool()).unsqueeze(-1)
    no_end = torch.stack(list(S["no"]))[last, ei]
    ns_end = torch.stack(list(S["ns"]))[last, ei]
    no = torch.where(tt, torch.stack(list(S["fo"]))[last, ei], no_end)
    ns = torch.where(tt, torch.stack(list(S["fs"]))[last, ei], ns_end)
    buf_store(S["o"][0], S["s"][0], S["a"][0], no, ns, ret, disc, boot)


csv_f = open(os.path.join(run_dir, "train_log.csv"), "w", newline="")  # noqa: SIM115
csv_w = csv.writer(csv_f)
csv_w.writerow(["step", "ep_return_mean", "ep_len_mean", "qf_loss", "actor_loss", "alpha", "entropy", "sps"])
write_run_meta(
    run_dir,
    algo="sac_continuous",
    obs_dim=obs_dim,
    n_actions=act_dim,
    seed=args.seed,
    num_envs=args.num_envs,
    timesteps=args.timesteps,
    n_step=args.n_step,
    continuous=True,
    act_dim=act_dim,
)

obs, _ = env.reset()
o_t, s_t = obs["policy"], _critic_obs(obs)
cur_ret = torch.zeros(N, device=device)
cur_len = torch.zeros(N, device=device)
ep_returns, ep_lens = deque(maxlen=200), deque(maxlen=200)
qf_loss_v = actor_loss_v = entropy_v = 0.0
t0, steps_t0 = time.time(), 0
next_ckpt = args.checkpoint_interval
upd = 0

for step in range(args.timesteps):
    with torch.no_grad():
        if step < args.random_timesteps:
            act = act_bias + act_range * (2 * torch.rand(N, act_dim, device=device) - 1)
        else:
            act, _ = actor.sample(o_t)
    nobs, rew, terminated, truncated, infos = env.step(act)
    no_t, ns_t = nobs["policy"], _critic_obs(nobs)
    done = (terminated | truncated).bool()
    fo = infos.get("final_obs")
    if fo is not None and done.any():
        fo_t = torch.where(done.unsqueeze(-1), fo["policy"], no_t)
        fs_t = torch.where(done.unsqueeze(-1), _critic_obs(fo), ns_t)
    else:
        fo_t, fs_t = no_t, ns_t

    for k, v in zip(
        ("o", "s", "a", "rew", "term", "trunc", "no", "ns", "fo", "fs"),
        (o_t, s_t, act, rew, terminated.float(), truncated.float(), no_t, ns_t, fo_t, fs_t),
    ):
        S[k].append(v)
    if len(S["rew"]) == n:
        flush_front()

    cur_ret += rew
    cur_len += 1
    if done.any():
        ep_returns.extend(cur_ret[done].tolist())
        ep_lens.extend(cur_len[done].tolist())
        cur_ret[done] = 0.0
        cur_len[done] = 0.0
    o_t, s_t = no_t, ns_t

    if step >= args.learning_starts:
        hi = CAP if buf_full else buf_ptr
        if hi >= args.batch_size:
            idx = torch.randint(0, hi, (args.batch_size,), device=device)
            b_o, b_s, b_a = buf_o[idx], buf_s[idx], buf_a[idx]
            b_no, b_ns = buf_no[idx], buf_ns[idx]
            b_ret, b_disc, b_boot = buf_ret[idx], buf_disc[idx], buf_boot[idx]
            alpha = log_alpha.exp().detach()

            with torch.no_grad():
                na, nlogp = actor.sample(b_no)
                nq = torch.min(qf1_t(torch.cat([b_ns, na], -1)), qf2_t(torch.cat([b_ns, na], -1))).squeeze(-1)
                target = b_ret + b_disc * b_boot * (nq - alpha * nlogp)
            q1 = qf1(torch.cat([b_s, b_a], -1)).squeeze(-1)
            q2 = qf2(torch.cat([b_s, b_a], -1)).squeeze(-1)
            qf_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
            q_opt.zero_grad()
            qf_loss.backward()
            torch.nn.utils.clip_grad_norm_(list(qf1.parameters()) + list(qf2.parameters()), args.max_grad_norm)
            q_opt.step()
            qf_loss_v = float(qf_loss)

            upd += 1
            if upd % args.policy_frequency == 0:
                pa, plogp = actor.sample(b_o)
                min_q = torch.min(qf1(torch.cat([b_s, pa], -1)), qf2(torch.cat([b_s, pa], -1))).squeeze(-1)
                actor_loss = (alpha * plogp - min_q).mean()
                a_opt.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), args.max_grad_norm)
                a_opt.step()
                alpha_loss = (-log_alpha * (plogp.detach() + target_entropy)).mean()
                alpha_opt.zero_grad()
                alpha_loss.backward()
                alpha_opt.step()
                actor_loss_v, entropy_v = float(actor_loss), float(-plogp.mean())
                with torch.no_grad():
                    for pt, p in zip(qf1_t.parameters(), qf1.parameters()):
                        pt.mul_(1 - args.polyak).add_(args.polyak * p)
                    for pt, p in zip(qf2_t.parameters(), qf2.parameters()):
                        pt.mul_(1 - args.polyak).add_(args.polyak * p)

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
            f"[sac_cont {step}/{args.timesteps}] ep_ret={rmean:+.3f} ep_len={lmean:.0f} "
            f"qf={qf_loss_v:.4f} pi={actor_loss_v:+.4f} alpha={log_alpha.exp().item():.4f} "
            f"H={entropy_v:+.3f}/{target_entropy:.3f} sps={sps:.0f}",
            flush=True,
        )

    if step + 1 >= next_ckpt:
        next_ckpt += args.checkpoint_interval
        torch.save(
            {
                "actor": actor.state_dict(),
                "qf1": qf1.state_dict(),
                "qf2": qf2.state_dict(),
                "act_bias": act_bias.cpu(),
                "act_range": act_range.cpu(),
                "log_alpha": log_alpha.detach().cpu(),
                "step": step + 1,
            },
            os.path.join(ckpt_dir, f"agent_{step + 1}.pt"),
        )

torch.save(
    {
        "actor": actor.state_dict(),
        "qf1": qf1.state_dict(),
        "qf2": qf2.state_dict(),
        "act_bias": act_bias.cpu(),
        "act_range": act_range.cpu(),
        "log_alpha": log_alpha.detach().cpu(),
        "step": args.timesteps,
    },
    os.path.join(ckpt_dir, "agent_final.pt"),
)
csv_f.close()
print(f"[sac_cont] done. outputs in {run_dir}; checkpoints: {sorted(os.listdir(ckpt_dir))}", flush=True)
env.close()
simulation_app.close()
