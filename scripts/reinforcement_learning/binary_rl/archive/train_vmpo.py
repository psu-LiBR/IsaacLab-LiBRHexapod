# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Discrete V-MPO on the hexapod binary contact-bit env (single file).

Hand-written after "V-MPO: On-Policy Maximum a Posteriori Policy Optimization for
Discrete and Continuous Control" (arXiv:1909.12238, 2019-09, ICLR 2020).

POSITIONING (per task spec): policy-family CONTROL run. Jackson's exclusion was PPO;
V-MPO is a close on-policy relative, run only as a reference point under the same
env / data conventions as the first-wave discrete-PPO trial.

Implementation choices (AI simplifications, recorded per project rules):
- On-policy rollouts of T=16 steps x num_envs; each batch consumed ONCE with a single
  full-batch gradient step (paper: large-batch single-use updates; no reuse, no IS).
- Advantages: GAE(lambda=0.95) with bootstrap masked on done=terminated|truncated
  (paper uses n-step returns; GAE is the local convention from the PPO trial and only
  changes the advantage estimator). Auto-reset teleport handled by the done mask.
- E-step: top-half advantages kept (paper's psi); temperature eta learned via dual
  L_eta = eta*eps_eta + eta*(logsumexp(A_top/eta) - log K), eps_eta = 0.1 (paper value).
- M-step: weighted max-likelihood -sum(psi * log pi) + KL trust region with learned
  Lagrange alpha: L_alpha = alpha*(eps_alpha - sg(KL)) + sg(alpha)*KL, where
  KL = KL(pi_old || pi_new) averaged over the FULL batch. eps_alpha = 0.01
  (paper's discrete range; upper end chosen for movement within a 2h budget).
- pi_old = behavior logits stored at collection time (equivalent to the paper's
  target-network variant because each batch is consumed immediately).
- eta/alpha parametrized via softplus(raw) to stay positive; init eta=1.0, alpha=1.0.
- Networks: separate policy/value MLP [128,128,128]+ELU (project convention);
  one Adam over everything, lr 5e-4 (AI compromise between paper 1e-4 and 2h budget).
- Checkpoints store the policy under "policy" with net.* names -> play_discrete.py
  replays greedily (argmax over logits) unchanged.

Smoke:  CUDA_VISIBLE_DEVICES=<idle> python scripts/reinforcement_learning/train_vmpo.py \
            --num_envs 16 --timesteps 200 --rollout_len 8 --checkpoint_interval 100
Trial:  CUDA_VISIBLE_DEVICES=<idle> python scripts/reinforcement_learning/train_vmpo.py \
            --num_envs 4096 --timesteps 100000
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-Goal-Flat-Hexapod-Binary-v0")
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--timesteps", type=int, default=100000, help="total env steps (per env)")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--rollout_len", type=int, default=16)
parser.add_argument("--lr", type=float, default=5e-4)
parser.add_argument("--discount", type=float, default=0.99)
parser.add_argument("--gae_lambda", type=float, default=0.95)
parser.add_argument("--eps_eta", type=float, default=0.1)
parser.add_argument("--eps_alpha", type=float, default=0.01)
parser.add_argument("--log_interval", type=int, default=50, help="in env steps")
parser.add_argument("--checkpoint_interval", type=int, default=2000, help="in env steps")
parser.add_argument("--out_dir", default="runs_discrete/vmpo_trial_20260830")
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
T = args.rollout_len
obs_dim = int(env.single_observation_space["policy"].shape[-1])
print(f"[train_vmpo] task={args.task} num_envs={N} obs_dim={obs_dim} actions={N_ACT} T={T}", flush=True)


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


policy = mlp(N_ACT)
value = mlp(1)
# softplus(raw) > 0 ; softplus(0.5413) ~= 1.0
raw_eta = torch.tensor([0.5413], requires_grad=True, device=device)
raw_alpha = torch.tensor([0.5413], requires_grad=True, device=device)
opt = torch.optim.Adam(
    list(policy.parameters()) + list(value.parameters()) + [raw_eta, raw_alpha], lr=args.lr
)

# rollout storage
ro_obs = torch.zeros(T + 1, N, obs_dim, device=device)
ro_act = torch.zeros(T, N, dtype=torch.long, device=device)
ro_logits = torch.zeros(T, N, N_ACT, device=device)
ro_rew = torch.zeros(T, N, device=device)
ro_done = torch.zeros(T, N, device=device)

os.makedirs(os.path.join(args.out_dir, "checkpoints"), exist_ok=True)
csv_path = os.path.join(args.out_dir, "train_log.csv")
csv_f = open(csv_path, "w", newline="")
csv_w = csv.writer(csv_f)
csv_w.writerow(["step", "ep_return_mean", "ep_len_mean", "policy_loss", "value_loss",
                "kl", "eta", "alpha", "entropy", "sps"])

obs, _ = env.reset()
obs_t = obs["policy"]
cur_ret = torch.zeros(N, device=device)
cur_len = torch.zeros(N, device=device)
ep_returns, ep_lens = deque(maxlen=200), deque(maxlen=200)
pol_loss_v = val_loss_v = kl_v = ent_v = 0.0
global_step = 0
last_log = 0
last_ckpt = 0
t0, steps_t0 = time.time(), 0

n_iters = args.timesteps // T
for it in range(n_iters):
    # ---- collect T on-policy steps ----
    with torch.no_grad():
        for t in range(T):
            ro_obs[t] = obs_t
            logits = policy(obs_t)
            act = torch.distributions.Categorical(logits=logits).sample()
            nobs, rew, terminated, truncated, _ = env.step(act)
            nobs_t = nobs["policy"]
            done = (terminated | truncated).float()
            ro_act[t] = act
            ro_logits[t] = logits
            ro_rew[t] = rew
            ro_done[t] = done
            cur_ret += rew
            cur_len += 1
            dmask = done.bool()
            if dmask.any():
                ep_returns.extend(cur_ret[dmask].tolist())
                ep_lens.extend(cur_len[dmask].tolist())
                cur_ret[dmask] = 0.0
                cur_len[dmask] = 0.0
            obs_t = nobs_t
            global_step += 1
        ro_obs[T] = obs_t

        # ---- GAE advantages / value targets ----
        vals = value(ro_obs.reshape(-1, obs_dim)).reshape(T + 1, N)
        adv = torch.zeros(T, N, device=device)
        last_gae = torch.zeros(N, device=device)
        for t in reversed(range(T)):
            notdone = 1.0 - ro_done[t]
            delta = ro_rew[t] + args.discount * notdone * vals[t + 1] - vals[t]
            last_gae = delta + args.discount * args.gae_lambda * notdone * last_gae
            adv[t] = last_gae
        vtarget = adv + vals[:T]

    # ---- single full-batch V-MPO update ----
    b_obs = ro_obs[:T].reshape(-1, obs_dim)
    b_act = ro_act.reshape(-1)
    b_adv = adv.reshape(-1)
    b_vt = vtarget.reshape(-1)
    b_old_logits = ro_logits.reshape(-1, N_ACT)
    B = b_obs.shape[0]
    K = B // 2

    eta = F.softplus(raw_eta) + 1e-8
    alpha = F.softplus(raw_alpha) + 1e-8

    # E-step: top-half advantages
    top_adv, top_idx = torch.topk(b_adv, K)
    # temperature dual (logsumexp for stability)
    l_eta = eta * args.eps_eta + eta * (torch.logsumexp(top_adv / eta, dim=0) - math.log(K))
    # nonparametric weights (detached from eta for the policy term)
    psi = F.softmax(top_adv / eta.detach(), dim=0)

    logits_new = policy(b_obs)
    logp_all = F.log_softmax(logits_new, dim=1)
    logp_taken = logp_all.gather(1, b_act.unsqueeze(1)).squeeze(1)
    l_pi = -(psi.detach() * logp_taken[top_idx]).sum()

    # M-step KL constraint over the full batch: KL(pi_old || pi_new)
    old_logp = F.log_softmax(b_old_logits, dim=1)
    kl = (old_logp.exp() * (old_logp - logp_all)).sum(1).mean()
    l_alpha = alpha * (args.eps_alpha - kl.detach()) + alpha.detach() * kl

    v_pred = value(b_obs).squeeze(1)
    l_v = 0.5 * F.mse_loss(v_pred, b_vt)

    loss = l_pi + l_v + l_eta + l_alpha
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
    torch.nn.utils.clip_grad_norm_(value.parameters(), 10.0)
    opt.step()

    with torch.no_grad():
        entropy = -(logp_all.exp() * logp_all).sum(1).mean()
    pol_loss_v, val_loss_v = float(l_pi), float(l_v)
    kl_v, ent_v = float(kl), float(entropy)

    if global_step - last_log >= args.log_interval:
        last_log = global_step
        sps = (global_step - steps_t0) * N / max(1e-9, time.time() - t0)
        t0, steps_t0 = time.time(), global_step
        rmean = sum(ep_returns) / len(ep_returns) if ep_returns else float("nan")
        lmean = sum(ep_lens) / len(ep_lens) if ep_lens else float("nan")
        csv_w.writerow([global_step, f"{rmean:.4f}", f"{lmean:.1f}", f"{pol_loss_v:.5f}",
                        f"{val_loss_v:.5f}", f"{kl_v:.6f}", f"{float(eta):.4f}",
                        f"{float(alpha):.4f}", f"{ent_v:.4f}", f"{sps:.0f}"])
        csv_f.flush()
        print(f"[vmpo {global_step}/{args.timesteps}] ep_ret={rmean:+.3f} ep_len={lmean:.0f} "
              f"pi={pol_loss_v:+.4f} v={val_loss_v:.4f} kl={kl_v:.5f} eta={float(eta):.3f} "
              f"alpha={float(alpha):.3f} H={ent_v:.3f} sps={sps:.0f}", flush=True)

    if global_step - last_ckpt >= args.checkpoint_interval:
        last_ckpt = global_step
        torch.save({"policy": policy.state_dict(), "value": value.state_dict(),
                    "raw_eta": raw_eta.detach().cpu(), "raw_alpha": raw_alpha.detach().cpu(),
                    "step": global_step},
                   os.path.join(args.out_dir, "checkpoints", f"agent_{global_step}.pt"))

torch.save({"policy": policy.state_dict(), "value": value.state_dict(),
            "raw_eta": raw_eta.detach().cpu(), "raw_alpha": raw_alpha.detach().cpu(),
            "step": global_step},
           os.path.join(args.out_dir, "checkpoints", "agent_final.pt"))
csv_f.close()
print(f"[train_vmpo] done at step {global_step}. outputs in {args.out_dir}", flush=True)
env.close()
simulation_app.close()
