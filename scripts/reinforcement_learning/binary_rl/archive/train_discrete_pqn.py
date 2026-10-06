# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train a PQN (Parallelised Q-Network) agent on the hexapod binary contact-bit env.

Port of CleanRL's PyTorch ``pqn.py`` (github.com/vwxyzjn/cleanrl) to the Isaac Lab
GPU-vectorized env, reusing the project's ``DiscreteBitsActionWrapper`` (Discrete(64),
LSB = hardware bit 1 = FrontRight).  PQN (arXiv 2407.04811) is Q-learning designed for
massively parallel envs: **no replay buffer, no target network**; stability comes from
LayerNorm in the Q-network and on-policy Q(lambda) returns over short rollouts.

Kept from CleanRL defaults: lr 2.5e-4 (annealed), gamma 0.99, q_lambda 0.65,
4 epochs x 4 minibatches, eps 1.0 -> 0.05 over 50% of training, grad-norm clip 10,
Linear -> LayerNorm -> ReLU blocks.  Changed (recorded): hidden dims [128,128,128] to
match the other routes (CleanRL toy: 120/84); rollout num_steps 32 (paper-style
vectorized config; CleanRL CartPole: 128).  Truncation counts as done (CleanRL
convention; slightly biased at the 25 s time limit).

TensorBoard tags mirror skrl's so the shared monitoring works unchanged.

Smoke run (wiring check only):
  CUDA_VISIBLE_DEVICES=<idle> ./isaaclab.sh -p scripts/reinforcement_learning/train_discrete_pqn.py \
      --num_envs 16 --timesteps 200 --checkpoint_interval 100

Full run example:
  CUDA_VISIBLE_DEVICES=<idle> ./isaaclab.sh -p scripts/reinforcement_learning/train_discrete_pqn.py \
      --num_envs 4096 --timesteps 100000
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-Goal-Flat-Hexapod-Binary-v0")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--timesteps", type=int, default=200, help="total env steps per env")
parser.add_argument("--seed", type=int, default=42)
# PQN hyperparameters (CleanRL pqn.py defaults unless noted)
parser.add_argument("--num_steps", type=int, default=32, help="rollout length (CleanRL: 128; paper vectorized: 32)")
parser.add_argument("--num_minibatches", type=int, default=4)
parser.add_argument("--update_epochs", type=int, default=4)
parser.add_argument("--learning_rate", type=float, default=2.5e-4)
parser.add_argument("--anneal_lr", action="store_true", default=True)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--q_lambda", type=float, default=0.65)
parser.add_argument("--start_e", type=float, default=1.0)
parser.add_argument("--end_e", type=float, default=0.05)
parser.add_argument("--exploration_fraction", type=float, default=0.5)
parser.add_argument("--max_grad_norm", type=float, default=10.0)
# experiment bookkeeping
parser.add_argument("--experiment_name", default="", help="default: pqn_<task>")
parser.add_argument("--directory", default="runs_discrete")
parser.add_argument("--checkpoint_interval", type=int, default=1000)
parser.add_argument("--write_interval", type=int, default=50)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import os
import time

import gymnasium as gym
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

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

# --- env ---
env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=args.num_envs)
env_cfg.seed = args.seed
env = gym.make(args.task, cfg=env_cfg)
env = DiscreteBitsActionWrapper(env, n_bits=6)
device = env.device
N = env.num_envs
N_ACTIONS = env.n_actions
print(f"[train_discrete_pqn] task={args.task} num_envs={N} rollout={args.num_steps} q_lambda={args.q_lambda}")


class QNetwork(torch.nn.Module):
    """Linear -> LayerNorm -> ReLU blocks (PQN's stabilizer), final plain Linear."""

    def __init__(self, obs_dim: int, n_actions: int, hidden=(128, 128, 128)):
        super().__init__()
        layers, in_dim = [], obs_dim
        for h in hidden:
            layers += [torch.nn.Linear(in_dim, h), torch.nn.LayerNorm(h), torch.nn.ReLU()]
            in_dim = h
        layers += [torch.nn.Linear(in_dim, n_actions)]
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


obs_dict, _ = env.reset()
obs = obs_dict["policy"]
obs_dim = obs.shape[-1]
qnet = QNetwork(obs_dim, N_ACTIONS).to(device)
optimizer = torch.optim.Adam(qnet.parameters(), lr=args.learning_rate)

exp_name = args.experiment_name or f"pqn_{args.task}"
exp_dir = os.path.join(args.directory, exp_name)
ckpt_dir = os.path.join(exp_dir, "checkpoints")
os.makedirs(ckpt_dir, exist_ok=True)
writer = SummaryWriter(exp_dir)

# rollout storage
T = args.num_steps
obs_buf = torch.zeros(T, N, obs_dim, device=device)
act_buf = torch.zeros(T, N, dtype=torch.long, device=device)
rew_buf = torch.zeros(T, N, device=device)
done_buf = torch.zeros(T, N, device=device)  # done AFTER action t (terminated or truncated)
val_buf = torch.zeros(T, N, device=device)  # max_a Q(obs[t]) at collection time

# episode bookkeeping
ep_ret = torch.zeros(N, device=device)
ep_len = torch.zeros(N, device=device)
finished_returns: list[float] = []
finished_lengths: list[float] = []

params_before = torch.cat([p.detach().flatten().clone() for p in qnet.parameters()])
global_step = 0
best_metric = None
last_ckpt = 0
n_iterations = max(1, args.timesteps // T)
eps_horizon = max(1, int(args.exploration_fraction * args.timesteps))
t_start = time.time()

for iteration in range(n_iterations):
    if args.anneal_lr:
        frac = 1.0 - global_step / max(1, args.timesteps)
        optimizer.param_groups[0]["lr"] = args.learning_rate * frac

    # --- collect rollout ---
    for t in range(T):
        epsilon = args.start_e + (args.end_e - args.start_e) * min(1.0, global_step / eps_horizon)
        with torch.no_grad():
            q = qnet(obs)
            max_q, greedy_a = q.max(dim=1)
        explore = torch.rand(N, device=device) < epsilon
        rand_a = torch.randint(0, N_ACTIONS, (N,), device=device)
        action = torch.where(explore, rand_a, greedy_a)

        obs_buf[t] = obs
        act_buf[t] = action
        val_buf[t] = max_q

        obs_dict, rew, terminated, truncated, _ = env.step(action)
        obs = obs_dict["policy"]
        done = (terminated.view(-1) | truncated.view(-1)).float()
        rew = rew.view(-1)
        rew_buf[t] = rew
        done_buf[t] = done
        global_step += 1

        ep_ret += rew
        ep_len += 1.0
        if done.any():
            idx = done.bool()
            finished_returns += ep_ret[idx].tolist()
            finished_lengths += ep_len[idx].tolist()
            ep_ret[idx] = 0.0
            ep_len[idx] = 0.0

    # --- Q(lambda) returns (backward, no target network) ---
    with torch.no_grad():
        final_next_value = qnet(obs).max(dim=1).values
        returns = torch.zeros_like(rew_buf)
        for t in reversed(range(T)):
            nextnonterminal = 1.0 - done_buf[t]
            if t == T - 1:
                returns[t] = rew_buf[t] + args.gamma * final_next_value * nextnonterminal
            else:
                returns[t] = rew_buf[t] + args.gamma * (
                    args.q_lambda * returns[t + 1] + (1 - args.q_lambda) * val_buf[t + 1]
                ) * nextnonterminal

    # --- update ---
    b_obs = obs_buf.reshape(T * N, obs_dim)
    b_act = act_buf.reshape(T * N)
    b_ret = returns.reshape(T * N)
    batch = T * N
    mb_size = batch // args.num_minibatches
    last_loss = 0.0
    for _epoch in range(args.update_epochs):
        perm = torch.randperm(batch, device=device)
        for s in range(0, batch, mb_size):
            idx = perm[s : s + mb_size]
            val = qnet(b_obs[idx]).gather(1, b_act[idx].unsqueeze(1)).squeeze(1)
            loss = F.mse_loss(b_ret[idx], val)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(qnet.parameters(), args.max_grad_norm)
            optimizer.step()
            last_loss = loss.item()

    # --- logging (skrl-compatible tags) ---
    if finished_returns and (global_step % args.write_interval < T or iteration == n_iterations - 1):
        mean_ret = sum(finished_returns) / len(finished_returns)
        mean_len = sum(finished_lengths) / len(finished_lengths)
        writer.add_scalar("Reward / Total reward (mean)", mean_ret, global_step)
        writer.add_scalar("Episode / Total timesteps (mean)", mean_len, global_step)
        finished_returns.clear()
        finished_lengths.clear()
        if best_metric is None or mean_ret > best_metric:
            best_metric = mean_ret
            torch.save({"q_network": qnet.state_dict()}, os.path.join(ckpt_dir, "best_agent.pt"))
    writer.add_scalar("Loss / Q-network loss", last_loss, global_step)
    writer.add_scalar("Exploration / Exploration epsilon", epsilon, global_step)
    writer.add_scalar("Learning / Learning rate", optimizer.param_groups[0]["lr"], global_step)
    writer.add_scalar("Stats / SPS (env steps per s, per env)", global_step / (time.time() - t_start), global_step)

    # --- checkpoints ---
    if global_step - last_ckpt >= args.checkpoint_interval or iteration == n_iterations - 1:
        torch.save({"q_network": qnet.state_dict()}, os.path.join(ckpt_dir, f"agent_{global_step}.pt"))
        last_ckpt = global_step

    if iteration % 10 == 0 or iteration == n_iterations - 1:
        sps = global_step / (time.time() - t_start)
        print(
            f"[pqn] iter {iteration + 1}/{n_iterations} gstep {global_step}/{args.timesteps} "
            f"eps {epsilon:.3f} loss {last_loss:.5f} sps {sps:.1f} "
            f"({sps * N:.0f} total env steps/s)",
            flush=True,
        )

params_after = torch.cat([p.detach().flatten().clone() for p in qnet.parameters()])
delta = (params_after - params_before).norm().item()
print(f"[train_discrete_pqn] q_network parameter L2 change over run: {delta:.6f} (must be > 0 if updates ran)")
print(f"[train_discrete_pqn] experiment dir: {exp_dir}")
print(f"[train_discrete_pqn] checkpoints: {sorted(os.listdir(ckpt_dir))}")

writer.close()
env.close()
simulation_app.close()
