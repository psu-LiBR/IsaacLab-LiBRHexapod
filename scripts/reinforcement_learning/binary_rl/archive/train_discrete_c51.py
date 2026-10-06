# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train a C51 (categorical distributional DQN) agent on the hexapod binary contact-bit env.

Rainbow stand-in route: skrl 2.1.0 ships no Rainbow/C51, and full Rainbow libraries
(e.g. Tianshou) need heavy glue for GPU-vectorized Isaac envs.  This script therefore
implements C51 — the distributional core of Rainbow — as a skrl ``DQN`` subclass,
following the standard algorithm (Bellemare et al. 2017; reference implementation:
CleanRL ``c51.py``), plus Rainbow-style **double** action selection (argmax by the
online net's expected Q).  Everything else (replay, epsilon-greedy, trainer,
checkpointing) is inherited from the DQN wiring in ``train_discrete.py``.

Smoke run (wiring check only):
  CUDA_VISIBLE_DEVICES=<idle> ./isaaclab.sh -p scripts/reinforcement_learning/train_discrete_c51.py \
      --num_envs 16 --timesteps 200 --learning_starts 20 --random_timesteps 10 \
      --batch_size 256 --memory_slots 200 --checkpoint_interval 100

Full run example:
  CUDA_VISIBLE_DEVICES=<idle> ./isaaclab.sh -p scripts/reinforcement_learning/train_discrete_c51.py \
      --num_envs 4096 --timesteps 100000
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-Goal-Flat-Hexapod-Binary-v0")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--timesteps", type=int, default=200)
parser.add_argument("--seed", type=int, default=42)
# C51 distributional head
parser.add_argument("--atoms", type=int, default=51)
parser.add_argument("--v_min", type=float, default=-10.0)
parser.add_argument("--v_max", type=float, default=10.0)
# agent hyperparameters (mirror train_discrete.py defaults)
parser.add_argument("--batch_size", type=int, default=4096)
parser.add_argument("--memory_slots", type=int, default=500)
parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument("--discount", type=float, default=0.99)
parser.add_argument("--polyak", type=float, default=0.005)
parser.add_argument("--target_update_interval", type=int, default=10)
parser.add_argument("--gradient_steps", type=int, default=1)
parser.add_argument("--random_timesteps", type=int, default=20)
parser.add_argument("--learning_starts", type=int, default=50)
parser.add_argument("--eps_initial", type=float, default=1.0)
parser.add_argument("--eps_final", type=float, default=0.05)
parser.add_argument("--eps_fraction", type=float, default=0.2)
# experiment bookkeeping
parser.add_argument("--experiment_name", default="", help="default: c51_<task>")
parser.add_argument("--directory", default="runs_discrete")
parser.add_argument("--checkpoint_interval", type=int, default=1000)
parser.add_argument("--write_interval", type=int, default=50)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import torch.nn.functional as F

import isaaclab_tasks  # noqa: F401

try:
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
except ImportError:
    from isaaclab_tasks.utils import parse_env_cfg

from skrl.agents.torch.dqn import DQN, DQN_CFG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, Model
from skrl.trainers.torch.sequential import SequentialTrainer, SequentialTrainerCfg
from skrl.utils import set_seed

# Archived script: add the parent binary_rl/ dir to sys.path so the shared
# discrete_action_wrapper module still resolves when run from archive/.
import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from discrete_action_wrapper import DiscreteBitsActionWrapper  # noqa: E402

set_seed(args.seed)

# --- env ---
env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=args.num_envs)
env_cfg.seed = args.seed
env = gym.make(args.task, cfg=env_cfg)
env = DiscreteBitsActionWrapper(env, n_bits=6)
env = wrap_env(env, wrapper="isaaclab-single-agent")
device = env.device
print(f"[train_discrete_c51] task={args.task} num_envs={env.num_envs} atoms={args.atoms} "
      f"support=[{args.v_min},{args.v_max}]")
assert isinstance(env.action_space, gym.spaces.Discrete) and env.action_space.n == 64

N_ACTIONS = 64
SUPPORT = torch.linspace(args.v_min, args.v_max, args.atoms, device=device)
DELTA_Z = (args.v_max - args.v_min) / (args.atoms - 1)


class C51Network(DeterministicMixin, Model):
    """MLP [128,128,128] -> 64 x atoms logits.  ``act`` returns expected Q-values, so the
    inherited DQN epsilon-greedy/argmax machinery works unchanged; the raw distribution
    is exposed via :meth:`dist_logits` for the C51 update."""

    def __init__(self, observation_space, action_space, device, atoms, hidden=(128, 128, 128)):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        DeterministicMixin.__init__(self)
        self.atoms = atoms
        layers, in_dim = [], self.num_observations
        for h in hidden:
            layers += [torch.nn.Linear(in_dim, h), torch.nn.ELU()]
            in_dim = h
        layers += [torch.nn.Linear(in_dim, N_ACTIONS * atoms)]
        self.net = torch.nn.Sequential(*layers)

    def dist_logits(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations).view(-1, N_ACTIONS, self.atoms)

    def compute(self, inputs, role):
        probs = F.softmax(self.dist_logits(inputs["observations"]), dim=-1)
        expected_q = (probs * SUPPORT).sum(-1)  # [B, N_ACTIONS]
        return expected_q, {}


class C51(DQN):
    """skrl DQN with the update replaced by the categorical distributional loss.

    Action selection for the target uses the online net's expected Q (double variant,
    as in Rainbow)."""

    def update(self, *, timestep: int, timesteps: int) -> None:
        for _ in range(self.cfg.gradient_steps):
            (
                sampled_obs,
                _sampled_states,
                sampled_actions,
                sampled_rewards,
                sampled_next_obs,
                _sampled_next_states,
                sampled_terminated,
            ) = self.memory.sample(names=self._tensors_names, batch_size=self.cfg.batch_size)[0]

            obs = self._observation_preprocessor(sampled_obs, train=True)
            next_obs = self._observation_preprocessor(sampled_next_obs, train=True)
            B = obs.shape[0]
            arange = torch.arange(B, device=self.device)

            with torch.no_grad():
                # double action selection: argmax of ONLINE net expected Q on next obs
                online_next_q, _ = self.q_network.act({"observations": next_obs, "states": None}, role="q_network")
                a_star = torch.argmax(online_next_q, dim=1)  # [B]
                # target distribution for those actions
                next_logits = self.target_q_network.dist_logits(next_obs)[arange, a_star]  # [B, atoms]
                p_next = F.softmax(next_logits, dim=-1)
                # project Bellman-updated support onto the fixed support
                not_done = sampled_terminated.logical_not().float().view(-1, 1)
                Tz = (sampled_rewards.view(-1, 1) + self.cfg.discount_factor * not_done * SUPPORT.view(1, -1)).clamp(
                    args.v_min, args.v_max
                )  # [B, atoms]
                b = (Tz - args.v_min) / DELTA_Z
                lo = b.floor().long().clamp(0, args.atoms - 1)
                hi = b.ceil().long().clamp(0, args.atoms - 1)
                # when lo == hi (b integral), give full mass to lo
                w_hi = b - lo.float()
                w_lo = 1.0 - w_hi
                m = torch.zeros(B, args.atoms, device=self.device)
                m.scatter_add_(1, lo, p_next * w_lo)
                m.scatter_add_(1, hi, p_next * w_hi)

            logits = self.q_network.dist_logits(obs)[arange, sampled_actions.view(-1).long()]  # [B, atoms]
            loss = -(m * F.log_softmax(logits, dim=-1)).sum(-1).mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self._update_counter += 1
            if not self._update_counter % self.cfg.target_update_interval:
                self.target_q_network.update_parameters(self.q_network, polyak=self.cfg.polyak)

            self.track_data("Loss / Q-network loss", loss.item())
            self.track_data("Target / Target (mean)", (m * SUPPORT).sum(-1).mean().item())


models = {
    "q_network": C51Network(env.observation_space, env.action_space, device, args.atoms),
    "target_q_network": C51Network(env.observation_space, env.action_space, device, args.atoms),
}

memory = RandomMemory(memory_size=args.memory_slots, num_envs=env.num_envs, device=device)


def epsilon_schedule(timestep: int, timesteps: int) -> float:
    horizon = max(1, int(args.eps_fraction * timesteps))
    frac = min(1.0, timestep / horizon)
    return args.eps_initial + (args.eps_final - args.eps_initial) * frac


agent_cfg = DQN_CFG(
    gradient_steps=args.gradient_steps,
    batch_size=args.batch_size,
    discount_factor=args.discount,
    polyak=args.polyak,
    learning_rate=args.learning_rate,
    random_timesteps=args.random_timesteps,
    learning_starts=args.learning_starts,
    target_update_interval=args.target_update_interval,
    exploration_scheduler=epsilon_schedule,
)
agent_cfg.experiment.directory = args.directory
agent_cfg.experiment.experiment_name = args.experiment_name or f"c51_{args.task}"
agent_cfg.experiment.write_interval = args.write_interval
agent_cfg.experiment.checkpoint_interval = args.checkpoint_interval

agent = C51(
    models=models,
    memory=memory,
    observation_space=env.observation_space,
    state_space=None,
    action_space=env.action_space,
    device=device,
    cfg=agent_cfg,
)

trainer_cfg = SequentialTrainerCfg(
    timesteps=args.timesteps,
    headless=True,
    environment_info="log",
    close_environment_at_exit=True,
)
trainer = SequentialTrainer(env=env, agents=agent, cfg=trainer_cfg)

params_before = torch.cat([p.detach().flatten().clone() for p in models["q_network"].parameters()])
trainer.train()
params_after = torch.cat([p.detach().flatten().clone() for p in models["q_network"].parameters()])
delta = (params_after - params_before).norm().item()
print(f"[train_discrete_c51] q_network parameter L2 change over run: {delta:.6f} (must be > 0 if updates ran)")
print(f"[train_discrete_c51] replay memory filled: {len(memory)} transitions")
print(f"[train_discrete_c51] experiment dir: {agent.experiment_dir}")

import os

ckpt_dir = os.path.join(agent.experiment_dir, "checkpoints")
if os.path.isdir(ckpt_dir):
    print(f"[train_discrete_c51] checkpoints: {sorted(os.listdir(ckpt_dir))}")
else:
    print("[train_discrete_c51] WARNING: no checkpoint directory was created")

simulation_app.close()
