# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train a discrete-action (DQN / DDQN) agent on the hexapod binary contact-bit env.

The Isaac Lab skrl backend only wires PPO/AMP/IPPO/MAPPO, so this standalone script
drives skrl's DQN / DDQN directly:

    ManagerBased env (Box(6), +-1 bit floats)
      -> DiscreteBitsActionWrapper (Discrete(64), LSB = hardware bit 1 = FrontRight)
      -> skrl IsaacLabWrapper
      -> skrl DQN / DDQN + RandomMemory + SequentialTrainer

Q-network: MLP [128, 128, 128] + ELU (same width as the existing rsl_rl policy),
64 outputs (one Q-value per 6-bit gait pattern).  Exploration: linear epsilon-greedy
1.0 -> eps_final over the first eps_fraction of total timesteps, then constant.

Smoke run (wiring check only):
  CUDA_VISIBLE_DEVICES=<idle> ./isaaclab.sh -p train_discrete.py --algo dqn \
      --num_envs 16 --timesteps 200 --learning_starts 20 --random_timesteps 10 \
      --batch_size 256 --memory_slots 200 --checkpoint_interval 100

Full run example:
  CUDA_VISIBLE_DEVICES=<idle> ./isaaclab.sh -p train_discrete.py --algo dqn \
      --num_envs 4096 --timesteps 100000
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-Goal-Flat-Hexapod-Binary-v0")
parser.add_argument("--algo", choices=["dqn", "ddqn"], default="dqn")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--timesteps", type=int, default=200)
parser.add_argument("--seed", type=int, default=42)
# agent hyperparameters (skrl defaults unless noted; batch/memory scaled for vec envs)
parser.add_argument("--batch_size", type=int, default=4096)
parser.add_argument("--memory_slots", type=int, default=500, help="replay slots per env (total = slots * num_envs)")
parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument("--discount", type=float, default=0.99)
parser.add_argument("--polyak", type=float, default=0.005)
parser.add_argument("--target_update_interval", type=int, default=10)
parser.add_argument("--gradient_steps", type=int, default=1)
parser.add_argument("--random_timesteps", type=int, default=20)
parser.add_argument("--learning_starts", type=int, default=50)
parser.add_argument("--eps_initial", type=float, default=1.0)
parser.add_argument("--eps_final", type=float, default=0.05)
parser.add_argument("--eps_fraction", type=float, default=0.2, help="fraction of timesteps over which epsilon decays")
# experiment bookkeeping
parser.add_argument("--experiment_name", default="", help="default: <algo>_<task>")
parser.add_argument("--resume", default="", help="checkpoint to warm-start from (nets+optimizer resume; replay buffer and epsilon schedule restart)")
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

import isaaclab_tasks  # noqa: F401  (registers tasks, incl. the Binary variants)

try:
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
except ImportError:
    from isaaclab_tasks.utils import parse_env_cfg

from skrl.agents.torch.ddqn import DDQN, DDQN_CFG
from skrl.agents.torch.dqn import DQN, DQN_CFG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, Model
from skrl.trainers.torch.sequential import SequentialTrainer, SequentialTrainerCfg
from skrl.utils import set_seed

from discrete_action_wrapper import DiscreteBitsActionWrapper

set_seed(args.seed)

# --- env ---
env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=args.num_envs)
env_cfg.seed = args.seed
env = gym.make(args.task, cfg=env_cfg)
env = DiscreteBitsActionWrapper(env, n_bits=6)
env = wrap_env(env, wrapper="isaaclab-single-agent")
device = env.device
print(f"[train_discrete] task={args.task} algo={args.algo} num_envs={env.num_envs}")
print(f"[train_discrete] observation_space={env.observation_space} action_space={env.action_space}")
assert isinstance(env.action_space, gym.spaces.Discrete) and env.action_space.n == 64


# --- Q-network: MLP [128,128,128] + ELU, one Q-value per discrete action ---
class QNetwork(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, hidden=(128, 128, 128)):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        DeterministicMixin.__init__(self)
        layers, in_dim = [], self.num_observations
        for h in hidden:
            layers += [torch.nn.Linear(in_dim, h), torch.nn.ELU()]
            in_dim = h
        layers += [torch.nn.Linear(in_dim, self.num_actions)]
        self.net = torch.nn.Sequential(*layers)

    def compute(self, inputs, role):
        return self.net(inputs["observations"]), {}


models = {
    "q_network": QNetwork(env.observation_space, env.action_space, device),
    "target_q_network": QNetwork(env.observation_space, env.action_space, device),
}

# --- memory ---
memory = RandomMemory(memory_size=args.memory_slots, num_envs=env.num_envs, device=device)


# --- agent ---
def epsilon_schedule(timestep: int, timesteps: int) -> float:
    """Linear decay eps_initial -> eps_final over the first eps_fraction of training."""
    horizon = max(1, int(args.eps_fraction * timesteps))
    frac = min(1.0, timestep / horizon)
    return args.eps_initial + (args.eps_final - args.eps_initial) * frac


cfg_cls = DQN_CFG if args.algo == "dqn" else DDQN_CFG
agent_cls = DQN if args.algo == "dqn" else DDQN
agent_cfg = cfg_cls(
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
agent_cfg.experiment.experiment_name = args.experiment_name or f"{args.algo}_{args.task}"
agent_cfg.experiment.write_interval = args.write_interval
agent_cfg.experiment.checkpoint_interval = args.checkpoint_interval

agent = agent_cls(
    models=models,
    memory=memory,
    observation_space=env.observation_space,
    state_space=None,
    action_space=env.action_space,
    device=device,
    cfg=agent_cfg,
)

# --- train ---
if args.resume:
    agent.load(args.resume)
    print(f"[train_discrete] warm-started from {args.resume}")

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
print(f"[train_discrete] q_network parameter L2 change over run: {delta:.6f} (must be > 0 if updates ran)")
print(f"[train_discrete] replay memory filled: {len(memory)} transitions")
print(f"[train_discrete] experiment dir: {agent.experiment_dir}")

import os

ckpt_dir = os.path.join(agent.experiment_dir, "checkpoints")
if os.path.isdir(ckpt_dir):
    print(f"[train_discrete] checkpoints: {sorted(os.listdir(ckpt_dir))}")
else:
    print("[train_discrete] WARNING: no checkpoint directory was created")

simulation_app.close()
