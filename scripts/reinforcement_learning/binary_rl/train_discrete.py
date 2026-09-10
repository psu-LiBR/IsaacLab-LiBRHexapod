# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train DQN or Double DQN on the hexapod binary contact-bit env.

Isaac Lab's skrl backend only wires PPO/AMP/IPPO/MAPPO, so this standalone script drives
skrl's DQN / DDQN directly:

    ManagerBased env (Box(6), +/-1 bit floats)
      -> DiscreteBitsActionWrapper (Discrete(64), LSB = hardware bit 1 = FrontRight)
      -> skrl IsaacLabWrapper
      -> skrl DQN / DDQN + RandomMemory + SequentialTrainer

Q-network: MLP [128, 128, 128] + ELU (same width as the rsl_rl goal policy), 64 outputs
(one Q-value per 6-bit gait pattern). Exploration: linear epsilon-greedy 1.0 -> eps_final
over the first eps_fraction of training, then constant.

Algorithm references: Mnih et al. 2015 (DQN); van Hasselt, Guez, Silver 2016
(arXiv:1509.06461, Double DQN). skrl DQN/DDQN docs: https://skrl.readthedocs.io.

Smoke run (wiring check only):
  isaaclab.bat -p scripts/reinforcement_learning/binary_rl/train_discrete.py --algo dqn ^
      --num_envs 512 --timesteps 1500 --checkpoint_interval 500

Full run:
  isaaclab.bat -p scripts/reinforcement_learning/binary_rl/train_discrete.py --algo dqn ^
      --num_envs 4096 --timesteps 100000 --experiment_name dqn_s42 --seed 42
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
parser.add_argument("--algo", choices=["dqn", "ddqn"], default="dqn")
# replay
parser.add_argument("--replay_size", type=int, default=1_000_000, help="total replay transitions across all envs")
parser.add_argument(
    "--memory_slots", type=int, default=0, help="override: replay slots per env (0 = derive from --replay_size)"
)
# agent hyperparameters (skrl defaults unless noted; batch scaled for vec envs)
parser.add_argument("--batch_size", type=int, default=4096)
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
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from binary_common import build_env, maybe_init_wandb, mlp, write_run_meta  # noqa: E402
from skrl.agents.torch.ddqn import DDQN, DDQN_CFG
from skrl.agents.torch.dqn import DQN, DQN_CFG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, Model
from skrl.trainers.torch.sequential import SequentialTrainer, SequentialTrainerCfg
from skrl.utils import set_seed

set_seed(args.seed)

# --- env ---
env = build_env(args.task, args.num_envs, args.seed, args.device)
env = wrap_env(env, wrapper="isaaclab-single-agent")
device = env.device
n_actions = int(env.action_space.n)
obs_dim = int(env.observation_space.shape[-1])
print(
    f"[train_discrete] task={args.task} algo={args.algo} num_envs={env.num_envs} obs_dim={obs_dim} actions={n_actions}"
)
assert isinstance(env.action_space, gym.spaces.Discrete) and n_actions == 64


# --- Q-network: MLP [128,128,128] + ELU, one Q-value per discrete action ---
class QNetwork(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        DeterministicMixin.__init__(self)
        self.net = mlp(self.num_observations, self.num_actions)

    def compute(self, inputs, role):
        return self.net(inputs["observations"]), {}


models = {
    "q_network": QNetwork(env.observation_space, env.action_space, device),
    "target_q_network": QNetwork(env.observation_space, env.action_space, device),
}

# --- memory: size the replay by absolute transition count, decoupled from num_envs ---
memory_slots = args.memory_slots or max(2, math.ceil(args.replay_size / env.num_envs))
memory = RandomMemory(memory_size=memory_slots, num_envs=env.num_envs, device=device)
print(
    f"[train_discrete] replay: {memory_slots} slots x {env.num_envs} envs = {memory_slots * env.num_envs} transitions"
)


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
experiment_name = args.experiment_name or f"{args.algo}_{args.task}"
agent_cfg.experiment.directory = args.directory
agent_cfg.experiment.experiment_name = experiment_name
agent_cfg.experiment.write_interval = args.write_interval
agent_cfg.experiment.checkpoint_interval = args.checkpoint_interval

agent = agent_cls(
    models=models,
    memory=memory,
    observation_space=env.observation_space,
    action_space=env.action_space,
    device=device,
    cfg=agent_cfg,
)

if args.resume:
    agent.load(args.resume)
    print(f"[train_discrete] warm-started from {args.resume}")

trainer = SequentialTrainer(
    env=env,
    agents=agent,
    cfg=SequentialTrainerCfg(
        timesteps=args.timesteps, headless=True, environment_info="log", close_environment_at_exit=True
    ),
)

write_run_meta(
    os.path.join(args.directory, experiment_name),
    algo=args.algo,
    obs_dim=obs_dim,
    n_actions=n_actions,
    seed=args.seed,
    num_envs=args.num_envs,
    timesteps=args.timesteps,
    replay_transitions=memory_slots * env.num_envs,
)

wandb_run = maybe_init_wandb(args, experiment_name, args.algo, {"replay_transitions": memory_slots * env.num_envs})

params_before = torch.cat([p.detach().flatten().clone() for p in models["q_network"].parameters()])
trainer.train()
params_after = torch.cat([p.detach().flatten().clone() for p in models["q_network"].parameters()])
print(f"[train_discrete] q_network parameter L2 change over run: {(params_after - params_before).norm().item():.6f}")
print(f"[train_discrete] replay memory filled: {len(memory)} transitions")
print(f"[train_discrete] experiment dir: {agent.experiment_dir}")

ckpt_dir = os.path.join(agent.experiment_dir, "checkpoints")
print(f"[train_discrete] checkpoints: {sorted(os.listdir(ckpt_dir)) if os.path.isdir(ckpt_dir) else 'NONE WRITTEN'}")

if wandb_run is not None:
    wandb_run.finish()

simulation_app.close()
