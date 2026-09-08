# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train a categorical (discrete) PPO agent on the hexapod binary contact-bit env.

Third wiring route next to DQN/DDQN (``train_discrete.py``): same
``DiscreteBitsActionWrapper`` (Discrete(64), LSB = hardware bit 1 = FrontRight), but
on-policy skrl PPO with a Categorical policy over the 64 six-bit gait patterns.

Hyperparameters follow the repo's rsl_rl goal-task PPO config
(``agents/rsl_rl_ppo_goal_cfg.py``) where transferable: rollouts 96, epochs 5,
mini-batches 4, gamma 0.9995, lambda 0.97, entropy 0.003, adaptive-KL lr (0.01),
grad-norm 1.0, MLP [128,128,128] + ELU, running obs normalization.  Differences:
the critic here is symmetric (uses the policy observations, not the "critic" group).

Smoke run (wiring check only):
  CUDA_VISIBLE_DEVICES=<idle> ./isaaclab.sh -p scripts/reinforcement_learning/train_discrete_ppo.py \
      --num_envs 16 --timesteps 200 --checkpoint_interval 100

Full run example:
  CUDA_VISIBLE_DEVICES=<idle> ./isaaclab.sh -p scripts/reinforcement_learning/train_discrete_ppo.py \
      --num_envs 4096 --timesteps 100000
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-Goal-Flat-Hexapod-Binary-v0")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--timesteps", type=int, default=200)
parser.add_argument("--seed", type=int, default=42)
# PPO hyperparameters (aligned to the repo rsl_rl goal cfg unless noted)
parser.add_argument("--rollouts", type=int, default=96)
parser.add_argument("--learning_epochs", type=int, default=5)
parser.add_argument("--mini_batches", type=int, default=4)
parser.add_argument("--discount", type=float, default=0.9995)
parser.add_argument("--gae_lambda", type=float, default=0.97)
parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument("--kl_threshold", type=float, default=0.01, help="adaptive-KL lr scheduler target")
parser.add_argument("--entropy_loss_scale", type=float, default=0.003)
parser.add_argument("--value_loss_scale", type=float, default=1.0)
parser.add_argument("--grad_norm_clip", type=float, default=1.0)
parser.add_argument("--ratio_clip", type=float, default=0.2)
# experiment bookkeeping
parser.add_argument("--experiment_name", default="", help="default: ppo_<task>")
parser.add_argument("--resume", default="", help="checkpoint to warm-start from (nets+optimizer+preprocessors resume)")
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

import isaaclab_tasks  # noqa: F401

try:
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
except ImportError:
    from isaaclab_tasks.utils import parse_env_cfg

from skrl.agents.torch.ppo import PPO, PPO_CFG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import CategoricalMixin, DeterministicMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveLR
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
print(f"[train_discrete_ppo] task={args.task} num_envs={env.num_envs}")
print(f"[train_discrete_ppo] observation_space={env.observation_space} action_space={env.action_space}")
assert isinstance(env.action_space, gym.spaces.Discrete) and env.action_space.n == 64


def _mlp(in_dim: int, out_dim: int, hidden=(128, 128, 128)) -> torch.nn.Sequential:
    layers = []
    for h in hidden:
        layers += [torch.nn.Linear(in_dim, h), torch.nn.ELU()]
        in_dim = h
    layers += [torch.nn.Linear(in_dim, out_dim)]
    return torch.nn.Sequential(*layers)


class Policy(CategoricalMixin, Model):
    """Categorical policy: logits over the 64 six-bit gait patterns."""

    def __init__(self, observation_space, action_space, device):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        CategoricalMixin.__init__(self, unnormalized_log_prob=True)
        self.net = _mlp(self.num_observations, self.num_actions)

    def compute(self, inputs, role):
        return self.net(inputs["observations"]), {}


class Value(DeterministicMixin, Model):
    """Symmetric critic on the policy observations."""

    def __init__(self, observation_space, device):
        Model.__init__(self, observation_space=observation_space, action_space=None, device=device)
        DeterministicMixin.__init__(self)
        self.net = _mlp(self.num_observations, 1)

    def compute(self, inputs, role):
        return self.net(inputs["observations"]), {}


models = {
    "policy": Policy(env.observation_space, env.action_space, device),
    "value": Value(env.observation_space, device),
}

memory = RandomMemory(memory_size=args.rollouts, num_envs=env.num_envs, device=device)

agent_cfg = PPO_CFG(
    rollouts=args.rollouts,
    learning_epochs=args.learning_epochs,
    mini_batches=args.mini_batches,
    discount_factor=args.discount,
    gae_lambda=args.gae_lambda,
    learning_rate=args.learning_rate,
    learning_rate_scheduler=KLAdaptiveLR,
    learning_rate_scheduler_kwargs={"kl_threshold": args.kl_threshold},
    grad_norm_clip=args.grad_norm_clip,
    ratio_clip=args.ratio_clip,
    value_clip=args.ratio_clip,
    entropy_loss_scale=args.entropy_loss_scale,
    value_loss_scale=args.value_loss_scale,
    observation_preprocessor=RunningStandardScaler,
    observation_preprocessor_kwargs={"size": env.observation_space, "device": device},
    value_preprocessor=RunningStandardScaler,
    value_preprocessor_kwargs={"size": 1, "device": device},
)
agent_cfg.experiment.directory = args.directory
agent_cfg.experiment.experiment_name = args.experiment_name or f"ppo_{args.task}"
agent_cfg.experiment.write_interval = args.write_interval
agent_cfg.experiment.checkpoint_interval = args.checkpoint_interval

agent = PPO(
    models=models,
    memory=memory,
    observation_space=env.observation_space,
    state_space=None,
    action_space=env.action_space,
    device=device,
    cfg=agent_cfg,
)

if args.resume:
    agent.load(args.resume)
    print(f"[train_discrete_ppo] warm-started from {args.resume}")

trainer_cfg = SequentialTrainerCfg(
    timesteps=args.timesteps,
    headless=True,
    environment_info="log",
    close_environment_at_exit=True,
)
trainer = SequentialTrainer(env=env, agents=agent, cfg=trainer_cfg)

params_before = torch.cat([p.detach().flatten().clone() for p in models["policy"].parameters()])
trainer.train()
params_after = torch.cat([p.detach().flatten().clone() for p in models["policy"].parameters()])
delta = (params_after - params_before).norm().item()
print(f"[train_discrete_ppo] policy parameter L2 change over run: {delta:.6f} (must be > 0 if updates ran)")
print(f"[train_discrete_ppo] experiment dir: {agent.experiment_dir}")

import os

ckpt_dir = os.path.join(agent.experiment_dir, "checkpoints")
if os.path.isdir(ckpt_dir):
    print(f"[train_discrete_ppo] checkpoints: {sorted(os.listdir(ckpt_dir))}")
else:
    print("[train_discrete_ppo] WARNING: no checkpoint directory was created")

simulation_app.close()
