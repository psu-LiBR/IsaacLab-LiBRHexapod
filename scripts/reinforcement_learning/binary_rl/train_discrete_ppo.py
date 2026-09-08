# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train categorical PPO (optionally with an action mask) on the hexapod binary env.

On-policy skrl PPO with a Categorical policy over the 64 six-bit gait patterns, next to
DQN/DDQN (``train_discrete.py``) and SAC-D (``train_sac_d.py``):

    ManagerBased env (Box(6), +/-1 bit floats)
      -> DiscreteBitsActionWrapper (Discrete(64), LSB = hardware bit 1 = FrontRight)
      -> skrl IsaacLabWrapper (routes the "policy" / "critic" obs groups to actor / critic)
      -> skrl PPO + RandomMemory + SequentialTrainer

Hyperparameters follow the repo's rsl_rl goal-task PPO config
(``agents/rsl_rl_ppo_goal_cfg.py`` / ``HexapodGoalPPORunnerCfg``): rollouts 96, epochs 5,
mini-batches 4, gamma 0.9995, lambda 0.97, entropy 0.003, KL-adaptive LR (target 0.01),
grad-norm 1.0, MLP [128,128,128] + ELU, running obs/value normalization. Unlike the
collaborator's first version the critic here is **asymmetric** — it reads the "critic"
observation group (adds ground-truth ``base_lin_vel``), matching every other hexapod task.

``--mask`` restricts the policy to the legal action set (>= ``--mask_min_stance`` stance
legs, plus ``--mask_whitelist``); the illegal logits are set to ``-inf`` before sampling
and before the PPO log-prob, and the entropy uses the masked-safe form. See
``binary_action_mask.py``. The chosen mask is recorded in ``run_meta.json`` so the
greedy evaluator restricts its argmax the same way.
Masked-action-space RL reference: Huang & Ontanon 2022 (arXiv:2006.14171).

Smoke run (wiring check only):
  isaaclab.bat -p scripts/reinforcement_learning/binary_rl/train_discrete_ppo.py ^
      --num_envs 512 --timesteps 1500 --checkpoint_interval 500

Full run (masked):
  isaaclab.bat -p scripts/reinforcement_learning/binary_rl/train_discrete_ppo.py ^
      --num_envs 4096 --timesteps 100000 --mask --experiment_name ppo_masked_s42 --seed 42
"""

import argparse
import os
import sys

# allow running from any CWD (e.g. the repo root, so relative asset paths resolve)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from binary_common import add_common_cli, add_mask_cli  # noqa: E402

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser()
add_common_cli(parser)
add_mask_cli(parser)
# PPO hyperparameters (aligned to HexapodGoalPPORunnerCfg unless noted)
parser.add_argument("--rollouts", type=int, default=96)
parser.add_argument("--learning_epochs", type=int, default=5)
parser.add_argument("--mini_batches", type=int, default=4)
parser.add_argument("--discount", type=float, default=0.9995)
parser.add_argument("--gae_lambda", type=float, default=0.97)
parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument("--kl_threshold", type=float, default=0.01, help="adaptive-KL LR scheduler target")
parser.add_argument("--entropy_loss_scale", type=float, default=0.003)
parser.add_argument("--value_loss_scale", type=float, default=1.0)
parser.add_argument("--grad_norm_clip", type=float, default=1.0)
parser.add_argument("--ratio_clip", type=float, default=0.2)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from binary_action_mask import legal_action_mask  # noqa: E402
from binary_common import MaskedCategoricalMixin, build_env, mask_meta, mlp, write_run_meta
from skrl.agents.torch.ppo import PPO, PPO_CFG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import CategoricalMixin, DeterministicMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveLR
from skrl.trainers.torch.sequential import SequentialTrainer, SequentialTrainerCfg
from skrl.utils import set_seed

set_seed(args.seed)

# --- env ---
env = build_env(args.task, args.num_envs, args.seed, args.device)
env = wrap_env(env, wrapper="isaaclab-single-agent")
device = env.device
n_actions = int(env.action_space.n)
obs_dim = int(env.observation_space.shape[-1])
state_dim = int(env.state_space.shape[-1]) if env.state_space is not None else obs_dim
print(
    f"[train_discrete_ppo] task={args.task} num_envs={env.num_envs} "
    f"obs_dim={obs_dim} state_dim={state_dim} actions={n_actions}"
)
assert isinstance(env.action_space, gym.spaces.Discrete) and n_actions == 64


# --- models: asymmetric actor (policy obs) / critic (critic obs group) ---
_mixin = MaskedCategoricalMixin if args.mask else CategoricalMixin


class Policy(_mixin, Model):
    """Categorical policy: logits over the 64 six-bit gait patterns."""

    def __init__(self, observation_space, action_space, device):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        _mixin.__init__(self, unnormalized_log_prob=True)
        self.net = mlp(self.num_observations, self.num_actions)

    def compute(self, inputs, role):
        return self.net(inputs["observations"]), {}


class Value(DeterministicMixin, Model):
    """Asymmetric critic: reads the "critic" observation group (env state)."""

    def __init__(self, state_space, device):
        Model.__init__(self, observation_space=state_space, action_space=None, device=device)
        DeterministicMixin.__init__(self)
        self.net = mlp(self.num_observations, 1)

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}


policy = Policy(env.observation_space, env.action_space, device)
if args.mask:
    mask = legal_action_mask(args.mask_min_stance, tuple(args.mask_whitelist), n_bits=6, device=device)
    policy.set_action_mask(mask)
    print(f"[train_discrete_ppo] action mask on: {int(mask.sum())}/{n_actions} legal actions")

models = {"policy": policy, "value": Value(env.state_space or env.observation_space, device)}

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
    state_preprocessor=RunningStandardScaler,
    state_preprocessor_kwargs={"size": env.state_space or env.observation_space, "device": device},
    value_preprocessor=RunningStandardScaler,
    value_preprocessor_kwargs={"size": 1, "device": device},
)
experiment_name = args.experiment_name or (f"ppo_masked_{args.task}" if args.mask else f"ppo_{args.task}")
agent_cfg.experiment.directory = args.directory
agent_cfg.experiment.experiment_name = experiment_name
agent_cfg.experiment.write_interval = args.write_interval
agent_cfg.experiment.checkpoint_interval = args.checkpoint_interval

agent = PPO(
    models=models,
    memory=memory,
    observation_space=env.observation_space,
    state_space=env.state_space,
    action_space=env.action_space,
    device=device,
    cfg=agent_cfg,
)

if args.resume:
    agent.load(args.resume)
    print(f"[train_discrete_ppo] warm-started from {args.resume}")

trainer = SequentialTrainer(
    env=env,
    agents=agent,
    cfg=SequentialTrainerCfg(
        timesteps=args.timesteps, headless=True, environment_info="log", close_environment_at_exit=True
    ),
)

extra_meta = mask_meta(args.mask_min_stance, args.mask_whitelist) if args.mask else {}
write_run_meta(
    os.path.join(args.directory, experiment_name),
    algo="ppo_masked" if args.mask else "ppo",
    obs_dim=obs_dim,
    n_actions=n_actions,
    seed=args.seed,
    num_envs=args.num_envs,
    timesteps=args.timesteps,
    asymmetric_critic=env.state_space is not None,
    **extra_meta,
)

params_before = torch.cat([p.detach().flatten().clone() for p in models["policy"].parameters()])
trainer.train()
params_after = torch.cat([p.detach().flatten().clone() for p in models["policy"].parameters()])
print(f"[train_discrete_ppo] policy parameter L2 change over run: {(params_after - params_before).norm().item():.6f}")
print(f"[train_discrete_ppo] experiment dir: {agent.experiment_dir}")

ckpt_dir = os.path.join(agent.experiment_dir, "checkpoints")
print(
    f"[train_discrete_ppo] checkpoints: {sorted(os.listdir(ckpt_dir)) if os.path.isdir(ckpt_dir) else 'NONE WRITTEN'}"
)

simulation_app.close()
