# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""QR-DQN (Quantile Regression DQN) on the hexapod binary env (single file).

Hand-written after "Distributional Reinforcement Learning with Quantile Regression"
(Dabney, Rowland, Bellemare, Munos; arXiv:1710.10044, 2017-10; AAAI 2018).

WHY THIS ALGORITHM (recorded per project rules — this is an AI selection, not a
teacher requirement): the task asked for "Rainbow or a near relative".  Rainbow
itself is not runnable here: the Isaac venv is read-only for this session and
neither Tianshou, CleanRL nor Dopamine is installed (`pip list` shows none), so
there is no packaged Rainbow to install.  C51 -- Rainbow's distributional
component -- already has its own corrected-env run (`c51_fixed_20260831`,
retrained by the MCTS session).  QR-DQN is the other distributional member of
that family and the exact fallback the task named.  It is therefore reported as
**QR-DQN, self-written**, NOT as "Rainbow".

Relation to the existing C51 run: `train_discrete_c51.py` is the direct template.
Everything outside the distributional head and its loss is byte-for-byte the same
structure and the same default hyperparameters, so the C51 vs QR-DQN comparison
differs only in how the return distribution is represented:
  - C51    : fixed support [v_min, v_max], 51 atoms, learned probabilities,
             cross-entropy against the projected Bellman target.
  - QR-DQN : fixed uniform probabilities 1/N, N learned quantile LOCATIONS,
             quantile Huber loss -- no support bounds, no projection step.
Double-DQN action selection for the target is kept identical to the C51 script
(Rainbow convention), so that is not a new difference either.

Checkpoints are written by skrl under "q_network"/"policy" state dicts with
net.* names, which is what eval_protocol.py's `make_net_policy` loads; the mean
over quantiles is returned by `compute`, so the inherited epsilon-greedy/argmax
machinery and the offline greedy replay both work unchanged.

Smoke:  CUDA_VISIBLE_DEVICES=<idle> python scripts/reinforcement_learning/train_discrete_qrdqn.py \
            --num_envs 16 --timesteps 200 --batch_size 256 --memory_slots 200 --checkpoint_interval 100
Full :  CUDA_VISIBLE_DEVICES=<idle> python scripts/reinforcement_learning/train_discrete_qrdqn.py \
            --num_envs 4096 --timesteps 100000 --checkpoint_interval 1000 --write_interval 50 \
            --experiment_name qrdqn_fixed_20260831 --directory runs_discrete_fixed
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-Goal-Flat-Hexapod-Binary-v0")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--timesteps", type=int, default=200)
parser.add_argument("--seed", type=int, default=42)
# --- distributional head (the only structural difference vs train_discrete_c51.py) ---
parser.add_argument("--quantiles", type=int, default=51,
                    help="N in the paper; 51 chosen to equal the C51 run's atom count")
parser.add_argument("--kappa", type=float, default=1.0, help="Huber threshold (paper default)")
# --- everything below: identical defaults to train_discrete_c51.py ---
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
parser.add_argument("--experiment_name", default="", help="default: qrdqn_<task>")
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

from discrete_action_wrapper import DiscreteBitsActionWrapper

set_seed(args.seed)

# --- env ---
env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=args.num_envs)
env_cfg.seed = args.seed
env = gym.make(args.task, cfg=env_cfg)
env = DiscreteBitsActionWrapper(env, n_bits=6)
env = wrap_env(env, wrapper="isaaclab-single-agent")
device = env.device
print(f"[train_discrete_qrdqn] task={args.task} num_envs={env.num_envs} "
      f"quantiles={args.quantiles} kappa={args.kappa}")
assert isinstance(env.action_space, gym.spaces.Discrete) and env.action_space.n == 64

N_ACTIONS = 64
# tau_hat_i = (i + 0.5) / N  (paper eq. 9, midpoint quantile targets)
TAU_HAT = ((torch.arange(args.quantiles, device=device, dtype=torch.float32) + 0.5)
           / args.quantiles)


class QRNetwork(DeterministicMixin, Model):
    """MLP [128,128,128] -> 64 x N quantile LOCATIONS.  ``act`` returns the mean over
    quantiles (= Q), so the inherited DQN epsilon-greedy/argmax machinery works
    unchanged; raw quantiles are exposed via :meth:`quantiles` for the QR update."""

    def __init__(self, observation_space, action_space, device, n_quantiles, hidden=(128, 128, 128)):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        DeterministicMixin.__init__(self)
        self.n_quantiles = n_quantiles
        layers, in_dim = [], self.num_observations
        for h in hidden:
            layers += [torch.nn.Linear(in_dim, h), torch.nn.ELU()]
            in_dim = h
        layers += [torch.nn.Linear(in_dim, N_ACTIONS * n_quantiles)]
        self.net = torch.nn.Sequential(*layers)

    def quantiles(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations).view(-1, N_ACTIONS, self.n_quantiles)

    def compute(self, inputs, role):
        q = self.quantiles(inputs["observations"]).mean(-1)  # [B, N_ACTIONS]
        return q, {}


class QRDQN(DQN):
    """skrl DQN with the update replaced by the quantile-regression Huber loss.

    Action selection for the target uses the online net's mean-Q (double variant,
    same convention as the C51 script)."""

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
                # double action selection: argmax of ONLINE net mean-Q on next obs
                online_next_q, _ = self.q_network.act({"observations": next_obs, "states": None}, role="q_network")
                a_star = torch.argmax(online_next_q, dim=1)  # [B]
                # target quantile locations for those actions
                theta_next = self.target_q_network.quantiles(next_obs)[arange, a_star]  # [B, N]
                not_done = sampled_terminated.logical_not().float().view(-1, 1)
                # distributional Bellman: T theta = r + gamma * (1-done) * theta'  (no projection)
                target = sampled_rewards.view(-1, 1) + self.cfg.discount_factor * not_done * theta_next  # [B, N]

            theta = self.q_network.quantiles(obs)[arange, sampled_actions.view(-1).long()]  # [B, N]

            # pairwise TD errors u_ij = target_j - theta_i    -> [B, N(pred), N(target)]
            u = target.unsqueeze(1) - theta.unsqueeze(2)
            huber = torch.where(
                u.abs() <= args.kappa,
                0.5 * u.pow(2),
                args.kappa * (u.abs() - 0.5 * args.kappa),
            )
            # rho^kappa_tau(u) = |tau - 1{u < 0}| * L_kappa(u) / kappa   (paper eq. 10)
            weight = (TAU_HAT.view(1, -1, 1) - (u.detach() < 0).float()).abs()
            loss = (weight * huber / args.kappa).sum(dim=1).mean(dim=1).mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self._update_counter += 1
            if not self._update_counter % self.cfg.target_update_interval:
                self.target_q_network.update_parameters(self.q_network, polyak=self.cfg.polyak)

            self.track_data("Loss / Q-network loss", loss.item())
            self.track_data("Target / Target (mean)", target.mean().item())


models = {
    "q_network": QRNetwork(env.observation_space, env.action_space, device, args.quantiles),
    "target_q_network": QRNetwork(env.observation_space, env.action_space, device, args.quantiles),
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
agent_cfg.experiment.experiment_name = args.experiment_name or f"qrdqn_{args.task}"
agent_cfg.experiment.write_interval = args.write_interval
agent_cfg.experiment.checkpoint_interval = args.checkpoint_interval

agent = QRDQN(
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
print(f"[train_discrete_qrdqn] q_network parameter L2 change over run: {delta:.6f} (must be > 0 if updates ran)")
print(f"[train_discrete_qrdqn] replay memory filled: {len(memory)} transitions")
print(f"[train_discrete_qrdqn] experiment dir: {agent.experiment_dir}")

import os

ckpt_dir = os.path.join(agent.experiment_dir, "checkpoints")
if os.path.isdir(ckpt_dir):
    print(f"[train_discrete_qrdqn] checkpoints: {sorted(os.listdir(ckpt_dir))}")
else:
    print("[train_discrete_qrdqn] WARNING: no checkpoint directory was created")

simulation_app.close()
