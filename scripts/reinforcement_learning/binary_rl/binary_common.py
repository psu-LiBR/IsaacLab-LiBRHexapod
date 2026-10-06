# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared harness for the binary-contact RL training scripts.

Keeps the env build, CLI surface, network builder, checkpoint sidecar, and the
masked-categorical policy mixin in one place so DQN / Double-DQN / PPO / masked-PPO /
SAC-D stay directly comparable. Isaac Sim is imported lazily inside :func:`build_env`
so this module can be imported (and its non-env helpers unit-tested) without launching
the simulator.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
from typing import Any

import torch
from binary_action_mask import apply_logit_mask, legal_action_mask

DEFAULT_TASK = "Isaac-Goal-Flat-Hexapod-Binary-v0"
DEFAULT_HIDDEN = (128, 128, 128)
ACTIVATIONS: dict[str, type[torch.nn.Module]] = {
    "elu": torch.nn.ELU,
    "relu": torch.nn.ReLU,
    "tanh": torch.nn.Tanh,
    "gelu": torch.nn.GELU,
}

CHECKPOINT_SCHEMA_VERSION = 1


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def add_common_cli(parser: argparse.ArgumentParser) -> None:
    """Add the CLI arguments every binary-RL training script shares."""
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument(
        "--timesteps", type=int, default=100_000, help="trainer iterations; env-steps = timesteps * num_envs"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experiment_name", default="", help="run folder name (default: <algo>_<task>)")
    parser.add_argument("--directory", default="runs_binary", help="parent folder for run outputs")
    parser.add_argument("--checkpoint_interval", type=int, default=2000)
    parser.add_argument("--write_interval", type=int, default=50)
    parser.add_argument("--resume", default="", help="checkpoint to warm-start from")
    # Weights & Biases (the binary-RL equivalent of rsl_rl's --logger wandb --log_project_name)
    parser.add_argument("--wandb", action="store_true", help="log this run to Weights & Biases")
    parser.add_argument("--wandb_project", default="discrete-RL", help="W&B project name")
    parser.add_argument("--wandb_group", default="", help="W&B group (default: the algorithm name)")
    parser.add_argument("--wandb_name", default="", help="W&B run name (default: --experiment_name)")
    # Contact-switch penalty knobs. Defaults leave the env's rewards untouched.
    parser.add_argument(
        "--action_rate_multiplier",
        type=float,
        default=1.0,
        help="multiply the env's action_rate_l2 weight by this factor (the main knob for limiting contact-bit "
        "switching; 1.0 keeps the env's weight, e.g. -5e-4 -> -0.05 at 100)",
    )
    parser.add_argument(
        "--action_switch_penalty",
        type=float,
        default=0.0,
        help="OFF by default. If > 0, add the action_switch_count reward with this cost per flipped leg per step "
        "[reward units]. For +-1 bits it duplicates action_rate_l2, so prefer --action_rate_multiplier",
    )


def add_mask_cli(parser: argparse.ArgumentParser) -> None:
    """Add the masked-categorical-policy arguments (used by masked PPO)."""
    parser.add_argument(
        "--mask",
        action="store_true",
        help="restrict the policy to the legal action set (>= --mask_min_stance stance legs, plus --mask_whitelist)",
    )
    parser.add_argument("--mask_min_stance", type=int, default=4, help="minimum stance-leg count kept legal")
    parser.add_argument(
        "--mask_whitelist",
        type=int,
        nargs="*",
        default=[25, 38],
        help="action ids always kept legal regardless of stance-leg count (default: the two tripods)",
    )


# --------------------------------------------------------------------------------------
# Contact-switch penalty
# --------------------------------------------------------------------------------------
def scale_action_rate_weight(env_cfg: Any, multiplier: float) -> None:
    """Multiply the env's ``action_rate_l2`` reward weight by ``multiplier``.

    For the +-1 leg bits ``action_rate_l2`` equals ``4 x`` the number of flipped legs, so this is the main knob for
    limiting contact-bit switching. ``1.0`` leaves the env untouched.

    Args:
        env_cfg: The parsed env config whose ``rewards.action_rate_l2`` term is scaled in place.
        multiplier: Non-negative factor applied to the term's current weight.

    Raises:
        ValueError: If ``multiplier`` is negative, or the env has no ``action_rate_l2`` term to scale.
    """
    if multiplier < 0:
        raise ValueError(f"action_rate_multiplier must be >= 0, got {multiplier}")
    if multiplier == 1.0:
        return
    term = getattr(env_cfg.rewards, "action_rate_l2", None)
    if term is None:
        raise ValueError("the env has no action_rate_l2 reward term to scale")
    term.weight = term.weight * multiplier


def add_action_switch_penalty(env_cfg: Any, penalty: float) -> None:
    """Add the optional :func:`switch_command_reward.action_switch_count` reward term.

    Args:
        env_cfg: The parsed env config; the term is added to ``rewards`` in place.
        penalty: Cost per flipped leg per step [reward units]. ``0`` adds nothing.

    Raises:
        ValueError: If ``penalty`` is negative.
    """
    if penalty < 0:
        raise ValueError(f"action_switch_penalty must be >= 0, got {penalty}")
    if penalty == 0:
        return
    from switch_command_reward import action_switch_count

    from isaaclab.managers import RewardTermCfg

    env_cfg.rewards.action_switch_count = RewardTermCfg(func=action_switch_count, weight=-penalty)


def switch_penalty_meta(env: Any, args: argparse.Namespace) -> dict[str, Any]:
    """Record the contact-switch penalty settings and the reward weights they resolved to, for ``run_meta.json``.

    Args:
        env: Either a ``DiscreteBitsActionWrapper`` (whose ``unwrapped`` is the wrapper itself, so the raw env is
            read from ``base_env``) or a plain gym env.
        args: The parsed CLI namespace holding the two switch-penalty flags.
    """
    base = getattr(env, "base_env", None) or env.unwrapped
    rewards = base.cfg.rewards
    action_rate = getattr(rewards, "action_rate_l2", None)
    switch_count = getattr(rewards, "action_switch_count", None)
    return {
        "action_rate_multiplier": args.action_rate_multiplier,
        "action_rate_l2_weight": None if action_rate is None else action_rate.weight,
        "action_switch_penalty": args.action_switch_penalty,
        "action_switch_count_weight": None if switch_count is None else switch_count.weight,
    }


# --------------------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------------------
def build_env(
    task: str,
    num_envs: int,
    seed: int,
    device: str = "cuda:0",
    n_bits: int = 6,
    compute_final_obs: bool = False,
    action_rate_multiplier: float = 1.0,
    action_switch_penalty: float = 0.0,
):
    """Build the binary-contact env wrapped as ``Discrete(2**n_bits)``.

    Args:
        task: Gym id of the binary env (train or play variant).
        num_envs: Number of parallel envs.
        seed: Environment seed.
        device: Simulation/compute device.
        n_bits: Number of leg contact bits (6 for the hexapod).
        compute_final_obs: When ``True``, sets ``env_cfg.compute_final_obs`` so
            ``env.step`` exposes the pre-reset observation in ``extras["final_obs"]``
            (needed for timeout-aware value bootstrapping in the off-policy trainers).
        action_rate_multiplier: Factor applied to the env's ``action_rate_l2`` weight (see
            :func:`scale_action_rate_weight`).
        action_switch_penalty: Cost per flipped leg per step of the optional ``action_switch_count`` reward;
            ``0`` leaves it off (see :func:`add_action_switch_penalty`).

    Returns:
        The ``DiscreteBitsActionWrapper``-wrapped environment.
    """
    import gymnasium as gym

    import isaaclab_tasks  # noqa: F401

    try:
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
    except ImportError:
        from isaaclab_tasks.utils import parse_env_cfg

    from discrete_action_wrapper import DiscreteBitsActionWrapper

    env_cfg = parse_env_cfg(task, device=device, num_envs=num_envs)
    env_cfg.seed = seed
    if compute_final_obs:
        env_cfg.compute_final_obs = True
    scale_action_rate_weight(env_cfg, action_rate_multiplier)
    add_action_switch_penalty(env_cfg, action_switch_penalty)
    env = gym.make(task, cfg=env_cfg)
    return DiscreteBitsActionWrapper(env, n_bits=n_bits)


# --------------------------------------------------------------------------------------
# Networks
# --------------------------------------------------------------------------------------
def mlp(
    in_dim: int,
    out_dim: int,
    hidden: tuple[int, ...] = DEFAULT_HIDDEN,
    activation: str = "elu",
) -> torch.nn.Sequential:
    """A plain MLP, ``in_dim -> hidden... -> out_dim``, matching the project convention."""
    act = ACTIVATIONS[activation]
    layers: list[torch.nn.Module] = []
    d = in_dim
    for h in hidden:
        layers += [torch.nn.Linear(d, h), act()]
        d = h
    layers.append(torch.nn.Linear(d, out_dim))
    return torch.nn.Sequential(*layers)


# --------------------------------------------------------------------------------------
# Masked-categorical policy mixin (for skrl PPO)
# --------------------------------------------------------------------------------------
try:
    from skrl.models.torch import CategoricalMixin

    class MaskedCategoricalMixin(CategoricalMixin):
        """``CategoricalMixin`` that only ever samples from a fixed legal action set.

        Register the legal-action mask with :meth:`set_action_mask` before the first
        forward pass. The mask is applied to the logits both when sampling actions and
        when the PPO update recomputes their log-probability, and the entropy is
        computed with the masked-safe formula (``p * log p`` is zeroed where ``p == 0``
        rather than evaluating ``0 * -inf``).
        """

        def set_action_mask(self, mask: torch.Tensor) -> None:
            self.register_buffer("_action_mask", mask.to(dtype=torch.bool))

        def act(self, inputs: dict[str, Any], *, role: str = "") -> tuple[torch.Tensor, dict[str, Any]]:
            from torch.distributions import Categorical

            net_output, outputs = self.compute(inputs, role)
            net_output = apply_logit_mask(net_output, self._action_mask)
            self._c_distribution = Categorical(logits=net_output)
            actions = self._c_distribution.sample()
            log_prob = self._c_distribution.log_prob(inputs.get("taken_actions", actions).view(-1))
            outputs["log_prob"] = log_prob.unsqueeze(-1)
            outputs["net_output"] = net_output
            return actions.unsqueeze(-1), outputs

        # get_entropy is inherited: the finite mask fill keeps torch's Categorical.entropy
        # (which clamps logits and computes logits * probs) NaN-free for the masked terms.

except ImportError:  # skrl not installed — non-env helpers still import fine
    MaskedCategoricalMixin = None  # type: ignore[assignment, misc]


# --------------------------------------------------------------------------------------
# Checkpoint sidecar
# --------------------------------------------------------------------------------------
def write_run_meta(run_dir: str, algo: str, obs_dim: int, n_actions: int, **extra: Any) -> str:
    """Write ``<run_dir>/run_meta.json`` describing the run for the evaluator / exporter.

    skrl owns the ``.pt`` checkpoint format, so run-level facts the greedy evaluator and
    the ONNX exporter need (algorithm, obs/action dims, whether an action mask applies)
    live in this sidecar instead of the checkpoint.
    """
    os.makedirs(run_dir, exist_ok=True)
    meta = {
        "schema": CHECKPOINT_SCHEMA_VERSION,
        "algo": algo,
        "obs_dim": int(obs_dim),
        "n_actions": int(n_actions),
        **extra,
    }
    path = os.path.join(run_dir, "run_meta.json")
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    return path


# --------------------------------------------------------------------------------------
# Weights & Biases
# --------------------------------------------------------------------------------------
def _bridge_skrl_writer_to_wandb(wandb_module: Any) -> None:
    """Forward skrl's tracking scalars to ``wandb.log``.

    skrl (>= 2.1) ships its **own** ``SummaryWriter`` (``skrl.utils.tensorboard``) that
    writes tfevents directly via ``tensorboard.summary.writer`` -- it is not
    ``torch.utils.tensorboard`` or ``tensorboardX``, so ``wandb.init(sync_tensorboard=True)``
    never sees it and the W&B run ends up with only the config and no metric history.
    This wraps skrl's ``add_scalar`` so every scalar it records (losses, rewards,
    episode stats -- whatever the agent tracks) is also sent to the active W&B run.
    Idempotent; a no-op if skrl is not importable (the SAC-D trainer, which logs
    explicitly).
    """
    try:
        from skrl.utils.tensorboard import SummaryWriter as _SkrlWriter
    except Exception:  # noqa: BLE001 -- skrl absent / API moved: nothing to bridge
        return
    if getattr(_SkrlWriter, "_wandb_bridged", False):
        return
    _orig = _SkrlWriter.add_scalar

    def add_scalar(self: Any, *, tag: str, value: float, timestep: int) -> None:
        _orig(self, tag=tag, value=value, timestep=timestep)
        with contextlib.suppress(Exception):  # never let logging break training
            wandb_module.log({tag: value}, step=int(timestep))

    _SkrlWriter.add_scalar = add_scalar  # type: ignore[method-assign]
    _SkrlWriter._wandb_bridged = True


def maybe_init_wandb(
    args: argparse.Namespace,
    experiment_name: str,
    algo: str,
    config: dict[str, Any] | None = None,
):
    """Start a Weights & Biases run when ``--wandb`` was passed, else return ``None``.

    Bridges skrl's custom TensorBoard writer to ``wandb.log`` (see
    :func:`_bridge_skrl_writer_to_wandb`) so the DQN / DDQN / PPO / masked-PPO runs report
    their loss / reward / episode metrics; the single-file SAC-D trainer additionally logs
    its metrics with :func:`wandb.log` directly. The caller must invoke ``.finish()`` on
    the returned handle at the end of the run.

    Args:
        args: Parsed CLI namespace carrying the ``--wandb*`` flags from :func:`add_common_cli`.
        experiment_name: Run-folder name; the default W&B run name.
        algo: Algorithm tag; the default W&B group, also recorded in the run config.
        config: Extra key/value pairs to record in the W&B run config.

    Returns:
        The ``wandb`` run handle, or ``None`` when logging is disabled or ``wandb`` is not
        installed.
    """
    if not getattr(args, "wandb", False):
        return None
    try:
        import wandb
    except ImportError:
        print("[binary_common] --wandb set but 'wandb' is not installed; continuing without it", flush=True)
        return None
    run = wandb.init(
        project=getattr(args, "wandb_project", "") or "discrete-RL",
        group=getattr(args, "wandb_group", "") or algo,
        name=getattr(args, "wandb_name", "") or experiment_name,
        config={**vars(args), "algo": algo, **(config or {})},
    )
    _bridge_skrl_writer_to_wandb(wandb)
    print(f"[binary_common] W&B: project={run.project} name={run.name} id={run.id}", flush=True)
    return run


# --------------------------------------------------------------------------------------
# n-step returns (for the off-policy trainers)
# --------------------------------------------------------------------------------------
def nstep_return(
    rew_win: torch.Tensor,
    term_win: torch.Tensor,
    trunc_win: torch.Tensor,
    gamma: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Uncorrected n-step return from a stacked ``n``-step window.

    All three inputs are shape ``[n, N]`` (window step first): ``rew_win`` the rewards,
    ``term_win`` / ``trunc_win`` the per-step termination / truncation masks (``{0, 1}``).
    Step 0 is the transition's own step. The window is cut at the first step that ends the
    episode (termination *or* truncation).

    Returns ``(ret, steps, bootstrap)``, each ``[N]``:

    - ``ret``: ``sum_{k < K} gamma**k * r_k`` where ``K`` is the number of steps summed.
    - ``steps``: ``K`` (long) — the bootstrap value is discounted by ``gamma**K``.
    - ``bootstrap``: ``0.0`` if a real termination cut the window short, else ``1.0``
      (a full window or a truncation still bootstraps off the value of the state ``K``
      steps ahead — the caller supplies the pre-reset observation for a truncation).
    """
    n, num = rew_win.shape
    ret = torch.zeros(num, device=rew_win.device)
    steps = torch.zeros(num, dtype=torch.long, device=rew_win.device)
    bootstrap = torch.ones(num, device=rew_win.device)
    disc = torch.ones(num, device=rew_win.device)
    ended = torch.zeros(num, dtype=torch.bool, device=rew_win.device)
    for k in range(n):
        add = ~ended
        ret = ret + torch.where(add, disc * rew_win[k], torch.zeros_like(ret))
        steps = steps + add.long()
        term_k = term_win[k].bool()
        end_k = (term_k | trunc_win[k].bool()) & add
        bootstrap = torch.where(end_k & term_k, torch.zeros_like(bootstrap), bootstrap)
        ended = ended | end_k
        disc = disc * gamma
    return ret, steps, bootstrap


def mask_meta(min_stance: int, whitelist: list[int], n_bits: int = 6) -> dict[str, Any]:
    """The action-mask fields to fold into :func:`write_run_meta` for a masked run."""
    mask = legal_action_mask(min_stance, tuple(whitelist), n_bits)
    return {
        "action_mask": {
            "min_stance_legs": int(min_stance),
            "whitelist": [int(a) for a in whitelist],
            "legal_actions": torch.nonzero(mask, as_tuple=False).flatten().tolist(),
        }
    }
