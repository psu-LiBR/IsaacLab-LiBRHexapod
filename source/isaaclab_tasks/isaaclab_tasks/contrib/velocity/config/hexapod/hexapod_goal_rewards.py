# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward and termination terms for the LiBR Hexapod goal-reaching task.

Task: reach a fixed 5m-forward goal as fast as possible.  Combines progress shaping
(velocity component toward the goal) with a sparse reach bonus and a per-step time
penalty.  The progress term gives a strong, position-independent gradient and avoids
the 1/r^2 singularity at the goal.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def progress_to_goal(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """(prev_distance - current_distance) / dt -- velocity component toward goal in m/s.

    The episode integral telescopes to the total displacement toward the goal, so
    the reward weight directly controls "reward per meter advanced".  Strong gradient
    everywhere along the approach, unlike 1/r^2 which vanishes when far away.
    """
    command = env.command_manager.get_command(command_name)
    curr_dist = torch.norm(command[:, :3], dim=1)

    if not hasattr(env, "_goal_prev_dist") or env._goal_prev_dist.shape != curr_dist.shape:
        env._goal_prev_dist = curr_dist.clone()

    # On episode reset, seed prev_dist with the fresh distance so the first-step delta is 0
    just_reset = env.episode_length_buf <= 1
    if just_reset.any():
        env._goal_prev_dist[just_reset] = curr_dist[just_reset]

    progress = (env._goal_prev_dist - curr_dist) / env.step_dt
    env._goal_prev_dist = curr_dist.clone()
    return progress


def constant_per_step(env: ManagerBasedRLEnv) -> torch.Tensor:
    """1.0 every env every step -- pair with a negative weight as a time penalty."""
    return torch.ones(env.num_envs, device=env.device)


def termination_signal(env: ManagerBasedRLEnv, termination_name: str) -> torch.Tensor:
    """Return a one-step float signal for a named termination condition."""
    return env.termination_manager.get_term(termination_name).float()


def time_decayed_termination_signal(
    env: ManagerBasedRLEnv,
    termination_name: str,
    episode_length_s: float,
    min_fraction: float = 0.5,
) -> torch.Tensor:
    """Terminal signal scaled down linearly as episode time elapses.

    Fires ``decay`` on the step ``termination_name`` is True, 0 otherwise, where ``decay``
    goes from 1.0 immediately after reset to ``min_fraction`` at ``episode_length_s``. Pair
    with a positive weight so a fast success is worth more than a slow one, concentrating
    the speed pressure into the single high-salience terminal transition instead of relying
    solely on the per-step time penalty.
    """
    fired = env.termination_manager.get_term(termination_name).float()
    elapsed_s = env.episode_length_buf.float() * env.step_dt
    frac_elapsed = torch.clamp(elapsed_s / episode_length_s, max=1.0)
    decay = 1.0 - (1.0 - min_fraction) * frac_elapsed
    return fired * decay


def reached_goal_done(env: ManagerBasedRLEnv, command_name: str, radius: float) -> torch.Tensor:
    """Termination: True when robot is inside `radius` of the goal."""
    command = env.command_manager.get_command(command_name)
    distance = torch.norm(command[:, :3], dim=1)
    return distance < radius
