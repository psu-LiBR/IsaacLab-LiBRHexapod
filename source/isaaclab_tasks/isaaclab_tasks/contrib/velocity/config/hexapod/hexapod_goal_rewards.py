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


def reached_goal_bonus(env: ManagerBasedRLEnv, command_name: str, radius: float) -> torch.Tensor:
    """Returns 1.0 on the step the robot is inside `radius` of the goal, 0 otherwise.

    Paired with the reach_goal termination, this fires exactly once per successful
    episode.  Set the reward weight to the desired bonus magnitude (e.g. 100.0).
    """
    command = env.command_manager.get_command(command_name)
    distance = torch.norm(command[:, :3], dim=1)
    return (distance < radius).float()


def constant_per_step(env: ManagerBasedRLEnv) -> torch.Tensor:
    """1.0 every env every step -- pair with a negative weight as a time penalty."""
    return torch.ones(env.num_envs, device=env.device)


def termination_signal(env: ManagerBasedRLEnv, termination_name: str) -> torch.Tensor:
    """Return a one-step float signal for a named termination condition."""
    return env.termination_manager.get_term(termination_name).float()


def reached_goal_done(env: ManagerBasedRLEnv, command_name: str, radius: float) -> torch.Tensor:
    """Termination: True when robot is inside `radius` of the goal."""
    command = env.command_manager.get_command(command_name)
    distance = torch.norm(command[:, :3], dim=1)
    return distance < radius
