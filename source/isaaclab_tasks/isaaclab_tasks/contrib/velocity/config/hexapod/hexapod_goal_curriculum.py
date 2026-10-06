# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Success-based goal-distance curriculum for the LiBR hexapod."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def goal_distance_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_name: str,
    distances: tuple[float, ...],
    success_threshold: float,
    window_sizes: tuple[int, ...],
    demotion_threshold: float | None = None,
) -> dict[str, float]:
    """Advance (or demote) the fixed goal distance after a non-overlapping episode window.

    ``window_sizes[stage]`` sets how many completed episodes are pooled before the running
    success rate at the current stage is evaluated -- easier/cheaper stages can use a smaller
    window than harder ones without changing the statistical confidence of the decision.

    If ``demotion_threshold`` is given and the windowed success rate falls below it, the stage
    steps back down one level instead of holding, so a policy that regresses after advancing
    (e.g. catastrophic forgetting of the shorter-distance gait) gets pushed back onto easier
    data rather than being stuck grinding at a stage it can no longer solve.
    """
    if not distances:
        raise ValueError("distances must contain at least one curriculum stage")
    if not 0.0 <= success_threshold <= 1.0:
        raise ValueError("success_threshold must be in [0, 1]")
    if len(window_sizes) != len(distances):
        raise ValueError("window_sizes must have the same length as distances")
    if any(size <= 0 for size in window_sizes):
        raise ValueError("window_sizes must all be positive")
    if demotion_threshold is not None and not 0.0 <= demotion_threshold <= success_threshold:
        raise ValueError("demotion_threshold must be in [0, success_threshold]")

    if not hasattr(env, "_goal_curriculum_stage"):
        env._goal_curriculum_stage = 0
        env._goal_curriculum_episodes = 0
        env._goal_curriculum_successes = 0
        env._goal_curriculum_falls = 0
        env._goal_curriculum_last_rate = 0.0

    # Initial scene setup also calls curriculum computation. Ignore environments
    # that have not executed an episode yet.
    completed = env.episode_length_buf[env_ids] > 0
    if completed.any():
        success = env.termination_manager.get_term("reach_goal")[env_ids][completed]
        fall = env.termination_manager.get_term("base_contact")[env_ids][completed]
        env._goal_curriculum_episodes += int(completed.sum().item())
        env._goal_curriculum_successes += int(success.sum().item())
        env._goal_curriculum_falls += int(fall.sum().item())

    report_episodes = env._goal_curriculum_episodes
    report_successes = env._goal_curriculum_successes
    report_falls = env._goal_curriculum_falls

    window_size = window_sizes[env._goal_curriculum_stage]
    if env._goal_curriculum_episodes >= window_size:
        success_rate = env._goal_curriculum_successes / env._goal_curriculum_episodes
        env._goal_curriculum_last_rate = success_rate
        if success_rate >= success_threshold and env._goal_curriculum_stage < len(distances) - 1:
            env._goal_curriculum_stage += 1
        elif demotion_threshold is not None and success_rate < demotion_threshold and env._goal_curriculum_stage > 0:
            env._goal_curriculum_stage -= 1
        env._goal_curriculum_episodes = 0
        env._goal_curriculum_successes = 0
        env._goal_curriculum_falls = 0

    distance = float(distances[env._goal_curriculum_stage])
    command_term = env.command_manager.get_term(command_name)
    command_term.cfg.ranges.pos_x = (distance, distance)

    current_rate = report_successes / report_episodes if report_episodes > 0 else env._goal_curriculum_last_rate
    return {
        "distance": distance,
        "success_rate": float(current_rate),
        "episodes": float(report_episodes),
        "successes": float(report_successes),
        "falls": float(report_falls),
    }
