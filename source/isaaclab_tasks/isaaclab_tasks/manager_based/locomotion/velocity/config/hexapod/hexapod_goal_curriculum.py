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
    window_size: int,
) -> dict[str, float]:
    """Advance fixed goal distance after a successful non-overlapping episode window."""
    if not distances:
        raise ValueError("distances must contain at least one curriculum stage")
    if not 0.0 <= success_threshold <= 1.0:
        raise ValueError("success_threshold must be in [0, 1]")
    if window_size <= 0:
        raise ValueError("window_size must be positive")

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

    if env._goal_curriculum_episodes >= window_size:
        success_rate = env._goal_curriculum_successes / env._goal_curriculum_episodes
        env._goal_curriculum_last_rate = success_rate
        if success_rate >= success_threshold and env._goal_curriculum_stage < len(distances) - 1:
            env._goal_curriculum_stage += 1
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
