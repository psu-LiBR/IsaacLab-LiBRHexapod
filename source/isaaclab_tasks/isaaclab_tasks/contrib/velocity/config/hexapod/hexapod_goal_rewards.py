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

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def ang_vel_z_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis (yaw) base angular velocity using an L2 squared kernel.

    Mirrors the generic ``mdp.ang_vel_xy_l2`` (roll/pitch) penalty, but for yaw. The goal task
    drops ``track_ang_vel_z_exp`` since there is no yaw command to track, which otherwise leaves
    yaw rate completely unpenalized -- nothing then discourages the robot from spinning in place
    en route to the goal.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_ang_vel_b.torch[:, 2])


def feet_air_time_ungated(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
) -> torch.Tensor:
    """Reward long steps taken by the feet using an L2 kernel, with no command gate.

    Identical to the generic ``mdp.feet_air_time`` except it drops the "zero reward when the
    command is near zero" gate. That gate assumes the command's first two components are a
    velocity (m/s) -- appropriate for the base-velocity task, where it is meaningless to reward
    stepping while the robot is commanded to stand still. The goal task has no equivalent
    "stand still" command state: its command's first two components are a position error (m)
    toward a fixed goal, and ``reach_goal`` already ends the episode the moment the robot
    arrives -- so gating on "close to the goal" only zeroes the stepping reward during the
    final approach, discouraging the steps needed to actually close the distance.

    Args:
        env: The RL environment instance.
        sensor_cfg: Scene entity config pointing at the contact-force sensor.
            Set ``body_names`` to the six leg bodies.
        threshold: Air-time threshold [s] above which a step is rewarded.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    return torch.sum((last_air_time - threshold) * first_contact, dim=1)


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
