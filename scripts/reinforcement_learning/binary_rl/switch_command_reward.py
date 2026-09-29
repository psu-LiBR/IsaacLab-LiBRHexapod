# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Actual contact transitions: output is events/s, integrated once by manager."""

import torch


def contact_switch_count(env, sensor_cfg, threshold=1.0, release_threshold=0.5):
    if not 0 <= release_threshold < threshold:
        raise ValueError("contact hysteresis thresholds must be ordered")
    forces = env.scene.sensors[sensor_cfg.name].data.net_forces_w
    if not isinstance(forces, torch.Tensor):
        forces = forces.torch
    magnitude = torch.linalg.vector_norm(forces[:, sensor_cfg.body_ids], dim=-1)
    first = not hasattr(env, "_contact_switch_state")
    if first:
        env._contact_switch_state = magnitude > threshold
        env._contact_switch_initialized = torch.zeros(env.num_envs, dtype=torch.bool, device=magnitude.device)
    previous = env._contact_switch_state
    state = torch.where(previous, magnitude > release_threshold, magnitude > threshold)
    reset = env.episode_length_buf <= 1
    # Seed reset environments independently: no hysteresis memory from old episodes.
    state = torch.where(reset[:, None], magnitude > threshold, state)
    count = (state != previous).sum(-1).float()
    count = torch.where(reset | ~env._contact_switch_initialized, 0.0, count)
    env._contact_switch_state = state
    env._contact_switch_initialized = torch.ones_like(reset)
    return count / env.step_dt


def action_switch_count(env):
    flips = (env.action_manager.action != env.action_manager.prev_action).sum(-1).float()
    return torch.where(env.episode_length_buf <= 1, 0.0, flips) / env.step_dt
