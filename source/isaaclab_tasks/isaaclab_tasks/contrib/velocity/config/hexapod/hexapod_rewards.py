# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom reward terms for the LiBR Hexapod."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.assets import RigidObject


def track_ang_vel_z_exp_deadzone(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    deadzone: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward yaw-rate tracking with a dead zone around the commanded value.

    The standard ``track_ang_vel_z_exp`` penalises any instantaneous yaw-rate
    error, including the oscillatory body rotation produced by the hexapod's
    spine undulation (FrontLink/BackLink sin wave).  That oscillation has zero
    net drift over a full gait cycle but is penalised at every step, discouraging
    the body undulation the gait relies on.

    This version applies a dead zone: yaw-rate errors smaller than ``deadzone``
    (rad/s) are treated as zero error and receive full reward.  Errors beyond the
    dead zone are penalised with the same exponential kernel as the original.

    Set ``deadzone`` to cover the peak yaw rate produced by body undulation
    (measure from ``playReal.py`` yaw_rate printout; typically 0.1–0.4 rad/s).

    Args:
        env: The RL environment instance.
        std: Standard deviation of the exponential kernel (same as original term).
        command_name: Name of the velocity command group.
        deadzone: Yaw-rate error magnitude (rad/s) below which no penalty is applied.
        asset_cfg: Scene entity config for the robot rigid body.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    commanded_yaw = env.command_manager.get_command(command_name)[:, 2]
    measured_yaw = asset.data.root_ang_vel_b[:, 2]

    error = measured_yaw - commanded_yaw
    # Clip error to zero inside the dead zone; penalise only the excess beyond it
    error_outside_dz = torch.sign(error) * torch.clamp(torch.abs(error) - deadzone, min=0.0)
    return torch.exp(-torch.square(error_outside_dz) / std**2)


def track_ang_vel_z_exp_moving_avg(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    window_steps: int = 50,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward yaw-rate tracking using a fixed-window moving average.

    Penalises the mean yaw rate over the last ``window_steps`` control steps rather
    than the instantaneous value.  Sinusoidal body undulation averages to zero over
    a gait cycle and receives full reward; sustained turning accumulates and is penalised.

    Note: ``window_steps`` encodes an assumed timescale.  Prefer
    ``track_ang_vel_z_exp_episode_avg`` if you want a cycle-time-free alternative.

    Args:
        env: The RL environment instance.
        std: Standard deviation of the exponential kernel.
        command_name: Name of the velocity command group.
        window_steps: Control steps to average over (~50 steps ≈ 1 s at 0.02 s dt).
        asset_cfg: Scene entity config for the robot rigid body.
    """
    asset: RigidObject = env.scene[asset_cfg.name]

    if not hasattr(env, "_hexapod_yaw_buf"):
        env._hexapod_yaw_buf = torch.zeros(env.num_envs, window_steps, device=env.device)
        env._hexapod_yaw_ptr: int = 0

    just_reset = env.episode_length_buf == 1
    if just_reset.any():
        env._hexapod_yaw_buf[just_reset] = 0.0

    slot = env._hexapod_yaw_ptr % window_steps
    env._hexapod_yaw_buf[:, slot] = asset.data.root_ang_vel_b[:, 2]
    env._hexapod_yaw_ptr += 1

    avg_yaw = env._hexapod_yaw_buf.mean(dim=1)
    commanded = env.command_manager.get_command(command_name)[:, 2]
    return torch.exp(-torch.square(avg_yaw - commanded) / std**2)


def track_ang_vel_z_exp_episode_avg(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward yaw-rate tracking using the cumulative average over the entire episode.

    No gait-cycle timescale is hardcoded.  A cumulative sum of yaw rate is divided
    by ``episode_length_buf`` (steps elapsed this episode) to give the true episode
    mean.  Sinusoidal body undulation with zero net drift converges to zero mean as
    the episode progresses and receives full reward.  Any sustained turning shifts
    the mean away from the command and is penalised.

    The cumulative sum is stored per-env on the env object and reset to zero
    whenever an episode ends, so drift never carries across episodes.

    Args:
        env: The RL environment instance.
        std: Standard deviation of the exponential kernel (same scale as original).
        command_name: Name of the velocity command group.
        asset_cfg: Scene entity config for the robot rigid body.
    """
    asset: RigidObject = env.scene[asset_cfg.name]

    # Lazy-init per-env cumulative yaw-rate sum
    if not hasattr(env, "_hexapod_yaw_cumsum"):
        env._hexapod_yaw_cumsum = torch.zeros(env.num_envs, device=env.device)

    # Reset cumulative sum for envs that just started a new episode
    just_reset = env.episode_length_buf == 1
    if just_reset.any():
        env._hexapod_yaw_cumsum[just_reset] = 0.0

    # Accumulate current yaw rate
    env._hexapod_yaw_cumsum += asset.data.root_ang_vel_b[:, 2]

    # Episode-mean yaw rate: cumsum / steps_elapsed (clamped to avoid div-by-zero at step 0)
    steps = env.episode_length_buf.float().clamp(min=1.0)
    avg_yaw = env._hexapod_yaw_cumsum / steps

    commanded = env.command_manager.get_command(command_name)[:, 2]
    return torch.exp(-torch.square(avg_yaw - commanded) / std**2)


def track_ang_vel_z_exp_ema(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    alpha: float = 0.97,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward yaw-rate tracking using an exponential moving average (EMA).

    Avoids two problems with the episode-average approach:

    * **Vanishing gradient**: cumsum/N gives each action a 1/N contribution at step N.
      EMA gives every step a constant (1-alpha) contribution -- the gradient signal
      stays strong throughout training.
    * **Non-Markovian critic**: cumsum/N depends on the entire episode history, which
      the critic cannot observe.  EMA state changes smoothly and is predictable from
      one step to the next.

    No explicit gait-cycle length is encoded.  ``alpha`` controls the decay timescale:
    effective window ≈ 1/(1-alpha) steps.  At 0.02 s control dt:

    * alpha=0.95 → ~20 steps (~0.4 s)
    * alpha=0.97 → ~33 steps (~0.66 s)
    * alpha=0.99 → ~100 steps (~2 s)

    Sinusoidal body undulation (equal +/- halves) drives the EMA toward zero because
    each new sample partially cancels the previous.  Sustained turning holds the EMA
    away from zero and is penalised.  The EMA is stored per-env and reset to zero at
    each episode boundary so old-episode drift never carries over.

    Args:
        env: The RL environment instance.
        std: Standard deviation of the exponential kernel (same scale as original).
        command_name: Name of the velocity command group.
        alpha: EMA decay factor in (0, 1).  Higher = longer memory, slower response.
        asset_cfg: Scene entity config for the robot rigid body.
    """
    asset: RigidObject = env.scene[asset_cfg.name]

    # Lazy-init per-env EMA state
    if not hasattr(env, "_hexapod_yaw_ema"):
        env._hexapod_yaw_ema = torch.zeros(env.num_envs, device=env.device)

    # Clear EMA for envs that just started a new episode
    just_reset = env.episode_length_buf == 1
    if just_reset.any():
        env._hexapod_yaw_ema[just_reset] = 0.0

    # Update EMA: constant gradient contribution (1-alpha) per step regardless of episode length
    current_yaw = asset.data.root_ang_vel_b[:, 2]
    env._hexapod_yaw_ema = alpha * env._hexapod_yaw_ema + (1.0 - alpha) * current_yaw

    commanded = env.command_manager.get_command(command_name)[:, 2]
    error = torch.square(env._hexapod_yaw_ema - commanded)
    return torch.exp(-error / std**2)


def feet_air_time_per_leg(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    target_ratio: float = 1.0,
) -> torch.Tensor:
    """Reward each leg for air time proportional to its own previous contact time.

    Unlike ``feet_air_time`` which compares every leg to a single fixed threshold,
    this compares each leg's ``last_air_time`` to that same leg's ``last_contact_time``
    scaled by ``target_ratio``.  This makes the reward duty-cycle aware: a leg that
    spends longer in contact (e.g. a stance leg in a slow gait) is expected to spend
    proportionally longer in the air before being rewarded.

    At the moment each leg first makes contact the contribution is:

        ``(last_air_time  -  target_ratio * last_contact_time) * first_contact``

    Positive values (air time exceeded the target fraction of contact time) are
    rewarded; negative values penalise legs that return to ground too quickly.
    Legs whose ``last_contact_time`` is still zero (never lifted before) are
    excluded so the very first touch-down does not distort the reward.

    Args:
        env: The RL environment instance.
        command_name: Name of the velocity command used to gate the reward
            (reward is zeroed when the commanded speed is near zero).
        sensor_cfg: Scene entity config pointing at the contact-force sensor.
            Set ``body_names`` to the six leg bodies.
        target_ratio: Desired ``air_time / contact_time`` ratio per leg.
            1.0 targets a 50 % duty cycle (equal time in air and on ground).
            Values > 1 encourage longer air phases; < 1 encourage shorter ones.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # [num_envs, num_legs] — True only on the step each leg first touches down
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]

    # Duration of the air phase that just ended (updated the moment contact is made)
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]

    # Duration of the most recent completed contact phase for this leg
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]

    # Only reward legs that have completed at least one contact phase; avoids
    # dividing-by-zero and distortion on the very first touch-down of an episode.
    has_prior_contact = last_contact_time > 0.0

    # Per-leg reward: positive when air time >= target_ratio * contact_time
    per_leg = (last_air_time - target_ratio * last_contact_time) * first_contact * has_prior_contact

    reward = torch.sum(per_leg, dim=1)

    # Zero reward when velocity command magnitude is negligible (robot should stand still)
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1

    return reward
