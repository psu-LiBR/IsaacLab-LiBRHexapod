# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""September 11 candidates on the upstream DCMotor and canonical wave definitions.

This module is opt-in. Existing tasks retain their original configuration unless
the generic overnight entry point explicitly supplies a candidate and spine.
Reward weights are rates integrated by RewardManager exactly once. Event and
distance metrics divide by dt so their effective magnitude is independent of dt.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

LEG_NAMES = ["FrontRight", "FrontLeft", "MiddleRight", "MiddleLeft", "BackRight", "BackLeft"]
COMMON_WEIGHTS = {"progress": 5.0, "reach": 2.0, "fall": -6.0, "time": -0.01}
RAW_WEIGHTS = {
    "contacts": -1.0,
    "limits": -1.0,
    "slide": -0.01,
    "tilt": -1.0,
    "vertical": -0.1,
    "lateral": -0.05,
    "roll_pitch": -0.005,
    "yaw": -0.0001,
    "torque": -0.00003,
    "acc": -2.5e-9,
    "switch": -0.0005,
}
BOUNDED_WEIGHTS = {
    "tilt": -0.080,
    "contacts": -0.060,
    "limits": -0.020,
    "vertical": -0.020,
    "lateral": -0.010,
    "roll_pitch": -0.005,
    "yaw": -0.001,
    "torque": -0.002,
    "switch": -0.002,
}


def tensor(value):
    """Access the Torch view of an Isaac Lab 3 ProxyArray or a Torch tensor."""
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value)
    if hasattr(value, "torch"):
        return value.torch
    import warp as wp

    return wp.to_torch(value)


def reference_profile(spine: str) -> dict:
    """Read canonical Fourier coefficients and the upstream extquad leg schedule."""
    from isaaclab_tasks.contrib.velocity.config.hexapod import hexapod_binary_env_cfg as canonical

    if spine not in ("wave1", "wave2"):
        raise ValueError(spine)
    prefix = "TRIPOD_" if spine == "wave2" else ""
    sin_coef = getattr(canonical, prefix + "SPINE_SIN_COEF")
    cos_coef = getattr(canonical, prefix + "SPINE_COS_COEF")
    offset = getattr(canonical, prefix + "SPINE_OFFSET")
    path = Path(__file__).with_name("tripod_bit_demos.npz")
    with np.load(path) as data:
        bits = data["bits"].astype(np.int64)
        actions = data["action_idx"].astype(np.int64)
    if len(bits) == 51:
        np.testing.assert_array_equal(bits[0], bits[-1])
        bits, actions = bits[:50], actions[:50]
    assert bits.shape == (50, 6)
    np.testing.assert_array_equal(actions, (bits * (1 << np.arange(6))).sum(axis=1))
    assert sorted(np.unique(actions).tolist()) == [25, 38]
    theta = 2 * np.pi * np.arange(50) / 50
    return {
        "spine": spine,
        "period_s": canonical.GAIT_PERIOD_S,
        "period_steps": 50,
        "sin_coef": sin_coef,
        "cos_coef": cos_coef,
        "offset": offset,
        "bits": bits.tolist(),
        "actions": actions.tolist(),
        "spine_targets": {
            name: (offset[name] + sin_coef[name][0] * np.sin(theta) + cos_coef[name][0] * np.cos(theta)).tolist()
            for name in sin_coef
        },
        "switch_rows": (np.nonzero(np.any(bits[1:] != bits[:-1], axis=1))[0] + 1).tolist(),
        "schedule_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "schedule": path.name,
    }


def _indices(env):
    if not hasattr(env, "_overnight_indices"):
        robot = env.scene["robot"]
        sensor = env.scene.sensors["contact_forces"]
        env._overnight_indices = {
            "legs": robot.find_joints([n + "_Joint" for n in LEG_NAMES], preserve_order=True)[0],
            "feet": robot.find_bodies(LEG_NAMES, preserve_order=True)[0],
            "contact_feet": sensor.find_bodies(LEG_NAMES, preserve_order=True)[0],
            "torso": sensor.find_bodies(["CenterLink", "BackLink", "FrontLink"])[0],
            "center": sensor.find_bodies(["CenterLink"])[0],
        }
        if len(env._overnight_indices["legs"]) != 6 or len(env._overnight_indices["torso"]) != 3:
            raise ValueError("Unexpected robot joint/body mapping")
    return env._overnight_indices


def fresh_failure(env):
    """Original center-body contact failure, evaluated from current force history [N]."""
    ids = _indices(env)
    forces = tensor(env.scene.sensors["contact_forces"].data.net_forces_w_history)
    return (forces[:, :, ids["center"]].norm(dim=-1).amax(dim=1) > 1.0).any(dim=1)


def motor_effort_limits(env):
    """Return motor caps [N m], not the deliberately unbounded PhysX solver caps."""
    robot = env.scene["robot"]
    limits = torch.zeros_like(tensor(robot.data.joint_effort_limits))
    for actuator in robot.actuators.values():
        limits[:, actuator.joint_indices] = tensor(actuator.effort_limit)
    return limits[:, _indices(env)["legs"]]


def fresh_reached_goal(env, radius: float = 0.2):
    """Use fresh root/goal positions [m]; simultaneous failure never counts as success."""
    command = env.command_manager.get_term("pose_command")
    goal = command.pos_command_w
    # Refresh the existing observation field without advancing time or resampling.
    # The manager captures final_obs before its normal post-reset command update.
    command._update_command()
    root = tensor(env.scene["robot"].data.root_pos_w)
    return ((goal - root).norm(dim=1) < radius) & ~fresh_failure(env)


def reward_metrics(env, candidate: str) -> dict[str, torch.Tensor]:
    """Compute fresh same-episode metrics once per control step, before any reset."""
    key = (env.common_step_counter, candidate)
    if getattr(env, "_overnight_metric_key", None) == key:
        return env._overnight_metrics
    ids = _indices(env)
    robot = env.scene["robot"]
    data = robot.data
    root = tensor(data.root_pos_w)
    goal = env.command_manager.get_term("pose_command").pos_command_w
    distance = (goal[:, :2] - root[:, :2]).norm(dim=1)
    first = env.episode_length_buf <= 1
    if not hasattr(env, "_overnight_prev_distance"):
        env._overnight_prev_distance = distance.clone()
    delta = torch.where(first, 0.0, env._overnight_prev_distance - distance)
    env._overnight_prev_distance = distance.clone()
    tilt = torch.acos((-tensor(data.projected_gravity_b)[:, 2]).clamp(-1.0, 1.0))
    excess_tilt = (tilt - math.radians(10)).clamp(min=0)
    lin_b, lin_w, ang = tensor(data.root_lin_vel_b), tensor(data.root_lin_vel_w), tensor(data.root_ang_vel_b)
    quality = torch.exp(-(excess_tilt / math.radians(20)).square() - (lin_w[:, 2] / 0.15).square())
    force = tensor(env.scene.sensors["contact_forces"].data.net_forces_w_history).norm(dim=-1).amax(dim=1)
    contacts = (force[:, ids["torso"]] > 1.0).sum(dim=1).float()
    limits = tensor(data.soft_joint_pos_limits)
    pos = tensor(data.joint_pos)
    limit_excess = ((limits[:, :, 0] - pos).clamp(min=0) + (pos - limits[:, :, 1]).clamp(min=0)).sum(dim=1)
    tau = tensor(data.applied_torque)[:, ids["legs"]]
    effort_limits = motor_effort_limits(env)
    switch = (env.action_manager.action - env.action_manager.prev_action).square().sum(dim=1)
    switch = torch.where(first, 0.0, switch)
    failed = fresh_failure(env)
    success = fresh_reached_goal(env)
    elapsed = env.episode_length_buf.float() * env.step_dt
    decay = 1.0 - 0.5 * (elapsed / 25.0).clamp(0, 1)
    q = quality * (contacts == 0).float() if candidate == "C2" else torch.ones_like(quality)
    metrics = {
        "progress": (q * delta.clamp(min=0) - (-delta).clamp(min=0)) / env.step_dt,
        "reach": success.float() * decay * q / env.step_dt,
        "fall": failed.float() / env.step_dt,
        "time": torch.ones_like(distance),
    }
    if candidate == "A2":
        metrics.update(
            {
                "contacts": contacts,
                "limits": limit_excess,
                "slide": (
                    tensor(data.body_lin_vel_w)[:, ids["feet"], :2].norm(dim=-1) * (force[:, ids["contact_feet"]] > 1.0)
                ).sum(dim=1),
                "tilt": excess_tilt.square(),
                "vertical": lin_w[:, 2].square(),
                "lateral": lin_b[:, 1].square(),
                "roll_pitch": ang[:, :2].square().sum(dim=1),
                "yaw": ang[:, 2].square(),
                "torque": tau.square().sum(dim=1),
                "acc": tensor(data.joint_acc)[:, ids["legs"]].square().sum(dim=1),
                "switch": switch,
            }
        )
    else:
        metrics.update(
            {
                "contacts": contacts,
                "limits": (limit_excess / 0.1).clamp(0, 1),
                "tilt": (excess_tilt / math.radians(20)).square().clamp(0, 1),
                "vertical": (lin_w[:, 2] / 0.15).square().clamp(0, 1),
                "lateral": (lin_b[:, 1] / 0.1).square().clamp(0, 1),
                "roll_pitch": (ang[:, :2].square().sum(dim=1) / 4.0).clamp(0, 1),
                "yaw": (ang[:, 2] / 2.0).square().clamp(0, 1),
                "torque": (tau / effort_limits.clamp(min=1e-8)).square().clamp(0, 1).mean(dim=1),
                "switch": switch / 24.0,
            }
        )
    env._overnight_quality = quality
    env._reward_progress_gate = q
    env._overnight_delta = delta
    env._overnight_failure = failed
    env._overnight_success = success
    env._overnight_metric_key, env._overnight_metrics = key, metrics
    return metrics


def reward_component(env, component: str, candidate: str):
    """Return one metric for RewardManager weighting and dt integration."""
    return reward_metrics(env, candidate)[component]


def configure_env(cfg, candidate: str, spine: str, evaluation: bool = False, no_reset: bool = False):
    """Apply an approved candidate without changing action/observation field definitions."""
    from isaaclab.managers import RewardTermCfg, TerminationTermCfg

    if candidate not in ("J", "A2", "B2", "C2"):
        raise ValueError(candidate)
    if not evaluation and spine != "wave1":
        raise ValueError("All learned policies must use the upstream Wave 1")
    profile = reference_profile(spine)
    wave = cfg.actions.spine_wave
    wave.sin_coef, wave.cos_coef, wave.offset = profile["sin_coef"], profile["cos_coef"], profile["offset"]
    wave.period = profile["period_s"]
    weights = {name: value.weight for name, value in vars(cfg.rewards).items() if isinstance(value, RewardTermCfg)}
    if candidate != "J":
        weights = COMMON_WEIGHTS | (RAW_WEIGHTS if candidate == "A2" else BOUNDED_WEIGHTS | {"contacts": -1.0})
        # Opt-in candidates replace the inherited terms; J retains upstream semantics.
        for name, value in vars(cfg.rewards).items():
            if isinstance(value, RewardTermCfg):
                setattr(cfg.rewards, name, None)
        for name, weight in weights.items():
            setattr(
                cfg.rewards,
                name,
                RewardTermCfg(func=reward_component, weight=weight, params={"component": name, "candidate": candidate}),
            )
        cfg.terminations.reach_goal = TerminationTermCfg(func=fresh_reached_goal, params={"radius": 0.2})
    material = cfg.events.physics_material.params
    assert tuple(material["static_friction_range"]) == (0.18, 0.25)
    assert tuple(material["dynamic_friction_range"]) == (0.18, 0.25)
    # Explicitly retain the backend's configured consistency policy and report it.
    if evaluation:
        for key in ("static_friction_range", "dynamic_friction_range", "restitution_range"):
            midpoint = sum(material[key]) / 2
            material[key] = (midpoint, midpoint)
        material["num_buckets"] = 1
        cfg.commands.pose_command.resampling_time_range = (1e9, 1e9)
        cfg.curriculum.goal_distance = None
        cfg.commands.pose_command.ranges.pos_x = (2.0, 2.0)
        cfg.commands.pose_command.ranges.pos_y = (0.0, 0.0)
        cfg.commands.pose_command.ranges.heading = (0.0, 0.0)
        cfg.observations.policy.enable_corruption = False
        cfg.observations.critic.enable_corruption = False
        cfg.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        cfg.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
        if no_reset:
            cfg.episode_length_s = 1e6
            cfg.terminations.base_contact = TerminationTermCfg(func=never_done)
            cfg.terminations.reach_goal = TerminationTermCfg(func=never_done)
    return {
        "candidate": candidate,
        "profile": profile,
        "weights": weights,
        "training": not evaluation,
        "no_reset": no_reset,
        "friction": {k: v for k, v in material.items() if k != "asset_cfg"},
        "terrain_material": cfg.scene.terrain.physics_material.to_dict(),
        "progress": "upstream native" if candidate == "J" else "fresh XY distance, first step zero",
        "success": "upstream native" if candidate == "J" else "fresh 3D radius .2, failure priority",
        "actuators": {name: type(value).__name__ for name, value in cfg.scene.robot.actuators.items()},
        "upstream_commit": "929be07e10d1f8097b48f06a68b1af9e3eb554ac",
        "dt": cfg.sim.dt * cfg.decimation,
    }


def never_done(env):
    """Disable resets for a finite physical measurement; judge failures separately."""
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)


def save_runtime_audit(env, audit: dict, path: str):
    """Save actual observation shapes, actuator limits and loaded materials."""
    data = env.scene["robot"].data
    ids = _indices(env)
    limits = motor_effort_limits(env)
    audit.update(
        {
            "num_envs": env.num_envs,
            "step_dt": env.step_dt,
            "joint_names": env.scene["robot"].joint_names,
            "indices": ids,
            "effort_limit_min": limits.min().item(),
            "effort_limit_max": limits.max().item(),
            "solver_effort_limit_max": tensor(data.joint_effort_limits).max().item(),
            "effort_normalization": "actuator.effort_limit; explicit DCMotor stall/continuous cap, not solver cap",
            "observations": {k: list(v.shape) for k, v in env.single_observation_space.items()},
            "reward_terms": env.reward_manager.active_terms,
            "loaded_actuators": {name: type(value).__name__ for name, value in env.scene["robot"].actuators.items()},
        }
    )
    try:
        values = env.scene["robot"].root_view.get_material_properties()
        values = tensor(values)
        audit["loaded_material_min"] = values.reshape(-1, 3).min(dim=0).values.tolist()
        audit["loaded_material_max"] = values.reshape(-1, 3).max(dim=0).values.tolist()
    except (AttributeError, TypeError) as exc:
        audit["loaded_material_error"] = str(exc)
    if not torch.isfinite(limits).all() or not (limits > 0).all():
        raise ValueError("Invalid loaded actuator effort limits")
    if any(value != "DCMotor" for value in audit["loaded_actuators"].values()):
        raise ValueError("Expected upstream DCMotor actuators")
    Path(path).write_text(json.dumps(audit, indent=2))
