# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation schemas matching the actor (`PolicyCfg`) obs groups the RL policies
were trained on.

Velocity profile mirrors `hexapod_obs_cfg.py::HexapodFlatObservationsCfg.PolicyCfg`.
Goal profile mirrors `hexapod_goal_obs_cfg.py::HexapodGoalObservationsCfg.PolicyCfg`.
Both concatenate terms in this fixed order: gyro(3), gravity(3), command(3 or 4),
joint_pos_rel(8), joint_vel(8), last_action(8). Noise is off at deploy time, matching
`HexapodFlatEnvCfg_PLAY`/`HexapodGoalEnvCfg_PLAY`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

NUM_JOINTS = 8


@dataclass(frozen=True)
class ProfileSpec:
    name: str
    command_dim: int
    command_field_names: tuple[str, ...]

    @property
    def obs_dim(self) -> int:
        return 3 + 3 + self.command_dim + NUM_JOINTS + NUM_JOINTS + NUM_JOINTS

    @property
    def action_dim(self) -> int:
        return NUM_JOINTS


PROFILES: dict[str, ProfileSpec] = {
    "velocity": ProfileSpec("velocity", command_dim=3, command_field_names=("vx", "vy", "yaw_rate")),
    "goal": ProfileSpec("goal", command_dim=4, command_field_names=("x", "y", "z", "heading")),
}


class ObsBuilder:
    """Concatenates sensor/command/state inputs into the obs vector in training order."""

    def __init__(self, spec: ProfileSpec):
        self.spec = spec

    def build(
        self,
        gyro_xyz: np.ndarray,
        gravity_body: np.ndarray,
        joint_pos_sim: np.ndarray,
        joint_vel_sim: np.ndarray,
        last_action_sim: np.ndarray,
        q_default_sim: np.ndarray,
        command: np.ndarray,
    ) -> np.ndarray:
        gyro_xyz = np.asarray(gyro_xyz, dtype=np.float64)
        gravity_body = np.asarray(gravity_body, dtype=np.float64)
        joint_pos_sim = np.asarray(joint_pos_sim, dtype=np.float64)
        joint_vel_sim = np.asarray(joint_vel_sim, dtype=np.float64)
        last_action_sim = np.asarray(last_action_sim, dtype=np.float64)
        q_default_sim = np.asarray(q_default_sim, dtype=np.float64)
        command = np.asarray(command, dtype=np.float64)

        if gyro_xyz.shape != (3,):
            raise ValueError(f"gyro_xyz must be shape (3,), got {gyro_xyz.shape}")
        if gravity_body.shape != (3,):
            raise ValueError(f"gravity_body must be shape (3,), got {gravity_body.shape}")
        for name, arr in (
            ("joint_pos_sim", joint_pos_sim),
            ("joint_vel_sim", joint_vel_sim),
            ("last_action_sim", last_action_sim),
            ("q_default_sim", q_default_sim),
        ):
            if arr.shape != (NUM_JOINTS,):
                raise ValueError(f"{name} must be shape ({NUM_JOINTS},), got {arr.shape}")
        if command.shape != (self.spec.command_dim,):
            raise ValueError(
                f"command must be shape ({self.spec.command_dim},) for profile '{self.spec.name}', got {command.shape}"
            )

        joint_pos_rel = joint_pos_sim - q_default_sim
        obs = np.concatenate([gyro_xyz, gravity_body, command, joint_pos_rel, joint_vel_sim, last_action_sim])
        assert obs.shape == (self.spec.obs_dim,)
        return obs.astype(np.float32)


class VelocityObsBuilder(ObsBuilder):
    def __init__(self):
        super().__init__(PROFILES["velocity"])


class GoalObsBuilder(ObsBuilder):
    def __init__(self):
        super().__init__(PROFILES["goal"])


_BUILDERS: dict[str, type[ObsBuilder]] = {
    "velocity": VelocityObsBuilder,
    "goal": GoalObsBuilder,
}


def make_obs_builder(profile_name: str) -> ObsBuilder:
    if profile_name not in _BUILDERS:
        raise ValueError(f"unknown profile '{profile_name}', expected one of {list(_BUILDERS)}")
    return _BUILDERS[profile_name]()
