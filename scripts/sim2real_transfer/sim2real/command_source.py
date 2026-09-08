# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Velocity/goal command source, kept as a swappable interface.

v1 only ships constant sources (hardcoded values from `deployment.yaml`). A future
`WirelessCommandSource` (SSH- or joystick-driven) implements the same interface;
`control_loop.py` only ever calls `get_command()`, so nothing else needs to change
when that lands.

For the velocity profile, `get_command()` returns the obs-ready [vx, vy, yaw_rate]
vector directly. For the goal profile, it returns the raw fixed target
[x, y, z, heading] in the localizer's world frame -- `localization.py` is what
turns that into the actual obs-ready relative pose_command each step.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .deployment_config import CommandsCfg


class CommandSource(ABC):
    @abstractmethod
    def get_command(self) -> np.ndarray: ...


class ConstantVelocityCommand(CommandSource):
    def __init__(self, cfg: CommandsCfg):
        self._command = np.array([cfg.velocity["vx"], cfg.velocity["vy"], cfg.velocity["yaw_rate"]])

    def get_command(self) -> np.ndarray:
        return self._command.copy()


class ConstantGoalCommand(CommandSource):
    def __init__(self, cfg: CommandsCfg):
        self._command = np.array([cfg.goal["x"], cfg.goal["y"], cfg.goal["z"], cfg.goal["heading"]])

    def get_command(self) -> np.ndarray:
        return self._command.copy()


def make_command_source(profile_name: str, cfg: CommandsCfg) -> CommandSource:
    if profile_name == "velocity":
        return ConstantVelocityCommand(cfg)
    if profile_name in ("goal", "binary"):
        # The binary-contact task is goal-reaching with a swapped action space, so it
        # takes the same 4-dim [x, y, z, heading] world-frame goal.
        return ConstantGoalCommand(cfg)
    raise ValueError(f"unknown profile '{profile_name}'")
