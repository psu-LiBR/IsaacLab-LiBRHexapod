# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Velocity/goal command source, kept as a swappable interface.

`control_loop.py` only ever calls `get_command()`, so a new source (SSH- or
joystick-driven, a receding goal, ...) drops in without touching the loop.

For the velocity profile, `get_command()` returns the obs-ready [vx, vy, yaw_rate]
vector directly. For the goal/binary profiles it returns a fixed target
[x, y, z, heading] in the localizer's world frame -- `localization.py` turns that
into the actual obs-ready body-frame pose_command each step.

Two goal sources:

- `ConstantGoalCommand` (`mode: fixed`, the default) -- a stationary world point.
  The body-frame pose_command shrinks toward zero as the robot approaches, so the
  policy runs its full "approach then arrive" behaviour and then has no goal left
  once it is on top of the point. Use it to walk a set distance and stop.
- `RecedingGoalCommand` (`mode: receding`) -- the goal is regenerated a fixed
  `lookahead_m` ahead of the robot's *current* dead-reckoned position every step,
  so the body-frame pose_command stays ~[lookahead, 0, 0, 0] and the policy never
  leaves its "far from the goal, keep walking" regime. Use it for continuous
  forward locomotion. Because the goal recedes with the position estimate, the
  localizer's absolute-position drift cancels out of the pose_command -- only its
  (accurate, gyro-integrated) heading estimate is actually used. See CLAUDE.md
  "Goal-Reaching System" / the sim2real notes.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import numpy as np

from .deployment_config import CommandsCfg
from .localization import DeadReckoningLocalizer


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
        self._command = np.array(
            [float(cfg.goal["x"]), float(cfg.goal["y"]), float(cfg.goal["z"]), float(cfg.goal["heading"])]
        )

    def get_command(self) -> np.ndarray:
        return self._command.copy()


class RecedingGoalCommand(CommandSource):
    """A goal pinned `lookahead_m` ahead of the robot's estimated position.

    The world goal each step is the localizer's current (x, y) plus `lookahead_m`
    along `world_heading` (config `commands.goal.heading`, default 0.0 -- i.e. hold
    the heading the robot started at). `world_heading` is the direction to walk,
    not a goal orientation: it is also returned as the command's heading field so
    the body-frame pose_command's heading term drives the policy back onto that
    line whenever the gyro-integrated heading drifts.

    Only the localizer's heading is load-bearing here; its position estimate
    cancels (goal and robot share the same drifting estimate), so the
    constant-forward-speed position assumption in `DeadReckoningLocalizer` does
    not matter for this mode.
    """

    def __init__(self, cfg: CommandsCfg, localizer: DeadReckoningLocalizer):
        self._localizer = localizer
        self._lookahead = float(cfg.goal.get("lookahead_m", 2.0))
        self._world_heading = float(cfg.goal.get("heading", 0.0))
        self._z = float(cfg.goal.get("z", 0.0))
        if self._lookahead <= 0.0:
            raise ValueError(f"commands.goal.lookahead_m must be positive, got {self._lookahead}")

    def get_command(self) -> np.ndarray:
        x = self._localizer.x + self._lookahead * math.cos(self._world_heading)
        y = self._localizer.y + self._lookahead * math.sin(self._world_heading)
        return np.array([x, y, self._z, self._world_heading])


def make_command_source(
    profile_name: str,
    cfg: CommandsCfg,
    localizer: DeadReckoningLocalizer | None = None,
) -> CommandSource:
    if profile_name == "velocity":
        return ConstantVelocityCommand(cfg)
    if profile_name in ("goal", "binary"):
        # The binary-contact task is goal-reaching with a swapped action space, so it
        # takes the same 4-dim [x, y, z, heading] world-frame goal.
        mode = str(cfg.goal.get("mode", "fixed")).lower()
        if mode == "receding":
            if localizer is None:
                raise ValueError("commands.goal.mode 'receding' requires a localizer")
            return RecedingGoalCommand(cfg, localizer)
        if mode != "fixed":
            raise ValueError(f"unknown commands.goal.mode '{mode}', expected 'fixed' or 'receding'")
        return ConstantGoalCommand(cfg)
    raise ValueError(f"unknown profile '{profile_name}'")
