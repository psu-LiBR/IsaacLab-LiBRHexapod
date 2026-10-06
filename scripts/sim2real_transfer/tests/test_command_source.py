# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import numpy as np
import pytest
from sim2real.command_source import (
    ConstantGoalCommand,
    ConstantVelocityCommand,
    RecedingGoalCommand,
    make_command_source,
)
from sim2real.deployment_config import CommandsCfg
from sim2real.localization import DeadReckoningLocalizer

CFG = CommandsCfg(
    velocity={"vx": 0.2, "vy": 0.0, "yaw_rate": 0.0},
    goal={"x": 5.0, "y": 0.0, "z": 0.0, "heading": 0.0},
)
RECEDING_CFG = CommandsCfg(
    velocity={"vx": 0.2, "vy": 0.0, "yaw_rate": 0.0},
    goal={"mode": "receding", "lookahead_m": 2.0, "x": 0.0, "y": 0.0, "z": 0.0, "heading": 0.0},
)


def test_constant_velocity_command():
    src = ConstantVelocityCommand(CFG)
    assert np.allclose(src.get_command(), [0.2, 0.0, 0.0])


def test_constant_goal_command():
    src = ConstantGoalCommand(CFG)
    assert np.allclose(src.get_command(), [5.0, 0.0, 0.0, 0.0])


def test_get_command_returns_a_copy():
    src = ConstantVelocityCommand(CFG)
    cmd = src.get_command()
    cmd[0] = 999.0
    assert src.get_command()[0] == 0.2


def test_make_command_source_factory():
    assert isinstance(make_command_source("velocity", CFG), ConstantVelocityCommand)
    assert isinstance(make_command_source("goal", CFG), ConstantGoalCommand)
    assert isinstance(make_command_source("binary", CFG), ConstantGoalCommand)
    with pytest.raises(ValueError):
        make_command_source("bogus", CFG)


def test_make_command_source_receding_needs_localizer():
    with pytest.raises(ValueError, match="localizer"):
        make_command_source("goal", RECEDING_CFG)
    src = make_command_source("goal", RECEDING_CFG, localizer=DeadReckoningLocalizer())
    assert isinstance(src, RecedingGoalCommand)


def test_make_command_source_rejects_unknown_goal_mode():
    bad = CommandsCfg(velocity=CFG.velocity, goal={**CFG.goal, "mode": "orbit"})
    with pytest.raises(ValueError, match="mode"):
        make_command_source("goal", bad, localizer=DeadReckoningLocalizer())


def test_receding_goal_stays_lookahead_ahead_of_estimated_position():
    loc = DeadReckoningLocalizer()
    src = RecedingGoalCommand(RECEDING_CFG, loc)
    # at the origin, heading 0: goal is straight ahead at x = lookahead
    assert np.allclose(src.get_command(), [2.0, 0.0, 0.0, 0.0])
    # advance the estimate forward; the goal recedes with it
    loc.x, loc.y = 1.3, 0.4
    assert np.allclose(src.get_command(), [3.3, 0.4, 0.0, 0.0])


def test_receding_goal_walks_along_world_heading():
    loc = DeadReckoningLocalizer()
    cfg = CommandsCfg(
        velocity=CFG.velocity,
        goal={"mode": "receding", "lookahead_m": 2.0, "heading": math.pi / 2, "z": 0.0},
    )
    src = RecedingGoalCommand(cfg, loc)
    got = src.get_command()
    assert np.allclose(got, [0.0, 2.0, 0.0, math.pi / 2], atol=1e-9)


def test_receding_goal_rejects_nonpositive_lookahead():
    loc = DeadReckoningLocalizer()
    with pytest.raises(ValueError, match="lookahead_m"):
        RecedingGoalCommand(CommandsCfg(velocity=CFG.velocity, goal={"mode": "receding", "lookahead_m": 0.0}), loc)
