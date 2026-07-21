# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
from sim2real.command_source import (
    ConstantGoalCommand,
    ConstantVelocityCommand,
    make_command_source,
)
from sim2real.deployment_config import CommandsCfg

CFG = CommandsCfg(
    velocity={"vx": 0.2, "vy": 0.0, "yaw_rate": 0.0},
    goal={"x": 5.0, "y": 0.0, "z": 0.0, "heading": 0.0},
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
    with pytest.raises(ValueError):
        make_command_source("bogus", CFG)
