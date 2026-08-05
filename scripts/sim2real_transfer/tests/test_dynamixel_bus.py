# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import sys

import numpy as np
import pytest
from sim2real.dynamixel_bus import DryRunDynamixelBus, RealDynamixelBus, _to_signed32

MOTOR_IDS = [3, 6, 2, 1, 4, 5, 7, 8]


def test_dry_run_default_state():
    bus = DryRunDynamixelBus(MOTOR_IDS)
    assert np.array_equal(bus.read_positions(), np.full(8, 2048))
    assert np.array_equal(bus.read_velocities(), np.zeros(8))
    assert bus.torque_on is False


def test_dry_run_torque_enable():
    bus = DryRunDynamixelBus(MOTOR_IDS)
    bus.torque_enable(True)
    assert bus.torque_on is True
    bus.torque_enable(False)
    assert bus.torque_on is False


def test_dry_run_write_then_read():
    bus = DryRunDynamixelBus(MOTOR_IDS)
    target = np.array([100, 200, 300, 400, 500, 600, 700, 800])
    bus.write_goal_positions(target)
    assert np.array_equal(bus.read_positions(), target)


def test_dry_run_write_clips_to_valid_tick_range():
    bus = DryRunDynamixelBus(MOTOR_IDS)
    bus.write_goal_positions(np.array([-100, 5000, 2048, 2048, 2048, 2048, 2048, 2048]))
    positions = bus.read_positions()
    assert positions[0] == 0
    assert positions[1] == 4095


def test_read_positions_and_velocities_matches_separate_calls():
    """DynamixelBus's base-class default combines read_positions()+read_velocities();
    RealDynamixelBus overrides it with a single merged bulk read for speed (see
    dynamixel_bus.py's LEN_PRESENT_VEL_AND_POS), but the two must always agree."""
    bus = DryRunDynamixelBus(MOTOR_IDS)
    target = np.array([100, 200, 300, 400, 500, 600, 700, 800])
    bus.write_goal_positions(target)
    ticks, radps = bus.read_positions_and_velocities()
    assert np.array_equal(ticks, bus.read_positions())
    assert np.array_equal(radps, bus.read_velocities())


def test_real_bus_requires_dynamixel_sdk(monkeypatch):
    """Simulate dynamixel_sdk being absent regardless of the environment.

    On a robot host the SDK *is* installed, and letting the constructor proceed
    would open the real serial port and write to live servos. Setting the
    sys.modules entry to None makes the `from dynamixel_sdk import ...` inside
    RealDynamixelBus.__init__ raise ImportError before any hardware is touched.
    """
    monkeypatch.setitem(sys.modules, "dynamixel_sdk", None)
    with pytest.raises(ImportError, match="dynamixel_sdk"):
        RealDynamixelBus(MOTOR_IDS, port="/dev/ttyUSB0", baud_rate=1_000_000, protocol_version=2.0)


def test_to_signed32():
    assert _to_signed32(0) == 0
    assert _to_signed32(100) == 100
    assert _to_signed32(0xFFFFFFFF) == -1
    assert _to_signed32(0x80000000) == -2147483648
