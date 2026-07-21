# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import numpy as np
from sim2real.localization import DeadReckoningLocalizer, wrap_to_pi


def test_wrap_to_pi():
    assert math.isclose(wrap_to_pi(0.0), 0.0)
    assert math.isclose(wrap_to_pi(3 * math.pi), -math.pi, abs_tol=1e-9) or math.isclose(
        wrap_to_pi(3 * math.pi), math.pi, abs_tol=1e-9
    )
    assert math.isclose(wrap_to_pi(-3 * math.pi), math.pi, abs_tol=1e-9) or math.isclose(
        wrap_to_pi(-3 * math.pi), -math.pi, abs_tol=1e-9
    )


def test_heading_integrates_gyro_z():
    loc = DeadReckoningLocalizer()
    for _ in range(50):
        loc.update(dt=0.02, gyro_z_sim_body=1.0)  # 1 rad/s for 1s total
    assert math.isclose(loc.heading, 1.0, abs_tol=1e-6)


def test_position_advances_forward_with_zero_heading():
    loc = DeadReckoningLocalizer(forward_speed_estimate=0.2)
    for _ in range(50):
        loc.update(dt=0.02, gyro_z_sim_body=0.0)  # 1s total, heading stays 0
    assert math.isclose(loc.x, 0.2 * 1.0, abs_tol=1e-6)
    assert math.isclose(loc.y, 0.0, abs_tol=1e-9)


def test_reset_zeroes_state():
    loc = DeadReckoningLocalizer()
    loc.update(dt=0.02, gyro_z_sim_body=1.0)
    loc.reset()
    assert loc.x == 0.0 and loc.y == 0.0 and loc.heading == 0.0


def test_pose_command_at_origin_facing_goal():
    loc = DeadReckoningLocalizer()
    command = loc.get_pose_command(goal_xyz_world=np.array([5.0, 0.0, 0.0]), goal_heading_world=0.0)
    assert command.shape == (4,)
    assert math.isclose(command[0], 5.0)
    assert math.isclose(command[1], 0.0)
    assert math.isclose(command[3], 0.0)


def test_pose_command_rotates_with_heading():
    loc = DeadReckoningLocalizer()
    loc.heading = math.pi / 2  # facing +y in world
    command = loc.get_pose_command(goal_xyz_world=np.array([5.0, 0.0, 0.0]), goal_heading_world=0.0)
    # Goal is due +x in world; a robot facing +y sees that as directly to its
    # right (y_b < 0), with nothing gained "ahead" (x_b ~ 0).
    assert math.isclose(command[0], 0.0, abs_tol=1e-9)
    assert command[1] < 0.0
    assert math.isclose(command[3], -math.pi / 2)
