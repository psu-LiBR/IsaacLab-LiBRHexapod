# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Dead-reckoning pose estimator -- goal-profile-only, and the weakest link in the
whole pipeline.

The sim's `pose_command` obs term (see `pose_2d_command.py::UniformPose2dCommand`)
needs the robot's own world position/heading, which the simulator gets for free
from ground truth. The real robot has no localization system (no GPS/mocap/AprilTag)
-- this class is a placeholder that integrates gyro-z for heading (reasonably
accurate over a 45s episode) and assumes a constant forward speed for position
(much less accurate; true accelerometer double-integration drifts even faster and
isn't used here). Bring up the velocity profile first, since it has zero dependency
on this module; validate this class separately (walk a known straight distance,
compare estimate vs. actual) before trusting it in the goal profile.

The weak part -- the constant-forward-speed position estimate -- only matters for
a FIXED goal, where its error directly corrupts "how far to the goal". With
`command_source.RecedingGoalCommand` the goal is regenerated a fixed distance
ahead of this same estimate every step, so the position error cancels out of the
body-frame pose_command and only `heading` (the gyro-integrated, reasonably
accurate term) is load-bearing.
"""

from __future__ import annotations

import math

import numpy as np

# m. Constant stand-in for the pose_command z term, which the real robot can't measure.
# UniformPose2dCommand sets goal z to the default root height (0.2 m) and reports
# goal z - current root z; the walking root sits ~0.06 m, so training saw ~0.14 m
# (sim traces: mean 0.140-0.151, range 0.115-0.164). Sending 0 instead is ~4.7 std
# off the goal-mimic policy's baked-in obs normalizer and shifts its actions by
# >0.25 rad on almost every step.
NOMINAL_GOAL_Z_B = 0.14


def wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class DeadReckoningLocalizer:
    def __init__(self, forward_speed_estimate: float = 0.15):
        """`forward_speed_estimate` (m/s) is a rough constant derived from the
        sim's measured gait speed (~0.14-0.16 m/s), not a measured quantity."""
        self.forward_speed_estimate = forward_speed_estimate
        self.x = 0.0
        self.y = 0.0
        self.heading = 0.0

    def reset(self) -> None:
        self.x = 0.0
        self.y = 0.0
        self.heading = 0.0

    def update(self, dt: float, gyro_z_sim_body: float) -> None:
        self.heading = wrap_to_pi(self.heading + gyro_z_sim_body * dt)
        self.x += self.forward_speed_estimate * math.cos(self.heading) * dt
        self.y += self.forward_speed_estimate * math.sin(self.heading) * dt

    def get_pose_command(self, goal_xyz_world: np.ndarray, goal_heading_world: float) -> np.ndarray:
        """Returns the 4-dim [x_b, y_b, z_b, heading_b] pose_command obs term, using
        the same relative-pose formula as UniformPose2dCommand._update_command()."""
        dx = goal_xyz_world[0] - self.x
        dy = goal_xyz_world[1] - self.y
        cos_h, sin_h = math.cos(self.heading), math.sin(self.heading)
        pos_b_x = cos_h * dx + sin_h * dy
        pos_b_y = -sin_h * dx + cos_h * dy
        # No z estimate is tracked -- use the training-time nominal value (see
        # NOMINAL_GOAL_Z_B), not 0.
        pos_b_z = NOMINAL_GOAL_Z_B
        heading_b = wrap_to_pi(goal_heading_world - self.heading)
        return np.array([pos_b_x, pos_b_y, pos_b_z, heading_b])
