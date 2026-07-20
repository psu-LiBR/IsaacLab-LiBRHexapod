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
"""

from __future__ import annotations

import math

import numpy as np


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
        # No z estimate is tracked -- goal z is defined to equal the robot's own
        # nominal height in sim, so this approximates to 0 by construction.
        pos_b_z = 0.0
        heading_b = wrap_to_pi(goal_heading_world - self.heading)
        return np.array([pos_b_x, pos_b_y, pos_b_z, heading_b])
