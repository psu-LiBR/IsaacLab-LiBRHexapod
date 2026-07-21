# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import sys
import types

import numpy as np
import pytest
from sim2real.imu import (
    DEFAULT_MAX_STALENESS_S,
    FakeImu,
    ImuStaleError,
    RosImuReader,
    _check_freshness,
    quat_apply_inverse,
    quat_to_rot_matrix,
)

IDENTITY_QUAT = np.array([1.0, 0.0, 0.0, 0.0])


def test_identity_quat_is_identity_matrix():
    R = quat_to_rot_matrix(IDENTITY_QUAT)
    assert np.allclose(R, np.eye(3))


def test_quat_apply_inverse_identity_no_op():
    v = np.array([1.0, 2.0, 3.0])
    assert np.allclose(quat_apply_inverse(IDENTITY_QUAT, v), v)


def test_level_robot_gravity_is_down():
    # A level robot's orientation quaternion is identity; projected gravity
    # (world [0,0,-1] rotated into body frame) should also read [0,0,-1].
    from sim2real.imu import GRAVITY_VEC_WORLD

    gravity_body = quat_apply_inverse(IDENTITY_QUAT, GRAVITY_VEC_WORLD)
    assert np.allclose(gravity_body, [0.0, 0.0, -1.0])


def test_90deg_yaw_rotates_gravity_axes_but_not_z():
    # Quaternion for 90deg rotation about Z: (w,x,y,z) = (cos45, 0, 0, sin45)
    half = math.pi / 4
    q = np.array([math.cos(half), 0.0, 0.0, math.sin(half)])
    from sim2real.imu import GRAVITY_VEC_WORLD

    gravity_body = quat_apply_inverse(q, GRAVITY_VEC_WORLD)
    # Pure yaw doesn't change the vertical component of a vector already along Z.
    assert math.isclose(gravity_body[2], -1.0, abs_tol=1e-9)


def test_fake_imu_reads_level_stationary():
    imu = FakeImu()
    gyro, gravity = imu.read()
    assert np.allclose(gyro, 0.0)
    assert np.allclose(gravity, [0.0, 0.0, -1.0])


def test_ros_imu_reader_requires_rclpy():
    with pytest.raises(ImportError, match="rclpy"):
        RosImuReader(topic="/imu", mount_offset_quat=(1.0, 0.0, 0.0, 0.0), negate_gyro_z=True)


# ---------------------------------------------------------------------------
# Staleness detection
# ---------------------------------------------------------------------------
# `_check_freshness` is the pure timestamp-checking core of RosImuReader.read()
# (no rclpy/threading involved), so it's directly unit-testable. The tests
# below it go one level up and exercise the real RosImuReader class end to
# end by installing fake `rclpy`/`sensor_msgs` modules into sys.modules --
# rclpy isn't installed on this dev machine, and RosImuReader.__init__ raises
# ImportError immediately without something importable under those names.


def test_check_freshness_no_data_ever_raises():
    with pytest.raises(ImuStaleError, match="no IMU data"):
        _check_freshness(have_data=False, last_sample_time=None, now=100.0, max_staleness_s=0.25, topic="/imu")


def test_check_freshness_fresh_sample_is_ok():
    # Should not raise.
    _check_freshness(have_data=True, last_sample_time=100.0, now=100.1, max_staleness_s=0.25, topic="/imu")


def test_check_freshness_stale_sample_raises():
    with pytest.raises(ImuStaleError, match="stale"):
        _check_freshness(have_data=True, last_sample_time=100.0, now=100.5, max_staleness_s=0.25, topic="/imu")


def test_check_freshness_exactly_at_threshold_is_ok():
    # age == max_staleness_s should not (yet) count as stale.
    _check_freshness(have_data=True, last_sample_time=100.0, now=100.25, max_staleness_s=0.25, topic="/imu")


class _FakeVec3:
    def __init__(self, x: float, y: float, z: float):
        self.x, self.y, self.z = x, y, z


class _FakeQuat:
    def __init__(self, w: float, x: float, y: float, z: float):
        self.w, self.x, self.y, self.z = w, x, y, z


class _FakeImuMsg:
    def __init__(self):
        self.angular_velocity = _FakeVec3(0.1, 0.2, 0.3)
        self.orientation = _FakeQuat(1.0, 0.0, 0.0, 0.0)


def _install_fake_ros_modules(monkeypatch):
    """Installs minimal fake `rclpy`/`rclpy.node`/`sensor_msgs.msg` modules so
    `RosImuReader.__init__`'s `import rclpy` / `from rclpy.node import Node` /
    `from sensor_msgs.msg import Imu` succeed without the real ROS2 stack.
    `start()`/`stop()` (which actually spin rclpy) are not exercised by these
    tests -- only construction + the `_on_imu` callback + `read()`."""
    rclpy_module = types.ModuleType("rclpy")
    rclpy_module.ok = lambda: False
    rclpy_module.init = lambda: None
    rclpy_module.shutdown = lambda: None
    rclpy_module.spin = lambda node: None

    node_module = types.ModuleType("rclpy.node")

    class _FakeNode:
        def __init__(self, name):
            self._name = name

        def create_subscription(self, msg_type, topic, callback, qos):
            return None

        def destroy_node(self):
            pass

    node_module.Node = _FakeNode

    sensor_msgs_module = types.ModuleType("sensor_msgs")
    sensor_msgs_msg_module = types.ModuleType("sensor_msgs.msg")
    sensor_msgs_msg_module.Imu = _FakeImuMsg
    sensor_msgs_module.msg = sensor_msgs_msg_module

    monkeypatch.setitem(sys.modules, "rclpy", rclpy_module)
    monkeypatch.setitem(sys.modules, "rclpy.node", node_module)
    monkeypatch.setitem(sys.modules, "sensor_msgs", sensor_msgs_module)
    monkeypatch.setitem(sys.modules, "sensor_msgs.msg", sensor_msgs_msg_module)


def test_ros_imu_reader_fresh_data_reads_successfully(monkeypatch):
    _install_fake_ros_modules(monkeypatch)
    clock = [100.0]
    reader = RosImuReader(
        topic="/imu",
        mount_offset_quat=(1.0, 0.0, 0.0, 0.0),
        negate_gyro_z=False,
        max_staleness_s=0.25,
        time_fn=lambda: clock[0],
    )
    reader._on_imu(_FakeImuMsg())

    clock[0] = 100.1  # 0.1s later -- within the 0.25s threshold
    gyro, gravity = reader.read()
    assert np.allclose(gyro, [0.1, 0.2, 0.3])
    assert np.allclose(gravity, [0.0, 0.0, -1.0])
    assert reader.seconds_since_last_sample() == pytest.approx(0.1)


def test_ros_imu_reader_stale_data_raises(monkeypatch):
    _install_fake_ros_modules(monkeypatch)
    clock = [100.0]
    reader = RosImuReader(
        topic="/imu",
        mount_offset_quat=(1.0, 0.0, 0.0, 0.0),
        negate_gyro_z=False,
        max_staleness_s=0.25,
        time_fn=lambda: clock[0],
    )
    reader._on_imu(_FakeImuMsg())

    clock[0] = 100.3  # 0.3s later -- past the 0.25s threshold
    with pytest.raises(ImuStaleError, match="stale"):
        reader.read()


def test_ros_imu_reader_no_data_ever_raises(monkeypatch):
    _install_fake_ros_modules(monkeypatch)
    reader = RosImuReader(topic="/imu", mount_offset_quat=(1.0, 0.0, 0.0, 0.0), negate_gyro_z=False)
    assert reader.seconds_since_last_sample() is None
    with pytest.raises(ImuStaleError, match="no IMU data"):
        reader.read()


def test_ros_imu_reader_default_max_staleness_matches_module_default(monkeypatch):
    _install_fake_ros_modules(monkeypatch)
    reader = RosImuReader(topic="/imu", mount_offset_quat=(1.0, 0.0, 0.0, 0.0), negate_gyro_z=False)
    assert reader._max_staleness_s == DEFAULT_MAX_STALENESS_S
