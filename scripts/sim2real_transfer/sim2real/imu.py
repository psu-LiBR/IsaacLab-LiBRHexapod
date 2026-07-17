"""IMU interface.

The Hexapi robot already runs a ROS2 node that publishes a fused orientation
quaternion + linear acceleration + angular velocity on `/imu` (`sensor_msgs/Imu`)
-- see `combined_logger.py`. There is no raw I2C driver or software sensor-fusion
filter to write; `RosImuReader` below just subscribes to that existing topic,
reusing the same rclpy.init()-once + background-spin-thread + lock-protected
latest-value pattern as `combined_logger.py`'s `_CombinedLoggerNode`/`CombinedLogger`.

The old `imu_processor.py` (Hexapi) mount facts (gyro_z negated, body-forward
`[0,1,0]`) are stale for the current mount -- the IMU board was physically
reseated during driver debugging. A fresh 4-heading rotation test (2026-07)
calibrated `negate_gyro_z: false` and a tilt-only `mount_offset_quat` into
config/deployment.yaml. The horizontal forward-axis mapping for this mount is
still uncalibrated -- run `tools/log_imu_axis_alignment_test.py` before trusting
it for closed-loop deployment.
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np

GRAVITY_VEC_WORLD = np.array([0.0, 0.0, -1.0])

# The IMU publishes at 100 Hz and the control loop runs at 50 Hz; 0.25s of
# silence is ~25 missed messages in a row -- unambiguously a dead topic
# (e.g. the bno08x_driver process crashing) rather than ordinary jitter.
DEFAULT_MAX_STALENESS_S = 0.25


class ImuStaleError(RuntimeError):
    """Raised by `ImuReader.read()` when no fresh IMU sample is available --
    either none has ever arrived, or the topic has gone silent for longer than
    the configured staleness threshold. Distinct from a bare RuntimeError so
    callers (control_loop.run) can catch specifically this and trip the
    Watchdog cleanly instead of letting an unrelated bug propagate unnoticed."""


def _check_freshness(
    have_data: bool,
    last_sample_time: float | None,
    now: float,
    max_staleness_s: float,
    topic: str,
) -> None:
    """Raises `ImuStaleError` if no sample has ever arrived, or if the latest
    one is older than `max_staleness_s`. Kept as a small pure function (no
    rclpy/threading involved) so the staleness logic is unit-testable without
    ROS2 installed."""
    if not have_data or last_sample_time is None:
        raise ImuStaleError(f"no IMU data received yet on topic '{topic}'")
    age = now - last_sample_time
    if age > max_staleness_s:
        raise ImuStaleError(
            f"IMU data on topic '{topic}' is stale: last sample {age:.3f}s ago (max allowed {max_staleness_s}s)"
        )


def quat_to_rot_matrix(q_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = q_wxyz
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def quat_apply_inverse(q_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotates world-frame vector `v` into the frame described by quaternion `q`
    (frame -> world orientation), matching `isaaclab.utils.math.quat_apply_inverse`
    semantics used by `mdp.projected_gravity`."""
    return quat_to_rot_matrix(q_wxyz).T @ v


class ImuReader(ABC):
    @abstractmethod
    def read(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns (gyro_xyz_radps, gravity_unit_vector), both in sim body frame."""

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass


class RosImuReader(ImuReader):
    def __init__(
        self,
        topic: str,
        mount_offset_quat: tuple[float, float, float, float],
        negate_gyro_z: bool,
        max_staleness_s: float = DEFAULT_MAX_STALENESS_S,
        time_fn: Callable[[], float] = time.monotonic,
    ):
        try:
            import rclpy
            from rclpy.node import Node
            from sensor_msgs.msg import Imu as ImuMsg
        except ImportError as exc:
            raise ImportError(
                "rclpy and sensor_msgs are required for RosImuReader (ROS2); "
                "use FakeImu for hardware-free testing"
            ) from exc

        self._rclpy = rclpy
        self._topic = topic
        self._mount_offset_R = quat_to_rot_matrix(np.array(mount_offset_quat, dtype=np.float64))
        self._negate_gyro_z = negate_gyro_z
        self._max_staleness_s = max_staleness_s
        self._time_fn = time_fn

        self._lock = threading.Lock()
        self._latest_orientation_wxyz = np.array([1.0, 0.0, 0.0, 0.0])
        self._latest_gyro_imu = np.zeros(3)
        self._have_data = False
        self._last_sample_time: float | None = None
        self._thread: threading.Thread | None = None

        on_imu = self._on_imu

        class _ImuSubscriberNode(Node):
            def __init__(self_node):
                super().__init__("sim2real_imu_reader")
                self_node.subscription = self_node.create_subscription(ImuMsg, topic, on_imu, 50)

        self._node_cls = _ImuSubscriberNode
        self._node = None

    def _on_imu(self, msg) -> None:
        gyro_imu = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])
        if self._negate_gyro_z:
            gyro_imu[2] = -gyro_imu[2]
        o = msg.orientation
        with self._lock:
            self._latest_orientation_wxyz = np.array([o.w, o.x, o.y, o.z])
            self._latest_gyro_imu = gyro_imu
            self._have_data = True
            self._last_sample_time = self._time_fn()

    def start(self) -> None:
        if not self._rclpy.ok():
            self._rclpy.init()
        self._node = self._node_cls()
        self._thread = threading.Thread(target=self._rclpy.spin, args=(self._node,), daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._node is not None:
            self._node.destroy_node()
        if self._rclpy.ok():
            self._rclpy.shutdown()
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    def seconds_since_last_sample(self) -> float | None:
        """Returns None if no sample has ever arrived, else the age (seconds) of
        the latest one -- useful for diagnostics/logging outside of `read()`."""
        with self._lock:
            last_sample_time = self._last_sample_time
        if last_sample_time is None:
            return None
        return self._time_fn() - last_sample_time

    def read(self) -> tuple[np.ndarray, np.ndarray]:
        with self._lock:
            orientation = self._latest_orientation_wxyz.copy()
            gyro_imu = self._latest_gyro_imu.copy()
            have_data = self._have_data
            last_sample_time = self._last_sample_time
        _check_freshness(have_data, last_sample_time, self._time_fn(), self._max_staleness_s, self._topic)

        gravity_imu = quat_apply_inverse(orientation, GRAVITY_VEC_WORLD)
        gyro_sim_body = self._mount_offset_R @ gyro_imu
        gravity_sim_body = self._mount_offset_R @ gravity_imu
        return gyro_sim_body, gravity_sim_body


class FakeImu(ImuReader):
    """Level, stationary robot -- for dry-run testing with no IMU attached."""

    def read(self) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros(3), GRAVITY_VEC_WORLD.copy()
