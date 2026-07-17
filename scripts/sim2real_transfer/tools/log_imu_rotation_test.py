#!/usr/bin/env python3
"""Log raw /imu messages while manually rotating the robot, for mount_offset_quat /
negate_gyro_z calibration (see CLAUDE.md's sim2real IMU bring-up notes).

Run this ON THE ROBOT HOST (wherever `bno08x_driver` publishes /imu), with the
driver already running in another terminal:

    ros2 launch bno08x_driver bno085_i2c.launch.py

Then, in a second terminal:

    python3 log_imu_rotation_test.py --out imu_rotation_log.csv

Watch the live "yaw_rel" readout and physically rotate the robot: hold still
~10s at your starting heading, then rotate 90 degrees counterclockwise as
viewed from above, hold ~10s, repeat to 180/270, optionally back to 360 (=0)
as a consistency check. The printed yaw_rel tells you when you've actually
hit each increment -- no need to time it by feel.

Everything (full orientation quaternion, angular velocity, linear
acceleration, wall-clock timestamp) is logged to CSV for later analysis --
this script only prints a live estimate for your convenience, it doesn't
decide anything on its own.
"""

from __future__ import annotations

import argparse
import csv
import math
import threading
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu as ImuMsg


def quat_to_yaw_deg(w: float, x: float, y: float, z: float) -> float:
    """Yaw (rotation about Z) from a quaternion, standard atan2 form, degrees in (-180, 180]."""
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.degrees(math.atan2(siny_cosp, cosy_cosp))


class _State:
    def __init__(self):
        self.count = 0
        self.yaw0 = None
        self.yaw_prev_raw = None
        self.yaw_unwrapped = None
        self.gyro_z = 0.0
        self.lock = threading.Lock()

    def update(self, yaw_raw: float, gyro_z: float) -> None:
        with self.lock:
            if self.yaw0 is None:
                self.yaw0 = yaw_raw
                self.yaw_prev_raw = yaw_raw
                self.yaw_unwrapped = yaw_raw
            else:
                delta = yaw_raw - self.yaw_prev_raw
                if delta > 180.0:
                    delta -= 360.0
                elif delta < -180.0:
                    delta += 360.0
                self.yaw_unwrapped += delta
                self.yaw_prev_raw = yaw_raw
            self.gyro_z = gyro_z
            self.count += 1

    def snapshot(self):
        with self.lock:
            if self.yaw0 is None:
                return None
            return self.yaw_unwrapped - self.yaw0, self.gyro_z, self.count


class ImuLoggerNode(Node):
    def __init__(self, topic: str, writer: "csv._writer", state: _State):
        super().__init__("imu_rotation_test_logger")
        self._writer = writer
        self._state = state
        self.subscription = self.create_subscription(ImuMsg, topic, self._on_imu, 50)

    def _on_imu(self, msg: ImuMsg) -> None:
        t = time.time()
        o = msg.orientation
        av = msg.angular_velocity
        la = msg.linear_acceleration
        self._writer.writerow(
            [f"{t:.6f}", o.w, o.x, o.y, o.z, av.x, av.y, av.z, la.x, la.y, la.z]
        )
        self._state.update(quat_to_yaw_deg(o.w, o.x, o.y, o.z), av.z)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--topic", default="/imu", help="IMU topic to subscribe to (default: /imu)")
    parser.add_argument("--duration", type=float, default=180.0, help="seconds to log (default: 180)")
    parser.add_argument("--out", default="imu_rotation_log.csv", help="output CSV path")
    args = parser.parse_args()

    state = _State()

    out_file = open(args.out, "w", newline="")
    writer = csv.writer(out_file)
    writer.writerow(["t_wall", "qw", "qx", "qy", "qz", "gyro_x", "gyro_y", "gyro_z", "acc_x", "acc_y", "acc_z"])

    rclpy.init()
    node = ImuLoggerNode(args.topic, writer, state)
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    print(f"Logging '{args.topic}' -> {args.out} for up to {args.duration:.0f}s.")
    print("Rotate the robot now: hold ~10s per heading, 90deg CCW-from-above each step.")
    print("Watch yaw_rel below; Ctrl+C stops early (already-logged data is kept).\n")

    start = time.time()
    try:
        while time.time() - start < args.duration:
            time.sleep(0.2)
            snap = state.snapshot()
            if snap is None:
                print("\r[waiting for first IMU message...]", end="", flush=True)
                continue
            yaw_rel, gyro_z, count = snap
            print(
                f"\rt={time.time() - start:5.1f}s  yaw_rel={yaw_rel:8.2f} deg  "
                f"gyro_z={gyro_z:+7.3f} rad/s  samples={count:6d}   ",
                end="",
                flush=True,
            )
    except KeyboardInterrupt:
        print("\nStopped early by user request.")

    print(f"\nDone. Wrote {state.count} samples to {args.out}.")

    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
    spin_thread.join(timeout=3.0)
    out_file.close()


if __name__ == "__main__":
    main()
