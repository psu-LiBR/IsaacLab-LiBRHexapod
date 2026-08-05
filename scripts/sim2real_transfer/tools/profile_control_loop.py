#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Per-phase timing breakdown for the control loop, to diagnose loop overruns.

Torque is never enabled -- this is a pure read/compute/write timing diagnostic,
safe to run against real hardware (same bring-up stage 4 risk profile as
`run_policy.py --no-torque`: goal positions are written every step regardless,
but the servos never receive torque so nothing moves). Mirrors run_policy.py's
object wiring but replaces control_loop.run()'s single loop with individually
timed phases (IMU read, position read, velocity read, obs build + policy
inference, tick conversion, goal write) so a "loop overrun" (see
control_loop.py's print of the same name) can be attributed to an actual
bottleneck instead of guessed at.

Usage (mirrors run_policy.py's flags):
    python3 tools/profile_control_loop.py --policy <policy.onnx> --profile velocity \
        --config config/deployment.yaml --steps 200

Add --dry-run --fake-imu for a hardware-free baseline -- isolates software-only
cost (obs build + ONNX inference) from serial/ROS I/O cost, useful for telling
apart "the bus is slow" from "the Pi's CPU is slow" from "the rclpy spin thread
is stealing the GIL".
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim2real.command_source import make_command_source  # noqa: E402
from sim2real.deployment_config import load_deployment_config  # noqa: E402
from sim2real.dynamixel_bus import DryRunDynamixelBus, RealDynamixelBus  # noqa: E402
from sim2real.imu import FakeImu, ImuStaleError, RosImuReader  # noqa: E402
from sim2real.joint_mapping import JointMapping, ordered_array  # noqa: E402
from sim2real.policy_runner import PolicyRunner  # noqa: E402
from sim2real.profiles import PROFILES, make_obs_builder  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy", required=True, help="Path to an exported policy.onnx checkpoint.")
    parser.add_argument("--profile", required=True, choices=sorted(PROFILES))
    parser.add_argument("--config", required=True, help="Path to deployment.yaml.")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument(
        "--dry-run", action="store_true", help="Use DryRunDynamixelBus instead of the real serial bus."
    )
    parser.add_argument("--fake-imu", action="store_true", help="Use FakeImu instead of subscribing to /imu.")
    return parser.parse_args()


def _summarize(name: str, samples_ms: list[float]) -> str:
    if not samples_ms:
        return f"{name:26s} {'--':>8s} {'--':>8s} {'--':>8s} {'--':>8s}"
    s = sorted(samples_ms)
    n = len(s)
    mean = sum(s) / n
    p50 = s[n // 2]
    p95 = s[min(n - 1, int(round(n * 0.95)))]
    return f"{name:26s} {mean:8.2f} {p50:8.2f} {p95:8.2f} {s[-1]:8.2f}"


def main() -> None:
    args = parse_args()

    cfg = load_deployment_config(args.config)
    joint_mapping = JointMapping(cfg.joints)
    profile = PROFILES[args.profile]

    policy = PolicyRunner(args.policy, profile)
    obs_builder = make_obs_builder(args.profile)
    command_source = make_command_source(args.profile, cfg.commands)

    if args.dry_run:
        bus = DryRunDynamixelBus(joint_mapping.motor_ids)
    else:
        bus = RealDynamixelBus(
            joint_mapping.motor_ids, cfg.serial.port, cfg.serial.baud_rate, cfg.serial.protocol_version
        )

    imu = (
        FakeImu()
        if args.fake_imu
        else RosImuReader(
            cfg.imu.topic, cfg.imu.mount_offset_quat, cfg.imu.negate_gyro_z, max_staleness_s=cfg.imu.max_staleness_s
        )
    )

    q_default_sim = ordered_array(cfg.control.q_default_sim, joint_mapping.sim_order)
    rate_hz = cfg.control.rate_hz
    period = 1.0 / rate_hz

    phases: dict[str, list[float]] = defaultdict(list)
    overruns = 0

    imu.start()
    try:
        # RosImuReader needs a moment for its background rclpy spin thread to
        # receive the first /imu message; control_loop.run() gets this for free
        # from soft_start_ramp's multi-second ramp before its first imu.read().
        # This tool has no ramp, so wait explicitly instead of tripping
        # ImuStaleError on step 0.
        warmup_deadline = time.perf_counter() + 3.0
        while True:
            try:
                imu.read()
                break
            except ImuStaleError:
                if time.perf_counter() > warmup_deadline:
                    raise
                time.sleep(0.05)

        for _step in range(args.steps):
            t0 = time.perf_counter()

            gyro, gravity = imu.read()
            t1 = time.perf_counter()

            ticks, raw_vel = bus.read_positions_and_velocities()
            t3 = time.perf_counter()
            measured_pos_sim = joint_mapping.ticks_to_sim_rad(ticks)
            measured_vel_sim = joint_mapping.real_radps_to_sim_radps(raw_vel)

            command = command_source.get_command()
            obs = obs_builder.build(
                gyro, gravity, measured_pos_sim, measured_vel_sim, policy.last_action, q_default_sim, command
            )
            t4 = time.perf_counter()

            raw_action = policy.step(obs)
            t5 = time.perf_counter()

            target_sim = q_default_sim + cfg.control.action_scale * cfg.control.action_scale_multiplier * raw_action
            target_sim = joint_mapping.clip_to_soft_limits(target_sim)
            target_ticks = joint_mapping.sim_target_to_ticks(target_sim)
            t6 = time.perf_counter()

            bus.write_goal_positions(target_ticks)
            t7 = time.perf_counter()

            phases["imu.read"].append((t1 - t0) * 1000)
            phases["bus.read_positions_and_velocities"].append((t3 - t1) * 1000)
            phases["obs_build+command"].append((t4 - t3) * 1000)
            phases["policy.step (onnx)"].append((t5 - t4) * 1000)
            phases["clip+tick_convert"].append((t6 - t5) * 1000)
            phases["bus.write_goal_positions"].append((t7 - t6) * 1000)
            phases["TOTAL"].append((t7 - t0) * 1000)

            remaining = period - (t7 - t0)
            if remaining <= 0:
                overruns += 1
            else:
                time.sleep(remaining)
    except ImuStaleError as exc:
        print(f"IMU went stale mid-run: {exc}")
    finally:
        imu.stop()
        bus.close()

    n = len(phases["TOTAL"])
    print(f"\n{'phase':26s} {'mean':>8s} {'p50':>8s} {'p95':>8s} {'max':>8s}   (ms, n={n})")
    for name in (
        "imu.read",
        "bus.read_positions_and_velocities",
        "obs_build+command",
        "policy.step (onnx)",
        "clip+tick_convert",
        "bus.write_goal_positions",
        "TOTAL",
    ):
        print(_summarize(name, phases[name]))
    print(f"\ncontrol budget per step: {period * 1000:.1f} ms (rate_hz={rate_hz})")
    print(f"steps that overran budget: {overruns}/{n}")


if __name__ == "__main__":
    main()
