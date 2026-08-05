# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The 50 Hz policy control loop.

`imu.RosImuReader` needs `rclpy` running (spinning in a background thread) while
the servo/inference loop drives the bus directly from the main thread -- the same
hybrid-threading shape as `hexapod_tripod_adaptive.py` (main thread) +
`combined_logger.py` (rclpy spin thread) already use together on the real robot.
This function owns exactly that lifecycle: `imu.start()`/`imu.stop()` bracket the
loop, and a `try`/`finally` around the whole body guarantees `soft_stop_ramp` +
`torque_enable(False)` run on normal completion, an unhandled exception, or
Ctrl+C alike.

`imu.read()` raising `ImuStaleError` (no data yet, or the topic has gone silent
past `RosImuReader`'s staleness threshold -- e.g. the `bno08x_driver` process
dying mid-run) is treated as a recognized failure mode, not a bug: it trips the
Watchdog with a clear reason and breaks the loop through the same clean-exit
path as the other `watchdog.check(extra_ok=False, ...)` sites below, rather
than propagating as a raw exception.
"""

from __future__ import annotations

import time

from .command_source import CommandSource
from .deployment_config import DeploymentConfig
from .dynamixel_bus import DynamixelBus
from .imu import ImuReader, ImuStaleError
from .joint_mapping import JointMapping, ordered_array
from .localization import DeadReckoningLocalizer
from .logging_utils import CsvRunLogger
from .policy_runner import PolicyRunner
from .profiles import ObsBuilder
from .safety import Watchdog, all_finite, soft_start_ramp, soft_stop_ramp, ticks_in_valid_range


def run(
    cfg: DeploymentConfig,
    profile_name: str,
    policy_runner: PolicyRunner,
    obs_builder: ObsBuilder,
    joint_mapping: JointMapping,
    bus: DynamixelBus,
    imu: ImuReader,
    command_source: CommandSource,
    localizer: DeadReckoningLocalizer | None = None,
    logger: CsvRunLogger | None = None,
    dry_run: bool = False,
    enable_torque: bool = True,
    rate_hz: float | None = None,
    action_scale_multiplier: float | None = None,
    duration_s: float | None = None,
) -> None:
    if profile_name == "goal" and localizer is None:
        raise ValueError("goal profile requires a localizer")

    rate_hz = rate_hz if rate_hz is not None else cfg.control.rate_hz
    action_scale_multiplier = (
        action_scale_multiplier if action_scale_multiplier is not None else cfg.control.action_scale_multiplier
    )
    period = 1.0 / rate_hz

    q_default_sim = ordered_array(cfg.control.q_default_sim, joint_mapping.sim_order)

    imu.start()
    watchdog = Watchdog(max_silence_s=max(1.0, 5.0 * period))
    step = 0

    try:
        soft_start_ramp(bus, joint_mapping, q_default_sim, cfg.control.soft_start_seconds, rate_hz)
        if enable_torque:
            bus.torque_enable(True)
        watchdog.feed()

        run_start = time.perf_counter()
        while duration_s is None or (time.perf_counter() - run_start) < duration_s:
            iter_start = time.perf_counter()

            try:
                gyro, gravity = imu.read()
            except ImuStaleError as exc:
                watchdog.check(extra_ok=False, reason=str(exc))
                print(f"[control_loop] watchdog tripped: {watchdog.trip_reason}")
                break

            ticks, raw_vel = bus.read_positions_and_velocities()
            measured_pos_sim = joint_mapping.ticks_to_sim_rad(ticks)
            measured_vel_sim = joint_mapping.real_radps_to_sim_radps(raw_vel)

            if profile_name == "velocity":
                command = command_source.get_command()
            else:
                goal = command_source.get_command()
                localizer.update(period, gyro[2])
                command = localizer.get_pose_command(goal[:3], goal[3])

            obs = obs_builder.build(
                gyro,
                gravity,
                measured_pos_sim,
                measured_vel_sim,
                policy_runner.last_action,
                q_default_sim,
                command,
            )
            if not watchdog.check(extra_ok=all_finite(obs), reason="non-finite observation"):
                print(f"[control_loop] watchdog tripped: {watchdog.trip_reason}")
                break

            raw_action = policy_runner.step(obs)
            target_sim = q_default_sim + cfg.control.action_scale * action_scale_multiplier * raw_action
            target_sim = joint_mapping.clip_to_soft_limits(target_sim)
            target_ticks = joint_mapping.sim_target_to_ticks(target_sim)

            ticks_ok = ticks_in_valid_range(target_ticks, joint_mapping.ticks_per_rev)
            if not watchdog.check(extra_ok=ticks_ok, reason="target tick out of hardware range"):
                print(f"[control_loop] watchdog tripped: {watchdog.trip_reason}")
                break

            if not dry_run:
                bus.write_goal_positions(target_ticks)
            watchdog.feed()

            loop_time_ms = (time.perf_counter() - iter_start) * 1000.0
            if logger is not None:
                logger.log(
                    step,
                    time.perf_counter() - run_start,
                    obs,
                    raw_action,
                    target_sim,
                    target_ticks,
                    measured_pos_sim,
                    measured_vel_sim,
                    loop_time_ms,
                )

            elapsed = time.perf_counter() - iter_start
            remaining = period - elapsed
            if remaining > 0:
                time.sleep(remaining)
            else:
                print(f"[control_loop] step {step}: loop overrun by {-remaining * 1000:.2f} ms")
            step += 1
    finally:
        soft_stop_ramp(bus, joint_mapping, q_default_sim, cfg.control.soft_stop_seconds, rate_hz)
        bus.torque_enable(False)
        imu.stop()
        if logger is not None:
            logger.close()
