# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Watchdog, startup/shutdown ramps.

Joint limit clipping lives on `JointMapping.clip_to_soft_limits` (it needs the
per-joint limit arrays already loaded there) rather than being duplicated here.

`ramp_to_target` (used by both `soft_start_ramp` and `soft_stop_ramp`) is modeled
on `hexapod_tripod_adaptive.py`'s shutdown sequence, which ramps to a safe
`post_up_pos` before disabling torque -- this package does the same thing
symmetrically at both startup (from the robot's arbitrary measured rest pose) and
shutdown (from wherever the policy left it), instead of slamming to `q_default`
instantly.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np

from .dynamixel_bus import DynamixelBus
from .joint_mapping import JointMapping


def all_finite(*arrays: np.ndarray) -> bool:
    return all(np.all(np.isfinite(a)) for a in arrays)


def ticks_in_valid_range(ticks: np.ndarray, ticks_per_rev: int) -> bool:
    """A computed tick outside [0, ticks_per_rev) usually means a calibration bug
    (wrong zero_tick/correction) rather than a real target -- silently clipping it
    (as DryRunDynamixelBus/hardware firmware may do) would command the joint to an
    arbitrary extreme instead of surfacing the error. Feed this into a Watchdog
    check before writing rather than trusting the clip."""
    return bool(np.all((ticks >= 0) & (ticks < ticks_per_rev)))


class Watchdog:
    """Trips on comms silence or a failed sanity check; the caller is responsible
    for actually disabling torque and stopping the loop when `tripped` is True."""

    def __init__(self, max_silence_s: float):
        self.max_silence_s = max_silence_s
        self.tripped = False
        self.trip_reason: str | None = None
        self._last_ok_time = time.monotonic()

    def feed(self) -> None:
        """Call after a successful sensor read + servo write this iteration."""
        self._last_ok_time = time.monotonic()

    def check(self, extra_ok: bool = True, reason: str = "") -> bool:
        """Returns True if healthy. `extra_ok=False` trips immediately regardless
        of comms timing (e.g. a NaN reading or an out-of-range value)."""
        if self.tripped:
            return False
        if not extra_ok:
            self.tripped = True
            self.trip_reason = reason or "sanity check failed"
            return False
        if time.monotonic() - self._last_ok_time > self.max_silence_s:
            self.tripped = True
            self.trip_reason = f"no successful iteration in {self.max_silence_s}s"
            return False
        return True


def ramp_to_target(
    bus: DynamixelBus,
    joint_mapping: JointMapping,
    target_sim: np.ndarray,
    duration_s: float,
    rate_hz: float,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> None:
    """Linearly interpolates from the robot's currently measured pose to
    `target_sim` (sim DOF order, radians) over `duration_s`, writing at `rate_hz`."""
    if duration_s <= 0:
        bus.write_goal_positions(joint_mapping.sim_target_to_ticks(target_sim))
        return

    n_steps = max(1, round(duration_s * rate_hz))
    start_sim = joint_mapping.ticks_to_sim_rad(bus.read_positions())
    period = 1.0 / rate_hz

    for step in range(1, n_steps + 1):
        alpha = step / n_steps
        interp_sim = start_sim + alpha * (target_sim - start_sim)
        bus.write_goal_positions(joint_mapping.sim_target_to_ticks(interp_sim))
        sleep_fn(period)


def soft_start_ramp(
    bus: DynamixelBus,
    joint_mapping: JointMapping,
    q_default_sim: np.ndarray,
    duration_s: float,
    rate_hz: float,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> None:
    ramp_to_target(bus, joint_mapping, q_default_sim, duration_s, rate_hz, sleep_fn)


def soft_stop_ramp(
    bus: DynamixelBus,
    joint_mapping: JointMapping,
    q_default_sim: np.ndarray,
    duration_s: float,
    rate_hz: float,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> None:
    ramp_to_target(bus, joint_mapping, q_default_sim, duration_s, rate_hz, sleep_fn)
