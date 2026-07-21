# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import time

import numpy as np
from sim2real.deployment_config import JointsCfg
from sim2real.dynamixel_bus import DryRunDynamixelBus
from sim2real.joint_mapping import JointMapping
from sim2real.safety import (
    Watchdog,
    all_finite,
    ramp_to_target,
    soft_start_ramp,
    soft_stop_ramp,
    ticks_in_valid_range,
)

SIM_ORDER = [
    "BackLink",
    "FrontLink",
    "MiddleLeft",
    "MiddleRight",
    "BackLeft",
    "BackRight",
    "FrontLeft",
    "FrontRight",
]
REAL_ORDER = [
    "FrontLink",
    "BackLink",
    "FrontRight",
    "FrontLeft",
    "MiddleRight",
    "MiddleLeft",
    "BackRight",
    "BackLeft",
]
MOTOR_IDS = {
    "FrontLink": 3,
    "BackLink": 6,
    "FrontRight": 2,
    "FrontLeft": 1,
    "MiddleRight": 4,
    "MiddleLeft": 5,
    "BackRight": 7,
    "BackLeft": 8,
}
CORRECTION_GROUP = {
    "BackLink": "unchanged",
    "FrontLink": "negate",
    "FrontRight": "leg_negate_plus_pi",
    "FrontLeft": "leg_negate_plus_pi",
    "MiddleRight": "leg_negate_plus_pi",
    "MiddleLeft": "leg_negate_plus_pi",
    "BackRight": "leg_negate_plus_pi",
    "BackLeft": "leg_negate_plus_pi",
}
# NOTE: 2048 is only a self-consistent placeholder for exercising the ramp math in
# this test, not a claim about the real robot's calibration -- see
# joint_mapping tests + CLAUDE.md for why leg zero_tick is a "# CALIBRATE" item.
# Body joints keep zero_tick=2048 (confirmed by imu_processor.py/gait_generator.py);
# legs use a different placeholder here specifically so that q_default=-0.47 rad,
# once run through leg_negate_plus_pi (real_rad = 0.47+pi ~= 3.61 rad), lands
# inside the valid 12-bit tick range instead of overflowing it.
ZERO_TICK = {name: 2048 for name in REAL_ORDER}
for _leg in ("FrontRight", "FrontLeft", "MiddleRight", "MiddleLeft", "BackRight", "BackLeft"):
    ZERO_TICK[_leg] = 1000
SOFT_LIMITS = {name: (-1.4, 1.4) for name in SIM_ORDER}


def make_mapping() -> JointMapping:
    cfg = JointsCfg(
        sim_order=SIM_ORDER,
        real_order=REAL_ORDER,
        motor_ids=MOTOR_IDS,
        correction_group=CORRECTION_GROUP,
        ticks_per_rev=4096,
        zero_tick=ZERO_TICK,
        soft_limits_rad=SOFT_LIMITS,
    )
    return JointMapping(cfg)


def test_all_finite():
    assert all_finite(np.array([1.0, 2.0]), np.array([3.0]))
    assert not all_finite(np.array([1.0, float("nan")]))
    assert not all_finite(np.array([1.0, float("inf")]))


def test_watchdog_healthy_when_fed():
    wd = Watchdog(max_silence_s=1.0)
    wd.feed()
    assert wd.check() is True
    assert wd.tripped is False


def test_watchdog_trips_on_silence():
    wd = Watchdog(max_silence_s=0.01)
    wd.feed()
    time.sleep(0.05)
    assert wd.check() is False
    assert wd.tripped is True


def test_watchdog_trips_on_sanity_failure():
    wd = Watchdog(max_silence_s=10.0)
    wd.feed()
    assert wd.check(extra_ok=False, reason="NaN in obs") is False
    assert wd.trip_reason == "NaN in obs"


def test_watchdog_stays_tripped():
    wd = Watchdog(max_silence_s=10.0)
    wd.check(extra_ok=False)
    wd.feed()
    assert wd.check() is False  # once tripped, stays tripped


def test_ticks_in_valid_range():
    assert ticks_in_valid_range(np.array([0, 2048, 4095]), ticks_per_rev=4096)
    assert not ticks_in_valid_range(np.array([0, 4402, 100]), ticks_per_rev=4096)
    assert not ticks_in_valid_range(np.array([-1, 100]), ticks_per_rev=4096)


def test_zero_tick_2048_for_legs_overflows_valid_range_reproducing_calibration_bug():
    # Documents the bug the ramp test below works around: a naive zero_tick=2048
    # for legs combined with leg_negate_plus_pi puts q_default well outside the
    # valid 12-bit tick range, which is exactly the kind of thing
    # ticks_in_valid_range() is meant to catch before it gets silently clipped.
    cfg = JointsCfg(
        sim_order=SIM_ORDER,
        real_order=REAL_ORDER,
        motor_ids=MOTOR_IDS,
        correction_group=CORRECTION_GROUP,
        ticks_per_rev=4096,
        zero_tick={name: 2048 for name in REAL_ORDER},
        soft_limits_rad=SOFT_LIMITS,
    )
    jm = JointMapping(cfg)
    q_default = np.full(8, -0.47)
    q_default[SIM_ORDER.index("BackLink")] = 0.0
    q_default[SIM_ORDER.index("FrontLink")] = 0.0
    ticks = jm.sim_target_to_ticks(q_default)
    assert not ticks_in_valid_range(ticks, ticks_per_rev=4096)


def test_ramp_to_target_converges_and_writes_expected_step_count():
    jm = make_mapping()
    bus = DryRunDynamixelBus(jm.motor_ids, initial_ticks=np.full(8, 2048))
    target_sim = np.full(8, -0.47)
    target_sim[SIM_ORDER.index("BackLink")] = 0.0
    target_sim[SIM_ORDER.index("FrontLink")] = 0.0

    sleeps = []
    ramp_to_target(bus, jm, target_sim, duration_s=1.0, rate_hz=50.0, sleep_fn=sleeps.append)

    assert len(sleeps) == 50
    final_sim = jm.ticks_to_sim_rad(bus.read_positions())
    tol = (2 * math.pi / 4096) * 1.0001
    assert np.max(np.abs(final_sim - target_sim)) < tol


def test_ramp_to_target_zero_duration_jumps_immediately():
    jm = make_mapping()
    bus = DryRunDynamixelBus(jm.motor_ids, initial_ticks=np.full(8, 2048))
    target_sim = np.zeros(8)
    ramp_to_target(bus, jm, target_sim, duration_s=0.0, rate_hz=50.0, sleep_fn=lambda s: None)
    final_sim = jm.ticks_to_sim_rad(bus.read_positions())
    tol = (2 * math.pi / 4096) * 1.0001
    assert np.max(np.abs(final_sim - target_sim)) < tol


def test_soft_start_and_stop_ramps_use_same_target():
    jm = make_mapping()
    bus = DryRunDynamixelBus(jm.motor_ids, initial_ticks=np.full(8, 1000))
    q_default = np.zeros(8)

    soft_start_ramp(bus, jm, q_default, duration_s=0.2, rate_hz=50.0, sleep_fn=lambda s: None)
    after_start = jm.ticks_to_sim_rad(bus.read_positions())

    bus.write_goal_positions(np.full(8, 3000))  # simulate the policy moving the robot
    soft_stop_ramp(bus, jm, q_default, duration_s=0.2, rate_hz=50.0, sleep_fn=lambda s: None)
    after_stop = jm.ticks_to_sim_rad(bus.read_positions())

    tol = (2 * math.pi / 4096) * 1.0001
    assert np.max(np.abs(after_start - q_default)) < tol
    assert np.max(np.abs(after_stop - q_default)) < tol
