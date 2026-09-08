# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import os

import numpy as np
import pytest
from sim2real.deployment_config import JointsCfg, load_deployment_config
from sim2real.joint_mapping import JointMapping, ordered_array

_EXAMPLE_CFG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "deployment.example.yaml")

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
ZERO_TICK = {name: 2048 for name in REAL_ORDER}
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


def test_permutation_round_trips():
    jm = make_mapping()
    identity = np.arange(len(SIM_ORDER))
    assert np.array_equal(jm.sim_to_real_idx[jm.real_to_sim_idx], identity)
    assert np.array_equal(jm.real_to_sim_idx[jm.sim_to_real_idx], identity)


def test_reorder_matches_expected_names():
    jm = make_mapping()
    # Encode each sim-order joint's index as its value, so we can read off
    # which physical joint ended up at each real-order slot.
    sim_rad = np.arange(len(SIM_ORDER), dtype=np.float64)
    real_reordered = sim_rad[jm.sim_to_real_idx]
    for i, name in enumerate(REAL_ORDER):
        assert real_reordered[i] == SIM_ORDER.index(name)


def test_correction_invertible():
    jm = make_mapping()
    rng = np.random.default_rng(0)
    sim_rad = rng.uniform(-1.0, 1.0, size=len(SIM_ORDER))
    real_rad = jm.sim_rad_to_real_rad(sim_rad)
    sim_rad_reconstructed = jm.real_rad_to_sim_rad(real_rad)
    assert np.allclose(sim_rad, sim_rad_reconstructed)


def test_leg_correction_formula():
    jm = make_mapping()
    sim_rad = np.zeros(len(SIM_ORDER))
    sim_rad[SIM_ORDER.index("FrontLeft")] = -0.47
    real_rad = jm.sim_rad_to_real_rad(sim_rad)
    expected = -(-0.47) + math.pi
    assert math.isclose(real_rad[REAL_ORDER.index("FrontLeft")], expected)


def test_leg_negate_minus_pi_formula():
    cfg = JointsCfg(
        sim_order=SIM_ORDER,
        real_order=REAL_ORDER,
        motor_ids=MOTOR_IDS,
        correction_group={**CORRECTION_GROUP, "FrontLeft": "leg_negate_minus_pi"},
        ticks_per_rev=4096,
        zero_tick=ZERO_TICK,
        soft_limits_rad=SOFT_LIMITS,
    )
    jm = JointMapping(cfg)
    sim_rad = np.zeros(len(SIM_ORDER))
    sim_rad[SIM_ORDER.index("FrontLeft")] = -0.47
    real_rad = jm.sim_rad_to_real_rad(sim_rad)
    expected = -(-0.47) - math.pi
    assert math.isclose(real_rad[REAL_ORDER.index("FrontLeft")], expected)
    # Same physical angle as leg_negate_plus_pi's real_rad, just the other
    # 2*pi-equivalent phase branch.
    assert math.isclose(expected + 2 * math.pi, -(-0.47) + math.pi)


def test_leg_minus_pi_formula():
    # HexapI (binary-profile) leg map: real = sim - pi. It is leg_negate_minus_pi
    # (real = -sim_old - pi) rewritten for the HexapI leg-sign flip (sim = -sim_old),
    # so it lands on the same physical angle / same zero_tick as the calibrated
    # velocity/goal profile.
    cfg = JointsCfg(
        sim_order=SIM_ORDER,
        real_order=REAL_ORDER,
        motor_ids=MOTOR_IDS,
        correction_group={**CORRECTION_GROUP, "FrontLeft": "leg_minus_pi"},
        ticks_per_rev=4096,
        zero_tick=ZERO_TICK,
        soft_limits_rad=SOFT_LIMITS,
    )
    jm = JointMapping(cfg)
    sim_rad = np.zeros(len(SIM_ORDER))
    sim_rad[SIM_ORDER.index("FrontLeft")] = 0.47  # HexapI leg rest pose (+0.47, not -0.47)
    real_rad = jm.sim_rad_to_real_rad(sim_rad)
    expected = 0.47 - math.pi
    assert math.isclose(real_rad[REAL_ORDER.index("FrontLeft")], expected)
    # Identical real angle to the old calibrated map at the same physical rest pose.
    assert math.isclose(expected, -(-0.47) - math.pi)
    # Rate: b has no effect on a derivative and a = +1, so velocity passes through.
    sim_vel = np.zeros(len(SIM_ORDER))
    sim_vel[SIM_ORDER.index("FrontLeft")] = 1.3
    real_vel = jm.sim_radps_to_real_radps(sim_vel)
    assert math.isclose(real_vel[REAL_ORDER.index("FrontLeft")], 1.3)


def test_body_correction_formula():
    jm = make_mapping()
    sim_rad = np.zeros(len(SIM_ORDER))
    sim_rad[SIM_ORDER.index("FrontLink")] = 0.3
    sim_rad[SIM_ORDER.index("BackLink")] = 0.3
    real_rad = jm.sim_rad_to_real_rad(sim_rad)
    assert math.isclose(real_rad[REAL_ORDER.index("FrontLink")], -0.3)
    assert math.isclose(real_rad[REAL_ORDER.index("BackLink")], 0.3)


def test_velocity_conversion_round_trip():
    jm = make_mapping()
    rng = np.random.default_rng(2)
    sim_radps = rng.uniform(-5.0, 5.0, size=len(SIM_ORDER))
    real_radps = jm.sim_radps_to_real_radps(sim_radps)
    sim_radps_reconstructed = jm.real_radps_to_sim_radps(real_radps)
    assert np.allclose(sim_radps, sim_radps_reconstructed)


def test_velocity_conversion_sign_matches_position_correction():
    jm = make_mapping()
    sim_radps = np.zeros(len(SIM_ORDER))
    sim_radps[SIM_ORDER.index("FrontLeft")] = 1.0
    real_radps = jm.sim_radps_to_real_radps(sim_radps)
    # leg_negate_plus_pi has a=-1, so velocity should flip sign (no +pi offset)
    assert math.isclose(real_radps[REAL_ORDER.index("FrontLeft")], -1.0)


def test_tick_round_trip():
    jm = make_mapping()
    rng = np.random.default_rng(1)
    real_rad = rng.uniform(-1.0, 1.0, size=len(SIM_ORDER))
    ticks = jm.real_rad_to_ticks(real_rad)
    real_rad_reconstructed = jm.ticks_to_real_rad(ticks)
    # tolerance = one tick's worth of angle (quantization)
    tol = (2 * math.pi / 4096) * 1.0001
    assert np.max(np.abs(real_rad - real_rad_reconstructed)) < tol


def test_end_to_end_sim_to_ticks_and_back():
    jm = make_mapping()
    q_default = ordered_array(
        {
            "BackLink": 0.0,
            "FrontLink": 0.0,
            "MiddleLeft": -0.47,
            "MiddleRight": -0.47,
            "BackLeft": -0.47,
            "BackRight": -0.47,
            "FrontLeft": -0.47,
            "FrontRight": -0.47,
        },
        SIM_ORDER,
    )
    ticks = jm.sim_target_to_ticks(q_default)
    sim_rad_reconstructed = jm.ticks_to_sim_rad(ticks)
    tol = (2 * math.pi / 4096) * 1.0001
    assert np.max(np.abs(q_default - sim_rad_reconstructed)) < tol


def test_clip_to_soft_limits():
    jm = make_mapping()
    sim_rad = np.full(len(SIM_ORDER), 10.0)
    clipped = jm.clip_to_soft_limits(sim_rad)
    assert np.all(clipped <= jm.soft_limits_sim[:, 1])


def test_mismatched_joint_names_raise():
    cfg = JointsCfg(
        sim_order=SIM_ORDER,
        real_order=SIM_ORDER[:-1] + ["NotAJoint"],
        motor_ids=MOTOR_IDS,
        correction_group=CORRECTION_GROUP,
        ticks_per_rev=4096,
        zero_tick=ZERO_TICK,
        soft_limits_rad=SOFT_LIMITS,
    )
    with pytest.raises(ValueError):
        JointMapping(cfg)


def test_example_yaml_loads_and_builds_mapping():
    cfg = load_deployment_config(_EXAMPLE_CFG_PATH)
    jm = JointMapping(cfg.joints)
    assert len(jm.sim_order) == 8
    assert len(jm.motor_ids) == 8
