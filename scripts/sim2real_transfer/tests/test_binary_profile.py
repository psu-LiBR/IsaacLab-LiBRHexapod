# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import os

import numpy as np
import pytest
from sim2real.binary_profile import BinaryActionAdapter, BinaryActionCfg, BinarySpineCfg
from sim2real.deployment_config import load_deployment_config

_BINARY_CFG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "deployment.binary.example.yaml")

# sim DOF order used by the binary example config
_SIM_ORDER = ["BackLink", "FrontLink", "MiddleLeft", "MiddleRight", "BackLeft", "BackRight", "FrontLeft", "FrontRight"]
_LEG_ORDER = ["FrontRight", "FrontLeft", "MiddleRight", "MiddleLeft", "BackRight", "BackLeft"]

STANCE, LIFT = 0.46, 1.18


def _cfg():
    return BinaryActionCfg(
        stance_pos=STANCE,
        lift_pos=LIFT,
        leg_joint_order=list(_LEG_ORDER),
        spine=BinarySpineCfg(
            amplitude={"BackLink": -0.85, "FrontLink": -0.84},
            phase={"BackLink": -0.73, "FrontLink": 0.46},
            offset={"BackLink": 0.012, "FrontLink": -0.006},
            period=1.0,
        ),
    )


def test_all_stance_bits_map_every_leg_to_stance():
    adapter = BinaryActionAdapter(_cfg(), _SIM_ORDER)
    legs = adapter.leg_targets(np.ones(6))
    assert np.allclose(legs, STANCE)


def test_all_lift_bits_map_every_leg_to_lift():
    adapter = BinaryActionAdapter(_cfg(), _SIM_ORDER)
    assert np.allclose(adapter.leg_targets(-np.ones(6)), LIFT)


def test_bit_order_places_targets_at_the_right_sim_dof():
    adapter = BinaryActionAdapter(_cfg(), _SIM_ORDER)
    # only policy index 0 (FrontRight) in stance, rest lifted
    bits = np.array([1.0, -1.0, -1.0, -1.0, -1.0, -1.0])
    target = adapter.targets_sim(bits, t_seconds=0.0)
    fr = _SIM_ORDER.index("FrontRight")
    assert target[fr] == pytest.approx(STANCE)
    for leg in ("FrontLeft", "MiddleRight", "MiddleLeft", "BackRight", "BackLeft"):
        assert target[_SIM_ORDER.index(leg)] == pytest.approx(LIFT)


def test_spine_sinusoid_matches_formula_and_is_period_periodic():
    cfg = _cfg()
    adapter = BinaryActionAdapter(cfg, _SIM_ORDER)
    t = 0.137
    got = adapter.spine_targets(t)
    for i, name in enumerate(("BackLink", "FrontLink")):
        expect = cfg.spine.offset[name] + cfg.spine.amplitude[name] * math.sin(
            2 * math.pi * t / cfg.spine.period + cfg.spine.phase[name]
        )
        assert got[i] == pytest.approx(expect)
    assert np.allclose(adapter.spine_targets(0.0), adapter.spine_targets(cfg.spine.period))


def test_targets_sim_covers_all_eight_dofs():
    adapter = BinaryActionAdapter(_cfg(), _SIM_ORDER)
    target = adapter.targets_sim(np.array([1, -1, 1, -1, 1, -1.0]), t_seconds=0.25)
    assert target.shape == (8,)
    assert np.all(np.isfinite(target))
    # spine dofs hold the sinusoid, not a leg angle
    spine = adapter.spine_targets(0.25)
    assert target[_SIM_ORDER.index("BackLink")] == pytest.approx(spine[0])
    assert target[_SIM_ORDER.index("FrontLink")] == pytest.approx(spine[1])


def test_adapter_rejects_incomplete_dof_coverage():
    cfg = _cfg()
    with pytest.raises(ValueError, match="cover all"):
        # sim_order has a joint no leg/spine entry references
        BinaryActionAdapter(cfg, _SIM_ORDER + ["Extra"])


def test_binary_example_config_parses_and_builds_adapter():
    cfg = load_deployment_config(_BINARY_CFG_PATH)
    assert cfg.binary is not None
    adapter = BinaryActionAdapter(cfg.binary, cfg.joints.sim_order)
    target = adapter.targets_sim(np.ones(6), t_seconds=0.0)
    assert target.shape == (8,)
    # config's leg_joint_order must be exactly the 6 legs
    assert sorted(cfg.binary.leg_joint_order) == sorted(
        j for j in cfg.joints.sim_order if j not in ("BackLink", "FrontLink")
    )


def test_non_binary_example_config_has_no_binary_section():
    cfg = load_deployment_config(os.path.join(os.path.dirname(__file__), "..", "config", "deployment.example.yaml"))
    assert cfg.binary is None
