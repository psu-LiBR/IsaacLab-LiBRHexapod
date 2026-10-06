# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CPU regressions for the per-leg, per-spine-cycle contact-command switch metric (numpy only, no Isaac Sim)."""

import math
import re
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPTS = REPO / "scripts/reinforcement_learning/binary_rl"
sys.path.insert(0, str(SCRIPTS))
import switch_metrics  # noqa: E402
from switch_metrics import summarize  # noqa: E402

BODY_LENGTH = 0.315


def _summarize(actions, phase, bl_per_cycle=0.7, heading=0.0, **kwargs):
    """Run :func:`summarize` on a straight, fall-free rollout, one env per column of ``actions``."""
    T, E = phase.shape
    steps = switch_metrics.DEFAULT_CONTROL_DT
    forward = np.linspace(0, bl_per_cycle * BODY_LENGTH * T * steps / switch_metrics.DEFAULT_PERIOD_S, T)
    trajectory = np.zeros((T, E, 3))
    trajectory[:, :, 0] = forward[:, None] * math.cos(heading)
    trajectory[:, :, 1] = forward[:, None] * math.sin(heading)
    quat = np.tile(np.array([0, 0, math.sin(heading / 2), math.cos(heading / 2)]), (E, 1))
    no_contact = np.zeros((T, E), dtype=bool)
    return summarize(
        actions,
        np.zeros(E, dtype=np.int64),
        phase,
        trajectory,
        quat,
        np.tile(quat, (T, 1, 1)),
        np.ones((T, E)),
        no_contact,
        no_contact,
        **kwargs,
    )


def _phase(steps, envs=1):
    """Phase counter 1..steps (the first step after reset is not measured), shape [steps, envs]."""
    return np.tile(np.arange(1, steps + 1)[:, None], (1, envs))


def _toggle_actions(phase, toggle_phases_by_leg):
    """Action index per step for legs that toggle whenever ``phase % 50`` is in the leg's toggle set.

    Starts every leg at bit 0 (matching ``previous=0``); legs not listed stay frozen at 0.
    """
    actions = np.zeros(phase.shape, dtype=np.int64)
    for leg, toggle_set in toggle_phases_by_leg.items():
        toggled = np.isin(phase % 50, list(toggle_set))
        actions |= (np.cumsum(toggled, axis=0) % 2).astype(np.int64) << leg
    return actions


@pytest.mark.parametrize("heading", [0.0, math.pi / 2])
def test_complete_cycles_and_initial_heading_projection(heading):
    phase = _phase(300)
    actions = ((phase // 25) % 2).astype(np.int64)
    result = _summarize(actions, phase, heading=heading)
    assert result["complete_cycles"] == 5
    assert result["per_leg_max"] == [2, 0, 0, 0, 0, 0]
    assert result["dwell_min_s"] == pytest.approx(0.5)
    assert result["forward_bl_per_cycle"] == pytest.approx(0.7)
    assert result["lateral_abs_max_bl"] == pytest.approx(0, abs=1e-12)
    assert result["yaw_abs_max_deg"] == pytest.approx(0, abs=1e-12)


def test_default_limit_is_five_switches_per_leg_per_cycle():
    phase = _phase(300)
    five = _summarize(_toggle_actions(phase, {0: {5, 15, 25, 35, 45}}), phase)
    assert five["per_leg_max"][0] == 5
    assert five["max_switches_per_leg_cycle"] == 5
    assert five["basic_screen_pass"], five["rejection_reasons"]
    assert five["leg_cycle_fraction_within_limit"] == 1.0

    six = _summarize(_toggle_actions(phase, {0: {5, 12, 20, 28, 36, 44}}), phase)
    assert six["per_leg_max"][0] == 6
    assert not six["basic_screen_pass"]
    assert six["rejection_reasons"] == ["per_leg_cycle_switch_exceeds_5"]
    assert six["all_legs_cycle_fraction_within_limit"] == 0.0


def test_limit_is_configurable():
    phase = _phase(300)
    actions = _toggle_actions(phase, {0: {5, 12, 20, 28, 36, 44}})
    assert _summarize(actions, phase, max_switches_per_leg_cycle=6)["basic_screen_pass"]
    result = _summarize(actions, phase, max_switches_per_leg_cycle=4)
    assert result["rejection_reasons"] == ["per_leg_cycle_switch_exceeds_4"]


def test_frozen_legs_are_flagged_but_do_not_fail_the_screen():
    phase = _phase(300)
    # Only legs 0 and 3 ever switch; legs 1, 2, 4, 5 hold bit 0 (lifted) for the whole run.
    result = _summarize(_toggle_actions(phase, {0: {10, 35}, 3: {10, 35}}), phase)
    assert result["frozen_legs"] == [1, 2, 4, 5]
    assert result["warnings"] == ["frozen_legs"]
    assert result["per_leg_frozen_env_fraction"] == [0.0, 1.0, 1.0, 0.0, 1.0, 1.0]
    assert result["per_leg_stance_fraction"][1] == 0.0
    assert result["basic_screen_pass"]


def test_leg_frozen_in_only_some_envs_is_not_a_frozen_leg():
    phase = _phase(300, envs=2)
    actions = _toggle_actions(phase, {0: {10, 35}})
    actions[:, 1] = 0  # env 1 never switches leg 0
    result = _summarize(actions, phase)
    assert result["per_leg_frozen_env_fraction"][0] == 0.5
    assert 0 not in result["frozen_legs"]


def test_cycle_length_follows_the_gait_period_and_control_step():
    phase = _phase(300)
    actions = ((phase // 25) % 2).astype(np.int64)
    assert _summarize(actions, phase)["period_steps"] == 50
    half = _summarize(actions, phase, period=0.5)
    assert half["period_steps"] == 25
    assert half["complete_cycles"] == 11  # steps 25..299 give 11 complete 25-step cycles
    with pytest.raises(ValueError, match="whole number"):
        _summarize(actions, phase, period=0.03)


def test_censored_dwell_spans_count_one_for_a_leg_that_never_switches():
    phase = _phase(300)
    # Leg 0 switches 12 times (2 censored end spans, 11 complete dwells); legs 1-5 never switch (1 censored span each).
    result = _summarize(_toggle_actions(phase, {0: {10, 20}}), phase)
    assert result["dwell_complete_count"] == 11  # toggles at phases 10, 20, 60, 70, ... -> 12 switches, 11 gaps
    assert result["dwell_censored_count"] == 2 + 5


def test_defaults_match_the_binary_env_timing():
    """Cycle = GAIT_PERIOD_S / (decimation * sim.dt) control steps; fail if the env timing drifts from the metric."""
    binary_cfg = (
        REPO / "source/isaaclab_tasks/isaaclab_tasks/contrib/velocity/config/hexapod/hexapod_binary_env_cfg.py"
    ).read_text()
    core_cfg = (REPO / "source/isaaclab_tasks/isaaclab_tasks/core/velocity/velocity_env_cfg.py").read_text()
    period = float(re.search(r"^GAIT_PERIOD_S\s*=\s*([0-9.]+)", binary_cfg, re.M).group(1))
    decimation = int(re.search(r"self\.decimation\s*=\s*(\d+)", core_cfg).group(1))
    sim_dt = float(re.search(r"self\.sim\.dt\s*=\s*([0-9.]+)", core_cfg).group(1))
    assert period == pytest.approx(switch_metrics.DEFAULT_PERIOD_S)
    assert decimation * sim_dt == pytest.approx(switch_metrics.DEFAULT_CONTROL_DT)
    assert round(period / (decimation * sim_dt)) == 50
