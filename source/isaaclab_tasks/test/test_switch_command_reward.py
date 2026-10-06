# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CPU tests for the optional per-flip contact-command penalty (needs ``torch``; no Isaac Sim)."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts/reinforcement_learning/binary_rl"))
from switch_command_reward import action_switch_count  # noqa: E402

STEP_DT = 0.02


def _env(action, prev_action, episode_length):
    return SimpleNamespace(
        action_manager=SimpleNamespace(action=torch.tensor(action), prev_action=torch.tensor(prev_action)),
        episode_length_buf=torch.tensor(episode_length),
        step_dt=STEP_DT,
    )


def test_counts_flipped_legs_per_second():
    env = _env(
        [[1.0, 1.0, 1.0, -1.0, -1.0, -1.0], [1.0, -1.0, 1.0, -1.0, 1.0, -1.0]],
        [[1.0, 1.0, 1.0, -1.0, -1.0, -1.0], [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0]],
        [10, 10],
    )
    # Env 0 flips nothing; env 1 flips all six legs. Output is flips / step_dt so the manager's * dt gives a count.
    assert action_switch_count(env).tolist() == pytest.approx([0.0, 6 / STEP_DT])


def test_first_steps_after_reset_are_not_penalized():
    flipped = [[1.0] * 6, [1.0] * 6]
    env = _env(flipped, [[-1.0] * 6, [-1.0] * 6], [0, 1])
    assert action_switch_count(env).tolist() == [0.0, 0.0]
    env = _env(flipped, [[-1.0] * 6, [-1.0] * 6], [2, 2])
    assert action_switch_count(env).tolist() == pytest.approx([6 / STEP_DT] * 2)


def test_equals_a_quarter_of_action_rate_l2_for_plus_minus_one_bits():
    # action_rate_l2 is sum((a - a_prev)^2); a +-1 flip contributes (+-2)^2 = 4, so the two terms carry one signal.
    action = torch.tensor([[1.0, -1.0, 1.0, -1.0, 1.0, -1.0]])
    prev = torch.tensor([[-1.0, -1.0, -1.0, 1.0, 1.0, -1.0]])
    env = _env(action.tolist(), prev.tolist(), [10])
    action_rate_l2 = torch.sum(torch.square(action - prev), dim=1)
    assert (action_rate_l2 / 4).item() == 3.0
    assert action_switch_count(env).item() * STEP_DT == pytest.approx((action_rate_l2 / 4).item())
