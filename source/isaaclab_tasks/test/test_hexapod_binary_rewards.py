# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch

HEXAPOD_DIR = Path(__file__).parents[1] / "isaaclab_tasks" / "contrib" / "velocity" / "config" / "hexapod"


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HEXAPOD_DIR / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


binary_rewards = load_module("hexapod_binary_rewards", "hexapod_binary_rewards.py")


class FakeActionManager:
    def __init__(self, action: torch.Tensor):
        self.action = action


def _stance_env(rows: list[list[float]]) -> SimpleNamespace:
    return SimpleNamespace(action_manager=FakeActionManager(torch.tensor(rows, dtype=torch.float32)))


def _penalty(rows: list[list[float]], **kwargs) -> list[float]:
    return binary_rewards.min_stance_legs_penalty(_stance_env(rows), **kwargs).tolist()


def test_min_stance_all_lift_penalizes_three():
    assert _penalty([[-1.0] * 6]) == [3.0]


def test_min_stance_one_and_two_down_penalize_deficit():
    assert _penalty([[1.0, -1.0, -1.0, -1.0, -1.0, -1.0]]) == [2.0]
    assert _penalty([[-1.0, 1.0, -1.0, -1.0, 1.0, -1.0]]) == [1.0]


def test_min_stance_zero_counts_as_stance():
    # BinaryJointAction.process_actions lifts only when a < 0, so exactly 0 (and NaN) is stance in the sim
    assert _penalty([[0.0] * 6]) == [0.0]
    assert _penalty([[-1.0, -1.0, -1.0, -1.0, 0.0, 0.0]]) == [1.0]
    assert _penalty([[float("nan")] * 6]) == [0.0]


def test_min_stance_matches_sim_lift_rule():
    values = torch.tensor([[-1.0, -1e-6, 0.0, 1e-6, 1.0, float("nan")]])
    sim_lift = values < 0  # BinaryJointAction.process_actions: close (lift) iff action < 0
    n_stance = int((~sim_lift).sum())
    expected = float(max(3 - n_stance, 0))
    assert binary_rewards.min_stance_legs_penalty(
        SimpleNamespace(action_manager=FakeActionManager(values))
    ).tolist() == [expected]


def test_min_stance_reference_tripod_patterns_incur_zero():
    bam = _load_binary_action_mask()
    for index in (25, 38):
        bits = bam.action_to_pm1(torch.tensor([index]), 6)
        assert int((bits > 0).sum()) == 3
        assert binary_rewards.min_stance_legs_penalty(
            SimpleNamespace(action_manager=FakeActionManager(bits))
        ).tolist() == [0.0]


def test_min_stance_four_five_six_down_incur_zero():
    for n_down in (4, 5, 6):
        row = [1.0] * n_down + [-1.0] * (6 - n_down)
        assert _penalty([row]) == [0.0]


def test_min_stance_min_legs_changes_threshold():
    assert _penalty([[1.0, 1.0, 1.0, -1.0, -1.0, -1.0]], min_legs=4) == [1.0]
    assert _penalty([[-1.0] * 6], min_legs=4) == [4.0]
    assert _penalty([[1.0] * 4 + [-1.0] * 2], min_legs=4) == [0.0]


def test_min_stance_power_squares_deficit():
    assert _penalty([[-1.0] * 6], power=2.0) == [9.0]
    assert _penalty([[1.0, -1.0, -1.0, -1.0, -1.0, -1.0]], power=2.0) == [4.0]


def test_min_stance_ignores_trailing_columns():
    assert _penalty([[-1.0] * 6 + [1.0, 1.0]]) == [3.0]
    assert _penalty([[1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, 1.0]]) == [0.0]


def test_min_stance_batch_returns_per_env_float_values():
    out = binary_rewards.min_stance_legs_penalty(
        _stance_env([[-1.0] * 6, [1.0, -1.0, -1.0, -1.0, -1.0, -1.0], [1.0] * 3 + [-1.0] * 3, [1.0] * 6])
    )
    assert out.dtype == torch.float32 and out.shape == (4,)
    torch.testing.assert_close(out, torch.tensor([3.0, 2.0, 0.0, 0.0]))


def _load_binary_action_mask():
    import sys

    binary_rl_dir = Path(__file__).parents[3] / "scripts" / "reinforcement_learning" / "binary_rl"
    sys.path.insert(0, str(binary_rl_dir))
    spec = importlib.util.spec_from_file_location("binary_action_mask", binary_rl_dir / "binary_action_mask.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
