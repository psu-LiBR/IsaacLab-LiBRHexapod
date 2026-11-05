# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the mimic-env reset event and reward-weight ramp.

Loads the hexapod modules through a synthetic package with ``isaaclab.managers`` stubbed, so only ``torch`` and
``numpy`` are needed (plain system Python works).
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

HEXAPOD_DIR = Path(__file__).parents[1] / "isaaclab_tasks" / "contrib" / "velocity" / "config" / "hexapod"
PKG = "_hexapod_mimic_under_test"


def _install_stubs() -> None:
    managers = types.ModuleType("isaaclab.managers")
    managers.SceneEntityCfg = lambda name, **kw: SimpleNamespace(name=name, joint_ids=slice(None))
    managers.ManagerTermBase = type("ManagerTermBase", (), {"__init__": lambda self, cfg, env: None})
    managers.CurriculumTermCfg = object
    sys.modules["isaaclab"] = types.ModuleType("isaaclab")
    sys.modules["isaaclab.managers"] = managers
    pkg = types.ModuleType(PKG)
    pkg.__path__ = [str(HEXAPOD_DIR)]
    sys.modules[PKG] = pkg


_saved = {k: sys.modules.get(k) for k in ("isaaclab", "isaaclab.managers")}
_install_stubs()
try:
    rewards = importlib.import_module(f"{PKG}.hexapod_mimic_rewards")
    curriculum = importlib.import_module(f"{PKG}.hexapod_mimic_curriculum")
    motion = importlib.import_module(f"{PKG}.hexapod_mimic_motion")
finally:  # do not leak the stubs into other test modules
    for _k, _v in _saved.items():
        if _v is None:
            sys.modules.pop(_k, None)
        else:
            sys.modules[_k] = _v


class FakeRewardManager:
    def __init__(self, term_cfg):
        self.term_cfg = term_cfg

    def get_term_cfg(self, _name):
        return self.term_cfg

    def set_term_cfg(self, _name, cfg):
        self.term_cfg = cfg


def _ramp(start_weight: float, end_weight: float, start_step: int, end_step: int):
    term_cfg = SimpleNamespace(weight=start_weight)
    env = SimpleNamespace(reward_manager=FakeRewardManager(term_cfg), common_step_counter=0)
    term = curriculum.ramp_reward_weight(SimpleNamespace(params={"term_name": "x"}), env)

    def at(step: int) -> float:
        env.common_step_counter = step
        return term(env, [0], "x", end_weight, start_step, end_step)

    return at


def test_ramp_holds_then_interpolates_then_stays():
    at = _ramp(3.0, 0.0, 1000, 2000)
    assert at(0) == 3.0
    assert at(1000) == 3.0
    assert abs(at(1500) - 1.5) < 1e-9
    assert at(2000) == 0.0
    assert at(10_000) == 0.0


def test_ramp_has_no_jump_at_start():
    at = _ramp(0.25, 1.0, 1000, 2000)
    assert at(1000) == 0.25
    assert 0.25 < at(1001) < 0.2508  # 0.75 / 1000 per step


class FakeAsset:
    def __init__(self, joint_names: list[str]):
        n = len(joint_names)
        self.data = SimpleNamespace(
            joint_names=joint_names,
            joint_pos=SimpleNamespace(torch=torch.zeros(4, n)),
            soft_joint_pos_limits=SimpleNamespace(torch=torch.tensor([-3.0, 3.0]).expand(4, n, 2).clone()),
            soft_joint_vel_limits=SimpleNamespace(torch=torch.full((4, n), 100.0)),
        )
        self.written: dict[str, torch.Tensor] = {}

    def write_joint_position_to_sim_index(self, position, joint_ids, env_ids):
        self.written["pos"] = position

    def write_joint_velocity_to_sim_index(self, velocity, joint_ids, env_ids):
        self.written["vel"] = velocity


def _reset(joint_names: list[str], lag, random_phase: bool = False, return_env: bool = False):
    asset = FakeAsset(joint_names)
    env = SimpleNamespace(scene={"robot": asset}, step_dt=0.02, episode_length_buf=torch.zeros(4))
    rewards._JOINT_IDX_CACHE.clear()
    rewards.reset_joints_to_mimic_reference(
        env,
        torch.arange(4),
        csv_path=None,
        gait_period=1.0,
        csv_col_order=None,
        servo_lag_s=lag,
        random_phase=random_phase,
    )
    return (asset, env) if return_env else asset.written


def test_reset_matches_reward_reference_at_counter_zero():
    names = list(motion.JOINT_NAMES)
    written = _reset(names, lag=0.06)
    ref = motion.MotionReference(csv_path=None, gait_period=1.0)
    phase = ref.get_phase(torch.tensor([-0.06]))
    expected = ref.get_reference(phase)[0]
    assert torch.allclose(written["pos"][0], expected)
    assert written["vel"].shape == (4, 8)
    assert np.isfinite(written["vel"].numpy()).all()


def test_reset_handles_reordered_asset_joints():
    names = list(motion.JOINT_NAMES)
    perm = [3, 0, 7, 1, 6, 2, 5, 4]
    shuffled = [names[i] for i in perm]  # asset column k holds Sim-DOF joint perm[k]
    in_sim_order = _reset(names, lag=0.0)["pos"]
    in_asset_order = _reset(shuffled, lag=0.0)["pos"]
    assert torch.allclose(in_asset_order, in_sim_order[:, perm])


def test_missing_reference_csv_fails_loudly():
    with pytest.raises(FileNotFoundError):
        motion.MotionReference(csv_path="does/not/exist.csv", gait_period=1.0)


LAG_LIST = [0.10, 0.10, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08]


def _imitation(asset, env, lag, buf: int) -> torch.Tensor:
    """Per-env imitation reward with the asset pose as written by the reset event."""
    asset.data.joint_pos.torch = asset.written["pos"].clone()
    env.episode_length_buf = torch.full((4,), buf)
    return rewards.joint_pos_imitation(env, csv_path=None, gait_period=1.0, joint_sigma=0.25, servo_lag_s=lag)


@pytest.mark.parametrize("lag", [0.06, LAG_LIST])
def test_random_phase_reset_sits_exactly_on_reference_at_counter_zero(lag):
    torch.manual_seed(0)
    asset, env = _reset(list(motion.JOINT_NAMES), lag, random_phase=True, return_env=True)
    offsets = env._mimic_clock_offset
    assert offsets.shape == (4,)
    assert ((offsets >= 0) & (offsets < 1.0)).all()
    assert offsets.unique().numel() == 4  # different offset per env
    assert torch.allclose(_imitation(asset, env, lag, buf=0), torch.ones(4), atol=1e-6)
    # one control step later the reference has moved on, so the (static) pose no longer sits exactly on it
    ref = motion.MotionReference(csv_path=None, gait_period=1.0)
    lag_t = rewards._lag_tensor(lag, "cpu")
    now = ref.get_reference_per_joint(ref.get_phase(offsets[:, None] - lag_t[None]))
    later = ref.get_reference_per_joint(ref.get_phase(offsets[:, None] + env.step_dt - lag_t[None]))
    assert not torch.allclose(now, later)
    r1 = _imitation(asset, env, lag, buf=1)
    expected = torch.exp(-((asset.written["pos"] - later) ** 2) / 0.25**2).mean(-1)
    assert torch.allclose(r1, expected, atol=1e-6)
    assert (r1 < 1.0).all()


def test_fixed_phase_reset_writes_zero_offset():
    asset, env = _reset(list(motion.JOINT_NAMES), 0.06, random_phase=False, return_env=True)
    assert torch.equal(env._mimic_clock_offset, torch.zeros(4))
    assert torch.allclose(_imitation(asset, env, 0.06, buf=0), torch.ones(4), atol=1e-6)


def test_fixed_phase_reset_clears_previous_random_offsets():
    asset, env = _reset(list(motion.JOINT_NAMES), 0.0, random_phase=True, return_env=True)
    assert (env._mimic_clock_offset != 0).any()
    rewards.reset_joints_to_mimic_reference(env, torch.arange(4), csv_path=None, gait_period=1.0, random_phase=False)
    assert torch.equal(env._mimic_clock_offset, torch.zeros(4))


def test_reward_without_reset_event_uses_zero_offset():
    asset = FakeAsset(list(motion.JOINT_NAMES))
    env = SimpleNamespace(scene={"robot": asset}, step_dt=0.02, episode_length_buf=torch.zeros(4))
    rewards._JOINT_IDX_CACHE.clear()
    r = rewards.joint_pos_imitation(env, csv_path=None, gait_period=1.0, servo_lag_s=0.0)
    assert r.shape == (4,)


def test_per_joint_lag_matches_scalar_lag_per_joint():
    ref = motion.MotionReference(csv_path=None, gait_period=1.0)
    clock = torch.tensor([0.0, 0.13, 0.5, 0.97])
    per_joint = rewards._reference_at(ref, clock, rewards._lag_tensor(LAG_LIST, "cpu"))
    for j, lag_j in enumerate(LAG_LIST):
        scalar = rewards._reference_at(ref, clock, rewards._lag_tensor(lag_j, "cpu"))
        assert torch.equal(per_joint[:, j], scalar[:, j])
    # the scalar path still equals the old single-phase computation
    old = ref.get_reference(ref.get_phase(clock - 0.06))
    assert torch.equal(rewards._reference_at(ref, clock, rewards._lag_tensor(0.06, "cpu")), old)


def test_per_joint_lag_reset_places_each_joint_at_its_own_time():
    written = _reset(list(motion.JOINT_NAMES), LAG_LIST)
    ref = motion.MotionReference(csv_path=None, gait_period=1.0)
    for j, lag_j in enumerate(LAG_LIST):
        expected = ref.get_reference(ref.get_phase(torch.tensor([-lag_j])))[0, j]
        assert torch.allclose(written["pos"][0, j], expected)


def test_bad_lag_length_is_rejected():
    with pytest.raises(ValueError):
        rewards._lag_tensor([0.1, 0.2], "cpu")


def test_spine_imitation_uses_clock_offset():
    asset, env = _reset(list(motion.JOINT_NAMES), 0.0, random_phase=True, return_env=True)
    asset.data.joint_pos.torch = asset.written["pos"].clone()
    r = rewards.spine_pos_imitation(env, csv_path=None, gait_period=1.0)
    assert torch.allclose(r, torch.ones(4), atol=1e-6)


def test_mimic_csv_col_order_maps_asset_order_to_sim_dof_order():
    env_source = (HEXAPOD_DIR / "hexapod_mimic_env_cfg.py").read_text()
    goal_source = (HEXAPOD_DIR / "hexapod_goal_mimic_env_cfg.py").read_text()
    assert "MIMIC_CSV_COL_ORDER: list[int] | None = [1, 0, 2, 3, 6, 7, 4, 5]" in env_source
    # every reward/reset param block must use the constant, never a bare None
    assert '"csv_col_order": None' not in env_source and '"csv_col_order": None' not in goal_source

    asset_order = [
        "FrontLink",
        "BackLink",
        "MiddleLeft",
        "MiddleRight",
        "FrontLeft",
        "FrontRight",
        "BackLeft",
        "BackRight",
    ]
    sim_order = [
        "BackLink",
        "FrontLink",
        "MiddleLeft",
        "MiddleRight",
        "BackLeft",
        "BackRight",
        "FrontLeft",
        "FrontRight",
    ]
    order = [1, 0, 2, 3, 6, 7, 4, 5]
    assert [asset_order[i] for i in order] == sim_order
