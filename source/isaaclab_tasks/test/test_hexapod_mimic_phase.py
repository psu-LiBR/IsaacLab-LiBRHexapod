# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the phase-tracked imitation reward ``joint_pos_imitation_phase``.

Loads the hexapod modules through a synthetic package with ``isaaclab.managers`` stubbed, so only ``torch`` and
``numpy`` are needed (plain system Python works).
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

HEXAPOD_DIR = Path(__file__).parents[1] / "isaaclab_tasks" / "contrib" / "velocity" / "config" / "hexapod"
PKG = "_hexapod_mimic_phase_under_test"


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
    motion = importlib.import_module(f"{PKG}.hexapod_mimic_motion")
finally:  # do not leak the stubs into other test modules
    for _k, _v in _saved.items():
        if _v is None:
            sys.modules.pop(_k, None)
        else:
            sys.modules[_k] = _v

NUM_ROWS = 200


class FakeAsset:
    def __init__(self, joint_names: list[str], n_envs: int):
        self.data = SimpleNamespace(
            joint_names=joint_names,
            joint_pos=SimpleNamespace(torch=torch.zeros(n_envs, len(joint_names))),
        )


def _make_env(n_envs: int = 1, joint_names: list[str] | None = None):
    asset = FakeAsset(list(joint_names or motion.JOINT_NAMES), n_envs)
    env = SimpleNamespace(
        scene={"robot": asset}, step_dt=0.02, episode_length_buf=torch.zeros(n_envs, dtype=torch.long)
    )
    rewards._JOINT_IDX_CACHE.clear()
    return env, asset


def _ref():
    return rewards._get_motion_ref(None, 1.0, None)


def _call(env, asset, buf: int, rows: torch.Tensor, **kw) -> torch.Tensor:
    """Set the pose to reference ``rows`` (long [N]) at counter ``buf`` and evaluate the reward."""
    env.episode_length_buf = torch.full((rows.shape[0],), buf, dtype=torch.long)
    asset.data.joint_pos.torch = _ref().get_reference_rows(rows % NUM_ROWS).clone()
    kw.setdefault("min_step_rows", 2)
    kw.setdefault("max_step_rows", 6)
    return rewards.joint_pos_imitation_phase(env, csv_path=None, gait_period=1.0, **kw)


def _run(row_fn, steps: int, n_envs: int = 1, **kw):
    """Run steps t=1..steps with the pose at ``row_fn(t)`` (long [n_envs]); return (rewards, phase idx, phase step)."""
    env, asset = _make_env(n_envs)
    r_all, k_all, s_all = [], [], []
    for t in range(1, steps + 1):
        r_all.append(_call(env, asset, t, row_fn(t), **kw).clone())
        k_all.append(env._mimic_phase_idx.clone())
        s_all.append(env._mimic_phase_step.clone())
    return torch.stack(r_all), torch.stack(k_all), torch.stack(s_all)


def _replay(start: int, speed: int):
    return lambda t: torch.tensor([start + speed * (t - 1)])


@pytest.mark.parametrize("speed", [3, 4, 5])
# The built-in gait is mirror-symmetric (row r and row 100 - r share a pose), so start rows are chosen where the
# seed (first argmax) is the true row. Start 0 wraps within the run.
@pytest.mark.parametrize("start", [0, 20, 40])
def test_replay_at_varied_speeds_scores_full_reward(speed, start):
    r, k, s = _run(_replay(start, speed), 60)
    assert (r[1:] > 0.99).all()
    assert (s[1:] == speed).all()
    assert (s[0] == 0).all()


def test_frozen_pose_loses_reward():
    r, k, s = _run(lambda t: torch.tensor([37]), 50)
    assert r[25, 0] < 0.8 * r[0, 0]
    assert (s[1:] >= 2).all()
    # the index really kept moving every step (modulo wrap)
    assert (((k[1:] - k[:-1]) % NUM_ROWS) >= 2).all()


def test_too_fast_replay_is_capped():
    r_fast, _, s_fast = _run(_replay(0, 10), 41)
    r_ref, _, _ = _run(_replay(0, 4), 41)
    # the advance saturates at the cap while the pose runs away (later it may re-lock on another similar row)
    assert (s_fast[1:6] == 6).all()
    assert ((s_fast[1:] >= 2) & (s_fast[1:] <= 6)).all()
    assert r_fast[9:40].mean() < r_ref[9:40].mean() - 0.05


@pytest.mark.parametrize("row", [0, 43, 101])
def test_seed_finds_nearest_row(row):
    env, asset = _make_env()
    r = _call(env, asset, 1, torch.tensor([row]))
    assert env._mimic_phase_idx.item() == row
    assert env._mimic_phase_step.item() == 0
    assert r.item() == pytest.approx(1.0, abs=1e-5)


def test_reset_reseeds_phase():
    env, asset = _make_env()
    for t in range(1, 8):
        _call(env, asset, t, torch.tensor([10 + 4 * (t - 1)]))
    assert env._mimic_phase_step.item() == 4
    r = _call(env, asset, 1, torch.tensor([140]))
    assert env._mimic_phase_idx.item() == 140
    assert env._mimic_phase_step.item() == 0
    assert r.item() == pytest.approx(1.0, abs=1e-5)


def test_phase_wraps():
    env, asset = _make_env()
    _call(env, asset, 1, torch.tensor([0]))
    env._mimic_phase_idx.fill_(193)  # force the phase next to the table end
    ks = []
    for t in range(2, 30):
        r = _call(env, asset, t, torch.tensor([193 + 4 * (t - 1)]))
        ks.append(env._mimic_phase_idx.item())
        assert r.item() > 0.99
    assert ks[0] == 197 and ks[1] == 1  # 197 -> 201 % 200
    assert all(0 <= k < NUM_ROWS for k in ks)


def test_seed_is_ambiguous_for_mirrored_rows_of_builtin_gait():
    env, asset = _make_env()
    r = _call(env, asset, 1, torch.tensor([57]))
    assert env._mimic_phase_idx.item() in (57, 43)
    assert r.item() == pytest.approx(1.0, abs=1e-5)


@pytest.mark.parametrize(
    "lo, hi",
    [(0, 6), (7, 6), (2, NUM_ROWS), (2, NUM_ROWS + 5)],
)
def test_bad_window_rejected(lo, hi):
    env, asset = _make_env()
    with pytest.raises(ValueError):
        _call(env, asset, 1, torch.tensor([0]), min_step_rows=lo, max_step_rows=hi)


def test_per_env_independence():
    speeds = torch.tensor([3, 4, 5])
    starts = torch.tensor([0, 20, 40])
    r, k, s = _run(lambda t: starts + speeds * (t - 1), 40, n_envs=3)
    assert (r[1:] > 0.99).all()
    for e in range(3):
        assert (s[1:, e] == speeds[e]).all()


def test_reordered_asset_joints():
    names = list(motion.JOINT_NAMES)
    perm = [3, 0, 7, 1, 6, 2, 5, 4]
    shuffled = [names[i] for i in perm]  # asset column c holds Sim-DOF joint perm[c]
    rows = lambda t: torch.tensor([20 + 4 * (t - 1)])  # noqa: E731

    env_a, asset_a = _make_env()
    env_b, asset_b = _make_env(joint_names=shuffled)
    for t in range(1, 15):
        env_a.episode_length_buf = env_b.episode_length_buf = torch.full((1,), t, dtype=torch.long)
        q = _ref().get_reference_rows(rows(t) % NUM_ROWS).clone()
        # a small perturbation so the reward is not trivially 1.0
        q = q + 0.05 * torch.sin(torch.arange(8.0) + t)
        asset_a.data.joint_pos.torch = q
        asset_b.data.joint_pos.torch = q[:, perm].clone()
        kw = dict(csv_path=None, gait_period=1.0, min_step_rows=2, max_step_rows=6)
        ra = rewards.joint_pos_imitation_phase(env_a, **kw)
        rb = rewards.joint_pos_imitation_phase(env_b, **kw)
        assert torch.allclose(ra, rb)
        assert torch.equal(env_a._mimic_phase_idx, env_b._mimic_phase_idx)
