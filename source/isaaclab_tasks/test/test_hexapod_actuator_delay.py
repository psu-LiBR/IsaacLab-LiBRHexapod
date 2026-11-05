# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the hexapod servo command delay (``DelayedDCMotor``, ``HEXAPOD_CFG`` and the ``_PLAY`` configs).

The source-text and delay-buffer tests need only ``torch``; the actuator / config tests additionally need the
(app-free) ``isaaclab.actuators`` package and are skipped when it cannot be imported.
"""

from __future__ import annotations

import ast
import contextlib
import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

SOURCE_ROOT = Path(__file__).parents[2]
HEXAPOD_CFG_DIR = SOURCE_ROOT / "isaaclab_tasks" / "isaaclab_tasks" / "contrib" / "velocity" / "config" / "hexapod"
ROBOTS_DIR = SOURCE_ROOT / "isaaclab_assets" / "isaaclab_assets" / "robots"
ACTUATORS_FILE = ROBOTS_DIR / "hexapod_actuators.py"
BUFFERS_DIR = SOURCE_ROOT / "isaaclab" / "isaaclab" / "utils" / "buffers"

PLAY_CFG_FILES = {
    "rough_env_cfg.py": "HexapodRoughEnvCfg_PLAY",
    "flat_env_cfg.py": "HexapodFlatEnvCfg_PLAY",
    "flat_env_cfg_rshape.py": "HexapodFlatRshapeEnvCfg_PLAY",
    "hexapod_mimic_env_cfg.py": "HexapodMimicEnvCfg_PLAY",
    "hexapod_goal_env_cfg.py": "HexapodGoalEnvCfg_PLAY",
}


def _class_source(path: Path, class_name: str) -> str:
    """Return the source text of a top-level class."""
    source = path.read_text(encoding="utf-8")
    for node in ast.parse(source).body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return ast.get_source_segment(source, node)
    raise AssertionError(f"class {class_name} not found in {path.name}")


def _module_constants(path: Path) -> dict:
    out = {}
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            with contextlib.suppress(ValueError):
                out[node.targets[0].id] = ast.literal_eval(node.value)
    return out


# ---------------------------------------------------------------------------
# (a) source-text assertions
# ---------------------------------------------------------------------------


def test_robot_asset_uses_the_delay_with_ordered_ranges():
    consts = _module_constants(ROBOTS_DIR / "hexapod.py")
    assert consts["HEXAPOD_ACTUATOR_DELAY"] is True
    assert consts["HEXAPOD_LEG_DELAY_RANGE_STEPS"] == (8, 12)
    assert consts["HEXAPOD_SPINE_DELAY_RANGE_STEPS"] == (11, 15)
    source = (ROBOTS_DIR / "hexapod.py").read_text(encoding="utf-8")
    assert source.count("_ACTUATOR_CFG(") == 2


def test_every_play_cfg_pins_the_delay():
    pin = "self.scene.robot.actuators = with_fixed_actuator_delay(self.scene.robot.actuators)"
    for file, cls in PLAY_CFG_FILES.items():
        assert pin in _class_source(HEXAPOD_CFG_DIR / file, cls), cls
    # binary and BigStep play configs inherit the pin from the goal play config
    for file, cls in (
        ("hexapod_binary_env_cfg.py", "HexapodBinaryEnvCfg_PLAY"),
        ("hexapod_goal_tuned_env_cfg.py", "HexapodGoalBigStepEnvCfg_PLAY"),
    ):
        assert "(HexapodGoalEnvCfg_PLAY)" in _class_source(HEXAPOD_CFG_DIR / file, cls), cls


def test_actuators_module_defines_the_public_api():
    tree = ast.parse(ACTUATORS_FILE.read_text(encoding="utf-8"))
    classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
    functions = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert [ast.unparse(b) for b in classes["DelayedDCMotor"].bases] == ["DCMotor"]
    assert [ast.unparse(b) for b in classes["DelayedDCMotorCfg"].bases] == ["DCMotorCfg"]
    assert "with_fixed_actuator_delay" in functions


# ---------------------------------------------------------------------------
# (b) delay-buffer semantics (torch only: buffers loaded standalone from source)
# ---------------------------------------------------------------------------


def _load_delay_buffer():
    """Return ``DelayBuffer``: the real ``isaaclab.utils`` one if importable, else loaded standalone from source.

    ``delay_buffer.py`` imports ``..array`` (``index_fill_``), so the standalone fallback also needs the sibling
    ``isaaclab.utils.array`` module; the test is skipped if neither route works.
    """
    with contextlib.suppress(ImportError):
        from isaaclab.utils import DelayBuffer

        return DelayBuffer
    pkg_name = "_standalone_isaaclab_utils"
    try:
        if pkg_name not in sys.modules:
            root = types.ModuleType(pkg_name)
            root.__path__ = [str(BUFFERS_DIR.parent)]
            sys.modules[pkg_name] = root
            for sub, path in (("array", BUFFERS_DIR.parent / "array.py"), ("buffers", BUFFERS_DIR)):
                if sub == "buffers":
                    mod = types.ModuleType(f"{pkg_name}.buffers")
                    mod.__path__ = [str(path)]
                    sys.modules[f"{pkg_name}.buffers"] = mod
                else:
                    spec = importlib.util.spec_from_file_location(f"{pkg_name}.{sub}", path)
                    mod = importlib.util.module_from_spec(spec)
                    sys.modules[f"{pkg_name}.{sub}"] = mod
                    spec.loader.exec_module(mod)
            for name in ("circular_buffer", "delay_buffer"):
                spec = importlib.util.spec_from_file_location(f"{pkg_name}.buffers.{name}", BUFFERS_DIR / f"{name}.py")
                module = importlib.util.module_from_spec(spec)
                sys.modules[f"{pkg_name}.buffers.{name}"] = module
                spec.loader.exec_module(module)
        return sys.modules[f"{pkg_name}.buffers.delay_buffer"].DelayBuffer
    except Exception as err:  # pragma: no cover - depends on the environment
        for key in [k for k in sys.modules if k.startswith(pkg_name)]:
            del sys.modules[key]
        pytest.skip(f"DelayBuffer cannot be loaded standalone: {err!r}")


@pytest.mark.parametrize("lag", [0, 1, 3, 12])
def test_delay_buffer_step_input_appears_exactly_lag_calls_later(lag):
    delay_buffer = _load_delay_buffer()(history_length=16, batch_size=2, device="cpu")
    delay_buffer.set_time_lag(lag)
    delay_buffer.reset()
    step_call = 5
    outputs = []
    for k in range(step_call + lag + 5):
        value = 0.0 if k < step_call else 1.0
        outputs.append(delay_buffer.compute(torch.full((2, 3), value))[0, 0].item())
    first_nonzero = next(i for i, v in enumerate(outputs) if v != 0.0)
    assert first_nonzero == step_call + lag
    assert all(v == 1.0 for v in outputs[first_nonzero:])


def test_delay_buffer_history_equal_to_max_delay_and_zero_delay_is_passthrough():
    cls = _load_delay_buffer()
    buf = cls(0, 3, device="cpu")  # max_delay = 0 -> history length 0
    buf.set_time_lag(0)
    for k in range(4):
        data = torch.full((3, 2), float(k))
        assert torch.equal(buf.compute(data), data)
    with pytest.raises(ValueError):
        buf.set_time_lag(1)  # lag may not exceed the history length


def test_delay_buffer_per_env_lags_and_partial_reset():
    buf = _load_delay_buffer()(history_length=6, batch_size=3, device="cpu")
    buf.set_time_lag(torch.tensor([0, 2, 4], dtype=torch.int))
    buf.reset()
    seen = []
    for k in range(8):
        seen.append(buf.compute(torch.full((3, 1), float(k)))[:, 0].tolist())
    # env e outputs the command pushed `lag_e` calls earlier (clamped by the back-filled first push)
    assert seen[7] == [7.0, 5.0, 3.0]
    # reset only env 1: its history is cleared (back-filled by the next push), others keep their lag
    buf.reset(torch.tensor([1]))
    out = buf.compute(torch.full((3, 1), 100.0))[:, 0].tolist()
    assert out == [100.0, 100.0, 4.0]


# ---------------------------------------------------------------------------
# Actuator / config tests (need app-free ``isaaclab.actuators``)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def actuators_module():
    pytest.importorskip("isaaclab.actuators")
    spec = importlib.util.spec_from_file_location("_hexapod_actuators_under_test", ACTUATORS_FILE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _dc_cfg(cfg_class=None, **overrides):
    from isaaclab.actuators import DCMotorCfg

    kwargs = dict(
        joint_names_expr=[".*"],
        saturation_effort=1.4,
        effort_limit=1.4,
        velocity_limit=5.97,
        stiffness=1.0,
        damping=0.0,
        armature=1.3e-3,
        friction=0.04,
        dynamic_friction=0.03,
    )
    kwargs.update(overrides)
    return (cfg_class or DCMotorCfg)(**kwargs)


def _make_actuator(module, min_delay, max_delay, num_envs=4, **cfg_overrides):
    cfg = _dc_cfg(module.DelayedDCMotorCfg, min_delay=min_delay, max_delay=max_delay, **cfg_overrides)
    return module.DelayedDCMotor(cfg, joint_names=["j0", "j1"], joint_ids=slice(None), num_envs=num_envs, device="cpu")


def _command(target: float, num_envs: int = 4):
    from isaaclab.utils.types import ArticulationActions

    return ArticulationActions(
        joint_positions=torch.full((num_envs, 2), target),
        joint_velocities=torch.zeros(num_envs, 2),
        joint_efforts=torch.zeros(num_envs, 2),
    )


def test_with_fixed_actuator_delay_pins_each_group_to_the_middle_of_its_range(actuators_module):
    delayed = actuators_module.DelayedDCMotorCfg
    actuators = {
        "legs": _dc_cfg(delayed, stiffness=50.0, min_delay=8, max_delay=12),
        "spine": _dc_cfg(delayed, stiffness=10.0, min_delay=11, max_delay=15),
        "plain": _dc_cfg(),  # delay switched off: copied unchanged
    }
    out = actuators_module.with_fixed_actuator_delay(actuators)
    assert (out["legs"].min_delay, out["legs"].max_delay) == (10, 10)
    assert (out["spine"].min_delay, out["spine"].max_delay) == (13, 13)
    assert type(out["plain"]).__name__ == "DCMotorCfg"
    assert out["legs"].stiffness == 50.0 and out["spine"].stiffness == 10.0
    # the input is not mutated
    assert (actuators["legs"].min_delay, actuators["legs"].max_delay) == (8, 12)
    assert (actuators["spine"].min_delay, actuators["spine"].max_delay) == (11, 15)


def test_delayed_dc_motor_applies_effort_exactly_lag_steps_later(actuators_module):
    lag, step_call = 7, 3
    actuator = _make_actuator(actuators_module, lag, lag)
    actuator.reset(slice(None))
    joint_pos, joint_vel = torch.zeros(4, 2), torch.zeros(4, 2)
    efforts = []
    for k in range(step_call + lag + 4):
        target = 0.0 if k < step_call else 0.5
        out = actuator.compute(_command(target), joint_pos, joint_vel)
        efforts.append(out.joint_efforts[0, 0].item())
    first = next(i for i, v in enumerate(efforts) if v != 0.0)
    assert first == step_call + lag
    assert efforts[first] == pytest.approx(0.5)  # stiffness 1.0 * target 0.5, inside the torque limit


def test_delayed_dc_motor_zero_delay_matches_plain_dc_motor(actuators_module):
    from isaaclab.actuators import DCMotor

    delayed = _make_actuator(actuators_module, 0, 0)
    plain = DCMotor(_dc_cfg(), joint_names=["j0", "j1"], joint_ids=slice(None), num_envs=4, device="cpu")
    delayed.reset(None)
    plain.reset(None)
    joint_pos, joint_vel = torch.zeros(4, 2), torch.full((4, 2), 2.0)
    for target in (0.0, 0.3, -0.8, 5.0):
        a = delayed.compute(_command(target), joint_pos, joint_vel).joint_efforts
        b = plain.compute(_command(target), joint_pos, joint_vel).joint_efforts
        assert torch.allclose(a, b)


def test_delayed_dc_motor_keeps_torque_speed_clipping_with_current_joint_velocity(actuators_module):
    actuator = _make_actuator(actuators_module, 4, 4, stiffness=100.0)
    actuator.reset(None)
    joint_pos = torch.zeros(4, 2)
    # at no-load speed the deliverable torque is ~0, whatever the (delayed) command asks for
    out = actuator.compute(_command(1.0), joint_pos, torch.full((4, 2), 5.97))
    assert out.joint_efforts.abs().max().item() == pytest.approx(0.0, abs=1e-5)
    # at standstill the same command saturates at the stall / effort limit
    actuator2 = _make_actuator(actuators_module, 4, 4, stiffness=100.0)
    actuator2.reset(None)
    out = actuator2.compute(_command(1.0), joint_pos, torch.zeros(4, 2))
    assert out.joint_efforts.max().item() == pytest.approx(1.4)


@pytest.mark.parametrize(
    "make_env_ids",
    [
        lambda: None,
        lambda: slice(None),
        lambda: slice(1, 3),
        lambda: [0, 2],
        lambda: torch.tensor([1, 3]),
        lambda: torch.tensor([], dtype=torch.long),
    ],
    ids=["None", "slice-all", "slice-part", "list", "tensor", "empty"],
)
def test_delayed_dc_motor_reset_accepts_all_env_id_forms(actuators_module, make_env_ids):
    actuator = _make_actuator(actuators_module, 2, 6)
    env_ids = make_env_ids()
    actuator.reset(env_ids)
    lags = actuator.positions_delay_buffer.time_lags
    assert ((lags >= 0) & (lags <= 6)).all()
    # a following compute() must not raise
    actuator.compute(_command(0.1), torch.zeros(4, 2), torch.zeros(4, 2))


def test_delayed_dc_motor_partial_reset_redraws_only_selected_envs(actuators_module):
    actuator = _make_actuator(actuators_module, 2, 14, num_envs=64)
    torch.manual_seed(0)
    actuator.reset(None)
    before = actuator.positions_delay_buffer.time_lags.clone()
    assert (before >= 2).all() and (before <= 14).all()
    assert before.unique().numel() > 1  # actually randomized
    ids = torch.tensor([3, 9, 20])
    actuator.reset(ids)
    after = actuator.positions_delay_buffer.time_lags
    untouched = torch.ones(64, dtype=torch.bool)
    untouched[ids] = False
    assert torch.equal(after[untouched], before[untouched])
    for buf in (actuator.velocities_delay_buffer, actuator.efforts_delay_buffer):
        assert torch.equal(buf.time_lags, after)


def test_delayed_dc_motor_rejects_invalid_delay_range(actuators_module):
    cfg = _dc_cfg(actuators_module.DelayedDCMotorCfg, min_delay=4, max_delay=2)
    with pytest.raises(ValueError):
        actuators_module.DelayedDCMotor(cfg, joint_names=["j0"], joint_ids=slice(None), num_envs=2, device="cpu")


def test_robot_asset_is_delayed_and_play_cfgs_pin_the_delay(actuators_module):
    try:
        from isaaclab_tasks.contrib.velocity.config.hexapod import (
            flat_env_cfg,
            hexapod_binary_env_cfg,
            hexapod_goal_env_cfg,
        )

        from isaaclab_assets import HEXAPOD_CFG
    except Exception as exc:  # pragma: no cover - depends on the local install
        pytest.skip(f"hexapod task package not importable without Isaac Sim: {exc}")

    def delays(cfg):
        return {k: (v.min_delay, v.max_delay) for k, v in cfg.scene.robot.actuators.items()}

    ranges = {"leg_joints": (8, 12), "body_joints": (11, 15)}
    for act in HEXAPOD_CFG.actuators.values():
        assert type(act).__name__ == "DelayedDCMotorCfg"
    assert delays(flat_env_cfg.HexapodFlatEnvCfg()) == ranges
    pinned = {"leg_joints": (10, 10), "body_joints": (13, 13)}
    for cfg in (
        flat_env_cfg.HexapodFlatEnvCfg_PLAY(),
        hexapod_goal_env_cfg.HexapodGoalEnvCfg_PLAY(),
        hexapod_binary_env_cfg.HexapodBinaryEnvCfg_PLAY(),
    ):
        assert delays(cfg) == pinned
    # the shared asset is never mutated by the play pin
    assert {k: (v.min_delay, v.max_delay) for k, v in HEXAPOD_CFG.actuators.items()} == ranges
