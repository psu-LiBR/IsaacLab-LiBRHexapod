# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the opt-in "servo-realistic" variant of the hexapod binary-contact task.

None of these tests launch Isaac Sim.  The source-text tests need nothing but the standard library; the
actuator tests need ``torch`` and the (app-free) ``isaaclab.actuators`` package and are skipped when it cannot be
imported.
"""

from __future__ import annotations

import ast
import importlib.util
import re
import sys
from pathlib import Path

import pytest

SOURCE_ROOT = Path(__file__).parents[2]
HEXAPOD_CFG_DIR = SOURCE_ROOT / "isaaclab_tasks" / "isaaclab_tasks" / "contrib" / "velocity" / "config" / "hexapod"
ACTUATORS_FILE = SOURCE_ROOT / "isaaclab_assets" / "isaaclab_assets" / "robots" / "hexapod_actuators.py"

SERVO_IDS = {
    "Isaac-Goal-Flat-Hexapod-Binary-Servo-v0": "HexapodBinaryServoEnvCfg",
    "Isaac-Goal-Flat-Hexapod-Binary-Servo-Play-v0": "HexapodBinaryServoEnvCfg_PLAY",
}


def _read(name: str) -> str:
    return (HEXAPOD_CFG_DIR / name).read_text(encoding="utf-8")


def _class_source(source: str, class_name: str) -> str:
    for node in ast.parse(source).body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return ast.get_source_segment(source, node)
    raise AssertionError(f"class {class_name} not found")


def _constants(source: str) -> dict:
    out = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id.startswith("SERVO_"):
                out[node.targets[0].id] = ast.literal_eval(node.value)
    return out


# ---------------------------------------------------------------------------
# source-text assertions
# ---------------------------------------------------------------------------


def test_servo_task_ids_are_registered_without_rl_entry_points():
    init = _read("__init__.py")
    for task_id, cls in SERVO_IDS.items():
        pattern = (
            rf'id="{re.escape(task_id)}",.*?'
            rf'"env_cfg_entry_point": f"\{{__name__\}}\.hexapod_binary_env_cfg:{cls}",'
        )
        assert re.search(pattern, init, flags=re.S), task_id
        start = init.index(f'id="{task_id}"')
        block = init[start : init.index("\n)\n", start)]
        assert "rsl_rl_cfg_entry_point" not in block and "skrl_cfg_entry_point" not in block


def test_servo_constants_match_the_fit():
    consts = _constants(_read("hexapod_binary_env_cfg.py"))
    assert consts["SERVO_LEG_STIFFNESS"] == pytest.approx(5.16)  # XL430 firmware-default P gain 640
    assert consts["SERVO_LEG_ARMATURE"] > 1.3e-3  # the fit needs more effective inertia than the base model
    lo, hi = consts["SERVO_LEG_LAG_RANGE_STEPS"]
    assert 0 <= lo <= consts["SERVO_LEG_LAG_PLAY_STEPS"] <= hi
    s_lo, s_hi = consts["SERVO_STIFFNESS_SCALE_RANGE"]
    assert 0 < s_lo < 1.0 < s_hi
    v_lo, v_hi = consts["SERVO_VOLTAGE_SCALE_RANGE"]
    assert 0 < v_lo <= consts["SERVO_VOLTAGE_SCALE_PLAY"] <= v_hi <= 1.0


def test_servo_classes_subclass_the_binary_classes_and_use_the_helper():
    source = _read("hexapod_binary_env_cfg.py")
    assert source.count("class HexapodBinaryServoEnvCfg(HexapodBinaryEnvCfg):") == 1
    assert source.count("class HexapodBinaryServoEnvCfg_PLAY(HexapodBinaryEnvCfg_PLAY):") == 1
    for cls in ("HexapodBinaryServoEnvCfg", "HexapodBinaryServoEnvCfg_PLAY"):
        body = _class_source(source, cls)
        assert "super().__post_init__()" in body
        assert "self.scene.robot.actuators = with_servo_model(" in body


def test_servo_module_defines_the_public_api():
    tree = ast.parse(ACTUATORS_FILE.read_text(encoding="utf-8"))
    classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
    functions = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert [ast.unparse(b) for b in classes["ServoDCMotor"].bases] == ["DelayedDCMotor"]
    assert [ast.unparse(b) for b in classes["ServoDCMotorCfg"].bases] == ["DelayedDCMotorCfg"]
    assert "with_servo_model" in functions


# ---------------------------------------------------------------------------
# actuator tests (need torch + app-free ``isaaclab.actuators``)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def actuators_module():
    pytest.importorskip("torch")
    pytest.importorskip("isaaclab.actuators")
    spec = importlib.util.spec_from_file_location("_hexapod_servo_under_test", ACTUATORS_FILE)
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
        stiffness=100.0,
        damping=0.0,
        armature=1.3e-3,
        friction=0.04,
        dynamic_friction=0.03,
    )
    kwargs.update(overrides)
    return (cfg_class or DCMotorCfg)(**kwargs)


def _make(module, num_envs=64, **overrides):
    cfg = module.with_servo_model({"g": _dc_cfg()}, {"g": overrides})["g"]
    return module.ServoDCMotor(cfg, joint_names=["j0", "j1"], joint_ids=slice(None), num_envs=num_envs, device="cpu")


def _command(target: float, num_envs: int):
    import torch

    from isaaclab.utils.types import ArticulationActions

    return ArticulationActions(
        joint_positions=torch.full((num_envs, 2), target),
        joint_velocities=torch.zeros(num_envs, 2),
        joint_efforts=torch.zeros(num_envs, 2),
    )


def test_with_servo_model_overrides_fields_without_mutating_the_source(actuators_module):
    src = {"legs": _dc_cfg(stiffness=50.0), "spine": _dc_cfg(stiffness=10.0)}
    out = actuators_module.with_servo_model(
        src, {"legs": {"stiffness": 5.0, "armature": 8e-3, "min_delay": 3, "max_delay": 5}, "spine": {}}
    )
    assert isinstance(out["legs"], actuators_module.ServoDCMotorCfg)
    assert out["legs"].class_type is actuators_module.ServoDCMotor
    assert (out["legs"].stiffness, out["legs"].armature) == (5.0, 8e-3)
    assert (out["legs"].min_delay, out["legs"].max_delay) == (3, 5)
    assert out["legs"].damping == src["legs"].damping  # untouched fields are copied
    assert out["spine"].stiffness == 10.0 and out["spine"].stiffness_scale_range == (1.0, 1.0)
    assert src["legs"].stiffness == 50.0 and type(src["legs"]).__name__ == "DCMotorCfg"


def test_with_servo_model_validates_groups_and_fields(actuators_module):
    from isaaclab.actuators import ImplicitActuatorCfg

    with pytest.raises(KeyError):
        actuators_module.with_servo_model({"a": _dc_cfg(), "b": _dc_cfg()}, {"a": {}})  # group b missing
    with pytest.raises(KeyError):
        actuators_module.with_servo_model({"a": _dc_cfg()}, {"a": {}, "c": {}})  # unknown group c
    with pytest.raises(ValueError):
        actuators_module.with_servo_model({"a": _dc_cfg()}, {"a": {"stifness": 1.0}})  # typo'd field
    with pytest.raises(TypeError):
        actuators_module.with_servo_model({"a": ImplicitActuatorCfg(joint_names_expr=[".*"])}, {"a": {}})


def test_servo_motor_rejects_invalid_scale_ranges(actuators_module):
    for bad in ({"stiffness_scale_range": (0.0, 1.0)}, {"voltage_scale_range": (1.2, 1.0)}):
        with pytest.raises(ValueError):
            _make(actuators_module, num_envs=2, **bad)


def test_servo_motor_defaults_match_the_plain_delayed_motor(actuators_module):
    import torch

    servo = _make(actuators_module, num_envs=4, min_delay=2, max_delay=2)
    delayed_cfg = _dc_cfg(actuators_module.DelayedDCMotorCfg, min_delay=2, max_delay=2)
    delayed = actuators_module.DelayedDCMotor(
        delayed_cfg, joint_names=["j0", "j1"], joint_ids=slice(None), num_envs=4, device="cpu"
    )
    servo.reset(None)
    delayed.reset(None)
    pos, vel = torch.zeros(4, 2), torch.full((4, 2), 1.5)
    for target in (0.0, 0.01, 0.3, -0.8):
        a = servo.compute(_command(target, 4), pos, vel).joint_efforts
        b = delayed.compute(_command(target, 4), pos, vel).joint_efforts
        assert torch.allclose(a, b)


def test_servo_motor_draws_stiffness_per_env_inside_range_and_never_compounds(actuators_module):
    actuator = _make(actuators_module, stiffness_scale_range=(0.5, 2.0))
    for _ in range(5):  # repeated resets must always scale the NOMINAL value
        actuator.reset(None)
        scale = actuator.stiffness / 100.0
        assert (scale >= 0.5 - 1e-6).all() and (scale <= 2.0 + 1e-6).all()
    assert scale.unique().numel() > 1  # drawn per env, not one shared value
    assert torch_equal_across_joints(scale)


def torch_equal_across_joints(scale) -> bool:
    return bool((scale[:, :1] == scale).all())  # both joints of one env share the same scale


def test_servo_motor_voltage_scales_stall_torque_and_speed_limit_together(actuators_module):
    import torch

    actuator = _make(actuators_module, voltage_scale_range=(0.8, 0.8))
    actuator.reset(None)
    assert torch.allclose(actuator.velocity_limit, torch.full((64, 2), 5.97 * 0.8))
    # stiff command at standstill saturates at the scaled stall torque
    out = actuator.compute(_command(10.0, 64), torch.zeros(64, 2), torch.zeros(64, 2))
    assert out.joint_efforts.max().item() == pytest.approx(1.4 * 0.8)
    # at the scaled no-load speed no torque is left
    out = actuator.compute(_command(10.0, 64), torch.zeros(64, 2), torch.full((64, 2), 5.97 * 0.8))
    assert out.joint_efforts.abs().max().item() == pytest.approx(0.0, abs=1e-5)


def test_servo_motor_partial_reset_only_redraws_the_reset_envs(actuators_module):
    import torch

    actuator = _make(actuators_module, stiffness_scale_range=(0.5, 2.0), voltage_scale_range=(0.8, 1.0))
    actuator.reset(None)
    before_k = actuator.stiffness.clone()
    before_v = actuator.velocity_limit.clone()
    actuator.reset(torch.tensor([3, 7]))
    changed = (actuator.stiffness != before_k).any(dim=1) | (actuator.velocity_limit != before_v).any(dim=1)
    assert set(torch.nonzero(changed).flatten().tolist()) <= {3, 7}
    actuator.reset(slice(None))  # all forms of env ids keep working
    actuator.reset(torch.tensor([], dtype=torch.long))


def test_servo_env_cfgs_apply_the_fitted_leg_model_and_leave_the_shared_asset_alone(actuators_module):
    pytest.importorskip("isaaclab_assets")
    try:
        from isaaclab_tasks.contrib.velocity.config.hexapod import hexapod_binary_env_cfg as binary

        from isaaclab_assets import HEXAPOD_CFG
    except Exception as exc:  # pragma: no cover - depends on the local install
        pytest.skip(f"hexapod task package not importable without Isaac Sim: {exc}")

    shared = {k: (type(v).__name__, v.stiffness, v.armature) for k, v in HEXAPOD_CFG.actuators.items()}
    train, play = binary.HexapodBinaryServoEnvCfg(), binary.HexapodBinaryServoEnvCfg_PLAY()
    assert {k: (type(v).__name__, v.stiffness, v.armature) for k, v in HEXAPOD_CFG.actuators.items()} == shared
    for cfg in (train, play):
        legs, spine = cfg.scene.robot.actuators["leg_joints"], cfg.scene.robot.actuators["body_joints"]
        assert type(legs).__name__ == type(spine).__name__ == "ServoDCMotorCfg"
        assert legs.stiffness == binary.SERVO_LEG_STIFFNESS and legs.armature == binary.SERVO_LEG_ARMATURE
        # the spine keeps the robot asset's gains, inertia and delay
        assert (spine.stiffness, spine.armature) == (
            HEXAPOD_CFG.actuators["body_joints"].stiffness,
            HEXAPOD_CFG.actuators["body_joints"].armature,
        )
    assert (
        train.scene.robot.actuators["leg_joints"].min_delay,
        train.scene.robot.actuators["leg_joints"].max_delay,
    ) == (binary.SERVO_LEG_LAG_RANGE_STEPS)
    legs = play.scene.robot.actuators["leg_joints"]
    assert legs.min_delay == legs.max_delay == binary.SERVO_LEG_LAG_PLAY_STEPS
    assert legs.stiffness_scale_range == (1.0, 1.0)  # play is deterministic
    assert legs.voltage_scale_range == (binary.SERVO_VOLTAGE_SCALE_PLAY,) * 2
    assert train.scene.robot.actuators["leg_joints"].stiffness_scale_range == binary.SERVO_STIFFNESS_SCALE_RANGE
    # the spine keeps the robot-wide delay: random in training, pinned in play
    train_spine, play_spine = train.scene.robot.actuators["body_joints"], play.scene.robot.actuators["body_joints"]
    assert (train_spine.min_delay, train_spine.max_delay) == (11, 15)
    assert play_spine.min_delay == play_spine.max_delay == 13
