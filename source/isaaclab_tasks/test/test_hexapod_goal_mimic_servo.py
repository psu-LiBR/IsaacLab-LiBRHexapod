# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the servo-realistic variant of the hexapod goal + mimic task.

None of these tests launch Isaac Sim. The source-text tests need nothing but the standard library; the config test
is skipped when the hexapod task package cannot be imported.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

SOURCE_ROOT = Path(__file__).parents[2]
HEXAPOD_CFG_DIR = SOURCE_ROOT / "isaaclab_tasks" / "isaaclab_tasks" / "contrib" / "velocity" / "config" / "hexapod"

SERVO_IDS = {
    "Isaac-Goal-Flat-Hexapod-Mimic-Servo-v0": "HexapodGoalMimicServoEnvCfg",
    "Isaac-Goal-Flat-Hexapod-Mimic-Servo-Play-v0": "HexapodGoalMimicServoEnvCfg_PLAY",
}


def _read(name: str) -> str:
    return (HEXAPOD_CFG_DIR / name).read_text(encoding="utf-8")


def _class_source(source: str, class_name: str) -> str:
    for node in ast.parse(source).body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return ast.get_source_segment(source, node)
    raise AssertionError(f"class {class_name} not found")


def test_servo_task_ids_are_registered_with_their_own_runner():
    init = _read("__init__.py")
    for task_id, cls in SERVO_IDS.items():
        pattern = (
            rf'id="{re.escape(task_id)}",.*?'
            rf'"env_cfg_entry_point": f"\{{__name__\}}\.hexapod_goal_mimic_env_cfg:{cls}",\s*'
            rf'"rsl_rl_cfg_entry_point": f"\{{agents\.__name__\}}\.rsl_rl_ppo_goal_mimic_cfg:'
            rf'HexapodGoalMimicServoPPORunnerCfg",'
        )
        assert re.search(pattern, init, flags=re.S), task_id


def test_servo_runner_logs_to_its_own_experiment():
    body = _class_source(_read("agents/rsl_rl_ppo_goal_mimic_cfg.py"), "HexapodGoalMimicServoPPORunnerCfg")
    assert 'self.experiment_name = "hexapod_goal_mimic_servo"' in body


def test_servo_classes_reuse_the_binary_servo_model():
    source = _read("hexapod_goal_mimic_env_cfg.py")
    assert source.count("class HexapodGoalMimicServoEnvCfg(HexapodGoalMimicEnvCfg):") == 1
    assert source.count("class HexapodGoalMimicServoEnvCfg_PLAY(HexapodGoalMimicEnvCfg_PLAY):") == 1
    assert "from .hexapod_binary_env_cfg import (" in source  # one source of truth for the SERVO_* fit
    for cls in SERVO_IDS.values():
        body = _class_source(source, cls)
        assert "super().__post_init__()" in body
        assert "self.scene.robot.actuators = with_servo_model(" in body
        assert "_servo_group_overrides(" in body


def test_servo_env_cfgs_match_the_binary_servo_actuators():
    pytest.importorskip("torch")
    try:
        from isaaclab_tasks.contrib.velocity.config.hexapod import hexapod_binary_env_cfg as binary
        from isaaclab_tasks.contrib.velocity.config.hexapod import hexapod_goal_mimic_env_cfg as mimic
    except Exception as exc:  # pragma: no cover - depends on the local install
        pytest.skip(f"hexapod task package not importable without Isaac Sim: {exc}")

    pairs = (
        (mimic.HexapodGoalMimicServoEnvCfg(), binary.HexapodBinaryServoEnvCfg()),
        (mimic.HexapodGoalMimicServoEnvCfg_PLAY(), binary.HexapodBinaryServoEnvCfg_PLAY()),
    )
    fields = ("stiffness", "armature", "min_delay", "max_delay", "stiffness_scale_range", "voltage_scale_range")
    for cont, disc in pairs:
        for group in ("leg_joints", "body_joints"):
            a, b = cont.scene.robot.actuators[group], disc.scene.robot.actuators[group]
            assert type(a).__name__ == "ServoDCMotorCfg"
            assert {f: getattr(a, f) for f in fields} == {f: getattr(b, f) for f in fields}, group
