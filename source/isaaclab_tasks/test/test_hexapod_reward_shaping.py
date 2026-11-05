# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import ast
import re
from pathlib import Path

SOURCE_ROOT = Path(__file__).parents[2]
HEXAPOD_CFG_DIR = SOURCE_ROOT / "isaaclab_tasks" / "isaaclab_tasks" / "contrib" / "velocity" / "config" / "hexapod"


def test_flat_rshape_keeps_reward_shaping_values():
    source = (HEXAPOD_CFG_DIR / "flat_env_cfg_rshape.py").read_text(encoding="utf-8")

    for expected in (
        'self.rewards.track_lin_vel_xy_exp.params["std"] = math.sqrt(0.25) * 0.3',
        "self.rewards.feet_air_time.weight = 2.5",
        'self.rewards.feet_air_time.params["threshold"] = 0.16',
        "self.rewards.action_rate_l2.weight = -3.0e-2",
        'self.events.physics_material.params["static_friction_range"] = (0.9, 1.0)',
        'self.events.physics_material.params["dynamic_friction_range"] = (0.7, 0.8)',
    ):
        assert expected in source


def test_flat_rshape_class_names_do_not_shadow_the_baseline():
    source = (HEXAPOD_CFG_DIR / "flat_env_cfg_rshape.py").read_text(encoding="utf-8")

    assert "class HexapodFlatRshapeEnvCfg(HexapodRoughEnvCfg):" in source
    assert "class HexapodFlatRshapeEnvCfg_PLAY(HexapodFlatRshapeEnvCfg):" in source
    # The baseline class name must not be redefined here; both files are imported
    # from the same package.
    assert "class HexapodFlatEnvCfg(" not in source


def test_foot_slide_scan_covers_the_intended_weights():
    source = (HEXAPOD_CFG_DIR / "flat_env_cfg_rshape.py").read_text(encoding="utf-8")

    assert "weight=-0.6," in source
    for cls, weight in (
        ("HexapodFlatRshapeSlide045EnvCfg", "-0.45"),
        ("HexapodFlatRshapeSlide035EnvCfg", "-0.35"),
        ("HexapodFlatRshapeSlide025EnvCfg", "-0.25"),
    ):
        assert f"class {cls}(HexapodFlatRshapeEnvCfg):" in source
        assert f"self.rewards.feet_slide.weight = {weight}" in source


def test_goal_variants_override_the_air_time_command_name():
    source = (HEXAPOD_CFG_DIR / "hexapod_goal_tuned_env_cfg.py").read_text(encoding="utf-8")

    # HexapodGoalEnvCfg disables feet_air_time, and the default term in
    # velocity_env_cfg.py hardcodes command_name="base_velocity", which the goal env
    # does not define. Rebuilding the term without this override raises KeyError.
    assert '"command_name": "pose_command"' in source
    assert '"command_name": "base_velocity"' not in source
    assert "self.rewards.feet_air_time = _feet_air_time_term()" in source


def test_goal_variants_cover_the_trained_arms():
    source = (HEXAPOD_CFG_DIR / "hexapod_goal_tuned_env_cfg.py").read_text(encoding="utf-8")

    assert "AIR_TIME_WEIGHT = 2.5" in source
    assert "AIR_TIME_THRESHOLD = 0.16" in source
    assert "ACTION_RATE_WEIGHT = -3.0e-2" in source
    for cls in (
        "HexapodGoalBigStepEnvCfg",
        "HexapodGoalBigStepSlide06EnvCfg",
        "HexapodGoalBigStepSlide035EnvCfg",
        "HexapodGoalBigStepMinimalEnvCfg",
        "HexapodGoalBigStepEnvCfg_PLAY",
    ):
        assert f"class {cls}(" in source
    assert "_feet_slide_term(-0.6)" in source
    assert "_feet_slide_term(-0.35)" in source


def test_reward_shaping_tasks_are_registered():
    source = (HEXAPOD_CFG_DIR / "__init__.py").read_text(encoding="utf-8")

    for task_id in (
        "Isaac-Velocity-Flat-Hexapod-Rshape-v0",
        "Isaac-Velocity-Flat-Hexapod-Rshape-Slide045-v0",
        "Isaac-Velocity-Flat-Hexapod-Rshape-Slide035-v0",
        "Isaac-Velocity-Flat-Hexapod-Rshape-Slide025-v0",
        "Isaac-Velocity-Flat-Hexapod-Rshape-Play-v0",
        "Isaac-Goal-Flat-Hexapod-BigStep-v0",
        "Isaac-Goal-Flat-Hexapod-BigStep-Slide06-v0",
        "Isaac-Goal-Flat-Hexapod-BigStep-Slide035-v0",
        "Isaac-Goal-Flat-Hexapod-BigStep-Minimal-v0",
        "Isaac-Goal-Flat-Hexapod-BigStep-Play-v0",
    ):
        assert f'id="{task_id}"' in source


def _binary_cfg_source() -> str:
    return (HEXAPOD_CFG_DIR / "hexapod_binary_env_cfg.py").read_text(encoding="utf-8")


def _module_constant(source: str, name: str):
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not assigned at module level")


def test_binary_cfg_registers_min_stance_legs_term():
    source = _binary_cfg_source()
    flat = re.sub(r"\s+", " ", source).replace("( ", "(").replace(", )", ")")

    assert "rewards.min_stance_legs = RewTerm(func=goal_rewards.min_stance_legs_penalty" in flat
    assert "weight=MIN_STANCE_PENALTY_WEIGHT" in flat
    assert 'params={"min_legs": MIN_STANCE_LEGS}' in flat
    assert _module_constant(source, "MIN_STANCE_LEGS") == 3
    assert _module_constant(source, "MIN_STANCE_PENALTY_WEIGHT") < 0


def test_binary_cfg_applies_min_stance_penalty_to_train_and_play():
    tree = ast.parse(_binary_cfg_source())
    classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}

    for cls_name in ("HexapodBinaryEnvCfg", "HexapodBinaryEnvCfg_PLAY"):
        post_init = next(
            n for n in classes[cls_name].body if isinstance(n, ast.FunctionDef) and n.name == "__post_init__"
        )
        assert "_rebalance_binary_rewards(self.rewards)" in [
            ast.unparse(c) for c in ast.walk(post_init) if isinstance(c, ast.Call)
        ], cls_name

    # The term is added inside _rebalance_binary_rewards so both variants stay identical.
    rebalance = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_rebalance_binary_rewards")
    assert "rewards.min_stance_legs" in {
        ast.unparse(t) for n in ast.walk(rebalance) if isinstance(n, ast.Assign) for t in n.targets
    }


def test_goal_rewards_defines_min_stance_legs_penalty():
    source = (HEXAPOD_CFG_DIR / "hexapod_goal_rewards.py").read_text(encoding="utf-8")

    assert "def min_stance_legs_penalty(" in source
