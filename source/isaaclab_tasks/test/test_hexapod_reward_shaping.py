from __future__ import annotations

from pathlib import Path


SOURCE_ROOT = Path(__file__).parents[2]
HEXAPOD_CFG_DIR = (
    SOURCE_ROOT
    / "isaaclab_tasks"
    / "isaaclab_tasks"
    / "manager_based"
    / "locomotion"
    / "velocity"
    / "config"
    / "hexapod"
)


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
