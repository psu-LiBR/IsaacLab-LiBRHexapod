from __future__ import annotations

from pathlib import Path


SOURCE_ROOT = Path(__file__).parents[2]
HEXAPOD_ASSET = SOURCE_ROOT / "isaaclab_assets" / "isaaclab_assets" / "robots" / "hexapod.py"
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


def test_hexapod_actuators_use_motor_rated_hard_limits():
    asset_source = HEXAPOD_ASSET.read_text(encoding="utf-8")

    assert "MOTOR_STALL_TORQUE_NM = 1.4" in asset_source
    assert "MOTOR_NO_LOAD_SPEED_RAD_S = 5.97" in asset_source
    assert asset_source.count("effort_limit_sim = MOTOR_STALL_TORQUE_NM") == 2
    assert asset_source.count("velocity_limit_sim = MOTOR_NO_LOAD_SPEED_RAD_S") == 2
    assert "effort_limit_sim = 4.5" not in asset_source
    assert "velocity_limit_sim = 15.0" not in asset_source


def test_flat_and_goal_tasks_keep_physical_behavior_penalties():
    flat_source = (HEXAPOD_CFG_DIR / "flat_env_cfg.py").read_text(encoding="utf-8")
    goal_source = (HEXAPOD_CFG_DIR / "hexapod_goal_env_cfg.py").read_text(encoding="utf-8")
    mimic_source = (HEXAPOD_CFG_DIR / "hexapod_mimic_env_cfg.py").read_text(encoding="utf-8")

    for expected in (
        "self.rewards.dof_torques_l2.weight = -2.0e-4",
        "self.rewards.dof_acc_l2.weight = -2.5e-7",
        "self.rewards.lin_vel_z_l2.weight = -0.5",
        "self.rewards.action_rate_l2.weight = -5.0e-2",
        "self.rewards.dof_pos_limits.weight = -1.0",
    ):
        assert expected in flat_source

    for expected in (
        "self.rewards.dof_torques_l2.weight = -2.0e-4",
        "self.rewards.dof_acc_l2.weight = -2.5e-7",
        "self.rewards.lin_vel_z_l2.weight = -3.0",
        "self.rewards.action_rate_l2.weight = -5.0e-2",
        "self.rewards.dof_pos_limits.weight = -1.0",
    ):
        assert expected in goal_source

    assert '"dof_torques_l2"' in mimic_source
    assert '"dof_acc_l2"' in mimic_source
    assert '"lin_vel_z_l2"' in mimic_source
    assert '"action_rate_l2"' in mimic_source
