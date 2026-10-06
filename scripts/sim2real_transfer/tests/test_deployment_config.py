# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

from sim2real.deployment_config import load_deployment_config

_EXAMPLE_CFG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "deployment.example.yaml")
_BINARY_CFG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "deployment.binary.example.yaml")

_MINIMAL_CFG_WITHOUT_LOCALIZATION = """
serial:
  port: /dev/ttyUSB0
  baud_rate: 1000000
  protocol_version: 2.0
joints:
  sim_order: [A]
  real_order: [A]
  motor_ids: {A: 1}
  correction_group: {A: unchanged}
  encoder:
    ticks_per_rev: 4096
    zero_tick: {A: 2048}
  soft_limits_rad: {A: [-1.0, 1.0]}
imu:
  topic: /imu
  mount_offset_quat: [1.0, 0.0, 0.0, 0.0]
  negate_gyro_z: false
control:
  rate_hz: 50.0
  action_scale: 0.5
  q_default_sim: {A: 0.0}
  soft_start_seconds: 3.0
  soft_stop_seconds: 2.0
  action_scale_multiplier: 1.0
commands:
  velocity: {vx: 0.0, vy: 0.0, yaw_rate: 0.0}
  goal: {mode: fixed, x: 0.0, y: 0.0, z: 0.0, heading: 0.0}
"""


def test_example_configs_pin_forward_speed_estimate_to_point_one_five():
    assert load_deployment_config(_EXAMPLE_CFG_PATH).localization.forward_speed_estimate == 0.15
    assert load_deployment_config(_BINARY_CFG_PATH).localization.forward_speed_estimate == 0.15


def test_omitted_localization_block_defaults_to_point_one_five(tmp_path):
    cfg_path = tmp_path / "deployment.yaml"
    cfg_path.write_text(_MINIMAL_CFG_WITHOUT_LOCALIZATION)
    cfg = load_deployment_config(str(cfg_path))
    assert cfg.localization.forward_speed_estimate == 0.15


def test_custom_forward_speed_estimate_parses(tmp_path):
    cfg_path = tmp_path / "deployment.yaml"
    cfg_path.write_text(_MINIMAL_CFG_WITHOUT_LOCALIZATION + "\nlocalization:\n  forward_speed_estimate: 0.42\n")
    cfg = load_deployment_config(str(cfg_path))
    assert cfg.localization.forward_speed_estimate == 0.42
