# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import csv

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("onnxruntime")

import os

from _onnx_test_utils import export_tiny_mlp
from sim2real import control_loop
from sim2real.binary_profile import BinaryActionAdapter
from sim2real.command_source import make_command_source
from sim2real.deployment_config import load_deployment_config
from sim2real.dynamixel_bus import DryRunDynamixelBus
from sim2real.imu import FakeImu, ImuStaleError
from sim2real.joint_mapping import JointMapping
from sim2real.localization import DeadReckoningLocalizer
from sim2real.logging_utils import CsvRunLogger
from sim2real.policy_runner import PolicyRunner
from sim2real.profiles import PROFILES, make_obs_builder

_EXAMPLE_CFG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "deployment.example.yaml")
_BINARY_CFG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "deployment.binary.example.yaml")


def _make_env(tmp_path, profile_name, weight_scale=0.0):
    cfg = load_deployment_config(_EXAMPLE_CFG_PATH)
    jm = JointMapping(cfg.joints)
    spec = PROFILES[profile_name]
    onnx_path = export_tiny_mlp(tmp_path, spec.obs_dim, spec.action_dim, weight_scale=weight_scale)
    policy = PolicyRunner(onnx_path, spec)
    obs_builder = make_obs_builder(profile_name)
    bus = DryRunDynamixelBus(jm.motor_ids, initial_ticks=np.full(8, 2048))
    imu = FakeImu()
    command_source = make_command_source(profile_name, cfg.commands)
    localizer = DeadReckoningLocalizer() if profile_name == "goal" else None
    return cfg, jm, policy, obs_builder, bus, imu, command_source, localizer


def _make_binary_env(tmp_path):
    cfg = load_deployment_config(_BINARY_CFG_PATH)
    jm = JointMapping(cfg.joints)
    spec = PROFILES["binary"]
    onnx_path = export_tiny_mlp(tmp_path, spec.obs_dim, spec.action_dim, weight_scale=1.0)
    policy = PolicyRunner(onnx_path, spec)
    obs_builder = make_obs_builder("binary")
    bus = DryRunDynamixelBus(jm.motor_ids, initial_ticks=np.full(8, 1500))
    imu = FakeImu()
    command_source = make_command_source("binary", cfg.commands)
    adapter = BinaryActionAdapter(cfg.binary, cfg.joints.sim_order)
    return cfg, jm, policy, obs_builder, bus, imu, command_source, adapter


def test_dry_run_binary_profile_requires_localizer_and_adapter(tmp_path):
    cfg, jm, policy, obs_builder, bus, imu, command_source, adapter = _make_binary_env(tmp_path)
    with pytest.raises(ValueError, match="localizer"):
        control_loop.run(
            cfg,
            "binary",
            policy,
            obs_builder,
            jm,
            bus,
            imu,
            command_source,
            localizer=None,
            dry_run=True,
            rate_hz=50.0,
            duration_s=0.1,
            binary_adapter=adapter,
        )
    with pytest.raises(ValueError, match="binary_adapter"):
        control_loop.run(
            cfg,
            "binary",
            policy,
            obs_builder,
            jm,
            bus,
            imu,
            command_source,
            localizer=DeadReckoningLocalizer(),
            dry_run=True,
            rate_hz=50.0,
            duration_s=0.1,
            binary_adapter=None,
        )


def test_dry_run_binary_profile_completes_and_targets_are_stance_or_lift(tmp_path):
    cfg, jm, policy, obs_builder, bus, imu, command_source, adapter = _make_binary_env(tmp_path)
    log_path = tmp_path / "binary_run.csv"
    with CsvRunLogger(str(log_path), obs_dim=PROFILES["binary"].obs_dim, action_dim=6) as logger:
        control_loop.run(
            cfg,
            "binary",
            policy,
            obs_builder,
            jm,
            bus,
            imu,
            command_source,
            localizer=DeadReckoningLocalizer(),
            logger=logger,
            dry_run=True,
            rate_hz=50.0,
            duration_s=0.2,
            binary_adapter=adapter,
        )
    assert bus.torque_on is False
    with open(log_path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) > 2
    # every logged leg target must be (near) one of the two fixed levels
    leg_names = [n for n in jm.sim_order if n not in ("BackLink", "FrontLink")]
    leg_idx = [jm.sim_order.index(n) for n in leg_names]
    for row in rows:
        legs = np.array([float(row[f"target_sim_{i}"]) for i in leg_idx])
        near_stance = np.abs(legs - cfg.binary.stance_pos) < 0.05
        near_lift = np.abs(legs - cfg.binary.lift_pos) < 0.05
        assert np.all(near_stance | near_lift)


def test_dry_run_velocity_profile_completes_and_logs(tmp_path):
    cfg, jm, policy, obs_builder, bus, imu, command_source, localizer = _make_env(tmp_path, "velocity")
    log_path = tmp_path / "run.csv"

    with CsvRunLogger(str(log_path), obs_dim=PROFILES["velocity"].obs_dim, action_dim=8) as logger:
        control_loop.run(
            cfg,
            "velocity",
            policy,
            obs_builder,
            jm,
            bus,
            imu,
            command_source,
            localizer=None,
            logger=logger,
            dry_run=True,
            rate_hz=50.0,
            duration_s=0.2,
        )

    with open(log_path, newline="") as f:
        rows = list(csv.reader(f))
    assert len(rows) > 2  # header + at least a few iterations in 0.2s @ 50Hz
    assert bus.torque_on is False  # guaranteed off after the run


def test_dry_run_goal_profile_requires_localizer(tmp_path):
    cfg, jm, policy, obs_builder, bus, imu, command_source, localizer = _make_env(tmp_path, "goal")
    with pytest.raises(ValueError, match="localizer"):
        control_loop.run(
            cfg,
            "goal",
            policy,
            obs_builder,
            jm,
            bus,
            imu,
            command_source,
            localizer=None,
            dry_run=True,
            rate_hz=50.0,
            duration_s=0.1,
        )


def test_dry_run_goal_profile_completes_with_localizer(tmp_path):
    cfg, jm, policy, obs_builder, bus, imu, command_source, localizer = _make_env(tmp_path, "goal")
    control_loop.run(
        cfg,
        "goal",
        policy,
        obs_builder,
        jm,
        bus,
        imu,
        command_source,
        localizer=localizer,
        dry_run=True,
        rate_hz=50.0,
        duration_s=0.1,
    )
    assert bus.torque_on is False


def test_torque_disabled_even_on_exception(tmp_path):
    cfg, jm, policy, obs_builder, bus, imu, command_source, localizer = _make_env(tmp_path, "velocity")

    class BoomImu(FakeImu):
        def read(self):
            raise RuntimeError("simulated IMU failure")

    with pytest.raises(RuntimeError, match="simulated IMU failure"):
        control_loop.run(
            cfg,
            "velocity",
            policy,
            obs_builder,
            jm,
            bus,
            BoomImu(),
            command_source,
            localizer=None,
            dry_run=True,
            rate_hz=50.0,
            duration_s=1.0,
        )
    assert bus.torque_on is False


def test_stale_imu_trips_watchdog_and_stops_cleanly(tmp_path, capsys):
    # Simulates the bno08x_driver process dying mid-run: the IMU reader starts
    # out fine, then every read() after `stale_after` calls raises
    # ImuStaleError (as RosImuReader would once its topic goes silent past the
    # staleness threshold). The loop must catch this, trip the watchdog, print
    # a trip reason, and break -- not propagate the exception -- while the
    # try/finally soft_stop_ramp + torque-disable still run either way.
    class GoesStaleImu(FakeImu):
        def __init__(self, stale_after):
            self.calls = 0
            self.stale_after = stale_after

        def read(self):
            self.calls += 1
            if self.calls > self.stale_after:
                raise ImuStaleError("IMU data on topic '/imu' is stale (simulated)")
            return super().read()

    cfg, jm, policy, obs_builder, bus, _imu, command_source, localizer = _make_env(tmp_path, "velocity")
    stale_imu = GoesStaleImu(stale_after=3)

    # duration_s=5.0 @ 50Hz would be ~250 iterations if the loop ran to completion;
    # it must instead stop shortly after the IMU goes stale.
    control_loop.run(
        cfg,
        "velocity",
        policy,
        obs_builder,
        jm,
        bus,
        stale_imu,
        command_source,
        localizer=None,
        dry_run=True,
        rate_hz=50.0,
        duration_s=5.0,
    )

    assert stale_imu.calls == 4  # 3 good reads, then the failing 4th one that breaks the loop
    assert bus.torque_on is False  # soft_stop_ramp + torque_enable(False) still ran

    captured = capsys.readouterr()
    assert "watchdog tripped" in captured.out
    assert "stale" in captured.out


def test_imu_never_going_stale_does_not_trip_watchdog_early(tmp_path):
    # Control: a FakeImu (always fresh) over the same short run must complete
    # all its intended iterations rather than being cut short spuriously.
    cfg, jm, policy, obs_builder, bus, imu, command_source, localizer = _make_env(tmp_path, "velocity")
    control_loop.run(
        cfg,
        "velocity",
        policy,
        obs_builder,
        jm,
        bus,
        imu,
        command_source,
        localizer=None,
        dry_run=True,
        rate_hz=50.0,
        duration_s=0.2,
    )
    assert bus.torque_on is False


def test_dry_run_policy_actions_never_reach_bus(tmp_path):
    cfg, jm, policy, obs_builder, bus, imu, command_source, localizer = _make_env(
        tmp_path,
        "velocity",
        weight_scale=1000.0,  # large action -> would clearly move the robot if applied
    )
    control_loop.run(
        cfg,
        "velocity",
        policy,
        obs_builder,
        jm,
        bus,
        imu,
        command_source,
        localizer=None,
        dry_run=True,
        rate_hz=50.0,
        duration_s=0.1,
    )
    # soft_start/soft_stop ramps still move the dry-run bus toward q_default (expected),
    # but the policy's own large actions must never have been written mid-loop.
    end_positions = bus.read_positions()
    q_default_ticks = jm.sim_target_to_ticks(np.array([cfg.control.q_default_sim[n] for n in jm.sim_order]))
    assert np.max(np.abs(end_positions.astype(float) - q_default_ticks.astype(float))) < 2
