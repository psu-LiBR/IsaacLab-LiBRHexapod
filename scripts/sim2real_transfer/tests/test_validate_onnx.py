# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import csv
import os
import sys

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("onnxruntime")

from _onnx_test_utils import export_tiny_mlp

_TOOLS_DIR = os.path.join(os.path.dirname(__file__), "..", "tools")
sys.path.insert(0, _TOOLS_DIR)
import validate_onnx  # noqa: E402
from sim2real.policy_runner import PolicyRunner
from sim2real.profiles import NUM_JOINTS, PROFILES, make_obs_builder


def _write_direct_trace(path, obs_rows, action_rows):
    obs_dim = len(obs_rows[0])
    action_dim = len(action_rows[0])
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step"] + [f"obs_{i}" for i in range(obs_dim)] + [f"action_{i}" for i in range(action_dim)])
        for i, (obs, action) in enumerate(zip(obs_rows, action_rows)):
            writer.writerow([i] + list(obs) + list(action))


def test_direct_check_passes_for_exact_trace(tmp_path):
    spec = PROFILES["velocity"]
    onnx_path = export_tiny_mlp(tmp_path, spec.obs_dim, spec.action_dim, weight_scale=2.0)
    policy = PolicyRunner(onnx_path, spec)

    rng = np.random.default_rng(0)
    obs_rows = [rng.uniform(-1, 1, spec.obs_dim) for _ in range(5)]
    action_rows = [policy.step(obs) for obs in obs_rows]

    trace_path = tmp_path / "trace.csv"
    _write_direct_trace(trace_path, obs_rows, action_rows)
    rows = validate_onnx.load_trace(str(trace_path))

    ok = validate_onnx.run_direct_check(rows, policy, spec.obs_dim, spec.action_dim, tol=1e-4)
    assert ok is True


def test_direct_check_fails_for_perturbed_trace(tmp_path):
    spec = PROFILES["velocity"]
    onnx_path = export_tiny_mlp(tmp_path, spec.obs_dim, spec.action_dim, weight_scale=2.0)
    policy = PolicyRunner(onnx_path, spec)

    rng = np.random.default_rng(1)
    obs_rows = [rng.uniform(-1, 1, spec.obs_dim) for _ in range(3)]
    action_rows = [policy.step(obs) + 5.0 for obs in obs_rows]  # deliberately wrong

    trace_path = tmp_path / "trace.csv"
    _write_direct_trace(trace_path, obs_rows, action_rows)
    rows = validate_onnx.load_trace(str(trace_path))

    ok = validate_onnx.run_direct_check(rows, policy, spec.obs_dim, spec.action_dim, tol=1e-4)
    assert ok is False


def test_pipeline_check_passes_for_consistent_trace(tmp_path):
    spec = PROFILES["velocity"]
    onnx_path = export_tiny_mlp(tmp_path, spec.obs_dim, spec.action_dim, weight_scale=1.0)
    policy = PolicyRunner(onnx_path, spec)
    obs_builder = make_obs_builder("velocity")

    rng = np.random.default_rng(2)
    q_default = np.zeros(NUM_JOINTS)
    trace_path = tmp_path / "pipeline_trace.csv"
    header = (
        ["step"]
        + [f"gyro_{i}" for i in range(3)]
        + [f"gravity_{i}" for i in range(3)]
        + [f"command_{i}" for i in range(spec.command_dim)]
        + [f"joint_pos_{i}" for i in range(NUM_JOINTS)]
        + [f"joint_vel_{i}" for i in range(NUM_JOINTS)]
        + [f"last_action_{i}" for i in range(NUM_JOINTS)]
        + [f"obs_{i}" for i in range(spec.obs_dim)]
        + [f"action_{i}" for i in range(spec.action_dim)]
    )
    with open(trace_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for step in range(4):
            gyro = rng.uniform(-1, 1, 3)
            gravity = np.array([0.0, 0.0, -1.0])
            command = rng.uniform(-1, 1, spec.command_dim)
            joint_pos = rng.uniform(-1, 1, NUM_JOINTS)
            joint_vel = rng.uniform(-1, 1, NUM_JOINTS)
            last_action = rng.uniform(-1, 1, NUM_JOINTS)
            obs = obs_builder.build(gyro, gravity, joint_pos, joint_vel, last_action, q_default, command)
            action = policy.step(obs)
            writer.writerow(
                [step]
                + list(gyro)
                + list(gravity)
                + list(command)
                + list(joint_pos)
                + list(joint_vel)
                + list(last_action)
                + list(obs)
                + list(action)
            )

    rows = validate_onnx.load_trace(str(trace_path))
    ok = validate_onnx.run_pipeline_check(
        rows, policy, spec.obs_dim, spec.action_dim, spec.command_dim, q_default, tol=1e-3
    )
    assert ok is True
