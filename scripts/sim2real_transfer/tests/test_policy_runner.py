# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("onnxruntime")

from _onnx_test_utils import export_tiny_mlp as _export_tiny_mlp
from sim2real.policy_runner import PolicyRunner
from sim2real.profiles import PROFILES


def test_matching_profile_loads_and_infers(tmp_path):
    spec = PROFILES["velocity"]
    onnx_path = _export_tiny_mlp(tmp_path, spec.obs_dim, spec.action_dim, weight_scale=2.0)

    runner = PolicyRunner(onnx_path, spec)
    obs = np.ones(spec.obs_dim, dtype=np.float32)
    action = runner.step(obs)

    assert action.shape == (spec.action_dim,)
    # weight = 2*eye(8, 33) applied to ones(33) -> first 8 obs entries * 2
    assert np.allclose(action, 2.0, atol=1e-4)
    assert np.allclose(runner.last_action, action)


def test_reset_zeroes_last_action(tmp_path):
    spec = PROFILES["velocity"]
    onnx_path = _export_tiny_mlp(tmp_path, spec.obs_dim, spec.action_dim)
    runner = PolicyRunner(onnx_path, spec)
    runner.step(np.ones(spec.obs_dim, dtype=np.float32))
    runner.reset()
    assert np.allclose(runner.last_action, 0.0)


def test_mismatched_obs_dim_raises_at_load(tmp_path):
    velocity_spec = PROFILES["velocity"]
    goal_spec = PROFILES["goal"]
    onnx_path = _export_tiny_mlp(tmp_path, velocity_spec.obs_dim, velocity_spec.action_dim)

    with pytest.raises(ValueError, match="obs_dim"):
        PolicyRunner(onnx_path, goal_spec)


def test_wrong_obs_shape_at_step_raises(tmp_path):
    spec = PROFILES["velocity"]
    onnx_path = _export_tiny_mlp(tmp_path, spec.obs_dim, spec.action_dim)
    runner = PolicyRunner(onnx_path, spec)
    with pytest.raises(ValueError):
        runner.step(np.ones(5, dtype=np.float32))


# HexapiFlattened.usd articulation order (robot.data.joint_names) and the old-USD order the deployment used to assume.
_HEXAPI_ORDER = [
    "FrontLink",
    "BackLink",
    "MiddleLeft",
    "MiddleRight",
    "FrontLeft",
    "FrontRight",
    "BackLeft",
    "BackRight",
]
_OLD_ORDER = ["BackLink", "FrontLink", "MiddleLeft", "MiddleRight", "BackLeft", "BackRight", "FrontLeft", "FrontRight"]


def _binary_onnx(tmp_path, sim_joint_names):
    pytest.importorskip("onnx")
    spec = PROFILES["binary"]
    return _export_tiny_mlp(tmp_path, spec.obs_dim, spec.action_dim, sim_joint_names=sim_joint_names), spec


def test_joint_order_matching_the_deployment_loads_despite_the_joint_suffix(tmp_path):
    onnx_path, spec = _binary_onnx(tmp_path, [f"{name}_Joint" for name in _HEXAPI_ORDER])
    PolicyRunner(onnx_path, spec, sim_joint_order=_HEXAPI_ORDER, require_joint_order=True)


def test_joint_order_mismatch_raises_at_load(tmp_path):
    onnx_path, spec = _binary_onnx(tmp_path, [f"{name}_Joint" for name in _HEXAPI_ORDER])
    with pytest.raises(ValueError, match="joint order mismatch"):
        PolicyRunner(onnx_path, spec, sim_joint_order=_OLD_ORDER)


def test_missing_joint_order_is_rejected_only_when_required(tmp_path):
    onnx_path, spec = _binary_onnx(tmp_path, None)
    PolicyRunner(onnx_path, spec, sim_joint_order=_HEXAPI_ORDER)
    with pytest.raises(ValueError, match="records no 'sim_joint_names'"):
        PolicyRunner(onnx_path, spec, sim_joint_order=_HEXAPI_ORDER, require_joint_order=True)


def test_joint_order_is_not_checked_without_a_deployment_order(tmp_path):
    onnx_path, spec = _binary_onnx(tmp_path, [f"{name}_Joint" for name in _HEXAPI_ORDER])
    PolicyRunner(onnx_path, spec)
    with pytest.raises(ValueError, match="needs sim_joint_order"):
        PolicyRunner(onnx_path, spec, require_joint_order=True)
