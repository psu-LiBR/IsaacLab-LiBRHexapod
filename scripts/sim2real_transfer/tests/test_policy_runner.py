import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("onnxruntime")

from sim2real.policy_runner import PolicyRunner
from sim2real.profiles import PROFILES

from _onnx_test_utils import export_tiny_mlp as _export_tiny_mlp


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
