# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the binary-contact ONNX exporter.

Loads ``scripts/reinforcement_learning/binary_rl/export_binary_onnx.py`` by path (it is a
script-tree module) and checks that the exported graph matches the training-time greedy
policy: obs -> optional running-normalisation -> MLP -> optional legal mask -> argmax ->
6-dim ``+-1`` bit vector. Needs ``torch`` + ``onnxruntime``; no Isaac Sim.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
ort = pytest.importorskip("onnxruntime")

BINARY_RL_DIR = Path(__file__).parents[3] / "scripts" / "reinforcement_learning" / "binary_rl"
sys.path.insert(0, str(BINARY_RL_DIR))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, BINARY_RL_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


E = _load("export_binary_onnx")
bam = _load("binary_action_mask")
bc = _load("binary_common")

OBS_DIM = 32


def _action_index_from_bits(bits) -> int:
    return int(sum((1 if b > 0 else 0) << i for i, b in enumerate(bits)))


def _write_run(tmp_path: Path, ckpt: dict, meta: dict, name: str = "run") -> Path:
    run_dir = tmp_path / name
    (run_dir / "checkpoints").mkdir(parents=True)
    ckpt_path = run_dir / "checkpoints" / "agent_10.pt"
    torch.save(ckpt, ckpt_path)
    (run_dir / "run_meta.json").write_text(json.dumps(meta))
    return ckpt_path


def _run_onnx(onnx_path: Path, obs):
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    import numpy as np

    return sess.run(None, {name: np.asarray(obs, dtype=np.float32)})[0]


def test_sacd_style_export_matches_argmax_decode(tmp_path):
    torch.manual_seed(0)
    net = bc.mlp(OBS_DIM, bam.N_ACTIONS)
    ckpt_path = _write_run(
        tmp_path,
        {"policy": net.state_dict(), "step": 10},
        {"schema": 1, "algo": "sac_d", "obs_dim": OBS_DIM, "n_actions": bam.N_ACTIONS},
        "sacd",
    )
    module, info = E.build_export_module(str(ckpt_path))
    assert info["obs_dim"] == OBS_DIM and info["n_actions"] == 64 and info["action_dim"] == 6
    assert info["obs_normalization"] is False and info["n_legal_actions"] == 64

    onnx_path = tmp_path / "sacd.onnx"
    torch.onnx.export(
        module,
        torch.zeros(1, OBS_DIM),
        str(onnx_path),
        input_names=["obs"],
        output_names=["action"],
        opset_version=18,
        dynamo=False,
    )

    import numpy as np

    rng = np.random.default_rng(1)
    for _ in range(50):
        obs = rng.uniform(-3, 3, size=(1, OBS_DIM)).astype(np.float32)
        out = _run_onnx(onnx_path, obs)
        assert out.shape == (1, 6)
        assert set(np.unique(out).tolist()).issubset({-1.0, 1.0})
        with torch.no_grad():
            want_idx = int(torch.argmax(net(torch.from_numpy(obs)), dim=-1))
        assert _action_index_from_bits(out[0]) == want_idx


def test_masked_ppo_style_export_folds_normalisation_and_respects_legal_set(tmp_path):
    torch.manual_seed(2)
    net = bc.mlp(OBS_DIM, bam.N_ACTIONS)
    policy_state = {f"net.{k}": v for k, v in net.state_dict().items()}
    running_mean = torch.randn(OBS_DIM, dtype=torch.float64)
    running_var = torch.rand(OBS_DIM, dtype=torch.float64) + 0.25
    legal = bam.legal_action_indices().tolist()
    ckpt_path = _write_run(
        tmp_path,
        {
            "policy": policy_state,
            "observation_preprocessor": {
                "running_mean": running_mean,
                "running_variance": running_var,
                "current_count": torch.tensor(7.0, dtype=torch.float64),
            },
        },
        {
            "schema": 1,
            "algo": "ppo_masked",
            "obs_dim": OBS_DIM,
            "n_actions": bam.N_ACTIONS,
            "action_mask": {"min_stance_legs": 4, "whitelist": [25, 38], "legal_actions": legal},
        },
        "ppo",
    )
    module, info = E.build_export_module(str(ckpt_path))
    assert info["obs_normalization"] is True
    assert info["n_legal_actions"] == 24

    onnx_path = tmp_path / "ppo.onnx"
    torch.onnx.export(
        module,
        torch.zeros(1, OBS_DIM),
        str(onnx_path),
        input_names=["obs"],
        output_names=["action"],
        opset_version=18,
        dynamo=False,
    )

    import numpy as np

    mean = running_mean.float()
    std = torch.sqrt(running_var.float()) + 1e-8
    legal_set = set(legal)
    rng = np.random.default_rng(3)
    for _ in range(100):
        obs = rng.uniform(-4, 4, size=(1, OBS_DIM)).astype(np.float32)
        out = _run_onnx(onnx_path, obs)
        assert _action_index_from_bits(out[0]) in legal_set
        with torch.no_grad():
            x = torch.clamp((torch.from_numpy(obs) - mean) / std, -5.0, 5.0)
            scores = net(x).clone()
            illegal = [a for a in range(64) if a not in legal_set]
            scores[0, illegal] = -1e9
            want_idx = int(torch.argmax(scores, dim=-1))
        assert _action_index_from_bits(out[0]) == want_idx


def test_export_rejects_non_64_output_head(tmp_path):
    net = bc.mlp(OBS_DIM, 51)  # e.g. a QR-DQN-ish head
    ckpt_path = _write_run(
        tmp_path, {"policy": net.state_dict()}, {"schema": 1, "algo": "qrdqn", "obs_dim": OBS_DIM}, "bad"
    )
    with pytest.raises(ValueError, match="64 outputs"):
        E.build_export_module(str(ckpt_path))
