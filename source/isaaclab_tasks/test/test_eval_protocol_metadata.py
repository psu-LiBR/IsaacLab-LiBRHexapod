# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CPU test that ``eval_protocol.py`` finds ``run_meta.json`` next to a packaged checkpoint (needs ``torch``)."""

import ast
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

EVAL_PROTOCOL = Path(__file__).resolve().parents[3] / "scripts/reinforcement_learning/binary_rl/eval_protocol.py"


def _make_net_policy():
    # Importing the CLI module starts Isaac Sim, so exec only the pure policy builders from its source.
    tree = ast.parse(EVAL_PROTOCOL.read_text())
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {"build_mlp", "make_net_policy"}]
    namespace = {
        "torch": torch,
        "os": os,
        "json": json,
        "device": "cpu",
        "obs_dim": 32,
        "N_ACT": 64,
        "args": SimpleNamespace(v_min=-10, v_max=10),
    }
    exec(compile(ast.Module(body=functions, type_ignores=[]), "policy_builders", "exec"), namespace)
    return namespace["make_net_policy"]


def _save_checkpoint(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    net = torch.nn.Sequential(torch.nn.Linear(32, 64))
    with torch.no_grad():
        net[0].weight.zero_()
        net[0].bias.copy_(torch.arange(64))
    torch.save({"policy": net.state_dict()}, path)


def test_reads_metadata_next_to_packaged_checkpoint(tmp_path):
    checkpoint = tmp_path / "candidate" / "policy.pt"
    _save_checkpoint(checkpoint)
    (checkpoint.parent / "run_meta.json").write_text(json.dumps({"action_mask": {"legal_actions": [2, 3]}}))
    policy, metadata = _make_net_policy()(str(checkpoint))
    assert metadata["action_mask_legal"] == 2
    assert policy(torch.zeros(1, 32), 0).item() == 3  # Unmasked argmax would be 63.


def test_metadata_next_to_checkpoint_beats_run_directory_metadata(tmp_path):
    checkpoint = tmp_path / "run" / "checkpoints" / "agent_10.pt"
    _save_checkpoint(checkpoint)
    (tmp_path / "run" / "run_meta.json").write_text(json.dumps({"action_mask": {"legal_actions": [5, 6]}}))
    (checkpoint.parent / "run_meta.json").write_text(json.dumps({"action_mask": {"legal_actions": [2, 3]}}))
    policy, _ = _make_net_policy()(str(checkpoint))
    assert policy(torch.zeros(1, 32), 0).item() == 3


def test_falls_back_to_run_directory_metadata(tmp_path):
    checkpoint = tmp_path / "run" / "checkpoints" / "agent_10.pt"
    _save_checkpoint(checkpoint)
    (tmp_path / "run" / "run_meta.json").write_text(json.dumps({"action_mask": {"legal_actions": [5, 6]}}))
    policy, metadata = _make_net_policy()(str(checkpoint))
    assert metadata["action_mask_legal"] == 2
    assert policy(torch.zeros(1, 32), 0).item() == 6
