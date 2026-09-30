# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CPU regressions for packaged metadata and complete-cycle measurements."""

import ast
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts/reinforcement_learning/binary_rl"
sys.path.insert(0, str(SCRIPTS))
from switch_metrics import summarize  # noqa: E402


@pytest.mark.parametrize("heading", [0.0, math.pi / 2])
def test_complete_cycles_and_initial_heading_projection(heading):
    phase = np.arange(1, 301)[:, None]
    actions = ((phase // 25) % 2).astype(np.int64)
    forward = np.linspace(0, 0.7 * 0.315 * 6, 300)
    trajectory = np.zeros((300, 1, 3))
    trajectory[:, 0, 0] = forward * math.cos(heading)
    trajectory[:, 0, 1] = forward * math.sin(heading)
    quat = np.array([[0, 0, math.sin(heading / 2), math.cos(heading / 2)]])
    no_contact = np.zeros((300, 1), dtype=bool)
    result = summarize(
        actions,
        np.array([0]),
        phase,
        trajectory,
        quat,
        np.tile(quat, (300, 1, 1)),
        np.ones((300, 1)),
        no_contact,
        no_contact,
    )
    assert result["complete_cycles"] == 5
    assert result["per_leg_max"] == [2, 0, 0, 0, 0, 0]
    assert result["dwell_min_s"] == pytest.approx(0.5)
    assert result["forward_bl_per_cycle"] == pytest.approx(0.7)
    assert result["lateral_abs_max_bl"] == pytest.approx(0, abs=1e-12)
    assert result["yaw_abs_max_deg"] == pytest.approx(0, abs=1e-12)


def test_general_evaluator_reads_metadata_next_to_packaged_checkpoint(tmp_path):
    # Import only pure builders; importing the CLI module starts Isaac Sim.
    tree = ast.parse((SCRIPTS / "eval_protocol.py").read_text())
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
    checkpoint = tmp_path / "candidate" / "policy.pt"
    checkpoint.parent.mkdir()
    net = torch.nn.Sequential(torch.nn.Linear(32, 64))
    with torch.no_grad():
        net[0].weight.zero_()
        net[0].bias.copy_(torch.arange(64))
    torch.save({"policy": net.state_dict()}, checkpoint)
    (checkpoint.parent / "run_meta.json").write_text(json.dumps({"action_mask": {"legal_actions": [2, 3]}}))
    policy, metadata = namespace["make_net_policy"](str(checkpoint))
    assert metadata["action_mask_legal"] == 2
    assert policy(torch.zeros(1, 32), 0).item() == 3  # Unmasked argmax would be 63.
