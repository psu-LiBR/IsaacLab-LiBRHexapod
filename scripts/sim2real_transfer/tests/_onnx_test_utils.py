# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared helper for tests that need a tiny synthetic ONNX 'policy' graph, mirroring
the shape/name conventions of isaaclab_rl's export_policy_as_onnx (obs->actions,
batch-size-1, opset 18)."""

import torch


def export_tiny_mlp(tmp_path, obs_dim, action_dim, weight_scale=1.0, filename="policy.onnx", sim_joint_names=None):
    class TinyPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(obs_dim, action_dim, bias=False)
            with torch.no_grad():
                self.linear.weight.copy_(weight_scale * torch.eye(action_dim, obs_dim))

        def forward(self, obs):
            return self.linear(obs)

    model = TinyPolicy()
    onnx_path = tmp_path / filename
    dummy_obs = torch.zeros(1, obs_dim)
    torch.onnx.export(
        model, dummy_obs, str(onnx_path),
        input_names=["obs"], output_names=["actions"], opset_version=18,
    )
    if sim_joint_names is not None:
        # Same metadata key and JSON encoding as export_binary_onnx.py.
        import json

        import onnx

        proto = onnx.load(str(onnx_path))
        entry = proto.metadata_props.add()
        entry.key, entry.value = "sim_joint_names", json.dumps(sim_joint_names)
        onnx.save(proto, str(onnx_path))
    return str(onnx_path)
