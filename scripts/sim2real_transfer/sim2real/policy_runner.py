# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""onnxruntime wrapper around an exported hexapod policy.

The ONNX graphs produced by `isaaclab_rl.rsl_rl.exporter.export_policy_as_onnx`
take raw (un-normalized) float32 obs of shape [1, obs_dim] and return raw
(pre action-scale, pre joint-reorder) actions of shape [1, action_dim] -- see
CLAUDE.md's sim2real plan for the full derivation. This wrapper validates the
loaded graph's shapes against the requested profile at construction time so a
mismatched --policy/--profile pairing fails immediately instead of silently
producing garbage actions.

Policies exported by ``export_binary_onnx.py`` also carry the sim's joint order as ONNX
metadata. The joint_pos / joint_vel observations are filled by position, so a policy fed
joints in a different order than it was trained on gets wrong observations without any
shape error; the optional ``sim_joint_order`` check catches that at construction.
"""

from __future__ import annotations

import json

import numpy as np
import onnxruntime as ort

from .profiles import ProfileSpec

# ONNX metadata key written by scripts/reinforcement_learning/binary_rl/export_binary_onnx.py.
SIM_JOINT_NAMES_KEY = "sim_joint_names"


def _joint_key(name: str) -> str:
    """The sim names its joints ``FrontLink_Joint``; the deployment config uses ``FrontLink``."""
    return name.removesuffix("_Joint")


class PolicyRunner:
    """
    Args:
        onnx_path: Exported policy.
        profile: Obs/action layout the policy must match.
        sim_joint_order: The joint order the deployment feeds the policy (``joints.sim_order``). If given and the
            graph records the order it was trained with, the two must match.
        require_joint_order: Also fail when the graph records no joint order. Needs ``sim_joint_order``.
    """

    def __init__(
        self,
        onnx_path: str,
        profile: ProfileSpec,
        sim_joint_order: list[str] | None = None,
        require_joint_order: bool = False,
    ):
        self.profile = profile
        self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError(
                f"expected a single-input/single-output ONNX graph, got {len(inputs)} inputs and {len(outputs)} outputs"
            )
        self._input_name = inputs[0].name
        self._output_name = outputs[0].name

        input_obs_dim = inputs[0].shape[-1]
        output_action_dim = outputs[0].shape[-1]
        if input_obs_dim != profile.obs_dim:
            raise ValueError(
                f"ONNX graph '{onnx_path}' expects obs_dim={input_obs_dim}, but "
                f"profile '{profile.name}' expects obs_dim={profile.obs_dim} -- "
                f"wrong --profile for this checkpoint?"
            )
        if output_action_dim != profile.action_dim:
            raise ValueError(
                f"ONNX graph '{onnx_path}' outputs action_dim={output_action_dim}, but "
                f"profile '{profile.name}' expects action_dim={profile.action_dim}"
            )

        self._check_joint_order(onnx_path, sim_joint_order, require_joint_order)

        self.last_action = np.zeros(profile.action_dim, dtype=np.float32)

    def _check_joint_order(self, onnx_path: str, sim_joint_order: list[str] | None, required: bool) -> None:
        if required and sim_joint_order is None:
            raise ValueError("require_joint_order needs sim_joint_order to compare against")
        if sim_joint_order is None:
            return
        raw = self.session.get_modelmeta().custom_metadata_map.get(SIM_JOINT_NAMES_KEY)
        if raw is None:
            if required:
                raise ValueError(
                    f"ONNX graph '{onnx_path}' records no '{SIM_JOINT_NAMES_KEY}', so its joint order cannot be "
                    "checked against joints.sim_order. Re-export it with export_binary_onnx.py from a run whose "
                    "run_meta.json has the joint order."
                )
            return
        trained = [_joint_key(name) for name in json.loads(raw)]
        deployed = [_joint_key(name) for name in sim_joint_order]
        if trained != deployed:
            raise ValueError(
                f"joint order mismatch for '{onnx_path}': the policy was trained with {trained} but "
                f"joints.sim_order is {deployed}. The joint_pos / joint_vel observations are filled by position, "
                "so the policy would see the wrong joints. Fix sim_order in the deployment config."
            )

    def reset(self) -> None:
        self.last_action = np.zeros(self.profile.action_dim, dtype=np.float32)

    def step(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        if obs.shape != (self.profile.obs_dim,):
            raise ValueError(f"obs must be shape ({self.profile.obs_dim},), got {obs.shape}")

        outputs = self.session.run([self._output_name], {self._input_name: obs[None, :]})
        raw_action = outputs[0][0].astype(np.float32)
        self.last_action = raw_action
        return raw_action
