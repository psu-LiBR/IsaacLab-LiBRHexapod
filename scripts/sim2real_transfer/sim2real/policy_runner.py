"""onnxruntime wrapper around an exported hexapod policy.

The ONNX graphs produced by `isaaclab_rl.rsl_rl.exporter.export_policy_as_onnx`
take raw (un-normalized) float32 obs of shape [1, obs_dim] and return raw
(pre action-scale, pre joint-reorder) actions of shape [1, action_dim] -- see
CLAUDE.md's sim2real plan for the full derivation. This wrapper validates the
loaded graph's shapes against the requested profile at construction time so a
mismatched --policy/--profile pairing fails immediately instead of silently
producing garbage actions.
"""

from __future__ import annotations

import numpy as np
import onnxruntime as ort

from .profiles import ProfileSpec


class PolicyRunner:
    def __init__(self, onnx_path: str, profile: ProfileSpec):
        self.profile = profile
        self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError(
                f"expected a single-input/single-output ONNX graph, got "
                f"{len(inputs)} inputs and {len(outputs)} outputs"
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

        self.last_action = np.zeros(profile.action_dim, dtype=np.float32)

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
