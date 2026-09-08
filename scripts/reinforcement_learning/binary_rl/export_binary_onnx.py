# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Export a trained binary-contact policy to ONNX for sim-to-real deployment.

Produces a single-input / single-output graph::

    obs [1, obs_dim] float32  ->  action [1, 6] float32 in {-1, +1}

matching the ``binary`` profile of ``scripts/sim2real_transfer`` (``obs_dim`` 32, six leg
contact bits). The graph folds in everything the deployment ``PolicyRunner`` should not
need to know about the training framework:

* running observation normalisation (PPO checkpoints only), as a fixed affine + clip,
* the greedy argmax over the 64 six-bit gait patterns,
* the legal-action restriction recorded in ``run_meta.json`` for a masked-PPO run,
* the bit decode (action index -> ``+-1`` vector), as a constant ``[64, 6]`` lookup
  (bit-identical to ``DiscreteBitsActionWrapper.decode`` / ``action_to_pm1``).

Works for the maintained comparison set -- DQN, Double-DQN, categorical PPO, masked PPO,
SAC-D -- all of which store a plain MLP (64 Q-values or 64 logits) under ``q_network`` or
``policy``. Torch-only; no Isaac Sim, no skrl.

Run (from the repo root)::

    isaaclab.bat -p scripts/reinforcement_learning/binary_rl/export_binary_onnx.py ^
        --checkpoint runs_binary/ppo_masked_s42/checkpoints/agent_100000.pt ^
        --out policies/ppo_masked_s42.onnx

then validate on the deployment host::

    python scripts/sim2real_transfer/tools/validate_onnx.py --mode direct ^
        --policy policies/ppo_masked_s42.onnx --profile binary --trace <sim_trace.csv>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from binary_action_mask import N_ACTIONS, action_to_pm1  # noqa: E402

# skrl RunningStandardScaler defaults (resources/preprocessors/torch/running_standard_scaler.py)
_SCALER_EPS = 1.0e-8
_SCALER_CLIP = 5.0


def _extract_mlp_state(ck: dict | object) -> dict[str, torch.Tensor]:
    """Pull the policy/Q MLP's ``Sequential`` state dict out of a checkpoint.

    Mirrors ``eval_protocol.py``: unwrap a ``q_network`` / ``policy`` sub-dict, strip a
    ``net.`` prefix, and keep only the numbered layer tensors (dropping mixin buffers
    such as ``MaskedCategoricalMixin._action_mask``).
    """
    state = ck
    if isinstance(ck, dict):
        for key in ("q_network", "policy"):
            if key in ck:
                state = ck[key]
                break
    state = {(k[4:] if k.startswith("net.") else k): v for k, v in state.items()}
    state = {k: v for k, v in state.items() if k.split(".")[0].isdigit()}
    if not state:
        raise ValueError("no numbered MLP layer tensors found in checkpoint")
    return state


def _build_mlp(state: dict[str, torch.Tensor]) -> torch.nn.Sequential:
    """Reconstruct the MLP from its flat state dict (indices as keys)."""
    has_ln = any(k.endswith(".weight") and v.dim() == 1 for k, v in state.items())
    idxs = sorted({int(k.split(".")[0]) for k in state})
    mods: list[torch.nn.Module] = []
    for i in range(max(idxs) + 1):
        w = state.get(f"{i}.weight")
        if w is None:
            mods.append(torch.nn.ReLU() if has_ln else torch.nn.ELU())
        elif w.dim() == 2:
            mods.append(torch.nn.Linear(w.shape[1], w.shape[0]))
        else:
            mods.append(torch.nn.LayerNorm(w.shape[0]))
    net = torch.nn.Sequential(*mods)
    net.load_state_dict(state)
    net.eval()
    return net


def _load_run_meta(checkpoint_path: str, explicit: str | None) -> dict:
    path = explicit or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(checkpoint_path))), "run_meta.json")
    if os.path.isfile(path):
        with open(path) as f:
            return json.load(f)
    return {}


class _BinaryPolicyExport(torch.nn.Module):
    """obs -> (optional normalise) -> MLP -> (optional legal mask) -> argmax -> +-1 bits."""

    def __init__(
        self,
        net: torch.nn.Sequential,
        *,
        obs_mean: torch.Tensor | None,
        obs_var: torch.Tensor | None,
        legal_bias: torch.Tensor,
        decode_table: torch.Tensor,
    ):
        super().__init__()
        self.net = net
        self.has_scaler = obs_mean is not None
        if self.has_scaler:
            self.register_buffer("obs_mean", obs_mean.float())
            self.register_buffer("obs_std", torch.sqrt(obs_var.float()) + _SCALER_EPS)
        self.register_buffer("legal_bias", legal_bias)  # [64]
        self.register_buffer("decode_table", decode_table)  # [64, 6] float32 in {-1, +1}

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = obs
        if self.has_scaler:
            x = torch.clamp((x - self.obs_mean) / self.obs_std, -_SCALER_CLIP, _SCALER_CLIP)
        scores = self.net(x) + self.legal_bias
        idx = torch.argmax(scores, dim=-1)
        return self.decode_table.index_select(0, idx)


def build_export_module(checkpoint_path: str, run_meta_path: str | None = None) -> tuple[_BinaryPolicyExport, dict]:
    """Build the exportable module + an info dict, without writing anything."""
    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    meta = _load_run_meta(checkpoint_path, run_meta_path)

    net = _build_mlp(_extract_mlp_state(ck))
    linears = [m for m in net if isinstance(m, torch.nn.Linear)]
    obs_dim = linears[0].in_features
    out_dim = linears[-1].out_features
    if out_dim != N_ACTIONS:
        raise ValueError(
            f"expected an MLP with {N_ACTIONS} outputs (one per 6-bit gait pattern), got {out_dim}. "
            "Distributional heads (C51 / QR-DQN) are archived and not exportable here."
        )
    if meta.get("obs_dim") not in (None, obs_dim):
        raise ValueError(f"run_meta obs_dim={meta['obs_dim']} disagrees with checkpoint obs_dim={obs_dim}")

    obs_mean = obs_var = None
    if isinstance(ck, dict) and "observation_preprocessor" in ck:
        pp = ck["observation_preprocessor"]
        obs_mean = pp["running_mean"].detach().cpu()
        obs_var = pp["running_variance"].detach().cpu()

    legal_bias = torch.zeros(N_ACTIONS)
    legal_actions = None
    am = meta.get("action_mask")
    if isinstance(am, dict) and am.get("legal_actions"):
        legal_actions = sorted(int(a) for a in am["legal_actions"])
        legal_bias = torch.full((N_ACTIONS,), -1.0e9)
        legal_bias[torch.tensor(legal_actions)] = 0.0

    decode_table = action_to_pm1(torch.arange(N_ACTIONS)).float()  # [64, 6] in {-1, +1}

    module = _BinaryPolicyExport(
        net,
        obs_mean=obs_mean,
        obs_var=obs_var,
        legal_bias=legal_bias,
        decode_table=decode_table,
    )
    module.eval()
    info = {
        "algo": meta.get("algo", "unknown"),
        "obs_dim": obs_dim,
        "n_actions": out_dim,
        "action_dim": 6,
        "obs_normalization": obs_mean is not None,
        "legal_actions": legal_actions,
        "n_legal_actions": len(legal_actions) if legal_actions is not None else N_ACTIONS,
    }
    return module, info


def _self_check(onnx_path: str, module: _BinaryPolicyExport, obs_dim: int, n: int = 64) -> float:
    """Run random obs through onnxruntime and the torch module; return the max mismatch."""
    import numpy as np
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    rng = np.random.default_rng(0)
    worst = 0.0
    with torch.no_grad():
        for _ in range(n):
            obs = rng.uniform(-3.0, 3.0, size=(1, obs_dim)).astype(np.float32)
            ort_out = sess.run(None, {in_name: obs})[0]
            torch_out = module(torch.from_numpy(obs)).numpy()
            worst = max(worst, float(np.abs(ort_out - torch_out).max()))
            if not set(np.unique(ort_out).tolist()).issubset({-1.0, 1.0}):
                raise AssertionError(f"ONNX output not in {{-1, +1}}: {np.unique(ort_out)}")
    return worst


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="trained .pt checkpoint")
    parser.add_argument("--out", default="", help="output .onnx path (default: checkpoint path with .onnx)")
    parser.add_argument("--run_meta", default="", help="override run_meta.json path (default: sibling of the run dir)")
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--no_check", action="store_true", help="skip the onnxruntime round-trip self-check")
    args = parser.parse_args()

    out_path = args.out or os.path.splitext(args.checkpoint)[0] + ".onnx"
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    module, info = build_export_module(args.checkpoint, args.run_meta or None)
    print(f"[export_binary_onnx] {info}")

    dummy = torch.zeros(1, info["obs_dim"])
    with warnings.catch_warnings():
        # The legacy TorchScript exporter is used deliberately: the new torch.export path
        # emits a non-ASCII progress banner that crashes on a cp1252 console (Windows), and
        # the graph here (affine + MLP + argmax + gather) is exactly what legacy handles well.
        warnings.filterwarnings("ignore", category=DeprecationWarning, module="torch.onnx")
        torch.onnx.export(
            module,
            dummy,
            out_path,
            input_names=["obs"],
            output_names=["action"],
            opset_version=args.opset,
            dynamo=False,
        )
    print(f"[export_binary_onnx] wrote {out_path}")

    if not args.no_check:
        worst = _self_check(out_path, module, info["obs_dim"])
        status = "OK" if worst == 0.0 else f"MAX DIFF {worst:.3e}"
        print(f"[export_binary_onnx] onnxruntime round-trip: {status}")
        if worst != 0.0:
            raise SystemExit("ONNX self-check failed: onnxruntime and torch disagree")


if __name__ == "__main__":
    main()
