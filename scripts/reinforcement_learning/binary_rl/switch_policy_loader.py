# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Strict inference for the six approved methods, including mask and normalization."""

import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F


class GreedyPolicy:
    def __init__(self, checkpoint, method_id, device):
        file = Path(checkpoint)
        ckpt = torch.load(file, map_location=device, weights_only=False)
        key = "q_network" if method_id in (1, 2) else "policy"
        state = ckpt[key]
        self.mask = state.get("_action_mask")
        weights = {(k[4:] if k.startswith("net.") else k): v for k, v in state.items() if k != "_action_mask"}
        dims = [32, 128, 128, 128, 64]
        layers = []
        for i in range(4):
            layers.append(torch.nn.Linear(dims[i], dims[i + 1]))
            if i < 3:
                layers.append(torch.nn.ELU())
        self.net = torch.nn.Sequential(*layers).to(device)
        self.net.load_state_dict(weights, strict=True)
        self.net.eval()
        self.scaler = None
        if method_id in (3, 4):
            from skrl.resources.preprocessors.torch import RunningStandardScaler

            self.scaler = RunningStandardScaler(size=32, device=device)
            self.scaler.load_state_dict(ckpt["observation_preprocessor"], strict=True)
            self.scaler.eval()
        elif "observation_preprocessor" in ckpt:
            raise ValueError("Unexpected off-policy observation scaler")
        meta = json.loads((file.parent / "run_meta.json").read_text())
        if method_id == 4:
            legal = meta["action_mask"]["legal_actions"]
            expected = torch.zeros(64, dtype=torch.bool, device=device)
            expected[legal] = True
            assert int(expected.sum()) == 24
            if self.mask is not None:
                assert torch.equal(self.mask, expected)
            self.mask = expected
        else:
            assert self.mask is None and "action_mask" not in meta
        # Compare a second direct linear/ELU route with the loaded Sequential.
        gen = torch.Generator(device=device).manual_seed(9137)
        x = torch.randn(1024, 32, device=device, generator=gen)
        with torch.no_grad():
            z = self.scaler(x, train=False) if self.scaler is not None else x
            manual = z
            for i in (0, 2, 4, 6):
                manual = F.linear(manual, weights[f"{i}.weight"], weights[f"{i}.bias"])
                if i < 6:
                    manual = F.elu(manual)
            torch.testing.assert_close(self.net(z), manual, rtol=1e-6, atol=1e-6)
            chosen = self(x)
            if self.mask is not None:
                assert self.mask[chosen].all()
        self.audit = {
            "checkpoint": str(file),
            "sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
            "method_id": method_id,
            "scaler_loaded": self.scaler is not None,
            "legal_actions": None if self.mask is None else self.mask.nonzero().flatten().tolist(),
            "direct_forward_crosscheck": True,
        }

    @torch.no_grad()
    def __call__(self, x):
        if self.scaler is not None:
            x = self.scaler(x, train=False)
        logits = self.net(x)
        assert torch.isfinite(logits).all()
        if self.mask is not None:
            logits = logits.masked_fill(~self.mask, float("-inf"))
        return logits.argmax(dim=-1)
