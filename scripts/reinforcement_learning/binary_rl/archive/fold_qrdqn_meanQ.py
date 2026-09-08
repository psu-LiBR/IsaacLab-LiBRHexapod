# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fold QR-DQN's quantile head into an equivalent plain-Q head, so the OFFICIAL ruler
(eval_protocol.py, byte-unchanged) scores it with the right math.

WHY THIS EXISTS (a silent-failure trap, caught before it produced a number)
--------------------------------------------------------------------------
eval_protocol.py:441-459 auto-detects a distributional head purely from the output width:

    atoms   = out_dim // 64  if out_dim > 64 else 1
    support = linspace(v_min, v_max, atoms)                # fixed [-10, 10]
    q       = (softmax(q.view(-1, 64, atoms), -1) * support).sum(-1)

That is **C51's** reduction: learned probabilities on a fixed support.
QR-DQN is the mirror image: probabilities are fixed at 1/N and the network learns the
**quantile LOCATIONS**, so its greedy value is the plain **mean over quantiles**.
A QR-DQN checkpoint is 64x51 wide, so the stock loader would happily treat it as C51,
softmax the quantile locations, dot them with a support they have nothing to do with,
and return a **wrong argmax with no error message at all** -- a fake QR-DQN score.

FIX -- convert the weights, never the ruler
-------------------------------------------
The greedy action is argmax_a mean_j theta(s)[a, j].  The mean is LINEAR, so it folds
exactly into the final layer:

    mean_j ( W[a,j] . h + b[a,j] )  ==  ( mean_j W[a,j] ) . h  +  mean_j b[a,j]

Averaging the 51 rows of each action's block turns the [3264, H] final layer into a
[64, H] one.  Output width becomes 64 -> the stock loader takes the `atoms == 1` branch
-> plain argmax over Q.  **eval_protocol.py stays byte-for-byte untouched**, so these
numbers stay on exactly the same ruler as every other run today.

Self-check (run automatically, aborts on failure): on random observations the folded
net must reproduce the original net's mean-over-quantiles Q values and its argmax.

usage: python fold_qrdqn_meanQ.py <ckpt_dir> <out_dir>
"""

import os
import sys

import torch

SRC, DST = sys.argv[1], sys.argv[2]
os.makedirs(DST, exist_ok=True)
N_ACT = 64

files = sorted(
    [f for f in os.listdir(SRC) if f.endswith(".pt")],
    key=lambda f: (0, int(f.split("_")[1].split(".")[0])) if f.split("_")[1].split(".")[0].isdigit() else (1, 0),
)
print(f"[fold] {len(files)} checkpoints in {SRC}")

n_ok = 0
for f in files:
    ck = torch.load(os.path.join(SRC, f), map_location="cpu", weights_only=False)
    key = next((k for k in ("q_network", "policy") if isinstance(ck, dict) and k in ck), None)
    if key is None:
        print(f"[fold] SKIP {f}: no q_network/policy key (keys={list(ck)[:6]})")
        continue
    state = dict(ck[key])

    # locate the final Linear ("net.<i>.weight" with the largest i)
    idxs = sorted({int(k.split(".")[1]) for k in state if k.startswith("net.") and k.endswith(".weight")})
    last = idxs[-1]
    wk, bk = f"net.{last}.weight", f"net.{last}.bias"
    W, b = state[wk], state[bk]
    out_dim = W.shape[0]
    if out_dim == N_ACT:
        print(f"[fold] SKIP {f}: already 64-wide (not a quantile head)")
        continue
    assert out_dim % N_ACT == 0, f"{f}: out_dim {out_dim} not a multiple of 64"
    n_q = out_dim // N_ACT

    # view is [action, quantile, ...] because QRNetwork does .view(-1, N_ACTIONS, n_quantiles)
    W_folded = W.view(N_ACT, n_q, -1).mean(dim=1)   # [64, H]
    b_folded = b.view(N_ACT, n_q).mean(dim=1)       # [64]

    # ---- self-check: folded net must match mean-over-quantiles of the original ----
    H = W.shape[1]
    h = torch.randn(4096, H)
    q_orig = (h @ W.T + b).view(-1, N_ACT, n_q).mean(-1)
    q_fold = h @ W_folded.T + b_folded
    max_abs = (q_orig - q_fold).abs().max().item()
    same_argmax = bool((q_orig.argmax(1) == q_fold.argmax(1)).all())
    assert max_abs < 1e-4 and same_argmax, f"{f}: fold check FAILED max|dq|={max_abs} argmax_ok={same_argmax}"

    state[wk], state[bk] = W_folded, b_folded
    out = {key: state}
    if isinstance(ck, dict) and "observation_preprocessor" in ck:
        out["observation_preprocessor"] = ck["observation_preprocessor"]
    torch.save(out, os.path.join(DST, f))
    n_ok += 1
    if n_ok <= 3 or n_ok % 20 == 0:
        print(f"[fold] {f}: {out_dim} -> {N_ACT} (n_quantiles={n_q}) max|dq|={max_abs:.2e} argmax identical")

print(f"[fold] DONE: {n_ok} checkpoints folded into {DST}; every one passed the equivalence check")
