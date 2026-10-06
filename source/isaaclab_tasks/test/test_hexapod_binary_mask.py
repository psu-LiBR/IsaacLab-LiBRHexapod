# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the binary-contact action mask / bit helpers.

Loads ``scripts/reinforcement_learning/binary_rl/binary_action_mask.py`` directly by
path (it is a script-tree module, not part of the ``isaaclab_tasks`` package) and only
needs ``torch``, so this runs under plain ``python -m pytest`` without Isaac Sim.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

BINARY_RL_DIR = Path(__file__).parents[3] / "scripts" / "reinforcement_learning" / "binary_rl"
# so binary_common's `from binary_action_mask import ...` resolves
sys.path.insert(0, str(BINARY_RL_DIR))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, BINARY_RL_DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bam = _load("binary_action_mask")
bam_common = _load("binary_common")


# --- expected legal set: popcount >= 4, plus the two tripod patterns -----------------
_POPCOUNT_GE_4 = sorted(a for a in range(64) if bin(a).count("1") >= 4)
_EXPECTED_LEGAL = sorted(set(_POPCOUNT_GE_4) | {25, 38})


def test_popcount_ge_4_has_22_actions():
    assert len(_POPCOUNT_GE_4) == 22  # C(6,4) + C(6,5) + C(6,6) = 15 + 6 + 1


def test_tripod_actions_are_complementary_triples():
    assert bam.TRIPOD_ACTIONS == (25, 38)
    assert 25 + 38 == 63
    assert bin(25).count("1") == 3 and bin(38).count("1") == 3


def test_legal_mask_has_exactly_24_actions():
    mask = bam.legal_action_mask()
    assert mask.dtype == torch.bool
    assert mask.shape == (64,)
    assert int(mask.sum()) == 24


def test_legal_indices_match_expected_set():
    idx = bam.legal_action_indices().tolist()
    assert idx == _EXPECTED_LEGAL
    # the two tripods are in, but only because they are whitelisted
    assert 25 in idx and 38 in idx
    assert bam.popcount(torch.tensor([25, 38])).tolist() == [3, 3]


def test_whitelist_none_drops_the_tripods():
    idx = bam.legal_action_indices(whitelist=None).tolist()
    assert idx == _POPCOUNT_GE_4
    assert 25 not in idx and 38 not in idx


def test_min_stance_legs_is_configurable():
    assert int(bam.legal_action_mask(min_stance_legs=6, whitelist=None).sum()) == 1  # only action 63
    assert int(bam.legal_action_mask(min_stance_legs=0, whitelist=None).sum()) == 64


def test_bit_action_roundtrip_for_all_64():
    actions = torch.arange(64)
    bits = bam.action_to_bits(actions)
    assert bits.shape == (64, 6)
    assert torch.equal(bam.bits_to_action(bits), actions)
    # LSB first
    assert bits[1].tolist() == [1, 0, 0, 0, 0, 0]
    assert bits[63].tolist() == [1, 1, 1, 1, 1, 1]


def test_action_to_pm1_matches_wrapper_decode():
    """``action_to_pm1`` must be bit-identical to ``DiscreteBitsActionWrapper.decode``."""
    actions = torch.arange(64)
    pm1 = bam.action_to_pm1(actions)
    assert pm1.shape == (64, 6)
    assert set(pm1.unique().tolist()) == {-1.0, 1.0}
    # reproduce the wrapper's arithmetic: (a >> arange(6)) & 1, then * 2 - 1
    shifts = torch.arange(6)
    ref = ((actions.unsqueeze(1) >> shifts) & 1).float() * 2.0 - 1.0
    assert torch.equal(pm1, ref)


def test_apply_logit_mask_zeros_illegal_actions_after_softmax():
    mask = bam.legal_action_mask()
    logits = torch.randn(8, 64)
    masked = bam.apply_logit_mask(logits, mask)
    probs = torch.softmax(masked, dim=-1)
    assert torch.allclose(probs[:, ~mask], torch.zeros_like(probs[:, ~mask]))
    assert torch.allclose(probs.sum(dim=-1), torch.ones(8))
    # legal-action logits are untouched
    assert torch.equal(masked[:, mask], logits[:, mask])


def test_nstep_return_full_window():
    """No episode boundary: full geometric sum, K = n, bootstrap = 1."""
    n, num, gamma = 4, 3, 0.9
    rew = torch.ones(n, num)
    ret, steps, boot = bam_common.nstep_return(rew, torch.zeros(n, num), torch.zeros(n, num), gamma)
    expected = sum(gamma**k for k in range(n))
    assert torch.allclose(ret, torch.full((num,), expected))
    assert torch.equal(steps, torch.full((num,), n))
    assert torch.allclose(boot, torch.ones(num))


def test_nstep_return_termination_cuts_window_and_kills_bootstrap():
    n, gamma = 5, 0.99
    rew = torch.ones(n, 1)
    term = torch.zeros(n, 1)
    term[2, 0] = 1.0  # episode ends at window step 2 (0-indexed) -> 3 rewards summed
    ret, steps, boot = bam_common.nstep_return(rew, term, torch.zeros(n, 1), gamma)
    assert int(steps[0]) == 3
    assert torch.allclose(ret[0], torch.tensor(1.0 + gamma + gamma**2))
    assert float(boot[0]) == 0.0


def test_nstep_return_truncation_cuts_window_but_keeps_bootstrap():
    n, gamma = 5, 0.99
    rew = torch.ones(n, 1)
    trunc = torch.zeros(n, 1)
    trunc[1, 0] = 1.0  # timeout at window step 1 -> 2 rewards, still bootstrap
    ret, steps, boot = bam_common.nstep_return(rew, torch.zeros(n, 1), trunc, gamma)
    assert int(steps[0]) == 2
    assert torch.allclose(ret[0], torch.tensor(1.0 + gamma))
    assert float(boot[0]) == 1.0


def test_apply_logit_mask_leaves_all_illegal_row_finite():
    mask = torch.zeros(64, dtype=torch.bool)
    logits = torch.randn(3, 64)
    masked = bam.apply_logit_mask(logits, mask)
    assert torch.isfinite(masked).all()
    assert torch.equal(masked, logits)


def test_apply_logit_mask_uses_finite_fill_and_has_no_nan_entropy_gradient():
    """The masked fill must be finite so the categorical entropy backward stays NaN-free."""
    mask = bam.legal_action_mask()
    logits = torch.randn(16, 64, requires_grad=True)
    masked = bam.apply_logit_mask(logits, mask)
    assert torch.isfinite(masked).all()
    dist = torch.distributions.Categorical(logits=masked)
    dist.entropy().sum().backward()
    assert torch.isfinite(logits.grad).all()
