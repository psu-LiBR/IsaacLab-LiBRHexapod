# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bit/action helpers and the legal-action mask for the hexapod binary-contact env.

The binary-contact env exposes a ``Discrete(64)`` action: one integer in ``[0, 63]``
whose six low bits are the per-leg stance/swing pattern (bit ``i`` -> action-vector
index ``i``; bit ``1`` = stance / foot down, bit ``0`` = swing / foot up). Bit ``i``
LSB-first maps to a leg in the term-declaration order of
:class:`HexapodBinaryActionsCfg` (``idx 0`` = ``FrontRight`` ... ``idx 5`` = ``BackLeft``).

This module is deliberately dependency-free (``torch`` only, no Isaac Sim / Isaac Lab
import) so it can be unit-tested under plain ``python -m pytest`` and imported by both
the training scripts and the sim-to-real export helper.

**Masked-categorical policy.** :func:`legal_action_mask` builds the boolean mask of
actions a masked-PPO policy is allowed to sample. The default rule keeps any pattern
with at least :data:`DEFAULT_MIN_STANCE_LEGS` legs in stance, plus an explicit
whitelist for the two reference tripod patterns (:data:`TRIPOD_ACTIONS`), which each
have only three legs down and would otherwise be masked out. Apply
:func:`apply_logit_mask` to the raw logits *before* sampling and *before* computing the
log-probability used in the PPO surrogate loss, so the disallowed actions carry zero
probability in both the rollout and the update.
"""

from __future__ import annotations

import torch

N_BITS = 6
"""Number of leg contact bits (one per leg)."""

N_ACTIONS = 2**N_BITS
"""Size of the discrete action space (``64``)."""

TRIPOD_ACTIONS: tuple[int, ...] = (25, 38)
"""The two complementary reference tripod contact patterns, ``25 + 38 == 63``.

``25`` = ``0b011001`` (legs at idx 0, 3, 4 in stance) and ``38`` = ``0b100110`` (legs
at idx 1, 2, 5 in stance). Each has only three legs down, so both are whitelisted
past the :data:`DEFAULT_MIN_STANCE_LEGS` rule.
"""

DEFAULT_MIN_STANCE_LEGS = 4
"""Minimum stance-leg count allowed by :func:`legal_action_mask` unless whitelisted."""


def action_to_bits(actions: torch.Tensor, n_bits: int = N_BITS) -> torch.Tensor:
    """Unpack integer actions into their bit vectors, LSB first.

    Args:
        actions: Integer action indices, any shape.
        n_bits: Number of bits to unpack.

    Returns:
        Long tensor of shape ``(*actions.shape, n_bits)`` with values in ``{0, 1}``;
        the last axis is ordered from bit ``0`` (LSB) to bit ``n_bits - 1``.
    """
    shifts = torch.arange(n_bits, dtype=torch.long, device=actions.device)
    return (actions.long().unsqueeze(-1) >> shifts) & 1


def bits_to_action(bits: torch.Tensor) -> torch.Tensor:
    """Pack a bit vector (LSB first, last axis) back into an integer action.

    Args:
        bits: Tensor whose last axis holds ``{0, 1}`` bits, LSB first.

    Returns:
        Long tensor of shape ``bits.shape[:-1]``.
    """
    n_bits = bits.shape[-1]
    weights = 1 << torch.arange(n_bits, dtype=torch.long, device=bits.device)
    return (bits.long() * weights).sum(dim=-1)


def action_to_pm1(actions: torch.Tensor, n_bits: int = N_BITS) -> torch.Tensor:
    """Decode integer actions into the ``+/-1`` float vector the env consumes.

    Bit value ``1`` -> ``+1.0`` (stance / foot down), bit value ``0`` -> ``-1.0``
    (swing / foot up). This is bit-for-bit identical to
    ``DiscreteBitsActionWrapper.decode``.

    Args:
        actions: Integer action indices, any shape.
        n_bits: Number of leg bits.

    Returns:
        Float tensor of shape ``(*actions.shape, n_bits)`` with values in
        ``{-1.0, +1.0}``.
    """
    return action_to_bits(actions, n_bits).to(torch.float32) * 2.0 - 1.0


def popcount(actions: torch.Tensor, n_bits: int = N_BITS) -> torch.Tensor:
    """Number of set bits per action = number of legs commanded into stance.

    Args:
        actions: Integer action indices, any shape.
        n_bits: Number of bits to count over.

    Returns:
        Long tensor of shape ``actions.shape``.
    """
    return action_to_bits(actions, n_bits).sum(dim=-1)


def legal_action_mask(
    min_stance_legs: int = DEFAULT_MIN_STANCE_LEGS,
    whitelist: tuple[int, ...] | None = TRIPOD_ACTIONS,
    n_bits: int = N_BITS,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Boolean mask over the ``2**n_bits`` actions a masked policy may sample.

    An action is legal if it has at least ``min_stance_legs`` legs in stance
    (``popcount >= min_stance_legs``) or it appears in ``whitelist``.

    Args:
        min_stance_legs: Inclusive lower bound on the stance-leg count.
        whitelist: Actions always kept legal regardless of their stance-leg count;
            pass ``None`` or an empty tuple to disable.
        n_bits: Number of leg bits.
        device: Device for the returned tensor.

    Returns:
        Boolean tensor of shape ``(2**n_bits,)``; ``True`` marks a legal action.
    """
    actions = torch.arange(2**n_bits, dtype=torch.long, device=device)
    mask = popcount(actions, n_bits) >= min_stance_legs
    for a in whitelist or ():
        mask[a] = True
    return mask


def legal_action_indices(
    min_stance_legs: int = DEFAULT_MIN_STANCE_LEGS,
    whitelist: tuple[int, ...] | None = TRIPOD_ACTIONS,
    n_bits: int = N_BITS,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Sorted indices of the legal actions; see :func:`legal_action_mask`.

    Returns:
        Long tensor of the legal action indices, ascending.
    """
    return torch.nonzero(legal_action_mask(min_stance_legs, whitelist, n_bits, device), as_tuple=False).flatten()


DEFAULT_MASK_FILL = -1.0e9
"""Fill value for masked logits.

A large finite negative number, **not** ``-inf``: ``softmax`` still drives the masked
probabilities to exactly ``0``, but the logits stay finite so the entropy's backward
pass does not hit ``0 * -inf -> NaN`` (the standard masked-categorical gradient trap;
see CleanRL's ``CategoricalMasked``).
"""


def apply_logit_mask(logits: torch.Tensor, mask: torch.Tensor, fill_value: float = DEFAULT_MASK_FILL) -> torch.Tensor:
    """Set the logits of illegal actions to ``fill_value`` (a large finite negative).

    Broadcasts a ``(n_actions,)`` mask over any leading batch dimensions of
    ``logits``. Rows that would become entirely masked are left untouched so a
    downstream ``softmax`` cannot produce ``NaN``; with the default 24-action mask
    this never triggers, but callers that pass a custom mask should guard against an
    all-illegal row themselves.

    Args:
        logits: Unnormalized log-probabilities, shape ``(*batch, n_actions)``.
        mask: Boolean tensor broadcastable to ``logits``; ``True`` = keep.
        fill_value: Value written into masked positions.

    Returns:
        Tensor the same shape as ``logits``.
    """
    mask = mask.to(dtype=torch.bool, device=logits.device)
    broadcast_mask = mask.expand_as(logits)
    row_has_legal = broadcast_mask.any(dim=-1, keepdim=True)
    effective_mask = broadcast_mask | ~row_has_legal
    return logits.masked_fill(~effective_mask, fill_value)
