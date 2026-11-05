# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward terms specific to the binary contact-bit hexapod env."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def min_stance_legs_penalty(
    env: ManagerBasedRLEnv,
    min_legs: int = 3,
    num_leg_bits: int = 6,
    power: float = 1.0,
) -> torch.Tensor:
    """Penalty magnitude for having fewer than ``min_legs`` feet down (stance).

    Sign convention follows the binary contact action space: a leg bit ``> 0`` (``+1`` from the
    discrete wrapper) is stance (foot down), ``< 0`` is lift (foot up); exactly ``0`` is stance,
    identical to ``BinaryJointAction.process_actions`` (lift only when the action is negative). The deficit is
    ``max(min_legs - n_stance, 0)``, so a tripod (exactly 3 stance legs) or any gait with more
    legs down incurs zero. The function returns a **non-negative** magnitude ``deficit ** power``;
    pair it with a **negative** weight.

    It assumes the six leg bits are the first ``num_leg_bits`` columns of the action vector. This
    holds for the binary env because the scripted spine term (``SpineSineAction``) is zero-width
    (``action_dim == 0``), so it adds no action columns; any extra trailing columns are ignored.

    Args:
        env: The RL environment instance.
        min_legs: Minimum number of stance legs that incurs no penalty.
        num_leg_bits: Number of leading action columns that are leg contact bits.
        power: Exponent applied to the deficit (1.0 linear, 2.0 quadratic).

    Returns:
        Non-negative penalty magnitude, shape ``[num_envs]``, float.
    """
    leg_bits = env.action_manager.action[:, :num_leg_bits]
    # stance is "not lift", matching BinaryJointAction.process_actions (lift only when a < 0, so 0 and NaN are stance)
    n_stance = (~(leg_bits < 0)).sum(dim=1)
    deficit = (min_legs - n_stance).clamp(min=0).float()
    return deficit**power
