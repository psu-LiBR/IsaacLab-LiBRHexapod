# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Optional per-flip contact-command penalty for the binary-contact env.

Nothing registers :func:`action_switch_count` by default; add it to ``env_cfg.rewards`` only when you want a
penalty on top of ``action_rate_l2``.

For the +-1 leg bits, ``action_rate_l2`` already equals ``4 x`` the number of flipped legs, so the two terms are
the same signal. Tuning the ``action_rate_l2`` weight is the primary way to limit switching; this term only adds a
second, separately scaled knob. With weight ``w`` on this term a flip costs ``w`` per flipped leg, which matches
``action_rate_l2`` weight ``w_ar`` when ``w = 4 * |w_ar| * step_dt``.
"""

import torch


def action_switch_count(env) -> torch.Tensor:
    """Number of legs whose commanded contact bit flipped this step, per second [1/s], shape [num_envs].

    The reward manager multiplies the output by ``weight * step_dt``, so a weight of ``w`` charges ``w`` per flipped
    leg. Steps right after a reset are zeroed because the previous action belongs to the previous episode.
    """
    flips = (env.action_manager.action != env.action_manager.prev_action).sum(-1).float()
    return torch.where(env.episode_length_buf <= 1, 0.0, flips) / env.step_dt
