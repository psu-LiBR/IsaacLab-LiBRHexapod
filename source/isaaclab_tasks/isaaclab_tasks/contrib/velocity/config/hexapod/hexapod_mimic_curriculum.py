# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Curriculum term for blending the imitation phase into the RL phase of the mimic envs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.managers import CurriculumTermCfg, ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class ramp_reward_weight(ManagerTermBase):
    """Linearly ramp a reward weight from its starting value to ``end_weight``.

    Unlike :class:`isaaclab.envs.mdp.modify_reward_weight`, which switches to the final weight in one step, the
    weight is held at its value when the term is built until ``start_step``, then moves linearly to ``end_weight`` by
    ``end_step`` and stays there.
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._term_cfg = env.reward_manager.get_term_cfg(cfg.params["term_name"])
        self._start_weight = float(self._term_cfg.weight)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        term_name: str,
        end_weight: float,
        start_step: int,
        end_step: int,
    ) -> float:
        """Update the weight for the current ``env.common_step_counter`` and return it.

        Args:
            env: The RL environment.
            env_ids: Environments being reset (unused; the weight is global).
            term_name: Name of the reward term.
            end_weight: Weight reached at ``end_step``.
            start_step: Env step at which the ramp starts.
            end_step: Env step at which the ramp ends.
        """
        frac = (env.common_step_counter - start_step) / max(end_step - start_step, 1)
        frac = min(max(frac, 0.0), 1.0)
        weight = self._start_weight + frac * (end_weight - self._start_weight)
        if weight != self._term_cfg.weight:
            self._term_cfg.weight = weight
            env.reward_manager.set_term_cfg(term_name, self._term_cfg)
        return self._term_cfg.weight
