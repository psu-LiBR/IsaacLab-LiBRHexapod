# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scripted (non-RL) spine action term for the hexapod binary-gait environment.

The binary-gait task hands the six leg joints to RL as one contact bit per leg
(:class:`isaaclab.envs.mdp.BinaryJointPositionActionCfg` terms) while the two spine
joints follow a fixed open-loop sinusoid extracted from the reference tripod gait CSV
(``hexapod-assets/Sim Gaits/tripod_B11BL0_sim.csv``).  This module provides that spine
term: a zero-width :class:`ActionTerm` (``action_dim == 0``) so the spine consumes no
slot in the RL action vector but still drives its joints every step through the action
pipeline (same PD-target path as every other joint target in this project).

Per-joint targets follow ``q(t) = offset + amplitude * sin(2*pi * t / period + phase)``
with ``t`` the elapsed episode time (``env.episode_length_buf * env.step_dt``), so the
wave restarts from phase 0 at every episode reset, matching how the reference CSV is
replayed from row 0.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.managers.manager_term_cfg import ActionTermCfg
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class SpineSineAction(ActionTerm):
    """Open-loop sinusoidal joint-position targets for the spine joints.

    Zero-width action term: contributes nothing to the policy action vector
    (``action_dim == 0``) and ignores the (empty) action slice it receives.
    """

    cfg: SpineSineActionCfg
    _asset: Articulation

    def __init__(self, cfg: SpineSineActionCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)

        # resolve the joints over which the term is applied
        self._joint_ids, self._joint_names = self._asset.find_joints(self.cfg.joint_names)
        self._num_joints = len(self._joint_ids)

        # per-joint sine parameters, resolved by joint name so the articulation DOF
        # order never matters
        def _param(table: dict[str, float], what: str) -> torch.Tensor:
            missing = set(self._joint_names) - set(table.keys())
            if missing:
                raise ValueError(f"SpineSineAction: missing {what} for joints {sorted(missing)}")
            return torch.tensor([table[n] for n in self._joint_names], device=self.device)

        self._amplitude = _param(self.cfg.amplitude, "amplitude")  # [J]
        self._phase = _param(self.cfg.phase, "phase")  # [J]
        self._offset = _param(self.cfg.offset, "offset")  # [J]
        if self.cfg.period <= 0.0:
            raise ValueError(f"SpineSineAction: period must be positive, got {self.cfg.period}")

        # empty raw-action buffer (this term owns no slice of the action vector)
        self._raw_actions = torch.zeros(self.num_envs, 0, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, self._num_joints, device=self.device)

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return 0

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        # actions is a zero-width slice [N, 0]; the target depends only on episode time.
        t = self._env.episode_length_buf.to(dtype=torch.float32) * self._env.step_dt  # [N]
        angle = (2.0 * math.pi / self.cfg.period) * t  # [N]
        # [N, J] = offset + A * sin(angle + phase)
        self._processed_actions = self._offset.unsqueeze(0) + self._amplitude.unsqueeze(0) * torch.sin(
            angle.unsqueeze(1) + self._phase.unsqueeze(0)
        )

    def apply_actions(self):
        self._asset.set_joint_position_target_index(target=self._processed_actions, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        # phase is derived from episode_length_buf, which the env resets itself
        pass


@configclass
class SpineSineActionCfg(ActionTermCfg):
    """Configuration for :class:`SpineSineAction`."""

    class_type: type[ActionTerm] = SpineSineAction

    joint_names: list[str] = None  # type: ignore[assignment]
    """Joint names (or regex) driven by the sinusoid."""

    amplitude: dict[str, float] = None  # type: ignore[assignment]
    """Per-joint-name amplitude A in rad."""

    phase: dict[str, float] = None  # type: ignore[assignment]
    """Per-joint-name phase offset phi in rad (applied as sin(2*pi*t/period + phi))."""

    offset: dict[str, float] = None  # type: ignore[assignment]
    """Per-joint-name constant offset C in rad."""

    period: float = 1.0
    """Gait period in seconds (time for one full sine cycle)."""
