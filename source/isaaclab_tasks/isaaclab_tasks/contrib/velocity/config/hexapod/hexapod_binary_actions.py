# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scripted (non-RL) spine action term for the hexapod binary-gait environment.

The binary-gait task hands the six leg joints to RL as one contact bit per leg
(:class:`isaaclab.envs.mdp.BinaryJointPositionActionCfg` terms) while the two spine
joints follow a fixed open-loop waveform.  :class:`SpineSineAction` is that term: a
zero-width :class:`ActionTerm` (``action_dim == 0``) so the spine consumes no slot in
the RL action vector but still drives its joints every step through the action pipeline
(same PD-target path as every other joint target in this project).

During ``Isaac-Goal-Flat-Hexapod-Binary-v0`` training (and ``-Play-v0``) the term plays
an **analytic traveling body wave**: ``FrontLink_Joint`` a pure sine and
``BackLink_Joint`` the same sine shifted +90 deg (a cosine), with a shared magnitude
(``A_SPINE = deg2rad(70) * 12/16 = deg2rad(52.5) = 0.9162978573`` rad, the exact
open-loop gait-generator value) carried with the HexapI global spine-joint-sign flip
(``sin_coef`` / ``cos_coef`` hold ``-A_SPINE``) and offset -- a quarter-cycle wave that
travels down the body.  It is not fitted to any CSV.
``eval_protocol.py``'s reference-tripod baseline instead swaps in an analytic *anti-phase*
body wave regenerated from the MATLAB gait generator -- ``BackLink_Joint = -FrontLink_Joint``,
``FrontLink_Joint(t) = -A_SPINE*sin(2*pi*t - pi/4)`` -- via :meth:`set_waveform`, not a CSV fit.

The waveform is a truncated Fourier series with period :attr:`SpineSineActionCfg.period`
(one gait cycle)::

    q(t) = offset + sum_k [ sin_coef[k] * sin((k + 1) * w * t) + cos_coef[k] * cos((k + 1) * w * t) ]

with ``w = 2 * pi / period``, ``k`` the 0-based harmonic index (``k = 0`` is the
fundamental), and ``t`` the elapsed episode time (``env.episode_length_buf *
env.step_dt``), so the wave restarts from phase 0 at every episode reset.  The canonical
coefficient values for both waves live in :mod:`.hexapod_binary_env_cfg`
("CANONICAL SPINE-WAVE DEFINITION" block).
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

        if self.cfg.period <= 0.0:
            raise ValueError(f"SpineSineAction: period must be positive, got {self.cfg.period}")

        # per-joint Fourier coefficients, resolved by joint name so the articulation DOF
        # order never matters
        self._sin_coef, self._cos_coef, self._offset, self._harmonics = self._resolve_waveform(
            self.cfg.sin_coef, self.cfg.cos_coef, self.cfg.offset
        )

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
        w = 2.0 * math.pi / self.cfg.period
        # [N, 1, H] angle per harmonic
        angle = w * t.view(-1, 1, 1) * self._harmonics.view(1, 1, -1)
        # [N, J] = offset + sum_H (sin_coef * sin(angle) + cos_coef * cos(angle))
        self._processed_actions = self._offset.view(1, -1) + (
            self._sin_coef.view(1, self._num_joints, -1) * torch.sin(angle)
            + self._cos_coef.view(1, self._num_joints, -1) * torch.cos(angle)
        ).sum(dim=-1)

    def apply_actions(self):
        self._asset.set_joint_position_target_index(target=self._processed_actions, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        # phase is derived from episode_length_buf, which the env resets itself
        pass

    def _resolve_waveform(
        self,
        sin_coef: dict[str, list[float]],
        cos_coef: dict[str, list[float]],
        offset: dict[str, float],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Resolve per-joint Fourier coefficient tables into device tensors.

        Coefficient lists are zero-padded to a common harmonic count; ``sin_coef`` and
        ``cos_coef`` must resolve to the same shape. Joints are ordered as
        :attr:`_joint_names` so the articulation DOF order is irrelevant.

        Args:
            sin_coef: Per-joint-name sine coefficient lists ``[a1, a2, ...]`` [rad].
            cos_coef: Per-joint-name cosine coefficient lists ``[b1, b2, ...]`` [rad].
            offset: Per-joint-name constant term ``c0`` [rad].

        Returns:
            Tuple ``(sin_coef, cos_coef, offset, harmonics)`` of tensors shaped ``[J, H]``,
            ``[J, H]``, ``[J]`` and ``[H]`` respectively, all on :attr:`device`.
        """

        def _coef_table(table: dict[str, list[float]], what: str) -> torch.Tensor:
            missing = set(self._joint_names) - set(table.keys())
            if missing:
                raise ValueError(f"SpineSineAction: missing {what} for joints {sorted(missing)}")
            rows = [list(table[n]) for n in self._joint_names]
            width = max(len(r) for r in rows)
            padded = [r + [0.0] * (width - len(r)) for r in rows]
            return torch.tensor(padded, device=self.device, dtype=torch.float32)  # [J, H]

        def _offset_table(table: dict[str, float]) -> torch.Tensor:
            missing = set(self._joint_names) - set(table.keys())
            if missing:
                raise ValueError(f"SpineSineAction: missing offset for joints {sorted(missing)}")
            return torch.tensor([table[n] for n in self._joint_names], device=self.device, dtype=torch.float32)

        sin_tensor = _coef_table(sin_coef, "sin_coef")  # [J, H]
        cos_tensor = _coef_table(cos_coef, "cos_coef")  # [J, H]
        if sin_tensor.shape != cos_tensor.shape:
            raise ValueError(
                "SpineSineAction: sin_coef and cos_coef must have matching per-joint lengths, got "
                f"{tuple(sin_tensor.shape)} vs {tuple(cos_tensor.shape)}"
            )
        offset_tensor = _offset_table(offset)  # [J]
        # harmonic multipliers 1..H (k = 0 is the fundamental)
        harmonics = torch.arange(1, sin_tensor.shape[1] + 1, device=self.device, dtype=torch.float32)  # [H]
        return sin_tensor, cos_tensor, offset_tensor, harmonics

    def set_waveform(
        self,
        sin_coef: dict[str, list[float]],
        cos_coef: dict[str, list[float]],
        offset: dict[str, float],
    ) -> None:
        """Replace the per-joint Fourier coefficients at runtime.

        Used by ``eval_protocol.py`` to swap the analytic RL-env traveling wave for the
        anti-phase ``tripod_extendedquad`` baseline wave without rebuilding the env. The new
        tables are resolved exactly as in :meth:`__init__` (same joint-name ordering,
        device, zero-pad-to-equal-length and shape check).

        Args:
            sin_coef: Per-joint-name sine coefficient lists ``[a1, a2, ...]`` [rad].
            cos_coef: Per-joint-name cosine coefficient lists ``[b1, b2, ...]`` [rad].
            offset: Per-joint-name constant term ``c0`` [rad].
        """
        self._sin_coef, self._cos_coef, self._offset, self._harmonics = self._resolve_waveform(
            sin_coef, cos_coef, offset
        )


@configclass
class SpineSineActionCfg(ActionTermCfg):
    """Configuration for :class:`SpineSineAction`."""

    class_type: type[ActionTerm] = SpineSineAction

    joint_names: list[str] = None  # type: ignore[assignment]
    """Joint names (or regex) driven by the waveform."""

    sin_coef: dict[str, list[float]] = None  # type: ignore[assignment]
    """Per-joint-name sine Fourier coefficients ``[a1, a2, ...]`` in rad; entry ``k`` (0-based)
    multiplies ``sin((k + 1) * 2 * pi * t / period)``."""

    cos_coef: dict[str, list[float]] = None  # type: ignore[assignment]
    """Per-joint-name cosine Fourier coefficients ``[b1, b2, ...]`` in rad; entry ``k`` (0-based)
    multiplies ``cos((k + 1) * 2 * pi * t / period)``. Must match :attr:`sin_coef` in length."""

    offset: dict[str, float] = None  # type: ignore[assignment]
    """Per-joint-name constant term ``c0`` in rad."""

    period: float = 1.0
    """Gait period in seconds (time for one full fundamental cycle)."""
