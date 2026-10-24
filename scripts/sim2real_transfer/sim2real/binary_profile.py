# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Binary contact-bit profile: policy-action decode + host-side spine wave.

The binary env (``Isaac-Goal-Flat-Hexapod-Binary-v0``) gives the policy six leg contact bits. Each bit snaps its
leg joint to one of two fixed angles (``STANCE_POS`` / ``LIFT_POS`` in ``hexapod_binary_env_cfg.py``), and the two
spine joints follow a fixed analytic wave the policy never sees (``SpineSineAction``, Wave 1 in the "CANONICAL
SPINE-WAVE DEFINITION" block there): a single harmonic of period 1.0 s and amplitude ``-A_SPINE``
(``A_SPINE`` = ``deg2rad(70) * 12/16`` = 0.9162978573 rad) on both joints, FrontLink = ``-A_SPINE*sin(w*t)`` and
BackLink the same shifted by +pi/2 (``-A_SPINE*cos(w*t)``).

``binary.spine.amplitude`` in the deployment YAML carries the sim's wave verbatim. The real spine motors are
mounted opposite to the sim convention, so both spine joints use correction group ``negate`` (real = -sim) and the
real encoders follow FrontLink = +A_SPINE*sin, BackLink = +A_SPINE*cos (hardware-confirmed 2026-10-20). Negating
through the correction group, rather than flipping the amplitudes, also keeps the encoder read-back converting to
the sim joint angles the policy was trained on.

The wave is single-harmonic, so a per-joint ``q(t) = offset + amplitude * sin(2*pi*t / period + phase)`` (FrontLink
phase 0, BackLink phase pi/2) represents it directly.

On hardware the exported ONNX graph emits the six-dim ``+-1`` bit vector (see
``scripts/reinforcement_learning/binary_rl/export_binary_onnx.py``). This module turns that vector plus the
elapsed time into the full eight-joint, sim-DOF-order position target, recreating ``SpineSineAction`` host-side.
The target is in the sim joint convention and flows through the usual ``JointMapping.clip_to_soft_limits`` ->
``sim_target_to_ticks`` path; nothing here touches encoder ticks or the sim<->real DOF permutation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BinarySpineCfg:
    """Per-joint parameters of Wave 1, ``q(t) = offset + amplitude * sin(2*pi*t / period + phase)``.

    ``amplitude`` / ``phase`` / ``offset`` are ``{joint_name: value}`` maps [rad] keyed by the sim-DOF names
    (``BackLink`` / ``FrontLink``, no ``_Joint`` suffix); ``period`` is one gait cycle [s].
    """

    amplitude: dict[str, float]
    phase: dict[str, float]
    offset: dict[str, float]
    period: float


@dataclass(frozen=True)
class BinaryActionCfg:
    """Everything needed to turn a six-bit policy action into eight joint targets."""

    stance_pos: float
    """Leg joint angle for a stance (foot-down) bit, rad. Matches ``STANCE_POS``."""

    lift_pos: float
    """Leg joint angle for a lift (foot-up) bit, rad. Matches ``LIFT_POS``."""

    leg_joint_order: list[str]
    """The six leg joint names in *policy action index* order. Must match the term
    declaration order of ``HexapodBinaryActionsCfg`` (FrontRight, FrontLeft,
    MiddleRight, MiddleLeft, BackRight, BackLeft)."""

    spine: BinarySpineCfg


class BinaryActionAdapter:
    """Decode a six-bit ``+-1`` policy action into an eight-joint sim-order target."""

    def __init__(self, cfg: BinaryActionCfg, sim_order: list[str]):
        self.sim_order = list(sim_order)
        n = len(self.sim_order)

        if len(cfg.leg_joint_order) != 6:
            raise ValueError(f"leg_joint_order must list 6 joints, got {len(cfg.leg_joint_order)}")
        self.stance_pos = float(cfg.stance_pos)
        self.lift_pos = float(cfg.lift_pos)

        missing = [j for j in cfg.leg_joint_order if j not in self.sim_order]
        if missing:
            raise ValueError(f"leg_joint_order joints not in sim_order: {missing}")
        self._leg_sim_idx = np.array([self.sim_order.index(j) for j in cfg.leg_joint_order], dtype=np.int64)

        spine_names = list(cfg.spine.amplitude.keys())
        if len(spine_names) != 2:
            raise ValueError(f"expected 2 spine joints, got {spine_names}")
        for what, table in (("phase", cfg.spine.phase), ("offset", cfg.spine.offset)):
            miss = set(spine_names) - set(table)
            if miss:
                raise ValueError(f"spine {what} missing joints {sorted(miss)}")
        miss_sim = [j for j in spine_names if j not in self.sim_order]
        if miss_sim:
            raise ValueError(f"spine joints not in sim_order: {miss_sim}")
        self._spine_names = spine_names
        self._spine_sim_idx = np.array([self.sim_order.index(j) for j in spine_names], dtype=np.int64)
        self._spine_amp = np.array([cfg.spine.amplitude[j] for j in spine_names], dtype=np.float64)
        self._spine_phase = np.array([cfg.spine.phase[j] for j in spine_names], dtype=np.float64)
        self._spine_offset = np.array([cfg.spine.offset[j] for j in spine_names], dtype=np.float64)
        if cfg.spine.period <= 0.0:
            raise ValueError(f"spine period must be positive, got {cfg.spine.period}")
        self._spine_period = float(cfg.spine.period)

        covered = set(self._leg_sim_idx.tolist()) | set(self._spine_sim_idx.tolist())
        if covered != set(range(n)):
            raise ValueError(
                f"leg + spine joints do not cover all {n} sim DOFs (covered {sorted(covered)}, sim_order has {n})"
            )

    def leg_targets(self, bits_pm1: np.ndarray) -> np.ndarray:
        """Six ``+-1`` bits -> six leg joint angles (``+1`` -> stance, ``<=0`` -> lift)."""
        bits_pm1 = np.asarray(bits_pm1, dtype=np.float64)
        if bits_pm1.shape != (6,):
            raise ValueError(f"bits_pm1 must be shape (6,), got {bits_pm1.shape}")
        return np.where(bits_pm1 > 0.0, self.stance_pos, self.lift_pos)

    def spine_targets(self, t_seconds: float) -> np.ndarray:
        """The two spine joint angles at elapsed time ``t_seconds`` (radians)."""
        angle = (2.0 * math.pi / self._spine_period) * float(t_seconds)
        return self._spine_offset + self._spine_amp * np.sin(angle + self._spine_phase)

    def targets_sim(self, bits_pm1: np.ndarray, t_seconds: float) -> np.ndarray:
        """Full eight-joint sim-DOF-order position target, radians."""
        out = np.zeros(len(self.sim_order), dtype=np.float64)
        out[self._leg_sim_idx] = self.leg_targets(bits_pm1)
        out[self._spine_sim_idx] = self.spine_targets(t_seconds)
        return out
