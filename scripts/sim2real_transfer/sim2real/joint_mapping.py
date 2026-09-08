# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sim<->Real DOF permutation, sign/offset corrections, and tick<->radian conversion.

This is the single most safety-critical file in the deployment package: a sign or
reorder mistake here sends the wrong joint the wrong direction. Every step is kept
separate and testable rather than folded into one formula, and the sim<->real
permutation is derived from named joint lists (matching the convention already used
by scripts/sim2sim_transfer/config/*.yaml) rather than hardcoded index arrays.
"""

from __future__ import annotations

import math

import numpy as np

from .deployment_config import JointsCfg

# real = a * sim + b (elementwise, in radians), applied after reordering sim->real.
# Inverse: sim = (real - b) / a.
_CORRECTIONS: dict[str, tuple[float, float]] = {
    "unchanged": (1.0, 0.0),
    "negate": (-1.0, 0.0),
    "leg_negate_plus_pi": (-1.0, math.pi),
    # Same sign flip as leg_negate_plus_pi (a=-1), but the other 2*pi-equivalent
    # phase branch (b=-pi instead of +pi) -- same physical joint angle, but lands
    # zero_tick on the other side of the ticks = zero_tick + real_rad*ticks_per_rad
    # formula's [0, ticks_per_rev) window. Calibrated 2026-07-17: this robot's leg
    # servos measured a physical rest-pose tick (~1720-1864) that only produces a
    # valid zero_tick in [0, 4095] under this -pi branch, not +pi -- see
    # tools/calibrate_encoders.py zero-tick output and CLAUDE.md.
    "leg_negate_minus_pi": (-1.0, -math.pi),
    # HexapI USD leg convention (used by the `binary` profile -- hexapod_binary_env_cfg.py
    # gives the legs positive angles). The HexapI USD flips the leg-joint sign vs the old
    # USD, so sim_hexapi = -sim_old. Substituting into this robot's *calibrated* old map
    # real = -sim_old - pi (leg_negate_minus_pi) gives real = sim_hexapi - pi. Same -pi
    # phase branch, so the identical real encoder angle and the identical calibrated leg
    # zero_tick carry over unchanged from the velocity/goal deployment.yaml
    # (HexapI stance +0.46 -> 0.46 - pi == old stance -0.46 -> -(-0.46) - pi).
    "leg_minus_pi": (1.0, -math.pi),
}


def ordered_array(values: dict[str, float], order: list[str]) -> np.ndarray:
    """Convert a {joint_name: value} dict into an array following `order`."""
    return np.array([values[name] for name in order], dtype=np.float64)


class JointMapping:
    """Sim DOF order <-> Real DOF order conversion, built from named joint lists."""

    def __init__(self, cfg: JointsCfg):
        if set(cfg.sim_order) != set(cfg.real_order):
            raise ValueError("sim_order and real_order must contain the same joint names")

        self.sim_order = list(cfg.sim_order)
        self.real_order = list(cfg.real_order)
        n = len(self.sim_order)

        sim_index_of = {name: i for i, name in enumerate(self.sim_order)}
        real_index_of = {name: i for i, name in enumerate(self.real_order)}

        # real_arr = sim_arr[sim_to_real_idx]
        self.sim_to_real_idx = np.array([sim_index_of[name] for name in self.real_order])
        # sim_arr = real_arr[real_to_sim_idx]
        self.real_to_sim_idx = np.array([real_index_of[name] for name in self.sim_order])

        identity = np.arange(n)
        if not np.array_equal(self.sim_to_real_idx[self.real_to_sim_idx], identity):
            raise ValueError("sim_to_real_idx / real_to_sim_idx are not inverse permutations")

        corr_a = np.empty(n, dtype=np.float64)
        corr_b = np.empty(n, dtype=np.float64)
        for i, name in enumerate(self.real_order):
            group = cfg.correction_group[name]
            if group not in _CORRECTIONS:
                raise ValueError(f"unknown correction_group '{group}' for joint '{name}'")
            corr_a[i], corr_b[i] = _CORRECTIONS[group]
        self._corr_a = corr_a
        self._corr_b = corr_b

        self.motor_ids = [cfg.motor_ids[name] for name in self.real_order]

        self.ticks_per_rev = cfg.ticks_per_rev
        ticks_per_rad = cfg.ticks_per_rev / (2.0 * math.pi)
        self._ticks_per_rad = ticks_per_rad
        self._zero_tick = np.array([cfg.zero_tick[name] for name in self.real_order], dtype=np.float64)

        self.soft_limits_sim = np.array([cfg.soft_limits_rad[name] for name in self.sim_order], dtype=np.float64)

    # ------------------------------------------------------------------
    # Sim <-> Real, in radians
    # ------------------------------------------------------------------

    def sim_rad_to_real_rad(self, sim_rad: np.ndarray) -> np.ndarray:
        """Reorder a sim-DOF-order radian array into real-DOF-order, applying corrections."""
        real_reordered = sim_rad[self.sim_to_real_idx]
        return self._corr_a * real_reordered + self._corr_b

    def real_rad_to_sim_rad(self, real_rad: np.ndarray) -> np.ndarray:
        """Undo corrections and reorder a real-DOF-order radian array back to sim-DOF-order."""
        real_uncorrected = (real_rad - self._corr_b) / self._corr_a
        return real_uncorrected[self.real_to_sim_idx]

    def sim_radps_to_real_radps(self, sim_radps: np.ndarray) -> np.ndarray:
        """Same as sim_rad_to_real_rad but for rates: the constant offset `b` has no
        effect on a derivative, so only the sign/scale factor `a` applies."""
        return self._corr_a * sim_radps[self.sim_to_real_idx]

    def real_radps_to_sim_radps(self, real_radps: np.ndarray) -> np.ndarray:
        return (real_radps / self._corr_a)[self.real_to_sim_idx]

    # ------------------------------------------------------------------
    # Real radians <-> encoder ticks
    # ------------------------------------------------------------------

    def real_rad_to_ticks(self, real_rad: np.ndarray) -> np.ndarray:
        ticks = np.round(self._zero_tick + real_rad * self._ticks_per_rad)
        return ticks.astype(np.int64)

    def ticks_to_real_rad(self, ticks: np.ndarray) -> np.ndarray:
        return (ticks.astype(np.float64) - self._zero_tick) / self._ticks_per_rad

    # ------------------------------------------------------------------
    # Convenience: sim radians <-> ticks, end to end
    # ------------------------------------------------------------------

    def sim_target_to_ticks(self, sim_rad: np.ndarray) -> np.ndarray:
        """Full pipeline for writing a policy-computed target: sim rad -> real ticks."""
        return self.real_rad_to_ticks(self.sim_rad_to_real_rad(sim_rad))

    def ticks_to_sim_rad(self, ticks: np.ndarray) -> np.ndarray:
        """Full pipeline for reading encoder feedback: real ticks -> sim rad."""
        return self.real_rad_to_sim_rad(self.ticks_to_real_rad(ticks))

    def clip_to_soft_limits(self, sim_rad: np.ndarray) -> np.ndarray:
        return np.clip(sim_rad, self.soft_limits_sim[:, 0], self.soft_limits_sim[:, 1])
