# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Imitation-learning reward terms for the LiBR Hexapod.

Reward params must contain only OmegaConf-compatible primitives (str, float,
list[int], etc.) so that Isaac Lab's Hydra config serialization does not fail.
The MotionReference object is built lazily on the first reward call and cached
in a module-level dict keyed by the config primitives.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# Module-level cache: built once per unique (csv_path, gait_period, col_order).
_MOTION_REF_CACHE: dict = {}

# Per-asset joint reorder index cache.
# Maps asset object id -> LongTensor of column indices that reorder
# asset.data.joint_pos columns to match JOINT_NAMES (Sim DOF order).
# None means no reordering needed (ordering already matches).
_JOINT_IDX_CACHE: dict[int, torch.Tensor | None] = {}


def _get_motion_ref(csv_path: str | None, gait_period: float, csv_col_order: list[int] | None):
    """Return a cached MotionReference, constructing it on first call."""
    from .hexapod_mimic_motion import MotionReference

    cache_key = (csv_path, gait_period, tuple(csv_col_order) if csv_col_order is not None else None)
    if cache_key not in _MOTION_REF_CACHE:
        _MOTION_REF_CACHE[cache_key] = MotionReference(
            csv_path=csv_path,
            gait_period=gait_period,
            csv_col_order=list(csv_col_order) if csv_col_order is not None else None,
        )
    return _MOTION_REF_CACHE[cache_key]


def _get_joint_reorder(asset) -> torch.Tensor | None:
    """Return a LongTensor that reorders asset.data.joint_pos to Sim DOF order.

    Called once per asset instance. Prints the detected ordering and warns if a
    reorder is needed (e.g. after an Isaac Lab update that changed joint sorting).

    Returns None if the ordering already matches JOINT_NAMES (no reorder needed).
    """
    from .hexapod_mimic_motion import JOINT_NAMES

    asset_id = id(asset)
    if asset_id in _JOINT_IDX_CACHE:
        return _JOINT_IDX_CACHE[asset_id]

    actual_names: list[str] = list(asset.data.joint_names)
    expected_names: list[str] = JOINT_NAMES  # Sim DOF order

    print(f"[MimicReward] asset.data.joint_names : {actual_names}")
    print(f"[MimicReward] expected Sim DOF order : {expected_names}")

    if actual_names == expected_names:
        print("[MimicReward] Joint ordering matches — no reorder needed.")
        _JOINT_IDX_CACHE[asset_id] = None
        return None

    # Build reorder: for each position in expected order, find the column in actual.
    print("[MimicReward] WARNING: joint ordering mismatch detected. Auto-correcting.")
    try:
        idx = [actual_names.index(name) for name in expected_names]
    except ValueError as e:
        raise ValueError(
            f"[MimicReward] Joint name not found in asset: {e}. Expected {expected_names}, got {actual_names}"
        ) from e

    idx_tensor = torch.tensor(idx, dtype=torch.long)
    print(f"[MimicReward] Reorder index applied: {idx}")
    _JOINT_IDX_CACHE[asset_id] = idx_tensor
    return idx_tensor


def joint_pos_imitation(
    env: ManagerBasedRLEnv,
    csv_path: str | None = None,
    gait_period: float = 0.5,
    csv_col_order: list[int] | None = None,
    joint_sigma: float = 0.25,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward joint positions for matching a reference gait trajectory.

    Uses a Gaussian kernel so the reward is 1.0 when all joints are at their
    reference angles and decays smoothly as error grows:

        r = exp( -||q - q_ref||^2 / sigma^2 )

    The reference is indexed by gait phase = (episode_elapsed_time % gait_period)
    / gait_period.

    On the first call this function validates that the joint ordering returned by
    asset.data.joint_pos matches the Sim DOF order expected by MotionReference.
    If not, it logs a warning and applies an automatic reorder so the comparison
    is always correct regardless of Isaac Lab version.

    Args:
        env:           The RL environment.
        csv_path:      Path to reference gait CSV, or None to use the built-in
                       sinusoidal tripod gait.
        gait_period:   Duration of one gait cycle in seconds.
        csv_col_order: Integer list to reorder CSV columns to Isaac Lab's
                       alphabetical joint order.  Pass SIM_DOF_TO_ALPHA if the
                       CSV was produced by play.py / playReal.py.
        joint_sigma:   Gaussian width in radians.  Smaller -> stricter tracking.
        asset_cfg:     Scene entity config for the robot articulation.
    """
    motion_ref = _get_motion_ref(csv_path, gait_period, csv_col_order)

    asset = env.scene[asset_cfg.name]

    # Validate/correct joint ordering on first call (cached after that).
    joint_idx = _get_joint_reorder(asset)

    q_current = asset.data.joint_pos  # [N, num_joints] in asset's native order
    if joint_idx is not None:
        q_current = q_current[:, joint_idx.to(q_current.device)]  # reorder to Sim DOF

    episode_time = env.episode_length_buf.float() * env.step_dt  # [N]
    phase = motion_ref.get_phase(episode_time)
    q_ref = motion_ref.get_reference(phase)  # [N, 8] Sim DOF order

    # Per-joint Gaussian averaged across joints.  Summing squared errors across
    # all joints causes two high-amplitude spine joints to kill the gradient for
    # all six leg joints — per-joint mean keeps the signal alive independently.
    per_joint = torch.exp(-((q_current - q_ref) ** 2) / (joint_sigma**2))  # [N, 8]
    return per_joint.mean(dim=-1)  # [N]


def spine_pos_imitation(
    env: ManagerBasedRLEnv,
    csv_path: str | None = None,
    gait_period: float = 0.5,
    csv_col_order: list[int] | None = None,
    spine_sigma: float = 0.15,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Like joint_pos_imitation but restricted to the two spine joints only.

    Spine joint indices in Sim DOF order: BackLink_Joint=0, FrontLink_Joint=1.
    """
    SPINE_COLS_SIM_DOF = [0, 1]  # BackLink_Joint, FrontLink_Joint in Sim DOF order

    motion_ref = _get_motion_ref(csv_path, gait_period, csv_col_order)

    asset = env.scene[asset_cfg.name]

    joint_idx = _get_joint_reorder(asset)

    q_current = asset.data.joint_pos
    if joint_idx is not None:
        q_current = q_current[:, joint_idx.to(q_current.device)]

    q_current_spine = q_current[:, SPINE_COLS_SIM_DOF]  # [N, 2]

    episode_time = env.episode_length_buf.float() * env.step_dt
    phase = motion_ref.get_phase(episode_time)
    q_ref = motion_ref.get_reference(phase)[:, SPINE_COLS_SIM_DOF]  # [N, 2]

    per_joint = torch.exp(-((q_current_spine - q_ref) ** 2) / (spine_sigma**2))  # [N, 2]
    return per_joint.mean(dim=-1)
