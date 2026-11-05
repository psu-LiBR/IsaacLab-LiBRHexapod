# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Imitation-learning reward terms for the LiBR Hexapod.

Reward params hold only OmegaConf-compatible primitives (str, float, list[int]) so Hydra serialization works;
the MotionReference is built lazily on the first call and cached by those primitives.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# MotionReference per (csv_path, gait_period, col_order).
_MOTION_REF_CACHE: dict = {}

# Per asset id: column indices that reorder asset.data.joint_pos to JOINT_NAMES order (None = already matching).
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


_CLOCK_OFFSET_ATTR = "_mimic_clock_offset"
"""Name of the per-env reference-clock offset buffer [s], shape [num_envs], created lazily on the env."""

_PHASE_IDX_ATTR = "_mimic_phase_idx"
"""Name of the per-env reference row index buffer (long, shape [num_envs]) used by :func:`joint_pos_imitation_phase`."""

_PHASE_STEP_ATTR = "_mimic_phase_step"
"""Name of the per-env buffer with the rows the phase advanced on the last step (long, shape [num_envs])."""


def _lag_tensor(servo_lag_s: float | list[float], device: torch.device | str) -> torch.Tensor:
    """Return the servo lag as a [8] tensor in sim DOF order [s] (a scalar is broadcast to every joint)."""
    from .hexapod_mimic_motion import NUM_JOINTS

    lag = torch.as_tensor(list(servo_lag_s) if not isinstance(servo_lag_s, (int, float)) else servo_lag_s)
    lag = lag.to(device=device, dtype=torch.float32)
    if lag.ndim == 0:
        lag = lag.expand(NUM_JOINTS)
    if lag.shape != (NUM_JOINTS,):
        raise ValueError(f"servo_lag_s must be a float or a list of {NUM_JOINTS} floats, got shape {tuple(lag.shape)}")
    return lag


def _reference_at(motion_ref, clock: torch.Tensor, lag: torch.Tensor) -> torch.Tensor:
    """Reference pose [N, 8] (sim DOF order) where joint j is evaluated at ``clock - lag[j]`` [s]."""
    phase = motion_ref.get_phase(clock[:, None] - lag[None, :])  # [N, 8]; % wraps negatives
    return motion_ref.get_reference_per_joint(phase)


def _clock_offset(env: ManagerBasedRLEnv, device: torch.device | str) -> torch.Tensor | float:
    """Per-env reference-clock offset [s] (shape [N]); 0.0 if the env has no reset event that sets one."""
    offset = getattr(env, _CLOCK_OFFSET_ATTR, None)
    return 0.0 if offset is None else offset.to(device)


def reset_joints_to_mimic_reference(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    csv_path: str | None = None,
    gait_period: float = 1.0,
    csv_col_order: list[int] | None = None,
    servo_lag_s: float | list[float] = 0.0,
    random_phase: bool = False,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Reset the joints to the reference gait pose (and velocity) the imitation reward expects at episode start.

    ``joint_pos_imitation`` restarts its clock at every reset (``episode_length_buf`` is zeroed after the reset
    events). The reference clock of an env is ``episode_length_buf * step_dt + offset``; this event draws the
    per-env ``offset`` and places each joint ``j`` on the reference at ``offset - servo_lag_s[j]``, which is exactly
    the reference time the reward uses at counter 0 (the first reward is computed at counter 1, one control step
    later). Resetting to the default pose instead leaves the robot ~1 rad away from the reference at step 0, so each
    episode would begin with a catch-up transient.

    Args:
        env: The RL environment. The offsets are stored in ``env._mimic_clock_offset`` (shape [num_envs], created
            on first use and zero for envs that were never reset by this event).
        env_ids: Environments being reset.
        csv_path: Path to the reference gait CSV (same value as the imitation reward).
        gait_period: Duration of one gait cycle [s].
        csv_col_order: Integer list that reorders the CSV columns to the sim DOF order, or None.
        servo_lag_s: Servo delay compensation [s] as in :func:`joint_pos_imitation`: a float, or a list of 8 floats in
            sim DOF order.
        random_phase: If True, start every reset env at a uniformly random point of the gait cycle (offset drawn in
            [0, ``gait_period``) s); if False the offset is 0 and every episode starts at phase 0.
        asset_cfg: Scene entity config for the robot articulation.
    """
    motion_ref = _get_motion_ref(csv_path, gait_period, csv_col_order)
    asset = env.scene[asset_cfg.name]
    joint_idx = _get_joint_reorder(asset)

    device = asset.data.joint_pos.torch.device
    num_envs = asset.data.joint_pos.torch.shape[0]
    offsets = getattr(env, _CLOCK_OFFSET_ATTR, None)
    if offsets is None or offsets.shape[0] != num_envs or offsets.device != device:
        offsets = torch.zeros(num_envs, device=device)
        setattr(env, _CLOCK_OFFSET_ATTR, offsets)
    env_ids = torch.as_tensor(env_ids, device=device)
    if random_phase:
        offsets[env_ids] = torch.rand(len(env_ids), device=device) * motion_ref.gait_period
    else:
        offsets[env_ids] = 0.0

    lag = _lag_tensor(servo_lag_s, device)
    clock = offsets[env_ids]  # [n]
    phase = motion_ref.get_phase(clock[:, None] - lag[None, :])  # [n, 8]
    q_ref = motion_ref.get_reference_per_joint(phase)  # [n, 8] Sim DOF order
    # forward difference over 2% of the cycle (4 table rows) for the joint velocity [rad/s]
    dphi = 0.02
    q_next = motion_ref.get_reference_per_joint((phase + dphi) % 1.0)
    dq_ref = (q_next - q_ref) / (dphi * motion_ref.gait_period)  # the table is periodic, so the wrap-around is fine

    if joint_idx is not None:  # back from Sim DOF order to the asset's native order
        inverse = torch.argsort(joint_idx).to(device)
        q_ref = q_ref[:, inverse]
        dq_ref = dq_ref[:, inverse]

    pos_limits = asset.data.soft_joint_pos_limits.torch[env_ids]
    joint_pos = q_ref.clamp(pos_limits[..., 0], pos_limits[..., 1])
    vel_limits = asset.data.soft_joint_vel_limits.torch[env_ids]
    joint_vel = dq_ref.clamp(-vel_limits, vel_limits)

    asset.write_joint_position_to_sim_index(position=joint_pos, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)
    asset.write_joint_velocity_to_sim_index(velocity=joint_vel, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)


def joint_pos_imitation(
    env: ManagerBasedRLEnv,
    csv_path: str | None = None,
    gait_period: float = 0.5,
    csv_col_order: list[int] | None = None,
    joint_sigma: float = 0.25,
    servo_lag_s: float | list[float] = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward joint positions for matching a reference gait trajectory.

    Uses a Gaussian kernel so the reward is 1.0 when all joints are at their
    reference angles and decays smoothly as error grows:

        r = exp( -||q - q_ref||^2 / sigma^2 )

    The reference of joint ``j`` is indexed by gait phase = (clock_j % gait_period) / gait_period with
    ``clock_j = episode_elapsed_time + clock_offset - servo_lag_s[j]``. ``clock_offset`` is the per-env start phase
    drawn by :func:`reset_joints_to_mimic_reference` (zero if the env has no such event).

    On the first call this function validates that the joint ordering returned by
    asset.data.joint_pos matches the Sim DOF order expected by MotionReference.
    If not, it logs a warning and applies an automatic reorder so the comparison
    is always correct regardless of Isaac Lab version.

    Args:
        env:           The RL environment.
        csv_path:      Path to reference gait CSV, or None to use the built-in
                       sinusoidal tripod gait.
        gait_period:   Duration of one gait cycle in seconds.
        csv_col_order: Integer list that reorders the CSV columns to the sim DOF order, or None.
        joint_sigma:   Gaussian width in radians.  Smaller -> stricter tracking.
        servo_lag_s:   Servo command-to-motion delay [s]: a float, or a list of 8 floats in sim DOF order. Joint
                       ``j``'s reference is evaluated at ``t - servo_lag_s[j]`` so a command issued at ``t`` is
                       rewarded when it shows up at the joint, i.e. the policy only has to command the reference
                       at its own current time (as playReal.py does).
        asset_cfg:     Scene entity config for the robot articulation.
    """
    motion_ref = _get_motion_ref(csv_path, gait_period, csv_col_order)

    asset = env.scene[asset_cfg.name]

    # Validate/correct joint ordering on first call (cached after that).
    joint_idx = _get_joint_reorder(asset)

    q_current = asset.data.joint_pos.torch  # [N, num_joints] in asset's native order
    if joint_idx is not None:
        q_current = q_current[:, joint_idx.to(q_current.device)]  # reorder to Sim DOF

    clock = env.episode_length_buf.float() * env.step_dt + _clock_offset(env, q_current.device)  # [N]
    q_ref = _reference_at(motion_ref, clock, _lag_tensor(servo_lag_s, q_current.device))  # [N, 8] Sim DOF order

    # Per-joint Gaussian averaged across joints.  Summing squared errors across
    # all joints causes two high-amplitude spine joints to kill the gradient for
    # all six leg joints — per-joint mean keeps the signal alive independently.
    per_joint = torch.exp(-((q_current - q_ref) ** 2) / (joint_sigma**2))  # [N, 8]
    return per_joint.mean(dim=-1)  # [N]


def joint_pos_imitation_phase(
    env: ManagerBasedRLEnv,
    csv_path: str | None = None,
    gait_period: float = 1.0,
    csv_col_order: list[int] | None = None,
    joint_sigma: float = 0.25,
    min_step_rows: int = 2,
    max_step_rows: int = 6,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward joint positions for matching a reference gait, with the gait phase tracked from the measured pose.

    The per-joint Gaussian kernel is the same as in :func:`joint_pos_imitation` (``exp(-(q - q_ref)^2 / sigma^2)``
    averaged over the 8 joints), but the reference row is not indexed by a wall clock. Each env keeps a phase index
    ``k`` into the reference table; every control step the phase advances by between ``min_step_rows`` and
    ``max_step_rows`` rows, choosing the advance whose reference pose best matches the measured joint positions,
    and the reward is that best kernel score. The policy therefore picks its own gait frequency within
    ``[min_step_rows, max_step_rows]`` rows per control step. The nominal advance is
    ``num_points * step_dt / gait_period`` rows (e.g. 200 * 0.02 / 1.0 = 4 rows/step), so the defaults allow
    0.5x-1.5x the reference speed. ``min_step_rows > 0`` forces the phase forward every step, so a frozen pose
    loses reward as the reference moves away from it.

    There is no servo-lag parameter: the measured pose already includes the servo lag. The first call of each
    episode seeds the phase from the globally nearest row of the whole table (the reset event places each joint
    behind the clock offset by its servo lag, so the offset alone would mis-seed it) and gives a zero advance.

    ``env._mimic_phase_step`` holds the last per-env advance [rows] for inspection (gait frequency [Hz] =
    step / num_points / step_dt).

    Args:
        env: The RL environment.
        csv_path: Path to the reference gait CSV, or None to use the built-in sinusoidal tripod gait.
        gait_period: Duration of one gait cycle of the reference table [s].
        csv_col_order: Integer list that reorders the CSV columns to the sim DOF order, or None.
        joint_sigma: Gaussian width [rad]. Smaller -> stricter tracking.
        min_step_rows: Minimum phase advance per control step [table rows]. Must be > 0.
        max_step_rows: Maximum phase advance per control step [table rows]. Must be < the table size.
        asset_cfg: Scene entity config for the robot articulation.

    Returns:
        Reward tensor of shape [num_envs], in [0, 1].
    """
    motion_ref = _get_motion_ref(csv_path, gait_period, csv_col_order)
    asset = env.scene[asset_cfg.name]
    joint_idx = _get_joint_reorder(asset)

    q = asset.data.joint_pos.torch  # [N, num_joints] in asset's native order
    if joint_idx is not None:
        q = q[:, joint_idx.to(q.device)]  # reorder to Sim DOF
    device = q.device
    n_envs = q.shape[0]
    num_rows = motion_ref.num_points

    if not 0 < min_step_rows <= max_step_rows < num_rows:
        raise ValueError(
            f"Need 0 < min_step_rows <= max_step_rows < num_points, got {min_step_rows}, {max_step_rows}, {num_rows}."
        )

    # Lazily create the per-env phase buffers.
    k = getattr(env, _PHASE_IDX_ATTR, None)
    created = k is None or k.shape[0] != n_envs or k.device != device
    if created:
        k = torch.zeros(n_envs, dtype=torch.long, device=device)
        setattr(env, _PHASE_IDX_ATTR, k)
    step_buf = getattr(env, _PHASE_STEP_ATTR, None)
    if step_buf is None or step_buf.shape[0] != n_envs or step_buf.device != device:
        step_buf = torch.zeros(n_envs, dtype=torch.long, device=device)
        setattr(env, _PHASE_STEP_ATTR, step_buf)

    # First reward call after a reset (episode_length_buf is incremented before rewards are computed).
    init = env.episode_length_buf == 1
    if created:
        init = torch.ones_like(init)

    # Best-matching advance within the allowed window.
    offs = torch.arange(min_step_rows, max_step_rows + 1, device=device)  # [W]
    rows = (k[:, None] + offs[None, :]) % num_rows  # [N, W]
    ref = motion_ref.get_reference_rows(rows)  # [N, W, 8]
    score = torch.exp(-((q[:, None, :] - ref) ** 2) / joint_sigma**2).mean(dim=-1)  # [N, W]
    best = score.argmax(dim=1)
    new_k = rows.gather(1, best[:, None]).squeeze(1)
    reward = score.gather(1, best[:, None]).squeeze(1)
    step = offs[best]

    # Newly reset envs: seed from the globally nearest table row.
    if init.any():
        ids = init.nonzero(as_tuple=True)[0]
        all_rows = motion_ref.get_reference_rows(torch.arange(num_rows, device=device))  # [R, 8]
        s = torch.exp(-((q[ids, None, :] - all_rows[None]) ** 2) / joint_sigma**2).mean(dim=-1)  # [n, R]
        b = s.argmax(dim=1)
        new_k[ids] = b
        reward[ids] = s.gather(1, b[:, None]).squeeze(1)
        step[ids] = 0

    k.copy_(new_k)
    step_buf.copy_(step)
    return reward


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

    q_current = asset.data.joint_pos.torch
    if joint_idx is not None:
        q_current = q_current[:, joint_idx.to(q_current.device)]

    q_current_spine = q_current[:, SPINE_COLS_SIM_DOF]  # [N, 2]

    episode_time = env.episode_length_buf.float() * env.step_dt + _clock_offset(env, q_current.device)
    phase = motion_ref.get_phase(episode_time)
    q_ref = motion_ref.get_reference(phase)[:, SPINE_COLS_SIM_DOF]  # [N, 2]

    per_joint = torch.exp(-((q_current_spine - q_ref) ** 2) / (spine_sigma**2))  # [N, 2]
    return per_joint.mean(dim=-1)
