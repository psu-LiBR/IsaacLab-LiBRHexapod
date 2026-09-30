# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""High-level command screening on complete spine-phase cycles; numpy only."""

import numpy as np


def euler(q):
    # Frozen IsaacLab math.py documents x,y,z,w (not legacy w,x,y,z).
    x, y, z, w = np.moveaxis(np.asarray(q), -1, 0)
    return np.stack(
        [
            np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)),
            np.arcsin(np.clip(2 * (w * y - z * x), -1, 1)),
            np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)),
        ],
        axis=-1,
    )


def summarize(
    actions,
    previous,
    phase_steps,
    trajectory,
    initial_q,
    quaternions,
    heights,
    failures,
    torso,
    dt=0.02,
    period=1.0,
    body_length=0.315,
):
    actions = np.asarray(actions, dtype=np.int64).reshape(len(actions), -1)
    T, E = actions.shape
    phases = np.asarray(phase_steps, dtype=np.int64).reshape(T, E)
    P = round(period / dt)
    assert abs(P * dt - period) < 1e-8 and P == 50
    bits = (actions[..., None] >> np.arange(6)) & 1
    prev = (np.asarray(previous, dtype=np.int64).reshape(E, 1) >> np.arange(6)) & 1
    flips = bits != np.concatenate([prev[None], bits[:-1]], axis=0)
    reset = np.concatenate([np.zeros((1, E), dtype=bool), np.diff(phases, axis=0) <= 0], axis=0) | (phases == 0)
    # phase==0 alone is only episode reset at step0, not every cycle boundary.
    flips[reset] = False
    counts = []
    events = []
    cycle_ids = []
    env_cycles = [0] * E
    dwell = []
    censored = 0
    for env in range(E):
        boundaries = np.r_[0, np.flatnonzero(np.diff(phases[:, env]) <= 0) + 1, T]
        for lo, hi in zip(boundaries[:-1], boundaries[1:]):
            for cyc in np.unique(phases[lo:hi, env] // P):
                ids = np.flatnonzero(phases[lo:hi, env] // P == cyc) + lo
                if len(ids) == P and np.array_equal(phases[ids, env] % P, np.arange(P)):
                    counts.append(flips[ids, env].sum(axis=0))
                    events.append(np.any(flips[ids, env], axis=-1).sum())
                    cycle_ids.append([env, int(cyc)])
                    env_cycles[env] += 1
            for leg in range(6):
                change = np.flatnonzero(flips[lo:hi, env, leg]) + lo
                # First and last spans are censored. Differences between observed changes are complete dwell runs.
                dwell.extend((np.diff(change) * dt).tolist())
                censored += 2
    c = np.asarray(counts, dtype=int).reshape(-1, 6)
    events = np.asarray(events, dtype=int)
    yaw0 = euler(initial_q)[:, 2]
    tr = np.asarray(trajectory)
    forward = tr[:, :, 0] * np.cos(yaw0) + tr[:, :, 1] * np.sin(yaw0)
    lateral = -tr[:, :, 0] * np.sin(yaw0) + tr[:, :, 1] * np.cos(yaw0)
    angles = euler(quaternions)
    yaw = np.arctan2(np.sin(angles[:, :, 2] - yaw0), np.cos(angles[:, :, 2] - yaw0)) * 180 / np.pi
    failures = np.asarray(failures, dtype=bool)
    torso = np.asarray(torso, dtype=bool)
    missing = sum(n == 0 for n in env_cycles)
    reasons = []
    if not len(c) or missing:
        reasons.append("insufficient_complete_cycles")
    if len(c) and c.max() > 8:
        reasons.append("per_leg_cycle_switch_exceeds_8")
    if failures.any():
        reasons.append("fall_observed")
    if torso.any():
        reasons.append("torso_contact_observed")
    if not np.all(forward[-1] > 0):
        reasons.append("nonforward_environment")
    if reset.any():
        reasons.append("reset_in_measurement")
    dw = np.asarray(dwell)

    def rate(mask):
        return float(np.mean(mask)) if len(c) else None

    out = dict(
        profile="complete-spine-cycle-command-screen",
        period_s=period,
        control_dt=dt,
        period_steps=P,
        complete_cycles=len(c),
        complete_cycles_per_env=env_cycles,
        missing_cycle_envs=missing,
        cycle_env_and_index=cycle_ids,
        per_leg_per_cycle=c.tolist(),
        per_cycle_six_leg_mean=c.mean(axis=1).tolist() if len(c) else [],
        whole_command_events_per_cycle=events.tolist(),
        per_leg_max=c.max(axis=0).tolist() if len(c) else None,
        per_leg_mean=c.mean(axis=0).tolist() if len(c) else None,
        per_leg_p95=np.percentile(c, 95, axis=0).tolist() if len(c) else None,
        worst_leg_cycle=int(c.max()) if len(c) else None,
        leg_cycle_fraction_le8=rate(c <= 8),
        leg_cycle_fraction_le6=rate(c <= 6),
        leg_cycle_fraction_3to4=rate((c >= 3) & (c <= 4)),
        all_legs_cycle_fraction_le8=rate(np.all(c <= 8, axis=1)),
        all_legs_cycle_fraction_le6=rate(np.all(c <= 6, axis=1)),
        dwell_complete_count=len(dwell),
        dwell_censored_count=censored,
        dwell_min_s=float(dw.min()) if len(dw) else None,
        dwell_p05_s=float(np.percentile(dw, 5)) if len(dw) else None,
        dwell_median_s=float(np.median(dw)) if len(dw) else None,
        single_step_pulse_fraction=float(np.mean(dw <= dt + 1e-8)) if len(dw) else None,
        dwell_histogram_counts=np.histogram(
            dw, bins=[0, 0.021, 0.041, 0.081, 0.161, 0.321, 0.641, 1.281, float("inf")]
        )[0].tolist(),
        dwell_histogram_edges_s=[0, 0.021, 0.041, 0.081, 0.161, 0.321, 0.641, 1.281, None],
        forward_bl_per_cycle=float(forward[-1].mean() / body_length / (T * dt / period)),
        per_env_forward_m=forward[-1].tolist(),
        lateral_abs_max_bl=float(np.abs(lateral).max() / body_length),
        lateral_rms_bl=float(np.sqrt(np.mean(lateral * lateral)) / body_length),
        yaw_abs_max_deg=float(np.abs(yaw).max()),
        yaw_rms_deg=float(np.sqrt(np.mean(yaw * yaw))),
        roll_abs_max_deg=float(np.abs(angles[:, :, 0]).max() * 180 / np.pi),
        pitch_abs_max_deg=float(np.abs(angles[:, :, 1]).max() * 180 / np.pi),
        height_min_m=float(np.min(heights)),
        fall_env_fraction=float(failures.any(axis=0).mean()),
        torso_contact_env_fraction=float(torso.any(axis=0).mean()),
        basic_screen_pass=not reasons,
        rejection_reasons=reasons,
        full_qualification="pending_thresholds",
        pending_thresholds=["minimum_dwell", "yaw", "lateral_drift", "pose"],
        notes=(
            "High-level commands only. Dwell and straightness recorded, hardware thresholds not specified. "
            "Basic pass is not deployment qualification."
        ),
    )
    out["preview_env0"] = {
        "bits": bits[:, 0].tolist(),
        "forward_m": forward[:, 0].tolist(),
        "lateral_m": lateral[:, 0].tolist(),
        "yaw_deg": yaw[:, 0].tolist(),
        "time_s": (np.arange(T) * dt).tolist(),
    }
    out["sensitivity"] = {
        "minimum_dwell_s": {str(x): bool(len(dw) and dw.min() >= x) for x in [0.04, 0.08, 0.12, 0.16]},
        "yaw_max_deg": {str(x): bool(np.abs(yaw).max() <= x) for x in [5, 10, 15, 20]},
        "lateral_max_bl": {str(x): bool(np.abs(lateral).max() / body_length <= x) for x in [0.1, 0.2, 0.3, 0.5]},
    }
    return out
