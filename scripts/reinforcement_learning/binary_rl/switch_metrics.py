# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Per-leg contact-command switch counts on complete spine-wave cycles; numpy only.

A *cycle* is one period of the scripted spine wave (``GAIT_PERIOD_S`` in ``hexapod_binary_env_cfg.py``).
It is measured in control steps: ``period / dt`` steps, with ``dt`` the policy step (``decimation * sim.dt``).
The spine wave is driven by ``t = episode_length_buf * step_dt``, so cycles are aligned to the episode start and
the ``phase_steps`` passed to :func:`summarize` are that ``episode_length_buf``.

A *switch* is a change of one leg's commanded contact bit (stance <-> lift) between consecutive control steps,
counted on both edges. The metric acts on commanded bits, not on measured foot contact.
"""

import numpy as np

DEFAULT_CONTROL_DT = 0.02
"""Policy step [s]: ``decimation (4) * sim.dt (0.005)``. Kept in sync with the env by ``test_switch_metrics.py``."""

DEFAULT_PERIOD_S = 1.0
"""Spine-wave (gait cycle) period [s]. Kept in sync with ``GAIT_PERIOD_S`` by ``test_switch_metrics.py``."""

DEFAULT_MAX_SWITCHES_PER_LEG_CYCLE = 5
"""Largest per-leg switch count in one complete cycle that still passes the basic screen."""

NUM_LEGS = 6


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
    dt=DEFAULT_CONTROL_DT,
    period=DEFAULT_PERIOD_S,
    body_length=0.315,
    max_switches_per_leg_cycle=DEFAULT_MAX_SWITCHES_PER_LEG_CYCLE,
):
    """Summarize command switching and drift for a batch of rollouts.

    Args:
        actions: Commanded 6-bit leg action index per step, shape [T, E] (bit ``i`` is leg ``i``).
        previous: Action index applied before the first step, shape [E].
        phase_steps: ``episode_length_buf`` at each step, shape [T, E].
        trajectory: Base position in the world frame [m], shape [T, E, 3].
        initial_q: Base orientation at the start (xyzw), shape [E, 4].
        quaternions: Base orientation per step (xyzw), shape [T, E, 4].
        heights: Base height [m], shape [T, E].
        failures: Fall flag per step, shape [T, E].
        torso: Torso-contact flag per step, shape [T, E].
        dt: Policy control step [s].
        period: Spine-wave (gait cycle) period [s]. Must be a whole number of control steps.
        body_length: Body length used to normalize distances [m].
        max_switches_per_leg_cycle: A complete cycle in which any leg switches more often than this rejects the run.

    Returns:
        A JSON-serializable dict of metrics. ``basic_screen_pass`` / ``rejection_reasons`` give the screen result;
        ``warnings`` lists non-failing flags such as ``frozen_legs``.

    Raises:
        ValueError: If ``period`` is not a whole number of control steps.
    """
    actions = np.asarray(actions, dtype=np.int64).reshape(len(actions), -1)
    T, E = actions.shape
    phases = np.asarray(phase_steps, dtype=np.int64).reshape(T, E)
    P = round(period / dt)
    if P < 1 or abs(P * dt - period) > 1e-8:
        raise ValueError(f"period {period} s is not a whole number of {dt} s control steps")
    bits = (actions[..., None] >> np.arange(NUM_LEGS)) & 1
    prev = (np.asarray(previous, dtype=np.int64).reshape(E, 1) >> np.arange(NUM_LEGS)) & 1
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
            for leg in range(NUM_LEGS):
                change = np.flatnonzero(flips[lo:hi, env, leg]) + lo
                # First and last spans are censored. Differences between observed changes are complete dwell runs.
                # A leg that never switches has one span, censored on both sides.
                dwell.extend((np.diff(change) * dt).tolist())
                censored += 2 if len(change) else 1
    c = np.asarray(counts, dtype=int).reshape(-1, NUM_LEGS)
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
    # A leg is frozen in an env when its command never switches over the measurement; flagged, not rejected.
    frozen_env_fraction = (~flips.any(axis=0)).mean(axis=0)
    frozen_legs = np.flatnonzero(frozen_env_fraction == 1.0).tolist()
    reasons = []
    if not len(c) or missing:
        reasons.append("insufficient_complete_cycles")
    if len(c) and c.max() > max_switches_per_leg_cycle:
        reasons.append(f"per_leg_cycle_switch_exceeds_{max_switches_per_leg_cycle}")
    if failures.any():
        reasons.append("fall_observed")
    if torso.any():
        reasons.append("torso_contact_observed")
    if not np.all(forward[-1] > 0):
        reasons.append("nonforward_environment")
    if reset.any():
        reasons.append("reset_in_measurement")
    warnings = ["frozen_legs"] if frozen_legs else []
    dw = np.asarray(dwell)

    def rate(mask):
        return float(np.mean(mask)) if len(c) else None

    out = dict(
        profile="complete-spine-cycle-command-screen",
        period_s=period,
        control_dt=dt,
        period_steps=P,
        max_switches_per_leg_cycle=max_switches_per_leg_cycle,
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
        leg_cycle_fraction_within_limit=rate(c <= max_switches_per_leg_cycle),
        all_legs_cycle_fraction_within_limit=rate(np.all(c <= max_switches_per_leg_cycle, axis=1)),
        frozen_legs=frozen_legs,
        per_leg_frozen_env_fraction=frozen_env_fraction.tolist(),
        per_leg_stance_fraction=bits.mean(axis=(0, 1)).tolist(),
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
        warnings=warnings,
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


BULKY_KEYS = (
    "cycle_env_and_index",
    "per_leg_per_cycle",
    "per_cycle_six_leg_mean",
    "whole_command_events_per_cycle",
    "per_env_forward_m",
    "complete_cycles_per_env",
    "preview_env0",
)
"""Per-cycle and per-step lists that :func:`summarize` returns but that are too long for a results table."""


def compact(summary):
    """Drop the long per-cycle and per-step lists from a :func:`summarize` result, keeping the scalar metrics."""
    return {k: v for k, v in summary.items() if k not in BULKY_KEYS}


def _numpy(x):
    """Copy a torch tensor (any device) or array-like to numpy, so later in-place updates to ``x`` do not leak in."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.array(x)


class SwitchRecorder:
    """Collects one fixed-window rollout step by step and summarizes it with :func:`summarize`.

    Per step, ``action`` and ``phase`` are the values *before* ``env.step`` (the commanded bits and the
    ``episode_length_buf`` the spine wave is driven from); ``position``, ``quaternion``, ``failure`` and ``torso``
    are read *after* it. Tensors are copied to numpy as they are recorded.

    Args:
        initial_position: Base position at the start of the window, in the env frame [m], shape [E, 3].
        initial_quaternion: Base orientation at the start of the window (xyzw), shape [E, 4].
        previous_action: Action index applied just before the window (the last warm-up action), shape [E]. If
            None, the first recorded action counts as no switch.
    """

    def __init__(self, initial_position, initial_quaternion, previous_action=None):
        self.initial_position = _numpy(initial_position)
        self.initial_quaternion = _numpy(initial_quaternion)
        self.previous_action = None if previous_action is None else _numpy(previous_action)
        self._steps = {k: [] for k in ("action", "phase", "position", "quaternion", "failure", "torso")}

    def record(self, action, phase, position, quaternion, failure, torso):
        """Append one step: ``action`` and ``phase`` [E] before the step, the rest after it (see class docs)."""
        for key, value in zip(self._steps, (action, phase, position, quaternion, failure, torso)):
            self._steps[key].append(_numpy(value))

    def summarize(self, **kwargs):
        """Run :func:`summarize` on the recorded window. ``kwargs`` are ``dt``, ``period``, ``body_length``, etc."""
        steps = {k: np.stack(v) for k, v in self._steps.items()}
        actions = steps["action"]
        return summarize(
            actions=actions,
            previous=actions[0] if self.previous_action is None else self.previous_action,
            phase_steps=steps["phase"],
            trajectory=steps["position"] - self.initial_position[None],
            initial_q=self.initial_quaternion,
            quaternions=steps["quaternion"],
            heights=steps["position"][..., 2],
            failures=steps["failure"],
            torso=steps["torso"],
            **kwargs,
        )
