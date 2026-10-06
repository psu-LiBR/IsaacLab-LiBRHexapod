#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Offline sim-vs-onnx trace validation. Run this (on the dev machine AND the
actual Pi -- onnxruntime/opset behavior can differ by platform) before ever
letting a policy touch the real robot.

Two modes:

  direct   -- feeds each recorded obs row straight into onnxruntime and diffs
              against the recorded action row. Validates the ONNX export +
              onnxruntime on this machine, independent of any obs-building code.
              Works against any CSV with obs_*/action_* columns, including
              sim2real's own CsvRunLogger output from a previous run.

  pipeline -- additionally rebuilds the obs vector via the real ObsBuilder from
              raw sensor-equivalent fields (gyro, gravity, joint_pos, joint_vel,
              last_action, command) and diffs the *rebuilt* obs against the
              recorded obs before even running inference -- isolates obs-
              construction bugs (wrong order, wrong sign, wrong q_default) from
              ONNX/export bugs. Requires a trace produced by
              `play.py --dump_obs_action_csv`, and --config (for q_default_sim).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim2real.deployment_config import load_deployment_config  # noqa: E402
from sim2real.joint_mapping import ordered_array  # noqa: E402
from sim2real.policy_runner import PolicyRunner  # noqa: E402
from sim2real.profiles import NUM_JOINTS, PROFILES, make_obs_builder  # noqa: E402


def _read_columns(row: dict, prefix: str, n: int) -> np.ndarray:
    return np.array([float(row[f"{prefix}_{i}"]) for i in range(n)], dtype=np.float64)


def load_trace(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def run_direct_check(rows: list[dict], policy: PolicyRunner, obs_dim: int, action_dim: int, tol: float) -> bool:
    max_err = 0.0
    worst_step = None
    for row in rows:
        obs = _read_columns(row, "obs", obs_dim)
        recorded_action = _read_columns(row, "action", action_dim)
        predicted_action = policy.step(obs).astype(np.float64)
        err = float(np.max(np.abs(predicted_action - recorded_action)))
        if err > max_err:
            max_err = err
            worst_step = row.get("step", "?")

    print(f"[direct] {len(rows)} rows, max_abs_action_err={max_err:.6f} (worst step={worst_step})")
    ok = max_err <= tol
    print("[direct] PASS" if ok else f"[direct] FAIL (tolerance={tol})")
    return ok


def run_pipeline_check(
    rows: list[dict],
    policy: PolicyRunner,
    obs_dim: int,
    action_dim: int,
    command_dim: int,
    q_default_sim: np.ndarray,
    tol: float,
) -> bool:
    obs_builder = make_obs_builder(policy.profile.name)
    max_obs_err = 0.0
    max_action_err = 0.0
    worst_step = None

    for row in rows:
        gyro = _read_columns(row, "gyro", 3)
        gravity = _read_columns(row, "gravity", 3)
        command = _read_columns(row, "command", command_dim)
        joint_pos = _read_columns(row, "joint_pos", NUM_JOINTS)
        joint_vel = _read_columns(row, "joint_vel", NUM_JOINTS)
        last_action = _read_columns(row, "last_action", policy.profile.last_action_dim)
        recorded_obs = _read_columns(row, "obs", obs_dim)
        recorded_action = _read_columns(row, "action", action_dim)

        rebuilt_obs = obs_builder.build(
            gyro, gravity, joint_pos, joint_vel, last_action, q_default_sim, command
        ).astype(np.float64)
        obs_err = float(np.max(np.abs(rebuilt_obs - recorded_obs)))
        max_obs_err = max(max_obs_err, obs_err)

        predicted_action = policy.step(rebuilt_obs).astype(np.float64)
        action_err = float(np.max(np.abs(predicted_action - recorded_action)))
        if action_err > max_action_err:
            max_action_err = action_err
            worst_step = row.get("step", "?")

    print(
        f"[pipeline] {len(rows)} rows, max_abs_obs_err={max_obs_err:.6f}, "
        f"max_abs_action_err={max_action_err:.6f} (worst step={worst_step})"
    )
    ok = max_obs_err <= tol and max_action_err <= tol
    print("[pipeline] PASS" if ok else f"[pipeline] FAIL (tolerance={tol})")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trace", required=True, help="CSV reference trace.")
    parser.add_argument("--policy", required=True, help="Path to policy.onnx.")
    parser.add_argument("--profile", required=True, choices=sorted(PROFILES))
    parser.add_argument("--mode", choices=["direct", "pipeline"], default="direct")
    parser.add_argument("--config", default=None, help="deployment.yaml, required for --mode pipeline.")
    parser.add_argument("--tol", type=float, default=1e-3)
    args = parser.parse_args()

    spec = PROFILES[args.profile]
    policy = PolicyRunner(args.policy, spec)
    rows = load_trace(args.trace)
    if not rows:
        print(f"[validate_onnx] trace '{args.trace}' has no rows", file=sys.stderr)
        sys.exit(1)

    if args.mode == "direct":
        ok = run_direct_check(rows, policy, spec.obs_dim, spec.action_dim, args.tol)
    else:
        if args.config is None:
            print("[validate_onnx] --mode pipeline requires --config", file=sys.stderr)
            sys.exit(1)
        cfg = load_deployment_config(args.config)
        q_default_sim = ordered_array(cfg.control.q_default_sim, cfg.joints.sim_order)
        ok = run_pipeline_check(rows, policy, spec.obs_dim, spec.action_dim, spec.command_dim, q_default_sim, args.tol)

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
