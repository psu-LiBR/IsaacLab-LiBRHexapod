#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Interactive, single-leg stance/lift direction check for the `binary` profile.

Answers one narrow question in isolation from the RL policy, the spine wave, and
dead reckoning: does `binary.stance_pos` (bit +1, "foot down") actually put the
physical foot on the ground, and does `binary.lift_pos` (bit -1) actually lift it?
This moves ONE leg at a time -- everything else holds at `control.q_default_sim` --
so there is no gait, no coordination, and nothing to interpret: the leg you're
watching should be doing exactly one thing.

Unlike the other tools/ scripts, THIS ONE MOVES THE ROBOT (torque is enabled, not
forced off) -- prop the robot with the tested leg free to swing before running it.
Every other joint is ramped to and held at its normal standing pose throughout.

Run ON THE ROBOT HOST, robot propped with all legs clear of the ground/obstacles.

USAGE
-----
  # One leg, a couple of stance/lift cycles:
  python3 check_leg_stance_lift.py --config ../config/deployment.binary.yaml --leg FrontRight

  # Cycle through all six legs in turn (confirms per-leg, not just the one you picked):
  python3 check_leg_stance_lift.py --config ../config/deployment.binary.yaml --all
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim2real.deployment_config import DeploymentConfig, load_deployment_config  # noqa: E402
from sim2real.dynamixel_bus import RealDynamixelBus  # noqa: E402
from sim2real.joint_mapping import JointMapping, ordered_array  # noqa: E402
from sim2real.safety import ramp_to_target  # noqa: E402


def _open_bus(cfg: DeploymentConfig) -> RealDynamixelBus:
    motor_ids = [cfg.joints.motor_ids[name] for name in cfg.joints.real_order]
    bus = RealDynamixelBus(
        motor_ids=motor_ids, port=cfg.serial.port, baud_rate=cfg.serial.baud_rate, protocol_version=cfg.serial.protocol_version
    )
    bus.torque_enable(False)  # stays off until the operator explicitly confirms below
    print(f"Connected to {cfg.serial.port} @ {cfg.serial.baud_rate} baud, {len(motor_ids)} motors.\n")
    return bus


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="path to deployment.binary.yaml")
    parser.add_argument("--leg", default=None, help="one leg joint name (e.g. FrontRight); required unless --all")
    parser.add_argument("--all", action="store_true", help="cycle through all six legs in turn")
    parser.add_argument("--cycles", type=int, default=2, help="stance/lift cycles per leg (default: 2)")
    parser.add_argument("--ramp-seconds", type=float, default=1.0, help="ramp duration per transition (default: 1.0)")
    parser.add_argument("--hold-seconds", type=float, default=2.0, help="pause after each ramp (default: 2.0)")
    parser.add_argument("--rate-hz", type=float, default=50.0, help="ramp write rate (default: 50.0)")
    args = parser.parse_args()

    if bool(args.leg) == bool(args.all):
        raise SystemExit("pass exactly one of --leg <name> or --all")

    cfg = load_deployment_config(args.config)
    if cfg.binary is None:
        raise SystemExit(f"--config must point at a config with a 'binary:' section, got {args.config}")
    mapping = JointMapping(cfg.joints)
    q_default_sim = ordered_array(cfg.control.q_default_sim, cfg.joints.sim_order)

    legs = list(cfg.binary.leg_joint_order) if args.all else [args.leg]
    for leg in legs:
        if leg not in cfg.joints.sim_order:
            raise SystemExit(f"unknown leg joint '{leg}' -- expected one of {cfg.binary.leg_joint_order}")
    leg_idx = {leg: cfg.joints.sim_order.index(leg) for leg in legs}

    print("=" * 78)
    print("LEG STANCE/LIFT DIRECTION CHECK")
    print("=" * 78)
    print(f"stance_pos (bit +1) = {cfg.binary.stance_pos:+.4f} rad  -- should be the ground-contact position")
    print(f"lift_pos   (bit -1) = {cfg.binary.lift_pos:+.4f} rad  -- should be the raised/swing position")
    print(f"Legs to test, in order: {legs}\n")
    print("TORQUE WILL BE ENABLED. All other joints hold at their normal standing pose throughout;")
    print("only the one leg under test moves. Make sure the robot is propped and that leg is clear.\n")

    bus = _open_bus(cfg)
    labels: dict[str, list[tuple[str, str]]] = {leg: [] for leg in legs}
    try:
        input("Press Enter once the robot is propped and ready (Ctrl+C to abort)...")
        print("\nRamping to standing pose...")
        ramp_to_target(bus, mapping, q_default_sim, args.ramp_seconds, args.rate_hz)
        bus.torque_enable(True)

        for leg in legs:
            print(f"\n--- {leg} ---")
            idx = leg_idx[leg]
            for cycle in range(1, args.cycles + 1):
                for phase_name, target_rad in (("STANCE (bit +1)", cfg.binary.stance_pos), ("LIFT (bit -1)", cfg.binary.lift_pos)):
                    target_sim = q_default_sim.copy()
                    target_sim[idx] = target_rad
                    ramp_to_target(bus, mapping, target_sim, args.ramp_seconds, args.rate_hz)
                    time.sleep(args.hold_seconds)
                    label = input(
                        f"  [{leg} cycle {cycle}/{args.cycles}] commanded {phase_name} ({target_rad:+.4f} rad) -- "
                        "what did the foot physically do? (e.g. 'touched ground' / 'lifted up'): "
                    ).strip() or "(no label given)"
                    labels[leg].append((phase_name, label))

            print(f"  Returning {leg} to standing pose...")
            ramp_to_target(bus, mapping, q_default_sim, args.ramp_seconds, args.rate_hz)
    finally:
        print("\nRamping to standing pose and disabling torque...")
        ramp_to_target(bus, mapping, q_default_sim, args.ramp_seconds, args.rate_hz)
        bus.torque_enable(False)
        bus.close()

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    for leg in legs:
        print(f"\n{leg}:")
        for phase_name, label in labels[leg]:
            print(f"    {phase_name:16s} -> observed: {label}")
    print(
        "\nIf 'STANCE (bit +1)' was observed lifting the foot (or vice versa) for any leg, that leg's "
        "effective stance/lift meaning is inverted on hardware -- independent of the spine wave and "
        "dead reckoning. The direct fix is swapping binary.stance_pos / binary.lift_pos in "
        "deployment.binary.yaml (do not touch joints.correction_group; that's the calibrated position "
        "mapping, not the stance/lift semantic)."
    )
    print("\nThis script did NOT modify deployment.binary.yaml.")


if __name__ == "__main__":
    main()
