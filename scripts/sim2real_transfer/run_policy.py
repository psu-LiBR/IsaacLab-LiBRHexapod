#!/usr/bin/env python3
"""CLI entry point: runs a trained hexapod policy on the real robot (or a
hardware-free dry run) via the sim2real package.

Swapping policies: point --policy at a different exported .onnx file (of the
same --profile); PolicyRunner shape-checks it against the profile at load, so a
mismatched pairing fails immediately instead of producing garbage actions.

Bring-up stages (see CLAUDE.md's sim2real plan for the full staged checklist):
  1-2. unit tests / tools/validate_onnx.py -- no hardware involved at all.
  3. --dry-run --fake-imu                    (no hardware, verifies loop/logging/timing)
  4. --no-torque                             (live IMU + live encoders, servos never move)
  5. --action-scale-mult 0.2 (or similar)    (first real motion, propped robot)
  6+. full scale, then ground contact
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sim2real import control_loop  # noqa: E402
from sim2real.command_source import make_command_source  # noqa: E402
from sim2real.deployment_config import load_deployment_config  # noqa: E402
from sim2real.dynamixel_bus import DryRunDynamixelBus, RealDynamixelBus  # noqa: E402
from sim2real.imu import FakeImu, RosImuReader  # noqa: E402
from sim2real.joint_mapping import JointMapping  # noqa: E402
from sim2real.localization import DeadReckoningLocalizer  # noqa: E402
from sim2real.logging_utils import CsvRunLogger  # noqa: E402
from sim2real.policy_runner import PolicyRunner  # noqa: E402
from sim2real.profiles import PROFILES, make_obs_builder  # noqa: E402

_DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "deployment.example.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy", required=True, help="Path to an exported policy.onnx checkpoint.")
    parser.add_argument("--profile", required=True, choices=sorted(PROFILES), help="Obs schema to use.")
    parser.add_argument("--config", default=_DEFAULT_CONFIG, help="Path to deployment.yaml.")
    parser.add_argument("--rate-hz", type=float, default=None, help="Overrides control.rate_hz from config.")
    parser.add_argument("--log-csv", default=None, help="Optional path to write a per-step CSV run log.")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Use an in-memory DryRunDynamixelBus -- no serial port, no motion.",
    )
    parser.add_argument(
        "--fake-imu", action="store_true",
        help="Use a stationary FakeImu instead of subscribing to the real /imu topic (no ROS2 needed).",
    )
    parser.add_argument(
        "--no-torque", action="store_true",
        help="Read real sensors and compute real targets, but never enable servo torque "
        "(bring-up stage 4: zero physical risk, verifies sensor/target sanity on live hardware).",
    )
    parser.add_argument("--duration", type=float, default=None, help="Auto-stop after this many seconds.")
    parser.add_argument(
        "--action-scale-mult", type=float, default=None,
        help="Overrides control.action_scale_multiplier from config -- <1.0 dampens motion for bring-up.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.config == _DEFAULT_CONFIG:
        print(f"[run_policy] --config not given, using bundled example config: {_DEFAULT_CONFIG}")
        print("[run_policy] this contains '# CALIBRATE' placeholders -- do not trust it on real hardware.")

    cfg = load_deployment_config(args.config)
    joint_mapping = JointMapping(cfg.joints)
    profile = PROFILES[args.profile]

    policy = PolicyRunner(args.policy, profile)
    obs_builder = make_obs_builder(args.profile)
    command_source = make_command_source(args.profile, cfg.commands)
    localizer = DeadReckoningLocalizer() if args.profile == "goal" else None

    if args.dry_run:
        bus = DryRunDynamixelBus(joint_mapping.motor_ids)
    else:
        bus = RealDynamixelBus(
            joint_mapping.motor_ids, cfg.serial.port, cfg.serial.baud_rate, cfg.serial.protocol_version
        )

    imu = (
        FakeImu()
        if args.fake_imu
        else RosImuReader(
            cfg.imu.topic, cfg.imu.mount_offset_quat, cfg.imu.negate_gyro_z, max_staleness_s=cfg.imu.max_staleness_s
        )
    )

    logger = CsvRunLogger(args.log_csv, obs_dim=profile.obs_dim, action_dim=profile.action_dim) if args.log_csv else None

    try:
        control_loop.run(
            cfg, args.profile, policy, obs_builder, joint_mapping, bus, imu, command_source,
            localizer=localizer,
            logger=logger,
            dry_run=args.dry_run,
            enable_torque=not args.no_torque,
            rate_hz=args.rate_hz,
            action_scale_multiplier=args.action_scale_mult,
            duration_s=args.duration,
        )
    finally:
        bus.close()
        if logger is not None:
            logger.close()


if __name__ == "__main__":
    main()
