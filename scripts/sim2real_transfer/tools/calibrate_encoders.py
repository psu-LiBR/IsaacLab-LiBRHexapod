#!/usr/bin/env python3
"""Interactive, read-only encoder calibration helper for the leg joints'
`zero_tick` and `soft_limits_rad` in config/deployment.yaml -- see CLAUDE.md's
sim2real bring-up notes and the "CALIBRATE" comments in deployment.example.yaml.

TORQUE IS FORCED OFF on every motor the moment the bus opens, and this script
never writes a goal position -- it only ever calls `read_positions()`. You
move the physical robot by hand; the script just reads encoder ticks and does
the tick<->radian math. It never touches config/deployment.yaml itself -- it
prints suggested values for a human to copy in by hand (same pattern as
log_imu_rotation_test.py / log_imu_axis_alignment_test.py).

WHY TWO SUBCOMMANDS, IN THIS ORDER
-----------------------------------
`zero-tick` must be run (and its results pasted into deployment.yaml) BEFORE
`rom`, because `rom` converts ticks -> sim radians via the full
sim2real.joint_mapping.JointMapping pipeline, which needs a correct zero_tick
to mean anything. `zero-tick` itself does NOT depend on zero_tick (that's the
value being solved for) -- it only needs the known q_default_sim rest pose and
each joint's correction_group.

Run ON THE ROBOT HOST, in the sim2real_transfer environment (needs
dynamixel_sdk, i.e. `pip install -r requirements.txt`).

USAGE
-----
  # 1) Physically pose the WHOLE robot at its q_default_sim rest pose
  #    (this is the robot's normal standing pose -- spine joints ~0 rad, all
  #    six leg joints ~-0.47 rad) and hold/prop it there.
  python3 calibrate_encoders.py --config ../config/deployment.yaml zero-tick

  # -> paste the printed zero_tick values into deployment.yaml, THEN:

  # 2) One leg joint at a time, move it (by hand, torque off) to its low and
  #    then high safe mechanical extreme when prompted.
  python3 calibrate_encoders.py --config ../config/deployment.yaml rom

  # Anytime after zero_tick is calibrated: read-only live sanity check --
  # move the robot by hand and watch each joint's sim_rad update, Ctrl+C to stop.
  python3 calibrate_encoders.py --config ../config/deployment.yaml read

  # ALTERNATIVE to `zero-tick`: a 3-pose fit (parallel/rest/perpendicular, all
  # six legs moved together each time) that also checks whether ticks_per_rad
  # actually matches the theoretical constant per leg, not just zero_tick.
  python3 calibrate_encoders.py --config ../config/deployment.yaml scale
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim2real.deployment_config import DeploymentConfig, load_deployment_config  # noqa: E402
from sim2real.dynamixel_bus import RealDynamixelBus  # noqa: E402
from sim2real.joint_mapping import JointMapping  # noqa: E402

# real = a * sim + b (radians). Kept in sync with joint_mapping.py's _CORRECTIONS
# table -- duplicated here (not imported) because zero-tick calibration needs
# just the correction formula, not a fully-formed JointMapping (which requires
# zero_tick, the very thing being solved for).
_CORRECTIONS: dict[str, tuple[float, float]] = {
    "unchanged": (1.0, 0.0),
    "negate": (-1.0, 0.0),
    "leg_negate_plus_pi": (-1.0, math.pi),
    "leg_negate_minus_pi": (-1.0, -math.pi),
}


def _open_bus(cfg: DeploymentConfig) -> RealDynamixelBus:
    motor_ids = [cfg.joints.motor_ids[name] for name in cfg.joints.real_order]
    bus = RealDynamixelBus(
        motor_ids=motor_ids,
        port=cfg.serial.port,
        baud_rate=cfg.serial.baud_rate,
        protocol_version=cfg.serial.protocol_version,
    )
    bus.torque_enable(False)  # safety: always the very first thing done with the bus
    print(f"Connected to {cfg.serial.port} @ {cfg.serial.baud_rate} baud, "
          f"{len(motor_ids)} motors, torque forced OFF.\n")
    return bus


def _read_settled(bus: RealDynamixelBus, real_order: list[str], samples: int, interval: float) -> np.ndarray:
    """Average `samples` reads (real DOF order), printing per-joint mean+std so
    the operator can see if the pose wasn't actually held still."""
    rows = []
    for _ in range(samples):
        rows.append(bus.read_positions())
        time.sleep(interval)
    arr = np.stack(rows, axis=0).astype(np.float64)  # (samples, n_joints)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    for name, m, s in zip(real_order, mean, std):
        flag = "  <-- WARNING: high spread, hold steadier" if s > 3.0 else ""
        print(f"    {name:12s} mean={m:8.2f} ticks  std={s:5.2f}{flag}")
    return mean


def cmd_zero_tick(args: argparse.Namespace) -> None:
    cfg = load_deployment_config(args.config)
    ticks_per_rad = cfg.joints.ticks_per_rev / (2.0 * math.pi)

    print("=" * 78)
    print("ZERO-TICK CALIBRATION")
    print("=" * 78)
    print("Physically pose the WHOLE robot at its q_default_sim rest pose:")
    for name in cfg.joints.sim_order:
        print(f"    {name:12s} = {cfg.control.q_default_sim[name]:+.3f} rad")
    print("(this is the robot's normal standing pose). Prop or hold it there, torque is off.\n")

    bus = _open_bus(cfg)
    try:
        input("Press Enter once the robot is settled in that pose (Ctrl+C to abort)...")
        print(f"\nReading {args.samples} samples over ~{args.samples * args.interval:.1f}s:")
        mean_ticks = _read_settled(bus, cfg.joints.real_order, args.samples, args.interval)
    finally:
        bus.close()

    print("\nSuggested zero_tick (paste into joints.encoder.zero_tick in deployment.yaml):")
    print("  zero_tick:")
    for i, name in enumerate(cfg.joints.real_order):
        sim_name = name  # real_order and sim_order are the same joint-name set
        group = cfg.joints.correction_group[name]
        a, b = _CORRECTIONS[group]
        expected_real_rad = a * cfg.control.q_default_sim[sim_name] + b
        suggested = round(mean_ticks[i] - expected_real_rad * ticks_per_rad)
        in_range = "" if 0 <= suggested <= 4095 else "  <-- OUT OF [0,4095] RANGE, something is wrong"
        print(f"    {name}: {suggested}{in_range}  "
              f"(measured {mean_ticks[i]:.1f} ticks, expected real_rad={expected_real_rad:+.4f})")
    print("\nSanity check: BackLink/FrontLink (body joints) should come out close to the already-")
    print("confirmed 2048 -- if they're far off, the robot likely wasn't actually at rest pose.")
    print("This script did NOT modify deployment.yaml -- copy the values in by hand, then run `rom`.")


def cmd_rom(args: argparse.Namespace) -> None:
    cfg = load_deployment_config(args.config)
    mapping = JointMapping(cfg.joints)

    print("=" * 78)
    print("RANGE-OF-MOTION / SOFT-LIMIT CALIBRATION")
    print("=" * 78)
    print("Requires zero_tick to already be calibrated and pasted into deployment.yaml --")
    print("this step converts ticks to sim radians via the full JointMapping pipeline.\n")
    print(f"Margin factor {args.margin}: each joint's raw measured range is shrunk by this factor")
    print("around its own midpoint before being suggested as the soft limit (matches CLAUDE.md's")
    print(f"soft_joint_pos_limit_factor convention). Override with --margin if you want the raw range.\n")

    bus = _open_bus(cfg)
    results: dict[str, tuple[float, float]] = {}
    try:
        for sim_name in cfg.joints.sim_order:
            print(f"--- {sim_name} ---")
            input(f"Move {sim_name} to its LOW-end safe physical extreme (no forcing past a hard "
                  f"stop), then press Enter...")
            lo_ticks = _read_settled(bus, cfg.joints.real_order, args.samples, args.interval)
            lo_sim_rad = mapping.ticks_to_sim_rad(lo_ticks)[cfg.joints.sim_order.index(sim_name)]

            input(f"Move {sim_name} to its HIGH-end safe physical extreme, then press Enter...")
            hi_ticks = _read_settled(bus, cfg.joints.real_order, args.samples, args.interval)
            hi_sim_rad = mapping.ticks_to_sim_rad(hi_ticks)[cfg.joints.sim_order.index(sim_name)]

            raw_lo, raw_hi = sorted([lo_sim_rad, hi_sim_rad])
            center = (raw_lo + raw_hi) / 2.0
            half = (raw_hi - raw_lo) / 2.0 * args.margin
            soft_lo, soft_hi = center - half, center + half
            results[sim_name] = (soft_lo, soft_hi)
            print(f"    raw measured range: [{raw_lo:+.4f}, {raw_hi:+.4f}] rad  ->  "
                  f"soft_limits_rad: [{soft_lo:+.4f}, {soft_hi:+.4f}] rad\n")
    finally:
        bus.close()

    print("\nSuggested soft_limits_rad (paste into joints.soft_limits_rad in deployment.yaml):")
    print("  soft_limits_rad:")
    for name in cfg.joints.sim_order:
        lo, hi = results[name]
        print(f"    {name}: [{lo:.4f}, {hi:.4f}]")
    print("\nThis script did NOT modify deployment.yaml -- copy the values in by hand.")


def cmd_read(args: argparse.Namespace) -> None:
    cfg = load_deployment_config(args.config)
    mapping = JointMapping(cfg.joints)

    print("=" * 78)
    print("LIVE TICKS -> SIM_RAD READER")
    print("=" * 78)
    print("Read-only sanity check: move the robot by hand (torque off) and watch each joint's")
    print("sim_rad update. At the normal rest pose, spine joints should read ~0.0 and all six")
    print("leg joints should read ~-0.47 -- if not, zero_tick is still wrong. Ctrl+C to stop.\n")

    bus = _open_bus(cfg)
    try:
        while True:
            ticks = bus.read_positions()
            sim_rad = mapping.ticks_to_sim_rad(ticks)
            line = "  ".join(
                f"{name}={sim_rad[i]:+.3f}" for i, name in enumerate(cfg.joints.sim_order)
            )
            print(f"\r{line}   ", end="", flush=True)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        bus.close()


def cmd_scale(args: argparse.Namespace) -> None:
    cfg = load_deployment_config(args.config)
    ticks_per_rad_theory = cfg.joints.ticks_per_rev / (2.0 * math.pi)

    leg_names = [
        name for name in cfg.joints.sim_order
        if cfg.joints.correction_group[name] in ("leg_negate_plus_pi", "leg_negate_minus_pi")
    ]
    if not leg_names:
        print("No leg joints found (correction_group leg_negate_plus_pi/leg_negate_minus_pi) -- nothing to do.")
        return

    print("=" * 78)
    print("THREE-POSE LEG SCALING CALIBRATION")
    print("=" * 78)
    print("Fits ticks = zero_tick + real_rad*ticks_per_rad per leg via least-squares over 3 poses,")
    print("instead of trusting a single measured pose + the theoretical ticks_per_rad constant. Also")
    print("reports how far each leg's FITTED ticks_per_rad is from that theoretical constant, as a")
    print("check on whether the shared, single global ticks_per_rev in deployment.yaml is even valid.\n")
    print("ASSUMED sim_rad targets for the two non-rest poses -- these are NOT a confirmed measurement,")
    print("just defaults sourced from deployment.example.yaml's placeholder soft_limits_rad. Override")
    print("with --parallel-sim-rad / --perpendicular-sim-rad if your CAD/USD limits say otherwise:")
    print(f"    PARALLEL_SPRAWLED       ~= {args.parallel_sim_rad:+.3f} rad")
    print("    REST_STANCE             =  each leg's own q_default_sim (normally -0.47 rad)")
    print(f"    PERPENDICULAR_VERTICAL  ~= {args.perpendicular_sim_rad:+.3f} rad\n")

    poses = [
        ("PARALLEL_SPRAWLED", "Move ALL SIX LEGS parallel to the ground (fully sprawled out, horizontal).",
         args.parallel_sim_rad),
        ("REST_STANCE", "Move ALL SIX LEGS to the normal rest/standing stance.", None),
        ("PERPENDICULAR_VERTICAL", "Move ALL SIX LEGS perpendicular to the ground (fully vertical).",
         args.perpendicular_sim_rad),
    ]

    bus = _open_bus(cfg)
    captures: list[tuple[str, dict[str, float], np.ndarray]] = []
    try:
        for pose_name, prompt, override in poses:
            print(f"=== {pose_name} ===\n{prompt}")
            input("  Get all six legs into position, then press Enter to start recording (no rush)...")
            print(f"  Reading {args.samples} samples over ~{args.samples * args.interval:.1f}s:")
            mean_ticks = _read_settled(bus, cfg.joints.real_order, args.samples, args.interval)
            sim_rad_per_leg = {
                name: (override if override is not None else cfg.control.q_default_sim[name])
                for name in leg_names
            }
            captures.append((pose_name, sim_rad_per_leg, mean_ticks))
            print()
    finally:
        bus.close()

    print("Per-leg least-squares fit across the 3 poses:\n")
    header = f"{'joint':12s} {'fit zero_tick':>14s} {'fit ticks_per_rad':>18s} {'theory':>10s} {'diff %':>8s} {'max resid (ticks)':>18s}"
    print(header)
    print("-" * len(header))

    fitted_zero_tick: dict[str, float] = {}
    for name in leg_names:
        real_idx = cfg.joints.real_order.index(name)
        a, b = _CORRECTIONS[cfg.joints.correction_group[name]]
        real_rads = np.array([a * sim_rad_per_leg[name] + b for _, sim_rad_per_leg, _ in captures])
        ticks_vals = np.array([mean_ticks[real_idx] for _, _, mean_ticks in captures])

        slope, intercept = np.polyfit(real_rads, ticks_vals, 1)
        predicted = intercept + slope * real_rads
        max_resid = float(np.max(np.abs(ticks_vals - predicted)))
        diff_pct = (slope - ticks_per_rad_theory) / ticks_per_rad_theory * 100.0
        fitted_zero_tick[name] = intercept

        flag = "  <-- check pose assumptions / hold steadier" if max_resid > 15.0 else ""
        print(f"{name:12s} {intercept:14.1f} {slope:18.2f} {ticks_per_rad_theory:10.2f} "
              f"{diff_pct:7.2f}% {max_resid:18.2f}{flag}")

    print("\nSuggested zero_tick (fit across all 3 poses -- paste into joints.encoder.zero_tick):")
    print("  zero_tick:")
    for name in leg_names:
        v = round(fitted_zero_tick[name])
        in_range = "" if 0 <= v <= 4095 else "  <-- OUT OF [0,4095] RANGE"
        print(f"    {name}: {v}{in_range}")

    print("\nIf 'diff %' is small (a percent or two) for every leg, the existing single global")
    print("ticks_per_rad (from joints.encoder.ticks_per_rev, shared by every joint) is fine -- just")
    print("update zero_tick above. If any leg's fit disagrees noticeably with the theoretical constant,")
    print("that leg's real scaling differs from the others, and this config's schema (one shared")
    print("ticks_per_rev for all joints) can't represent that yet -- don't silently trust that leg's")
    print("zero_tick number, flag it for a schema change instead.")
    print("\nThis script did NOT modify deployment.yaml -- copy values in by hand.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="path to deployment.yaml")
    parser.add_argument("--samples", type=int, default=15, help="reads to average per capture (default: 15)")
    parser.add_argument("--interval", type=float, default=0.05, help="seconds between reads (default: 0.05)")

    sub = parser.add_subparsers(dest="mode", required=True)

    p_zt = sub.add_parser("zero-tick", help="Calibrate joints.encoder.zero_tick from the q_default_sim rest pose.")
    p_zt.set_defaults(func=cmd_zero_tick)

    p_rom = sub.add_parser("rom", help="Calibrate joints.soft_limits_rad by sweeping each joint's physical range.")
    p_rom.add_argument("--margin", type=float, default=0.9,
                        help="shrink factor applied around each joint's measured-range midpoint (default: 0.9)")
    p_rom.set_defaults(func=cmd_rom)

    p_read = sub.add_parser("read", help="Live-stream ticks_to_sim_rad() for all joints -- read-only sanity "
                                          "check of an already-calibrated zero_tick.")
    p_read.set_defaults(func=cmd_read)

    p_scale = sub.add_parser("scale", help="Alternative to `zero-tick`: fit zero_tick (and check ticks_per_rad) "
                                            "per leg from 3 poses -- parallel/rest/perpendicular -- instead of "
                                            "just the rest pose.")
    p_scale.add_argument("--parallel-sim-rad", type=float, default=0.3,
                          help="assumed sim_rad for ALL legs at the 'parallel to ground / sprawled' pose "
                               "(default: 0.3, matching deployment.example.yaml's placeholder soft_limits_rad "
                               "upper bound -- override if your CAD/USD limits differ)")
    p_scale.add_argument("--perpendicular-sim-rad", type=float, default=-1.4,
                          help="assumed sim_rad for ALL legs at the 'perpendicular to ground / vertical' pose "
                               "(default: -1.4, matching deployment.example.yaml's placeholder soft_limits_rad "
                               "lower bound -- override if your CAD/USD limits differ)")
    p_scale.set_defaults(func=cmd_scale)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
