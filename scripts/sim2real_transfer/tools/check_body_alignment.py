#!/usr/bin/env python3
"""Read-only sanity check for the spine (body) joints -- BackLink/FrontLink --
against a real reference gait trajectory, so a sign error like the one found
in the leg calibration (2026-07-17: correction_group was wrong, not just the
zero_tick offset) doesn't go unnoticed for the body joints too.

WHY THIS IS DIFFERENT FROM THE LEG CALIBRATION TOOLS
------------------------------------------------------
For legs, "parallel to ground" and "perpendicular to ground" are unambiguous
physical descriptions with a known sim_rad value (0 = perpendicular, per the
operator's own domain knowledge), so the scaling/sign could be solved exactly.

For BackLink/FrontLink there is no equivalent physically-unambiguous
"positive direction" documented anywhere in this repo -- this script can NOT
tell you on its own which way is "correct". What it CAN do:
  1. Measure the physically-achievable range of motion for each spine joint
     (by hand, torque off) and compare its MAGNITUDE against a real reference
     gait trajectory already in this repo (hexapod-assets/Sim Gaits/*.csv,
     used for imitation learning -- genuine ground truth for what the trained
     policy actually commands, unlike a guessed value).
  2. Ask you to label, in your own words, which physical direction each
     measured extreme corresponds to (e.g. "front tip curls up"), and print
     that alongside the measured sim_rad sign and the reference's sign at the
     same point in its phase cycle -- so YOU can cross-check it against
     something you've actually seen the trained policy do (a pyvista render,
     a training video, or your own mental model), rather than this script
     asserting an answer it has no way to actually know.

Run ON THE ROBOT HOST. TORQUE IS FORCED OFF the moment the bus opens, and
this script never writes a goal position -- read-only, same as
calibrate_encoders.py. Does not touch config/deployment.yaml.

USAGE
-----
  # --reference is required: this package is deployed standalone (no
  # hexapod-assets/ alongside it), so copy the CSV onto the robot host first, e.g.:
  #   scp "hexapod-assets/Sim Gaits/forward3_lleg30_amp65_sim.csv" \\
  #       hexapi@<host>:~/sim2real_transfer/reference_gaits/
  python3 check_body_alignment.py --config ../config/deployment.yaml \\
      --reference ../reference_gaits/forward3_lleg30_amp65_sim.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim2real.deployment_config import DeploymentConfig, load_deployment_config  # noqa: E402
from sim2real.dynamixel_bus import RealDynamixelBus  # noqa: E402
from sim2real.joint_mapping import JointMapping  # noqa: E402

# Sim DOF order columns in hexapod-assets/Sim Gaits/*.csv, per CLAUDE.md's
# MotionReference joint ordering: 0 BackLink, 1 FrontLink, 2 MiddleLeft,
# 3 MiddleRight, 4 BackLeft, 5 BackRight, 6 FrontLeft, 7 FrontRight.
#
# No default path here on purpose: this package is deployed to the robot host
# as a flat standalone copy (no hexapod-assets/ alongside it, confirmed on the
# Pi 2026-07-17), unlike the main IsaacLab checkout where the two live side by
# side -- a path guessed from this script's own location would silently be
# wrong there. Copy the specific CSV you want to compare against onto the
# robot host and pass it explicitly via --reference.
_GAIT_CSV_COLUMNS = ["BackLink", "FrontLink"]
_BODY_JOINTS = ["BackLink", "FrontLink"]


def _load_reference_range(path: str) -> dict[str, tuple[float, float]]:
    with open(path, newline="") as f:
        rows = [[float(x) for x in r] for r in csv.reader(f) if r]
    cols = np.array(rows)  # (n_rows, 8), sim DOF order
    return {name: (float(cols[:, i].min()), float(cols[:, i].max())) for i, name in enumerate(_GAIT_CSV_COLUMNS)}


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
    rows = []
    for _ in range(samples):
        rows.append(bus.read_positions())
        time.sleep(interval)
    arr = np.stack(rows, axis=0).astype(np.float64)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    for name, m, s in zip(real_order, mean, std):
        flag = "  <-- WARNING: high spread, hold steadier" if s > 3.0 else ""
        print(f"    {name:12s} mean={m:8.2f} ticks  std={s:5.2f}{flag}")
    return mean


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="path to deployment.yaml")
    parser.add_argument("--reference", required=True,
                         help="path to a reference gait CSV, sim DOF order (e.g. a copy of "
                              "hexapod-assets/Sim Gaits/forward3_lleg30_amp65_sim.csv pushed onto this host)")
    parser.add_argument("--samples", type=int, default=15, help="reads to average per capture (default: 15)")
    parser.add_argument("--interval", type=float, default=0.05, help="seconds between reads (default: 0.05)")
    args = parser.parse_args()

    cfg = load_deployment_config(args.config)
    mapping = JointMapping(cfg.joints)
    ref_range = _load_reference_range(args.reference)

    print("=" * 78)
    print("BODY (SPINE) JOINT ALIGNMENT CHECK")
    print("=" * 78)
    print(f"Reference gait: {args.reference}")
    for name in _BODY_JOINTS:
        lo, hi = ref_range[name]
        print(f"    {name:12s} reference sim_rad range: [{lo:+.4f}, {hi:+.4f}]  (amplitude {(hi - lo) / 2:.4f} rad)")
    print("\nThis script measures YOUR achievable range by hand and reports it next to the reference")
    print("range above, for magnitude comparison -- it does NOT know which physical direction is")
    print("'positive' and will ask you to label each extreme yourself so you can judge alignment.\n")

    bus = _open_bus(cfg)
    results: dict[str, list[tuple[str, float]]] = {name: [] for name in _BODY_JOINTS}
    try:
        for name in _BODY_JOINTS:
            print(f"--- {name} ---")
            for extreme_num in (1, 2):
                input(f"Move {name} to physical extreme #{extreme_num} (no forcing past a hard stop), "
                      f"then press Enter...")
                label = input(f"  In a few words, describe this extreme's physical direction "
                               f"(e.g. 'front tip curls up'): ").strip() or f"extreme #{extreme_num} (unlabeled)"
                ticks = _read_settled(bus, cfg.joints.real_order, args.samples, args.interval)
                sim_rad = mapping.ticks_to_sim_rad(ticks)[cfg.joints.sim_order.index(name)]
                results[name].append((label, sim_rad))
                print(f"    -> measured sim_rad = {sim_rad:+.4f}\n")
    finally:
        bus.close()

    print("\n" + "=" * 78)
    print("SUMMARY -- compare against what you've seen the trained policy actually do")
    print("=" * 78)
    for name in _BODY_JOINTS:
        ref_lo, ref_hi = ref_range[name]
        print(f"\n{name}:")
        print(f"    reference gait sim_rad range: [{ref_lo:+.4f}, {ref_hi:+.4f}]")
        for label, sim_rad in results[name]:
            print(f"    measured '{label}': sim_rad = {sim_rad:+.4f}")
        measured_vals = [v for _, v in results[name]]
        measured_amp = (max(measured_vals) - min(measured_vals)) / 2.0
        ref_amp = (ref_hi - ref_lo) / 2.0
        amp_ratio = measured_amp / ref_amp if ref_amp > 1e-6 else float("nan")
        print(f"    measured amplitude {measured_amp:.4f} rad vs. reference amplitude {ref_amp:.4f} rad "
              f"(ratio {amp_ratio:.2f})")
        if amp_ratio < 0.5 or amp_ratio > 2.0:
            print("    <-- WARNING: measured range is very different in magnitude from the reference gait --")
            print("        check the joint isn't hitting a mechanical limit before the intended extreme.")
    print("\nSign/direction alignment: match each 'measured' label above against what you've seen this")
    print("joint do in a trained rollout (e.g. via render_pyvista.py or a training video) -- if the")
    print("physical direction you labeled as positive-going doesn't match the sim's positive direction,")
    print("this joint's correction_group sign in deployment.yaml is likely wrong, the same class of bug")
    print("found in the leg calibration on 2026-07-17.")
    print("\nThis script did NOT modify deployment.yaml.")


if __name__ == "__main__":
    main()
