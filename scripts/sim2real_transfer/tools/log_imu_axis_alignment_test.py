#!/usr/bin/env python3
"""Static-tilt calibration test to determine which raw IMU axis is the robot
body's FORWARD direction (and which is LEFT), for the *current* physical
mount -- see CLAUDE.md's sim2real IMU bring-up notes.

WHY THIS TEST EXISTS
--------------------
`config/deployment.yaml`'s `mount_offset_quat` was last calibrated as a
TILT-ONLY correction (a small, near-identity rotation computed from an at-rest
gravity average) -- it deliberately contains no yaw / 90-degree axis remap.
If the IMU board has since been physically reseated, the true horizontal
(forward/left) axis mapping could be off by ~90 degrees (or any arbitrary
yaw) from what the tilt-only correction assumes, which would rotate the
policy's projected-gravity and gyro x/y observations away from what it was
trained on -- the single biggest risk before closed-loop deployment. This
script measures that mapping directly and empirically, without relying on
any old CAD/mount assumption.

TEST PROCEDURE (torque OFF; a human holds/tips the robot by hand)
-------------------------------------------------------------------
  1. LEVEL       -- hold the robot level and stationary (~3s)
  2. NOSE DOWN   -- tip the front of the robot down ~20-30 deg about the
                    body's left-right axis and hold (~3s)
  3. LEVEL       -- return level and hold (~3s)
  4. NOSE UP     -- tip the front up ~20-30 deg and hold (~3s)
  5. LEVEL       -- return level and hold (~3s)
  6. ROLL LEFT   -- (recommended, not required) tip the robot's LEFT side
                    down ~20-30 deg and hold (~3s)
  7. LEVEL       -- return level and hold (~3s)

Run `log` mode ON THE ROBOT HOST (wherever `bno08x_driver`/the fusion node
publishes `/imu`) while the driver is already running in another terminal;
it prints timed prompts for each step above and logs everything to CSV.
Run `analyze` mode ANYWHERE with just numpy (no ROS2/rclpy needed) against
the resulting CSV.

SIGN CONVENTIONS (worked out from scratch, not assumed -- see CLAUDE.md and
the accompanying report for the full derivation)
-------------------------------------------------------------------------
Sim body frame: +X forward, +Y left, +Z up (right-handed: X x Y = Z).
`projected_gravity` reads [0, 0, -1] in body frame when level (gravity world
vector is [0, 0, -1]; `quat_apply_inverse(q, v) = R(q)^T @ v` rotates a
world-frame vector into the frame described by orientation quaternion q,
matching `sim2real/imu.py`).

Tipping the NOSE DOWN is a rotation about the body's +Y (left) axis. Working
through R_y(theta) analytically: gravity expressed in the body frame becomes
(sin(theta), 0, -cos(theta)) for nose-down pitch angle theta > 0. So gravity
in BODY frame gains a POSITIVE x (forward) component when the nose tips down
-- confirmed by direct rotation math, not just intuition.
Symmetrically, tipping the LEFT side down is a rotation about the body's +X
(forward) axis, and gravity in body frame gains a POSITIVE y (left)
component. General rule: tipping any body axis "down" (below the horizontal
plane) makes body-frame gravity gain a POSITIVE component along that axis.

Angular velocity (gyro) during the transitions is a second, independent
signal: nose pitching down is (by the right-hand rule) a POSITIVE rotation
about the body's +Y (left) axis; the left side going down is a NEGATIVE
rotation about the body's +X (forward) axis. Both facts are pure geometry,
independent of any parameterization choice, and are used below as a
cross-check against the gravity-based axis identification.

This script does NOT write to `config/deployment.yaml`. It only prints a
suggested `mount_offset_quat` for a human to sanity-check and copy in by
hand.
"""

from __future__ import annotations

import argparse
import csv
import math
import threading
import time

import numpy as np

# ---------------------------------------------------------------------------
# Quaternion / rotation-matrix helpers.
#
# Duplicated here (not imported from sim2real.imu) so this standalone tool
# has no import-time dependency on the rest of the sim2real package -- the
# same pattern log_imu_rotation_test.py uses (it reimplements its own small
# quat_to_yaw_deg helper rather than importing). This also means `analyze`
# mode has zero dependency on rclpy/ROS2 being installed.
#
# Convention (matches sim2real/imu.py exactly): q is [w, x, y, z];
# quat_to_rot_matrix(q) is the "frame -> world" rotation R, i.e. for a
# vector v_frame expressed in the frame the quaternion describes,
# v_world = R @ v_frame. quat_apply_inverse(q, v) = R^T @ v rotates a
# WORLD-frame vector into that local frame (used to get gravity_imu from
# the fused orientation quaternion + the fixed world gravity vector).
# ---------------------------------------------------------------------------

GRAVITY_VEC_WORLD = np.array([0.0, 0.0, -1.0])

# Current tilt-only mount_offset_quat (wxyz), as calibrated from an at-rest
# gravity average -- see module docstring. This is only a *default*; pass
# --tilt-quat to override with whatever is actually in the robot's own
# (gitignored) config/deployment.yaml.
DEFAULT_TILT_QUAT_WXYZ = (0.999929, -0.001400, 0.011801, 0.0)


def quat_to_rot_matrix(q_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = q_wxyz
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def quat_apply_inverse(q_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
    return quat_to_rot_matrix(q_wxyz).T @ v


def rot_matrix_to_quat(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> quaternion [w, x, y, z]. Standard branch-selecting
    (Shepperd) method; inverse of quat_to_rot_matrix above."""
    m00, m01, m02 = R[0]
    m10, m11, m12 = R[1]
    m20, m21, m22 = R[2]
    tr = m00 + m11 + m22

    if tr > 0:
        S = math.sqrt(tr + 1.0) * 2
        w = 0.25 * S
        x = (m21 - m12) / S
        y = (m02 - m20) / S
        z = (m10 - m01) / S
    elif m00 > m11 and m00 > m22:
        S = math.sqrt(1.0 + m00 - m11 - m22) * 2
        w = (m21 - m12) / S
        x = 0.25 * S
        y = (m01 + m10) / S
        z = (m02 + m20) / S
    elif m11 > m22:
        S = math.sqrt(1.0 + m11 - m00 - m22) * 2
        w = (m02 - m20) / S
        x = (m01 + m10) / S
        y = 0.25 * S
        z = (m12 + m21) / S
    else:
        S = math.sqrt(1.0 + m22 - m00 - m11) * 2
        w = (m10 - m01) / S
        x = (m02 + m20) / S
        y = (m12 + m21) / S
        z = 0.25 * S

    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def axis_label(v: np.ndarray) -> tuple[str, float]:
    """Nearest standard basis axis (+/-X/Y/Z) to unit vector v, and the
    cosine similarity to it (1.0 = exact match)."""
    names = ["X", "Y", "Z"]
    best_i = int(np.argmax(np.abs(v)))
    cos = float(v[best_i])
    best_name = ("+" if cos >= 0 else "-") + names[best_i]
    return best_name, abs(cos)


# ---------------------------------------------------------------------------
# Log mode -- run on the robot host (rclpy required). Import is deferred to
# this function (unlike log_imu_rotation_test.py's top-level import) so that
# `analyze` mode keeps working on machines with no ROS2 install at all.
# ---------------------------------------------------------------------------

# (name, hold_seconds, operator prompt)
DEFAULT_SCHEDULE = [
    ("LEVEL", 3.0, "LEVEL - hold the robot level and stationary"),
    ("NOSE_DOWN", 3.0, "TIP NOSE DOWN ~20-30 deg about the left-right axis - hold"),
    ("LEVEL", 3.0, "LEVEL - hold still"),
    ("NOSE_UP", 3.0, "TIP NOSE UP ~20-30 deg about the left-right axis - hold"),
    ("LEVEL", 3.0, "LEVEL - hold still"),
    ("ROLL_LEFT_DOWN", 3.0, "ROLL: tip the LEFT side DOWN ~20-30 deg - hold"),
    ("LEVEL", 3.0, "LEVEL - hold still - test complete"),
]

# Same, without the (optional) roll segment -- pitch-only fallback.
NO_ROLL_SCHEDULE = [seg for seg in DEFAULT_SCHEDULE if seg[0] != "ROLL_LEFT_DOWN"]


class _State:
    def __init__(self):
        self.count = 0
        self.gravity_imu = None
        self.lock = threading.Lock()

    def update(self, gravity_imu: np.ndarray) -> None:
        with self.lock:
            self.gravity_imu = gravity_imu
            self.count += 1

    def snapshot(self):
        with self.lock:
            if self.gravity_imu is None:
                return None
            return self.gravity_imu.copy(), self.count


def cmd_log(args: argparse.Namespace) -> None:
    try:
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import Imu as ImuMsg
    except ImportError as exc:
        raise ImportError(
            "rclpy and sensor_msgs are required for `log` mode (ROS2, on the robot host). "
            "Use `analyze` mode (plain numpy) to process an already-recorded CSV instead."
        ) from exc

    state = _State()
    out_file = open(args.out, "w", newline="")
    writer = csv.writer(out_file)
    writer.writerow(["t_wall", "qw", "qx", "qy", "qz", "gyro_x", "gyro_y", "gyro_z", "acc_x", "acc_y", "acc_z"])

    class _Node(Node):
        def __init__(self_node):
            super().__init__("imu_axis_alignment_test_logger")
            self_node.subscription = self_node.create_subscription(ImuMsg, args.topic, self_node_on_imu, 50)

    def self_node_on_imu(msg) -> None:
        t = time.time()
        o = msg.orientation
        av = msg.angular_velocity
        la = msg.linear_acceleration
        writer.writerow([f"{t:.6f}", o.w, o.x, o.y, o.z, av.x, av.y, av.z, la.x, la.y, la.z])
        q = np.array([o.w, o.x, o.y, o.z])
        state.update(quat_apply_inverse(q, GRAVITY_VEC_WORLD))

    rclpy.init()
    node = _Node()
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    schedule = NO_ROLL_SCHEDULE if args.skip_roll else DEFAULT_SCHEDULE
    print(f"Logging '{args.topic}' -> {args.out}. Torque should be OFF; a human holds/tips the robot by hand.")
    print(f"{len(schedule)} segments, ~{sum(s[1] for s in schedule):.0f}s of holds "
          f"(plus however long you take to move between them -- no rush).\n")

    try:
        for name, hold_s, prompt in schedule:
            print(f"=== {name} ===\n{prompt}")
            seg_start = time.time()
            while time.time() - seg_start < hold_s:
                time.sleep(0.1)
                snap = state.snapshot()
                remain = hold_s - (time.time() - seg_start)
                if snap is None:
                    print("\r  [waiting for first IMU message...]", end="", flush=True)
                    continue
                g, count = snap
                print(
                    f"\r  hold {remain:4.1f}s more   gravity_imu=({g[0]:+.3f}, {g[1]:+.3f}, {g[2]:+.3f})"
                    f"   samples={count:6d}   ",
                    end="",
                    flush=True,
                )
            print()
        print("\nSchedule complete. Ctrl+C was not needed -- stopping now.")
    except KeyboardInterrupt:
        print("\nStopped early by user request (already-logged data is kept).")

    print(f"Wrote {state.count} samples to {args.out}.")

    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
    spin_thread.join(timeout=3.0)
    out_file.close()


# ---------------------------------------------------------------------------
# Analyze mode -- plain numpy, no rclpy/ROS2 needed.
# ---------------------------------------------------------------------------


def load_csv(path: str) -> dict[str, np.ndarray]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path} has no data rows")
    cols = {}
    for key in ["t_wall", "qw", "qx", "qy", "qz", "gyro_x", "gyro_y", "gyro_z", "acc_x", "acc_y", "acc_z"]:
        cols[key] = np.array([float(r[key]) for r in rows], dtype=np.float64)
    return cols


def find_stationary_segments(t: np.ndarray, gyro_mag: np.ndarray, gyro_thresh: float, min_duration: float):
    """Contiguous runs where gyro magnitude stays below gyro_thresh for at
    least min_duration seconds. Returns list of (i, j) half-open index
    ranges. Auto-detection (rather than trusting exact prompt timing) is
    more robust to human reaction lag / network delay than fixed offsets."""
    stationary = gyro_mag < gyro_thresh
    segments = []
    n = len(t)
    i = 0
    while i < n:
        if stationary[i]:
            j = i
            while j < n and stationary[j]:
                j += 1
            if t[j - 1] - t[i] >= min_duration:
                segments.append((i, j))
            i = j
        else:
            i += 1
    return segments


def segment_core_mean(t: np.ndarray, gravity_imu: np.ndarray, i: int, j: int, trim_seconds: float) -> np.ndarray:
    """Mean gravity_imu over the core of a held segment, trimming
    trim_seconds off each end to avoid transition bleed-through. Falls back
    to the untrimmed segment if it's too short to trim."""
    t0, t1 = t[i], t[j - 1]
    mask = (t[i:j] >= t0 + trim_seconds) & (t[i:j] <= t1 - trim_seconds)
    core = gravity_imu[i:j][mask]
    if len(core) == 0:
        core = gravity_imu[i:j]
    return core.mean(axis=0)


def project_out(v: np.ndarray, axis: np.ndarray) -> np.ndarray:
    return v - np.dot(v, axis) * axis


def cmd_analyze(args: argparse.Namespace) -> None:
    cols = load_csv(args.csv)
    t = cols["t_wall"]
    q = np.stack([cols["qw"], cols["qx"], cols["qy"], cols["qz"]], axis=1)
    gyro = np.stack([cols["gyro_x"], cols["gyro_y"], cols["gyro_z"]], axis=1)

    gravity_imu = np.array([quat_apply_inverse(q[k], GRAVITY_VEC_WORLD) for k in range(len(t))])
    gyro_mag = np.linalg.norm(gyro, axis=1)

    segments = find_stationary_segments(t, gyro_mag, args.gyro_thresh, args.min_hold_duration)
    print(f"Loaded {len(t)} samples over {t[-1] - t[0]:.1f}s from {args.csv}")
    print(f"Auto-detected {len(segments)} stationary (held) segment(s) "
          f"(gyro_thresh={args.gyro_thresh} rad/s, min_hold_duration={args.min_hold_duration}s).\n")

    if len(segments) < 2:
        print("ERROR: need at least 2 held segments (a LEVEL baseline + at least one tip).")
        print("Try relaxing --gyro-thresh or --min-hold-duration, or check the CSV/segment schedule.")
        return

    means = [segment_core_mean(t, gravity_imu, i, j, args.trim_seconds) for i, j in segments]

    print("Detected segments (chronological):")
    for k, ((i, j), m) in enumerate(zip(segments, means)):
        print(f"  [{k}] t=({t[i]:.1f}-{t[j - 1]:.1f})s dur={t[j - 1] - t[i]:.1f}s "
              f"gravity_imu_mean=({m[0]:+.4f}, {m[1]:+.4f}, {m[2]:+.4f})")
    print()

    # --- classify LEVEL vs TIPPED, refine baseline over all LEVEL-ish segs ---
    baseline0 = means[0]
    level_thresh = args.level_thresh
    level_idxs = [k for k, m in enumerate(means) if np.linalg.norm(m - baseline0) < level_thresh]
    tipped_idxs = [k for k in range(len(means)) if k not in level_idxs]
    baseline = np.mean([means[k] for k in level_idxs], axis=0)
    baseline /= np.linalg.norm(baseline)  # renormalize (measured gravity should be unit length)

    up_in_imu = -baseline
    up_in_imu /= np.linalg.norm(up_in_imu)
    print(f"LEVEL segments: {level_idxs} (baseline gravity_imu = "
          f"({baseline[0]:+.4f}, {baseline[1]:+.4f}, {baseline[2]:+.4f}))")
    print(f"up_in_imu (from -baseline) = ({up_in_imu[0]:+.4f}, {up_in_imu[1]:+.4f}, {up_in_imu[2]:+.4f}) "
          f"[nearest raw axis: {axis_label(up_in_imu)[0]}, cos={axis_label(up_in_imu)[1]:.4f}]")
    print(f"TIPPED segments (by detection order): {tipped_idxs}\n")

    if not tipped_idxs:
        print("ERROR: no tipped segments detected -- only LEVEL holds found. Re-run the procedure.")
        return

    # Positional hypothesis: 1st tip = NOSE_DOWN, 2nd = NOSE_UP, 3rd = ROLL_LEFT_DOWN
    # (matches DEFAULT_SCHEDULE order). Verified below rather than blindly trusted.
    deltas = [means[k] - baseline for k in tipped_idxs]

    nose_down_delta = deltas[0] if len(deltas) >= 1 else None
    nose_up_delta = deltas[1] if len(deltas) >= 2 else None
    roll_delta = deltas[2] if len(deltas) >= 3 else None

    # --- forward axis, from nose-down/up (gravity gains +forward on nose-down,
    #     -forward on nose-up; averaging the pair cancels common-mode noise) ---
    if nose_down_delta is not None and nose_up_delta is not None:
        cos_opposite = float(
            np.dot(nose_down_delta, nose_up_delta)
            / (np.linalg.norm(nose_down_delta) * np.linalg.norm(nose_up_delta) + 1e-12)
        )
        print(f"Consistency check: nose-down delta vs nose-up delta cosine = {cos_opposite:+.3f} "
              f"(expect close to -1.0, i.e. opposite directions)")
        forward_raw = (nose_down_delta - nose_up_delta) / 2.0
    elif nose_down_delta is not None:
        print("Only a NOSE_DOWN tip detected (no NOSE_UP) -- forward axis estimate is less averaged.")
        forward_raw = nose_down_delta
    else:
        print("Only a NOSE_UP tip detected (no NOSE_DOWN) -- sign-flipping its delta to estimate forward.")
        forward_raw = -nose_up_delta

    forward_raw = project_out(forward_raw, up_in_imu)  # discard any up-axis leakage
    forward_in_imu = forward_raw / np.linalg.norm(forward_raw)
    fname, fcos = axis_label(forward_in_imu)
    print(f"forward_in_imu = ({forward_in_imu[0]:+.4f}, {forward_in_imu[1]:+.4f}, {forward_in_imu[2]:+.4f}) "
          f"[nearest raw axis: {fname}, cos={fcos:.4f}]\n")

    # --- left axis, from roll (gravity gains +left when left side dips down) ---
    if roll_delta is not None:
        left_raw = project_out(project_out(roll_delta, up_in_imu), forward_in_imu)
        left_in_imu = left_raw / np.linalg.norm(left_raw)
        left_measured = True
    else:
        print("No ROLL_LEFT_DOWN segment detected -- deriving left axis as up x forward "
              "(right-handed completion). Re-run with the roll segment for an independently measured cross-check.")
        left_in_imu = np.cross(up_in_imu, forward_in_imu)
        left_in_imu /= np.linalg.norm(left_in_imu)
        left_measured = False
    lname, lcos = axis_label(left_in_imu)
    tag = "(measured)" if left_measured else "(derived)"
    print(f"left_in_imu {tag} = ({left_in_imu[0]:+.4f}, {left_in_imu[1]:+.4f}, {left_in_imu[2]:+.4f}) "
          f"[nearest raw axis: {lname}, cos={lcos:.4f}]")

    # --- orthonormality / right-handedness check ---
    up_refined = np.cross(forward_in_imu, left_in_imu)
    up_refined /= np.linalg.norm(up_refined)
    up_angle_deg = math.degrees(math.acos(np.clip(np.dot(up_refined, up_in_imu), -1.0, 1.0)))
    print(f"\nOrthogonality/right-handedness check: forward.left={np.dot(forward_in_imu, left_in_imu):+.4f}, "
          f"angle between (forward x left) and measured up = {up_angle_deg:.2f} deg (expect ~0).")
    if up_angle_deg > 10.0:
        print("  WARNING: large mismatch -- check the roll segment was really a LEFT-down roll, "
              "not right-down, and that pitch/roll weren't mixed together.")

    # --- gyro cross-check on the tipping *transitions* (not the holds) ---
    print("\nGyro cross-check (transition windows between held segments):")

    def transition_gyro_mean(seg_a_end: int, seg_b_start: int) -> np.ndarray | None:
        if seg_b_start - seg_a_end < 3:
            return None
        return gyro[seg_a_end:seg_b_start].mean(axis=0)

    # transition into first tip (LEVEL -> NOSE_DOWN or NOSE_UP): expect rotation
    # about +left, sign = +1 (nose pitching down is a positive rotation about
    # +Y/left by the right-hand rule -- true for down; for an up-only single
    # segment the expected sign flips to -1).
    first_tip_k = tipped_idxs[0]
    prev_level_k = max([k for k in level_idxs if k < first_tip_k], default=None)
    if prev_level_k is not None:
        _, prev_end = segments[prev_level_k]
        tip_start, _ = segments[first_tip_k]
        g_trans = transition_gyro_mean(prev_end, tip_start)
        if g_trans is not None and np.linalg.norm(g_trans) > 1e-6:
            g_dir = g_trans / np.linalg.norm(g_trans)
            cos_left = float(np.dot(g_dir, left_in_imu))
            expected_sign = +1 if nose_down_delta is not None else -1
            print(f"  LEVEL->tip[0] transition mean gyro direction . left_in_imu = {cos_left:+.3f} "
                  f"(expect ~{expected_sign:+d}.0; nose-down is a +left-axis rotation, "
                  f"right-hand rule)")
            if abs(cos_left) < 0.6:
                print("    WARNING: weak alignment -- transition window may be too short/noisy.")
            elif (cos_left > 0) != (expected_sign > 0):
                print("    WARNING: sign mismatch vs gravity-based left axis -- investigate.")
        else:
            print("  LEVEL->tip[0] transition too short/noisy to use for gyro cross-check.")

    # transition into roll segment (LEVEL -> ROLL_LEFT_DOWN): expect rotation
    # about forward axis, sign = -1 (left-down is a NEGATIVE rotation about
    # +X/forward by the right-hand rule).
    if roll_delta is not None:
        roll_k = tipped_idxs[2]
        prev_level_k2 = max([k for k in level_idxs if k < roll_k], default=None)
        if prev_level_k2 is not None:
            _, prev_end2 = segments[prev_level_k2]
            roll_start, _ = segments[roll_k]
            g_trans2 = transition_gyro_mean(prev_end2, roll_start)
            if g_trans2 is not None and np.linalg.norm(g_trans2) > 1e-6:
                g_dir2 = g_trans2 / np.linalg.norm(g_trans2)
                cos_fwd = float(np.dot(g_dir2, forward_in_imu))
                print(f"  LEVEL->ROLL transition mean gyro direction . forward_in_imu = {cos_fwd:+.3f} "
                      f"(expect ~-1.0; left-down is a -forward-axis rotation, right-hand rule)")
                if abs(cos_fwd) < 0.6:
                    print("    WARNING: weak alignment -- transition window may be too short/noisy.")
                elif cos_fwd > 0:
                    print("    WARNING: sign mismatch vs gravity-based forward axis -- investigate.")
            else:
                print("  LEVEL->ROLL transition too short/noisy to use for gyro cross-check.")

    # --- build the suggested mount_offset_quat ---
    tilt_quat = np.array(args.tilt_quat, dtype=np.float64)
    tilt_quat /= np.linalg.norm(tilt_quat)
    R_tilt_old = quat_to_rot_matrix(tilt_quat)

    # Self-contained: build directly from raw-IMU-frame axes measured above.
    R_selfcontained = np.stack([forward_in_imu, left_in_imu, up_refined], axis=0)
    quat_selfcontained = rot_matrix_to_quat(R_selfcontained)

    # Composed: repeat the same axis-finding logic on the *tilt-corrected*
    # gravity (R_tilt_old @ gravity_imu), so the already-validated tilt
    # correction's precise "up" axis is preserved, and only the missing
    # horizontal remap is contributed by this test.
    gravity_tc = (R_tilt_old @ gravity_imu.T).T
    means_tc = [segment_core_mean(t, gravity_tc, i, j, args.trim_seconds) for i, j in segments]
    baseline_tc = np.mean([means_tc[k] for k in level_idxs], axis=0)
    baseline_tc /= np.linalg.norm(baseline_tc)
    up_tc = -baseline_tc
    up_tc /= np.linalg.norm(up_tc)
    deltas_tc = [means_tc[k] - baseline_tc for k in tipped_idxs]
    nd_tc = deltas_tc[0] if len(deltas_tc) >= 1 else None
    nu_tc = deltas_tc[1] if len(deltas_tc) >= 2 else None
    roll_tc = deltas_tc[2] if len(deltas_tc) >= 3 else None
    if nd_tc is not None and nu_tc is not None:
        fwd_raw_tc = (nd_tc - nu_tc) / 2.0
    elif nd_tc is not None:
        fwd_raw_tc = nd_tc
    else:
        fwd_raw_tc = -nu_tc
    fwd_raw_tc = project_out(fwd_raw_tc, up_tc)
    forward_tc = fwd_raw_tc / np.linalg.norm(fwd_raw_tc)
    if roll_tc is not None:
        left_raw_tc = project_out(project_out(roll_tc, up_tc), forward_tc)
        left_tc = left_raw_tc / np.linalg.norm(left_raw_tc)
    else:
        left_tc = np.cross(up_tc, forward_tc)
        left_tc /= np.linalg.norm(left_tc)
    up_refined_tc = np.cross(forward_tc, left_tc)
    up_refined_tc /= np.linalg.norm(up_refined_tc)

    R_remap = np.stack([forward_tc, left_tc, up_refined_tc], axis=0)  # tilt-corrected-frame -> body
    R_composed = R_remap @ R_tilt_old  # raw imu -> body
    quat_composed = rot_matrix_to_quat(R_composed)

    agree_dot = abs(float(np.dot(quat_selfcontained, quat_composed)))
    agree_deg = math.degrees(2 * math.acos(min(1.0, agree_dot)))

    print("\n" + "=" * 78)
    print("SUGGESTED mount_offset_quat (wxyz)")
    print("=" * 78)
    print(f"  composed (R_remap @ existing tilt-only R):  "
          f"[{quat_composed[0]:.6f}, {quat_composed[1]:.6f}, {quat_composed[2]:.6f}, {quat_composed[3]:.6f}]")
    print(f"  self-contained (from-scratch, no old tilt):  "
          f"[{quat_selfcontained[0]:.6f}, {quat_selfcontained[1]:.6f}, {quat_selfcontained[2]:.6f}, "
          f"{quat_selfcontained[3]:.6f}]")
    print(f"  angle between the two candidate quats: {agree_deg:.2f} deg "
          f"(should be small -- tilt correction is a near-identity nudge; large disagreement means "
          f"something is inconsistent, re-check the segments above)")
    print()
    print("Recommended value to consider: the COMPOSED one (reuses the existing, already-averaged")
    print("tilt calibration for the 'up' axis; this test only supplies the missing horizontal remap).")
    print()
    print("*** DO NOT paste this into config/deployment.yaml without a live sanity check first: ***")
    print("*** with the robot level, projected gravity (RosImuReader.read()) should read ~[0,0,-1];  ***")
    print("*** tipping the nose down should make it read ~[+sin(theta), 0, -cos(theta)].            ***")
    print("This script does not modify config/deployment.yaml -- a human must copy the value in by hand.")

    # --- caveat ---
    if len(tipped_idxs) < 3:
        print("\nCAVEAT: no independent roll measurement -- left axis above was DERIVED (up x forward), "
              "not measured. Re-run including the ROLL_LEFT_DOWN segment for a full cross-check.")
    print("CAVEAT: this method recovers an ARBITRARY (not just 90-degree) yaw offset correctly -- forward/left")
    print("are computed as continuous unit vectors (via atan2-equivalent projection), not snapped to the")
    print("nearest cardinal axis. The '+X/-Y/...' labels above are only a human-readable diagnostic;")
    print("the numeric quaternion is exact for whatever angle the physical remount actually introduced.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)

    p_log = sub.add_parser("log", help="Run ON THE ROBOT HOST: guide the operator through the tilt procedure "
                                        "and log raw IMU messages to CSV. Requires rclpy.")
    p_log.add_argument("--topic", default="/imu", help="IMU topic to subscribe to (default: /imu)")
    p_log.add_argument("--out", default="imu_axis_alignment_log.csv", help="output CSV path")
    p_log.add_argument("--skip-roll", action="store_true",
                        help="omit the (optional) ROLL_LEFT_DOWN segment -- pitch-only test")
    p_log.set_defaults(func=cmd_log)

    p_an = sub.add_parser("analyze", help="Run ANYWHERE (plain numpy): analyze a CSV from `log` mode.")
    p_an.add_argument("--csv", required=True, help="path to the CSV produced by `log` mode")
    p_an.add_argument("--tilt-quat", type=float, nargs=4, default=list(DEFAULT_TILT_QUAT_WXYZ),
                       metavar=("W", "X", "Y", "Z"),
                       help="existing tilt-only mount_offset_quat (wxyz) to compose with (default: the value "
                            "documented in this script's docstring -- pass the real one from your "
                            "config/deployment.yaml if different)")
    p_an.add_argument("--gyro-thresh", type=float, default=0.15,
                       help="rad/s below which a sample counts as 'stationary' (default: 0.15)")
    p_an.add_argument("--min-hold-duration", type=float, default=1.5,
                       help="minimum seconds a stationary run must last to count as a held segment (default: 1.5)")
    p_an.add_argument("--trim-seconds", type=float, default=0.4,
                       help="seconds trimmed off each end of a held segment before averaging (default: 0.4)")
    p_an.add_argument("--level-thresh", type=float, default=0.12,
                       help="gravity_imu delta-from-baseline norm below which a segment counts as "
                            "'level' rather than 'tipped' (default: 0.12)")
    p_an.set_defaults(func=cmd_analyze)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
