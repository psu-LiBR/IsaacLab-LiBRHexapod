#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Convert a real-robot (HexapI) open-loop gait CSV into a mimic-env reference CSV.

Input: headerless 8-column CSV (``#`` comment lines allowed), one gait cycle, columns
``body_1, body_2, FR, FL, MR, ML, HR, HL``. ``body_*`` are "raw/unnegated" spine angles [rad]; the six leg
columns are contact bits (1 = foot on ground / stance, 0 = in the air / lift).

Output: headerless 8-column CSV [rad], one row per input row, in the mimic joint order
(:data:`MIMIC_JOINT_ORDER`), consumable by ``hexapod_mimic_motion.MotionReference``.

Derivation (nothing here is re-derived independently; the real robot's own pipeline is replayed and then
inverted with the repo's existing joint map):

1. ``LiBR-hexapi/scripts/physics_gait_di.py::get_contact_Di`` turns the gait into servo ticks:
   spine ``tick = (-body - bodyampshift) * 4095/(2*pi) + 4095/2``; legs ``tick = 1747.5`` (contact = 1) or
   ``air_enc`` (default ``0.6 * 1747.5 = 1048.5``, contact = 0). Motor order is
   ``[FrontLink, BackLink, FrontRight, FrontLeft, MiddleRight, MiddleLeft, BackRight, BackLeft]``, so
   body_1 -> FrontLink, body_2 -> BackLink, and FR/FL/MR/ML/HR/HL -> FrontRight/FrontLeft/MiddleRight/MiddleLeft/
   BackRight/BackLeft. Optional row rotations (start phase, body-vs-contact shift) are reproduced.
2. ``sim2real.joint_mapping.JointMapping.ticks_to_sim_rad`` with ``config/deployment.binary.example.yaml`` (all
   eight joints use correction group ``negate``, ticks_per_rev 4096) maps ticks to sim radians in the HexapI USD
   convention.

Closed form in ``--zero_tick_mode nominal`` (every zero_tick = 2047.5, the physics_gait_di midpoint; ticks are left
unrounded so the constants below are exact): the tick encodes ``-body`` and both spine joints are ``negate``, so
sim FrontLink = +body_1 and sim BackLink = +body_2 (up to the 4095/4096 scale between the robot's spine formula and
the 4096 ticks/rev mapping, i.e. < 0.03 %). Legs: ``sim = (2047.5 - tick) / (4096 / (2*pi))``, i.e. stance
0.460194 rad (tick 1747.5), lift 1.53250 rad (air tick 1048.5) or 1.180398 rad (``--air_enc 1278``), matching
STANCE_POS / LIFT_POS in hexapod_binary_env_cfg.py. ``calibrated`` mode uses the yaml's measured zero_ticks instead.

Note: the older ``hexapod-assets/Sim Gaits/vertical_K3_rank1_lleg30-93_amp70_hexapi_sim.csv`` used the OPPOSITE
spine sign (sim = -body) and predates the 2026-10-20 spine-sign settlement; do not mix it with files from this tool.

Tick rounding: the real robot writes ``int(round(tick))``. By default this tool keeps ticks as floats (so the
nominal stance/lift constants are exact); ``--round_ticks`` applies half-to-even rounding (identical to Python's
built-in ``round`` and ``np.round``) to reproduce exactly what the servos receive.

The output has no time column (a time column would make MotionReference infer period = t[-1] - t[0], which is one
row short). The real cycle period is ``rows * step_period``; set ``MIMIC_GAIT_PERIOD`` in
hexapod_mimic_env_cfg.py to it to replay at real-robot speed.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import numpy as np

_SIM2REAL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SIM2REAL_DIR))

from sim2real.deployment_config import load_deployment_config  # noqa: E402
from sim2real.joint_mapping import JointMapping  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = _SIM2REAL_DIR / "config" / "deployment.binary.example.yaml"
DEFAULT_OUT_ROOT = REPO_ROOT / "hexapod-assets" / "Sim Gaits"

# Must equal hexapod_mimic_motion.JOINT_NAMES (guarded by tests/test_real_gait_to_sim.py).
MIMIC_JOINT_ORDER: list[str] = [
    "BackLink_Joint",
    "FrontLink_Joint",
    "MiddleLeft_Joint",
    "MiddleRight_Joint",
    "BackLeft_Joint",
    "BackRight_Joint",
    "FrontLeft_Joint",
    "FrontRight_Joint",
]

# Real-gait CSV contact columns (FR, FL, MR, ML, HR, HL) -> joint names (HR/HL = hind = Back).
CONTACT_JOINTS: list[str] = ["FrontRight", "FrontLeft", "MiddleRight", "MiddleLeft", "BackRight", "BackLeft"]
# body_1 -> FrontLink (motor 3), body_2 -> BackLink (motor 6).
BODY_JOINTS: list[str] = ["FrontLink", "BackLink"]

ZERO_TICK_NOMINAL = 2047.5
LEG_GROUND_TICK = 4095.0 / 2.0 - 300.0
AIR_FRACTION = 0.6  # robot default: val0 = val1 * 0.6


def load_real_gait(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a real-robot gait CSV.

    Args:
        path: CSV path; ``#`` comment lines are skipped, no header row.

    Returns:
        ``(body, contact)`` with ``body`` of shape [N, 2] [rad] and ``contact`` of shape [N, 6] (0/1 ints).

    Raises:
        ValueError: If the file does not have 8 columns or the contact columns are not 0/1.
    """
    if Path(path).suffix.lower() == ".mat":
        return _load_mat_gait(path)
    data = np.loadtxt(path, delimiter=",", comments="#", ndmin=2)
    if data.ndim != 2 or data.shape[1] != 8 or data.shape[0] < 1:
        raise ValueError(f"{path}: expected 8 columns (body_1, body_2, FR, FL, MR, ML, HR, HL), got shape {data.shape}")
    raw = data[:, 2:8]
    rounded = np.round(raw)
    if not np.all(np.abs(raw - rounded) <= 1e-6) or not np.all(np.isin(rounded, (0.0, 1.0))):
        raise ValueError(f"{path}: contact columns must be 0/1 bits")
    return data[:, 0:2].astype(np.float64), rounded.astype(np.int64)


def _load_mat_gait(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a MATLAB gait file with ``body_1``, ``body_2`` and ``Contact`` (columns FR, FL, MR, ML, HR, HL).

    The MATLAB export stores ``body_*`` with one extra closing sample (row N equals row 0) while ``Contact`` has N
    rows; the closing sample is dropped so the result matches the CSV layout of :func:`load_real_gait`.
    """
    from scipy.io import loadmat  # local import: only .mat inputs need scipy

    m = loadmat(str(path))
    for key in ("body_1", "body_2", "Contact"):
        if key not in m:
            raise ValueError(f"{path}: missing variable {key!r}")
    body = np.stack([np.asarray(m["body_1"]).ravel(), np.asarray(m["body_2"]).ravel()], axis=1).astype(np.float64)
    contact = np.asarray(m["Contact"], dtype=np.float64)
    if contact.ndim != 2 or contact.shape[1] != 6:
        raise ValueError(f"{path}: Contact must be [N, 6], got {contact.shape}")
    n = contact.shape[0]
    if body.shape[0] == n + 1 and np.allclose(body[-1], body[0]):
        body = body[:n]
    if body.shape[0] != n:
        raise ValueError(f"{path}: body has {body.shape[0]} rows but Contact has {n}")
    if not np.all(np.isin(contact, (0.0, 1.0))):
        raise ValueError(f"{path}: Contact must be 0/1 bits")
    return body, contact.astype(np.int64)


def real_gait_to_ticks(
    body: np.ndarray,
    contact: np.ndarray,
    *,
    air_enc: float | None = None,
    bodyampshift: float = 0.0,
    shiftnum: int = 0,
    body_offset_deg: float = 0.0,
    phase_offset_deg: float = 0.0,
    round_ticks: bool = False,
) -> dict[str, np.ndarray]:
    """Reproduce ``physics_gait_di.get_contact_Di`` and return servo ticks per joint name.

    Args:
        body: Spine angles [rad], shape [N, 2] (body_1, body_2).
        contact: Contact bits, shape [N, 6] (FR, FL, MR, ML, HR, HL).
        air_enc: Leg tick for contact = 0 [ticks]; ``None`` -> ``0.6 * 1747.5`` as on the robot.
        bodyampshift: Spine amplitude shift [deg].
        shiftnum: Body-vs-contact row shift [rows].
        body_offset_deg: Body-vs-contact shift [deg of the cycle], converted to rows.
        phase_offset_deg: Start phase [deg of the cycle], applied to body and contacts together.
        round_ticks: Round to integer ticks (half-to-even) like the robot's ``int(round(tick))``.

    Returns:
        Mapping joint name -> tick array of shape [N] (float64, or integer-valued if ``round_ticks``).
    """
    body = np.asarray(body, dtype=np.float64)
    contact = np.asarray(contact, dtype=np.float64)
    n = body.shape[0]
    w = -body - bodyampshift * 2.0 * np.pi / 360.0
    body_ticks = w * (4095.0 / (2.0 * np.pi)) + 4095.0 / 2.0

    start_rows = int(round(phase_offset_deg / 360.0 * n))
    body_rows = shiftnum + int(round(body_offset_deg / 360.0 * n))
    contact = np.roll(contact, -start_rows, axis=0)
    body_ticks = np.roll(body_ticks, body_rows - start_rows, axis=0)

    val1 = LEG_GROUND_TICK
    val0 = air_enc if air_enc is not None else AIR_FRACTION * val1
    leg_ticks = contact * val1 + (1.0 - contact) * val0

    ticks: dict[str, np.ndarray] = {}
    for i, name in enumerate(BODY_JOINTS):
        ticks[name] = body_ticks[:, i]
    for i, name in enumerate(CONTACT_JOINTS):
        ticks[name] = leg_ticks[:, i]
    if round_ticks:
        ticks = {k: np.round(v) for k, v in ticks.items()}
    return ticks


def build_mapping(config_path: str | Path, zero_tick_mode: str = "calibrated") -> JointMapping:
    """Build a :class:`JointMapping` from a deployment yaml.

    Args:
        config_path: Deployment yaml (default: the HexapI binary example).
        zero_tick_mode: ``calibrated`` (yaml zero_ticks) or ``nominal`` (all zero_ticks = 2047.5).

    Returns:
        The joint mapping.
    """
    if zero_tick_mode not in ("calibrated", "nominal"):
        raise ValueError(f"zero_tick_mode must be 'calibrated' or 'nominal', got {zero_tick_mode!r}")
    cfg = load_deployment_config(str(config_path))
    joints = cfg.joints
    if zero_tick_mode == "nominal":
        joints = dataclasses.replace(joints, zero_tick={k: ZERO_TICK_NOMINAL for k in joints.zero_tick})
    return JointMapping(joints)


def ticks_to_mimic_rad(ticks_by_name: dict[str, np.ndarray], mapping: JointMapping) -> np.ndarray:
    """Convert per-joint ticks to sim radians in :data:`MIMIC_JOINT_ORDER`.

    Args:
        ticks_by_name: Joint name (no ``_Joint`` suffix) -> tick array [N].
        mapping: Joint mapping to use.

    Returns:
        Array of shape [N, 8] [rad], columns in mimic order.
    """
    n = len(next(iter(ticks_by_name.values())))
    # Arrange by NAME in the mapping's real order, never assuming it equals the DXL order.
    real_ticks = np.stack([np.asarray(ticks_by_name[name], dtype=np.float64) for name in mapping.real_order], axis=1)
    sim_order = mapping.sim_order
    out = np.empty((n, len(MIMIC_JOINT_ORDER)), dtype=np.float64)
    for r in range(n):
        sim = mapping.ticks_to_sim_rad(real_ticks[r])
        by_name = dict(zip(sim_order, sim))
        out[r] = [by_name[j.removesuffix("_Joint")] for j in MIMIC_JOINT_ORDER]
    return out


def write_mimic_csv(path: str | Path, arr: np.ndarray) -> None:
    """Write a headerless CSV with ``%.15g`` values and ``\\n`` line endings."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="\n") as f:
        for row in arr:
            f.write(",".join(f"{v:.15g}" for v in row) + "\n")


def smooth_legs_circular(arr: np.ndarray, sigma_s: float, step_period: float) -> np.ndarray:
    """Gaussian-smooth the six leg columns of a mimic-order array over the gait cycle; the spine is untouched.

    The cycle is periodic, so the filter wraps around. The contact-bit gaits are two-level square waves; this turns
    the 1 rad steps into ramps the servos can follow.

    Args:
        arr: Mimic-order array of shape [N, 8] [rad] (columns 0-1 spine, 2-7 legs).
        sigma_s: Gaussian standard deviation [s]; 0 returns ``arr`` unchanged.
        step_period: Time per row [s].

    Returns:
        A copy of ``arr`` with the leg columns smoothed.
    """
    if sigma_s <= 0.0:
        return arr
    sigma = sigma_s / step_period
    radius = int(np.ceil(4.0 * sigma))
    kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    kernel /= kernel.sum()
    out = arr.copy()
    for j in range(2, arr.shape[1]):
        padded = np.pad(arr[:, j], radius, mode="wrap")
        out[:, j] = np.convolve(padded, kernel, mode="valid")
    return out


def convert_file(
    in_path: str | Path,
    out_path: str | Path | None,
    mapping: JointMapping,
    leg_smooth_s: float = 0.0,
    step_period: float = 0.02,
    **opts,
) -> np.ndarray:
    """Convert one real gait CSV. ``out_path=None`` skips writing (dry run). ``opts`` go to real_gait_to_ticks.

    ``leg_smooth_s`` / ``step_period`` control :func:`smooth_legs_circular` (leg columns only).
    """
    body, contact = load_real_gait(in_path)
    ticks = real_gait_to_ticks(body, contact, **opts)
    arr = ticks_to_mimic_rad(ticks, mapping)
    arr = smooth_legs_circular(arr, leg_smooth_s, step_period)
    if out_path is not None:
        write_mimic_csv(out_path, arr)
    return arr


def _collect_jobs(inputs: list[str], out: str | None) -> list[tuple[Path, Path]]:
    """Resolve (input file, output file) pairs."""
    out_p = Path(out) if out else None
    multi = len(inputs) > 1
    jobs: list[tuple[Path, Path]] = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            root = out_p if out_p is not None else DEFAULT_OUT_ROOT / p.resolve().name
            for f in sorted([*p.rglob("*.csv"), *p.rglob("*.mat")]):
                if f.name.startswith("_"):
                    continue
                jobs.append((f, root / f.relative_to(p).parent / f"{f.stem}_hexapi_sim.csv"))
        elif p.is_file():
            name = f"{p.stem}_hexapi_sim.csv"
            if out_p is None:
                dest = DEFAULT_OUT_ROOT / name
            elif out_p.is_dir() or multi or out_p.suffix.lower() != ".csv":
                dest = out_p / name
            else:
                dest = out_p
            jobs.append((p, dest))
        else:
            raise FileNotFoundError(f"input not found: {item}")
    return jobs


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns the process exit code."""
    ap = argparse.ArgumentParser(description="Convert real-robot (HexapI) gait CSVs/.mat to mimic-env reference CSVs.")
    ap.add_argument(
        "inputs", nargs="+", help="gait .csv/.mat files and/or directories (recursed; '_'-prefixed skipped)"
    )
    ap.add_argument("--out", default=None, help="output file (single input) or directory")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG), help="deployment yaml (default: binary example)")
    ap.add_argument("--zero_tick_mode", choices=["calibrated", "nominal"], default="nominal")
    ap.add_argument("--air_enc", type=float, default=None, help="leg tick in the air (default 0.6*1747.5 = 1048.5)")
    ap.add_argument("--bodyampshift", type=float, default=0.0, help="spine amplitude shift [deg]")
    ap.add_argument("--shiftnum", type=int, default=3, help="body-vs-contact row shift [rows] (robot runs use 3)")
    ap.add_argument("--body_offset_deg", type=float, default=0.0, help="body-vs-contact shift [deg of cycle]")
    ap.add_argument(
        "--phase_offset_deg",
        type=float,
        default=180.0,
        help="start phase [deg of cycle] applied to body and contacts together (robot default 180)",
    )
    ap.add_argument("--step_period", type=float, default=0.02, help="seconds per row (only for the printed period)")
    ap.add_argument(
        "--leg_smooth_s",
        type=float,
        default=0.05,
        help="Gaussian sigma [s] smoothing the six LEG columns only (spine untouched); 0 disables. Default 0.05",
    )
    ap.add_argument("--round_ticks", action="store_true", help="round ticks to integers like the robot does")
    ap.add_argument("--dry_run", action="store_true", help="convert and report, write nothing")
    ap.add_argument("--overwrite", action="store_true", help="allow overwriting existing outputs")
    args = ap.parse_args(argv)

    try:
        jobs = _collect_jobs(args.inputs, args.out)
        if not jobs:
            raise ValueError("no gait CSV files found in the given inputs")
        if not args.dry_run and not args.overwrite:
            existing = [str(o) for _, o in jobs if o.exists()]
            if existing:
                raise FileExistsError("refusing to overwrite (use --overwrite): " + ", ".join(existing))
        mapping = build_mapping(args.config, args.zero_tick_mode)
        opts = dict(
            air_enc=args.air_enc,
            bodyampshift=args.bodyampshift,
            shiftnum=args.shiftnum,
            body_offset_deg=args.body_offset_deg,
            phase_offset_deg=args.phase_offset_deg,
            round_ticks=args.round_ticks,
        )
        verbose = len(jobs) <= 3
        for src, dst in jobs:
            arr = convert_file(
                src,
                None if args.dry_run else dst,
                mapping,
                leg_smooth_s=args.leg_smooth_s,
                step_period=args.step_period,
                **opts,
            )
            period = arr.shape[0] * args.step_period
            tag = "would write" if args.dry_run else "wrote"
            print(f"{src.name} -> {tag} {dst} [{arr.shape[0]} rows, period {period:.4g} s]")
            if verbose:
                for j, name in enumerate(MIMIC_JOINT_ORDER):
                    print(f"  {name:18s} min {arr[:, j].min():+.4f}  max {arr[:, j].max():+.4f} rad")
        print(
            f"Converted {len(jobs)} gait(s). Cycle period = rows * step_period ({args.step_period} s/row). "
            "Set MIMIC_GAIT_PERIOD in hexapod_mimic_env_cfg.py (currently 1.0) to it to replay at real-robot speed."
        )
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
