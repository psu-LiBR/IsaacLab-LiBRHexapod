# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Convert the reference tripod gait CSV into a 6-bit contact demonstration sequence.

Raw material for demonstration-augmented RL (DQfD-style) and for the ``eval_protocol.py``
tripod baseline. Produces **leg contact bits only** -- the two spine joints are handled
separately by :class:`SpineSineAction` and this script never touches them. The hardware
tripod gait (``hexapod-assets/Sim Gaits/`` CSV, 51 rows x 8 cols, 0.02 s/row, no header,
Sim DOF column order -- ``CSV_DEFAULT`` below picks which one) snaps each leg between
exactly two joint levels, so each leg column thresholds cleanly into a contact bit:

    stance level -0.4602 rad / lift level -1.1804 rad (pre-HexapI sign convention;
    the sign flip does not matter here because only the two-level split is used)
    -> bit = 1 (stance) iff value > midpoint (-0.8203).

Output bit order matches the binary env action layout / DiscreteBitsActionWrapper:
FR, FL, MR, ML, BR, BL with the integer action's LSB = FR.  CSV columns are
[0 BackLink, 1 FrontLink, 2 MiddleLeft, 3 MiddleRight, 4 BackLeft, 5 BackRight,
6 FrontLeft, 7 FrontRight]; legs = columns 2..7.

No simulation is touched -- plain numpy.  Run with any python that has numpy:
  python scripts/reinforcement_learning/extract_bit_demos.py
"""

import argparse
import hashlib
import os

import numpy as np

# Baseline gait source. tripod_extendedquad_sim.csv (spine joints move in phase; leg
# swing/stance switch at steps 13/37) is the comparison reference; tripod_B11BL0_sim.csv
# (spine out of phase; switch at 7/31) is the older variant. This script only ever reads
# the LEG columns (2..7) and only ever writes LEG contact bits -- the npz carries no spine
# data. The spine wave is configured separately in hexapod_binary_env_cfg.py and is NOT
# changed by re-running this script: the RL env plays a fixed analytic traveling wave
# (Wave 1). Wave 2 (the reference-tripod wave) is an analytic anti-phase wave
# (BackLink = -FrontLink) regenerated from the MATLAB gait generator -- NOT a fit of this
# CSV's byte-identical / in-phase spine columns -- and is swapped in only by
# eval_protocol.py's tripod baseline and play_discrete_closeup.py's --gait_npz tripod
# replay (via SpineSineAction.set_waveform).
CSV_DEFAULT = os.path.join("hexapod-assets", "Sim Gaits", "tripod_extendedquad_sim.csv")
LEG_COLS = {"FrontRight": 7, "FrontLeft": 6, "MiddleRight": 3, "MiddleLeft": 2, "BackRight": 5, "BackLeft": 4}
BIT_ORDER = ["FrontRight", "FrontLeft", "MiddleRight", "MiddleLeft", "BackRight", "BackLeft"]
STANCE_LEVEL = -0.460194236365692
LIFT_LEVEL = -1.180398216278
THRESHOLD = 0.5 * (STANCE_LEVEL + LIFT_LEVEL)  # -0.8203
DT = 0.02

parser = argparse.ArgumentParser()
parser.add_argument("--csv", default=CSV_DEFAULT)
parser.add_argument(
    "--out_dir",
    default="runs_discrete/demos_tripod_bits",
    help="where to write tripod_bit_demos.npz/.md; the copy consumed by eval_protocol.py "
    "and play_discrete_closeup.py is scripts/reinforcement_learning/binary_rl/tripod_bit_demos.npz "
    "-- pass --out_dir there to refresh it in place",
)
parser.add_argument("--level_tol", type=float, default=0.02, help="max |value - nearest level| accepted, rad")
args = parser.parse_args()

data = np.loadtxt(args.csv, delimiter=",")
assert data.ndim == 2 and data.shape[1] == 8, f"unexpected CSV shape {data.shape}"
T = data.shape[0]

legs = data[:, [LEG_COLS[n] for n in BIT_ORDER]]  # [T, 6] in bit order
dev = np.minimum(np.abs(legs - STANCE_LEVEL), np.abs(legs - LIFT_LEVEL))
assert dev.max() <= args.level_tol, f"leg values are not two-level within tol: max deviation {dev.max():.4f} rad"
bits = (legs > THRESHOLD).astype(np.uint8)  # 1 = stance
action_idx = (bits * (1 << np.arange(6))[None, :]).sum(axis=1).astype(np.int16)  # LSB = FR

periodic = bool(np.array_equal(bits[0], bits[-1]))
period = T - 1 if periodic else T
stance_frac = bits[:period].mean(axis=0)

with open(args.csv, "rb") as f:
    sha = hashlib.sha256(f.read()).hexdigest()

os.makedirs(args.out_dir, exist_ok=True)
np.savez(
    os.path.join(args.out_dir, "tripod_bit_demos.npz"),
    bits=bits,
    action_idx=action_idx,
    dt=DT,
    period_steps=period,
    bit_order=np.array(BIT_ORDER),
    leg_csv_cols=np.array([LEG_COLS[n] for n in BIT_ORDER]),
    threshold=THRESHOLD,
    stance_level=STANCE_LEVEL,
    lift_level=LIFT_LEVEL,
    source_csv=args.csv,
    source_sha256=sha,
)
with open(os.path.join(args.out_dir, "tripod_bit_demos.md"), "w") as f:
    f.write("# Tripod 6-bit demonstration sequence\n\n")
    f.write(f"- source: `{args.csv}` (sha256 `{sha[:16]}...`), {T} rows x 8 cols, {DT} s/row\n")
    f.write(
        f"- levels: stance {STANCE_LEVEL:.4f} / lift {LIFT_LEVEL:.4f} rad, "
        f"threshold {THRESHOLD:.4f}, max level deviation {dev.max():.5f} rad\n"
    )
    f.write(f"- bit order (int LSB first): {BIT_ORDER}\n")
    f.write(f"- periodic: row[0] == row[-1] -> {periodic}, period = {period} steps ({period * DT:.2f} s)\n")
    f.write(f"- per-leg stance fraction over one period: {np.round(stance_frac, 3).tolist()}\n")
    f.write(f"- distinct action ids: {sorted(set(action_idx.tolist()))}\n\n")
    f.write("step | action | " + " ".join(n[:2] for n in BIT_ORDER) + "\n")
    f.write("-----|--------|------------------\n")
    for t in range(T):
        f.write(f"{t:4d} | {int(action_idx[t]):6d} | " + "  ".join(str(int(b)) for b in bits[t]) + "\n")

print(f"[extract_bit_demos] {T} rows -> bits {bits.shape}, periodic={periodic} period={period}")
print(f"[extract_bit_demos] stance fractions {np.round(stance_frac, 3).tolist()}")
print(f"[extract_bit_demos] distinct actions: {sorted(set(action_idx.tolist()))}")
print(f"[extract_bit_demos] outputs in {args.out_dir}")
