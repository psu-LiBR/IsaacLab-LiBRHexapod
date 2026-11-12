"""Extract cycle features from terrain recording folders using only the standard library.

Manifest columns: path,session_id,split. Paths name folders containing meta.json,
imu.csv, joints.csv and cycles.csv. Labels come from meta.json surface.
"""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

MOTORS = ("link1", "link2", "FR", "FL", "MR", "ML", "HR", "HL")


def read_csv(path: Path) -> list[dict]:
    """Read CSV rows, accepting an optional UTF-8 byte order mark."""
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def numeric(row: dict, key: str) -> float:
    """Read a numeric field and reject nonfinite sensor values."""
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"Nonfinite {key}")
    return value


def check_clock(rows: list[dict], key: str) -> None:
    """Require a nonempty sequence of strictly increasing timestamps."""
    times = [numeric(row, key) for row in rows]
    if not times or any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError(f"Empty or non-increasing {key}")


def summarize(output: dict, name: str, values: list[float]) -> None:
    """Add mean, standard deviation, RMS, and range features for one channel."""
    output[name + "_mean"] = statistics.fmean(values)
    output[name + "_std"] = statistics.pstdev(values)
    output[name + "_rms"] = math.sqrt(statistics.fmean(v * v for v in values))
    output[name + "_range"] = max(values) - min(values)


def extract(folder: Path, session: str, split: str) -> list[dict]:
    """Validate a recording and extract one labeled feature row per gait cycle."""
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8-sig"))
    if meta.get("schema_version") != 1 or meta.get("completed") is not True:
        raise ValueError(f"Unsupported or incomplete recording: {folder}")
    label = meta["surface"].strip()
    if not label:
        raise ValueError("Missing surface label")
    if tuple(meta["motors"]["names"]) != MOTORS:
        raise ValueError("Unexpected motor ordering")
    joints, imu = read_csv(folder / "joints.csv"), read_csv(folder / "imu.csv")
    check_clock(joints, "t_cmd")
    check_clock(joints, "t_read")
    check_clock(imu, "t_recv")
    period = float(meta["args"]["step_period"])
    steps = int(meta["gait"]["n_steps"])
    if period <= 0 or steps <= 0:
        raise ValueError("Invalid gait timing")
    result = []
    cycles = sorted({int(row["cycle"]) for row in joints})
    for cycle in cycles:
        rows = [r for r in joints if int(r["cycle"]) == cycle]
        if [int(r["step"]) for r in rows] != list(range(steps)):
            raise ValueError(f"Incomplete cycle {cycle}")
        if any(r["read_ok"] != "1" for r in rows):
            raise ValueError(f"Failed joint read in cycle {cycle}")
        start = numeric(rows[0], "t_cmd")
        following = [numeric(r, "t_cmd") for r in joints if int(r["cycle"]) > cycle]
        end = min(following) if following else float(meta["gait_end_epoch"])
        samples = [r for r in imu if start <= numeric(r, "t_recv") < end]
        times = [numeric(r, "t_recv") for r in samples]
        if (
            len(times) < 3
            or times[0] - start > 0.05
            or end - times[-1] > 0.05
            or any(b - a > 0.05 for a, b in zip(times, times[1:]))
        ):
            raise ValueError(f"Insufficient IMU coverage in cycle {cycle}")
        if end <= start or any(numeric(r, "t_read") - numeric(r, "t_cmd") > 0.1 for r in rows):
            raise ValueError(f"Invalid timing in cycle {cycle}")
        output = dict(
            label=label,
            session_id=session,
            split=split,
            run_id=meta["run_id"],
            cycle=cycle,
            start_s=start,
            end_s=end,
            imu_samples=len(samples),
            joint_samples=len(rows),
            gait_sha256=meta["gait"]["sha256"],
            source=str(folder),
        )
        for channel in ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"):
            summarize(output, "feature_" + channel, [numeric(r, channel) for r in samples])
        for motor in MOTORS:
            for kind, scale in [("vel", 0.229 * 2 * math.pi / 60), ("load", 0.1)]:
                summarize(output, f"feature_{kind}_{motor}", [numeric(r, kind + "_" + motor) * scale for r in rows])
            errors = [(numeric(r, "cmd_" + motor) - numeric(r, "pos_" + motor)) * 2 * math.pi / 4096 for r in rows]
            summarize(output, f"feature_error_{motor}", errors)
        result.append(output)
    return result


def build(manifest: Path, destination: Path) -> None:
    """Write cycle features from a session-isolated recording manifest to CSV."""
    runs = read_csv(manifest)
    output, groups, seen = [], {}, set()
    for run in runs:
        session, split = run["session_id"].strip(), run["split"]
        if not session or split not in ("train", "validation", "test"):
            raise ValueError("Require session_id and train/validation/test split")
        if groups.setdefault(session, split) != split:
            raise ValueError("Session assigned to multiple splits")
        folder = (manifest.parent / run["path"]).resolve()
        if folder in seen:
            raise ValueError("Duplicate recording in manifest")
        seen.add(folder)
        output.extend(extract(folder, session, split))
    if not output:
        raise ValueError("No cycles found")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    print(f"Saved {len(output)} cycles, 120 features each, to {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="Path to the recording manifest CSV.")
    parser.add_argument("output", type=Path, help="Path to the output cycle-feature CSV.")
    args = parser.parse_args()
    build(args.manifest, args.output)
