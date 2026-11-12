"""Build terrain features from CsvRunLogger recordings; requires NumPy only."""

import argparse
import csv
from pathlib import Path

import numpy as np


def build(manifest: Path, output: Path, window_s: float = 1.0, stride_s: float = 0.5, max_gap_s: float = 0.1) -> None:
    """Validate CSV runs and save windowed sensor features with their dataset splits."""
    if min(window_s, stride_s, max_gap_s) <= 0:
        raise ValueError("Window, stride and maximum gap must be positive")
    with manifest.open(newline="") as stream:
        runs = list(csv.DictReader(stream))
    if not runs:
        raise ValueError("Manifest is empty")
    groups = {}
    features, labels, splits, sessions, starts, sources = [], [], [], [], [], []
    names: list[str] = []
    for run in runs:
        session, split = run["session_id"], run["split"]
        if not session or split not in {"train", "validation", "test"}:
            raise ValueError("Each run needs a session_id and train/validation/test split")
        if groups.setdefault(session, split) != split:
            raise ValueError(f"Session {session} occurs in multiple splits")
        if not run["label"].strip():
            raise ValueError("Every recording must have a terrain label")
        path = (manifest.parent / run["path"]).resolve()
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            raise ValueError(f"Empty recording: {path}")
        columns = [f"obs_{i}" for i in range(6)]
        columns += [f"measured_pos_sim_{i}" for i in range(8)]
        columns += [f"measured_vel_sim_{i}" for i in range(8)]
        targets = [f"target_sim_{i}" for i in range(8)]
        t = np.array([float(r["t"]) for r in rows])
        x = np.array([[float(r[c]) for c in columns] for r in rows])
        target = np.array([[float(r[c]) for c in targets] for r in rows])
        x = np.column_stack((x, target - x[:, 6:14]))
        channels = ["gyro_x", "gyro_y", "gyro_z", "gravity_x", "gravity_y", "gravity_z"]
        channels += [f"position_{i}" for i in range(8)]
        channels += [f"velocity_{i}" for i in range(8)]
        channels += [f"tracking_error_{i}" for i in range(8)]
        names = [f"{channel}_{stat}" for stat in ("mean", "std", "rms", "range") for channel in channels]
        if not np.isfinite(t).all() or not np.isfinite(x).all() or np.any(np.diff(t) <= 0):
            raise ValueError(f"Nonfinite values or non-increasing timestamps: {path}")
        count = 0
        for start in np.arange(t[0], t[-1] - window_s + 1e-9, stride_s):
            lo, hi = np.searchsorted(t, [start, start + window_s])
            times, values = t[lo:hi], x[lo:hi]
            if len(times) < 3 or np.max(np.diff(times)) > max_gap_s:
                continue
            if times[0] - start > max_gap_s or start + window_s - times[-1] > max_gap_s:
                continue
            features.append(
                np.concatenate((values.mean(0), values.std(0), np.sqrt((values**2).mean(0)), np.ptp(values, axis=0)))
            )
            labels.append(run["label"])
            splits.append(split)
            sessions.append(session)
            sources.append(str(path))
            starts.append(start)
            count += 1
        if not count:
            raise ValueError(f"No valid windows in {path}; check duration and sampling gaps")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        X=np.asarray(features),
        y=np.asarray(labels),
        split=np.asarray(splits),
        session_id=np.asarray(sessions),
        start_s=np.asarray(starts),
        source=np.asarray(sources),
        feature_names=np.asarray(names),
        window_s=window_s,
        stride_s=stride_s,
        max_gap_s=max_gap_s,
        schema_version=1,
    )
    print(f"Saved {len(features)} windows with {len(names)} features to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--window-s", type=float, default=1.0)
    parser.add_argument("--stride-s", type=float, default=0.5)
    parser.add_argument("--max-gap-s", type=float, default=0.1)
    args = parser.parse_args()
    build(args.manifest, args.output, args.window_s, args.stride_s, args.max_gap_s)
