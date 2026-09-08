# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Watch a training run's checkpoint directory and periodically record a short progress video.

This is a plain-Python script (no Isaac Sim import) meant to be run in a second terminal,
alongside a live ``isaaclab.bat train ...`` process, so training itself never has to enable
camera rendering. Enabling ``--video`` directly on a large-``num_envs`` training run boots the
RTX/Kit rendering pipeline on top of the whole training scene and can push memory usage over
the edge; a separate ``play --num_envs 1`` pass against a saved checkpoint has a much smaller
footprint and can safely run concurrently with training.

The script polls ``<experiment_root>/<latest_run>/`` for new ``model_<iteration>.pt``
checkpoints. Whenever a checkpoint at or past the next multiple of ``--interval_iterations``
appears, it shells out to ``isaaclab.bat play --num_envs 1 --checkpoint <path> --video`` and
moves the resulting mp4 into ``<run_dir>/videos/progress/model_<iteration>.mp4`` (play always
records to the same file name inside the same run directory, so without this move step every
invocation would overwrite the previous checkpoint's video).

Example:

.. code-block:: bat

    python scripts/tools/watch_and_record_video.py ^
        --experiment_root logs/rsl_rl/hexapod_goal ^
        --task Isaac-Goal-Flat-Hexapod-Play-v0 ^
        --interval_iterations 250
"""

from __future__ import annotations

import argparse
import re
import subprocess
import time
from pathlib import Path

CHECKPOINT_RE = re.compile(r"model_(\d+)\.pt$")
REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--experiment_root",
        required=True,
        help="Training log root, e.g. logs/rsl_rl/hexapod_goal. The most recently created run"
        " subdirectory inside it is watched, re-resolved on every poll.",
    )
    parser.add_argument(
        "--task",
        required=True,
        help="Play task id to record with, e.g. Isaac-Goal-Flat-Hexapod-Play-v0.",
    )
    parser.add_argument("--interval_iterations", type=int, default=250, help="Record a video every N iterations.")
    parser.add_argument("--video_length", type=int, default=200, help="Length of each recorded video (in steps).")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of envs for the play pass.")
    parser.add_argument("--poll_seconds", type=float, default=30.0, help="Filesystem poll interval.")
    parser.add_argument("--isaaclab_bat", default=str(REPO_ROOT / "isaaclab.bat"), help="Path to isaaclab.bat.")
    return parser.parse_args()


def latest_run_dir(experiment_root: Path) -> Path | None:
    if not experiment_root.is_dir():
        return None
    runs = sorted(d for d in experiment_root.iterdir() if d.is_dir())
    return runs[-1] if runs else None


def find_checkpoints(run_dir: Path) -> dict[int, Path]:
    checkpoints = {}
    for entry in run_dir.iterdir():
        match = CHECKPOINT_RE.match(entry.name)
        if match:
            checkpoints[int(match.group(1))] = entry
    return checkpoints


def record_video(args: argparse.Namespace, run_dir: Path, iteration: int, checkpoint: Path) -> None:
    play_video_dir = run_dir / "videos" / "play"
    before = set(play_video_dir.glob("*.mp4")) if play_video_dir.is_dir() else set()

    cmd = [
        "cmd",
        "/c",
        args.isaaclab_bat,
        "play",
        "--rl_library",
        "rsl_rl",
        "--task",
        args.task,
        "--num_envs",
        str(args.num_envs),
        "--checkpoint",
        str(checkpoint),
        "--video",
        "--video_length",
        str(args.video_length),
        "--headless",
    ]
    print(f"[watch] iteration {iteration}: recording -> {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    if result.returncode != 0:
        print(f"[watch] WARNING: play exited with code {result.returncode} for iteration {iteration}; skipping move")
        return

    after = set(play_video_dir.glob("*.mp4")) if play_video_dir.is_dir() else set()
    new_files = after - before
    if not new_files:
        print(f"[watch] WARNING: no new mp4 found in {play_video_dir} for iteration {iteration}")
        return

    progress_dir = run_dir / "videos" / "progress"
    progress_dir.mkdir(parents=True, exist_ok=True)
    dest = progress_dir / f"model_{iteration:05d}.mp4"
    # only one new file is expected per play invocation (fixed video_folder, step_trigger at step 0)
    next(iter(new_files)).replace(dest)
    print(f"[watch] saved {dest}")


def main() -> None:
    args = parse_args()
    experiment_root = Path(args.experiment_root).resolve()

    current_run_dir: Path | None = None
    seen_iterations: set[int] = set()
    next_target = args.interval_iterations

    print(f"[watch] watching {experiment_root} every {args.poll_seconds}s (interval={args.interval_iterations} iters)")
    try:
        while True:
            run_dir = latest_run_dir(experiment_root)
            if run_dir is None:
                print(f"[watch] no run directory under {experiment_root} yet, waiting...")
                time.sleep(args.poll_seconds)
                continue

            if run_dir != current_run_dir:
                print(f"[watch] tracking run: {run_dir}")
                current_run_dir = run_dir
                seen_iterations = set()
                next_target = args.interval_iterations

            checkpoints = find_checkpoints(run_dir)
            candidates = sorted(it for it in checkpoints if it >= next_target and it not in seen_iterations)
            if candidates:
                iteration = candidates[0]
                record_video(args, run_dir, iteration, checkpoints[iteration])
                seen_iterations.add(iteration)
                next_target = ((iteration // args.interval_iterations) + 1) * args.interval_iterations

            time.sleep(args.poll_seconds)
    except KeyboardInterrupt:
        print("[watch] stopped")


if __name__ == "__main__":
    main()
