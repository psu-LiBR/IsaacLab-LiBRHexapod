# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Portable per-step CSV logs for the hexapod fork's play scripts.

The hexapod play scripts (``play_rsl_rl``, ``playReal``, ``playpyvista``, ``play_physicsGait``,
``playTracking``) can log joint positions and base displacement of env 0 to CSV. The logging is **off by
default**: pass ``--save_play_logs`` to write the logs to ``<checkpoint_dir>/play_logs/``, or ``--play_log_dir``
to write them to a directory of your choice (this also switches the logging on). Every file name carries a
run timestamp, so repeated runs never overwrite each other. A log that cannot be written only prints a
warning; it never stops the playback.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import logging
import os
import time
from collections.abc import Sequence
from typing import Any

logger = logging.getLogger(__name__)

PLAY_LOG_SUBDIR = "play_logs"
"""Default sub-directory (of the checkpoint directory) for the play CSV logs."""

DISPLACEMENT_HEADER = [
    "step",
    "t_s",
    "cycle_num",
    "cycle_step",
    "gait_idx",
    "x_w",
    "y_w",
    "z_w",
    "dx",
    "dy",
    "dz",
    "forward_disp_x",
]
"""Header of the base-displacement log (``sim_displacement_log_*.csv``)."""


def joint_pos_header(num_joints: int) -> list[str]:
    """Header of a joint-position log with ``num_joints`` columns."""
    return [f"joint_{i}_pos_rad" for i in range(num_joints)]


def add_play_log_dir_arg(parser: argparse.ArgumentParser) -> None:
    """Add the opt-in ``--save_play_logs`` and ``--play_log_dir`` arguments to a play-script parser."""
    parser.add_argument(
        "--save_play_logs",
        action="store_true",
        default=False,
        help=(
            "Write the per-step CSV logs (joint positions, base displacement). Off by default."
            f" Files go to '<checkpoint_dir>/{PLAY_LOG_SUBDIR}/' unless --play_log_dir is given."
        ),
    )
    parser.add_argument(
        "--play_log_dir",
        type=str,
        default=None,
        help="Directory for the per-step CSV logs. Giving it also enables the logging. File names are timestamped.",
    )


def resolve_play_log_dir(args_cli: argparse.Namespace, default_parent_dir: str) -> str | None:
    """Return the directory for the play CSV logs, or ``None`` when logging is not requested.

    Args:
        args_cli: Parsed CLI arguments (``save_play_logs`` / ``play_log_dir`` are used).
        default_parent_dir: Directory under which the default ``play_logs`` sub-directory is placed,
            normally the checkpoint directory.

    Returns:
        The absolute log directory (created lazily by :class:`PlayCsvLog`), or ``None`` if neither
        ``--save_play_logs`` nor ``--play_log_dir`` was given.
    """
    override = getattr(args_cli, "play_log_dir", None)
    if not override and not getattr(args_cli, "save_play_logs", False):
        return None
    log_dir = override if override else os.path.join(default_parent_dir, PLAY_LOG_SUBDIR)
    return os.path.abspath(os.path.expanduser(log_dir))


def run_timestamp() -> str:
    """Timestamp used to tag all CSV logs of one play run."""
    return time.strftime("%Y-%m-%d_%H-%M-%S")


class PlayCsvLog:
    """Line-buffered CSV log that disables itself with a warning instead of raising on I/O errors.

    A log created with ``log_dir=None`` is permanently disabled and silently ignores all rows.
    """

    def __init__(self, log_dir: str | None, prefix: str, header: Sequence[str], timestamp: str):
        """Create ``<log_dir>/<prefix>_<timestamp>.csv`` and write the header row.

        Args:
            log_dir: Output directory, created if missing. ``None`` disables the log.
            prefix: File-name prefix, e.g. ``"HexapodRL_Rad"``.
            header: Column names of the header row.
            timestamp: Run timestamp shared by all logs of the run (see :func:`run_timestamp`).
        """
        self.path = None if log_dir is None else os.path.join(log_dir, f"{prefix}_{timestamp}.csv")
        self._file = None
        self._writer = None
        if log_dir is None:
            return
        try:
            os.makedirs(log_dir, exist_ok=True)
            # line buffering keeps the rows on disk if the run is interrupted
            self._file = open(self.path, "w", newline="", buffering=1)  # noqa: SIM115 -- closed in close()
            self._writer = csv.writer(self._file)
            self._writer.writerow(list(header))
            print(f"[INFO] Logging to: {self.path}")
        except OSError as err:
            logger.warning("Could not create play log '%s', logging disabled: %s", self.path, err)
            self.close()

    @property
    def enabled(self) -> bool:
        """Whether rows are still being written."""
        return self._writer is not None

    def writerow(self, row: Sequence[Any]) -> None:
        """Append one row. Disables the log on an I/O error."""
        if self._writer is None:
            return
        try:
            self._writer.writerow(list(row))
        except OSError as err:
            logger.warning("Could not write play log '%s', logging disabled: %s", self.path, err)
            self.close()

    def close(self) -> None:
        """Close the file. Safe to call more than once."""
        self._writer = None
        if self._file is not None:
            with contextlib.suppress(OSError):
                self._file.close()
            self._file = None
