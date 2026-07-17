"""Per-step CSV run logger.

The `obs_*`/`action_*` columns double as the reference-trace format
`tools/validate_onnx.py`'s direct mode reads -- any run logged with this class
(sim or real) can later be replayed through `PolicyRunner` to check for onnxruntime
export/platform drift.
"""

from __future__ import annotations

import csv
import os

import numpy as np

from .profiles import NUM_JOINTS


class CsvRunLogger:
    def __init__(self, path: str, obs_dim: int, action_dim: int):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._file = open(path, "w", newline="")
        self._writer = csv.writer(self._file)
        header = (
            ["step", "t", "loop_time_ms"]
            + [f"obs_{i}" for i in range(obs_dim)]
            + [f"action_{i}" for i in range(action_dim)]
            + [f"target_sim_{i}" for i in range(NUM_JOINTS)]
            + [f"target_ticks_{i}" for i in range(NUM_JOINTS)]
            + [f"measured_pos_sim_{i}" for i in range(NUM_JOINTS)]
            + [f"measured_vel_sim_{i}" for i in range(NUM_JOINTS)]
        )
        self._writer.writerow(header)

    def log(
        self,
        step: int,
        t: float,
        obs: np.ndarray,
        raw_action: np.ndarray,
        target_sim: np.ndarray,
        target_ticks: np.ndarray,
        measured_pos_sim: np.ndarray,
        measured_vel_sim: np.ndarray,
        loop_time_ms: float,
    ) -> None:
        row = (
            [step, t, loop_time_ms]
            + list(obs)
            + list(raw_action)
            + list(target_sim)
            + list(target_ticks)
            + list(measured_pos_sim)
            + list(measured_vel_sim)
        )
        self._writer.writerow(row)

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> "CsvRunLogger":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
