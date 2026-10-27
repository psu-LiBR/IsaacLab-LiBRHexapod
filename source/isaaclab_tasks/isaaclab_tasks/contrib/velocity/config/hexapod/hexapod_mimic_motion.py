# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reference motion data for hexapod imitation learning.

Provides a MotionReference that returns target joint angles at a given gait phase, loaded from a CSV or
generated as a built-in sinusoidal tripod gait.

Joint order is the sim DOF order of ``asset.data.joint_pos``:

    0 BackLink_Joint   1 FrontLink_Joint   2 MiddleLeft_Joint  3 MiddleRight_Joint
    4 BackLeft_Joint   5 BackRight_Joint   6 FrontLeft_Joint   7 FrontRight_Joint

CSV: either headerless rows in that order, or a header row with joint names (columns matched by name, any
order). A leading "time" column is optional; without it the rows are spaced evenly over one gait period.

Typical usage inside a reward function::

    ref = MotionReference(csv_path="C:/path/to/gait.csv", gait_period=0.5)
    phase = ref.get_phase(env.episode_length_buf * env.step_dt)  # [N]
    q_target = ref.get_reference(phase)  # [N, 8]
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

# Sim DOF order, as in asset.data.joint_pos.
JOINT_NAMES: list[str] = [
    "BackLink_Joint",  # 0  (spine)
    "FrontLink_Joint",  # 1  (spine)
    "MiddleLeft_Joint",  # 2
    "MiddleRight_Joint",  # 3
    "BackLeft_Joint",  # 4
    "BackRight_Joint",  # 5
    "FrontLeft_Joint",  # 6
    "FrontRight_Joint",  # 7
]
NUM_JOINTS: int = 8

# Alternating tripod groups (indices into JOINT_NAMES).
TRIPOD_A: list[int] = [3, 4, 6]  # MiddleRight_Joint, BackLeft_Joint, FrontLeft_Joint
TRIPOD_B: list[int] = [2, 5, 7]  # MiddleLeft_Joint,  BackRight_Joint, FrontRight_Joint
SPINE_IDX: list[int] = [0, 1]  # BackLink_Joint, FrontLink_Joint


class MotionReference:
    """Cyclic gait reference for hexapod imitation learning.

    The reference is indexed by a normalised gait phase phi ∈ [0, 1) computed
    from elapsed episode time.  The internal table is stored as a float32 NumPy
    array and is transferred to GPU lazily on the first call to get_reference().

    Args:
        csv_path:  Path to CSV file with reference joint trajectory.
                   If None, a built-in sinusoidal tripod gait is generated
                   instead. A path that does not exist raises FileNotFoundError.
        gait_period: Duration of one gait cycle in seconds.  If a time column is
                     present in the CSV this value is overwritten by the data.
        num_interp_points: Number of uniformly spaced phase samples in the table.
    """

    def __init__(
        self,
        csv_path: str | None = None,
        gait_period: float = 1.0,
        num_interp_points: int = 200,
        csv_col_order: list[int] | None = None,
    ) -> None:
        """
        Args:
            csv_path:       Path to reference gait CSV (headerless or headered).
            gait_period:    Gait cycle duration in seconds.  Overridden if the
                            CSV has a time column spanning one cycle.
            num_interp_points: Resolution of the internal phase table.
            csv_col_order:  Optional integer list that reorders the CSV columns to the sim DOF order
                            (not needed for CSVs already in that order).
        """
        self.gait_period = gait_period
        self._n = num_interp_points
        self._csv_col_order = csv_col_order
        self._ref_np: np.ndarray | None = None
        self._ref_gpu: torch.Tensor | None = None  # populated lazily

        if csv_path is not None:
            p = Path(csv_path)
            if p.exists():
                self._ref_np = self._load_csv(str(p))
            else:
                raise FileNotFoundError(
                    f"Mimic reference CSV not found at '{csv_path}' (resolved against the current directory "
                    f"'{Path.cwd()}'; relative paths assume you launch from the repo root). Pass csv_path=None "
                    "to use the built-in sinusoidal tripod gait instead."
                )

        if self._ref_np is None:
            self._ref_np = self._generate_tripod()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_phase(self, episode_time: torch.Tensor) -> torch.Tensor:
        """Convert elapsed episode time (seconds) to gait phase in [0, 1).

        Args:
            episode_time: Tensor of shape [num_envs].
        Returns:
            Phase tensor of shape [num_envs], values in [0, 1).
        """
        return (episode_time % self.gait_period) / self.gait_period

    def get_reference(self, phase: torch.Tensor) -> torch.Tensor:
        """Return target joint positions for the given phases.

        Args:
            phase: Tensor of shape [num_envs] with values in [0, 1).
        Returns:
            Tensor of shape [num_envs, 8] — target absolute joint angles (rad)
            in alphabetical joint order (BackLeft_Joint … MiddleRight_Joint).
        """
        if self._ref_gpu is None or self._ref_gpu.device != phase.device:
            self._ref_gpu = torch.from_numpy(self._ref_np).to(phase.device)
        indices = (phase * self._n).long().clamp(0, self._n - 1)
        return self._ref_gpu[indices]  # [num_envs, 8]

    def get_reference_per_joint(self, phase: torch.Tensor) -> torch.Tensor:
        """Return target joint positions with a separate phase for every joint.

        Args:
            phase: Tensor of shape [num_envs, 8] with values in [0, 1); column j is the phase joint j is read at.
        Returns:
            Tensor of shape [num_envs, 8] — joint j taken from the table at ``phase[:, j]`` [rad].
        """
        if self._ref_gpu is None or self._ref_gpu.device != phase.device:
            self._ref_gpu = torch.from_numpy(self._ref_np).to(phase.device)
        indices = (phase * self._n).long().clamp(0, self._n - 1)  # [N, 8]
        joints = torch.arange(NUM_JOINTS, device=phase.device).expand_as(indices)
        return self._ref_gpu[indices, joints]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_csv(self, path: str) -> np.ndarray:
        """Load and resample joint trajectory from CSV to a uniform phase grid.

        Returns ndarray of shape [num_interp_points, 8], dtype float32.
        """
        # Detect whether the first row is a header by trying to parse it as floats.
        with open(path) as f:
            first_line = f.readline().strip()
        try:
            [float(x.strip()) for x in first_line.split(",") if x.strip()]
            has_header = False
        except ValueError:
            has_header = True

        if has_header:
            raw = np.genfromtxt(path, delimiter=",", names=True, dtype=float)
            col_names = [n.lower() for n in raw.dtype.names]
            data = np.zeros((len(raw), NUM_JOINTS), dtype=np.float32)
            for j, jname in enumerate(JOINT_NAMES):
                matches = [raw.dtype.names[i] for i, n in enumerate(col_names) if n == jname.lower()]
                if not matches:
                    raise ValueError(f"Column '{jname}' not found in CSV header. Available columns: {raw.dtype.names}")
                data[:, j] = raw[matches[0]].astype(np.float32)
            time_cols = [raw.dtype.names[i] for i, n in enumerate(col_names) if n == "time"]
            if time_cols:
                times = raw[time_cols[0]].astype(np.float64)
                self.gait_period = float(times[-1] - times[0])
                phases = (times - times[0]) / self.gait_period
            else:
                phases = np.linspace(0.0, 1.0, len(data), endpoint=False)
        else:
            # No header: pure numeric array.
            arr = np.loadtxt(path, delimiter=",")
            if arr.ndim == 1:
                arr = arr[None, :]
            if arr.shape[1] == NUM_JOINTS + 1:
                times = arr[:, 0].astype(np.float64)
                self.gait_period = float(times[-1] - times[0])
                phases = (times - times[0]) / self.gait_period
                data = arr[:, 1:].astype(np.float32)
            elif arr.shape[1] == NUM_JOINTS:
                phases = np.linspace(0.0, 1.0, len(arr), endpoint=False)
                data = arr.astype(np.float32)
            else:
                raise ValueError(
                    f"CSV has {arr.shape[1]} columns. Expected {NUM_JOINTS} (joints only) "
                    f"or {NUM_JOINTS + 1} (time + joints)."
                )

        # Reorder columns to Isaac Lab alphabetical joint order if requested.
        if self._csv_col_order is not None:
            data = data[:, self._csv_col_order]

        # Resample to uniform phase grid via linear interpolation.
        grid = np.linspace(0.0, 1.0, self._n, endpoint=False)
        resampled = np.zeros((self._n, NUM_JOINTS), dtype=np.float32)
        for j in range(NUM_JOINTS):
            # Wrap phase: duplicate first point at end for continuity.
            ph = np.append(phases % 1.0, 1.0)
            vals = np.append(data[:, j], data[0, j])
            resampled[:, j] = np.interp(grid, ph, vals)
        return resampled

    def _generate_tripod(self) -> np.ndarray:
        """Generate a sinusoidal tripod gait reference.

        Joint ordering matches asset.data.joint_pos (Sim DOF order, confirmed via playReal.py):
          [0] BackLink_Joint (spine)  [1] FrontLink_Joint (spine)
          [2] MiddleLeft_Joint        [3] MiddleRight_Joint
          [4] BackLeft_Joint          [5] BackRight_Joint
          [6] FrontLeft_Joint         [7] FrontRight_Joint

        Tripod A (swing at phase [0, π]):  MiddleRight_Joint[3], BackLeft_Joint[4], FrontLeft_Joint[6]
        Tripod B (swing at phase [π, 2π]): MiddleLeft_Joint[2],  BackRight_Joint[5], FrontRight_Joint[7]

        Spine undulation: BackLink_Joint[0] and FrontLink_Joint[1] oscillate in opposition.
        """
        phase = np.linspace(0.0, 1.0, self._n, endpoint=False)
        omega = 2.0 * np.pi * phase

        LEG_OFFSET: float = 0.47  # rad  nominal standing pose for all leg joints (HexapI: legs positive)
        LEG_AMP: float = 0.30  # rad  peak swing amplitude above standing (positive = foot lifts)
        SPINE_AMP: float = 0.10  # rad  lateral spine oscillation

        ref = np.zeros((self._n, NUM_JOINTS), dtype=np.float32)

        # --- Spine joints ---
        ref[:, 0] = SPINE_AMP * np.sin(omega)  # BackLink_Joint
        ref[:, 1] = SPINE_AMP * np.sin(omega + np.pi)  # FrontLink_Joint (opposite)

        # --- Leg joints ---
        # Tripod A: sin > 0 during first half-cycle → leg lifts above standing pose
        ref[:, 3] = LEG_OFFSET + LEG_AMP * np.sin(omega)  # MiddleRight_Joint (A)
        ref[:, 4] = LEG_OFFSET + LEG_AMP * np.sin(omega)  # BackLeft_Joint    (A)
        ref[:, 6] = LEG_OFFSET + LEG_AMP * np.sin(omega)  # FrontLeft_Joint   (A)

        # Tripod B: opposite phase
        ref[:, 2] = LEG_OFFSET + LEG_AMP * np.sin(omega + np.pi)  # MiddleLeft_Joint  (B)
        ref[:, 5] = LEG_OFFSET + LEG_AMP * np.sin(omega + np.pi)  # BackRight_Joint   (B)
        ref[:, 7] = LEG_OFFSET + LEG_AMP * np.sin(omega + np.pi)  # FrontRight_Joint  (B)

        return ref
