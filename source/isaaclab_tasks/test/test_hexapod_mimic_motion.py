# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

HEXAPOD_DIR = Path(__file__).parents[1] / "isaaclab_tasks" / "contrib" / "velocity" / "config" / "hexapod"
REPO_ROOT = Path(__file__).parents[3]
GAIT_CSV = (
    REPO_ROOT / "hexapod-assets" / "Sim Gaits" / "SMOOTH_vertical_K3_NoNoise_rank1_lleg30-93_amp70_res50_hexapi_sim.csv"
)

# HexapiFlattened.usd leg limits are [-5, 110] deg
HEXAPI_LEG_LIMITS = (np.deg2rad(-5.0), np.deg2rad(110.0))
LEG_COLS = slice(2, 8)
CSV_COL_ORDER = [1, 0, 2, 3, 6, 7, 4, 5]  # MIMIC_CSV_COL_ORDER: asset order -> sim DOF order


def _load_motion_module():
    spec = importlib.util.spec_from_file_location("hexapod_mimic_motion", HEXAPOD_DIR / "hexapod_mimic_motion.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


motion = _load_motion_module()


def test_hexapi_csv_is_used_as_is_within_hexapi_leg_limits():
    ref = motion.MotionReference(str(GAIT_CSV), csv_col_order=CSV_COL_ORDER)._ref_np
    assert ref[:, LEG_COLS].min() >= HEXAPI_LEG_LIMITS[0]
    assert ref[:, LEG_COLS].max() <= HEXAPI_LEG_LIMITS[1]
    assert ref[:, LEG_COLS].mean() > 0.0  # positive-leg convention; a negated load would be < 0


def test_csv_col_order_moves_asset_order_to_sim_dof_order():
    raw = motion.MotionReference(str(GAIT_CSV))._ref_np
    reordered = motion.MotionReference(str(GAIT_CSV), csv_col_order=CSV_COL_ORDER)._ref_np
    np.testing.assert_allclose(reordered, raw[:, CSV_COL_ORDER])


def test_builtin_gait_stands_in_hexapi_convention():
    ref = motion.MotionReference(None)._ref_np
    np.testing.assert_allclose(ref[:, LEG_COLS].mean(), 0.47, atol=1e-3)
    assert ref[:, LEG_COLS].min() >= HEXAPI_LEG_LIMITS[0]
    assert ref[:, LEG_COLS].max() <= HEXAPI_LEG_LIMITS[1]
