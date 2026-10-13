# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

HEXAPOD_DIR = Path(__file__).parents[1] / "isaaclab_tasks" / "contrib" / "velocity" / "config" / "hexapod"
REPO_ROOT = Path(__file__).parents[3]
GAIT_CSV = REPO_ROOT / "hexapod-assets" / "Sim Gaits" / "forward3_lleg35_amp65_sim.csv"

# HexapiFlattened.usd leg limits are [-5, 110] deg
HEXAPI_LEG_LIMITS = (np.deg2rad(-5.0), np.deg2rad(110.0))
LEG_COLS = slice(2, 8)


def _load_motion_module():
    spec = importlib.util.spec_from_file_location("hexapod_mimic_motion", HEXAPOD_DIR / "hexapod_mimic_motion.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


motion = _load_motion_module()


def test_legacy_csv_is_negated_into_hexapi_leg_limits():
    ref = motion.MotionReference(str(GAIT_CSV), csv_convention="legacy")._ref_np
    assert ref[:, LEG_COLS].min() >= HEXAPI_LEG_LIMITS[0]
    assert ref[:, LEG_COLS].max() <= HEXAPI_LEG_LIMITS[1]
    assert ref[:, LEG_COLS].mean() > 0.0


def test_hexapi_convention_uses_csv_as_is():
    legacy = motion.MotionReference(str(GAIT_CSV), csv_convention="legacy")._ref_np
    as_is = motion.MotionReference(str(GAIT_CSV), csv_convention="hexapi")._ref_np
    np.testing.assert_allclose(legacy, -as_is)
    assert as_is[:, LEG_COLS].max() < 0.0  # the legacy legs read raw are outside the HexapI limits


def test_builtin_gait_stands_in_hexapi_convention():
    ref = motion.MotionReference(None)._ref_np
    np.testing.assert_allclose(ref[:, LEG_COLS].mean(), 0.47, atol=1e-3)
    assert ref[:, LEG_COLS].min() >= HEXAPI_LEG_LIMITS[0]
    assert ref[:, LEG_COLS].max() <= HEXAPI_LEG_LIMITS[1]


def test_unknown_convention_is_rejected():
    with pytest.raises(ValueError, match="csv_convention"):
        motion.MotionReference(None, csv_convention="bogus")
