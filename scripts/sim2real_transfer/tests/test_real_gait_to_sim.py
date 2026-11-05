# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import ast
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

_TOOLS_DIR = os.path.join(os.path.dirname(__file__), "..", "tools")
sys.path.insert(0, _TOOLS_DIR)
import real_gait_to_sim as rg  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
TPR = 4096 / (2 * math.pi)
A_SPINE = 0.9162978573
REAL_FILE = Path(
    r"C:\Users\jrh6552\OneDrive - The Pennsylvania State University\LiBR PSU - Documents\Hexapod\Data\Hexapi"
    r"\Robot Opt Gaits\robot24_antislip\noise0\robot24_antislip_noise0_MD1_K3_r01_011011-110101-111010.csv"
)


def _tripod_gait(n=50):
    t = np.arange(n) / n
    body = np.stack([A_SPINE * np.sin(2 * np.pi * t), A_SPINE * np.cos(2 * np.pi * t)], axis=1)
    half = (np.arange(n) < n // 2).astype(int)
    contact = np.stack([half, 1 - half, 1 - half, half, half, 1 - half], axis=1)
    return body, contact


def _nominal():
    return rg.build_mapping(rg.DEFAULT_CONFIG, "nominal")


def _nominal_rad(body, contact, **kw):
    return rg.ticks_to_mimic_rad(rg.real_gait_to_ticks(body, contact, **kw), _nominal())


def test_mapping_round_trip():
    mapping = rg.build_mapping(rg.DEFAULT_CONFIG, "calibrated")
    lim = mapping.soft_limits_sim
    rng = np.random.default_rng(0)
    for _ in range(20):
        sim = lim[:, 0] + rng.random(len(lim)) * (lim[:, 1] - lim[:, 0])
        back = mapping.ticks_to_sim_rad(mapping.sim_target_to_ticks(sim))
        assert np.allclose(back, sim, atol=math.pi / 4096 + 1e-12, rtol=0)
        assert np.allclose(mapping.real_rad_to_sim_rad(mapping.sim_rad_to_real_rad(sim)), sim, atol=1e-12, rtol=0)


def test_real_gait_round_trip():
    body, contact = _tripod_gait()
    out = _nominal_rad(body, contact, round_ticks=True)
    col = {n: i for i, n in enumerate(rg.MIMIC_JOINT_ORDER)}
    tol = 2.0 / TPR
    assert np.allclose(out[:, col["FrontLink_Joint"]], body[:, 0], atol=tol, rtol=0)
    assert np.allclose(out[:, col["BackLink_Joint"]], body[:, 1], atol=tol, rtol=0)
    mid = 0.5 * (0.460194236365692 + 1.5325)
    for i, name in enumerate(rg.CONTACT_JOINTS):
        lifted = out[:, col[f"{name}_Joint"]] > mid
        assert np.array_equal(lifted, contact[:, i] == 0)


def test_nominal_constants():
    body, contact = _tripod_gait()
    out = _nominal_rad(body, contact, air_enc=1278.0)
    legs = out[:, 2:]
    assert np.isclose(legs.min(), 0.460194236365692, atol=1e-9)
    assert np.isclose(legs.max(), 1.180398216278, atol=1e-9)
    default = _nominal_rad(body, contact)
    assert np.isclose(default[:, 2:].max(), 999 / TPR, atol=1e-9)
    assert abs(999 / TPR - 1.5325) < 1e-4


def test_column_order_matches_mimic_motion():
    src = REPO_ROOT / ("source/isaaclab_tasks/isaaclab_tasks/contrib/velocity/config/hexapod/hexapod_mimic_motion.py")
    tree = ast.parse(src.read_text(encoding="utf-8"))
    names = None
    for node in tree.body:
        target = (
            node.target
            if isinstance(node, ast.AnnAssign)
            else (node.targets[0] if isinstance(node, ast.Assign) else None)
        )
        if isinstance(target, ast.Name) and target.id == "JOINT_NAMES":
            names = ast.literal_eval(node.value)
    assert names is not None
    assert names == rg.MIMIC_JOINT_ORDER


def test_contact_mapping_single_leg():
    n = 10
    body = np.zeros((n, 2))
    contact = np.ones((n, 6), dtype=int)
    contact[:, 0] = 0  # FR in the air
    out = _nominal_rad(body, contact)
    stance = 0.460194236365692
    assert np.allclose(out[:, :7][:, 2:], stance, atol=1e-9)
    assert np.allclose(out[:, 7], 999 / TPR, atol=1e-9)
    assert np.allclose(out[:, :2], 0.0, atol=1e-9)


def test_rotations():
    body, contact = _tripod_gait(50)
    base = _nominal_rad(body, contact)
    shifted = _nominal_rad(body, contact, shiftnum=7)
    assert np.allclose(shifted[:, 2:], base[:, 2:])
    assert np.allclose(shifted[:, :2], np.roll(base[:, :2], 7, axis=0))
    phase = _nominal_rad(body, contact, phase_offset_deg=180.0)
    assert np.allclose(phase, np.roll(base, 25, axis=0))


def test_cli_end_to_end(tmp_path, capsys):
    body, contact = _tripod_gait(50)
    src = tmp_path / "gaits"
    (src / "noise0").mkdir(parents=True)
    rows = np.hstack([body, contact.astype(float)])
    with open(src / "noise0" / "g1.csv", "w") as f:
        f.write("# header line\n# contact order: foo\n")
        np.savetxt(f, rows, delimiter=",", fmt="%.10g")
    (src / "_gait_index_MD1.csv").write_text("a,b,c\n1,2,3\n")
    out_dir = tmp_path / "out"

    assert rg.main([str(src), "--out", str(out_dir)]) == 0
    outs = list(out_dir.rglob("*.csv"))
    assert [o.name for o in outs] == ["g1_hexapi_sim.csv"]
    assert outs[0].parent.name == "noise0"
    text = outs[0].read_text()
    assert "\r" not in text and not text.startswith("#")
    arr = np.loadtxt(outs[0], delimiter=",")
    assert arr.shape == (50, 8)

    assert rg.main([str(src), "--out", str(out_dir)]) != 0  # overwrite refused
    assert rg.main([str(src), "--out", str(out_dir), "--overwrite"]) == 0
    assert rg.main([str(tmp_path / "missing.csv")]) != 0
    capsys.readouterr()


def test_bad_contacts_rejected(tmp_path):
    p = tmp_path / "bad.csv"
    np.savetxt(p, np.full((4, 8), 0.5), delimiter=",")
    with pytest.raises(ValueError):
        rg.load_real_gait(p)


@pytest.mark.skipif(not REAL_FILE.exists(), reason="real robot gait file not available")
def test_real_file_conversion(tmp_path):
    arr = rg.convert_file(REAL_FILE, tmp_path / "o.csv", _nominal())
    assert arr.shape == (50, 8)
    for j in range(2, 8):
        assert len(np.unique(np.round(arr[:, j], 9))) <= 2
