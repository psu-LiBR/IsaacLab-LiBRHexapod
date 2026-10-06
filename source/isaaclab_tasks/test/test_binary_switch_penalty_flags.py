# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CPU tests for the contact-switch penalty flags shared by the binary-RL training scripts (no Isaac Sim)."""

import argparse
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
managers = pytest.importorskip("isaaclab.managers")

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts/reinforcement_learning/binary_rl"
sys.path.insert(0, str(SCRIPTS))
import binary_common as bc  # noqa: E402
from switch_command_reward import action_switch_count  # noqa: E402

CURRENT_ACTION_RATE_WEIGHT = -5.0e-4  # HexapodGoalEnvCfg


def _cfg(action_rate_weight=CURRENT_ACTION_RATE_WEIGHT):
    term = None
    if action_rate_weight is not None:
        term = managers.RewardTermCfg(func=lambda env: 0.0, weight=action_rate_weight)
    return SimpleNamespace(rewards=SimpleNamespace(action_rate_l2=term))


def _args(argv=()):
    parser = argparse.ArgumentParser()
    bc.add_common_cli(parser)
    return parser.parse_args(list(argv))


def test_defaults_leave_the_env_rewards_untouched():
    args = _args()
    assert args.action_rate_multiplier == 1.0
    assert args.action_switch_penalty == 0.0
    cfg = _cfg()
    bc.scale_action_rate_weight(cfg, args.action_rate_multiplier)
    bc.add_action_switch_penalty(cfg, args.action_switch_penalty)
    assert cfg.rewards.action_rate_l2.weight == CURRENT_ACTION_RATE_WEIGHT
    assert not hasattr(cfg.rewards, "action_switch_count")


def test_action_rate_multiplier_scales_the_env_weight():
    args = _args(["--action_rate_multiplier", "100"])
    cfg = _cfg()
    bc.scale_action_rate_weight(cfg, args.action_rate_multiplier)
    # 100x the current -5e-4 is the -0.05 that Robin's sweep called "10x" (her reference was -0.005).
    assert cfg.rewards.action_rate_l2.weight == pytest.approx(-0.05)


def test_action_switch_penalty_adds_the_term_only_when_requested():
    cfg = _cfg()
    bc.add_action_switch_penalty(cfg, _args(["--action_switch_penalty", "0.004"]).action_switch_penalty)
    term = cfg.rewards.action_switch_count
    assert term.func is action_switch_count
    assert term.weight == pytest.approx(-0.004)


def test_invalid_values_are_rejected():
    with pytest.raises(ValueError, match="action_rate_multiplier"):
        bc.scale_action_rate_weight(_cfg(), -1.0)
    with pytest.raises(ValueError, match="action_switch_penalty"):
        bc.add_action_switch_penalty(_cfg(), -0.1)
    with pytest.raises(ValueError, match="no action_rate_l2"):
        bc.scale_action_rate_weight(_cfg(action_rate_weight=None), 10.0)


def test_run_meta_records_the_resolved_weights():
    cfg = _cfg()
    args = _args(["--action_rate_multiplier", "100", "--action_switch_penalty", "0.004"])
    bc.scale_action_rate_weight(cfg, args.action_rate_multiplier)
    bc.add_action_switch_penalty(cfg, args.action_switch_penalty)
    env = SimpleNamespace(unwrapped=SimpleNamespace(cfg=cfg))
    meta = bc.switch_penalty_meta(env, args)
    assert meta["action_rate_l2_weight"] == pytest.approx(-0.05)
    assert meta["action_switch_count_weight"] == pytest.approx(-0.004)
    assert meta["action_rate_multiplier"] == 100.0
    assert meta["action_switch_penalty"] == 0.004


@pytest.mark.parametrize("script", ["train_discrete.py", "train_discrete_ppo.py", "train_sac_d.py"])
def test_discrete_training_scripts_forward_both_flags_to_build_env(script):
    source = (SCRIPTS / script).read_text()
    call = re.search(r"env = build_env\((.*?)\n\)", source, re.S).group(1)
    assert "action_rate_multiplier=args.action_rate_multiplier" in call
    assert "action_switch_penalty=args.action_switch_penalty" in call
    assert "**switch_penalty_meta(env, args)" in source


def test_continuous_sac_applies_the_multiplier_and_rejects_the_switch_penalty():
    source = (SCRIPTS / "train_sac_continuous.py").read_text()
    assert "scale_action_rate_weight(env_cfg, args.action_rate_multiplier)" in source
    assert "parser.error(" in source


def test_pipeline_forwards_both_flags_to_every_trainer():
    source = (SCRIPTS / "run_discrete_pipeline.py").read_text()
    assert '"--action_rate_multiplier"' in source and '"--action_switch_penalty"' in source
