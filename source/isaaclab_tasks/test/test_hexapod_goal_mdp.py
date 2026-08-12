# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch

HEXAPOD_DIR = Path(__file__).parents[1] / "isaaclab_tasks" / "contrib" / "velocity" / "config" / "hexapod"


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HEXAPOD_DIR / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


goal_rewards = load_module("hexapod_goal_rewards", "hexapod_goal_rewards.py")
goal_curriculum = load_module("hexapod_goal_curriculum", "hexapod_goal_curriculum.py")


class FakeCommandManager:
    def __init__(self, command: torch.Tensor):
        self.command = command
        self.term = SimpleNamespace(cfg=SimpleNamespace(ranges=SimpleNamespace(pos_x=(1.0, 1.0))))

    def get_command(self, _name: str) -> torch.Tensor:
        return self.command

    def get_term(self, _name: str):
        return self.term


class FakeTerminationManager:
    def __init__(self, terms: dict[str, torch.Tensor]):
        self.terms = terms

    def get_term(self, name: str) -> torch.Tensor:
        return self.terms[name]


class FakeEnv:
    def __init__(self):
        self.step_dt = 0.02
        self.device = "cpu"
        self.num_envs = 2
        self.episode_length_buf = torch.tensor([2, 2])
        self.command_manager = FakeCommandManager(torch.tensor([[5.0, 0.0, 0.0, 0.0], [5.0, 0.0, 0.0, 0.0]]))
        self.termination_manager = FakeTerminationManager(
            {
                "reach_goal": torch.tensor([True, False]),
                "base_contact": torch.tensor([False, True]),
            }
        )


def test_progress_reports_signed_speed_toward_goal():
    env = FakeEnv()
    torch.testing.assert_close(goal_rewards.progress_to_goal(env, "pose_command"), torch.zeros(2))
    env.episode_length_buf[:] = 3
    env.command_manager.command[:, 0] = torch.tensor([4.8, 5.2])
    torch.testing.assert_close(goal_rewards.progress_to_goal(env, "pose_command"), torch.tensor([10.0, -10.0]))


def test_progress_is_zero_on_first_step_after_individual_reset():
    env = FakeEnv()
    goal_rewards.progress_to_goal(env, "pose_command")
    env.episode_length_buf[:] = torch.tensor([1, 3])
    env.command_manager.command[:, 0] = torch.tensor([1.0, 4.8])
    torch.testing.assert_close(goal_rewards.progress_to_goal(env, "pose_command"), torch.tensor([0.0, 10.0]))


def test_terminal_signal_mirrors_named_termination_term():
    env = FakeEnv()
    torch.testing.assert_close(goal_rewards.termination_signal(env, "reach_goal"), torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(goal_rewards.termination_signal(env, "base_contact"), torch.tensor([0.0, 1.0]))


def test_time_decayed_termination_signal_decays_linearly_to_min_fraction():
    env = FakeEnv()
    env.episode_length_buf = torch.tensor([450, 2250])  # 9s and 45s elapsed at step_dt=0.02

    result = goal_rewards.time_decayed_termination_signal(
        env, "reach_goal", episode_length_s=45.0, min_fraction=0.5
    )

    torch.testing.assert_close(result, torch.tensor([0.9, 0.0]))


def test_time_decayed_termination_signal_clamps_past_episode_length():
    env = FakeEnv()
    env.episode_length_buf = torch.tensor([9000, 2])  # far past episode_length_s
    env.termination_manager = FakeTerminationManager(
        {"reach_goal": torch.tensor([True, True]), "base_contact": torch.tensor([False, False])}
    )

    result = goal_rewards.time_decayed_termination_signal(
        env, "reach_goal", episode_length_s=45.0, min_fraction=0.5
    )

    torch.testing.assert_close(result, torch.tensor([0.5, 1.0 - 0.5 * (0.04 / 45.0)]))


def make_curriculum_env(success_count: int, episode_count: int = 10) -> FakeEnv:
    env = FakeEnv()
    env.num_envs = episode_count
    env.episode_length_buf = torch.ones(episode_count, dtype=torch.long)
    success = torch.zeros(episode_count, dtype=torch.bool)
    success[:success_count] = True
    env.termination_manager = FakeTerminationManager(
        {
            "reach_goal": success,
            "base_contact": ~success,
        }
    )
    return env


def apply_curriculum(env: FakeEnv, window_size: int = 10, demotion_threshold: float | None = None):
    return goal_curriculum.goal_distance_curriculum(
        env,
        torch.arange(env.num_envs),
        command_name="pose_command",
        distances=(1.0, 2.0, 3.5, 5.0),
        success_threshold=0.7,
        window_sizes=(window_size, window_size, window_size, window_size),
        demotion_threshold=demotion_threshold,
    )


def test_curriculum_ignores_initial_resets():
    env = make_curriculum_env(success_count=0)
    env.episode_length_buf.zero_()

    state = apply_curriculum(env)

    assert state["episodes"] == 0.0
    assert state["distance"] == 1.0


def test_curriculum_requires_seventy_percent_success():
    env = make_curriculum_env(success_count=6)

    state = apply_curriculum(env)

    assert state["success_rate"] == 0.6
    assert state["distance"] == 1.0
    assert env.command_manager.term.cfg.ranges.pos_x == (1.0, 1.0)


def test_curriculum_advances_one_stage_and_clears_window():
    env = make_curriculum_env(success_count=7)

    state = apply_curriculum(env)

    assert state["success_rate"] == 0.7
    assert state["distance"] == 2.0
    assert env._goal_curriculum_episodes == 0
    assert env.command_manager.term.cfg.ranges.pos_x == (2.0, 2.0)


def test_curriculum_never_advances_past_five_meters():
    env = make_curriculum_env(success_count=10)

    for _ in range(5):
        state = apply_curriculum(env)

    assert state["distance"] == 5.0
    assert env._goal_curriculum_stage == 3


def test_curriculum_demotes_on_low_success_rate():
    env = make_curriculum_env(success_count=7)
    apply_curriculum(env, demotion_threshold=0.4)
    assert env._goal_curriculum_stage == 1

    env = make_curriculum_env(success_count=2)
    apply_curriculum(env, demotion_threshold=None)  # seed the ad hoc state attrs
    env._goal_curriculum_stage = 1
    state = apply_curriculum(env, demotion_threshold=0.4)

    assert state["success_rate"] == 0.2
    assert env._goal_curriculum_stage == 0
    assert state["distance"] == 1.0


def test_curriculum_does_not_demote_below_first_stage_or_without_threshold():
    env = make_curriculum_env(success_count=0)
    apply_curriculum(env, demotion_threshold=0.4)
    assert env._goal_curriculum_stage == 0

    env = make_curriculum_env(success_count=2)
    apply_curriculum(env, demotion_threshold=None)  # seed the ad hoc state attrs
    env._goal_curriculum_stage = 1
    apply_curriculum(env, demotion_threshold=None)
    assert env._goal_curriculum_stage == 1


def test_curriculum_uses_per_stage_window_size():
    env = make_curriculum_env(success_count=3, episode_count=5)
    state = goal_curriculum.goal_distance_curriculum(
        env,
        torch.arange(env.num_envs),
        command_name="pose_command",
        distances=(1.0, 2.0, 3.5, 5.0),
        success_threshold=0.7,
        window_sizes=(5, 10, 10, 10),
    )

    assert state["success_rate"] == 0.6
    assert env._goal_curriculum_episodes == 0  # stage-0 window of 5 already closed


def test_phase_one_source_configuration():
    env_source = (HEXAPOD_DIR / "hexapod_goal_env_cfg.py").read_text()
    obs_source = (HEXAPOD_DIR / "hexapod_goal_obs_cfg.py").read_text()
    agent_source = (HEXAPOD_DIR / "agents" / "rsl_rl_ppo_goal_cfg.py").read_text()

    for expected in (
        "GOAL_DISTANCES = (1.0, 2.0, 3.5, 5.0)",
        "EPISODE_LENGTH_S = 45.0",
        "CURRICULUM_DEMOTION_THRESHOLD = 0.4",
        "weight=10.0",
        "weight=2500.0",
        "weight=-1250.0",
        "weight=-1.0",
        "self.rewards.action_rate_l2.weight = -0.002",
        "self.rewards.feet_air_time.weight = 1.0",
        "self.rewards.dof_pos_limits.weight = -1.0",
        "self.curriculum.goal_distance = None",
    ):
        assert expected in env_source

    assert "class PolicyCfg" in obs_source and "self.enable_corruption = True" in obs_source
    assert "class CriticCfg" in obs_source and "self.enable_corruption = False" in obs_source
    assert "self.num_steps_per_env = 96" in agent_source
    assert "self.algorithm.gamma = 0.9995" in agent_source
    assert "self.algorithm.lam = 0.97" in agent_source
    assert "self.algorithm.entropy_coef = 0.003" in agent_source
    assert "self.policy.actor_obs_normalization = True" in agent_source
    assert "self.policy.critic_obs_normalization = True" in agent_source
