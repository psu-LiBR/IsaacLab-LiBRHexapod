# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO runner configuration for the hexapod goal + mimic task.

The network and observation space match the goal task, so a checkpoint can be resumed under
Isaac-Goal-Flat-Hexapod-v0.
"""

from isaaclab.utils.configclass import configclass

from .rsl_rl_ppo_goal_cfg import HexapodGoalPPORunnerCfg


@configclass
class HexapodGoalMimicPPORunnerCfg(HexapodGoalPPORunnerCfg):
    """Runner config for HexapodGoalMimicEnvCfg: flat-mimic PPO settings on the goal network, logged separately."""

    def __post_init__(self) -> None:
        super().__post_init__()

        self.experiment_name = "hexapod_goal_mimic"
        self.save_interval = 50
        # The goal runner's gamma 0.9995 / lam 0.97 / entropy 0.003 / 96 steps suit sparse terminal rewards, not a dense
        # per-step imitation reward: use the flat/mimic values (0.99 / 0.95 / 0.01 / 48).
        self.num_steps_per_env = 48
        self.algorithm.gamma = 0.99
        self.algorithm.lam = 0.95
        self.algorithm.entropy_coef = 0.01
        # imitation gives a dense signal: start from a narrower action distribution than the inherited 1.0
        self.max_iterations = 1500
        self.actor.distribution_cfg.init_std = 0.6
