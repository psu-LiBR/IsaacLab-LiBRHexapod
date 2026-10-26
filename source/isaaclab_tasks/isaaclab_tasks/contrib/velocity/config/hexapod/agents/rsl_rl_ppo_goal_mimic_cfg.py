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
    """Runner config for HexapodGoalMimicEnvCfg: goal PPO settings, logged separately."""

    def __post_init__(self) -> None:
        super().__post_init__()

        self.experiment_name = "hexapod_goal_mimic"
        self.save_interval = 50
