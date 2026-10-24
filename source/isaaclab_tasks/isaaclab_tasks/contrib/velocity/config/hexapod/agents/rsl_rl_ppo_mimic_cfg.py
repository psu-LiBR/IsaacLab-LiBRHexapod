# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO runner configuration for the hexapod mimic + RL task.

The network and observation space match the flat task, so a checkpoint can be resumed under
Isaac-Velocity-Flat-Hexapod-v0.
"""

from isaaclab.utils.configclass import configclass

from .rsl_rl_ppo_cfg import HexapodFlatPPORunnerCfg


@configclass
class HexapodMimicPPORunnerCfg(HexapodFlatPPORunnerCfg):
    """Runner config for HexapodMimicEnvCfg: 3000 iterations (800 imitation + 2200 RL), logged separately."""

    def __post_init__(self) -> None:
        super().__post_init__()

        self.max_iterations = 3000
        self.experiment_name = "hexapod_mimic"
        self.save_interval = 50
