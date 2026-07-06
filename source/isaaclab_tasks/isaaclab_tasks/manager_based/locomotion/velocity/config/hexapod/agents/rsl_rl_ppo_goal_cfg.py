# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from .rsl_rl_ppo_cfg import HexapodRoughPPORunnerCfg


@configclass
class HexapodGoalPPORunnerCfg(HexapodRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 3000
        self.num_steps_per_env = 96
        self.experiment_name = "hexapod_goal"
        self.policy.actor_hidden_dims = [128, 128, 128]
        self.policy.critic_hidden_dims = [128, 128, 128]
        self.policy.actor_obs_normalization = True
        self.policy.critic_obs_normalization = True
        self.algorithm.gamma = 0.999
        self.algorithm.entropy_coef = 0.003
        # Route asymmetric observations: actor sees "policy" group, critic sees "critic" group
        self.obs_groups = {"policy": ["policy"], "critic": ["critic"]}
