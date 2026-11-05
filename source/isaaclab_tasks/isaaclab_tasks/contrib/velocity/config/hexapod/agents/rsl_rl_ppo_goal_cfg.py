# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils.configclass import configclass

from .rsl_rl_ppo_cfg import HexapodRoughPPORunnerCfg


@configclass
class HexapodGoalPPORunnerCfg(HexapodRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 1000
        self.num_steps_per_env = 96
        self.experiment_name = "hexapod_goal"
        self.actor.hidden_dims = [128, 128, 128]
        self.critic.hidden_dims = [128, 128, 128]
        self.actor.obs_normalization = True
        self.critic.obs_normalization = True
        # Longer horizon than the 0.99/0.95 flat baseline, so the sparse terminal reach_bonus / fall_penalty
        # propagate back over the ~1250-step episode.
        self.algorithm.gamma = 0.9995
        self.algorithm.lam = 0.97
        # Below the 0.01 baseline: at 0.01 the action std ran away (>10), because the surrogate loss is too noisy
        # with sparse terminal rewards and a long horizon to counter the entropy bonus.
        self.algorithm.entropy_coef = 0.003
        # asymmetric observations: the actor sees "policy", the critic "critic"
        self.obs_groups = {"actor": ["policy"], "critic": ["critic"]}
