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

        self.max_iterations = 3000
        self.num_steps_per_env = 96
        self.experiment_name = "hexapod_goal"
        self.policy.actor_hidden_dims = [128, 128, 128]
        self.policy.critic_hidden_dims = [128, 128, 128]
        self.policy.actor_obs_normalization = True
        self.policy.critic_obs_normalization = True
        # gamma/lam raised further above the 0.99/0.95 rough/flat baseline: the terminal
        # reach_bonus/fall_penalty are an order of magnitude larger than any per-step term,
        # and at the old gamma=0.999 they were already discounted to ~10% of value by early
        # episode states (0.999**2250 ~= 0.10 over the ~2250-step, 45s episode) -- pushing
        # both further out extends the horizon over which that terminal credit propagates.
        self.algorithm.gamma = 0.9995
        self.algorithm.lam = 0.97
        # Raising this to the rough/flat baseline of 0.01 caused runaway action std (>10):
        # the entropy term in PPO's loss (-entropy_coef * entropy.mean()) pulls std up
        # unconditionally every step, and the only counterweight is the surrogate loss's
        # advantage-driven pull downward -- which is unusually noisy here because of the
        # sparse +-2500/-1250 terminal rewards (reach_bonus/fall_penalty) combined with the
        # raised gamma/lam above. Kept below the rough/flat baseline to keep that pull weak
        # enough for the surrogate loss to counteract it reliably.
        self.algorithm.entropy_coef = 0.003
        # Route asymmetric observations: actor sees "policy" group, critic sees "critic" group
        self.obs_groups = {"policy": ["policy"], "critic": ["critic"]}
