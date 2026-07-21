# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""PPO runner configuration for the hexapod mimic + RL task.

Total training is split into two logical phases controlled by the curriculum:
  Phase 1  (iterations 0 … MIMIC_ITERATIONS-1):   imitation reward active.
  Phase 2  (iterations MIMIC_ITERATIONS … end):    pure RL rewards only.

The network architecture and observation space are identical to the standard
flat env, so a checkpoint from this task can be resumed directly under
Isaac-Velocity-Flat-Hexapod-v0 for further RL fine-tuning.
"""

from isaaclab.utils.configclass import configclass

from .rsl_rl_ppo_cfg import HexapodFlatPPORunnerCfg


@configclass
class HexapodMimicPPORunnerCfg(HexapodFlatPPORunnerCfg):
    """Runner config for HexapodMimicEnvCfg.

    Extends the flat-env config with:
      - A longer max_iterations to cover both the imitation and RL phases.
        Default: 800 mimic + 2200 RL = 3000 total iterations.
      - A distinctive experiment_name so logs land in a separate directory.
    """

    def __post_init__(self) -> None:
        super().__post_init__()

        self.max_iterations = 3000
        self.experiment_name = "hexapod_mimic"
        self.save_interval = 50

        # init_noise_std and entropy_coef left at parent defaults (1.0 and 0.01)
        # to match the May 2026 working run configuration.
