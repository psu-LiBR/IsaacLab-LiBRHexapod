# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Velocity-Flat-Hexapod-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:HexapodFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HexapodFlatPPORunnerCfg",
        #"rsl_rl_with_symmetry_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HexapodFlatPPORunnerWithSymmetryCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-Hexapod-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:HexapodFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HexapodFlatPPORunnerCfg",
        #"rsl_rl_with_symmetry_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HexapodFlatPPORunnerWithSymmetryCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Hexapod-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:HexapodRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HexapodRoughPPORunnerCfg",
        #"rsl_rl_with_symmetry_cfg_entry_point": (
        #    f"{agents.__name__}.rsl_rl_ppo_cfg:HexapodRoughPPORunnerWithSymmetryCfg"
        #),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Hexapod-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:HexapodRoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HexapodRoughPPORunnerCfg",
        #"rsl_rl_with_symmetry_cfg_entry_point": (
        #    f"{agents.__name__}.rsl_rl_ppo_cfg:HexapodRoughPPORunnerWithSymmetryCfg"
        #),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
    },
)

# ---------------------------------------------------------------------------
# Mimic + RL environments
# ---------------------------------------------------------------------------

gym.register(
    id="Isaac-Velocity-Flat-Hexapod-Mimic-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hexapod_mimic_env_cfg:HexapodMimicEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_mimic_cfg:HexapodMimicPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-Hexapod-Mimic-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hexapod_mimic_env_cfg:HexapodMimicEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_mimic_cfg:HexapodMimicPPORunnerCfg",
    },
)

# ---------------------------------------------------------------------------
# Goal-reaching environment (reach 5m forward, as fast as possible)
# ---------------------------------------------------------------------------

gym.register(
    id="Isaac-Goal-Flat-Hexapod-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hexapod_goal_env_cfg:HexapodGoalEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_goal_cfg:HexapodGoalPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Goal-Flat-Hexapod-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hexapod_goal_env_cfg:HexapodGoalEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_goal_cfg:HexapodGoalPPORunnerCfg",
    },
)

##
# Reward-tuning variants (Robin, 2026-07). Flat: foot-slide weight scan on top of
# the reward-shaping config. Goal: the same shaping carried to the goal-reaching task.
# See REWARD_SHAPING.md for what each arm changes and what was measured.
##

gym.register(
    id="Isaac-Velocity-Flat-Hexapod-Rshape-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg_rshape:HexapodFlatRshapeEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HexapodFlatPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-Hexapod-Rshape-Slide045-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg_rshape:HexapodFlatRshapeSlide045EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HexapodFlatPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-Hexapod-Rshape-Slide035-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg_rshape:HexapodFlatRshapeSlide035EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HexapodFlatPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-Hexapod-Rshape-Slide025-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg_rshape:HexapodFlatRshapeSlide025EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HexapodFlatPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-Hexapod-Rshape-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg_rshape:HexapodFlatRshapeEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HexapodFlatPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Goal-Flat-Hexapod-BigStep-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hexapod_goal_tuned_env_cfg:HexapodGoalBigStepEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_goal_cfg:HexapodGoalPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Goal-Flat-Hexapod-BigStep-Slide06-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hexapod_goal_tuned_env_cfg:HexapodGoalBigStepSlide06EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_goal_cfg:HexapodGoalPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Goal-Flat-Hexapod-BigStep-Slide035-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hexapod_goal_tuned_env_cfg:HexapodGoalBigStepSlide035EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_goal_cfg:HexapodGoalPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Goal-Flat-Hexapod-BigStep-Minimal-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hexapod_goal_tuned_env_cfg:HexapodGoalBigStepMinimalEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_goal_cfg:HexapodGoalPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Goal-Flat-Hexapod-BigStep-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hexapod_goal_tuned_env_cfg:HexapodGoalBigStepEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_goal_cfg:HexapodGoalPPORunnerCfg",
    },
)

# ---------------------------------------------------------------------------
# Binary (contact-bit) action-space variant of the goal-reaching environment.
# Action-space swap only: 6 leg bits (RL) + scripted spine sinusoid.
# No RL library config wired up yet (deliberate -- algorithm choice pending).
# ---------------------------------------------------------------------------

gym.register(
    id="Isaac-Goal-Flat-Hexapod-Binary-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hexapod_binary_env_cfg:HexapodBinaryEnvCfg",
    },
)

gym.register(
    id="Isaac-Goal-Flat-Hexapod-Binary-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hexapod_binary_env_cfg:HexapodBinaryEnvCfg_PLAY",
    },
)
