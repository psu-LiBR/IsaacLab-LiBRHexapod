# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
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



# =========================================================================
# 🚀 注册沙地训练环境 (Hexapod Sand Train v0)
# =========================================================================
gym.register(
    id="Isaac-Velocity-Sand-Hexapod-v0",
    entry_point="isaaclab_tasks.manager_based.locomotion.velocity.config.hexapod.hexapod_sand_train_env:HexapodSandTrainEnv",
    disable_env_checker=True,
    kwargs={
        # 📌【核心修正】：训练阶段的配置入口，指向真正的沙地配置类
        "env_cfg_entry_point": f"{__name__}.sand_env_cfg:HexapodSandEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HexapodFlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
    },
)

# =========================================================================
# 🎥 注册沙地成果演示环境 (Hexapod Sand Play/Demo v0)
# =========================================================================
gym.register(
    id="Isaac-Velocity-Sand-Hexapod-Play-v0",
    entry_point="isaaclab_tasks.manager_based.locomotion.velocity.config.hexapod.hexapod_sand_play_env:HexapodSandPlayEnv",
    disable_env_checker=True,
    kwargs={
        # 📌【核心修正】：成果演示的配置入口，必须指向 sand_env_cfg 文件中的 HexapodSandEnvCfg_PLAY 参数类！
        "env_cfg_entry_point": f"{__name__}.sand_env_cfg:HexapodSandEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HexapodFlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
    },
)