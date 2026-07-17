
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from .flat_env_cfg import HexapodFlatEnvCfg


@configclass
class HexapodSandImprovedEnvCfg(HexapodFlatEnvCfg):
    def __post_init__(self) -> None:
        # 直接继承最初的平地环境配置
        super().__post_init__()

        # 使用generator类型地形，只生成平坦地形，这样可以使用大理石材质
        self.scene.terrain.terrain_type = "generator"
        # 修改terrain_generator，只包含平坦地形
        self.scene.terrain.terrain_generator = ROUGH_TERRAINS_CFG.replace(
            curriculum=False,
            sub_terrains={
                "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0)
            },
            num_rows=1,
            num_cols=1,
        )
        # 不设置terrain_generator为None了
        self.scene.height_scanner = None
        self.curriculum.terrain_levels = None
        
        # 设置大理石白色瓷砖材质，类似参考图片
        self.scene.terrain.visual_material = sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        )

        # 沙地的物理材质属性（摩擦力系数范围）
        self.events.physics_material.params["static_friction_range"] = (0.35, 0.45)
        self.events.physics_material.params["dynamic_friction_range"] = (0.22, 0.32)

        # 关键：启用外部力！这样我们的RFT力才能被物理引擎接收
        self.sim.physx.enable_external_forces_every_iteration = True

        # 保持奖励函数参数对齐
        self.rewards.feet_air_time.params["threshold"] = 0.08
        self.rewards.dof_torques_l2.weight = -6.0e-5
        self.rewards.action_rate_l2.weight = -3.0e-2


@configclass
class HexapodSandImprovedEnvCfg_PLAY(HexapodSandImprovedEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        # 单环境成果演示的 Play 模式基础配置
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None

        # 设定单向行走测试速度
        self.commands.base_velocity.ranges.lin_vel_x = (0.16, 0.16)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.resampling_time_range = (1000.0, 1000.0)

        # 基础视窗视角
        self.viewer.eye = (2.5, 2.5, 2.5)
        self.viewer.lookat = (0.0, 0.0, 0.0)

