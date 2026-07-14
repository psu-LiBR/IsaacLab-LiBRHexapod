# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run an environment with zero action agent and direct MP4 video recording."""

import argparse
import os
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Zero agent for Isaac Lab environments with MP4 recording.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Sand-Hexapod-Play-v0", help="Name of the task.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import carb
import gymnasium as gym
import torch
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

# 关键修复：在创建环境之前就启用 rtx_sensors
carb.settings.get_settings().set_bool("/isaaclab/render/rtx_sensors", True)

def main():
    """Zero actions agent that records the real robot rendered in the sand play environment."""
    task_name = args_cli.task or "Isaac-Velocity-Sand-Hexapod-Play-v0"

    # 1. 解析基础配置
    env_cfg = parse_env_cfg(
        task_name, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    
    # 关键：启用外部力！
    env_cfg.sim.physx.enable_external_forces_every_iteration = True

    # 设置相机视角（参考 play_sand_headless.py 的黄金视角）
    if hasattr(env_cfg, "viewer"):
        env_cfg.viewer.eye = [-1.5, 0.5, 0.7]
        env_cfg.viewer.lookat = [0.0, 0.0, 0.1]
        env_cfg.viewer.origin_type = "asset_root"
        env_cfg.viewer.asset_name = "robot"

    # 2. 使用 gym.make 创建环境并启用 rgb_array 渲染模式
    base_env = gym.make(task_name, cfg=env_cfg, render_mode="rgb_array")
    print(f"[INFO]: 使用 gym.make 创建环境: {task_name}")

    print(f"[INFO]: Gym observation space: {base_env.observation_space}")
    print(f"[INFO]: Gym action space: {base_env.action_space}")

    # 明确指定视频输出路径
    video_output_dir = "/home/ubuntu/IsaacLab-LiBRHexapod/my_videos"
    os.makedirs(video_output_dir, exist_ok=True)

    # 使用 Gym 的 RecordVideo 包装器录制视频
    env = gym.wrappers.RecordVideo(
        env=base_env,
        video_folder=video_output_dir,
        step_trigger=lambda step: step == 0,
        video_length=300,
        name_prefix="hexapod_sand_demo"
    )

    env.reset()

    sim_steps = 0
    print("[INFO]: 正在录制真实沙地环境中的机器人 RGB 帧...")
    print("[INFO]: 机器人将执行波浪式前进步态（左右交替摆动），向前行走...")
    print("[INFO]: RFT阻力已在环境中启用！")
    
    # 记录初始位置
    initial_pos = None
    final_pos = None
    HEXAPOD_BODY_LENGTH = 0.5  # 估计六足机器人的体长（米）

    while simulation_app.is_running():
        with torch.inference_mode():
            # 波浪式前进步态（左侧和右侧有相位差，让机器人真正向前移动）
            t = sim_steps * 0.02  # 时间，dt=0.02
            gait_period = 0.65    # 步态周期（秒）
            # 创建 Tensor 类型的 omega
            device = env.unwrapped.device
            omega = torch.tensor(2.0 * torch.pi * (t % gait_period) / gait_period, device=device)
            pi_tensor = torch.tensor(torch.pi, device=device)
            
            actions = torch.zeros(env.action_space.shape, device=device)
            
            # 步态参数
            LEG_OFFSET = -0.47    # 腿部关节默认位置（rad）
            LEG_AMP = 0.50        # 更大的腿部摆动幅度
            SPINE_AMP = 0.20      # 更大的脊柱摆动幅度
            ACTION_SCALE = 0.5    # 动作缩放因子（与环境配置一致）
            
            # 计算目标关节位置（绝对角度）
            q_target = torch.zeros(8, device=device)
            
            # 脊柱关节（反向摆动，推动前进）
            q_target[0] = SPINE_AMP * torch.sin(omega)          # BackLink
            q_target[1] = SPINE_AMP * torch.sin(omega + pi_tensor) # FrontLink (反向)
            
            # 左侧腿（MiddleLeft[2], BackLeft[4], FrontLeft[6]）- 相位 A
            q_target[2] = LEG_OFFSET + LEG_AMP * torch.sin(omega)               # MiddleLeft
            q_target[4] = LEG_OFFSET + LEG_AMP * torch.sin(omega + 0.2*pi_tensor) # BackLeft
            q_target[6] = LEG_OFFSET + LEG_AMP * torch.sin(omega - 0.2*pi_tensor) # FrontLeft
            
            # 右侧腿（MiddleRight[3], BackRight[5], FrontRight[7]）- 相位 B（反相）
            q_target[3] = LEG_OFFSET + LEG_AMP * torch.sin(omega + pi_tensor)    # MiddleRight
            q_target[5] = LEG_OFFSET + LEG_AMP * torch.sin(omega + 1.2*pi_tensor) # BackRight
            q_target[7] = LEG_OFFSET + LEG_AMP * torch.sin(omega + 0.8*pi_tensor) # FrontRight
            
            # 默认关节位置
            q_default = torch.tensor([0.0, 0.0, LEG_OFFSET, LEG_OFFSET, LEG_OFFSET, LEG_OFFSET, LEG_OFFSET, LEG_OFFSET], 
                                    device=device)
            
            # 转换为动作空间：action = (q_target - q_default) / ACTION_SCALE
            actions[0, :] = (q_target - q_default) / ACTION_SCALE
            
            # 获取初始位置
            if sim_steps == 0:
                initial_pos = env.unwrapped.scene["robot"].data.root_pos_w[0].clone()
            
            env.step(actions)
            
            # 获取最终位置
            if sim_steps == 299:
                final_pos = env.unwrapped.scene["robot"].data.root_pos_w[0].clone()

            sim_steps += 1
            if sim_steps >= 300:
                print(f"[INFO]: 录制成功完成！MP4 已保存到: {video_output_dir}")
                
                # 计算并输出移动距离
                if initial_pos is not None and final_pos is not None:
                    # 计算 x 方向移动距离
                    distance_x = abs(final_pos[0] - initial_pos[0])
                    # 计算二维平面移动距离
                    distance_2d = torch.sqrt((final_pos[0] - initial_pos[0])**2 + 
                                             (final_pos[1] - initial_pos[1])**2)
                    # 计算移动了多少个体长
                    body_lengths = distance_2d / HEXAPOD_BODY_LENGTH
                    
                    print("\n" + "="*60)
                    print("沙地环境（zero_agent1）移动统计")
                    print("="*60)
                    print(f"总模拟时间：300步 = 6.0秒")
                    print(f"初始位置：x={initial_pos[0]:.3f}m, y={initial_pos[1]:.3f}m")
                    print(f"最终位置：x={final_pos[0]:.3f}m, y={final_pos[1]:.3f}m")
                    print(f"x方向移动距离：{distance_x:.3f}米")
                    print(f"二维平面总移动距离：{distance_2d:.3f}米")
                    print(f"相当于移动了：{body_lengths:.1f}个体长（体长≈{HEXAPOD_BODY_LENGTH}米）")
                    print("="*60 + "\n")
                
                break

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
