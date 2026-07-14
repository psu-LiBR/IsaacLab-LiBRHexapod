# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run an environment with zero action agent and direct MP4 video recording on Sand Terrain."""

import argparse
import os
import gymnasium as gym
import torch
from isaaclab.app import AppLauncher

# 1. 命令行参数配置
parser = argparse.ArgumentParser(description="Zero agent for Isaac Lab Sand environment with MP4 recording.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Sand-Hexapod-Play-v0", help="Name of the task.")
# 添加 AppLauncher 必需的命令行参数
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 2. 启动 Omniverse 仿真器后台
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 启动后才能安全导入的拓展库
import isaaclab_tasks  # noqa: F401

def main():
    """直接导入本地的 Sand 配置文件，并在无窗口状态下录制 MP4 视频"""
    print("[INFO]: 正在绕过注册表，直接从本地硬核加载 HexapodSandEnvCfg_PLAY 配置...")
    try:
        # 直接通过 Python 绝对路径导入你刚刚上传成功的沙地配置
        from isaaclab_tasks.manager_based.locomotion.velocity.config.hexapod.sand_env_cfg import HexapodSandEnvCfg_PLAY
        env_cfg = HexapodSandEnvCfg_PLAY()
    except Exception as e:
        print(f"[ERROR]: 导入 sand_env_cfg.py 失败！错误原因: {e}")
        return

    # 覆盖并行环境数量（由于是录屏，强制为 1）
    env_cfg.scene.num_envs = args_cli.num_envs
    
    # 🎯 黄金相机视角：从侧后方稍微拉近并抬高，能够完美观察六足机器人在 3D 沙丘上的高低跌宕
    if hasattr(env_cfg, "viewer"):
        env_cfg.viewer.eye = [-1.5, 0.5, 0.7]
        env_cfg.viewer.lookat = [0.0, 0.0, 0.1]
        env_cfg.viewer.origin_type = "asset_root"
        env_cfg.viewer.asset_name = "robot"
        
    # 【核心伪装借壳】利用系统已经注册的 Flat 标识符，但注入带有 3D 沙丘和超低摩擦力的 env_cfg 内核
    print("[INFO]: 正在通过 'Isaac-Velocity-Flat-Hexapod-Play-v0' 标头借壳创建沙地仿真环境...")
    base_env = gym.make("Isaac-Velocity-Flat-Hexapod-Play-v0", cfg=env_cfg, render_mode="rgb_array")

    # 指定视频保存路径
    video_output_dir = "/home/ubuntu/IsaacLab-LiBRHexapod/my_videos"
    os.makedirs(video_output_dir, exist_ok=True)
    
    # 挂载 Gym 官方的高清 MP4 录制包装器
    env = gym.wrappers.RecordVideo(
        env=base_env,
        video_folder=video_output_dir,
        step_trigger=lambda step: step == 0, # 从第 0 步开始录制
        video_length=300                     # 明确录制 300 步 (约 6 秒)
    )

    print(f"[INFO]: 成功建立沙地环境。Observation 空间: {env.observation_space}")
    print(f"[INFO]: 成功建立沙地环境。Action 空间: {env.action_space}")
    
    env.reset()
    sim_steps = 0
    
    print("\n" + "="*60)
    print("[INFO]: >>>>>>> 正在无窗口后台直接录制【3D沙丘混合地形】MP4 视频，请稍候... <<<<<<<")
    print("="*60 + "\n")
    
    while simulation_app.is_running():
        with torch.inference_mode():
            # 发送全 0 动作量
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            env.step(actions)
            
            sim_steps += 1
            if sim_steps >= 300:
                print(f"[INFO]: 🏁 录制成功完成！沙地视频已保存在: {video_output_dir}/rl-video-step-0.mp4")
                break

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
