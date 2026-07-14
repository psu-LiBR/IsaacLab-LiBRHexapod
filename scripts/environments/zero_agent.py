# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
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
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

def main():
    """Zero actions agent with Isaac Lab environment and direct MP4 recording."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    
    # 强制配置虚拟相机视角
    if hasattr(env_cfg, "viewer"):
        env_cfg.viewer.eye = [2.5, 2.5, 2.5]
        env_cfg.viewer.lookat = [0.0, 0.0, 0.0]
        
    # 【核心修正】在创建环境时显式指定 render_mode 为 "rgb_array"
    base_env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")

    # 明确指定你的视频输出路径
    video_output_dir = "/home/ubuntu/IsaacLab-LiBRHexapod/my_videos"
    os.makedirs(video_output_dir, exist_ok=True)
    
    # 用 Gym 的 RecordVideo 直接包装环境
    env = gym.wrappers.RecordVideo(
        env=base_env,
        video_folder=video_output_dir,
        step_trigger=lambda step: step == 0, # 从第 0 步开始录
        video_length=300                     # 明确录制 300 步
    )

    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    
    env.reset()
    sim_steps = 0
    
    print("[INFO]: >>>>>>> 正在无窗口后台直接录制 MP4 视频，请稍候... <<<<<<<")
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            env.step(actions)
            
            sim_steps += 1
            if sim_steps >= 300:
                print(f"[INFO]: 录制成功完成！MP4 视频已保存在: {video_output_dir}")
                break

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
