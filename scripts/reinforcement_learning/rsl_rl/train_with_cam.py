import os
import sys
import argparse

# =========================================================================
# 1. 纯净阶段：仅仅解析参数，绝不污染任何 isaac/omni 依赖
# =========================================================================
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL and fixed camera patch.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video.")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="Log data every n iterations.")

args_cli, unknown = parser.parse_known_args()

# =========================================================================
# 2. 正式初始化：遵循新版规范，从 isaacsim 唤醒底层引擎 (仅实例化一次)
# =========================================================================
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

# =========================================================================
# 3. 注入相机劫持热补丁：在任务创建前，强行将相机改写为黄金后视视角
# =========================================================================
import gymnasium as gym
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

if not hasattr(gym, '_camera_patched'):
    original_load_cfg = load_cfg_from_registry
    
    def patched_load_cfg(task_name, entry_point_key):
        cfg = original_load_cfg(task_name, entry_point_key)
        if hasattr(cfg, "viewer") and "Hexapod" in task_name:
            print("\n" + "="*60)
            print("[INFO] ✨ 热补丁拦截成功：已强制将六足机器人相机参数注入为【黄金后视跟随视角】")
            print("="*60 + "\n")
            cfg.viewer.origin_type = "asset_root"
            cfg.viewer.eye = (-1.2, 0.0, 0.6)
            cfg.viewer.lookat = (0.0, 0.0, 0.1)
            cfg.viewer.asset_name = "robot"
        return cfg

    import isaaclab_tasks.utils.parse_cfg as parse_cfg_mod
    parse_cfg_mod.load_cfg_from_registry = patched_load_cfg
    gym._camera_patched = True

# =========================================================================
# 4. 核心平替：直接就地执行官方训练器的初始化与迭代流程
# =========================================================================
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.wrappers.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_onnx
from isaaclab_tasks.utils import parse_env_cfg
from rsl_rl.runners import OnPolicyRunner

def main():
    # 解析并获取劫持修改后的任务环境配置
    env_cfg = parse_env_cfg(args_cli.task, use_gpu=True, num_envs=args_cli.num_envs)
    
    # 覆盖录制开关
    if args_cli.video:
        env_cfg.viewer.render_mode = "rgb_array"
    
    # 唤醒创建强化学习物理环境
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    
    # 如果开启录像，挂载 Gym 录制包装器
    if args_cli.video:
        video_output_dir = os.path.join("/home/ubuntu/IsaacLab-LiBRHexapod/logs/rsl_rl/hexapod_flat/videos/train")
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_output_dir,
            step_trigger=lambda step: step % args_cli.video_interval == 0,
            video_length=args_cli.video_length,
        )
    
    # 包装为 RSL_RL 专属矢量环境
    env = RslRlVecEnvWrapper(env)
    
    # 加载 PPO 智能体配置
    agent_cfg: RslRlOnPolicyRunnerCfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg")
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations
        
    # 实例化官方 RSL_RL 训练器
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir="/home/ubuntu/IsaacLab-LiBRHexapod/logs/rsl_rl/hexapod_flat")
    
    # 开始痛快训练！
    print("[INFO] 🚀 正在以完美的相机跟随视角启动强化学习正式训练...")
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_log_dir=runner.log_dir)
    
    # 训练完安全关闭
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
