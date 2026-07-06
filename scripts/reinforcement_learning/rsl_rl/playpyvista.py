# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint of an RL agent from RSL-RL and log per-frame body world
poses (position + quaternion) to an .npz file for offline PyVista rendering.

This is a fork of play.py kept as a separate file so play.py can stay close to the
upstream Isaac Lab version and be updated/merged without conflicts. Differences from
play.py:
  - Always logs body poses (no flag needed) to <log_dir>/body_pose_log/.
  - Adds --num_steps to stop the play loop after a fixed number of steps instead of
    running indefinitely (play.py only stops early when --video is used).
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import csv
import numpy as np

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play an RL agent with RSL-RL and log body poses for PyVista.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--num_steps",
    type=int,
    default=None,
    help=(
        "Stop the play loop after this many env steps. If --video is also set, the video"
        " still stops at --video_length steps; whichever limit is hit first ends the run."
        " If omitted, the loop runs until the window is closed (same as play.py)."
    ),
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for installed RSL-RL version."""

import importlib.metadata as metadata

from packaging import version

installed_version = metadata.version("rsl-rl-lib")

"""Rest everything follows."""

import os
import time

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
    handle_deprecated_rsl_rl_cfg,
    handle_deprecated_rsl_rl_checkpoint,
)
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # handle deprecated configurations
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # Body pose log setup (same level as videos/)
    body_pos_frames: list = []
    body_quat_frames: list = []
    body_pose_log_dir = os.path.join(log_dir, "body_pose_log")
    os.makedirs(body_pose_log_dir, exist_ok=True)
    print(f"[INFO] Body pose logging enabled. Output: {body_pose_log_dir}")

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    # convert pre-5.0 published checkpoints to the layout expected by rsl-rl >= 5.0 (no-op otherwise)
    resume_path = handle_deprecated_rsl_rl_checkpoint(resume_path, installed_version)
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # export the trained policy to JIT and ONNX formats
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")

    if version.parse(installed_version) >= version.parse("4.0.0"):
        # use the new export functions for rsl-rl >= 4.0.0
        runner.export_policy_to_jit(path=export_model_dir, filename="policy.pt")
        runner.export_policy_to_onnx(path=export_model_dir, filename="policy.onnx")
    else:
        # extract the neural network for rsl-rl < 4.0.0
        if version.parse(installed_version) >= version.parse("2.3.0"):
            policy_nn = runner.alg.policy
        else:
            policy_nn = runner.alg.actor_critic

        # extract the normalizer
        if hasattr(policy_nn, "actor_obs_normalizer"):
            normalizer = policy_nn.actor_obs_normalizer
        elif hasattr(policy_nn, "student_obs_normalizer"):
            normalizer = policy_nn.student_obs_normalizer
        else:
            normalizer = None

        # export to JIT and ONNX
        export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt
    device = env.unwrapped.device

    # ---------- NEW: setup for saving joint positions (radians) ----------
    # File where we'll store joint positions (not raw actions)
    actionFileName = "C:/Users/jrh6552/Hexapod/IsaacLab/Position Files/HexapodRL_Rad_5-3-26_actiontest.csv"
    commandFileName = "C:/Users/jrh6552/Hexapod/IsaacLab/Position Files/HexapodRL_command_1-2-26_test.csv"
    # Number of joints we care about (first 8 from env 0)
    num_joints = 8

    # Default joint positions in radians:
    # joint 0,1 -> 0.0 rad; joints 2..7 -> 0.35 rad
    # This will be used as q_default in: q = q_default + 0.5 * action
    q_default_list = [0.0, 0.0] + [0.35] * (num_joints - 2)

    # Create/overwrite file and write header once
    os.makedirs(os.path.dirname(actionFileName), exist_ok=True)
    with open(actionFileName, "w", newline="") as f:
        writer = csv.writer(f)
        header = [f"joint_{i}_pos_rad" for i in range(num_joints)]
        writer.writerow(header)

    # ---- sanity prints ----
    print("[INFO] sim dt:", env.unwrapped.step_dt)
    try:
        robot = env.unwrapped.scene["robot"]
        if hasattr(robot.data, "joint_names"):
            print("[INFO] robot joint names (first 8):", robot.data.joint_names[:8])
    except Exception as e:
        print("[WARN] Couldn't print robot joint names:", e)

    # open file to track displacement
    disp_log_path = "C:/Users/jrh6552/Hexapod/IsaacLab/Position Files/sim_displacement_log_5-5-26_test.csv"
    os.makedirs(os.path.dirname(disp_log_path), exist_ok=True)

    with open(disp_log_path, "w", newline="") as d:
        w = csv.writer(d)
        w.writerow([
            "step",
            "t_s",
            "cycle_num",
            "cycle_step",
            "gait_idx",
            "x_w", "y_w", "z_w",
            "dx", "dy", "dz",
            "forward_disp_x",
        ])

    # reset environment
    obs = env.get_observations()
    timestep = 0

    # capture initial base position for env 0
    start_pos = robot.data.root_pos_w.clone()  # [num_envs, 3]

    try:
        if hasattr(env, "num_actions"):
            print("[INFO] action_dim:", env.num_actions)
        else:
            print("[INFO] action_dim:", int(env.action_space.shape[0]))
    except Exception as e:
        print("[WARN] Couldn't print action dim:", e)
    # -----------------------

    reward_sum_per_env = torch.zeros(env.num_envs, device=device, dtype=torch.float32)

    rm = getattr(env.unwrapped, "reward_manager", None)
    term_names = list(rm.active_terms) if rm is not None else []
    reward_term_sums = {
        name: torch.zeros(env.num_envs, device=device, dtype=torch.float32)
        for name in term_names
    }

    # reset environment
    obs = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)

            # env stepping
            obs, rew, dones, _ = env.step(actions)
            # reset recurrent states for episodes that have terminated
            if version.parse(installed_version) >= version.parse("4.0.0"):
                policy.reset(dones)
            else:
                policy_nn.reset(dones)

            # Access the underlying robot in the scene
            robot = env.unwrapped.scene["robot"]  # adjust name if needed

            reward_sum_per_env += rew

            rm = getattr(env.unwrapped, "reward_manager", None)
            if rm is not None:
                step_reward_terms = rm._step_reward  # [num_envs, num_terms]

                for term_idx, name in enumerate(term_names):
                    reward_term_sums[name] += step_reward_terms[:, term_idx] * dt

            # actual joint positions (env 0, first 8)
            joint_positions = robot.data.joint_pos[0, :num_joints]
            joint_positions_list = joint_positions.detach().cpu().numpy().tolist()

            with open(actionFileName, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(joint_positions_list)

            # Body pose logging (env 0 only)
            body_pos_frames.append(robot.data.body_pos_w[0].detach().cpu().numpy())
            body_quat_frames.append(robot.data.body_quat_w[0].detach().cpu().numpy())

            # -----Added to track the displacement of robot --------------
            # ---- step / cycle bookkeeping (warmup-aware) ----
            step = timestep  # raw sim step (includes warmup)

            active_step = ""
            cycle_num = ""
            cycle_step = ""
            gait_idx = ""
            # current base position in world
            t_s = step * dt

            pos = robot.data.root_pos_w
            dpos = pos - start_pos

            x, y, z = pos[0].detach().cpu().numpy().tolist()
            dx, dy, dz = dpos[0].detach().cpu().numpy().tolist()
            forward_disp_x = dx

            with open(disp_log_path, "a", newline="") as d:
                w = csv.writer(d)
                w.writerow([
                    step,
                    t_s,
                    cycle_num,
                    cycle_step,
                    gait_idx,
                    x, y, z,
                    dx, dy, dz,
                    forward_disp_x,
                ])

        # advance the step counter once per loop iteration (covers video and num_steps limits)
        timestep += 1
        if args_cli.video and timestep == args_cli.video_length:
            break
        if args_cli.num_steps is not None and timestep >= args_cli.num_steps:
            break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    print("\n===== REWARD SUMMARY =====")
    print("reward_sum_per_env:", reward_sum_per_env.detach().cpu().numpy())
    print("\n===== INDIVIDUAL REWARD TERM SUMMARY =====")
    for name in term_names:
        print(f"{name}:")
        print("  sum_per_env:", reward_term_sums[name].detach().cpu().numpy())

    # Save body pose log to npz
    if body_pos_frames:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        npz_path = os.path.join(body_pose_log_dir, f"body_poses_{timestamp}.npz")
        np.savez(
            npz_path,
            body_pos_w=np.array(body_pos_frames),    # (T, num_bodies, 3)
            body_quat_w=np.array(body_quat_frames),  # (T, num_bodies, 4) wxyz
            body_names=np.array(robot.data.body_names),
            dt=np.float32(dt),
        )
        print(f"[INFO] Body pose log saved: {npz_path}  ({len(body_pos_frames)} frames)")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
