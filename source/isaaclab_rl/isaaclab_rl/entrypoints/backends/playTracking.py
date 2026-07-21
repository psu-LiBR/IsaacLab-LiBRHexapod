# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import csv
import sys

from isaaclab.app import AppLauncher

# local imports
from isaaclab_rl.entrypoints.backends import cli_args_rsl_rl as cli_args

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
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
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

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
    runner.load(resume_path)

    # switch runner to eval mode (this script drives the robot open-loop, not via the trained policy)
    runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    """
    # ---------- NEW: setup for saving joint positions (radians) ----------
    # File where we'll store joint positions (not raw actions)
    actionFileName = "C:/Users/jrh6552/Hexapod/IsaacLab/Position Files/HexapodRL_Rad_12-17-25.csv"
    commandFileName = "C:/Users/jrh6552/Hexapod/IsaacLab/Position Files/HexapodRL_command_12-17-25.csv"
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

    os.makedirs(os.path.dirname(commandFileName), exist_ok=True)
    with open(commandFileName, "w", newline="") as f:
        writer = csv.writer(f)
        header = [f"joint_{i}_pos_rad" for i in range(num_joints)]
        writer.writerow(header)
    # --------------------------------------------------------------------
    """

    # ---------- Logging setup ----------
    logFileName = (
        "C:/Users/jrh6552/Hexapod/IsaacLab/Position Files/Hexapod_sin_cmdvsact_stiff37_damp.32_step_amp0.3_1-9-26.csv"
    )
    num_joints = 8

    q_default_list = [0.0, 0.0] + [0.35] * (num_joints - 2)
    q_default = torch.tensor(q_default_list, device=env.device, dtype=torch.float32)

    os.makedirs(os.path.dirname(logFileName), exist_ok=True)
    with open(logFileName, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["t_s"]
        header += [f"cmd_j{i}_rad" for i in range(num_joints)]
        header += [f"act_j{i}_rad" for i in range(num_joints)]
        writer.writerow(header)
    # -----------------------------------

    # ---------- Sine command parameters ----------
    A = 0.3  # amplitude in radians (keep <= your joint limits!)
    action_scale = 0.5  # from your comment: q = q_default + 0.5 * action
    t = 0.0
    # which joints to excite (example: first motor only)
    excite = torch.zeros(num_joints, device=env.device, dtype=torch.float32)
    excite[[1, 3, 6]] = 1.0  # set to 1.0 for joints you want to move
    # e.g., excite[:] = 1.0 to move all first 8 joints
    # --------------------------------------------
    log_f = open(logFileName, "a", newline="")  # noqa: SIM115 -- lives across the sim step loop, closed at exit
    writer = csv.writer(log_f)

    # reset environment
    obs = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            # actions = policy(obs) #-- uncomment for normal running
            # actions = actionsList[index] -- create actions list corresponding to the gait, scale and shift
            # Write the list of actions to a file - actions is a torch tensor
            # deploy at the same hz as the physical robot, and angle scaling/shifting, joint order
            # Build commanded joint positions (rad)

            # sin_val = -np.sin(freq_hz * t + phase) #
            # q_cmd = q_default + (A * sin_val) * excite  # shape [8]

            q_cmd = (
                q_default + A / 2 * excite
            )  # step response - 0.3 rad (in negative direction which follows experimentation)

            # Convert commanded joint positions -> actions
            # q = q_default + action_scale * action  => action = (q_cmd - q_default)/action_scale
            action_8 = (q_cmd - q_default) / action_scale
            action_8 = torch.clamp(action_8, -1.0, 1.0)

            # Build full action tensor: [num_envs, action_dim]
            # Get action dimension robustly (works with the wrapper)

            if hasattr(env, "num_actions"):
                action_dim = env.num_actions
            else:
                action_dim = int(env.action_space.shape[0])

            device = env.unwrapped.device if hasattr(env, "unwrapped") else env.device

            actions = torch.zeros((env.num_envs, action_dim), device=device, dtype=torch.float32)
            actions[:, :num_joints] = action_8

            # env stepping
            obs, _, _, _ = env.step(actions)

            # Access the underlying robot in the scene
            robot = env.unwrapped.scene["robot"]  # adjust name if needed

            # Get the joint target positions that Isaac is actually using
            # This is a tensor of shape [num_envs, num_joints]
            # joint_targets = robot.data.joint_pos_target[0, :8]  # env 0, first 8 joints
            # actual joint positions (env 0, first 8)
            act = robot.data.joint_pos[0, :num_joints].detach().cpu().numpy().tolist()

            # commanded joint targets if available, otherwise log the q_cmd you computed
            if hasattr(robot.data, "joint_pos_target"):
                cmd = robot.data.joint_pos_target[0, :num_joints].detach().cpu().numpy().tolist()
            else:
                cmd = q_cmd.detach().cpu().numpy().tolist()

            writer.writerow([t] + cmd + act)
            # optional: flush occasionally so you don't lose data on crash
            if timestep % 20 == 0:
                log_f.flush()

        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        t += dt  # COMMENT THIS OUT WHEN NOT USING

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
