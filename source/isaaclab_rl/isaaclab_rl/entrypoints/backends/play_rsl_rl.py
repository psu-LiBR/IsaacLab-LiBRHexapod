# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint of an RL agent from RSL-RL."""

import argparse
import contextlib
import csv
import importlib.metadata as metadata
import os
import sys
import time

import gymnasium as gym
import torch
from packaging import version
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.seed import configure_seed
from isaaclab.utils.string import list_intersection, string_to_callable

from isaaclab_rl.entrypoints.backends import cli_args_rsl_rl as cli_args
from isaaclab_rl.entrypoints.common import CHECKPOINT_SELECTORS, resolve_checkpoint_selector
from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
    handle_deprecated_rsl_rl_cfg,
)
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import (
    get_checkpoint_path,
    setup_preset_cli,
)
from isaaclab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)
with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401

# -- argparse ----------------------------------------------------------------
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during play.")
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
    "--dump_obs_action_csv",
    type=str,
    default=None,
    help="Optional path to dump per-step (obs, action) pairs to CSV, for offline "
    "validation via scripts/sim2real_transfer/tools/validate_onnx.py. Only "
    "meaningful for the hexapod flat/goal policy obs layout (gyro, gravity, "
    "command, joint_pos_rel, joint_vel, last_action, in that fixed order) -- "
    "use with --num_envs 1.",
)
parser.add_argument("--external_callback", default=None, help="Fully qualified path to an externally defined callback.")
cli_args.add_rsl_rl_args(parser)
add_launcher_args(parser)
args_cli, remaining_args = setup_preset_cli(parser)

if args_cli.video:
    args_cli.enable_cameras = True


# an external callback lets downstream code register its environments; it returns
# the arguments it did not consume
remaining_args_env_registration = None
if args_cli.external_callback:
    external_callback_function = string_to_callable(args_cli.external_callback, separator=".")
    remaining_args_env_registration = external_callback_function()

# hand the arguments consumed by neither this parser nor the callback over to Hydra
remaining_args = list_intersection(remaining_args, remaining_args_env_registration)
sys.argv = [sys.argv[0]] + remaining_args

installed_version = metadata.version("rsl-rl-lib")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    with launch_simulation(env_cfg, args_cli):
        task_name = args_cli.task.split(":")[-1]
        train_task_name = task_name.replace("-Play", "")

        agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

        # note: certain randomizations occur in the environment initialization so we set the seed here
        env_cfg.seed = agent_cfg.seed
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

        log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        log_root_path = os.path.abspath(log_root_path)
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        if args_cli.use_pretrained_checkpoint:
            resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
            if not resume_path:
                print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
                return
        elif args_cli.checkpoint in CHECKPOINT_SELECTORS:
            resume_path = resolve_checkpoint_selector(
                log_root_path,
                args_cli.checkpoint,
                library="rsl_rl",
                task=train_task_name,
                checkpoint_pattern=r"model_.*\.pt",
                metadata={"agent": args_cli.agent},
            )
        elif args_cli.checkpoint:
            resume_path = retrieve_file_path(args_cli.checkpoint)
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

        log_dir = os.path.dirname(resume_path)

        env_cfg.log_dir = log_dir

        env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

        if isinstance(env.unwrapped.cfg, DirectMARLEnvCfg):
            from isaaclab.envs import multi_agent_to_single_agent

            env = multi_agent_to_single_agent(env)

        if args_cli.video:
            video_kwargs = {
                "video_folder": os.path.join(log_dir, "videos", "play"),
                "step_trigger": lambda step: step == 0,
                "video_length": args_cli.video_length,
                "disable_logger": True,
            }
            print("[INFO] Recording videos during play.")
            print_dict(video_kwargs, nesting=4)
            env = gym.wrappers.RecordVideo(env, **video_kwargs)

        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        if agent_cfg.class_name == "OnPolicyRunner":
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        elif agent_cfg.class_name == "DistillationRunner":
            runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        else:
            raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
        # configure_seed must run after runner construction so torch determinism does not disturb its initialization
        if args_cli.deterministic:
            configure_seed(env_cfg.seed, torch_deterministic=True)
        runner.load(resume_path)

        policy = runner.get_inference_policy(device=env.unwrapped.device)

        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")

        if version.parse(installed_version) >= version.parse("4.0.0"):
            runner.export_policy_to_jit(path=export_model_dir, filename="policy.pt")
            runner.export_policy_to_onnx(path=export_model_dir, filename="policy.onnx")
            policy_nn = None  # Not needed for rsl-rl >= 4.0.0
        else:
            if version.parse(installed_version) >= version.parse("2.3.0"):
                policy_nn = runner.alg.policy
            else:
                policy_nn = runner.alg.actor_critic

            if hasattr(policy_nn, "actor_obs_normalizer"):
                normalizer = policy_nn.actor_obs_normalizer
            elif hasattr(policy_nn, "student_obs_normalizer"):
                normalizer = policy_nn.student_obs_normalizer
            else:
                normalizer = None

            export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
            export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

        dt = env.unwrapped.step_dt
        device = env.unwrapped.device

        # ---------- hexapod fork: joint-position (rad) + displacement logging ----------
        num_joints = 8
        action_log_path = "C:/Users/jrh6552/Hexapod/IsaacLab/Position Files/HexapodRL_Rad_5-3-26_actiontest.csv"
        disp_log_path = "C:/Users/jrh6552/Hexapod/IsaacLab/Position Files/sim_displacement_log_5-5-26_test.csv"

        os.makedirs(os.path.dirname(action_log_path), exist_ok=True)
        with open(action_log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([f"joint_{i}_pos_rad" for i in range(num_joints)])

        os.makedirs(os.path.dirname(disp_log_path), exist_ok=True)
        with open(disp_log_path, "w", newline="") as d:
            writer = csv.writer(d)
            writer.writerow(
                [
                    "step",
                    "t_s",
                    "cycle_num",
                    "cycle_step",
                    "gait_idx",
                    "x_w",
                    "y_w",
                    "z_w",
                    "dx",
                    "dy",
                    "dz",
                    "forward_disp_x",
                ]
            )

        robot = env.unwrapped.scene["robot"]
        start_pos = robot.data.root_pos_w.clone()  # [num_envs, 3]

        # ---------- hexapod fork: per-term reward breakdown, printed on exit ----------
        reward_sum_per_env = torch.zeros(env.num_envs, device=device, dtype=torch.float32)
        rm = getattr(env.unwrapped, "reward_manager", None)
        term_names = list(rm.active_terms) if rm is not None else []
        reward_term_sums = {name: torch.zeros(env.num_envs, device=device, dtype=torch.float32) for name in term_names}

        # ---------- hexapod fork: optional (obs, action) dump for offline ONNX validation ----------
        obs = env.get_observations()
        timestep = 0
        dump_writer = None
        dump_file = None
        dump_step = 0
        cmd_dim = 0
        if args_cli.dump_obs_action_csv:
            obs_dim = obs["policy"].shape[1]
            cmd_dim = obs_dim - 3 - 3 - num_joints - num_joints - num_joints
            action_dim = env.num_actions
            os.makedirs(os.path.dirname(os.path.abspath(args_cli.dump_obs_action_csv)), exist_ok=True)
            dump_file = open(args_cli.dump_obs_action_csv, "w", newline="")  # noqa: SIM115 -- closed below
            dump_writer = csv.writer(dump_file)
            dump_writer.writerow(
                ["step", "t"]
                + [f"gyro_{i}" for i in range(3)]
                + [f"gravity_{i}" for i in range(3)]
                + [f"command_{i}" for i in range(cmd_dim)]
                + [f"joint_pos_{i}" for i in range(num_joints)]
                + [f"joint_vel_{i}" for i in range(num_joints)]
                + [f"last_action_{i}" for i in range(num_joints)]
                + [f"obs_{i}" for i in range(obs_dim)]
                + [f"action_{i}" for i in range(action_dim)]
            )

        try:
            while True:
                start_time = time.time()
                with torch.inference_mode():
                    actions = policy(obs)

                    if dump_writer is not None:
                        obs_row = obs["policy"][0].detach().cpu().numpy()
                        action_row = actions[0].detach().cpu().numpy()
                        gyro, gravity = obs_row[0:3], obs_row[3:6]
                        command = obs_row[6 : 6 + cmd_dim]
                        joint_pos = obs_row[6 + cmd_dim : 6 + cmd_dim + num_joints]
                        joint_vel = obs_row[6 + cmd_dim + num_joints : 6 + cmd_dim + 2 * num_joints]
                        last_action = obs_row[6 + cmd_dim + 2 * num_joints : 6 + cmd_dim + 3 * num_joints]
                        dump_writer.writerow(
                            [dump_step, dump_step * dt]
                            + list(gyro)
                            + list(gravity)
                            + list(command)
                            + list(joint_pos)
                            + list(joint_vel)
                            + list(last_action)
                            + list(obs_row)
                            + list(action_row)
                        )
                        dump_step += 1

                    obs, rew, dones, _ = env.step(actions)
                    # reset recurrent states for episodes that have terminated
                    if version.parse(installed_version) >= version.parse("4.0.0"):
                        policy.reset(dones)
                    else:
                        policy_nn.reset(dones)

                    reward_sum_per_env += rew
                    if rm is not None:
                        step_reward_terms = rm._step_reward  # [num_envs, num_terms]
                        for term_idx, name in enumerate(term_names):
                            reward_term_sums[name] += step_reward_terms[:, term_idx] * dt

                    # actual joint positions (env 0, first num_joints)
                    joint_positions_list = robot.data.joint_pos[0, :num_joints].detach().cpu().numpy().tolist()
                    with open(action_log_path, "a", newline="") as f:
                        csv.writer(f).writerow(joint_positions_list)

                    # displacement of env 0 relative to its start-of-play position
                    pos = robot.data.root_pos_w
                    dpos = pos - start_pos
                    x, y, z = pos[0].detach().cpu().numpy().tolist()
                    dx, dy, dz = dpos[0].detach().cpu().numpy().tolist()
                    t_s = timestep * dt
                    with open(disp_log_path, "a", newline="") as d:
                        csv.writer(d).writerow([timestep, t_s, "", "", "", x, y, z, dx, dy, dz, dx])

                if args_cli.video:
                    timestep += 1
                    if timestep == args_cli.video_length:
                        break
                else:
                    timestep += 1

                sleep_time = dt - (time.time() - start_time)
                if args_cli.real_time and sleep_time > 0:
                    time.sleep(sleep_time)

            env.close()
        except KeyboardInterrupt:
            pass
        finally:
            print("\n===== REWARD SUMMARY =====")
            print("reward_sum_per_env:", reward_sum_per_env.detach().cpu().numpy())
            print("\n===== INDIVIDUAL REWARD TERM SUMMARY =====")
            for name in term_names:
                print(f"{name}:")
                print("  sum_per_env:", reward_term_sums[name].detach().cpu().numpy())

            if dump_file is not None:
                dump_file.close()
                print(f"[INFO] wrote {dump_step} (obs, action) rows to {args_cli.dump_obs_action_csv}")


if __name__ == "__main__":
    main()
