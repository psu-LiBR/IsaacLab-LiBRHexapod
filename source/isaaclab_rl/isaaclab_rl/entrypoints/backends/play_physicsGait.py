# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play a checkpoint of an RL agent from RSL-RL and log the joint position targets and base displacement
of env 0 to CSV (used for comparing the policy against the physical robot's gait).

Fork-only backend with no ``--rl_library`` registration, so it is not reachable through the unified ``play``
subcommand. Run it as a module, e.g.::

    uv run python -m isaaclab_rl.entrypoints.backends.play_physicsGait \
        --task Isaac-Velocity-Flat-Hexapod-Play-v0 --num_envs 1
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import os
import sys
import time

import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.envs import DirectMARLEnvCfg
from isaaclab.utils import to_dict
from isaaclab.utils.string import list_intersection

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

from ...rsl_rl import RslRlVecEnvWrapper
from ..common import (
    add_common_play_args,
    apply_env_overrides,
    apply_video_recording,
    close_env,
    create_isaaclab_env,
    enable_cameras_for_video,
    pre_launch_video_config,
    set_hydra_args,
    show_run_summary,
    startup_screen,
    video_playback_steps,
)
from . import cli_args_rsl_rl as cli_args
from .play_rsl_rl import _resolve_checkpoint


def _parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse the RSL-RL playback arguments."""
    parser = argparse.ArgumentParser(description="Play an RL agent with RSL-RL.")
    add_common_play_args(
        parser,
        agent_default="rsl_rl_cfg_entry_point",
        agent_help="Name of the RL agent configuration entry point.",
    )
    parser.add_argument(
        "--external_callback", default=None, help="Fully qualified path to an externally defined callback."
    )
    cli_args.add_rsl_rl_args(parser)
    add_launcher_args(parser)
    remaining_args_env_registration = cli_args.register_external_tasks(argv)
    args_cli, remaining_args = setup_preset_cli(parser, argv)
    enable_cameras_for_video(args_cli)
    set_hydra_args(list_intersection(remaining_args, remaining_args_env_registration))
    return args_cli


def run(argv: list[str] | None = None) -> None:
    """Load the checkpoint, create the environment and run the playback loop."""
    argv = sys.argv[1:] if argv is None else argv
    args_cli = _parse_args(argv)

    with startup_screen(args_cli, num_stages=3) as screen:
        env_cfg, agent_cfg = resolve_task_config(args_cli.task, args_cli.agent, play_mode=not args_cli.train_env_cfg)
        pre_launch_video_config(env_cfg, args_cli)
        screen.stage("Launching simulation")
        with launch_simulation(env_cfg, args_cli), contextlib.ExitStack() as cleanup:
            show_run_summary(screen, args_cli, env_cfg, library="rsl_rl", action="play")
            agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
            apply_env_overrides(args_cli, env_cfg)
            # certain randomizations occur in the environment initialization so we set the seed here
            env_cfg.seed = agent_cfg.seed

            log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
            print(f"[INFO] Loading experiment from directory: {log_root_path}")
            resume_path = _resolve_checkpoint(args_cli, agent_cfg, env_cfg, log_root_path)
            if resume_path is None:
                return
            log_dir = os.path.dirname(resume_path)
            env_cfg.log_dir = log_dir
            apply_video_recording(env_cfg, log_dir, args_cli, subdir="play", checkpoint_path=resume_path)

            screen.stage("Creating environment")
            env = create_isaaclab_env(
                args_cli.task,
                env_cfg,
                args_cli,
                convert_marl_to_single_agent=isinstance(env_cfg, DirectMARLEnvCfg),
            )
            cleanup.callback(lambda: close_env(env))

            screen.stage("Loading policy")
            env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
            print(f"[INFO]: Loading model checkpoint from: {resume_path}")
            if agent_cfg.class_name == "OnPolicyRunner":
                runner = OnPolicyRunner(env, to_dict(agent_cfg), log_dir=None, device=agent_cfg.device)
            elif agent_cfg.class_name == "DistillationRunner":
                runner = DistillationRunner(env, to_dict(agent_cfg), log_dir=None, device=agent_cfg.device)
            else:
                raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
            runner.load(resume_path)
            # obtain the trained policy for inference (also switches the runner to eval mode)
            policy = runner.get_inference_policy(device=env.unwrapped.device)

            # export policy to onnx/jit
            export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
            runner.export_policy_to_jit(path=export_model_dir, filename="policy.pt")
            runner.export_policy_to_onnx(path=export_model_dir, filename="policy.onnx")

            screen.close()
            with contextlib.suppress(KeyboardInterrupt):
                _play(env, env_cfg, policy, args_cli)


def _play(env, env_cfg, policy, args_cli: argparse.Namespace) -> None:
    """Run the playback loop."""
    dt = env.unwrapped.step_dt

    # ---------- NEW: setup for saving joint positions (radians) ----------
    # File where we'll store joint positions (not raw actions)
    actionFileName = "C:/Users/jrh6552/Hexapod/IsaacLab/Position Files/HexapodRL_Rad_1-15-26_test.csv"
    # Number of joints we care about (first 8 from env 0)
    num_joints = 8

    # Create/overwrite file and write header once
    os.makedirs(os.path.dirname(actionFileName), exist_ok=True)
    with open(actionFileName, "w", newline="") as f:
        writer = csv.writer(f)
        header = [f"joint_{i}_pos_rad" for i in range(num_joints)]
        writer.writerow(header)

    # os.makedirs(os.path.dirname(commandFileName), exist_ok=True)
    # with open(commandFileName, "w", newline="") as f:
    #    writer = csv.writer(f)
    #    header = [f"joint_{i}_pos_rad" for i in range(num_joints)]
    #    writer.writerow(header)

    # ---- sanity prints ----
    print("[INFO] sim dt:", env.unwrapped.step_dt)
    try:
        robot = env.unwrapped.scene["robot"]
        if hasattr(robot.data, "joint_names"):
            print("[INFO] robot joint names (first 8):", robot.data.joint_names[:8])
    except Exception as e:
        print("[WARN] Couldn't print robot joint names:", e)

    # open file to track displacement
    disp_log_path = "C:/Users/jrh6552/Hexapod/IsaacLab/Position Files/sim_displacement_log_1-22-26_test.csv"
    os.makedirs(os.path.dirname(disp_log_path), exist_ok=True)

    with open(disp_log_path, "w", newline="") as d:
        w = csv.writer(d)
        w.writerow(
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
    # --------------------------------------------------------------------

    """
    # ---------- Logging setup ----------
    logFileName = "C:/Users/jrh6552/Hexapod/IsaacLab/Position Files/Hexapod_sin_cmd_vs_actual_stiff20.19_12-17-25.csv"
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
    A = 0.3          # amplitude in radians (keep <= your joint limits!)
    freq_hz = 1.0    # sine frequency (Hz)
    phase = 0.0       # phase (rad)
    action_scale = 0.5  # from your comment: q = q_default + 0.5 * action
    t = 0.0
    # which joints to excite (example: first motor only)
    excite = torch.zeros(num_joints, device=env.device, dtype=torch.float32)
    excite[[1, 2, 5]] = 1.0 # set to 1.0 for joints you want to move
    # e.g., excite[:] = 1.0 to move all first 8 joints
    # --------------------------------------------
    log_f = open(logFileName, "a", newline="")
    writer = csv.writer(log_f)
    """

    # reset environment
    obs = env.get_observations()
    timestep = 0
    # --video plays until every video recorder has finished its first clip
    max_steps = video_playback_steps(args_cli, env_cfg)
    # simulate environment
    while max_steps is None or timestep < max_steps:
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)  # -- uncomment for normal running
            # actions = actionsList[index] -- create actions list corresponding to the gait, scale and shift
            # Write the list of actions to a file - actions is a torch tensor
            # deploy at the same hz as the physical robot, and angle scaling/shifting, joint order
            # Build commanded joint positions (rad)
            """
            # sin_val = np.sin(freq_hz * t + phase) #
            # q_cmd = q_default + (A * sin_val) * excite  # shape [8]

            # Convert commanded joint positions -> actions
            # q = q_default + action_scale * action  => action = (q_cmd - q_default)/action_scale
            # action_8 = (q_cmd - q_default) / action_scale
            # action_8 = torch.clamp(action_8, -1.0, 1.0)

            # Build full action tensor: [num_envs, action_dim]
            # Get action dimension robustly (works with the wrapper)

            if hasattr(env, "num_actions"):
                action_dim = env.num_actions
            else:
                action_dim = int(env.action_space.shape[0])

            device = env.unwrapped.device if hasattr(env, "unwrapped") else env.device

            actions = torch.zeros((env.num_envs, action_dim), device=device, dtype=torch.float32)
            actions[:, :num_joints] = action_8
            """

            # env stepping
            obs, reward, _, _ = env.step(
                actions
            )  # will return the computed reward as a float - set environments to ten and average across (or set higher)

            # sum the rewards over the timesteps and average everything together at the end

            # Access the underlying robot in the scene
            robot = env.unwrapped.scene["robot"]  # adjust name if needed

            # Get the joint target positions that Isaac is actually using
            # This is a tensor of shape [num_envs, num_joints]
            joint_targets = robot.data.joint_pos_target[0, :8]  # env 0, first 8 joints

            joint_positions_list = [float(x.item()) for x in joint_targets]

            with open(actionFileName, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(joint_positions_list)

            # -----Added to track the displacement of robot --------------
            # ---- step / cycle bookkeeping (warmup-aware) ----
            step = timestep  # raw sim step (includes warmup)

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
                w.writerow(
                    [
                        step,
                        t_s,
                        cycle_num,
                        cycle_step,
                        gait_idx,
                        x,
                        y,
                        z,
                        dx,
                        dy,
                        dz,
                        forward_disp_x,
                    ]
                )

        timestep += 1

        # t += dt #COMMENT THIS OUT WHEN NOT USING

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)


if __name__ == "__main__":
    run()
