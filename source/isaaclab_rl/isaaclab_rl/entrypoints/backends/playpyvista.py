# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play a checkpoint of an RL agent from RSL-RL and log per-frame body world poses (position + quaternion)
to an .npz file for offline PyVista rendering.

This is a fork of ``play_rsl_rl.py`` kept as a separate file so ``play_rsl_rl.py`` can stay close to the
upstream Isaac Lab version and be updated/merged without conflicts. Differences from ``play_rsl_rl.py``:

- Always logs body poses (no flag needed) to ``<log_dir>/body_pose_log/``.
- Adds ``--num_steps`` to stop the play loop after a fixed number of steps instead of running indefinitely
  (``play_rsl_rl.py`` only stops early when ``--video`` is used). Ctrl+C also ends the run, and the body pose
  log is still written.

Fork-only backend with no ``--rl_library`` registration, so it is not reachable through the unified ``play``
subcommand. Run it as a module, e.g.::

    uv run python -m isaaclab_rl.entrypoints.backends.playpyvista --task Isaac-Velocity-Flat-Hexapod-Play-v0 \\
        --num_envs 1 --num_steps 500
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import os
import sys
import time

import numpy as np
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
    """Parse playback arguments (RSL-RL playback arguments plus ``--num_steps``)."""
    parser = argparse.ArgumentParser(description="Play an RL agent with RSL-RL and log body poses for PyVista.")
    add_common_play_args(
        parser,
        agent_default="rsl_rl_cfg_entry_point",
        agent_help="Name of the RL agent configuration entry point.",
    )
    parser.add_argument(
        "--external_callback", default=None, help="Fully qualified path to an externally defined callback."
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=None,
        help=(
            "Stop the play loop after this many env steps. If --video is also set, the video"
            " still stops after its clip; whichever limit is hit first ends the run."
            " If omitted, the loop runs until interrupted with Ctrl+C."
        ),
    )
    cli_args.add_rsl_rl_args(parser)
    add_launcher_args(parser)
    remaining_args_env_registration = cli_args.register_external_tasks(argv)
    args_cli, remaining_args = setup_preset_cli(parser, argv)
    enable_cameras_for_video(args_cli)
    set_hydra_args(list_intersection(remaining_args, remaining_args_env_registration))
    return args_cli


def run(argv: list[str] | None = None) -> None:
    """Play an RSL-RL checkpoint and save the per-frame body poses of env 0."""
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

            # Body pose log setup (same level as videos/)
            body_pose_log_dir = os.path.join(log_dir, "body_pose_log")
            os.makedirs(body_pose_log_dir, exist_ok=True)
            print(f"[INFO] Body pose logging enabled. Output: {body_pose_log_dir}")

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
            policy = runner.get_inference_policy(device=env.unwrapped.device)

            # export the trained policy to JIT and ONNX formats
            export_model_dir = os.path.join(log_dir, "exported")
            runner.export_policy_to_jit(path=export_model_dir, filename="policy.pt")
            runner.export_policy_to_onnx(path=export_model_dir, filename="policy.onnx")

            screen.close()
            _play(env, env_cfg, policy, args_cli, body_pose_log_dir)


def _play(env, env_cfg, policy, args_cli: argparse.Namespace, body_pose_log_dir: str) -> None:
    """Step the policy, log joint positions, displacement and body poses, then save the pose log."""
    dt = env.unwrapped.step_dt
    device = env.unwrapped.device

    # ---------- setup for saving joint positions (radians) ----------
    # File where we'll store joint positions (not raw actions)
    action_log_path = "C:/Users/jrh6552/Hexapod/IsaacLab/Position Files/HexapodRL_Rad_5-3-26_actiontest.csv"
    # Number of joints we care about (first 8 from env 0)
    num_joints = 8

    # Create/overwrite file and write header once
    os.makedirs(os.path.dirname(action_log_path), exist_ok=True)
    with open(action_log_path, "w", newline="") as f:
        csv.writer(f).writerow([f"joint_{i}_pos_rad" for i in range(num_joints)])

    # ---- sanity prints ----
    print("[INFO] sim dt:", dt)
    robot = env.unwrapped.scene["robot"]
    try:
        if hasattr(robot.data, "joint_names"):
            print("[INFO] robot joint names (first 8):", robot.data.joint_names[:8])
    except Exception as e:
        print("[WARN] Couldn't print robot joint names:", e)

    # open file to track displacement
    disp_log_path = "C:/Users/jrh6552/Hexapod/IsaacLab/Position Files/sim_displacement_log_5-5-26_test.csv"
    os.makedirs(os.path.dirname(disp_log_path), exist_ok=True)
    with open(disp_log_path, "w", newline="") as f:
        csv.writer(f).writerow(
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

    obs = env.get_observations()

    # capture initial base position for env 0
    start_pos = robot.data.root_pos_w.clone()  # [num_envs, 3]

    try:
        if hasattr(env, "num_actions"):
            print("[INFO] action_dim:", env.num_actions)
        else:
            print("[INFO] action_dim:", int(env.action_space.shape[0]))
    except Exception as e:
        print("[WARN] Couldn't print action dim:", e)

    reward_sum_per_env = torch.zeros(env.num_envs, device=device, dtype=torch.float32)
    rm = getattr(env.unwrapped, "reward_manager", None)
    term_names = list(rm.active_terms) if rm is not None else []
    reward_term_sums = {name: torch.zeros(env.num_envs, device=device, dtype=torch.float32) for name in term_names}

    body_pos_frames: list = []
    body_quat_frames: list = []

    # step budget: --num_steps and the --video clip length, whichever comes first
    limits = [n for n in (args_cli.num_steps, video_playback_steps(args_cli, env_cfg)) if n is not None]
    max_steps = min(limits) if limits else None

    timestep = 0
    # simulate environment (Ctrl+C ends the loop; the summary and pose log below are still written)
    with contextlib.suppress(KeyboardInterrupt):
        while max_steps is None or timestep < max_steps:
            start_time = time.time()
            # run everything in inference mode
            with torch.inference_mode():
                # agent stepping
                actions = policy(obs)

                # env stepping
                obs, rew, dones, _ = env.step(actions)
                # reset recurrent states for episodes that have terminated
                policy.reset(dones)

                reward_sum_per_env += rew
                if rm is not None:
                    step_reward_terms = rm._step_reward  # [num_envs, num_terms]
                    for term_idx, name in enumerate(term_names):
                        reward_term_sums[name] += step_reward_terms[:, term_idx] * dt

                # actual joint positions (env 0, first 8)
                joint_positions_list = robot.data.joint_pos[0, :num_joints].detach().cpu().numpy().tolist()
                with open(action_log_path, "a", newline="") as f:
                    csv.writer(f).writerow(joint_positions_list)

                # Body pose logging (env 0 only)
                body_pos_frames.append(robot.data.body_pos_w[0].detach().cpu().numpy())
                body_quat_frames.append(robot.data.body_quat_w[0].detach().cpu().numpy())

                # displacement of env 0 relative to its start-of-play position
                pos = robot.data.root_pos_w
                dpos = pos - start_pos
                x, y, z = pos[0].detach().cpu().numpy().tolist()
                dx, dy, dz = dpos[0].detach().cpu().numpy().tolist()
                with open(disp_log_path, "a", newline="") as f:
                    csv.writer(f).writerow([timestep, timestep * dt, "", "", "", x, y, z, dx, dy, dz, dx])

            timestep += 1

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
            body_pos_w=np.array(body_pos_frames),  # (T, num_bodies, 3)
            body_quat_w=np.array(body_quat_frames),  # (T, num_bodies, 4) xyzw
            body_names=np.array(robot.data.body_names),
            dt=np.float32(dt),
        )
        print(f"[INFO] Body pose log saved: {npz_path}  ({len(body_pos_frames)} frames)")


if __name__ == "__main__":
    run()
