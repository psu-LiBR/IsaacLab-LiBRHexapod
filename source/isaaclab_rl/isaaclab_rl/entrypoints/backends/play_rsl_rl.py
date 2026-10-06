# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL playback backend of the unified reinforcement learning entrypoint."""

from __future__ import annotations

import argparse
import contextlib
import csv
import logging
import os

import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.envs import DirectMARLEnvCfg
from isaaclab.utils import to_dict
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.seed import configure_seed
from isaaclab.utils.string import list_intersection

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, resolve_task_config, setup_preset_cli

from ...rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from ...utils.wandb import is_wandb_checkpoint, resolve_wandb_checkpoint
from ..common import (
    CHECKPOINT_SELECTORS,
    add_common_play_args,
    apply_env_overrides,
    apply_video_recording,
    close_env,
    create_isaaclab_env,
    enable_cameras_for_video,
    normalize_task_name,
    pre_launch_video_config,
    resolve_checkpoint_selector,
    resolve_published_checkpoint,
    run_playback,
    set_hydra_args,
    show_run_summary,
    startup_screen,
)
from . import cli_args_rsl_rl as cli_args

logger = logging.getLogger(__name__)

# PLACEHOLDER: Extension template (do not remove this comment)
with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401


def _parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse RSL-RL playback arguments."""
    parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from RSL-RL.")
    add_common_play_args(
        parser,
        agent_default="rsl_rl_cfg_entry_point",
        agent_help="Name of the RL agent configuration entry point.",
    )
    parser.add_argument(
        "--external_callback", default=None, help="Fully qualified path to an externally defined callback."
    )
    parser.add_argument(
        "--dump_obs_action_csv",
        type=str,
        default=None,
        help="Optional path to dump per-step (obs, action) pairs to CSV, for offline "
        "validation via scripts/sim2real_transfer/tools/validate_onnx.py. Only "
        "meaningful for the hexapod flat/goal policy obs layout (gyro, gravity, "
        "command, joint_pos_rel, joint_vel, last_action, in that fixed order) -- "
        "use with --num_envs 1. The dumped joint_pos_N columns are the raw absolute "
        "joint positions (matching robot.data.joint_pos), not the joint_pos_rel term "
        "baked into obs_N -- validate_onnx.py's pipeline mode subtracts q_default_sim "
        "itself to reconstruct joint_pos_rel from these.",
    )
    cli_args.add_rsl_rl_args(parser)
    add_launcher_args(parser)
    remaining_args_env_registration = cli_args.register_external_tasks(argv)
    args_cli, remaining_args = setup_preset_cli(parser, argv)
    enable_cameras_for_video(args_cli)
    set_hydra_args(list_intersection(remaining_args, remaining_args_env_registration))
    return args_cli


def _resolve_checkpoint(
    args_cli: argparse.Namespace, agent_cfg: RslRlBaseRunnerCfg, env_cfg: object, log_root_path: str
) -> str | None:
    """Resolve the checkpoint to play, or None when no published checkpoint exists."""
    if args_cli.checkpoint and is_wandb_checkpoint(args_cli.checkpoint):
        return resolve_wandb_checkpoint(args_cli.checkpoint)
    if args_cli.checkpoint == "pretrained":
        return resolve_published_checkpoint("rsl_rl", args_cli.task, env_cfg)
    if args_cli.checkpoint in CHECKPOINT_SELECTORS:
        return resolve_checkpoint_selector(
            log_root_path,
            args_cli.checkpoint,
            library="rsl_rl",
            task=normalize_task_name(args_cli.task),
            checkpoint_pattern=r"model_.*\.pt",
            metadata={"agent": args_cli.agent},
        )
    if args_cli.checkpoint and os.path.isdir(args_cli.checkpoint):
        return get_checkpoint_path(
            os.path.dirname(args_cli.checkpoint), os.path.basename(args_cli.checkpoint), agent_cfg.load_checkpoint
        )
    if args_cli.checkpoint:
        return retrieve_file_path(args_cli.checkpoint)
    return get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)


def run(argv: list[str]) -> None:
    """Play a checkpoint of an RSL-RL agent."""
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
            logger.info(f"Loading experiment from directory: {log_root_path}")
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
            logger.info(f"Loading model checkpoint from: {resume_path}")
            if agent_cfg.class_name == "OnPolicyRunner":
                runner = OnPolicyRunner(env, to_dict(agent_cfg), log_dir=None, device=agent_cfg.device)
            elif agent_cfg.class_name == "DistillationRunner":
                runner = DistillationRunner(env, to_dict(agent_cfg), log_dir=None, device=agent_cfg.device)
            else:
                raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
            # configure_seed must run after runner construction so torch determinism does not disturb its initialization
            if args_cli.deterministic:
                configure_seed(env_cfg.seed, torch_deterministic=True)
            runner.load(resume_path)
            policy = runner.get_inference_policy(device=env.unwrapped.device)

            export_model_dir = os.path.join(log_dir, "exported")
            runner.export_policy_to_jit(path=export_model_dir, filename="policy.pt")
            runner.export_policy_to_onnx(path=export_model_dir, filename="policy.onnx")

            obs = env.get_observations()
            dt = env.unwrapped.step_dt
            device = env.unwrapped.device

            # ---------- hexapod fork: joint-position (rad) + displacement logging ----------
            num_joints = 8
            action_log_path = "C:/Users/jrh6552/Hexapod/IsaacLab/Position Files/HexapodRL_Rad_5-3-26_actiontest.csv"
            disp_log_path = "C:/Users/jrh6552/Hexapod/IsaacLab/Position Files/sim_displacement_log_5-5-26_test.csv"

            os.makedirs(os.path.dirname(action_log_path), exist_ok=True)
            with open(action_log_path, "w", newline="") as f:
                csv.writer(f).writerow([f"joint_{i}_pos_rad" for i in range(num_joints)])

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

            robot = env.unwrapped.scene["robot"]
            start_pos = robot.data.root_pos_w.clone()  # [num_envs, 3]

            # ---------- hexapod fork: per-term reward breakdown, printed on exit ----------
            reward_sum_per_env = torch.zeros(env.num_envs, device=device, dtype=torch.float32)
            rm = getattr(env.unwrapped, "reward_manager", None)
            term_names = list(rm.active_terms) if rm is not None else []
            reward_term_sums = {
                name: torch.zeros(env.num_envs, device=device, dtype=torch.float32) for name in term_names
            }

            # ---------- hexapod fork: optional (obs, action) dump for offline ONNX validation ----------
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

            timestep = 0

            def step() -> None:
                nonlocal obs, timestep, dump_step
                actions = policy(obs)

                if dump_writer is not None:
                    obs_row = obs["policy"][0].detach().cpu().numpy()
                    action_row = actions[0].detach().cpu().numpy()
                    gyro, gravity = obs_row[0:3], obs_row[3:6]
                    command = obs_row[6 : 6 + cmd_dim]
                    # NOTE: obs_row's joint-pos slice is already joint_pos_rel (joint_pos - default), per
                    # profiles.py's obs term order. validate_onnx.py's pipeline mode re-subtracts
                    # q_default_sim itself, so this column must hold the raw absolute joint position, not
                    # the obs-vector slice, or leg joints (nonzero q_default) get double-subtracted.
                    joint_pos = robot.data.joint_pos[0, :num_joints].detach().cpu().numpy()
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
                policy.reset(dones)

                reward_sum_per_env.add_(rew)
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
                with open(disp_log_path, "a", newline="") as f:
                    csv.writer(f).writerow([timestep, timestep * dt, "", "", "", x, y, z, dx, dy, dz, dx])
                timestep += 1

            screen.close()
            try:
                run_playback(step, dt=dt, args_cli=args_cli, env_cfg=env_cfg)
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
