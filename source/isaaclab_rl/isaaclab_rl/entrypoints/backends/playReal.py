# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play a checkpoint of an RL agent from RSL-RL OR drive the robot from a gait CSV.

Fork-only backend with no ``--rl_library`` registration, so it is not reachable through the unified ``play``
subcommand. Run it as a module, e.g.::

    uv run python -m isaaclab_rl.entrypoints.backends.playReal --task Isaac-Velocity-Flat-Hexapod-Play-v0 \\
        --num_envs 1 --gait_csv <path_to_csv> --gait_mode pos --gait_dt <seconds_per_row> --warmup_time 1.0
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

from ...rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
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
    """Parse playback arguments (RSL-RL playback arguments plus the gait CSV options)."""
    parser = argparse.ArgumentParser(description="Play an RL agent with RSL-RL, or play a gait from CSV.")
    add_common_play_args(
        parser,
        agent_default="rsl_rl_cfg_entry_point",
        agent_help="Name of the RL agent configuration entry point.",
    )
    parser.add_argument(
        "--external_callback", default=None, help="Fully qualified path to an externally defined callback."
    )

    # ---------- gait CSV playback args ----------
    parser.add_argument(
        "--gait_csv",
        type=str,
        default=None,
        help="Path to CSV containing either joint position targets (rad) or raw normalized actions.",
    )
    parser.add_argument(
        "--gait_mode",
        type=str,
        choices=["pos", "action"],
        default="pos",
        help="Interpret CSV as joint positions (rad) or as raw normalized actions.",
    )
    parser.add_argument(
        "--gait_dt",
        type=float,
        default=None,
        help="Seconds per row in the gait CSV. If provided, each row is held for round(gait_dt / sim_dt) sim steps.",
    )
    parser.add_argument(
        "--gait_num_joints",
        type=int,
        default=8,
        help="Number of joints provided in the CSV (columns).",
    )
    parser.add_argument(
        "--warmup_time",
        type=float,
        default=0.0,
        help="Seconds to hold a stand-still pose at the start so the robot can settle onto the ground.",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=None,
        help="If set, overrides warmup_time and uses an exact number of sim steps for warmup.",
    )
    parser.add_argument(
        "--run_time",
        type=float,
        default=None,
        help="Run duration in seconds. Stops automatically without requiring video mode.",
    )
    # ------------------------------------------------

    cli_args.add_rsl_rl_args(parser)
    add_launcher_args(parser)
    remaining_args_env_registration = cli_args.register_external_tasks(argv)
    args_cli, remaining_args = setup_preset_cli(parser, argv)
    enable_cameras_for_video(args_cli)
    set_hydra_args(list_intersection(remaining_args, remaining_args_env_registration))
    return args_cli


def _load_gait_csv(csv_path: str) -> np.ndarray:
    """Loads a numeric CSV with no header. Returns shape [T, D]."""
    data = np.loadtxt(csv_path, delimiter=",")
    if data.ndim == 1:
        data = data[None, :]
    return data


def _infer_joint_pos_offset_scale(unwrapped_env, num_joints: int, device: torch.device):
    """
    Tries to infer the joint-position action mapping:
        joint_target = offset + scale * action
    Returns (offset[num_joints], scale[num_joints]).
    Falls back to offset=zeros, scale=0.5 if it can't find better.
    """
    offset_t = None
    scale_t = None

    # 1) Try action manager terms (robust when available)
    try:
        am = getattr(unwrapped_env, "action_manager", None)
        terms = None
        if am is not None:
            terms = getattr(am, "_terms", None)
            if terms is None:
                terms = getattr(am, "terms", None)

        if isinstance(terms, dict):
            # Prefer something that looks like joint position
            for name, term in terms.items():
                lname = str(name).lower()
                if ("joint" in lname) and ("pos" in lname or "position" in lname):
                    scale = getattr(term, "scale", None) or getattr(term, "_scale", None)
                    offset = getattr(term, "offset", None) or getattr(term, "_offset", None)

                    def _to_vec(x):
                        if x is None:
                            return None
                        if isinstance(x, (float, int)):
                            return torch.full((num_joints,), float(x), device=device, dtype=torch.float32)
                        if isinstance(x, torch.Tensor):
                            v = x.detach().to(device=device, dtype=torch.float32).flatten()
                            return v[:num_joints] if v.numel() >= num_joints else v.repeat(num_joints)[:num_joints]
                        if isinstance(x, (list, tuple, np.ndarray)):
                            v = torch.tensor(x, device=device, dtype=torch.float32).flatten()
                            return v[:num_joints] if v.numel() >= num_joints else v.repeat(num_joints)[:num_joints]
                        return None

                    scale_t = _to_vec(scale)
                    offset_t = _to_vec(offset)
                    break
    except Exception:
        pass

    # 2) Try robot defaults as offset if action term offset wasn't found
    if offset_t is None:
        try:
            robot = unwrapped_env.scene["robot"]
            if hasattr(robot.data, "default_joint_pos"):
                offset_t = robot.data.default_joint_pos[0, :num_joints].to(device=device, dtype=torch.float32)
            elif hasattr(robot.data, "joint_pos_default"):
                offset_t = robot.data.joint_pos_default[0, :num_joints].to(device=device, dtype=torch.float32)
        except Exception:
            pass

    # 3) Fallbacks
    if offset_t is None:
        offset_t = torch.zeros((num_joints,), device=device, dtype=torch.float32)
    if scale_t is None:
        scale_t = torch.full((num_joints,), 0.5, device=device, dtype=torch.float32)

    return offset_t, scale_t


def _load_policy_and_export(env, agent_cfg: RslRlBaseRunnerCfg, resume_path: str):
    """Loads the RSL-RL runner checkpoint, exports the policy, and returns the inference policy."""
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, to_dict(agent_cfg), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, to_dict(agent_cfg), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # export policy (rsl-rl >= 4 runners export themselves)
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    runner.export_policy_to_jit(path=export_model_dir, filename="policy.pt")
    runner.export_policy_to_onnx(path=export_model_dir, filename="policy.onnx")

    return policy


def _setup_gait_playback(args_cli, dt: float, unwrapped_env, device, num_joints: int):
    """Loads and validates the gait CSV, and infers joint offset/scale for 'pos' mode.

    Returns (gait_np, num_joints, gait_len, hold_steps, joint_offset, joint_scale).
    """
    gait_np = _load_gait_csv(args_cli.gait_csv)

    num_joints = int(args_cli.gait_num_joints)
    if gait_np.shape[1] < num_joints:
        raise ValueError(f"CSV has {gait_np.shape[1]} columns, but --gait_num_joints={num_joints}")

    gait_np = gait_np[:, :num_joints]
    gait_len = gait_np.shape[0]

    hold_steps = 1
    if args_cli.gait_dt is not None:
        hold_steps = max(1, int(round(float(args_cli.gait_dt) / float(dt))))

    if args_cli.gait_mode == "pos":
        joint_offset, joint_scale = _infer_joint_pos_offset_scale(unwrapped_env, num_joints, device)
        print(f"[INFO] Gait CSV mode=pos | hold_steps={hold_steps} | len={gait_len}")
    else:
        joint_offset, joint_scale = None, None
        print(f"[INFO] Gait CSV mode=action | hold_steps={hold_steps} | len={gait_len}")

    return gait_np, num_joints, gait_len, hold_steps, joint_offset, joint_scale


def _clamp_to_action_space(actions: torch.Tensor, env, device) -> torch.Tensor:
    """Clamps actions to the environment's action space bounds, if available."""
    try:
        low = torch.as_tensor(env.action_space.low, device=device, dtype=torch.float32)
        high = torch.as_tensor(env.action_space.high, device=device, dtype=torch.float32)
        return torch.max(torch.min(actions, high), low)
    except Exception:
        return actions


def _compute_actions(
    env,
    device,
    action_dim: int,
    timestep: int,
    warmup_steps: int,
    use_gait: bool,
    hold_steps: int,
    gait_len: int,
    gait_np,
    gait_mode: str,
    joint_offset,
    joint_scale,
    policy,
    obs,
) -> torch.Tensor:
    """Selects the per-step action: warmup stand-still, gait CSV replay, or trained policy."""
    if timestep < warmup_steps:
        # Stand still: zero action -> hold offset pose (starting pose)
        actions = torch.zeros((env.num_envs, action_dim), device=device, dtype=torch.float32)
        return _clamp_to_action_space(actions, env, device)

    # normal behavior (gait or policy)
    if not use_gait:
        return policy(obs)

    active_step = timestep - warmup_steps  # 0-based step counter AFTER warmup
    gait_idx = (active_step // hold_steps) % gait_len

    row = torch.tensor(gait_np[gait_idx], device=device, dtype=torch.float32)
    action_1d = (row - joint_offset) / joint_scale if gait_mode == "pos" else row

    actions = torch.zeros((env.num_envs, action_dim), device=device, dtype=torch.float32)
    actions[:, : action_1d.numel()] = action_1d
    return _clamp_to_action_space(actions, env, device)


def _step_cycle_bookkeeping(timestep: int, warmup_steps: int, use_gait: bool, gait_len: int, hold_steps: int):
    """Warmup-aware step/cycle bookkeeping for gait CSV playback logging."""
    step = timestep  # raw sim step (includes warmup)

    if not use_gait:
        return step, "", "", "", ""

    steps_per_cycle = gait_len * hold_steps

    if step < warmup_steps:
        # still warming up: no gait cycles yet
        return step, -1, 0, "", ""

    # step counter that starts at 0 right after warmup
    active_step = step - warmup_steps
    cycle_num = (active_step // steps_per_cycle) + 1  # 1-based
    cycle_step = active_step % steps_per_cycle  # 0..steps_per_cycle-1
    gait_idx = (active_step // hold_steps) % gait_len  # 0..gait_len-1
    return step, active_step, cycle_num, cycle_step, gait_idx


def run(argv: list[str] | None = None) -> None:
    """Play an RSL-RL agent, or replay a gait CSV, in the selected hexapod task."""
    argv = sys.argv[1:] if argv is None else argv
    args_cli = _parse_args(argv)
    use_gait = args_cli.gait_csv is not None

    with startup_screen(args_cli, num_stages=3) as screen:
        env_cfg, agent_cfg = resolve_task_config(args_cli.task, args_cli.agent, play_mode=not args_cli.train_env_cfg)
        pre_launch_video_config(env_cfg, args_cli)
        screen.stage("Launching simulation")
        with launch_simulation(env_cfg, args_cli), contextlib.ExitStack() as cleanup:
            show_run_summary(screen, args_cli, env_cfg, library="rsl_rl", action="play")
            agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
            apply_env_overrides(args_cli, env_cfg)
            env_cfg.seed = agent_cfg.seed

            log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
            print(f"[INFO] Loading experiment from directory: {log_root_path}")
            if use_gait:
                # No checkpoint required for gait playback
                resume_path = None
                log_dir = os.path.abspath(os.getcwd())
                print(f"[INFO] Gait playback enabled. Using gait CSV: {args_cli.gait_csv}")
            else:
                resume_path = _resolve_checkpoint(args_cli, agent_cfg, env_cfg, log_root_path)
                if resume_path is None:
                    return
                log_dir = os.path.dirname(resume_path)
                print(f"[INFO]: Loading model checkpoint from: {resume_path}")
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
            # wrap around environment for rsl-rl (still useful even for gait playback)
            env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
            screen.close()
            with contextlib.suppress(KeyboardInterrupt):
                _play(env, env_cfg, args_cli, agent_cfg, resume_path)


def _play(env, env_cfg, args_cli, agent_cfg: RslRlBaseRunnerCfg, resume_path: str | None) -> None:
    """Run the policy or gait CSV playback loop and print the reward summary."""
    use_gait = args_cli.gait_csv is not None

    # ---- sanity prints ----
    print("[INFO] sim dt:", env.unwrapped.step_dt)
    try:
        robot = env.unwrapped.scene["robot"]
        if hasattr(robot.data, "joint_names"):
            print("[INFO] robot joint names (first 8):", robot.data.joint_names[:8])
        try:
            masses = robot.root_physx_view.get_masses()  # [num_envs, num_bodies]
            total_mass = masses[0].sum().item()
            print(f"[INFO] total robot mass (kg): {total_mass:.4f}")
            per_body = dict(zip(robot.data.body_names, [round(m, 4) for m in masses[0].tolist()]))
            print(f"[INFO] per-body masses (kg): {per_body}")
        except Exception as mass_err:
            print(f"[WARN] Could not read robot masses: {mass_err}")
    except Exception as e:
        print("[WARN] Couldn't print robot joint names:", e)

    # open file to track displacement
    disp_log_path = "C:/Users/jrh6552/Hexapod/IsaacLab/Position Files/sim_displacement_log_3-26-26_physicsSim.csv"
    os.makedirs(os.path.dirname(disp_log_path), exist_ok=True)

    with open(disp_log_path, "w", newline="") as f:
        w = csv.writer(f)
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

    # ---------- RL runner/policy only if NOT gait playback ----------
    policy = None
    if not use_gait:
        assert resume_path is not None
        policy = _load_policy_and_export(env, agent_cfg, resume_path)
    # ---------------------------------------------------------------

    dt = env.unwrapped.step_dt
    device = env.unwrapped.device

    # ---------- NEW: setup for saving joint positions (radians) ----------
    # File where we'll store joint positions (not raw actions)
    actionFileName = "C:/Users/jrh6552/Hexapod/IsaacLab/Position Files/HexapodReal_unscaled_Rad_3-27-26_randtest.csv"
    # Number of joints we care about (first 8 from env 0)
    num_joints = 8

    # Create/overwrite file and write header once
    os.makedirs(os.path.dirname(actionFileName), exist_ok=True)
    with open(actionFileName, "w", newline="") as f:
        writer = csv.writer(f)
        header = [f"joint_{i}_pos_rad" for i in range(num_joints)]
        writer.writerow(header)

    # --- warmup / settle ---
    if args_cli.warmup_steps is not None:
        warmup_steps = int(args_cli.warmup_steps)
    else:
        warmup_steps = int(round(float(args_cli.warmup_time) / float(dt)))
    warmup_steps = max(0, warmup_steps)

    if warmup_steps > 0:
        print(
            f"[INFO] Warmup enabled: holding stand-still for {warmup_steps} steps "
            f"({warmup_steps * dt:.3f} s) to let the robot settle."
        )
    # -----------------------

    # ---------- NEW: load gait if requested ----------
    gait_len = 0
    hold_steps = 1
    gait_np, joint_offset, joint_scale = None, None, None
    if use_gait:
        gait_np, num_joints, gait_len, hold_steps, joint_offset, joint_scale = _setup_gait_playback(
            args_cli, dt, env.unwrapped, device, num_joints
        )
    # -----------------------------------------------

    max_steps = None
    if args_cli.run_time is not None:
        max_steps = int(round(float(args_cli.run_time) / float(dt)))
        max_steps = max(1, max_steps)
        print(f"[INFO] Fixed run_time enabled: {args_cli.run_time:.3f} s ({max_steps} steps)")
    # --video plays until every video recorder has finished its first clip
    video_steps = video_playback_steps(args_cli, env_cfg)
    if video_steps is not None:
        max_steps = video_steps if max_steps is None else min(max_steps, video_steps)

    # reset environment
    obs = env.get_observations()
    timestep = 0

    # collect reward of real gait
    # collect reward statistics
    reward_sum_per_env = torch.zeros(env.num_envs, device=device, dtype=torch.float32)
    reward_step_count_per_env = torch.zeros(env.num_envs, device=device, dtype=torch.float32)

    rm = getattr(env.unwrapped, "reward_manager", None)
    term_names = list(rm.active_terms) if rm is not None else []
    reward_term_sums = {name: torch.zeros(env.num_envs, device=device, dtype=torch.float32) for name in term_names}

    print("[INFO] reward terms:", term_names)

    # simulate environment
    while max_steps is None or timestep < max_steps:
        start_time = time.time()

        # run everything in inference mode
        with torch.inference_mode():
            # Get action dimension robustly (works with wrapper)
            if hasattr(env, "num_actions"):
                action_dim = int(env.num_actions)
            else:
                action_dim = int(env.action_space.shape[0])

            # ---- NEW: warmup / gait / policy action selection ----
            actions = _compute_actions(
                env,
                device,
                action_dim,
                timestep,
                warmup_steps,
                use_gait,
                hold_steps,
                gait_len,
                gait_np,
                args_cli.gait_mode,
                joint_offset,
                joint_scale,
                policy,
                obs,
            )
            # -----------------------------

            # env stepping
            obs, rew, dones, infos = env.step(actions)

            # actual joint positions (env 0, first 8)
            joint_positions = robot.data.joint_pos[0, :num_joints]
            joint_positions_list = joint_positions.detach().cpu().numpy().tolist()

            with open(actionFileName, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(joint_positions_list)

            # print applied torques and base velocity every 20 steps
            if timestep % 20 == 0:
                applied_torques = robot.data.applied_torque[0, :num_joints].detach().cpu().numpy()
                joint_names_short = robot.data.joint_names[:num_joints]
                print(
                    f"[step {timestep:5d}] applied torques (N·m): "
                    + "  ".join(f"{n}={t:+.3f}" for n, t in zip(joint_names_short, applied_torques))
                )
                # root_lin_vel_b is velocity in the robot's base frame (x=forward, y=lateral)
                vel_b = robot.data.root_lin_vel_b[0].detach().cpu().numpy()
                ang_vel_w = robot.data.root_ang_vel_w[0].detach().cpu().numpy()
                print(
                    f"[step {timestep:5d}] base velocity (body frame): "
                    f"vx={vel_b[0]:+.4f} m/s  vy={vel_b[1]:+.4f} m/s  vz={vel_b[2]:+.4f} m/s"
                    f"  yaw_rate={ang_vel_w[2]:+.4f} rad/s"
                )

            if timestep >= warmup_steps:
                reward_sum_per_env += rew
                reward_step_count_per_env += 1

                rm = getattr(env.unwrapped, "reward_manager", None)
                if rm is not None:
                    step_reward_terms = rm._step_reward  # [num_envs, num_terms]

                    for term_idx, name in enumerate(term_names):
                        reward_term_sums[name] += step_reward_terms[:, term_idx] * dt

            # ---- step / cycle bookkeeping (warmup-aware) ----
            step, active_step, cycle_num, cycle_step, gait_idx = _step_cycle_bookkeeping(
                timestep, warmup_steps, use_gait, gait_len, hold_steps
            )
            # -----------------------------------------------

            # current base position in world
            t_s = step * dt

            pos = robot.data.root_pos_w
            dpos = pos - start_pos

            x, y, z = pos[0].detach().cpu().numpy().tolist()
            dx, dy, dz = dpos[0].detach().cpu().numpy().tolist()
            forward_disp_x = dx

            with open(disp_log_path, "a", newline="") as f:
                w = csv.writer(f)
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

        # ALWAYS increment timestep so CSV playback advances even without video
        timestep += 1

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    valid_mask = reward_step_count_per_env > 0

    avg_reward_per_env = torch.zeros_like(reward_sum_per_env)
    avg_reward_per_env[valid_mask] = reward_sum_per_env[valid_mask] / reward_step_count_per_env[valid_mask]

    mean_reward_all_trials = avg_reward_per_env[valid_mask].mean().item()

    print("\n===== REWARD SUMMARY =====")
    print("reward_sum_per_env:", reward_sum_per_env.detach().cpu().numpy())
    print("reward_step_count_per_env:", reward_step_count_per_env.detach().cpu().numpy())
    print("avg_reward_per_env:", avg_reward_per_env.detach().cpu().numpy())
    print("mean_reward_all_trials:", mean_reward_all_trials)

    print("\n===== INDIVIDUAL REWARD TERM SUMMARY =====")
    for name in term_names:
        avg_term_per_env = torch.zeros_like(reward_term_sums[name])
        avg_term_per_env[valid_mask] = reward_term_sums[name][valid_mask] / reward_step_count_per_env[valid_mask]
        mean_term_all_trials = avg_term_per_env[valid_mask].mean().item()

        print(f"{name}:")
        print("  sum_per_env:", reward_term_sums[name].detach().cpu().numpy())
        print("  avg_per_env:", avg_term_per_env.detach().cpu().numpy())
        print("  mean_all_trials:", mean_term_all_trials)


if __name__ == "__main__":
    run()
