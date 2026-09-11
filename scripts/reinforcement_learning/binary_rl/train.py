# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generic numbered-run entry point; full scientific identity is in config.json."""

import argparse
import json
import runpy
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
args = parser.parse_args()
config = json.loads(Path(args.config).read_text())
run_dir = Path(config["run_dir"])
run_dir.mkdir(parents=True, exist_ok=True)

import binary_common  # noqa: E402


def build_candidate_env(task, num_envs, seed, device="cuda:0", n_bits=6, compute_final_obs=False):
    import gymnasium as gym
    import torch
    from discrete_action_wrapper import DiscreteBitsActionWrapper
    from overnight_config import configure_env, save_runtime_audit, tensor

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    cfg = parse_env_cfg(task, device=device, num_envs=num_envs)
    cfg.seed = seed
    cfg.compute_final_obs = compute_final_obs
    audit = configure_env(cfg, config["reward"], config["spine"])
    env = gym.make(task, cfg=cfg)
    base = env.unwrapped
    assert abs(base.step_dt - 0.02) < 1e-10
    assert base.single_observation_space["policy"].shape == (32,)
    assert base.single_observation_space["critic"].shape == (35,)
    save_runtime_audit(base, audit, str(run_dir / "runtime_audit.json"))
    wrapped = DiscreteBitsActionWrapper(env, n_bits=n_bits)
    original_step = wrapped.step
    original_reset = wrapped.reset

    def reset(**kwargs):
        _, info = original_reset(**kwargs)
        base.scene.update(dt=base.step_dt)
        base.command_manager.compute(dt=0.0)
        return base.observation_manager.compute(), info

    wrapped.reset = reset
    started = time.monotonic()
    calls = 0
    metrics_file = (run_dir / "progress.jsonl").open("a", buffering=1)

    def step(action):
        nonlocal calls
        result = original_step(action)
        calls += 1
        if calls % 250 == 0 or calls == 1:
            if not torch.isfinite(result[1]).all():
                raise RuntimeError("Non-finite reward")
            elapsed = time.monotonic() - started
            row = {
                "vector_steps": calls,
                "elapsed_s": elapsed,
                "vector_steps_per_s": calls / elapsed,
                "mean_step_reward": result[1].mean().item(),
                "curriculum_stage": getattr(base, "_goal_curriculum_stage", 0),
                "mean_root_height_m": tensor(base.scene["robot"].data.root_pos_w)[:, 2].mean().item(),
            }
            row["reward_terms_step"] = {
                name: (base.reward_manager._step_reward[:, i] * base.step_dt).mean().item()
                for i, name in enumerate(base.reward_manager.active_terms)
            }
            if hasattr(base, "_overnight_quality"):
                row["quality_mean"] = base._overnight_quality.mean().item()
                row["progress_gate_mean"] = base._reward_progress_gate.mean().item()
            metrics_file.write(json.dumps(row) + "\n")
        return result

    wrapped.step = step
    return wrapped


binary_common.build_env = build_candidate_env
methods = {
    1: ("train_discrete.py", ["--algo", "dqn"]),
    2: ("train_discrete.py", ["--algo", "ddqn"]),
    3: ("train_discrete_ppo.py", []),
    4: ("train_discrete_ppo.py", ["--mask"]),
    5: ("train_sac_d.py", ["--n_step", "1"]),
    6: ("train_sac_d.py", ["--n_step", "5"]),
}
script, extra = methods[config["method_id"]]
if config["method_id"] in (3, 4):
    extra = [*extra, "--no_value_preprocessor"]
sys.argv = [
    str(HERE / script),
    "--num_envs",
    str(config.get("num_envs", 4096)),
    "--timesteps",
    str(config.get("vector_steps", 192000)),
    "--seed",
    str(config["seed"]),
    "--directory",
    str(run_dir.parent),
    "--experiment_name",
    run_dir.name,
    "--checkpoint_interval",
    str(config.get("checkpoint_interval", 2000)),
    "--device",
    "cuda:0",
    *extra,
]
sys.argv += config.get("extra_args", [])
runpy.run_path(str(HERE / script), run_name="__main__")
(run_dir / "completed.json").write_text(
    json.dumps({"status": "completed", "vector_steps": config.get("vector_steps", 192000)})
)
