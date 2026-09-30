# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""One independent simulator per checkpoint/reference; real mesh rendering optional.

Uses the frozen training task and the same 6-second physical protocol as validate.py.
Evaluation reward is diagnostic only: disabled resets mean terminal indicators may
remain true for several steps, so it is not a training episodic return.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

RELEASE = Path(__file__).resolve().parent
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--candidate", required=True, help="Candidate ID in candidate_manifest.json")
parser.add_argument("--out", required=True, help="New output JSON path; also writes audit and trajectory")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
batch = Path(__file__).resolve().parents[3] / "switch-penalty-5x-to-100x"
manifest = json.loads((batch / "candidate_manifest.json").read_text())
record = next((r for r in manifest["candidates"] if r["id"] == args.candidate), None)
if record is None:
    parser.error("Unknown candidate. See switch-penalty-5x-to-100x/candidate_manifest.json")
folder = batch / record["directory"]
for filename, expected in record["artifact_sha256"].items():
    if hashlib.sha256((folder / filename).read_bytes()).hexdigest() != expected:
        raise ValueError("Artifact hash mismatch: " + filename)
config = json.loads((folder / "evaluation.json").read_text())
config.update(checkpoint=str(folder / "policy.pt"), run_id=record["id"], output=args.out)
args.headless = True
args.enable_cameras = bool(config.get("video", False))


class ReviewLauncher(AppLauncher):
    def _resolve_device_settings(self, launcher_args):
        super()._resolve_device_settings(launcher_args)
        if config.get("video"):
            physical = int(os.environ["REVIEW_RENDER_GPU"])
            launcher_args["active_gpu"] = physical
            launcher_args["physics_gpu"] = 0
            launcher_args["multi_gpu"] = False


app = ReviewLauncher(args).app

import gymnasium as gym
import numpy as np
import torch
from discrete_action_wrapper import DiscreteBitsActionWrapper
from overnight_config import configure_env, reward_metrics, save_runtime_audit, tensor
from switch_policy_loader import GreedyPolicy

from isaaclab_tasks.utils import parse_env_cfg

output = Path(config["output"])
if output.exists():
    raise FileExistsError("Preserve existing evaluation; use a new evaluation ID")
output.parent.mkdir(parents=True, exist_ok=True)
torch.manual_seed(config.get("seed", 7))
np.random.seed(config.get("seed", 7))
cfg = parse_env_cfg("Isaac-Goal-Flat-Hexapod-Binary-v0", device=args.device, num_envs=config.get("num_envs", 64))
cfg.seed = config.get("seed", 7)
audit = configure_env(cfg, config["reward"], config["spine"], evaluation=True, no_reset=True)
# Reproduce this training's actual reward override, preserving evaluation no-reset settings.
import copy

from isaaclab.managers import RewardTermCfg

from isaaclab_tasks.contrib.velocity.config.hexapod.hexapod_goal_env_cfg import HexapodGoalEnvCfg

for name, weight in config["override"].items():
    if getattr(cfg.rewards, name, None) is None:
        assert name == "feet_slide"
        cfg.rewards.feet_slide = copy.deepcopy(HexapodGoalEnvCfg().rewards.feet_slide)
    getattr(cfg.rewards, name).weight = weight
from switch_command_reward import action_switch_count

switch_name = "action_rate_l2" if config["reward"] == "J" else "switch"
if config.get("native_switch"):
    from isaaclab.envs.mdp.rewards import action_rate_l2

    getattr(cfg.rewards, switch_name).func = action_rate_l2
    getattr(cfg.rewards, switch_name).params = {}
getattr(cfg.rewards, switch_name).weight = config["switch_weight"]
if config["new_switch_cost"] > 0:
    cfg.rewards.command_switch = RewardTermCfg(func=action_switch_count, weight=-config["new_switch_cost"])
audit["weights"] = {k: v.weight for k, v in vars(cfg.rewards).items() if isinstance(v, RewardTermCfg)}
training_audit = {"weights": config["expected_reward_weights"]}
assert audit["weights"] == training_audit["weights"], (audit["weights"], training_audit["weights"])
audit["training_run_id"] = config["run_id"]
audit["override"] = config["override"]
evaluation_friction = float(config["evaluation_friction"])
assert evaluation_friction == 0.21
material = cfg.events.physics_material.params
material["static_friction_range"] = (evaluation_friction, evaluation_friction)
material["dynamic_friction_range"] = (evaluation_friction, evaluation_friction)
material["num_buckets"] = 1
audit["friction"] = {k: v for k, v in material.items() if k != "asset_cfg"}
audit["source_commit"] = config["source_commit"]
audit["source_commit_basis"] = "frozen training config; source archive has no .git"
audit["environment_code_sha256"] = hashlib.sha256((RELEASE / "overnight_config.py").read_bytes()).hexdigest()
audit["environment_code_basis"] = "Same release module imported by active training"
assert audit["source_commit"] == config["source_commit"]
audit["evaluation_code_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
audit["loader_code_sha256"] = hashlib.sha256(
    (Path(__file__).parent / "switch_policy_loader.py").read_bytes()
).hexdigest()
audit["seed"] = cfg.seed
audit["event_configuration"] = json.loads(json.dumps(cfg.events.to_dict(), default=str))
if args.enable_cameras:
    cfg.viewer.origin_type = "world"
    cfg.viewer.asset_name = "robot"
    cfg.viewer.env_index = 0
    cfg.viewer.resolution = (960, 720)
    cfg.viewer.eye = (0.9, 0.9, 0.45)
    cfg.viewer.lookat = (0.0, 0.0, 0.08)
simenv = gym.make(
    "Isaac-Goal-Flat-Hexapod-Binary-v0", cfg=cfg, render_mode="rgb_array" if args.enable_cameras else None
)
base = simenv.unwrapped
assert base.single_observation_space["policy"].shape == (32,)
assert base.single_observation_space["critic"].shape == (35,)
assert abs(base.step_dt - 0.02) < 1e-10
save_runtime_audit(base, audit, str(output.with_suffix(".audit.json")))
np.testing.assert_allclose(audit["loaded_material_min"][:2], [evaluation_friction, evaluation_friction], rtol=1e-6)
np.testing.assert_allclose(audit["loaded_material_max"][:2], [evaluation_friction, evaluation_friction], rtol=1e-6)
steps = config.get("steps", 300)
body_length_m = float(config.get("body_length_m", 0.315))
gait_period_s = float(config.get("gait_period_s", 1.0))
assert body_length_m > 0 and gait_period_s > 0
if args.enable_cameras:
    from isaaclab_physx.renderers.kit_viewport_utils import set_kit_renderer_camera_view

    simenv = gym.wrappers.RecordVideo(
        simenv,
        video_folder=str(output.parent / "videos" / output.stem),
        step_trigger=lambda step: step == 0,
        video_length=steps + 1,
        disable_logger=True,
    )
env = DiscreteBitsActionWrapper(simenv)
policy_kind = config.get("policy", "checkpoint")
net = GreedyPolicy(config["checkpoint"], config["method_id"], base.device) if policy_kind == "checkpoint" else None
table = torch.tensor(audit["profile"]["actions"], device=base.device)
obs, _ = env.reset()
base.scene.update(dt=base.step_dt)
base.command_manager.compute(dt=0.0)
obs = base.observation_manager.compute()


def select():
    if net is not None:
        return net(obs["policy"])
    if policy_kind == "reference":
        return table[(base.episode_length_buf % 50).long()]
    if policy_kind == "stand":
        return torch.full((base.num_envs,), 63, device=base.device, dtype=torch.long)
    if policy_kind == "random":
        return torch.randint(64, (base.num_envs,), device=base.device)
    raise ValueError(policy_kind)


def camera():
    if not args.enable_cameras:
        return
    origin = tensor(base.scene["robot"].data.root_pos_w)[0].cpu().numpy().astype(float)
    eye = origin + np.array([0.9, 0.9, 0.45])
    look = origin + np.array([0.0, 0.0, 0.08])
    base.sim.set_camera_view(eye=tuple(eye), target=tuple(look))
    set_kit_renderer_camera_view(eye=eye, target=look, camera_prim_path="/OmniverseKit_Persp")


camera()
obs, _, _, _, _ = env.step(select())
start = tensor(base.scene["robot"].data.root_pos_w).clone()
prev = start.clone()
initial_q = tensor(base.scene["robot"].data.root_quat_w).cpu().numpy().copy()
previous_command = base.action_manager.action.gt(0).long()
previous_command = (previous_command * (2 ** torch.arange(6, device=base.device))).sum(-1).cpu().numpy()
phase_history = []
height_history = []
trajectory = []
rewards = []
qualities = []
actions = []
joints = []
falls = []
torso_contacts = []
body_contacts = []
body_positions = []
body_quaternions = []
root_quaternions = []
root_velocities = []
ever_fall = torch.zeros(base.num_envs, device=base.device, dtype=torch.bool)
ever_success = ever_fall.clone()
first_fall = torch.full((base.num_envs,), float(steps), device=base.device)
path = torch.zeros(base.num_envs, device=base.device)
max_jump = 0.0
resets = 0
term_totals = {k: 0.0 for k in base.reward_manager.active_terms}
foot_ids, _ = base.scene.sensors["contact_forces"].find_bodies(
    ["FrontRight", "FrontLeft", "MiddleRight", "MiddleLeft", "BackRight", "BackLeft"], preserve_order=True
)
assert len(foot_ids) == 6
foot_history = []
foot_state = tensor(base.scene.sensors["contact_forces"].data.net_forces_w)[:, foot_ids].norm(dim=-1) > 1.0
initial_foot_state = foot_state.cpu().numpy().copy()
for step in range(steps):
    camera()
    phase_history.append(base.episode_length_buf.cpu().numpy().copy())
    chosen = select()
    obs, reward, term, trunc, _ = env.step(chosen)
    force = tensor(base.scene.sensors["contact_forces"].data.net_forces_w)[:, foot_ids].norm(dim=-1)
    foot_state = torch.where(foot_state, force > 0.5, force > 1.0)
    foot_history.append(foot_state.cpu().numpy().copy())
    root = tensor(base.scene["robot"].data.root_pos_w).clone()
    height_history.append(root[:, 2].cpu().numpy().copy())
    delta = root - prev
    max_jump = max(max_jump, delta.norm(dim=1).max().item())
    path += delta[:, :2].norm(dim=1)
    prev = root
    if config["reward"] == "J":
        reward_metrics(base, "B2")
    failure = base._overnight_failure
    first_fall = torch.where(failure & ~ever_fall, float(step), first_fall)
    ever_fall |= failure
    ever_success |= base._overnight_success & ~ever_fall
    resets += int((term | trunc).sum().item())
    assert torch.isfinite(reward).all() and all(torch.isfinite(v).all() for v in obs.values())
    parts = {
        name: base.reward_manager._step_reward[:, i] * base.step_dt
        for i, name in enumerate(base.reward_manager.active_terms)
    }
    torch.testing.assert_close(sum(parts.values()), reward, rtol=2e-5, atol=2e-6)
    for name, value in parts.items():
        term_totals[name] += value.mean().item()
    assert not (base._overnight_success & failure).any()
    trajectory.append((root - start).cpu().numpy())
    rewards.append(reward.cpu().numpy())
    qualities.append(base._overnight_quality.cpu().numpy())
    actions.append(chosen.cpu().numpy())
    joints.append(tensor(base.scene["robot"].data.joint_pos).cpu().numpy().copy())
    falls.append(failure.cpu().numpy())
    contact = (
        tensor(base.scene.sensors["contact_forces"].data.net_forces_w_history)[:, :, base._overnight_indices["torso"]]
        .norm(dim=-1)
        .amax(dim=1)
        > 1.0
    )
    torso_contacts.append(contact.any(dim=1).cpu().numpy())
    body_contacts.append(contact.cpu().numpy())
    body_positions.append(tensor(base.scene["robot"].data.body_pos_w)[0].cpu().numpy().copy())
    body_quaternions.append(tensor(base.scene["robot"].data.body_quat_w)[0].cpu().numpy().copy())
    root_quaternions.append(tensor(base.scene["robot"].data.root_quat_w).cpu().numpy().copy())
    root_velocities.append(tensor(base.scene["robot"].data.root_lin_vel_w).cpu().numpy().copy())
assert resets == 0, "Reset invalidates physical measurement"
distance = (prev - start)[:, 0].cpu().numpy()
continuous = max_jump < 0.05
hist = np.bincount(np.array(actions).flatten(), minlength=64)
probs = hist / hist.sum()
entropy = float(-(probs[probs > 0] * np.log(probs[probs > 0])).sum())
window_s = steps * base.step_dt
n_cycles = window_s / gait_period_s
result = {
    **{k: v for k, v in config.items() if k != "output"},
    "evaluation_id": output.stem,
    "source_commit": audit["source_commit"],
    "evaluation_code_sha256": audit["evaluation_code_sha256"],
    "policy_audit": None if net is None else net.audit,
    "warmup_steps": 1,
    "resets": resets,
    "max_step_jump_m": max_jump,
    "measurement_valid": continuous,
    "valid": continuous and not ever_fall.any().item(),
    "x_displacement_m": float(distance.mean()),
    "x_disp_std_m": float(distance.std(ddof=1)),
    "bl_per_cycle": float(distance.mean() / (body_length_m * n_cycles)),
    "fall_rate": ever_fall.float().mean().item(),
    "survival_s": first_fall.mean().item() * base.step_dt,
    "reach_rate": ever_success.float().mean().item(),
    "action_entropy_nats": entropy,
    "dominant_action_fraction": float(probs.max()),
    "action_counts": hist.tolist(),
    "mean_path_length_m": path.mean().item(),
    "body_contact_frame_fraction": float(np.mean(torso_contacts)),
    "quality_mean": float(np.mean(qualities)),
    "reward_terms_total": term_totals,
    "reward_total_diagnostic": float(np.array(rewards).sum(axis=0).mean()),
    "reward_note": "No-reset diagnostic; not training episodic return",
    "per_environment_x_m": distance.tolist(),
    "per_environment_fall": ever_fall.cpu().numpy().tolist(),
}
result["window_s"] = float(window_s)
result["step_dt"] = float(base.step_dt)
result["n_cycles"] = float(n_cycles)
result["body_length_m"] = body_length_m
result["gait_period_s"] = gait_period_s
result["body_names"] = base.scene["robot"].body_names
np.savez_compressed(
    output.with_suffix(".npz"),
    phase_steps=np.array(phase_history),
    previous_command=previous_command,
    initial_quaternion=initial_q,
    root_height=np.array(height_history),
    foot_contact=np.array(foot_history),
    initial_foot_contact=initial_foot_state,
    trajectory=np.array(trajectory),
    rewards=np.array(rewards),
    quality=np.array(qualities),
    actions=np.array(actions),
    joint_positions=np.array(joints),
    failures=np.array(falls),
    torso_contact=np.array(torso_contacts),
    body_contact_parts=np.array(body_contacts),
    body_positions_env0=np.array(body_positions),
    body_quaternions_env0=np.array(body_quaternions),
    root_quaternions=np.array(root_quaternions),
    root_velocity_world=np.array(root_velocities),
)
contact_array = np.concatenate([initial_foot_state[None], np.array(foot_history)], axis=0)
result["contact_switch_per_leg_s"] = float(
    np.count_nonzero(np.diff(contact_array.astype(int), axis=0)) / (base.num_envs * 6 * window_s)
)
action_array = np.array(actions).astype(int)
bits = (action_array[..., None] >> np.arange(6)) & 1
result["action_switch_per_leg_s"] = float(
    np.count_nonzero(np.diff(bits, axis=0)) / (base.num_envs * 6 * ((steps - 1) * base.step_dt))
)
result["contact_threshold_N"] = 1.0
result["contact_release_N"] = 0.5
from switch_metrics import summarize

result["screening"] = summarize(
    actions,
    previous_command,
    phase_history,
    trajectory,
    initial_q,
    root_quaternions,
    height_history,
    falls,
    torso_contacts,
    base.step_dt,
    gait_period_s,
    body_length_m,
)
result["screening_code_sha256"] = hashlib.sha256((Path(__file__).parent / "switch_metrics.py").read_bytes()).hexdigest()
result["raw_trajectory_path"] = str(output.with_suffix(".npz"))
result["historical_world_x_bl_per_cycle"] = result["bl_per_cycle"]
result["bl_per_cycle"] = result["screening"]["forward_bl_per_cycle"]
result["basic_screen_pass"] = bool(continuous and result["screening"]["basic_screen_pass"])
result["best_eligible"] = False
result["best_eligibility_rule"] = (
    "Full qualification pending dwell and straightness thresholds. "
    "basic_screen_pass is separate, not deployment clearance."
)
result["checkpoint_sha256"] = (
    hashlib.sha256(Path(config["checkpoint"]).read_bytes()).hexdigest() if config.get("checkpoint") else None
)
env.close()
if config.get("video"):
    videos = list((output.parent / "videos" / output.stem).glob("*.mp4"))
    if videos:
        result["video_url"] = "videos/" + output.stem + "/" + videos[0].name
output.write_text(json.dumps(result, indent=2))
print(json.dumps(result), flush=True)
app.close()
