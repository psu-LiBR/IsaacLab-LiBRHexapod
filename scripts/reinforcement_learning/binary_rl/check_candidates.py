# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Independent edge-case checks for the reward adapter; no simulator needed."""

import json
from types import SimpleNamespace as NS

import torch
from overnight_config import BOUNDED_WEIGHTS, COMMON_WEIGHTS, RAW_WEIGHTS, reward_metrics


class Scene(dict):
    pass


def fake_env():
    data = NS(
        root_pos_w=torch.zeros(2, 3),
        projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]]).repeat(2, 1),
        root_lin_vel_b=torch.zeros(2, 3),
        root_lin_vel_w=torch.zeros(2, 3),
        root_ang_vel_b=torch.zeros(2, 3),
        soft_joint_pos_limits=torch.tensor([[[-2.0, 2.0]]]).repeat(2, 8, 1),
        joint_pos=torch.zeros(2, 8),
        applied_torque=torch.zeros(2, 8),
        joint_effort_limits=torch.full((2, 8), 4.5),
        body_lin_vel_w=torch.zeros(2, 9, 3),
        joint_acc=torch.zeros(2, 8),
    )
    # Explicit motors deliberately retain huge solver limits. Reward normalization
    # must use the physical motor cap even when the solver cap changes.
    data.joint_effort_limits[:] = 1e9
    scene = Scene(
        robot=NS(data=data, actuators={"all": NS(joint_indices=list(range(8)), effort_limit=torch.full((2, 8), 1.4))})
    )
    scene.sensors = {"contact_forces": NS(data=NS(net_forces_w_history=torch.zeros(2, 3, 9, 3)))}
    command = NS(pos_command_w=torch.tensor([[0.5, 0.0, 0.0]]).repeat(2, 1), _update_command=lambda: None)
    env = NS(
        scene=scene,
        step_dt=0.02,
        num_envs=2,
        device="cpu",
        common_step_counter=1,
        episode_length_buf=torch.ones(2, dtype=torch.long),
        command_manager=NS(get_term=lambda name: command),
        action_manager=NS(action=torch.ones(2, 6), prev_action=torch.zeros(2, 6)),
        _overnight_indices={
            "legs": list(range(2, 8)),
            "feet": list(range(3, 9)),
            "contact_feet": list(range(3, 9)),
            "torso": [0, 1, 2],
            "center": [0],
        },
    )
    return env


def step(env):
    env.common_step_counter += 1
    env.episode_length_buf += 1


def run_checks():
    for candidate in ("A2", "B2", "C2"):
        env = fake_env()
        weights = COMMON_WEIGHTS | (RAW_WEIGHTS if candidate == "A2" else BOUNDED_WEIGHTS | {"contacts": -1.0})
        metric = reward_metrics(env, candidate)
        assert (metric["progress"] == 0).all() and (metric["switch"] == 0).all()
        env.scene["robot"].data.root_pos_w[:, 0] += 0.02
        env.action_manager.prev_action[:] = env.action_manager.action
        step(env)
        metric = reward_metrics(env, candidate)
        torch.testing.assert_close(metric["progress"] * 5 * 0.02, torch.full((2,), 0.1))
        # A second term read must not consume the delta a second time.
        assert reward_metrics(env, candidate) is metric
        env.scene["robot"].data.root_pos_w[:, 0] -= 0.02
        step(env)
        torch.testing.assert_close(reward_metrics(env, candidate)["progress"] * 5 * 0.02, torch.full((2,), -0.1))
        # Fresh root at goal + physical failure: actual penalty -6, no success.
        env.scene["robot"].data.root_pos_w[:, 0] = 0.5
        env.scene.sensors["contact_forces"].data.net_forces_w_history[0, :, 0, 2] = 2
        step(env)
        metric = reward_metrics(env, candidate)
        assert metric["reach"][0].item() == 0
        assert metric["fall"][0].item() * weights["fall"] * 0.02 == -6
        assert 1 <= metric["reach"][1].item() * weights["reach"] * 0.02 <= 2
        # Episode reset must not turn teleporting backward into progress.
        step(env)
        env.episode_length_buf[:] = 1
        env.scene["robot"].data.root_pos_w[:, 0] = -100
        metric = reward_metrics(env, candidate)
        assert (metric["progress"] == 0).all()
        # Inversion cannot hide from tilt through sin(pi)==0.
        step(env)
        env.scene["robot"].data.projected_gravity_b[:, 2] = 1
        metric = reward_metrics(env, candidate)
        assert (metric["tilt"] > 0).all()
        if candidate != "A2":
            assert all((metric[k] >= 0).all() and (metric[k] <= 1).all() for k in BOUNDED_WEIGHTS if k != "contacts")
        # A front/back contact is costly even without the center termination.
        env = fake_env()
        reward_metrics(env, candidate)
        env.scene.sensors["contact_forces"].data.net_forces_w_history[:, :, 1, 2] = 2
        env.scene["robot"].data.root_pos_w[:, 0] += 0.02
        step(env)
        metric = reward_metrics(env, candidate)
        torch.testing.assert_close(metric["contacts"] * weights["contacts"] * 0.02, torch.full((2,), -0.02))
        expected = 0.0 if candidate == "C2" else 0.1
        torch.testing.assert_close(metric["progress"] * 5 * 0.02, torch.full((2,), expected))
        # The C2 gate is reversible, not a permanent reward latch.
        env.scene.sensors["contact_forces"].data.net_forces_w_history[:] = 0
        env.scene["robot"].data.root_pos_w[:, 0] += 0.02
        step(env)
        torch.testing.assert_close(reward_metrics(env, candidate)["progress"] * 5 * 0.02, torch.full((2,), 0.1))
        env.scene["robot"].data.applied_torque[:] = 1.4
        step(env)
        metric = reward_metrics(env, candidate)
        if candidate != "A2":
            torch.testing.assert_close(metric["torque"], torch.ones(2))
    print(json.dumps({"edge_case_checks": "passed", "candidates": ["A2", "B2", "C2"]}))


if __name__ == "__main__":
    run_checks()
