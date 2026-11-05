# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the LiBR Hexapod robot."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets import ArticulationCfg

from .hexapod_actuators import DelayedDCMotorCfg

##
# Servo command delay
##

HEXAPOD_ACTUATOR_DELAY = True
"""Whether the servos lag their commands (sim-to-real latency). Set to False for the instantaneous DCMotor."""

HEXAPOD_LEG_DELAY_RANGE_STEPS = (8, 12)
"""Leg command delay ``(min, max)`` [physics steps], redrawn per env at reset (40-60 ms)."""

HEXAPOD_SPINE_DELAY_RANGE_STEPS = (11, 15)
"""Spine command delay ``(min, max)`` [physics steps], redrawn per env at reset (55-75 ms)."""

# Fitted to the 2026-10-19 robot logs with the DCMotor parameters below unchanged (best pure delay ~10 steps for
# the legs, ~13 for the spine). A delay-only stop-gap: part of it may be servo read-back latency. It sits inside
# the actuator model, so it hits RL actions and scripted motion alike. Play / eval configs pin it to the middle
# of each range (legs 10, spine 13) with ``with_fixed_actuator_delay``.


def _delay_kwargs(delay_range: tuple[int, int]) -> dict:
    """Return the ``DelayedDCMotorCfg`` delay fields for ``delay_range`` (empty if the delay is switched off)."""
    return {"min_delay": delay_range[0], "max_delay": delay_range[1]} if HEXAPOD_ACTUATOR_DELAY else {}


_ACTUATOR_CFG = DelayedDCMotorCfg if HEXAPOD_ACTUATOR_DELAY else DCMotorCfg

##
# Configuration
##

HEXAPOD_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path="hexapod-assets/USD/HexapiFlattened.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=1.0,
            enable_gyroscopic_forces=True,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=1,
        ),
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.2),
        joint_pos={
            # legs at +0.47 rad (HexapI convention)
            "FrontLink_Joint": 0.0,
            "BackLink_Joint": 0.0,
            "MiddleLeft_Joint": 0.47,
            "MiddleRight_Joint": 0.47,
            "BackLeft_Joint": 0.47,
            "BackRight_Joint": 0.47,
            "FrontLeft_Joint": 0.47,
            "FrontRight_Joint": 0.47,
        },
    ),
    soft_joint_pos_limit_factor=1.0,
    actuators={
        # Dynamixel XL430-W250-T as an explicit DC motor (datasheet: https://emanual.robotis.com/docs/en/dxl/x/xl430-w250/).
        # The 11.1 V column is used directly (this robot's 3S LiPo): stall torque 1.4 N.m, no-load speed
        # 5.97 rad/s, gear ratio 258.5:1. DCMotor limits torque linearly with speed,
        #   tau_max(qd) = clip(saturation_effort * (1 - qd / velocity_limit), -inf, effort_limit),
        # so torque collapses near no-load speed like the real servo and the sim will not perfectly track an
        # aggressive open-loop gait. effort_limit_sim / velocity_limit_sim stay unset (no solver double-clip).
        #
        # Estimates (the datasheet lists neither): armature 1.3e-3 kg.m^2 is the reflected rotor inertia
        # (~2e-8 * 258.5^2), which also keeps explicit integration stable at sim.dt = 5e-3; friction 0.04 /
        # dynamic_friction 0.03 N.m model geartrain breakaway as a torque.
        #
        # Gains map the firmware PID to joint units: k_p = saturation_effort * (P_gain/128) * (4096/2pi) / 885
        # (P gain 640 -> 5.16, 1240 -> 10, 6200 -> 50), and k_d ~ 0.35 from D gain 4000 (internal loop period
        # unpublished, so uncertain to ~2x). Legs use k_p = 50, a bump over the firmware default that tracks
        # open-loop tripod gaits better. The spine keeps 10: its undulation is smooth and the torque-speed curve
        # now limits the peaks. The spine wave peaks near the no-load speed, so it under-tracks its amplitude,
        # as on the real servo.
        "body_joints": _ACTUATOR_CFG(
            joint_names_expr=["FrontLink_Joint", "BackLink_Joint"],
            saturation_effort=1.4,
            effort_limit=1.4,
            velocity_limit=5.97,
            stiffness=10.0,
            damping=0.35,
            armature=1.3e-3,
            friction=0.04,
            dynamic_friction=0.03,
            **_delay_kwargs(HEXAPOD_SPINE_DELAY_RANGE_STEPS),
        ),
        "leg_joints": _ACTUATOR_CFG(
            joint_names_expr=[
                "MiddleLeft_Joint",
                "MiddleRight_Joint",
                "BackLeft_Joint",
                "BackRight_Joint",
                "FrontLeft_Joint",
                "FrontRight_Joint",
            ],
            saturation_effort=1.4,
            effort_limit=1.4,
            velocity_limit=5.97,
            stiffness=50.0,
            damping=0.35,
            armature=1.3e-3,
            friction=0.04,
            dynamic_friction=0.03,
            **_delay_kwargs(HEXAPOD_LEG_DELAY_RANGE_STEPS),
        ),
    },
)
"""Configuration for hexapod robot."""
