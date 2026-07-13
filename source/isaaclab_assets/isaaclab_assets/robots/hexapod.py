# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the LiBR Hexapod robot."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Configuration
##

MOTOR_STALL_TORQUE_NM = 1.4
MOTOR_NO_LOAD_SPEED_RAD_S = 5.97

HEXAPOD_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"hexapod-assets/USD/Hexapod_Flattened.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=1.0,
            enable_gyroscopic_forces=True,
            max_angular_velocity = 1000.0,
            max_linear_velocity = 1000.0,
            #enable_ccd=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=1,
            #sleep_threshold=0.1,
            #stabilization_threshold=0.01, #was 0.001 -- check this value
        ),
        #copy_from_source=False,
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.2),
        joint_pos={
            #".*": 0.0,
            "FrontLink": 0.0,
            "BackLink":0.0,

            #"MiddleLeft":0.0,
            #"MiddleRight": 0.0,
            #"BackLeft": 0.0,
            #"BackRight": 0.0,
            #"FrontLeft": 0.0,
            #"FrontRight": 0.0,

            "MiddleLeft":-0.47,
            "MiddleRight": -0.47,
            "BackLeft": -0.47,
            "BackRight": -0.47,
            "FrontLeft": -0.47,
            "FrontRight": -0.47,
            #"Front Middle Joint": 1.57079,
            #"Front Left Joint": 0,
            #"Front Right Joint": 0,
            #"Back Middle Joint": 1.57079,
            #"Back Left Joint": 0,
            #"Back Right Joint": 0,
            #"Middle Left Joint": 0,
            #"Middle Right Joint": 0,
            
        },
    ),
    soft_joint_pos_limit_factor = 0.9,
    actuators={
        # Spine joints use the same motor-rated output speed and stall torque caps as the legs.
        "body_joints": ImplicitActuatorCfg(
            joint_names_expr=["FrontLink", "BackLink"],
            stiffness = 40,
            damping = 0.4,
            velocity_limit_sim = MOTOR_NO_LOAD_SPEED_RAD_S,
            effort_limit_sim = MOTOR_STALL_TORQUE_NM,
        ),
        # Leg joints: motor-rated hard limits keep RL from learning impossible ground impulses.
        "leg_joints": ImplicitActuatorCfg(
            joint_names_expr=["MiddleLeft", "MiddleRight", "BackLeft", "BackRight", "FrontLeft", "FrontRight"],
            stiffness = 80,
            damping = 0.9,
            velocity_limit_sim = MOTOR_NO_LOAD_SPEED_RAD_S,
            effort_limit_sim = MOTOR_STALL_TORQUE_NM,
        ),
    },
)
"""Configuration for hexapod robot."""
