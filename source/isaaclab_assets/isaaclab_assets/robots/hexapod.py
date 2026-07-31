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

HEXAPOD_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"hexapod-assets/USD/Hexapod_Flattened.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=1.0,
            enable_gyroscopic_forces=True,
            max_angular_velocity = 1000.0, #57 rev/min is max unloaded motor speed = 5.96 rad/s **@11.1 V
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
            "FrontLink_Joint": 0.0,
            "BackLink_Joint":0.0,

            #"MiddleLeft_Joint":0.0,
            #"MiddleRight_Joint": 0.0,
            #"BackLeft_Joint": 0.0,
            #"BackRight_Joint": 0.0,
            #"FrontLeft_Joint": 0.0,
            #"FrontRight_Joint": 0.0,

            "MiddleLeft_Joint":-0.47,
            "MiddleRight_Joint": -0.47,
            "BackLeft_Joint": -0.47,
            "BackRight_Joint": -0.47,
            "FrontLeft_Joint": -0.47,
            "FrontRight_Joint": -0.47,
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
        # Spine joints: sinusoidal body undulation sustains high torque at wave peaks against body inertia.
        # Lower stiffness reduces peak torque demand (at stiffness=40, error=0.15 rad before saturation vs 0.075 at 80).
        # The real servo's internal firmware handles gravity loading better than a pure PD controller.
        "body_joints": ImplicitActuatorCfg(
            joint_names_expr=["FrontLink_Joint", "BackLink_Joint"],
            stiffness = 40,
            # stiffness = 80,   # too stiff -- small tracking lag generates huge torques; consistently saturates
            damping = 0.4,
            velocity_limit_sim = 15.0,   # raised: if sin wave step changes require >5.5 rad/s, velocity cap accumulates lag → torque saturates regardless of effort limit
            # velocity_limit_sim = 5.5,  # original -- may be binding constraint for body undulation
            effort_limit_sim = 4.5,
        ),
        # Leg joints: intermittent ground contact, shorter duration at peak torque
        "leg_joints": ImplicitActuatorCfg(
            joint_names_expr=[
                "MiddleLeft_Joint",
                "MiddleRight_Joint",
                "BackLeft_Joint",
                "BackRight_Joint",
                "FrontLeft_Joint",
                "FrontRight_Joint",
            ],
            stiffness = 80,
            # stiffness = 37,   # original -- too low: max correctable error = 1.4/37 = 0.038 rad before torque saturation
            damping = 0.9,
            # damping = 0.32,   # original
            velocity_limit_sim = 6.0,
            effort_limit_sim = 4.5,
            # effort_limit_sim = 1.4,  # XL430-W250-T rated stall torque at 12V (physical spec)
        ),
    },
)
"""Configuration for hexapod robot."""
