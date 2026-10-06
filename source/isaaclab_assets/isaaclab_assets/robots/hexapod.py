# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the LiBR Hexapod robot."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets import ArticulationCfg

##
# Configuration
##

HEXAPOD_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        # usd_path="hexapod-assets/USD/Hexapod_Flattened.usd",
        usd_path="hexapod-assets/USD/HexapiFlattened.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=1.0,
            enable_gyroscopic_forces=True,
            max_angular_velocity=1000.0,  # XL430 no-load: 57 rev/min = 5.97 rad/s @11.1 V (61 rev/min = 6.39 @12 V)
            max_linear_velocity=1000.0,
            # enable_ccd=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=1,
            # sleep_threshold=0.1,
            # stabilization_threshold=0.01, #was 0.001 -- check this value
        ),
        # copy_from_source=False,
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.2),
        joint_pos={
            # ".*": 0.0,
            # For Trad Hexapod Implementation ------
            # "FrontLink_Joint": 0.0,
            # "BackLink_Joint": 0.0,
            # "MiddleLeft_Joint": -0.47,
            # "MiddleRight_Joint": -0.47,
            # "BackLeft_Joint": -0.47,
            # "BackRight_Joint": -0.47,
            # "FrontLeft_Joint": -0.47,
            # "FrontRight_Joint": -0.47,
            # For HEXAPI Implementation -------------
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
        # ------------------------------------------------------------------------------------
        # Dynamixel XL430-W250-T explicit DC-motor model (sim2real-oriented).
        #
        # Datasheet (Robotis e-Manual, https://emanual.robotis.com/docs/en/dxl/x/xl430-w250/).
        # 11.1 V is the Robotis-recommended voltage and this robot's 3S LiPo pack, so the
        # 11.1 V column is used DIRECTLY -- no further voltage scaling:
        #   stall torque      1.4 N.m @ 1.3 A    no-load speed   57 rev/min = 5.969 rad/s
        #   gear ratio        258.5 : 1          resolution      4096 pulse/rev
        #   operating voltage 6.5 - 12.0 V       weight          57.2 g   standby current 52 mA
        #   position PID control-table defaults: P Gain=640, I Gain=0, D Gain=4000
        #     (KPP = P/128 = 5.0, KPI = I/65536 = 0, KPD = D/16 = 250; goal-PWM limit 885)
        #   -> velocity_limit    = 5.97   (== no-load speed; torque-speed curve x-intercept)
        #   -> saturation_effort = 1.4    (== stall torque)
        #  (12 V column, for reference: 1.5 N.m stall, 61 rev/min = 6.39 rad/s.)
        #
        # DCMotor enforces the linear four-quadrant torque-speed curve
        #   tau_max(qd) = clip(saturation_effort * (1 - qd / velocity_limit), -inf, effort_limit)
        # so deliverable torque collapses to ~0 as the joint approaches no-load speed, exactly
        # like the real servo. This is the main sim2real gain over the previous ImplicitActuator
        # (which could deliver full torque at any speed). Consequence, by design: the sim will
        # NOT perfectly track an aggressive open-loop reference gait -- the real robot cannot
        # either.
        #
        # effort_limit = saturation_effort (1.4): the e-Manual gives no separate continuous /
        # thermal torque rating, so the curve is a pure stall->no-load line with no extra flat
        # derate. For long continuous operation a conservative thermal choice would be ~0.6-0.9.
        #
        # Why effort_limit_sim is no longer 4.5: under the implicit model a single PhysX clamp
        # saw (stiffness*err + damping*qd) combined, and 4.5 (3.2x physical stall) was inflated
        # headroom for the damping term. DCMotor clips the physical envelope separately, so
        # effort_limit is now the real motor number and effort_limit_sim / velocity_limit_sim
        # are left unset (explicit-actuator default -> no solver double-clip; the torque-speed
        # curve governs joint speed).
        #
        # Non-datasheet estimates (the e-Manual lists neither rotor inertia nor running
        # no-load current, so these are engineering estimates -- tune against bench data):
        #   armature = 1.3e-3 kg.m^2  ~= J_rotor(~2e-8) * 258.5^2 (reflected rotor inertia).
        #     Dominant inertia at this gear ratio (~20x the leg-link inertia) AND needed for
        #     explicit-integration stability at sim.dt = 5e-3 s (bare-leg k_max ~ 4*I/dt^2).
        #   friction 0.04 / dynamic_friction 0.03 N.m: geartrain Coulomb / breakaway. Isaac
        #     Sim 5.0+ models these as a torque, not a coefficient.
        #
        # Stiffness / damping = the XL430 firmware PID mapped to physical joint units:
        #   k_p = saturation_effort * KPP * (4096 / 2pi) / 885,  KPP = P Gain / 128
        #     P Gain = 640  -> k_p = 5.16   (firmware default)
        #     P Gain ~ 1240 -> k_p = 10     (spine)
        #     P Gain ~ 6200 -> k_p = 50     (legs)
        #   k_d = saturation_effort * KPD * T_loop * (4096 / 2pi) / 885,  KPD = D Gain / 16
        #     D Gain = 4000, T_loop ~ 1.4 ms -> k_d ~ 0.35  (firmware default D term, both
        #     groups; T_loop is the internal-loop period, not published -- k_d is uncertain
        #     to a factor of ~2 and was cross-checked against the tripod-gait sweep).
        # Legs use k_p = 50 (a mild bump over the P Gain=640 default, motivated by open-loop
        # tripod-gait tracking tests and trivially settable on the real servo). The spine keeps
        # k_p = 10: the undulation is smooth and lightly loaded, and the torque-speed curve now
        # handles the peak-torque saturation that the old stiffness=10 was a workaround for.
        # NOTE: the spine sinusoid peaks near ~5.3 rad/s (~ the no-load speed), so it will
        # under-track its commanded amplitude -- physically accurate for the real servo.
        # ------------------------------------------------------------------------------------
        "body_joints": DCMotorCfg(
            joint_names_expr=["FrontLink_Joint", "BackLink_Joint"],
            saturation_effort=1.4,
            effort_limit=1.4,
            velocity_limit=5.97,
            stiffness=10.0,
            damping=0.35,
            armature=1.3e-3,
            friction=0.04,
            dynamic_friction=0.03,
        ),
        "leg_joints": DCMotorCfg(
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
        ),
    },
)
"""Configuration for hexapod robot."""
