# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward-tuning variants layered on top of the hexapod goal-reaching environment.

Base: `HexapodGoalEnvCfg` (hexapod_goal_env_cfg.py, unmodified).

These configs carry the flat-locomotion reward shaping (longer swing phase, smoother
actions, optional foot-slide penalty) over to the goal-reaching task.

Two details the base class forces us to handle explicitly:

1. `HexapodGoalEnvCfg` only tweaks ``rewards.feet_air_time.weight``/threshold, so
   the term is rebuilt here (rather than edited in place) to also swap its
   ``command_name`` param (see point 2) in one shot.
2. The default term in ``velocity_env_cfg.py`` hardcodes
   ``command_name="base_velocity"``. The goal environment replaces that command with
   ``pose_command``, so reusing the default parameters raises
   ``KeyError: base_velocity`` at startup.

Variants (each trained for 3000 iterations, 4096 envs, single seed)
------------------------------------------------------------------
| Class                             | action_rate_l2 | feet_air_time | feet_slide |
|-----------------------------------|----------------|---------------|------------|
| HexapodGoalBigStepEnvCfg          | -0.03          | 2.5 @ 0.16 s  | none       |
| HexapodGoalBigStepSlide06EnvCfg   | -0.03          | 2.5 @ 0.16 s  | -0.6       |
| HexapodGoalBigStepSlide035EnvCfg  | -0.03          | 2.5 @ 0.16 s  | -0.35      |
| HexapodGoalBigStepMinimalEnvCfg   | inherited      | 2.5 @ 0.16 s  | none       |

The `feet_air_time` weight/threshold and the `feet_slide` term follow
`flat_env_cfg_rshape.py`.

Dependency note
---------------
These variants were trained on the goal environment revision from the
`sihan-physical-goal-training` branch (distance-proportional goal command). The
classes below override reward terms only, so they also apply to the revision
currently on `main`, but the recorded training results correspond to the former.
"""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

import isaaclab_tasks.core.velocity.mdp as mdp

from .hexapod_goal_env_cfg import HexapodGoalEnvCfg, HexapodGoalEnvCfg_PLAY

LEG_BODIES = ["MiddleLeft", "MiddleRight", "BackLeft", "BackRight", "FrontLeft", "FrontRight"]

AIR_TIME_WEIGHT = 2.5
AIR_TIME_THRESHOLD = 0.16
ACTION_RATE_WEIGHT = -3.0e-2


def _feet_air_time_term() -> RewTerm:
    """Rebuild the swing-phase reward for the goal task (see module docstring)."""
    return RewTerm(
        func=mdp.feet_air_time,
        weight=AIR_TIME_WEIGHT,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=LEG_BODIES),
            "command_name": "pose_command",
            "threshold": AIR_TIME_THRESHOLD,
        },
    )


def _feet_slide_term(weight: float) -> RewTerm:
    """Penalize foot motion while the foot is in contact with the ground."""
    return RewTerm(
        func=mdp.feet_slide,
        weight=weight,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=LEG_BODIES),
            "asset_cfg": SceneEntityCfg("robot", body_names=LEG_BODIES),
        },
    )


@configclass
class HexapodGoalBigStepEnvCfg(HexapodGoalEnvCfg):
    """Longer swing phase and smoother actions; no foot-slide penalty."""

    def __post_init__(self):
        super().__post_init__()

        self.rewards.action_rate_l2.weight = ACTION_RATE_WEIGHT
        self.rewards.feet_air_time = _feet_air_time_term()


@configclass
class HexapodGoalBigStepSlide06EnvCfg(HexapodGoalBigStepEnvCfg):
    """Big-step shaping with the flat-task foot-slide weight (-0.6)."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.feet_slide = _feet_slide_term(-0.6)


@configclass
class HexapodGoalBigStepSlide035EnvCfg(HexapodGoalBigStepEnvCfg):
    """Big-step shaping with a reduced foot-slide weight (-0.35)."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.feet_slide = _feet_slide_term(-0.35)


@configclass
class HexapodGoalBigStepMinimalEnvCfg(HexapodGoalEnvCfg):
    """Minimal-diff arm: swing-phase reward only, base action_rate weight kept."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.feet_air_time = _feet_air_time_term()


@configclass
class HexapodGoalBigStepEnvCfg_PLAY(HexapodGoalEnvCfg_PLAY):
    """Play/evaluation variant of :class:`HexapodGoalBigStepEnvCfg`."""

    def __post_init__(self):
        super().__post_init__()

        self.rewards.action_rate_l2.weight = ACTION_RATE_WEIGHT
        self.rewards.feet_air_time = _feet_air_time_term()
