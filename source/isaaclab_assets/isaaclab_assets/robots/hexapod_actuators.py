# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hexapod actuator models for sim-to-real latency.

The real Dynamixel servos start moving several control periods after a command, while the simulated
:class:`~isaaclab.actuators.DCMotor` responds in the same physics step. :class:`DelayedDCMotor` adds a
per-environment command delay in front of the unchanged torque-speed model, identically for every joint in the
group. ``HEXAPOD_CFG`` uses :class:`DelayedDCMotorCfg` for both groups; evaluation configs pin the delay with
:func:`with_fixed_actuator_delay`, and :func:`with_servo_model` builds the opt-in servo-realistic variant. None
of these touch the shared robot asset configuration.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from copy import deepcopy

import torch

from isaaclab.actuators import DCMotor, DCMotorCfg
from isaaclab.utils import DelayBuffer
from isaaclab.utils.configclass import configclass
from isaaclab.utils.types import ArticulationActions


class DelayedDCMotor(DCMotor):
    """DC motor actuator whose position / velocity / effort commands are applied with a random delay.

    The torque-speed clipping of :class:`~isaaclab.actuators.DCMotor` is kept as is.  The delay is
    implemented with one :class:`~isaaclab.utils.DelayBuffer` per command type (as in
    :class:`~isaaclab.actuators.DelayedPDActuator`): at every call of :meth:`compute` (once per physics
    step) the newest command is pushed to the buffer and the command from ``lag`` calls earlier is used to
    compute the torque.  ``lag`` is drawn per environment, uniformly in ``[min_delay, max_delay]``
    physics steps, at every :meth:`reset`.

    .. note::
        Only the *commands* are delayed.  The measured joint position and velocity passed to the PD law
        and used by the torque-speed curve are the current ones, which is the physically correct
        behaviour: the motor sees its own instantaneous speed, but acts on a stale set-point.

    .. note::
        After a reset the buffer is back-filled with the first command it receives, so the delay does
        not hold the previous default pose for ``lag`` steps; it only shifts subsequent command changes.
    """

    cfg: DelayedDCMotorCfg
    """The configuration for the actuator model."""

    def __init__(self, cfg: DelayedDCMotorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        if cfg.min_delay < 0 or cfg.max_delay < cfg.min_delay:
            raise ValueError(
                f"Invalid actuator delay range: min_delay={cfg.min_delay}, max_delay={cfg.max_delay}."
                " Expected 0 <= min_delay <= max_delay."
            )
        # instantiate the delay buffers (history length = largest lag that can be drawn; 0 is valid)
        self.positions_delay_buffer = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
        self.velocities_delay_buffer = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
        self.efforts_delay_buffer = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)

    def _resolve_env_ids(self, env_ids: Sequence[int] | torch.Tensor | slice | None) -> tuple[torch.Tensor | None, int]:
        """Normalise ``env_ids`` to ``(index tensor or None for all envs, number of envs)``.

        ``env_ids`` may be None / slice(None) (all envs), a slice, a list, or a tensor.  DelayBuffer /
        CircularBuffer index with ``None`` or a sequence (they call ``len()`` on it), so a slice is normalised
        to None (all envs) or to an explicit index tensor.
        """
        if env_ids is None or (isinstance(env_ids, slice) and env_ids == slice(None)):
            return None, self._num_envs
        if isinstance(env_ids, slice):
            env_ids = range(self._num_envs)[env_ids]
        buffer_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self._device).reshape(-1)
        return buffer_ids, buffer_ids.numel()

    def reset(self, env_ids: Sequence[int] | torch.Tensor | slice | None):
        super().reset(env_ids)
        buffer_ids, num_envs = self._resolve_env_ids(env_ids)
        # draw a new random delay [physics steps] for the environments being reset
        time_lags = torch.randint(
            low=self.cfg.min_delay,
            high=self.cfg.max_delay + 1,
            size=(num_envs,),
            dtype=torch.int,
            device=self._device,
        )
        for buffer in (self.positions_delay_buffer, self.velocities_delay_buffer, self.efforts_delay_buffer):
            buffer.set_time_lag(time_lags, buffer_ids)
            buffer.reset(buffer_ids)

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        # delay the commands; the measured joint state is left untouched (see class docstring)
        control_action.joint_positions = self.positions_delay_buffer.compute(control_action.joint_positions)
        control_action.joint_velocities = self.velocities_delay_buffer.compute(control_action.joint_velocities)
        control_action.joint_efforts = self.efforts_delay_buffer.compute(control_action.joint_efforts)
        # DCMotor stores the *current* joint velocity for its torque-speed clipping, then runs the PD law
        return super().compute(control_action, joint_pos, joint_vel)


@configclass
class DelayedDCMotorCfg(DCMotorCfg):
    """Configuration for :class:`DelayedDCMotor`: a DC motor with a random command-to-motion delay."""

    class_type: type[DelayedDCMotor] | str = DelayedDCMotor

    min_delay: int = 0
    """Minimum command delay [physics steps]. Defaults to 0 (no delay)."""

    max_delay: int = 0
    """Maximum command delay [physics steps], inclusive. Defaults to 0 (no delay)."""


class ServoDCMotor(DelayedDCMotor):
    """Delayed DC motor with per-environment randomised gain and supply-voltage scale.

    On top of :class:`DelayedDCMotor`, every :meth:`reset` also draws, per environment:

    * a stiffness scale in ``cfg.stiffness_scale_range`` applied to the configured PD stiffness (the real
      XL430 P gain is a firmware register that was never calibrated to a single value), and
    * a supply-voltage scale in ``cfg.voltage_scale_range`` applied together to ``velocity_limit``,
      ``saturation_effort`` and ``effort_limit`` (the servo's no-load speed and stall torque both scale with
      the battery voltage; the sim's nominal values are the 11.1 V datasheet column, the robot logs show
      10.0-10.4 V under load, i.e. a scale of about 0.91).

    Both ranges default to ``(1.0, 1.0)``, which makes this class identical to :class:`DelayedDCMotor`.
    """

    cfg: ServoDCMotorCfg
    """The configuration for the actuator model."""

    def __init__(self, cfg: ServoDCMotorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        for name in ("stiffness_scale_range", "voltage_scale_range"):
            lo, hi = getattr(cfg, name)
            if not 0.0 < lo <= hi:
                raise ValueError(f"Invalid {name}={getattr(cfg, name)}. Expected 0 < min <= max.")
        # nominal values; the per-env scale is always applied to these, never compounded
        self._nominal_stiffness = self.stiffness.clone()
        self._nominal_velocity_limit = self.velocity_limit.clone()
        self._nominal_effort_limit = self.effort_limit.clone()
        # DCMotor keeps the stall torque as a python float; make it a per-env tensor so it can be scaled per env
        self._saturation_effort = torch.full_like(self.velocity_limit, float(cfg.saturation_effort))
        self._nominal_saturation_effort = self._saturation_effort.clone()

    def reset(self, env_ids: Sequence[int] | torch.Tensor | slice | None):
        super().reset(env_ids)
        idx, num_envs = self._resolve_env_ids(env_ids)
        if idx is None:
            idx = torch.arange(self._num_envs, device=self._device)

        def draw(bounds: tuple[float, float]) -> torch.Tensor:
            lo, hi = bounds
            return (lo + (hi - lo) * torch.rand(num_envs, 1, device=self._device)).to(self.stiffness.dtype)

        stiffness_scale = draw(self.cfg.stiffness_scale_range)
        voltage_scale = draw(self.cfg.voltage_scale_range)
        self.stiffness[idx] = self._nominal_stiffness[idx] * stiffness_scale
        self.velocity_limit[idx] = self._nominal_velocity_limit[idx] * voltage_scale
        self.effort_limit[idx] = self._nominal_effort_limit[idx] * voltage_scale
        self._saturation_effort[idx] = self._nominal_saturation_effort[idx] * voltage_scale
        # the corner velocity depends on all three scaled quantities
        self._vel_at_effort_lim = self.velocity_limit * (1 + self.effort_limit / self._saturation_effort)


@configclass
class ServoDCMotorCfg(DelayedDCMotorCfg):
    """Configuration for :class:`ServoDCMotor`: a delayed DC motor with randomised gain and voltage scale."""

    class_type: type[ServoDCMotor] | str = ServoDCMotor

    stiffness_scale_range: tuple[float, float] = (1.0, 1.0)
    """Per-env multiplier range on :attr:`stiffness`, redrawn at every reset. Defaults to no randomisation."""

    voltage_scale_range: tuple[float, float] = (1.0, 1.0)
    """Per-env multiplier range on velocity limit, stall torque and effort limit (battery voltage / 11.1 V),
    redrawn at every reset. Defaults to no randomisation."""


def with_servo_model(actuators: dict, group_overrides: dict[str, dict]) -> dict:
    """Convert :class:`~isaaclab.actuators.DCMotorCfg` actuator groups to :class:`ServoDCMotorCfg`.

    Like :func:`with_fixed_actuator_delay`, nothing in the input is mutated. Each group is copied field by field
    and then the entries of ``group_overrides[name]`` replace fields of the copy, so a group can change its
    ``stiffness``, ``armature``, ``min_delay`` / ``max_delay``, ``stiffness_scale_range``, ... without the
    shared robot asset being touched.

    Args:
        actuators: Mapping from actuator-group name to its configuration, e.g. ``robot_cfg.actuators``.
        group_overrides: Mapping from actuator-group name to a dict of :class:`ServoDCMotorCfg` field
            overrides. Every group of ``actuators`` must have an entry (an empty dict keeps it as is).

    Returns:
        A new dict with the same keys and :class:`ServoDCMotorCfg` values.

    Raises:
        TypeError: If an entry is not a :class:`~isaaclab.actuators.DCMotorCfg`.
        KeyError: If a group has no entry in ``group_overrides`` or an override names an unknown group.
        ValueError: If an override names a field :class:`ServoDCMotorCfg` does not have.
    """
    missing = sorted(set(actuators) - set(group_overrides))
    unknown = sorted(set(group_overrides) - set(actuators))
    if missing or unknown:
        raise KeyError(
            f"group_overrides must match the actuator groups {sorted(actuators)}; missing={missing}, unknown={unknown}."
        )
    field_names = {f.name for f in dataclasses.fields(ServoDCMotorCfg)}
    servo = {}
    for name, cfg in actuators.items():
        if not isinstance(cfg, DCMotorCfg):
            raise TypeError(f"Actuator group '{name}' is {type(cfg).__name__}, expected a DCMotorCfg.")
        bad = sorted(set(group_overrides[name]) - field_names)
        if bad:
            raise ValueError(f"Actuator group '{name}': unknown ServoDCMotorCfg field(s) {bad}.")
        kwargs = {f.name: deepcopy(getattr(cfg, f.name)) for f in dataclasses.fields(cfg) if f.name != "class_type"}
        kwargs.update(deepcopy(group_overrides[name]))
        servo[name] = ServoDCMotorCfg(**kwargs)
    return servo


def with_fixed_actuator_delay(actuators: dict) -> dict:
    """Pin every :class:`DelayedDCMotorCfg` group to the middle of its delay range.

    Used by play / evaluation configs so every rollout sees the same command delay. Groups that are not
    :class:`DelayedDCMotorCfg` (e.g. when the delay is switched off in ``hexapod.py``) are copied unchanged.
    Nothing in the input is mutated, so the shared robot asset configuration stays untouched.

    Args:
        actuators: Mapping from actuator-group name to its configuration, e.g. ``robot_cfg.actuators``.

    Returns:
        A new dict with the same keys; each delayed group has ``min_delay == max_delay == (min + max) // 2``
        [physics steps].
    """
    fixed = {}
    for name, cfg in actuators.items():
        cfg = deepcopy(cfg)
        if isinstance(cfg, DelayedDCMotorCfg):
            cfg.min_delay = cfg.max_delay = (cfg.min_delay + cfg.max_delay) // 2
        fixed[name] = cfg
    return fixed
