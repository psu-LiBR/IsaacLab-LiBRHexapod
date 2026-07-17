"""Typed loader for deployment.yaml -- the single place hardware facts live.

Every other module in this package receives already-typed config objects and
never touches YAML directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import yaml

from .imu import DEFAULT_MAX_STALENESS_S


@dataclass(frozen=True)
class SerialCfg:
    port: str
    baud_rate: int
    protocol_version: float


@dataclass(frozen=True)
class JointsCfg:
    sim_order: list[str]
    real_order: list[str]
    motor_ids: dict[str, int]
    correction_group: dict[str, str]
    ticks_per_rev: int
    zero_tick: dict[str, int]
    soft_limits_rad: dict[str, tuple[float, float]]


@dataclass(frozen=True)
class ImuCfg:
    topic: str
    mount_offset_quat: tuple[float, float, float, float]
    negate_gyro_z: bool
    # Optional: max age (seconds) of the last received /imu sample before
    # RosImuReader.read() raises ImuStaleError instead of returning a frozen
    # stale reading forever. Absent from older/calibrated deployment.yaml files
    # is fine -- this must keep parsing them unchanged, so it defaults below.
    max_staleness_s: float = DEFAULT_MAX_STALENESS_S


@dataclass(frozen=True)
class ControlCfg:
    rate_hz: float
    action_scale: float
    q_default_sim: dict[str, float]
    soft_start_seconds: float
    soft_stop_seconds: float
    action_scale_multiplier: float


@dataclass(frozen=True)
class CommandsCfg:
    velocity: dict[str, float]
    goal: dict[str, float]


@dataclass(frozen=True)
class DeploymentConfig:
    serial: SerialCfg
    joints: JointsCfg
    imu: ImuCfg
    control: ControlCfg
    commands: CommandsCfg


def _require(d: dict, key: str, path: str):
    if key not in d:
        raise ValueError(f"deployment config missing required key: {path}.{key}")
    return d[key]


def load_deployment_config(path: str) -> DeploymentConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)

    serial_raw = _require(raw, "serial", "")
    serial = SerialCfg(
        port=_require(serial_raw, "port", "serial"),
        baud_rate=int(_require(serial_raw, "baud_rate", "serial")),
        protocol_version=float(_require(serial_raw, "protocol_version", "serial")),
    )

    joints_raw = _require(raw, "joints", "")
    encoder_raw = _require(joints_raw, "encoder", "joints")
    joints = JointsCfg(
        sim_order=list(_require(joints_raw, "sim_order", "joints")),
        real_order=list(_require(joints_raw, "real_order", "joints")),
        motor_ids=dict(_require(joints_raw, "motor_ids", "joints")),
        correction_group=dict(_require(joints_raw, "correction_group", "joints")),
        ticks_per_rev=int(_require(encoder_raw, "ticks_per_rev", "joints.encoder")),
        zero_tick={k: int(v) for k, v in _require(encoder_raw, "zero_tick", "joints.encoder").items()},
        soft_limits_rad={
            k: (float(v[0]), float(v[1]))
            for k, v in _require(joints_raw, "soft_limits_rad", "joints").items()
        },
    )

    imu_raw = _require(raw, "imu", "")
    quat = _require(imu_raw, "mount_offset_quat", "imu")
    imu = ImuCfg(
        topic=_require(imu_raw, "topic", "imu"),
        mount_offset_quat=(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])),
        negate_gyro_z=bool(_require(imu_raw, "negate_gyro_z", "imu")),
        # Optional key -- keeps existing (already-calibrated) deployment.yaml
        # files, which predate this field, parsing unchanged.
        max_staleness_s=float(imu_raw["max_staleness_s"]) if "max_staleness_s" in imu_raw else DEFAULT_MAX_STALENESS_S,
    )

    control_raw = _require(raw, "control", "")
    control = ControlCfg(
        rate_hz=float(_require(control_raw, "rate_hz", "control")),
        action_scale=float(_require(control_raw, "action_scale", "control")),
        q_default_sim={k: float(v) for k, v in _require(control_raw, "q_default_sim", "control").items()},
        soft_start_seconds=float(_require(control_raw, "soft_start_seconds", "control")),
        soft_stop_seconds=float(_require(control_raw, "soft_stop_seconds", "control")),
        action_scale_multiplier=float(_require(control_raw, "action_scale_multiplier", "control")),
    )

    commands_raw = _require(raw, "commands", "")
    commands = CommandsCfg(
        velocity=dict(_require(commands_raw, "velocity", "commands")),
        goal=dict(_require(commands_raw, "goal", "commands")),
    )

    return DeploymentConfig(serial=serial, joints=joints, imu=imu, control=control, commands=commands)
