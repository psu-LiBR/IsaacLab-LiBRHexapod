# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Dynamixel servo bus interface.

`RealDynamixelBus` is a direct port of the working driver in
`hexapod_tripod_adaptive.py` (the Hexapi workspace): same `dynamixel_sdk`
PortHandler/PacketHandler/GroupBulkWrite/GroupBulkRead usage, same control-table
addresses, Protocol 2.0, 1 Mbaud. It adds `read_velocities()` (bulk-reading
ADDR_PRESENT_VELOCITY), which the original gait script never needed since it only
ever wrote goal positions -- the RL policy's `joint_vel` observation term requires
live velocity feedback.

Units: `read_positions()`/`write_goal_positions()` work in raw encoder ticks
(real DOF order); `read_velocities()` converts the device's native 0.229 rev/min
per LSB units (standard X-series control table) to rad/s. Sim<->real semantic
conversion (sign, offset, DOF reordering) is `joint_mapping.py`'s job, not this
module's -- this module only knows about the physical bus.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_POSITION = 116
ADDR_PROFILE_VELOCITY = 112
ADDR_PRESENT_VELOCITY = 128
ADDR_PRESENT_POSITION = 132

LEN_GOAL_POSITION = 4
LEN_PRESENT_VELOCITY = 4
LEN_PRESENT_POSITION = 4

TORQUE_ENABLE = 1
TORQUE_DISABLE = 0

# X-series present-velocity register: 0.229 rev/min per LSB.
_VELOCITY_UNIT_RPM = 0.229
_RPM_TO_RADPS = 2.0 * np.pi / 60.0


def _to_signed32(raw: int) -> int:
    """Dynamixel SDK bulk-read getData() returns an unsigned 32-bit int; present
    position/velocity registers are signed two's complement."""
    raw &= 0xFFFFFFFF
    return raw - 0x100000000 if raw & 0x80000000 else raw


class DynamixelBus(ABC):
    """Hardware interface for the 8-servo bus, real DOF order throughout."""

    @abstractmethod
    def torque_enable(self, on: bool) -> None: ...

    @abstractmethod
    def read_positions(self) -> np.ndarray:
        """Returns raw encoder ticks, real DOF order, int64."""

    @abstractmethod
    def read_velocities(self) -> np.ndarray:
        """Returns rad/s in the device's native sign convention, real DOF order."""

    @abstractmethod
    def write_goal_positions(self, ticks: np.ndarray) -> None:
        """Writes raw encoder ticks (0-4095), real DOF order."""

    def close(self) -> None:
        pass


class RealDynamixelBus(DynamixelBus):
    def __init__(
        self,
        motor_ids: list[int],
        port: str,
        baud_rate: int,
        protocol_version: float,
        profile_velocity: int = 0,
    ):
        try:
            from dynamixel_sdk import (
                COMM_SUCCESS,
                DXL_HIBYTE,
                DXL_HIWORD,
                DXL_LOBYTE,
                DXL_LOWORD,
                GroupBulkRead,
                GroupBulkWrite,
                PacketHandler,
                PortHandler,
            )
        except ImportError as exc:
            raise ImportError(
                "dynamixel_sdk is required for RealDynamixelBus "
                "(pip install dynamixel-sdk); use DryRunDynamixelBus for "
                "hardware-free testing"
            ) from exc

        self._dxl_lobyte = DXL_LOBYTE
        self._dxl_hibyte = DXL_HIBYTE
        self._dxl_loword = DXL_LOWORD
        self._dxl_hiword = DXL_HIWORD
        self._comm_success = COMM_SUCCESS

        self.motor_ids = list(motor_ids)
        self.port_handler = PortHandler(port)
        self.packet_handler = PacketHandler(protocol_version)
        self._bulk_write = GroupBulkWrite(self.port_handler, self.packet_handler)
        self._bulk_read_pos = GroupBulkRead(self.port_handler, self.packet_handler)
        self._bulk_read_vel = GroupBulkRead(self.port_handler, self.packet_handler)

        if not self.port_handler.openPort():
            raise RuntimeError(f"failed to open Dynamixel port {port}")
        if not self.port_handler.setBaudRate(baud_rate):
            self.port_handler.closePort()
            raise RuntimeError(f"failed to set baud rate {baud_rate}")

        for dxl_id in self.motor_ids:
            self.packet_handler.write4ByteTxRx(self.port_handler, dxl_id, ADDR_PROFILE_VELOCITY, profile_velocity)

        for dxl_id in self.motor_ids:
            if not self._bulk_read_pos.addParam(dxl_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION):
                raise RuntimeError(f"GroupBulkRead addParam failed for ID {dxl_id} (position)")
        for dxl_id in self.motor_ids:
            if not self._bulk_read_vel.addParam(dxl_id, ADDR_PRESENT_VELOCITY, LEN_PRESENT_VELOCITY):
                raise RuntimeError(f"GroupBulkRead addParam failed for ID {dxl_id} (velocity)")

    def torque_enable(self, on: bool) -> None:
        value = TORQUE_ENABLE if on else TORQUE_DISABLE
        for dxl_id in self.motor_ids:
            self.packet_handler.write1ByteTxRx(self.port_handler, dxl_id, ADDR_TORQUE_ENABLE, value)

    def read_positions(self) -> np.ndarray:
        self._bulk_read_pos.txRxPacket()
        ticks = np.empty(len(self.motor_ids), dtype=np.int64)
        for i, dxl_id in enumerate(self.motor_ids):
            raw = self._bulk_read_pos.getData(dxl_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)
            ticks[i] = _to_signed32(raw)
        return ticks

    def read_velocities(self) -> np.ndarray:
        self._bulk_read_vel.txRxPacket()
        radps = np.empty(len(self.motor_ids), dtype=np.float64)
        for i, dxl_id in enumerate(self.motor_ids):
            raw = self._bulk_read_vel.getData(dxl_id, ADDR_PRESENT_VELOCITY, LEN_PRESENT_VELOCITY)
            rpm = _to_signed32(raw) * _VELOCITY_UNIT_RPM
            radps[i] = rpm * _RPM_TO_RADPS
        return radps

    def write_goal_positions(self, ticks: np.ndarray) -> None:
        self._bulk_write.clearParam()
        for i, dxl_id in enumerate(self.motor_ids):
            value = int(np.clip(round(ticks[i]), 0, 4095))
            data = [
                self._dxl_lobyte(self._dxl_loword(value)),
                self._dxl_hibyte(self._dxl_loword(value)),
                self._dxl_lobyte(self._dxl_hiword(value)),
                self._dxl_hibyte(self._dxl_hiword(value)),
            ]
            if not self._bulk_write.addParam(dxl_id, ADDR_GOAL_POSITION, LEN_GOAL_POSITION, data):
                raise RuntimeError(f"GroupBulkWrite addParam failed for ID {dxl_id}")
        self._bulk_write.txPacket()
        self._bulk_write.clearParam()

    def close(self) -> None:
        self.port_handler.closePort()


class DryRunDynamixelBus(DynamixelBus):
    """No serial I/O -- in-memory simulated state for hardware-free testing."""

    def __init__(self, motor_ids: list[int], initial_ticks: np.ndarray | None = None):
        self.motor_ids = list(motor_ids)
        n = len(self.motor_ids)
        self._ticks = (
            np.full(n, 2048, dtype=np.int64) if initial_ticks is None else np.array(initial_ticks, dtype=np.int64)
        )
        self.torque_on = False

    def torque_enable(self, on: bool) -> None:
        self.torque_on = bool(on)

    def read_positions(self) -> np.ndarray:
        return self._ticks.copy()

    def read_velocities(self) -> np.ndarray:
        # No physical dynamics simulated -- goal positions are applied instantaneously,
        # so there's no meaningful "current velocity" to report.
        return np.zeros(len(self.motor_ids), dtype=np.float64)

    def write_goal_positions(self, ticks: np.ndarray) -> None:
        self._ticks = np.clip(np.round(ticks), 0, 4095).astype(np.int64)
