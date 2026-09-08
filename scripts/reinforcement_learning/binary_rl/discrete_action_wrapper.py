# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Discrete action wrapper for bit-vector continuous action envs.

Maps a single ``Discrete(2**n_bits)`` action to an ``n_bits``-dim float vector of
+1.0 / -1.0 (as consumed by ``BinaryJointPositionAction`` terms: positive = "open"
command, negative = "close" command).

Bit convention (documented for the hexapod binary env):
    integer action ``a`` -> action-vector index ``i`` receives ``(a >> i) & 1``;
    bit value 1 -> +1.0 (stance / foot down), bit value 0 -> -1.0 (lift / foot up).
    The LSB is action index 0.  With the hardware-aligned term order this means
    LSB = hardware bit 1 = FrontRight, ..., bit 6 (0x20) = BackLeft.
    Example: a = 0b000001 = 1 -> only FrontRight in stance; a = 63 -> all-stance.

Why ``unwrapped`` returns the wrapper itself: skrl's ``IsaacLabWrapper`` reads
``env.unwrapped.single_action_space`` (and calls ``env.step``/``env.reset`` on the
object handed to ``wrap_env``).  With the standard ``gym.Wrapper.unwrapped`` chain the
underlying ``ManagerBasedRLEnv`` would be reached and skrl would see the original
``Box(6)`` space instead of ``Discrete(64)``.  Overriding ``unwrapped`` keeps every
space/attribute lookup on this wrapper (unknown attributes still fall through to the
wrapped env via ``gym.Wrapper.__getattr__``).  The raw env stays reachable via
``.base_env`` for anything that needs sim internals.
"""

from __future__ import annotations

import gymnasium as gym
import torch


class DiscreteBitsActionWrapper(gym.Wrapper):
    """Expose a Box(n_bits) +-1 action env as a Discrete(2**n_bits) action env."""

    def __init__(self, env: gym.Env, n_bits: int = 6):
        super().__init__(env)
        self.n_bits = n_bits
        self.n_actions = 2**n_bits
        base = env.unwrapped
        self.base_env = base
        self._device = base.device
        self._num_envs = base.num_envs
        # bit weights [1, 2, 4, ...] for decoding on-device
        self._bit_shifts = torch.arange(n_bits, device=self._device, dtype=torch.long)
        # spaces: single = Discrete(2**n), batched = one Discrete per env
        self.single_action_space = gym.spaces.Discrete(self.n_actions)
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self._num_envs)
        # observation spaces pass through unchanged
        self.single_observation_space = base.single_observation_space
        self.observation_space = base.observation_space

    # -- make skrl (which reads spaces off env.unwrapped) see THIS wrapper's spaces --
    @property
    def unwrapped(self):
        return self

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def device(self):
        return self._device

    def decode(self, actions: torch.Tensor) -> torch.Tensor:
        """[N] or [N,1] integer actions in [0, 2**n_bits) -> [N, n_bits] float +-1."""
        a = actions.reshape(-1).long().to(self._device)
        bits = (a.unsqueeze(1) >> self._bit_shifts.unsqueeze(0)) & 1  # [N, n_bits]
        return bits.to(torch.float32) * 2.0 - 1.0

    def step(self, actions):
        return self.env.step(self.decode(torch.as_tensor(actions)))

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
