"""Gymnasium environment for Minecraft survival."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from minecraft_gym.actions import decode_action, make_action_space
from minecraft_gym.backend import (
    BackendObservation,
    ControlMode,
    MinecraftBackend,
    MockMinecraftBackend,
)
from minecraft_gym.rewards import RewardFunction, SparseSurvivalReward
from minecraft_gym.transport import DEFAULT_HOST, DEFAULT_PORT, SocketMinecraftBackend


class MinecraftSurvivalEnv(gym.Env[dict[str, Any], np.ndarray]):
    """A single-player Survival environment with a pluggable game backend."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 5}

    def __init__(
        self,
        *,
        backend: MinecraftBackend | str = "socket",
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        bridge_timeout: float = 30.0,
        width: int = 128,
        height: int = 128,
        frame_skip: int = 4,
        max_episode_steps: int = 9000,
        item_registry_size: int = 4096,
        control_mode: ControlMode = "agent",
        render_mode: str | None = None,
        reward_function: RewardFunction | None = None,
    ) -> None:
        super().__init__()
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive")
        if frame_skip <= 0:
            raise ValueError("frame_skip must be positive")
        if max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive")
        if control_mode not in ("agent", "human", "dagger"):
            raise ValueError(f"Unknown control mode: {control_mode}")
        if render_mode not in (None, "rgb_array"):
            raise ValueError(f"Unsupported render mode: {render_mode}")

        if backend == "mock":
            self.backend: MinecraftBackend = MockMinecraftBackend(
                item_registry_size=item_registry_size
            )
        elif backend == "socket":
            self.backend = SocketMinecraftBackend(host, port, timeout=bridge_timeout)
        elif isinstance(backend, MinecraftBackend):
            self.backend = backend
        else:
            raise ValueError(
                "backend must be 'socket', 'mock', or a MinecraftBackend instance"
            )

        self.width = width
        self.height = height
        self.frame_skip = frame_skip
        self.max_episode_steps = max_episode_steps
        self.item_registry_size = item_registry_size
        self.control_mode = control_mode
        self.render_mode = render_mode
        self.reward_function = reward_function or SparseSurvivalReward()

        self.action_space = make_action_space()
        self.observation_space = spaces.Dict(
            {
                "rgb": spaces.Box(0, 255, shape=(height, width, 3), dtype=np.uint8),
                "vitals": spaces.Box(0.0, 1.0, shape=(8,), dtype=np.float32),
                "pose": spaces.Box(-1.0, 1.0, shape=(10,), dtype=np.float32),
                "inventory_ids": spaces.Box(
                    0, item_registry_size - 1, shape=(36,), dtype=np.int32
                ),
                "inventory_counts": spaces.Box(0, 64, shape=(36,), dtype=np.int32),
                "equipped_slot": spaces.Discrete(9),
                "ui_mode": spaces.Discrete(8),
            }
        )
        self._last_backend_observation: BackendObservation | None = None
        self._episode_steps = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        super().reset(seed=seed)
        actual_seed = int(seed if seed is not None else self.np_random.integers(0, 2**31 - 1))
        actual_options = dict(options or {})
        transition = self.backend.reset(
            seed=actual_seed,
            options=actual_options,
            width=self.width,
            height=self.height,
        )
        self._episode_steps = 0
        self._last_backend_observation = transition.observation
        observation = transition.observation.as_gym_observation()
        self._assert_observation(observation)
        info = self._make_info(transition.info, transition.events)
        return observation, info

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        if self._last_backend_observation is None:
            raise RuntimeError("reset() must be called before step()")
        if not self.action_space.contains(action):
            raise ValueError(f"Action is outside action_space: {action!r}")

        control = decode_action(action)
        previous = self._last_backend_observation
        transition = self.backend.step(
            control,
            frame_skip=self.frame_skip,
            control_mode=self.control_mode,
        )
        self._episode_steps += 1
        truncated = bool(
            transition.truncated or self._episode_steps >= self.max_episode_steps
        )
        terminated = bool(transition.terminated)
        observation = transition.observation.as_gym_observation()
        self._assert_observation(observation)
        reward_result = self.reward_function(previous, transition)
        self._last_backend_observation = transition.observation
        info = self._make_info(transition.info, transition.events)
        info["reward_terms"] = reward_result.terms
        if truncated and not transition.truncated:
            info["truncation_reason"] = "max_episode_steps"
        return observation, reward_result.total, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        if self.render_mode is None:
            return None
        return self.backend.render()

    def close(self) -> None:
        self.backend.close()
        self._last_backend_observation = None

    def set_control_mode(self, mode: ControlMode) -> None:
        if mode not in ("agent", "human", "dagger"):
            raise ValueError(f"Unknown control mode: {mode}")
        self.control_mode = mode

    def _assert_observation(self, observation: dict[str, Any]) -> None:
        if not self.observation_space.contains(observation):
            raise ValueError("Backend returned an observation outside observation_space")

    def _make_info(
        self, backend_info: dict[str, Any], events: list[dict[str, Any]]
    ) -> dict[str, Any]:
        observation = self._last_backend_observation
        info = dict(backend_info)
        info.update(
            {
                "tick": observation.tick if observation is not None else None,
                "world_seed": observation.world_seed if observation is not None else None,
                "events": list(events),
                "control_mode": self.control_mode,
            }
        )
        return info
