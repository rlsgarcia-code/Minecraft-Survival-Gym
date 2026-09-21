"""Backend contract plus a deterministic in-process backend for testing."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from minecraft_gym.actions import ControlState


ControlMode = Literal["agent", "human", "dagger"]


@dataclass(slots=True)
class BackendObservation:
    rgb: np.ndarray
    vitals: np.ndarray
    pose: np.ndarray
    inventory_ids: np.ndarray
    inventory_counts: np.ndarray
    equipped_slot: int
    ui_mode: int
    tick: int
    world_seed: int

    def as_gym_observation(self) -> dict[str, np.ndarray | int]:
        return {
            "rgb": np.asarray(self.rgb, dtype=np.uint8),
            "vitals": np.asarray(self.vitals, dtype=np.float32),
            "pose": np.asarray(self.pose, dtype=np.float32),
            "inventory_ids": np.asarray(self.inventory_ids, dtype=np.int32),
            "inventory_counts": np.asarray(self.inventory_counts, dtype=np.int32),
            "equipped_slot": int(self.equipped_slot),
            "ui_mode": int(self.ui_mode),
        }


@dataclass(slots=True)
class BackendTransition:
    observation: BackendObservation
    terminated: bool = False
    truncated: bool = False
    events: list[dict[str, Any]] = field(default_factory=list)
    info: dict[str, Any] = field(default_factory=dict)


class MinecraftBackend(ABC):
    @abstractmethod
    def reset(
        self, *, seed: int, options: dict[str, Any], width: int, height: int
    ) -> BackendTransition:
        raise NotImplementedError

    @abstractmethod
    def step(
        self,
        control: ControlState,
        *,
        frame_skip: int,
        control_mode: ControlMode,
    ) -> BackendTransition:
        raise NotImplementedError

    @abstractmethod
    def render(self) -> np.ndarray | None:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError


class MockMinecraftBackend(MinecraftBackend):
    """Small deterministic world used to validate the Gymnasium contract.

    It is not intended to approximate Minecraft physics. The real bridge and
    this backend implement the same protocol, allowing unit tests without a
    running game.
    """

    def __init__(self, *, item_registry_size: int = 4096) -> None:
        self.item_registry_size = item_registry_size
        self._rng = np.random.default_rng()
        self._observation: BackendObservation | None = None
        self._closed = False

    def reset(
        self, *, seed: int, options: dict[str, Any], width: int, height: int
    ) -> BackendTransition:
        self._closed = False
        self._rng = np.random.default_rng(seed)
        vitals = np.asarray([1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        pose = np.zeros(10, dtype=np.float32)
        pose[1] = 0.5
        self._observation = BackendObservation(
            rgb=self._make_frame(width, height, tick=0),
            vitals=vitals,
            pose=pose,
            inventory_ids=np.zeros(36, dtype=np.int32),
            inventory_counts=np.zeros(36, dtype=np.int32),
            equipped_slot=0,
            ui_mode=0,
            tick=0,
            world_seed=seed,
        )
        return BackendTransition(
            observation=self._observation,
            info={"backend": "mock", "difficulty": options.get("difficulty", "normal")},
        )

    def step(
        self,
        control: ControlState,
        *,
        frame_skip: int,
        control_mode: ControlMode,
    ) -> BackendTransition:
        if self._closed or self._observation is None:
            raise RuntimeError("Backend must be reset before step")
        previous = self._observation
        pose = previous.pose.copy()
        pose[0] = np.clip(pose[0] + control.move_x * frame_skip / 100.0, -1.0, 1.0)
        pose[2] = np.clip(pose[2] + control.move_z * frame_skip / 100.0, -1.0, 1.0)
        pose[6] = np.clip(pose[6] + control.yaw_delta / 180.0, -1.0, 1.0)
        pose[7] = np.clip(pose[7] + control.pitch_delta / 90.0, -1.0, 1.0)
        tick = previous.tick + frame_skip
        vitals = previous.vitals.copy()
        vitals[1] = max(0.0, float(vitals[1]) - frame_skip / 24000.0)
        ui_mode = 1 - previous.ui_mode if control.inventory else previous.ui_mode
        height, width, _ = previous.rgb.shape
        self._observation = BackendObservation(
            rgb=self._make_frame(width, height, tick=tick),
            vitals=vitals,
            pose=pose,
            inventory_ids=previous.inventory_ids.copy(),
            inventory_counts=previous.inventory_counts.copy(),
            equipped_slot=control.hotbar_slot if control.hotbar_slot >= 0 else previous.equipped_slot,
            ui_mode=ui_mode,
            tick=tick,
            world_seed=previous.world_seed,
        )
        info: dict[str, Any] = {
            "backend": "mock",
            "control_mode": control_mode,
            "control_source": "agent",
            "policy_action": control.to_dict(),
            "human_action": None,
            "executed_action": control.to_dict(),
        }
        return BackendTransition(observation=self._observation, info=info)

    def render(self) -> np.ndarray | None:
        return None if self._observation is None else self._observation.rgb.copy()

    def close(self) -> None:
        self._closed = True

    def _make_frame(self, width: int, height: int, *, tick: int) -> np.ndarray:
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[..., 0] = np.uint8((tick // 4) % 256)
        frame[..., 1] = np.arange(width, dtype=np.uint8)[None, :]
        frame[..., 2] = np.arange(height, dtype=np.uint8)[:, None]
        return frame
