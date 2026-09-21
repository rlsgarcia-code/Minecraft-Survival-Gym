"""Canonical Minecraft controls and Gymnasium action encoding.

The canonical control object is intentionally richer than the first policy
space. Human keyboard/mouse demonstrations retain continuous camera and cursor
motion, while RL policies can use a stable MultiDiscrete projection.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from gymnasium import spaces


ACTION_FIELDS = (
    "strafe",
    "move",
    "jump",
    "sprint",
    "sneak",
    "attack",
    "use",
    "inventory",
    "drop",
    "yaw",
    "pitch",
    "hotbar",
    "cursor_x",
    "cursor_y",
    "gui_click",
)

CONTROL_FIELDS = (
    "move_x",
    "move_z",
    "jump",
    "sprint",
    "sneak",
    "attack",
    "use",
    "inventory",
    "drop",
    "yaw_delta",
    "pitch_delta",
    "hotbar_slot",
    "cursor_x_delta",
    "cursor_y_delta",
    "primary_click",
    "secondary_click",
)

# Every component uses zero as NOOP. This is useful for wrappers and policies.
ACTION_NVECS = np.asarray(
    [3, 3, 2, 2, 2, 2, 2, 2, 2, 5, 5, 10, 5, 5, 3], dtype=np.int64
)


def make_action_space() -> spaces.MultiDiscrete:
    return spaces.MultiDiscrete(ACTION_NVECS.copy(), dtype=np.int64)


@dataclass(frozen=True, slots=True)
class ControlState:
    """Device-independent control state applied for one environment step."""

    move_x: float = 0.0
    move_z: float = 0.0
    jump: bool = False
    sprint: bool = False
    sneak: bool = False
    attack: bool = False
    use: bool = False
    inventory: bool = False
    drop: bool = False
    yaw_delta: float = 0.0
    pitch_delta: float = 0.0
    hotbar_slot: int = -1
    cursor_x_delta: float = 0.0
    cursor_y_delta: float = 0.0
    primary_click: bool = False
    secondary_click: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ControlState":
        known = {field: value[field] for field in cls.__dataclass_fields__ if field in value}
        return cls(**known)


def control_to_array(control: ControlState) -> np.ndarray:
    """Losslessly encode canonical keyboard/mouse controls as 16 floats."""

    return np.asarray([getattr(control, field) for field in CONTROL_FIELDS], dtype=np.float32)


def noop_action() -> np.ndarray:
    return np.zeros(len(ACTION_NVECS), dtype=np.int64)


def decode_action(
    action: Sequence[int] | np.ndarray,
    *,
    camera_slow_degrees: float = 6.0,
    camera_fast_degrees: float = 18.0,
    cursor_slow: float = 0.25,
    cursor_fast: float = 0.75,
) -> ControlState:
    """Convert the policy MultiDiscrete action into canonical controls."""

    array = np.asarray(action, dtype=np.int64)
    space = make_action_space()
    if not space.contains(array):
        raise ValueError(
            f"Invalid Minecraft action {array.tolist()}; expected MultiDiscrete "
            f"with nvec={ACTION_NVECS.tolist()}"
        )

    strafe = (0.0, -1.0, 1.0)[array[0]]
    move = (0.0, 1.0, -1.0)[array[1]]
    camera_values = (
        0.0,
        -camera_slow_degrees,
        camera_slow_degrees,
        -camera_fast_degrees,
        camera_fast_degrees,
    )
    cursor_values = (0.0, -cursor_slow, cursor_slow, -cursor_fast, cursor_fast)
    gui_click = int(array[14])

    return ControlState(
        move_x=strafe,
        move_z=move,
        jump=bool(array[2]),
        sprint=bool(array[3]),
        sneak=bool(array[4]),
        attack=bool(array[5]),
        use=bool(array[6]),
        inventory=bool(array[7]),
        drop=bool(array[8]),
        yaw_delta=camera_values[array[9]],
        pitch_delta=camera_values[array[10]],
        hotbar_slot=int(array[11]) - 1,
        cursor_x_delta=cursor_values[array[12]],
        cursor_y_delta=cursor_values[array[13]],
        primary_click=gui_click == 1,
        secondary_click=gui_click == 2,
    )


def encode_human_control(
    control: ControlState,
    *,
    camera_slow_degrees: float = 6.0,
    cursor_slow: float = 0.25,
) -> np.ndarray:
    """Project continuous keyboard/mouse input onto the policy action space.

    Recorders should always keep ``ControlState`` as well as this projection.
    """

    def direction(value: float) -> int:
        if value < -0.25:
            return 1
        if value > 0.25:
            return 2
        return 0

    def five_way(value: float, slow: float) -> int:
        if value <= -slow * 2:
            return 3
        if value < -1e-6:
            return 1
        if value >= slow * 2:
            return 4
        if value > 1e-6:
            return 2
        return 0

    gui_click = 1 if control.primary_click else 2 if control.secondary_click else 0
    action = np.asarray(
        [
            direction(control.move_x),
            direction(-control.move_z) if control.move_z < 0 else (1 if control.move_z > 0.25 else 0),
            int(control.jump),
            int(control.sprint),
            int(control.sneak),
            int(control.attack),
            int(control.use),
            int(control.inventory),
            int(control.drop),
            five_way(control.yaw_delta, camera_slow_degrees),
            five_way(control.pitch_delta, camera_slow_degrees),
            control.hotbar_slot + 1 if 0 <= control.hotbar_slot <= 8 else 0,
            five_way(control.cursor_x_delta, cursor_slow),
            five_way(control.cursor_y_delta, cursor_slow),
            gui_click,
        ],
        dtype=np.int64,
    )
    if not make_action_space().contains(action):
        raise ValueError(f"Human control projected to invalid action: {action.tolist()}")
    return action
