"""Public policy contracts for the Navigation Academy.

Modern policies consume two jointly encoded lattices: a metric crop of the
persistent world map and the current gnomonic view.  Evaluator-only
values (absolute coordinates, shortest-path cost and world deltas) never enter
these encoders.  The legacy global grid remains available only so previously
saved policies can still be inspected and evaluated.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import math
from typing import Any, Iterable, Mapping

import numpy as np

from gymnasium import spaces

from minecraft_gym.actions import ControlState, noop_action


GRID_CHANNELS = (
    "solid_height",
    "void",
    "breakable",
    "placed_block",
    "traversable",
    "agent",
    "goal",
    "observed",
)

VIEW_LATTICE_CHANNELS = (
    "evidence",
    "elevation_or_inverse_depth",
    "variation",
    "target_probability",
    "material_id",
)
MAP_LATTICE_CHANNELS = (
    *VIEW_LATTICE_CHANNELS,
    "solid_stack_height",
    "head_clearance",
)
# Historical name retained for consumers that only need the view contract.
LATTICE_CHANNELS = VIEW_LATTICE_CHANNELS
PYRAMID_SCALES = (1, 4, 16)
PYRAMID_SIZE = 33
GNOMONIC_SHAPE = (18, 32)
POLICY_LATTICE_SIZE = 17

# The trainable map is a single metric crop.  Every pixel is exactly one
# Minecraft X/Z block; evidence outside this crop remains in the sparse world
# map owned by SpatialLatticeMap and re-enters the crop when the actor returns.
WORLD_MAP_WINDOW_SIZE = 65
WORLD_MAP_CHANNELS = (
    "evidence",
    "visited",
    "relative_elevation",
    "material_id",
    "solid_stack_height",
    "head_clearance",
    "traversability",
    "target_or_direction",
)


class AcademyAction(IntEnum):
    WAIT = 0
    FORWARD = 1
    BACKWARD = 2
    STRAFE_LEFT = 3
    STRAFE_RIGHT = 4
    SPRINT_FORWARD = 5
    JUMP_FORWARD = 6
    SPRINT_JUMP = 7
    TURN_LEFT_90 = 8
    TURN_RIGHT_90 = 9
    BREAK = 10
    PLACE = 11

    @property
    def label(self) -> str:
        return {
            self.WAIT: "esperar",
            self.FORWARD: "avancar",
            self.BACKWARD: "recuar",
            self.STRAFE_LEFT: "lateral_esquerda",
            self.STRAFE_RIGHT: "lateral_direita",
            self.SPRINT_FORWARD: "correr",
            self.JUMP_FORWARD: "pular_avancando",
            self.SPRINT_JUMP: "correr_e_pular",
            self.TURN_LEFT_90: "girar_esquerda_90",
            self.TURN_RIGHT_90: "girar_direita_90",
            self.BREAK: "quebrar",
            self.PLACE: "colocar",
        }[self]


ACTION_COUNT = len(AcademyAction)

# Direct Minecraft controls used by the factorized navigation policy.  Each
# branch is selected independently, so the policy can press several keys in
# the same decision without materializing their Cartesian product.
CONTROL_BRANCHES = (
    ("strafe", 3),
    ("move", 3),
    ("longitude", 5),
    ("latitude", 5),
    ("jump", 2),
    ("sneak", 2),
    ("sprint", 2),
    ("attack", 2),
    ("use", 2),
)
CONTROL_BRANCH_SIZES = tuple(size for _, size in CONTROL_BRANCHES)
CONTROL_LOGIT_COUNT = sum(CONTROL_BRANCH_SIZES)


# A compact catalogue of complete, physically valid keyboard/mouse chords for
# target navigation.  Unlike CONTROL_BRANCHES, each entry is one joint action:
# the Q estimator assigns credit to the whole chord that was executed.
JOINT_CONTROL_LABELS = (
    "w",
    "w_ctrl",
    "s",
    "a",
    "d",
    "w_a",
    "w_d",
    "w_ctrl_a",
    "w_ctrl_d",
    "space",
    "w_space",
    "w_ctrl_space",
    "w_a_space",
    "w_d_space",
    "w_ctrl_a_space",
    "w_ctrl_d_space",
    "shift",
    "w_shift",
    "mouse_left",
    "yaw_left_slow",
    "yaw_right_slow",
    "yaw_left_fast",
    "yaw_right_fast",
    "pitch_up",
    "pitch_down",
)
JOINT_CONTROL_ACTION_COUNT = len(JOINT_CONTROL_LABELS)


class HybridBodyMode(IntEnum):
    """Keyboard/button choice paired with a continuous camera displacement."""

    LOOK = 0
    FORWARD = 1
    SPRINT_FORWARD = 2
    JUMP_FORWARD = 3
    SPRINT_JUMP = 4
    SNEAK_FORWARD = 5
    ATTACK_FORWARD = 6

    @property
    def label(self) -> str:
        return {
            self.LOOK: "olhar",
            self.FORWARD: "avancar",
            self.SPRINT_FORWARD: "correr",
            self.JUMP_FORWARD: "pular_avancando",
            self.SPRINT_JUMP: "correr_e_pular",
            self.SNEAK_FORWARD: "agachar_avancando",
            self.ATTACK_FORWARD: "avancar_e_quebrar",
        }[self]


HYBRID_BODY_MODE_COUNT = len(HybridBodyMode)
HYBRID_LONGITUDE_LIMIT_DEGREES = 180.0
HYBRID_LATITUDE_LIMIT_DEGREES = 90.0


@dataclass(frozen=True, slots=True)
class HybridNavigationAction:
    """Independent mouse and keyboard factors for one policy decision.

    Camera displacement does not imply forward motion.  ``strafe`` represents
    A/D and ``move`` represents S/W in ``{-1, 0, 1}``; buttons are independent
    factors so looking, backing, strafing and jumping can be combined without
    enumerating their Cartesian product.
    """

    delta_longitude: float
    delta_latitude: float
    strafe: int = 0
    move: int = 0
    jump: bool = False
    sprint: bool = False
    sneak: bool = False
    attack: bool = False
    use: bool = False

    def __post_init__(self) -> None:
        longitude = float(self.delta_longitude)
        latitude = float(self.delta_latitude)
        if not math.isfinite(longitude) or not math.isfinite(latitude):
            raise ValueError("hybrid camera displacement must be finite")
        if not -HYBRID_LONGITUDE_LIMIT_DEGREES <= longitude <= HYBRID_LONGITUDE_LIMIT_DEGREES:
            raise ValueError("hybrid longitude displacement is outside [-180, 180]")
        if not -HYBRID_LATITUDE_LIMIT_DEGREES <= latitude <= HYBRID_LATITUDE_LIMIT_DEGREES:
            raise ValueError("hybrid latitude displacement is outside [-90, 90]")
        object.__setattr__(self, "delta_longitude", longitude)
        object.__setattr__(self, "delta_latitude", latitude)
        if int(self.strafe) not in (-1, 0, 1) or int(self.move) not in (-1, 0, 1):
            raise ValueError("hybrid movement axes must be in {-1, 0, 1}")
        object.__setattr__(self, "strafe", int(self.strafe))
        object.__setattr__(self, "move", int(self.move))
        for field in ("jump", "sprint", "sneak", "attack", "use"):
            object.__setattr__(self, field, bool(getattr(self, field)))

    def vector(self) -> np.ndarray:
        """Stable dataset projection for independent camera and controls."""

        return np.asarray(
            [
                self.delta_longitude,
                self.delta_latitude,
                self.strafe,
                self.move,
                int(self.jump),
                int(self.sprint),
                int(self.sneak),
                int(self.attack),
                int(self.use),
            ],
            dtype=np.float32,
        )

    def dataset_value(self) -> dict[str, float | int]:
        """Lossless policy-level label stored beside executed device controls."""

        return {
            "delta_longitude": self.delta_longitude,
            "delta_latitude": self.delta_latitude,
            "strafe": self.strafe,
            "move": self.move,
            "jump": self.jump,
            "sprint": self.sprint,
            "sneak": self.sneak,
            "attack": self.attack,
            "use": self.use,
        }


def make_hybrid_navigation_action_space() -> spaces.Dict:
    """Gymnasium space for ``minecraft-control-camera-keyboard/v3``."""

    return spaces.Dict(
        {
            "camera_delta": spaces.Box(
                low=np.asarray(
                    [-HYBRID_LONGITUDE_LIMIT_DEGREES, -HYBRID_LATITUDE_LIMIT_DEGREES],
                    dtype=np.float32,
                ),
                high=np.asarray(
                    [HYBRID_LONGITUDE_LIMIT_DEGREES, HYBRID_LATITUDE_LIMIT_DEGREES],
                    dtype=np.float32,
                ),
                dtype=np.float32,
            ),
            "movement_axes": spaces.MultiDiscrete([3, 3]),
            "buttons": spaces.MultiBinary(5),
        }
    )


def hybrid_control_action(action: HybridNavigationAction) -> ControlState:
    """Translate a hybrid policy action without quantizing camera motion."""

    selected = HybridNavigationAction(**action.dataset_value())
    return ControlState(
        move_x=float(selected.strafe),
        move_z=float(selected.move),
        jump=selected.jump,
        sprint=selected.sprint,
        sneak=selected.sneak,
        attack=selected.attack,
        use=selected.use,
        yaw_delta=selected.delta_longitude,
        pitch_delta=selected.delta_latitude,
    )


def _joint_control_catalogue() -> tuple[np.ndarray, ...]:
    def chord(
        *,
        strafe: int = 0,
        move: int = 0,
        jump: int = 0,
        sprint: int = 0,
        sneak: int = 0,
        attack: int = 0,
        yaw: int = 0,
        pitch: int = 0,
    ) -> np.ndarray:
        action = noop_action()
        action[0] = strafe
        action[1] = move
        action[2] = jump
        action[3] = sprint
        action[4] = sneak
        action[5] = attack
        action[9] = yaw
        action[10] = pitch
        return action

    return (
        chord(move=1),
        chord(move=1, sprint=1),
        chord(move=2),
        chord(strafe=1),
        chord(strafe=2),
        chord(move=1, strafe=1),
        chord(move=1, strafe=2),
        chord(move=1, sprint=1, strafe=1),
        chord(move=1, sprint=1, strafe=2),
        chord(jump=1),
        chord(move=1, jump=1),
        chord(move=1, sprint=1, jump=1),
        chord(move=1, strafe=1, jump=1),
        chord(move=1, strafe=2, jump=1),
        chord(move=1, sprint=1, strafe=1, jump=1),
        chord(move=1, sprint=1, strafe=2, jump=1),
        chord(sneak=1),
        chord(move=1, sneak=1),
        chord(attack=1),
        chord(yaw=1),
        chord(yaw=2),
        chord(yaw=3),
        chord(yaw=4),
        chord(pitch=1),
        chord(pitch=2),
    )


JOINT_CONTROL_ACTIONS = _joint_control_catalogue()


def joint_control_action(action: int) -> np.ndarray:
    """Return a copy of one complete 25-way navigation control chord."""

    index = int(action)
    if not 0 <= index < JOINT_CONTROL_ACTION_COUNT:
        raise ValueError("joint navigation action is outside its discrete range")
    return JOINT_CONTROL_ACTIONS[index].copy()


def factorized_control_action(branches: np.ndarray | Iterable[int]) -> np.ndarray:
    """Compose independent control heads into the canonical keyboard action."""

    values = np.asarray(tuple(branches), dtype=np.int64)
    if values.shape != (len(CONTROL_BRANCHES),):
        raise ValueError("factorized control action has an incompatible shape")
    if any(value < 0 or value >= size for value, size in zip(values, CONTROL_BRANCH_SIZES)):
        raise ValueError("factorized control branch is outside its discrete range")
    action = noop_action()
    action[0] = values[0]
    action[1] = values[1]
    action[9] = values[2]
    action[10] = values[3]
    action[2] = values[4]
    action[4] = values[5]
    action[3] = values[6]
    action[5] = values[7]
    action[6] = values[8]
    return action


def abstract_action(action: int | AcademyAction) -> np.ndarray:
    """Map one of the 12 body actions to canonical Minecraft controls.

    The environment decodes the fast yaw bins as 90 degrees for this policy
    contract; the canonical action format itself is unchanged.
    """

    selected = AcademyAction(action)
    value = noop_action()
    if selected is AcademyAction.FORWARD:
        value[1] = 1
    elif selected is AcademyAction.BACKWARD:
        value[1] = 2
    elif selected is AcademyAction.STRAFE_LEFT:
        value[0] = 1
    elif selected is AcademyAction.STRAFE_RIGHT:
        value[0] = 2
    elif selected is AcademyAction.SPRINT_FORWARD:
        value[1], value[3] = 1, 1
    elif selected is AcademyAction.JUMP_FORWARD:
        value[1], value[2] = 1, 1
    elif selected is AcademyAction.SPRINT_JUMP:
        value[1], value[2], value[3] = 1, 1, 1
    elif selected is AcademyAction.TURN_LEFT_90:
        value[9] = 3
    elif selected is AcademyAction.TURN_RIGHT_90:
        value[9] = 4
    elif selected is AcademyAction.BREAK:
        value[5] = 1
    elif selected is AcademyAction.PLACE:
        value[6] = 1
    return value


@dataclass(frozen=True, slots=True)
class PolicyState:
    grid: np.ndarray
    joint_lattice: np.ndarray
    auxiliary: np.ndarray
    action_mask: np.ndarray

    def vector(self) -> np.ndarray:
        return np.concatenate(
            (self.joint_lattice.reshape(-1), self.auxiliary, [1.0])
        ).astype(
            np.float32, copy=False
        )

    def legacy_vector(self) -> np.ndarray:
        return np.concatenate((self.grid.reshape(-1), self.auxiliary, [1.0])).astype(
            np.float32, copy=False
        )


def _nearest_resize(values: np.ndarray, size: int) -> np.ndarray:
    """Dependency-free nearest resize for policy encoding, not visualization."""

    if values.ndim != 3:
        raise ValueError("lattice must have channel, height and width dimensions")
    rows = np.rint(np.linspace(0, values.shape[1] - 1, size)).astype(np.intp)
    columns = np.rint(np.linspace(0, values.shape[2] - 1, size)).astype(np.intp)
    return values[:, rows][:, :, columns]


def encode_policy_state(
    observation: dict[str, Any], *, include_elapsed: bool = False
) -> PolicyState:
    """Encode the metric world-map crop and current gnomonic observation.

    ``grid_pyramid`` is still accepted by environments that have not migrated,
    but every new academy/runtime observation supplies ``map_window``.  The
    modern path never averages blocks into coarse bins.
    """

    grid = np.asarray(observation["global_grid"], dtype=np.float32)
    map_window_value = observation.get("map_window")
    map_window = (
        np.asarray(map_window_value, dtype=np.float32)
        if map_window_value is not None
        else None
    )
    pyramid = np.asarray(observation.get("grid_pyramid", ()), dtype=np.float32)
    gnomonic = np.asarray(observation["gnomonic_lattice"], dtype=np.float32)
    orientation = np.asarray(observation["orientation"], dtype=np.float32)
    inventory = np.asarray(observation["inventory"], dtype=np.float32)
    inventory = np.asarray(
        [np.clip(inventory[0] / 8.0, 0.0, 1.0), np.clip(inventory[1], 0.0, 1.0)],
        dtype=np.float32,
    )
    auxiliary_parts = [orientation, inventory]
    if include_elapsed:
        auxiliary_parts.append(
            np.asarray(
                [np.clip(float(observation.get("elapsed_fraction", 0.0)), 0.0, 1.0)],
                dtype=np.float32,
            )
        )
    auxiliary = np.concatenate(auxiliary_parts).astype(np.float32)
    mask = np.asarray(observation["action_mask"], dtype=np.int8)
    if grid.ndim != 3 or grid.shape[0] != len(GRID_CHANNELS):
        raise ValueError("global_grid is incompatible with academy-grid/v2")
    expected_window = (
        len(WORLD_MAP_CHANNELS),
        WORLD_MAP_WINDOW_SIZE,
        WORLD_MAP_WINDOW_SIZE,
    )
    if map_window is not None and map_window.shape != expected_window:
        raise ValueError("map_window is incompatible with world-map-window/v1")
    expected_pyramid = (
        len(PYRAMID_SCALES), len(MAP_LATTICE_CHANNELS), PYRAMID_SIZE, PYRAMID_SIZE
    )
    if map_window is None and pyramid.shape != expected_pyramid:
        raise ValueError("observation has neither world-map-window/v1 nor dual-lattice/v1")
    if gnomonic.shape != (len(VIEW_LATTICE_CHANNELS), *GNOMONIC_SHAPE):
        raise ValueError("gnomonic_lattice is incompatible with topology-view/v1")
    if orientation.shape != (4,) or mask.shape != (ACTION_COUNT,):
        raise ValueError("academy orientation or action mask is incompatible")
    if map_window is not None:
        spatial = map_window
        lattice_size = WORLD_MAP_WINDOW_SIZE
    else:
        spatial = pyramid.reshape(
            len(PYRAMID_SCALES) * len(MAP_LATTICE_CHANNELS),
            PYRAMID_SIZE,
            PYRAMID_SIZE,
        )
        lattice_size = POLICY_LATTICE_SIZE
        spatial = _nearest_resize(spatial, lattice_size)
    visual = _nearest_resize(gnomonic, lattice_size)
    joint_lattice = np.concatenate((spatial, visual), axis=0).astype(
        np.float32, copy=False
    )
    return PolicyState(
        grid=grid,
        joint_lattice=joint_lattice,
        auxiliary=auxiliary,
        action_mask=mask,
    )


def _material_value(block_counts: Mapping[str, Any]) -> float:
    names = " ".join(str(name).lower() for name in block_counts)
    if "water" in names or "lava" in names:
        return 0.1
    if "wood" in names or "log" in names or "leaves" in names:
        return 0.75
    if "stone" in names or "quartz" in names or "ore" in names:
        return 1.0
    return 0.4 if names else 0.25


def project_world_map_window(
    cells: Iterable[Mapping[str, Any]],
    *,
    current_cell: tuple[int, int],
    target_cell: tuple[int, int],
    current_y: float | None = None,
    size: int = WORLD_MAP_WINDOW_SIZE,
) -> np.ndarray:
    """Crop persistent world evidence into a north-up, one-block lattice.

    The source iterable may contain the complete accumulated world map.  Only
    the crop is exposed to the policy; records outside it are not mutated or
    discarded.  If the target lies outside the crop, its bearing is clipped to
    the border so the policy retains a directional objective.
    """

    if size <= 0 or size % 2 == 0:
        raise ValueError("world map window size must be a positive odd number")
    current_x, current_z = current_cell
    records = list(cells)
    current_surface = (
        float(current_y) - 1.0
        if current_y is not None
        else next(
            (
                float(cell["surface_y_min"])
                for cell in records
                if int(cell["cell_x"]) == current_x
                and int(cell["cell_z"]) == current_z
                and cell.get("surface_y_min") is not None
            ),
            64.0,
        )
    )
    result = np.zeros((len(WORLD_MAP_CHANNELS), size, size), dtype=np.float32)
    half = size // 2
    for cell in records:
        dx = int(cell["cell_x"]) - current_x
        dz = int(cell["cell_z"]) - current_z
        row, column = half - dz, half + dx
        if not (0 <= row < size and 0 <= column < size):
            continue
        free = int(cell.get("free_count") or 0)
        hits = int(cell.get("hit_count") or 0)
        visited = int(cell.get("visited_count") or 0)
        surface = cell.get("surface_y_max")
        stack = float(max(0.0, float(cell.get("solid_stack_height") or 0.0)))
        clearance = float(
            max(
                0.0,
                float(
                    cell.get("head_clearance")
                    if cell.get("head_clearance") is not None
                    else 4.0
                ),
            )
        )
        observations = free + hits + visited
        observed = bool(observations or cell.get("block_counts"))
        result[0, row, column] = (
            1.0 - math.exp(-observations / 8.0) if observations else float(observed)
        )
        result[1, row, column] = float(visited > 0)
        if surface is not None:
            result[2, row, column] = np.clip(
                (float(surface) - current_surface + 4.0) / 8.0, 0.0, 1.0
            )
        result[3, row, column] = _material_value(cell.get("block_counts") or {})
        result[4, row, column] = np.clip(stack / 4.0, 0.0, 1.0)
        result[5, row, column] = np.clip(clearance / 4.0, 0.0, 1.0)
        if visited > 0:
            traversability = 1.0
        elif clearance < 2.0 or stack >= 2.0:
            traversability = 0.0
        elif free > 0 and hits == 0:
            traversability = 1.0
        elif observed:
            traversability = 0.5
        else:
            traversability = 0.0
        result[6, row, column] = traversability

    target_dx = target_cell[0] - current_x
    target_dz = target_cell[1] - current_z
    if target_dx == 0 and target_dz == 0:
        target_row = target_column = half
    else:
        magnitude = max(abs(target_dx), abs(target_dz))
        scale = min(1.0, half / float(magnitude))
        target_column = half + int(round(target_dx * scale))
        target_row = half - int(round(target_dz * scale))
    result[7, target_row, target_column] = 1.0
    return result


class WorldMapWindowCache:
    """Incremental crop over a persistent sparse world map.

    Repeated raycasts usually increase counters without changing topology.
    Once confidence is high, those cells remain in memory but are not
    reprojected.  A one-block actor displacement rolls the previous crop and
    fills only the newly exposed strip plus cells whose topology changed.
    """

    def __init__(
        self,
        *,
        size: int = WORLD_MAP_WINDOW_SIZE,
        confidence_threshold: float = 0.95,
    ) -> None:
        if size <= 0 or size % 2 == 0:
            raise ValueError("world map cache size must be a positive odd number")
        self.size = int(size)
        self.confidence_threshold = float(confidence_threshold)
        self._memory: dict[tuple[int, int], tuple[float, ...]] = {}
        self._dirty: set[tuple[int, int]] = set()
        self._window: np.ndarray | None = None
        self._center: tuple[int, int] | None = None
        self._surface: float | None = None
        self.last_metrics = {"updated": 0, "reused": 0, "window_cells_written": 0}

    @property
    def world_cell_count(self) -> int:
        return len(self._memory)

    @staticmethod
    def _record(cell: Mapping[str, Any]) -> tuple[float, ...]:
        free = int(cell.get("free_count") or 0)
        hits = int(cell.get("hit_count") or 0)
        visited = int(cell.get("visited_count") or 0)
        observations = free + hits + visited
        confidence = 1.0 - math.exp(-observations / 8.0)
        surface_value = cell.get("surface_y_max")
        surface = float(surface_value) if surface_value is not None else math.nan
        material = _material_value(cell.get("block_counts") or {})
        stack = float(max(0.0, float(cell.get("solid_stack_height") or 0.0)))
        clearance = float(
            max(
                0.0,
                float(
                    cell.get("head_clearance")
                    if cell.get("head_clearance") is not None
                    else 4.0
                ),
            )
        )
        if visited > 0:
            traversability = 1.0
        elif clearance < 2.0 or stack >= 2.0:
            traversability = 0.0
        elif free > 0 and hits == 0:
            traversability = 1.0
        elif observations:
            traversability = 0.5
        else:
            traversability = 0.0
        return (
            confidence,
            float(visited > 0),
            surface,
            material,
            np.clip(stack / 4.0, 0.0, 1.0),
            np.clip(clearance / 4.0, 0.0, 1.0),
            traversability,
        )

    @staticmethod
    def _topology_signature(record: tuple[float, ...]) -> tuple[float, ...]:
        surface = record[2]
        return (
            round(surface, 2) if math.isfinite(surface) else -1.0e30,
            *record[1:2],
            *record[3:7],
        )

    def update(self, cells: Iterable[Mapping[str, Any]]) -> None:
        updated = reused = 0
        for cell in cells:
            key = (int(cell["cell_x"]), int(cell["cell_z"]))
            candidate = self._record(cell)
            previous = self._memory.get(key)
            self._memory[key] = candidate
            if previous is None:
                self._dirty.add(key)
                updated += 1
                continue
            topology_changed = self._topology_signature(previous) != self._topology_signature(candidate)
            confidence_changed = (
                previous[0] < self.confidence_threshold
                and abs(candidate[0] - previous[0]) >= 0.01
            )
            if topology_changed or confidence_changed:
                self._dirty.add(key)
                updated += 1
            else:
                reused += 1
        self.last_metrics = {
            "updated": updated,
            "reused": reused,
            "window_cells_written": 0,
        }

    def _write_cell(
        self,
        window: np.ndarray,
        key: tuple[int, int],
        *,
        center: tuple[int, int],
        current_surface: float,
    ) -> bool:
        record = self._memory.get(key)
        if record is None:
            return False
        half = self.size // 2
        column = half + key[0] - center[0]
        row = half - (key[1] - center[1])
        if not (0 <= row < self.size and 0 <= column < self.size):
            return False
        window[:, row, column] = 0.0
        window[0, row, column] = record[0]
        window[1, row, column] = record[1]
        if math.isfinite(record[2]):
            window[2, row, column] = np.clip(
                (record[2] - current_surface + 4.0) / 8.0, 0.0, 1.0
            )
        window[3:7, row, column] = record[3:7]
        return True

    def render(
        self,
        *,
        current_cell: tuple[int, int],
        target_cell: tuple[int, int],
        current_y: float | None = None,
    ) -> np.ndarray:
        current_surface = float(current_y) - 1.0 if current_y is not None else 64.0
        half = self.size // 2
        written = 0
        previous_center = self._center
        small_shift = (
            self._window is not None
            and previous_center is not None
            and abs(current_cell[0] - previous_center[0]) < self.size
            and abs(current_cell[1] - previous_center[1]) < self.size
        )
        if not small_shift:
            window = np.zeros(
                (len(WORLD_MAP_CHANNELS), self.size, self.size), dtype=np.float32
            )
            for x in range(current_cell[0] - half, current_cell[0] + half + 1):
                for z in range(current_cell[1] - half, current_cell[1] + half + 1):
                    written += self._write_cell(
                        window,
                        (x, z),
                        center=current_cell,
                        current_surface=current_surface,
                    )
            # A full rebuild already consumed the latest value of every
            # in-range cell. Out-of-range dirty cells will be read directly
            # from world memory when they later enter the crop.
            self._dirty.clear()
        else:
            assert self._window is not None and previous_center is not None
            dx = current_cell[0] - previous_center[0]
            dz = current_cell[1] - previous_center[1]
            window = np.roll(self._window, shift=(dz, -dx), axis=(1, 2)).copy()
            window[7] = 0.0
            if dz > 0:
                window[:, :dz, :] = 0.0
            elif dz < 0:
                window[:, dz:, :] = 0.0
            if dx > 0:
                window[:, :, -dx:] = 0.0
            elif dx < 0:
                window[:, :, : -dx] = 0.0

            if self._surface is not None and current_surface != self._surface:
                mask = window[0] > 0.0
                window[2, mask] = np.clip(
                    window[2, mask] + (self._surface - current_surface) / 8.0,
                    0.0,
                    1.0,
                )

            # Only the newly exposed rows/columns are looked up in world
            # memory. The stable overlapping core is already in ``window``.
            exposed: set[tuple[int, int]] = set()
            if dz:
                z_values = (
                    range(current_cell[1] + half - abs(dz) + 1, current_cell[1] + half + 1)
                    if dz > 0
                    else range(current_cell[1] - half, current_cell[1] - half + abs(dz))
                )
                exposed.update(
                    (x, z)
                    for z in z_values
                    for x in range(current_cell[0] - half, current_cell[0] + half + 1)
                )
            if dx:
                x_values = (
                    range(current_cell[0] + half - abs(dx) + 1, current_cell[0] + half + 1)
                    if dx > 0
                    else range(current_cell[0] - half, current_cell[0] - half + abs(dx))
                )
                exposed.update(
                    (x, z)
                    for x in x_values
                    for z in range(current_cell[1] - half, current_cell[1] + half + 1)
                )
            for key in exposed:
                written += self._write_cell(
                    window,
                    key,
                    center=current_cell,
                    current_surface=current_surface,
                )

        for key in tuple(self._dirty):
            written += self._write_cell(
                window,
                key,
                center=current_cell,
                current_surface=current_surface,
            )
        self._dirty.clear()

        target_dx = target_cell[0] - current_cell[0]
        target_dz = target_cell[1] - current_cell[1]
        magnitude = max(abs(target_dx), abs(target_dz))
        scale = 1.0 if magnitude <= half or magnitude == 0 else half / float(magnitude)
        target_column = half + int(round(target_dx * scale))
        target_row = half - int(round(target_dz * scale))
        window[7, target_row, target_column] = 1.0

        self._window = window
        self._center = current_cell
        self._surface = current_surface
        self.last_metrics["window_cells_written"] = written
        return window.copy()


def project_sparse_policy_lattices(
    cells: Iterable[Mapping[str, Any]],
    *,
    current_cell: tuple[int, int],
    target_cell: tuple[int, int],
    inventory: np.ndarray,
    action_mask: np.ndarray,
    elapsed_fraction: float = 0.0,
    yaw_degrees: float = 0.0,
    current_y: float | None = None,
    window_cache: WorldMapWindowCache | None = None,
) -> dict[str, Any]:
    """Project a Gym sparse map into the world-window policy contract.

    This adapter is shared by runtime consumers.  It intentionally uses only
    accumulated actor-scoped evidence; route cost and other oracle channels
    are not accepted by the function signature.  ``window_cache`` preserves
    the stable core between calls and only reprojects new or changed cells.
    """

    current_x, current_z = current_cell
    if math.hypot(target_cell[0] - current_x, target_cell[1] - current_z) < 0.5:
        raise ValueError("target must differ from the current cell")
    radians = math.radians(float(yaw_degrees))
    forward_x, forward_z = -math.sin(radians), math.cos(radians)
    right_x, right_z = forward_z, -forward_x
    records = list(cells)
    current_surface = (
        float(current_y) - 1.0
        if current_y is not None
        else next(
            (
                float(cell["surface_y_min"])
                for cell in records
                if int(cell["cell_x"]) == current_x
                and int(cell["cell_z"]) == current_z
                and cell.get("surface_y_min") is not None
            ),
            64.0,
        )
    )
    pyramid = np.zeros(
        (
            len(PYRAMID_SCALES),
            len(MAP_LATTICE_CHANNELS),
            PYRAMID_SIZE,
            PYRAMID_SIZE,
        ),
        dtype=np.float32,
    )
    half = PYRAMID_SIZE // 2

    for level, scale in enumerate(PYRAMID_SCALES):
        buckets: dict[
            tuple[int, int], list[tuple[float, float, bool, float, float]]
        ] = {}
        for cell in records:
            offset_x = int(cell["cell_x"]) - current_x
            offset_z = int(cell["cell_z"]) - current_z
            right = offset_x * right_x + offset_z * right_z
            forward = offset_x * forward_x + offset_z * forward_z
            column = half + math.floor((right + scale / 2.0) / scale)
            row = half - math.floor((forward + scale / 2.0) / scale)
            if not (0 <= row < PYRAMID_SIZE and 0 <= column < PYRAMID_SIZE):
                continue
            surface = cell.get("surface_y_max")
            elevation = float(surface) if surface is not None else 0.0
            buckets.setdefault((row, column), []).append(
                (
                    elevation,
                    _material_value(cell.get("block_counts") or {}),
                    (int(cell["cell_x"]), int(cell["cell_z"])) == target_cell,
                    float(max(0.0, float(cell.get("solid_stack_height") or 0.0))),
                    float(
                        max(
                            0.0,
                            float(
                                cell.get("head_clearance")
                                if cell.get("head_clearance") is not None
                                else 4.0
                            ),
                        )
                    ),
                )
            )
        for (row, column), values in buckets.items():
            elevations = [value[0] for value in values]
            pyramid[level, 0, row, column] = min(1.0, len(values) / float(scale * scale))
            # Match the academy convention: zero-height ground maps to 0.25,
            # with each additional block adding another quarter.
            center = float(np.median(elevations))
            pyramid[level, 1, row, column] = np.clip(
                (center - current_surface + 1.0) / 4.0, 0.0, 1.0
            )
            pyramid[level, 2, row, column] = np.clip(
                (max(elevations) - min(elevations)) / 8.0, 0.0, 1.0
            )
            pyramid[level, 3, row, column] = sum(value[2] for value in values) / len(values)
            pyramid[level, 4, row, column] = float(np.mean([value[1] for value in values]))
            pyramid[level, 5, row, column] = np.clip(
                max(value[3] for value in values) / 4.0, 0.0, 1.0
            )
            pyramid[level, 6, row, column] = np.clip(
                min(value[4] for value in values) / 4.0, 0.0, 1.0
            )

    gnomonic = np.zeros(
        (len(VIEW_LATTICE_CHANNELS), *GNOMONIC_SHAPE), dtype=np.float32
    )
    view_height, view_width = GNOMONIC_SHAPE
    candidates: list[tuple[float, int, Mapping[str, Any], float]] = []
    for cell in records:
        offset_x = int(cell["cell_x"]) - current_x
        offset_z = int(cell["cell_z"]) - current_z
        right = offset_x * right_x + offset_z * right_z
        forward = offset_x * forward_x + offset_z * forward_z
        if forward <= 0.0:
            continue
        angle = math.degrees(math.atan2(right, forward))
        if abs(angle) > 45.0:
            continue
        column = int(np.clip(round((angle / 90.0 + 0.5) * (view_width - 1)), 0, view_width - 1))
        candidates.append((math.hypot(right, forward), column, cell, forward))
    for ray_distance, column, cell, forward in sorted(candidates, reverse=True):
        surface = float(cell.get("surface_y_max") or current_surface)
        elevation = math.degrees(math.atan2(surface - current_surface, forward))
        row = int(np.clip(round((0.5 - elevation / 60.0) * (view_height - 1)), 0, view_height - 1))
        gnomonic[0, row:, column] = 1.0
        gnomonic[1, row:, column] = 1.0 - min(1.0, ray_distance / 32.0)
        gnomonic[2, row:, column] = np.clip((surface - current_surface + 4.0) / 8.0, 0.0, 1.0)
        if (int(cell["cell_x"]), int(cell["cell_z"])) == target_cell:
            gnomonic[3, max(0, row - 1) : row + 1, column] = 1.0
        gnomonic[4, row:, column] = _material_value(cell.get("block_counts") or {})

    legacy_grid = np.zeros((len(GRID_CHANNELS), 14, 9), dtype=np.float32)
    orientation = np.zeros(4, dtype=np.float32)
    orientation[int(round(float(yaw_degrees) / 90.0)) % 4] = 1.0
    if window_cache is None:
        map_window = project_world_map_window(
            records,
            current_cell=current_cell,
            target_cell=target_cell,
            current_y=current_y,
        )
    else:
        window_cache.update(records)
        map_window = window_cache.render(
            current_cell=current_cell,
            target_cell=target_cell,
            current_y=current_y,
        )

    return {
        "global_grid": legacy_grid,
        "map_window": map_window,
        "grid_pyramid": pyramid,
        "gnomonic_lattice": gnomonic,
        "orientation": orientation,
        "inventory": np.asarray(inventory, dtype=np.int32),
        "elapsed_fraction": np.float32(np.clip(elapsed_fraction, 0.0, 1.0)),
        "action_mask": np.asarray(action_mask, dtype=np.int8),
    }


def epsilon_greedy_probabilities(
    values: np.ndarray, mask: np.ndarray, epsilon: float
) -> np.ndarray:
    valid = np.flatnonzero(np.asarray(mask, dtype=bool))
    if not len(valid):
        raise ValueError("at least one academy action must be valid")
    action_count = int(np.asarray(values).size)
    if np.asarray(mask).size != action_count:
        raise ValueError("action values and validity mask must have matching sizes")
    probabilities = np.zeros(action_count, dtype=np.float32)
    probabilities[valid] = float(epsilon) / len(valid)
    valid_values = np.asarray(values)[valid]
    greedy = valid[np.flatnonzero(np.isclose(valid_values, valid_values.max()))]
    probabilities[greedy] += (1.0 - float(epsilon)) / len(greedy)
    return probabilities


def choose_epsilon_greedy(
    values: np.ndarray,
    mask: np.ndarray,
    *,
    epsilon: float,
    rng: np.random.Generator,
) -> int:
    probabilities = epsilon_greedy_probabilities(values, mask, epsilon)
    return int(rng.choice(len(probabilities), p=probabilities))
