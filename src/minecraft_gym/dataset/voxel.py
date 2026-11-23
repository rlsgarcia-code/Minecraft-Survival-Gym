"""Sparse, actor-scoped voxel memory reconstructed from Minecraft raycasts.

The voxel map is the lossless spatial memory.  Two-dimensional navigation
maps are projections of this memory, not an independent source of truth.
Only voxels traversed or hit by the deployable ray sensor become known.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import math
from typing import Iterable, Literal

import numpy as np

from .contracts import RaycastFrameV1


Vector3 = tuple[float, float, float]
VoxelIndex = tuple[int, int, int]
ChunkIndex = tuple[int, int, int]

VOXEL_VOLUME_CHANNELS = (
    "known",
    "solid_probability",
    "material_class",
    "confidence",
    "visited",
)

VOXEL_MINIMAP_CHANNELS = (
    "known_column",
    "relative_floor_height",
    "obstacle_height",
    "head_clearance",
    "traversability",
    "material_class",
)


def trace_voxels_3d(
    origin: Vector3,
    endpoint: Vector3,
    *,
    resolution: float = 1.0,
) -> tuple[VoxelIndex, ...]:
    """Return voxels crossed by a segment using 3-D DDA traversal.

    The endpoint is interpreted as a point on the open segment boundary.  For
    a block hit this therefore returns the free voxels before the exact hit
    voxel, whose integer position is supplied separately by the renderer.
    """

    if not math.isfinite(resolution) or resolution <= 0:
        raise ValueError("resolution must be finite and positive")
    if any(not math.isfinite(value) for value in (*origin, *endpoint)):
        raise ValueError("segment coordinates must be finite")

    delta = tuple(endpoint[i] - origin[i] for i in range(3))
    start = tuple(
        math.nextafter(origin[i], endpoint[i]) if delta[i] else origin[i]
        for i in range(3)
    )
    end = tuple(
        math.nextafter(endpoint[i], origin[i]) if delta[i] else endpoint[i]
        for i in range(3)
    )
    cell = [math.floor(value / resolution) for value in start]
    end_cell = tuple(math.floor(value / resolution) for value in end)
    cells = [tuple(cell)]
    if tuple(cell) == end_cell:
        return tuple(cells)

    step: list[int] = []
    t_max: list[float] = []
    t_delta: list[float] = []
    for axis in range(3):
        change = delta[axis]
        direction = 1 if change > 0 else -1 if change < 0 else 0
        step.append(direction)
        if direction:
            boundary = (cell[axis] + (1 if direction > 0 else 0)) * resolution
            t_max.append((boundary - origin[axis]) / change)
            t_delta.append(resolution / abs(change))
        else:
            t_max.append(math.inf)
            t_delta.append(math.inf)

    max_cells = sum(abs(end_cell[i] - cell[i]) for i in range(3)) + 4
    while tuple(cell) != end_cell:
        next_t = min(t_max)
        # Crossing an edge or corner advances every tied axis.  This is the
        # same thin-ray convention used by Minecraft's block intersection.
        for axis in range(3):
            if math.isclose(t_max[axis], next_t, abs_tol=1e-12):
                cell[axis] += step[axis]
                t_max[axis] += t_delta[axis]
        cells.append(tuple(cell))
        if len(cells) > max_cells:
            raise RuntimeError("voxel traversal exceeded deterministic bound")
    return tuple(cells)


def _material_class(block_name: str | None) -> float:
    """Return a stable coarse material class in [0, 1]."""

    name = (block_name or "").lower()
    if not name:
        return 0.0
    if "water" in name:
        return 0.10
    if "lava" in name:
        return 0.20
    if any(token in name for token in ("leaves", "vine", "plant", "grass")):
        return 0.35
    if any(token in name for token in ("wood", "log", "plank")):
        return 0.50
    if any(token in name for token in ("sand", "snow", "gravel", "dirt")):
        return 0.65
    if any(token in name for token in ("stone", "ore", "quartz", "brick")):
        return 0.80
    return 1.0


@dataclass(slots=True)
class VoxelCellState:
    """Evidence for exactly one Minecraft block-sized volume."""

    free_count: int = 0
    solid_count: int = 0
    visited_count: int = 0
    block_counts: Counter[str] = field(default_factory=Counter)
    last_seen_tick: int = 0
    source: Literal["raycast", "body_occupancy", "authoritative_event"] = "raycast"

    @property
    def evidence_count(self) -> int:
        return self.free_count + self.solid_count

    @property
    def confidence(self) -> float:
        return 1.0 - math.exp(-self.evidence_count / 4.0)

    @property
    def solid_probability(self) -> float:
        """Compatibility projection: whether a block was confirmed here.

        A ray traversing a voxel is not negative evidence for block presence:
        Minecraft blocks may have partial collision/selection shapes.  The raw
        free and hit counters remain separate and ``partial_geometry`` exposes
        cells for which both kinds of evidence exist.
        """

        if not self.evidence_count:
            return 0.0
        return float(self.solid_count > 0)

    @property
    def partial_geometry(self) -> bool:
        return self.free_count > 0 and self.solid_count > 0

    @property
    def dominant_block(self) -> str | None:
        return self.block_counts.most_common(1)[0][0] if self.block_counts else None


@dataclass(frozen=True, slots=True)
class VoxelMapUpdate:
    actor_id: str
    server_tick: int
    changed_voxels: tuple[VoxelIndex, ...]
    free_observations: int
    solid_observations: int


class SparseVoxelWorldMemory:
    """Unbounded sparse voxel memory stored in versioned 16³ chunks."""

    def __init__(
        self,
        map_id: str,
        actor_id: str,
        *,
        resolution: float = 1.0,
        chunk_size: int = 16,
    ) -> None:
        if not map_id or not actor_id:
            raise ValueError("map_id and actor_id cannot be empty")
        if not math.isfinite(resolution) or resolution <= 0:
            raise ValueError("resolution must be finite and positive")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self.map_id = map_id
        self.actor_id = actor_id
        self.resolution = float(resolution)
        self.chunk_size = int(chunk_size)
        self._chunks: dict[ChunkIndex, dict[VoxelIndex, VoxelCellState]] = {}
        self._chunk_versions: Counter[ChunkIndex] = Counter()
        self.last_tick: int | None = None
        self.last_visited_voxel: VoxelIndex | None = None

    def _chunk_and_local(self, voxel: VoxelIndex) -> tuple[ChunkIndex, VoxelIndex]:
        chunk = tuple(value // self.chunk_size for value in voxel)
        local = tuple(value % self.chunk_size for value in voxel)
        return chunk, local  # type: ignore[return-value]

    def _state(self, voxel: VoxelIndex) -> VoxelCellState:
        chunk, local = self._chunk_and_local(voxel)
        return self._chunks.setdefault(chunk, {}).setdefault(local, VoxelCellState())

    def get(self, voxel: VoxelIndex) -> VoxelCellState | None:
        chunk, local = self._chunk_and_local(voxel)
        return self._chunks.get(chunk, {}).get(local)

    @property
    def voxel_count(self) -> int:
        return sum(len(chunk) for chunk in self._chunks.values())

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    def chunk_version(self, chunk: ChunkIndex) -> int:
        return int(self._chunk_versions[chunk])

    def _touch(self, voxel: VoxelIndex) -> None:
        chunk, _ = self._chunk_and_local(voxel)
        self._chunk_versions[chunk] += 1

    def integrate(
        self,
        raycast: RaycastFrameV1,
        *,
        visited_position: Vector3 | None = None,
    ) -> VoxelMapUpdate:
        if raycast.actor_id != self.actor_id:
            raise ValueError("raycast actor does not match voxel map actor")
        if self.last_tick is not None and raycast.server_tick < self.last_tick:
            raise ValueError("voxel map cannot integrate a reversed server tick")

        changed: set[VoxelIndex] = set()
        free_observations = solid_observations = 0
        for sample in raycast.samples:
            endpoint = tuple(
                sample.origin[index]
                + sample.direction[index] * sample.traversed_distance
                for index in range(3)
            )
            traversed = trace_voxels_3d(
                sample.origin,
                endpoint,  # type: ignore[arg-type]
                resolution=self.resolution,
            )
            hit_voxel: VoxelIndex | None = None
            if sample.termination == "block_hit":
                assert sample.block_position is not None
                hit_voxel = tuple(
                    math.floor(value / self.resolution)
                    for value in sample.block_position
                )  # type: ignore[assignment]
            free_voxels: Iterable[VoxelIndex] = (
                voxel for voxel in traversed if voxel != hit_voxel
            )
            if sample.termination == "unloaded_chunk":
                free_voxels = traversed[:-1]
            for voxel in free_voxels:
                state = self._state(voxel)
                state.free_count += 1
                state.last_seen_tick = raycast.server_tick
                state.source = "raycast"
                self._touch(voxel)
                changed.add(voxel)
                free_observations += 1

            if hit_voxel is not None:
                assert sample.block_name is not None
                state = self._state(hit_voxel)
                state.solid_count += 1
                state.block_counts[sample.block_name] += 1
                state.last_seen_tick = raycast.server_tick
                state.source = "raycast"
                self._touch(hit_voxel)
                changed.add(hit_voxel)
                solid_observations += 1

        if visited_position is not None:
            if any(not math.isfinite(value) for value in visited_position):
                raise ValueError("visited_position must contain finite values")
            visited = tuple(
                math.floor(value / self.resolution) for value in visited_position
            )
            feet_voxel = visited  # type: ignore[assignment]
            head_voxel = (visited[0], visited[1] + 1, visited[2])
            for voxel in (feet_voxel, head_voxel):
                state = self._state(voxel)
                state.free_count += 1
                state.last_seen_tick = raycast.server_tick
                state.source = "body_occupancy"
                self._touch(voxel)
                changed.add(voxel)
            self._state(feet_voxel).visited_count += 1
            self.last_visited_voxel = visited  # type: ignore[assignment]

        self.last_tick = raycast.server_tick
        return VoxelMapUpdate(
            actor_id=self.actor_id,
            server_tick=raycast.server_tick,
            changed_voxels=tuple(sorted(changed)),
            free_observations=free_observations,
            solid_observations=solid_observations,
        )

    def set_authoritative_block(
        self,
        voxel: VoxelIndex,
        block_name: str | None,
        *,
        server_tick: int,
    ) -> None:
        """Apply a break/place event without retaining stale occupancy."""

        if self.last_tick is not None and server_tick < self.last_tick:
            raise ValueError("authoritative event cannot reverse server tick")
        state = self._state(voxel)
        state.free_count = int(block_name is None)
        state.solid_count = int(block_name is not None)
        state.block_counts.clear()
        if block_name is not None:
            state.block_counts[block_name] = 1
        state.last_seen_tick = server_tick
        state.source = "authoritative_event"
        self._touch(voxel)
        self.last_tick = server_tick

    def iter_voxels(self) -> Iterable[tuple[VoxelIndex, VoxelCellState]]:
        for chunk, cells in self._chunks.items():
            base = tuple(value * self.chunk_size for value in chunk)
            for local, state in cells.items():
                yield (
                    tuple(base[i] + local[i] for i in range(3)),  # type: ignore[misc]
                    state,
                )

    def summary(self) -> dict[str, object]:
        """Return compact geometry metrics without exposing the dense volume."""

        voxels = list(self.iter_voxels())
        solid = [
            (position, state)
            for position, state in voxels
            if state.evidence_count and state.solid_probability >= 0.5
        ]
        free = [
            (position, state)
            for position, state in voxels
            if state.evidence_count and not state.solid_count
        ]
        partial = [
            (position, state)
            for position, state in voxels
            if state.partial_geometry
        ]
        columns: dict[tuple[int, int], list[int]] = {}
        for (x, y, z), _state in solid:
            columns.setdefault((x, z), []).append(y)
        vertically_ambiguous = 0
        for levels in columns.values():
            ordered = sorted(set(levels))
            if any(upper > lower + 1 for lower, upper in zip(ordered, ordered[1:])):
                vertically_ambiguous += 1
        if voxels:
            positions = [position for position, _state in voxels]
            bounds = (
                min(item[0] for item in positions),
                min(item[1] for item in positions),
                min(item[2] for item in positions),
                max(item[0] for item in positions),
                max(item[1] for item in positions),
                max(item[2] for item in positions),
            )
        else:
            bounds = None
        return {
            "interface": "minecraft-voxel-world-summary/v1",
            "map_id": self.map_id,
            "actor_id": self.actor_id,
            "server_tick": self.last_tick or 0,
            "resolution": self.resolution,
            "known_voxel_count": len(voxels),
            "free_voxel_count": len(free),
            "solid_voxel_count": len(solid),
            "partial_geometry_voxel_count": len(partial),
            "chunk_count": self.chunk_count,
            "vertical_column_count": len(columns),
            "vertically_ambiguous_column_count": vertically_ambiguous,
            "bounds_xyz": bounds,
            "current_voxel": self.last_visited_voxel,
        }

    def snapshot_voxels(
        self,
        *,
        center: VoxelIndex | None = None,
        horizontal_radius: int = 16,
        vertical_radius: int = 8,
    ) -> tuple[dict[str, object], ...]:
        """Return a bounded JSON-ready crop for Observer diagnostics."""

        if horizontal_radius < 0 or vertical_radius < 0:
            raise ValueError("voxel snapshot radii cannot be negative")
        center = center or self.last_visited_voxel
        records: list[dict[str, object]] = []
        for (x, y, z), state in self.iter_voxels():
            if center is not None and (
                abs(x - center[0]) > horizontal_radius
                or abs(z - center[2]) > horizontal_radius
                or abs(y - center[1]) > vertical_radius
            ):
                continue
            records.append(
                {
                    "x": x,
                    "y": y,
                    "z": z,
                    "free_count": state.free_count,
                    "solid_count": state.solid_count,
                    "solid_probability": state.solid_probability,
                    "block_present": bool(state.solid_count),
                    "partial_geometry": state.partial_geometry,
                    "confidence": state.confidence,
                    "visited_count": state.visited_count,
                    "block_name": state.dominant_block,
                    "last_seen_tick": state.last_seen_tick,
                    "source": state.source,
                }
            )
        records.sort(key=lambda item: (int(item["y"]), int(item["z"]), int(item["x"])))
        return tuple(records)

    def local_volume(
        self,
        *,
        center: VoxelIndex,
        shape: tuple[int, int, int] = (33, 17, 33),
    ) -> np.ndarray:
        """Return C×Y×Z×X policy channels centered on a voxel."""

        width, height, depth = shape
        if min(shape) <= 0 or any(value % 2 == 0 for value in shape):
            raise ValueError("local volume dimensions must be positive and odd")
        volume = np.zeros(
            (len(VOXEL_VOLUME_CHANNELS), height, depth, width), dtype=np.float32
        )
        half_x, half_y, half_z = width // 2, height // 2, depth // 2
        for out_y in range(height):
            world_y = center[1] + out_y - half_y
            for out_z in range(depth):
                world_z = center[2] + out_z - half_z
                for out_x in range(width):
                    world_x = center[0] + out_x - half_x
                    state = self.get((world_x, world_y, world_z))
                    if state is None:
                        continue
                    volume[0, out_y, out_z, out_x] = float(state.evidence_count > 0)
                    volume[1, out_y, out_z, out_x] = state.solid_probability
                    volume[2, out_y, out_z, out_x] = _material_class(
                        state.dominant_block
                    )
                    volume[3, out_y, out_z, out_x] = state.confidence
                    volume[4, out_y, out_z, out_x] = float(state.visited_count > 0)
        return volume

    def navigation_minimap(
        self,
        *,
        center_feet: VoxelIndex,
        size: int = 65,
        vertical_radius: int = 8,
    ) -> np.ndarray:
        """Derive a 2-D navigation map from the canonical voxel memory.

        A column is traversable only when its support and both body voxels are
        known.  Unknown is never silently promoted to free space.
        """

        if size <= 0 or size % 2 == 0:
            raise ValueError("minimap size must be positive and odd")
        if vertical_radius < 2:
            raise ValueError("vertical_radius must be at least two")
        result = np.zeros(
            (len(VOXEL_MINIMAP_CHANNELS), size, size), dtype=np.float32
        )
        half = size // 2
        cx, feet_y, cz = center_feet
        for row in range(size):
            z = cz + half - row
            for column in range(size):
                x = cx + column - half
                states = {
                    y: self.get((x, y, z))
                    for y in range(feet_y - vertical_radius, feet_y + vertical_radius + 1)
                }
                known_states = [state for state in states.values() if state is not None]
                if not known_states:
                    continue
                result[0, row, column] = max(
                    state.confidence for state in known_states
                )
                support_levels = [
                    y
                    for y, state in states.items()
                    if y < feet_y and state is not None and state.solid_probability >= 0.5
                ]
                if not support_levels:
                    continue
                floor_y = max(support_levels)
                result[1, row, column] = np.clip(
                    (floor_y - (feet_y - 1) + 8.0) / 16.0, 0.0, 1.0
                )
                obstacle = 0
                y = floor_y + 1
                while (
                    states.get(y) is not None
                    and states[y].solid_probability >= 0.5  # type: ignore[union-attr]
                ):
                    obstacle += 1
                    y += 1
                result[2, row, column] = np.clip(obstacle / 4.0, 0.0, 1.0)
                clearance = 0
                y = floor_y + 1
                while y <= feet_y + vertical_radius:
                    state = states.get(y)
                    if state is None or state.solid_probability >= 0.5:
                        break
                    clearance += 1
                    y += 1
                result[3, row, column] = np.clip(clearance / 4.0, 0.0, 1.0)
                body_states = (states.get(floor_y + 1), states.get(floor_y + 2))
                body_known_free = all(
                    state is not None and state.solid_probability < 0.5
                    for state in body_states
                )
                step_height = floor_y - (feet_y - 1)
                result[4, row, column] = float(
                    body_known_free and -3 <= step_height <= 1
                )
                support = states[floor_y]
                result[5, row, column] = _material_class(
                    support.dominant_block if support is not None else None
                )
        return result
