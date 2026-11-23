"""Pixel rays and actor-scoped spatial-semantic lattice maps.

The renderer/world adapter owns block intersection.  This module owns the
portable Gym contract: camera projection, deterministic X/Z grid traversal,
and additive map evidence that can be replayed from an episode.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import json
import math
import threading
from typing import Iterable
import urllib.error
import urllib.parse
import urllib.request

from pydantic import ValidationError

from .contracts import (
    CameraCalibrationV1,
    CameraPoseV1,
    RaycastFrameV1,
    SpatialMapCellDeltaV1,
    SpatialMapSummaryV1,
    SpatialMapUpdateV1,
    SphericalValueCellV1,
    SphericalValueLatticeV1,
)
from .voxel import SparseVoxelWorldMemory


Vector3 = tuple[float, float, float]

SPATIAL_PYRAMID_LEVELS = (
    ("local", "Entorno", 1),
    ("regional", "Região", 4),
    ("world", "Mundo", 16),
)
SPATIAL_PYRAMID_VIEW_SIZE = 33


class SpatialStreamError(RuntimeError):
    pass


def ray_direction_from_pixel(
    pixel_x: int,
    pixel_y: int,
    calibration: CameraCalibrationV1,
    pose: CameraPoseV1,
) -> Vector3:
    """Project a pixel center into a unit world-space ray direction.

    ``forward/right/up`` come from the renderer camera itself, avoiding any
    ambiguity in Minecraft, Mineflayer, and Three.js yaw conventions.
    """

    if not 0 <= pixel_x < calibration.width:
        raise ValueError("pixel_x is outside camera width")
    if not 0 <= pixel_y < calibration.height:
        raise ValueError("pixel_y is outside camera height")
    ndc_x = 2.0 * (pixel_x + 0.5) / calibration.width - 1.0
    ndc_y = 1.0 - 2.0 * (pixel_y + 0.5) / calibration.height
    tan_half_fov = math.tan(math.radians(calibration.vertical_fov_degrees) / 2.0)
    aspect = calibration.width / calibration.height
    camera_x = ndc_x * aspect * tan_half_fov
    camera_y = ndc_y * tan_half_fov
    direction = tuple(
        pose.forward[index]
        + camera_x * pose.right[index]
        + camera_y * pose.up[index]
        for index in range(3)
    )
    norm = math.sqrt(sum(value * value for value in direction))
    if norm == 0:
        raise ValueError("camera basis produced a zero-length ray")
    return tuple(value / norm for value in direction)  # type: ignore[return-value]


def trace_lattice_cells_xz(
    origin: Vector3,
    endpoint: Vector3,
    *,
    resolution: float = 1.0,
) -> tuple[tuple[int, int], ...]:
    """Return X/Z cells crossed by a segment using 2-D voxel traversal."""

    if not math.isfinite(resolution) or resolution <= 0:
        raise ValueError("resolution must be finite and positive")
    if any(not math.isfinite(value) for value in (*origin, *endpoint)):
        raise ValueError("segment coordinates must be finite")

    x0, z0 = origin[0], origin[2]
    x1, z1 = endpoint[0], endpoint[2]
    dx, dz = x1 - x0, z1 - z0
    # Pick the cell just inside the segment when the origin lies on a boundary.
    start_x = math.nextafter(x0, x1) if dx else x0
    start_z = math.nextafter(z0, z1) if dz else z0
    cell_x = math.floor(start_x / resolution)
    cell_z = math.floor(start_z / resolution)
    end_x = math.floor(math.nextafter(x1, x0) / resolution) if dx else cell_x
    end_z = math.floor(math.nextafter(z1, z0) / resolution) if dz else cell_z
    cells = [(cell_x, cell_z)]
    if (cell_x, cell_z) == (end_x, end_z):
        return tuple(cells)

    step_x = 1 if dx > 0 else -1 if dx < 0 else 0
    step_z = 1 if dz > 0 else -1 if dz < 0 else 0
    if step_x:
        boundary_x = (cell_x + (1 if step_x > 0 else 0)) * resolution
        t_max_x = (boundary_x - x0) / dx
        t_delta_x = resolution / abs(dx)
    else:
        t_max_x = math.inf
        t_delta_x = math.inf
    if step_z:
        boundary_z = (cell_z + (1 if step_z > 0 else 0)) * resolution
        t_max_z = (boundary_z - z0) / dz
        t_delta_z = resolution / abs(dz)
    else:
        t_max_z = math.inf
        t_delta_z = math.inf

    # The bound makes malformed floating-point inputs fail rather than loop.
    max_cells = abs(end_x - cell_x) + abs(end_z - cell_z) + 3
    while (cell_x, cell_z) != (end_x, end_z):
        if t_max_x < t_max_z:
            cell_x += step_x
            t_max_x += t_delta_x
        elif t_max_z < t_max_x:
            cell_z += step_z
            t_max_z += t_delta_z
        else:
            cell_x += step_x
            cell_z += step_z
            t_max_x += t_delta_x
            t_max_z += t_delta_z
        cells.append((cell_x, cell_z))
        if len(cells) > max_cells:
            raise RuntimeError("lattice traversal exceeded deterministic bound")
    return tuple(cells)


@dataclass(slots=True)
class SpatialCellState:
    visited_count: int = 0
    free_count: int = 0
    hit_count: int = 0
    block_counts: Counter[str] = field(default_factory=Counter)
    block_y_counts: Counter[int] = field(default_factory=Counter)
    surface_y_min: float | None = None
    surface_y_max: float | None = None
    visited_feet_y: float | None = None
    topology_stack_height: int | None = None
    topology_head_clearance: float | None = None
    last_seen_tick: int = 0


def _vertical_topology(state: SpatialCellState) -> tuple[int, float]:
    """Derive compact wall/ceiling evidence from exact hit block levels.

    The first value is the longest observed contiguous vertical run of blocks.
    The second is the smallest observed free vertical gap, capped at four
    blocks. Four therefore means "open or not measured closely enough"; a
    value below two is known to be too low for an upright Minecraft body.
    """

    if (
        state.topology_stack_height is not None
        and state.topology_head_clearance is not None
    ):
        return state.topology_stack_height, state.topology_head_clearance

    levels = sorted(level for level, count in state.block_y_counts.items() if count)
    if not levels:
        return 0, 4.0

    longest_run = run = 1
    free_gaps: list[float] = []
    for lower, upper in zip(levels, levels[1:], strict=False):
        if upper == lower + 1:
            run += 1
            longest_run = max(longest_run, run)
        else:
            run = 1
            gap = upper - lower - 1
            if gap > 0:
                free_gaps.append(float(gap))

    if state.visited_feet_y is not None:
        feet_level = math.floor(state.visited_feet_y + 1e-6)
        overhead = [level for level in levels if level >= feet_level]
        if overhead:
            free_gaps.append(max(0.0, float(min(overhead)) - state.visited_feet_y))

    clearance = min(free_gaps) if free_gaps else 4.0
    # A single observed ground block is the floor, not a barrier.  The map
    # channel measures additional contiguous blocks stacked above that base,
    # matching the academy's ``top_y - base_y`` convention.
    return max(0, longest_run - 1), min(4.0, clearance)


@dataclass(slots=True)
class SphericalCellState:
    observation_count: int = 0
    hit_count: int = 0
    block_counts: Counter[str] = field(default_factory=Counter)
    distance_sum: float = 0.0
    last_seen_tick: int | None = None


class SphericalValueLattice:
    """Replayable world-aligned angular memory for active exploration.

    Azimuth wraps around the horizon and elevation spans pole to pole.  All
    bins exist in snapshots, so unseen directions retain an explicit value.
    """

    def __init__(
        self,
        lattice_id: str,
        actor_id: str,
        *,
        azimuth_bins: int = 24,
        elevation_bins: int = 6,
    ) -> None:
        if not lattice_id or not actor_id:
            raise ValueError("spherical lattice and actor ids cannot be empty")
        if azimuth_bins <= 0 or elevation_bins <= 0:
            raise ValueError("spherical lattice dimensions must be positive")
        self.lattice_id = lattice_id
        self.actor_id = actor_id
        self.azimuth_bins = azimuth_bins
        self.elevation_bins = elevation_bins
        self.cells: dict[tuple[int, int], SphericalCellState] = {}
        self.last_tick = 0

    @staticmethod
    def direction_angles(direction: Vector3) -> tuple[float, float]:
        norm = math.sqrt(sum(value * value for value in direction))
        if norm == 0:
            raise ValueError("direction cannot be zero")
        x, y, z = (value / norm for value in direction)
        return math.degrees(math.atan2(x, z)), math.degrees(math.asin(max(-1.0, min(1.0, y))))

    def indices_for_angles(self, azimuth: float, elevation: float) -> tuple[int, int]:
        wrapped = ((azimuth + 180.0) % 360.0) - 180.0
        clamped = max(-90.0, min(math.nextafter(90.0, -math.inf), elevation))
        azimuth_bin = int((wrapped + 180.0) / 360.0 * self.azimuth_bins)
        elevation_bin = int((clamped + 90.0) / 180.0 * self.elevation_bins)
        return azimuth_bin, elevation_bin

    def integrate(self, raycast: RaycastFrameV1) -> None:
        if raycast.actor_id != self.actor_id:
            raise ValueError("raycast actor does not match spherical lattice actor")
        if raycast.server_tick < self.last_tick:
            raise ValueError("spherical lattice cannot integrate a reversed tick")
        for sample in raycast.samples:
            azimuth, elevation = self.direction_angles(sample.direction)
            cell = self.cells.setdefault(
                self.indices_for_angles(azimuth, elevation), SphericalCellState()
            )
            cell.observation_count += 1
            cell.distance_sum += sample.traversed_distance
            if sample.termination == "block_hit":
                assert sample.block_name is not None
                cell.hit_count += 1
                cell.block_counts[sample.block_name] += 1
            cell.last_seen_tick = raycast.server_tick
        self.last_tick = raycast.server_tick

    def summary(
        self, *, current_yaw_degrees: float, current_pitch_degrees: float
    ) -> SphericalValueLatticeV1:
        cells: list[SphericalValueCellV1] = []
        for elevation_bin in range(self.elevation_bins):
            for azimuth_bin in range(self.azimuth_bins):
                state = self.cells.get((azimuth_bin, elevation_bin), SphericalCellState())
                cells.append(
                    SphericalValueCellV1(
                        azimuth_bin=azimuth_bin,
                        elevation_bin=elevation_bin,
                        azimuth_center_degrees=(azimuth_bin + 0.5) * 360.0 / self.azimuth_bins - 180.0,
                        elevation_center_degrees=(elevation_bin + 0.5) * 180.0 / self.elevation_bins - 90.0,
                        observation_count=state.observation_count,
                        hit_count=state.hit_count,
                        block_counts=dict(state.block_counts),
                        mean_distance=(
                            state.distance_sum / state.observation_count
                            if state.observation_count
                            else None
                        ),
                        last_seen_tick=state.last_seen_tick,
                        novelty_value=1.0 / math.sqrt(state.observation_count + 1.0),
                    )
                )
        observed = sum(cell.observation_count > 0 for cell in cells)
        total = self.azimuth_bins * self.elevation_bins
        return SphericalValueLatticeV1(
            lattice_id=self.lattice_id,
            actor_id=self.actor_id,
            server_tick=self.last_tick,
            azimuth_bins=self.azimuth_bins,
            elevation_bins=self.elevation_bins,
            current_yaw_degrees=current_yaw_degrees,
            current_pitch_degrees=current_pitch_degrees,
            observed_bin_count=observed,
            total_bin_count=total,
            coverage_ratio=observed / total,
            cells=tuple(cells),
        )


@dataclass(slots=True)
class _DeltaAccumulator:
    visited: int = 0
    free: int = 0
    hits: int = 0
    blocks: Counter[str] = field(default_factory=Counter)
    block_y: Counter[int] = field(default_factory=Counter)
    y_values: list[float] = field(default_factory=list)


class SpatialLatticeMap:
    """Deterministic actor-scoped map rebuilt from raycast episode records."""

    def __init__(self, map_id: str, actor_id: str, *, resolution: float = 1.0):
        if not map_id:
            raise ValueError("map_id cannot be empty")
        if not actor_id:
            raise ValueError("actor_id cannot be empty")
        if not math.isfinite(resolution) or resolution <= 0:
            raise ValueError("resolution must be finite and positive")
        self.map_id = map_id
        self.actor_id = actor_id
        self.resolution = resolution
        self.cells: dict[tuple[int, int], SpatialCellState] = {}
        self.last_tick: int | None = None
        self.last_visited_cell: tuple[int, int] | None = None

    def integrate(
        self,
        raycast: RaycastFrameV1,
        *,
        visited_position: Vector3 | None = None,
        scope: str = "privileged_derived",
    ) -> SpatialMapUpdateV1:
        """Apply one raycast frame and return its replayable additive delta."""

        if raycast.actor_id != self.actor_id:
            raise ValueError("raycast actor does not match spatial map actor")
        if self.last_tick is not None and raycast.server_tick < self.last_tick:
            raise ValueError("spatial map cannot integrate a reversed server tick")
        aggregate: dict[tuple[int, int], _DeltaAccumulator] = {}

        def accumulator(cell: tuple[int, int]) -> _DeltaAccumulator:
            return aggregate.setdefault(cell, _DeltaAccumulator())

        for sample in raycast.samples:
            endpoint = tuple(
                sample.origin[index]
                + sample.direction[index] * sample.traversed_distance
                for index in range(3)
            )
            cells = trace_lattice_cells_xz(
                sample.origin,
                endpoint,  # type: ignore[arg-type]
                resolution=self.resolution,
            )
            if sample.termination == "block_hit":
                assert sample.block_position is not None
                assert sample.block_name is not None
                assert sample.contact_point is not None
                hit_cell = (
                    math.floor(sample.block_position[0] / self.resolution),
                    math.floor(sample.block_position[2] / self.resolution),
                )
                for cell in cells:
                    if cell != hit_cell:
                        accumulator(cell).free += 1
                hit = accumulator(hit_cell)
                hit.hits += 1
                hit.blocks[sample.block_name] += 1
                hit.block_y[sample.block_position[1]] += 1
                hit.y_values.append(sample.contact_point[1])
            else:
                free_cells: Iterable[tuple[int, int]] = cells
                # The final cell is unknown when traversal ended at an
                # unloaded-chunk boundary; earlier cells remain valid evidence.
                if sample.termination == "unloaded_chunk":
                    free_cells = cells[:-1]
                for cell in free_cells:
                    accumulator(cell).free += 1

        if visited_position is not None:
            if any(not math.isfinite(value) for value in visited_position):
                raise ValueError("visited_position must contain finite values")
            visited_cell = (
                math.floor(visited_position[0] / self.resolution),
                math.floor(visited_position[2] / self.resolution),
            )
            accumulator(visited_cell).visited += 1
            self.last_visited_cell = visited_cell

        deltas: list[SpatialMapCellDeltaV1] = []
        for cell, evidence in sorted(aggregate.items()):
            delta = SpatialMapCellDeltaV1(
                cell_x=cell[0],
                cell_z=cell[1],
                resolution=self.resolution,
                visited_count_delta=evidence.visited,
                free_count_delta=evidence.free,
                hit_count_delta=evidence.hits,
                block_counts_delta=dict(evidence.blocks),
                surface_y_min=min(evidence.y_values) if evidence.y_values else None,
                surface_y_max=max(evidence.y_values) if evidence.y_values else None,
                last_seen_tick=raycast.server_tick,
            )
            deltas.append(delta)
            state = self.cells.setdefault(cell, SpatialCellState())
            state.visited_count += evidence.visited
            state.free_count += evidence.free
            state.hit_count += evidence.hits
            state.block_counts.update(evidence.blocks)
            state.block_y_counts.update(evidence.block_y)
            if evidence.y_values:
                low, high = min(evidence.y_values), max(evidence.y_values)
                state.surface_y_min = (
                    low if state.surface_y_min is None else min(state.surface_y_min, low)
                )
                state.surface_y_max = (
                    high if state.surface_y_max is None else max(state.surface_y_max, high)
                )
            state.last_seen_tick = raycast.server_tick

        if visited_position is not None:
            visited_state = self.cells.setdefault(visited_cell, SpatialCellState())
            visited_state.visited_feet_y = float(visited_position[1])

        self.last_tick = raycast.server_tick
        return SpatialMapUpdateV1(
            map_id=self.map_id,
            actor_id=self.actor_id,
            server_tick=raycast.server_tick,
            resolution=self.resolution,
            scope=scope,
            raycast=raycast,
            deltas=tuple(deltas),
        )

    def snapshot(self) -> dict[tuple[int, int], SpatialCellState]:
        """Return defensive copies suitable for visualization or assertions."""

        return {
            cell: SpatialCellState(
                visited_count=state.visited_count,
                free_count=state.free_count,
                hit_count=state.hit_count,
                block_counts=Counter(state.block_counts),
                block_y_counts=Counter(state.block_y_counts),
                surface_y_min=state.surface_y_min,
                surface_y_max=state.surface_y_max,
                visited_feet_y=state.visited_feet_y,
                topology_stack_height=state.topology_stack_height,
                topology_head_clearance=state.topology_head_clearance,
                last_seen_tick=state.last_seen_tick,
            )
            for cell, state in self.cells.items()
        }

    def summary(self, *, frontier_limit: int = 12) -> SpatialMapSummaryV1:
        """Return a compact frontier-oriented view without raw ray samples."""

        if frontier_limit < 0:
            raise ValueError("frontier_limit cannot be negative")
        observed = set(self.cells)
        traversable = {
            cell
            for cell, state in self.cells.items()
            if state.free_count > 0 or state.visited_count > 0
        }
        frontier = [
            cell
            for cell in traversable
            if any(
                neighbor not in observed
                for neighbor in (
                    (cell[0] - 1, cell[1]),
                    (cell[0] + 1, cell[1]),
                    (cell[0], cell[1] - 1),
                    (cell[0], cell[1] + 1),
                )
            )
        ]
        current = self.last_visited_cell
        frontier.sort(
            key=lambda cell: (
                abs(cell[0] - current[0]) + abs(cell[1] - current[1])
                if current is not None
                else 0,
                cell[0],
                cell[1],
            )
        )
        block_counts: Counter[str] = Counter()
        for state in self.cells.values():
            block_counts.update(state.block_counts)
        if observed:
            xs = [cell[0] for cell in observed]
            zs = [cell[1] for cell in observed]
            bounds = (min(xs), min(zs), max(xs), max(zs))
        else:
            bounds = None
        return SpatialMapSummaryV1(
            map_id=self.map_id,
            actor_id=self.actor_id,
            server_tick=self.last_tick or 0,
            resolution=self.resolution,
            observed_cell_count=len(observed),
            visited_cell_count=sum(
                state.visited_count > 0 for state in self.cells.values()
            ),
            free_evidence_cell_count=sum(
                state.free_count > 0 for state in self.cells.values()
            ),
            semantic_cell_count=sum(
                state.hit_count > 0 for state in self.cells.values()
            ),
            bounds_xz=bounds,
            current_cell=current,
            frontier_cells=tuple(frontier[:frontier_limit]),
            block_counts=dict(block_counts),
        )

    def multiscale_levels(self) -> tuple[dict[str, object], ...]:
        """Return three views of the same sparse map at fixed metric scales.

        Aggregation is deterministic and world aligned.  These are projections,
        not independent maps, so changing scale can never create conflicting
        evidence or a second source of truth.
        """

        levels: list[dict[str, object]] = []
        for key, label, bin_size in SPATIAL_PYRAMID_LEVELS:
            aggregated: dict[tuple[int, int], SpatialCellState] = {}
            for (cell_x, cell_z), state in self.cells.items():
                coarse = (math.floor(cell_x / bin_size), math.floor(cell_z / bin_size))
                target = aggregated.setdefault(coarse, SpatialCellState())
                target.visited_count += state.visited_count
                target.free_count += state.free_count
                target.hit_count += state.hit_count
                target.block_counts.update(state.block_counts)
                target.block_y_counts.update(state.block_y_counts)
                stack_height, head_clearance = _vertical_topology(state)
                target.topology_stack_height = max(
                    target.topology_stack_height or 0, stack_height
                )
                target.topology_head_clearance = min(
                    4.0
                    if target.topology_head_clearance is None
                    else target.topology_head_clearance,
                    head_clearance,
                )
                if state.surface_y_min is not None:
                    target.surface_y_min = (
                        state.surface_y_min
                        if target.surface_y_min is None
                        else min(target.surface_y_min, state.surface_y_min)
                    )
                if state.surface_y_max is not None:
                    target.surface_y_max = (
                        state.surface_y_max
                        if target.surface_y_max is None
                        else max(target.surface_y_max, state.surface_y_max)
                    )
                target.last_seen_tick = max(target.last_seen_tick, state.last_seen_tick)
            current = (
                (
                    math.floor(self.last_visited_cell[0] / bin_size),
                    math.floor(self.last_visited_cell[1] / bin_size),
                )
                if self.last_visited_cell is not None
                else None
            )
            cells = tuple(
                {
                    "cell_x": cell_x,
                    "cell_z": cell_z,
                    "visited_count": state.visited_count,
                    "free_count": state.free_count,
                    "hit_count": state.hit_count,
                    "block_counts": dict(state.block_counts),
                    "solid_y_levels": sorted(state.block_y_counts),
                    "solid_stack_height": _vertical_topology(state)[0],
                    "head_clearance": _vertical_topology(state)[1],
                    "surface_y_min": state.surface_y_min,
                    "surface_y_max": state.surface_y_max,
                    "last_seen_tick": state.last_seen_tick,
                }
                for (cell_x, cell_z), state in sorted(aggregated.items())
            )
            levels.append(
                {
                    "key": key,
                    "label": label,
                    "bin_size_blocks": bin_size,
                    "coverage_blocks": SPATIAL_PYRAMID_VIEW_SIZE * bin_size,
                    "view_size": SPATIAL_PYRAMID_VIEW_SIZE,
                    "current_cell": current,
                    "cells": cells,
                }
            )
        return tuple(levels)


class HttpSpatialStreamClient:
    """Fetch raycasts paired to an immutable frame and maintain its Gym map."""

    def __init__(
        self,
        url: str = "http://127.0.0.1:3008/v1/spatial/latest",
        *,
        actor_id: str = "actor:bot",
        resolution: float = 1.0,
        timeout: float = 2.0,
    ) -> None:
        self.url = url
        self.actor_id = actor_id
        self.resolution = resolution
        self.timeout = timeout
        self._map: SpatialLatticeMap | None = None
        self._voxel_map: SparseVoxelWorldMemory | None = None
        self._latest_frame: RaycastFrameV1 | None = None
        self._lock = threading.RLock()

    def start_episode(self, episode_id: str) -> None:
        if not episode_id:
            raise ValueError("episode_id cannot be empty")
        with self._lock:
            self._map = SpatialLatticeMap(
                f"map:{self.actor_id}:{episode_id}",
                self.actor_id,
                resolution=self.resolution,
            )
            self._voxel_map = SparseVoxelWorldMemory(
                f"voxels:{self.actor_id}:{episode_id}",
                self.actor_id,
                resolution=self.resolution,
            )
            self._latest_frame = None

    def latest(
        self,
        *,
        server_tick: int,
        camera_frame_id: str,
        camera_stream_id: str,
        visited_position: Vector3 | None = None,
    ) -> SpatialMapUpdateV1:
        with self._lock:
            spatial_map = self._map
            voxel_map = self._voxel_map
        if spatial_map is None or voxel_map is None:
            raise SpatialStreamError("start_episode() must precede spatial capture")
        separator = "&" if "?" in self.url else "?"
        query = urllib.parse.urlencode(
            {
                "server_tick": server_tick,
                "camera_frame_id": camera_frame_id,
                "camera_stream_id": camera_stream_id,
            }
        )
        request = urllib.request.Request(f"{self.url}{separator}{query}", method="GET")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read())
        except (urllib.error.URLError, OSError, TimeoutError) as error:
            raise SpatialStreamError(f"spatial stream unavailable: {error}") from error
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise SpatialStreamError("spatial stream returned invalid JSON") from error
        try:
            frame = RaycastFrameV1.model_validate(payload)
        except ValidationError as error:
            raise SpatialStreamError("spatial stream contract mismatch") from error
        if frame.server_tick != server_tick:
            raise SpatialStreamError("spatial frame server tick mismatch")
        if frame.camera_frame_id != camera_frame_id:
            raise SpatialStreamError("spatial frame does not match visual frame")
        if frame.camera_stream_id != camera_stream_id:
            raise SpatialStreamError("spatial frame does not match visual stream")
        # X/Z of the egocentric camera and body are identical.  Y is not: the
        # renderer origin is the eye, approximately 1.62 blocks above the
        # player's feet.  Treating eye Y as feet Y makes the floor look like a
        # low ceiling in the accumulated topology map.
        position = visited_position or (
            frame.pose.origin[0],
            frame.pose.origin[1] - 1.62,
            frame.pose.origin[2],
        )
        with self._lock:
            if self._map is not spatial_map:
                raise SpatialStreamError("spatial episode changed during capture")
            update = spatial_map.integrate(frame, visited_position=position)
            voxel_map.integrate(frame, visited_position=position)
            self._latest_frame = frame
            return update

    def view_state(self) -> dict[str, object] | None:
        """Expose the latest camera basis used to build the spatial memory.

        The Observer can render a frustum from this defensive projection
        without inventing an orientation or reading private renderer state.
        """

        with self._lock:
            frame = self._latest_frame
            if frame is None:
                return None
            return {
                "server_tick": frame.server_tick,
                "camera_frame_id": frame.camera_frame_id,
                "sampling_columns": frame.sampling_columns,
                "sampling_rows": frame.sampling_rows,
                "calibration": frame.calibration.model_dump(mode="json"),
                "pose": frame.pose.model_dump(mode="json"),
            }

    def raycast_frame(self) -> RaycastFrameV1 | None:
        """Return a defensive copy of the most recently validated ray frame."""

        with self._lock:
            return (
                None
                if self._latest_frame is None
                else self._latest_frame.model_copy(deep=True)
            )

    def summary(self, *, frontier_limit: int = 12) -> SpatialMapSummaryV1:
        with self._lock:
            if self._map is None:
                raise SpatialStreamError("start_episode() must precede spatial summary")
            return self._map.summary(frontier_limit=frontier_limit)

    def snapshot_cells(self) -> tuple[dict[str, object], ...]:
        """Return a stable, JSON-ready visualization snapshot.

        The map remains owned by the Gym collector. Consumers such as an
        Observer UI receive defensive values and cannot mutate training state.
        """

        with self._lock:
            if self._map is None:
                raise SpatialStreamError("start_episode() must precede spatial snapshot")
            snapshot = self._map.snapshot()
        return tuple(
            {
                "cell_x": cell_x,
                "cell_z": cell_z,
                "visited_count": state.visited_count,
                "free_count": state.free_count,
                "hit_count": state.hit_count,
                "block_counts": dict(state.block_counts),
                "solid_y_levels": sorted(state.block_y_counts),
                "solid_stack_height": _vertical_topology(state)[0],
                "head_clearance": _vertical_topology(state)[1],
                "surface_y_min": state.surface_y_min,
                "surface_y_max": state.surface_y_max,
                "last_seen_tick": state.last_seen_tick,
            }
            for (cell_x, cell_z), state in sorted(snapshot.items())
        )

    def multiscale_levels(self) -> tuple[dict[str, object], ...]:
        """Expose Gym-owned projections for Observer and dataset consumers."""

        with self._lock:
            if self._map is None:
                raise SpatialStreamError("start_episode() must precede spatial snapshot")
            return self._map.multiscale_levels()

    def voxel_summary(self) -> dict[str, object]:
        """Return Gym-owned voxel geometry metrics for the Observer."""

        with self._lock:
            if self._voxel_map is None:
                raise SpatialStreamError("start_episode() must precede voxel summary")
            return self._voxel_map.summary()

    def snapshot_voxels(
        self,
        *,
        horizontal_radius: int = 16,
        vertical_radius: int = 8,
    ) -> tuple[dict[str, object], ...]:
        """Return a bounded local voxel crop without copying world memory."""

        with self._lock:
            if self._voxel_map is None:
                raise SpatialStreamError("start_episode() must precede voxel snapshot")
            return self._voxel_map.snapshot_voxels(
                horizontal_radius=horizontal_radius,
                vertical_radius=vertical_radius,
            )
