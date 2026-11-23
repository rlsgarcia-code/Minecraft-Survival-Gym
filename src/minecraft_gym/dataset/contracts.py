"""Versioned public contracts for Minecraft research episodes.

These models are deliberately independent from the Observer database.  A
dataset can therefore be audited, replayed, or exported using only the files
stored alongside an episode.
"""

from __future__ import annotations

from datetime import UTC, datetime
import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


DataTier = Literal[
    "exact_visual",
    "rendered_visual",
    "exact_structured",
    "semantic_weak",
    "privileged_tool",
]
Consumer = Literal[
    "behavior_cloning",
    "offline_rl",
    "dreamerv3",
    "semantic_imitation",
    "reward_model",
    "evaluation",
    "tool_policy",
]


def utc_now() -> datetime:
    return datetime.now(UTC)


_PRIVACY_PROTECTED_KEYS = {
    "chat",
    "message",
    "player_name",
    "username",
    "display_name",
    "player_list",
    "players",
}


def privacy_protected_keys(value: Any) -> set[str]:
    """Find identity/chat fields recursively before data reaches disk."""

    if isinstance(value, dict):
        found = {str(key).lower() for key in value}.intersection(
            _PRIVACY_PROTECTED_KEYS
        )
        for child in value.values():
            found.update(privacy_protected_keys(child))
        return found
    if isinstance(value, (list, tuple)):
        found: set[str] = set()
        for child in value:
            found.update(privacy_protected_keys(child))
        return found
    return set()


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=True)


class ConsentRecord(ContractModel):
    actor_id: str
    granted: bool
    scope: Literal["structured", "visual", "visual_and_input"]
    granted_at: datetime = Field(default_factory=utc_now)
    policy_version: str = "privacy-default/v1"


class ActorDescriptor(ContractModel):
    actor_id: str
    role: Literal["bot", "expert_player", "observer", "environment"]
    platform: Literal["java", "android", "bedrock", "server", "unknown"] = "unknown"
    identity_scope: Literal["public_bot", "dataset_pseudonym", "environment"]

    @model_validator(mode="after")
    def identity_is_not_a_player_name(self) -> "ActorDescriptor":
        if self.identity_scope == "dataset_pseudonym" and not self.actor_id.startswith(
            ("player:", "actor:pseudonym:")
        ):
            raise ValueError("dataset actors must use a pseudonymous actor_id")
        if self.identity_scope == "public_bot" and self.role != "bot":
            raise ValueError("only bots may use a public identity")
        return self


class ArtifactDescriptor(ContractModel):
    path: str
    media_type: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)
    role: Literal[
        "steps",
        "events",
        "video",
        "depth",
        "frame_index",
        "manifest",
        "quality_report",
        "checkpoint",
        "other",
    ]


class VisualStreamDescriptorV1(ContractModel):
    """Declare one stable camera domain before frames enter an episode."""

    interface: Literal["minecraft-visual-stream/v1"] = "minecraft-visual-stream/v1"
    stream_id: str
    actor_id: str
    viewpoint: Literal["egocentric", "observer", "third_person"]
    source: Literal["fabric_native", "prismarine_renderer", "bedrock_capture"]
    visual_domain: Literal["native_java", "rendered_prismarine", "native_bedrock"]
    width: int = Field(gt=0, le=7680)
    height: int = Field(gt=0, le=4320)
    nominal_fps: float = Field(gt=0, le=240)
    policy_eligible: bool
    input_aligned: bool = False

    @model_validator(mode="after")
    def source_matches_domain(self) -> "VisualStreamDescriptorV1":
        expected = {
            "fabric_native": "native_java",
            "prismarine_renderer": "rendered_prismarine",
            "bedrock_capture": "native_bedrock",
        }
        if expected[self.source] != self.visual_domain:
            raise ValueError("visual stream source does not match its visual domain")
        if self.source == "prismarine_renderer" and self.input_aligned:
            raise ValueError("Prismarine frames cannot claim native input alignment")
        return self


class ViewpointFrameRefV1(ContractModel):
    """One immutable frame aligned to the authoritative episode clock."""

    interface: Literal["minecraft-viewpoint-frame/v1"] = (
        "minecraft-viewpoint-frame/v1"
    )
    stream_id: str
    actor_id: str
    viewpoint: Literal["egocentric", "observer", "third_person"]
    source: Literal["fabric_native", "prismarine_renderer", "bedrock_capture"]
    visual_domain: Literal["native_java", "rendered_prismarine", "native_bedrock"]
    frame_id: str
    rgb_ref: str
    server_tick: int = Field(ge=0)
    monotonic_ns: int = Field(ge=0)
    capture_sequence: int = Field(ge=0)
    width: int = Field(gt=0, le=7680)
    height: int = Field(gt=0, le=4320)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    alignment_ticks: int = Field(ge=0)
    alignment_method: Literal[
        "exact_client_tick",
        "nearest_monotonic",
        "server_reconstructed",
    ]
    policy_eligible: bool

    @model_validator(mode="after")
    def source_matches_domain(self) -> "ViewpointFrameRefV1":
        expected = {
            "fabric_native": "native_java",
            "prismarine_renderer": "rendered_prismarine",
            "bedrock_capture": "native_bedrock",
        }
        if expected[self.source] != self.visual_domain:
            raise ValueError("viewpoint source does not match its visual domain")
        if self.source == "fabric_native" and self.alignment_method != "exact_client_tick":
            raise ValueError("Fabric native frames require the exact client tick")
        return self


class CameraCalibrationV1(ContractModel):
    """Pinhole camera parameters used to turn a rendered pixel into a ray."""

    interface: Literal["minecraft-camera-calibration/v1"] = (
        "minecraft-camera-calibration/v1"
    )
    width: int = Field(gt=0, le=7680)
    height: int = Field(gt=0, le=4320)
    vertical_fov_degrees: float = Field(gt=0, lt=180)
    near_plane: float = Field(default=0.1, gt=0)
    far_plane: float = Field(default=1000.0, gt=0)
    projection: Literal["perspective"] = "perspective"
    renderer: Literal["prismarine_viewer"] = "prismarine_viewer"
    pixel_center_convention: Literal["half_pixel"] = "half_pixel"

    @model_validator(mode="after")
    def planes_are_ordered(self) -> "CameraCalibrationV1":
        if self.far_plane <= self.near_plane:
            raise ValueError("camera far_plane must be greater than near_plane")
        return self


class CameraPoseV1(ContractModel):
    """Camera origin and world-space orthonormal basis for one server tick.

    The basis is authoritative for projection.  Yaw and pitch are retained for
    audit/debugging only because renderer and protocol conventions can differ.
    """

    interface: Literal["minecraft-camera-pose/v1"] = "minecraft-camera-pose/v1"
    actor_id: str
    server_tick: int = Field(ge=0)
    monotonic_ns: int = Field(ge=0)
    origin: tuple[float, float, float]
    forward: tuple[float, float, float]
    right: tuple[float, float, float]
    up: tuple[float, float, float]
    yaw_degrees: float | None = None
    pitch_degrees: float | None = None
    source: Literal["prismarine_camera", "fabric_camera"]

    @model_validator(mode="after")
    def basis_is_finite_orthonormal(self) -> "CameraPoseV1":
        vectors = (self.origin, self.forward, self.right, self.up)
        if any(not math.isfinite(value) for vector in vectors for value in vector):
            raise ValueError("camera pose vectors must contain finite values")
        basis = (self.forward, self.right, self.up)
        for vector in basis:
            norm = math.sqrt(sum(value * value for value in vector))
            if not math.isclose(norm, 1.0, abs_tol=1e-3):
                raise ValueError("camera basis vectors must be unit length")
        for left, right in (
            (self.forward, self.right),
            (self.forward, self.up),
            (self.right, self.up),
        ):
            dot = sum(a * b for a, b in zip(left, right, strict=True))
            if not math.isclose(dot, 0.0, abs_tol=1e-3):
                raise ValueError("camera basis vectors must be orthogonal")
        return self


class RaycastSampleV1(ContractModel):
    """One pixel ray and the structured world result returned by Prismarine."""

    interface: Literal["minecraft-raycast-sample/v1"] = (
        "minecraft-raycast-sample/v1"
    )
    pixel_x: int = Field(ge=0)
    pixel_y: int = Field(ge=0)
    origin: tuple[float, float, float]
    direction: tuple[float, float, float]
    max_distance: float = Field(gt=0)
    traversed_distance: float = Field(ge=0)
    termination: Literal["block_hit", "max_range", "unloaded_chunk"]
    contact_point: tuple[float, float, float] | None = None
    block_position: tuple[int, int, int] | None = None
    block_name: str | None = None
    block_state_id: int | None = Field(default=None, ge=0)
    face: int | str | None = None
    intersection_source: Literal["prismarine_world_shapes"] = (
        "prismarine_world_shapes"
    )
    confidence: float = Field(default=1.0, ge=0, le=1)

    @model_validator(mode="after")
    def result_matches_termination(self) -> "RaycastSampleV1":
        if any(
            not math.isfinite(value)
            for vector in (self.origin, self.direction)
            for value in vector
        ):
            raise ValueError("ray vectors must contain finite values")
        norm = math.sqrt(sum(value * value for value in self.direction))
        if not math.isclose(norm, 1.0, abs_tol=1e-3):
            raise ValueError("ray direction must be unit length")
        if self.traversed_distance > self.max_distance + 1e-6:
            raise ValueError("ray traversed_distance exceeds max_distance")
        block_fields = (
            self.contact_point,
            self.block_position,
            self.block_name,
            self.block_state_id,
        )
        if self.termination == "block_hit" and any(
            value is None for value in block_fields
        ):
            raise ValueError("block_hit rays require contact and block identity")
        if self.termination != "block_hit" and any(
            value is not None for value in block_fields
        ):
            raise ValueError("non-hit rays cannot claim block identity")
        return self


class RaycastFrameV1(ContractModel):
    """Sparse pixel-to-world projection aligned with one visual frame."""

    interface: Literal["minecraft-raycast-frame/v1"] = "minecraft-raycast-frame/v1"
    sensor_id: str
    actor_id: str
    server_tick: int = Field(ge=0)
    monotonic_ns: int = Field(ge=0)
    camera_stream_id: str
    camera_frame_id: str
    calibration: CameraCalibrationV1
    pose: CameraPoseV1
    sampling_pattern: Literal["uniform_grid"] = "uniform_grid"
    sampling_columns: int = Field(gt=0)
    sampling_rows: int = Field(gt=0)
    samples: tuple[RaycastSampleV1, ...]

    @model_validator(mode="after")
    def frame_is_internally_aligned(self) -> "RaycastFrameV1":
        if self.pose.actor_id != self.actor_id:
            raise ValueError("raycast pose actor does not match frame actor")
        if self.pose.server_tick != self.server_tick:
            raise ValueError("raycast pose tick does not match frame tick")
        pixels: set[tuple[int, int]] = set()
        if len(self.samples) != self.sampling_columns * self.sampling_rows:
            raise ValueError("raycast sample count does not match sampling grid")
        for sample in self.samples:
            pixel = (sample.pixel_x, sample.pixel_y)
            if pixel in pixels:
                raise ValueError("raycast frame contains duplicate sampled pixels")
            pixels.add(pixel)
            if sample.pixel_x >= self.calibration.width:
                raise ValueError("raycast pixel_x is outside camera width")
            if sample.pixel_y >= self.calibration.height:
                raise ValueError("raycast pixel_y is outside camera height")
            if any(
                not math.isclose(a, b, abs_tol=1e-6)
                for a, b in zip(sample.origin, self.pose.origin, strict=True)
            ):
                raise ValueError("ray origin does not match camera pose")
        return self


class DepthMapV1(ContractModel):
    """Metric depth image derived from the ray grid paired to one RGB frame.

    Values are row-major and remain in Minecraft blocks. ``status_codes``
    distinguishes a measured surface (1), a ray censored at max range (2),
    and an unknown ray that reached an unloaded chunk (0).  A learner can use
    depth plus a validity plane without treating unknown space as nearby
    geometry.
    """

    interface: Literal["minecraft-depth-map/v1"] = "minecraft-depth-map/v1"
    sensor_id: str
    actor_id: str
    server_tick: int = Field(ge=0)
    monotonic_ns: int = Field(ge=0)
    camera_stream_id: str
    camera_frame_id: str
    source_width: int = Field(gt=0, le=7680)
    source_height: int = Field(gt=0, le=4320)
    width: int = Field(gt=0, le=7680)
    height: int = Field(gt=0, le=4320)
    unit: Literal["minecraft_block"] = "minecraft_block"
    sampling_pattern: Literal["uniform_grid"] = "uniform_grid"
    max_distance: float = Field(gt=0)
    depth_blocks: tuple[float, ...]
    status_codes: tuple[Literal[0, 1, 2], ...]
    confidence: tuple[float, ...]
    policy_eligible: bool = True

    @model_validator(mode="after")
    def raster_is_aligned_and_finite(self) -> "DepthMapV1":
        expected = self.width * self.height
        if not (
            len(self.depth_blocks)
            == len(self.status_codes)
            == len(self.confidence)
            == expected
        ):
            raise ValueError("depth arrays do not match raster dimensions")
        if any(
            not math.isfinite(value) or value < 0 or value > self.max_distance + 1e-6
            for value in self.depth_blocks
        ):
            raise ValueError("depth values must be finite and inside max_distance")
        if any(not 0 <= value <= 1 for value in self.confidence):
            raise ValueError("depth confidence must lie in [0, 1]")
        return self


class DenseDepthFrameRefV1(ContractModel):
    """Episode-relative reference to one RGB-aligned dense metric depth frame.

    Dense rasters are intentionally stored as binary artifacts instead of
    being repeated inside every Parquet/JSON step.  One pixel occupies three
    bytes before zlib: uint16 little-endian centiblocks followed by a status
    byte (1 measured surface, 2 censored at max range, 0 invalid).
    """

    interface: Literal["minecraft-dense-depth-frame/v1"] = (
        "minecraft-dense-depth-frame/v1"
    )
    sensor_id: str
    actor_id: str
    server_tick: int = Field(ge=0)
    monotonic_ns: int = Field(ge=0)
    camera_stream_id: str
    camera_frame_id: str
    depth_ref: str
    width: int = Field(gt=0, le=7680)
    height: int = Field(gt=0, le=4320)
    unit: Literal["minecraft_block"] = "minecraft_block"
    encoding: Literal[
        "uint16_le_centiblocks+uint8_status_interleaved+zlib"
    ] = "uint16_le_centiblocks+uint8_status_interleaved+zlib"
    max_distance: float = Field(gt=0, le=655.35)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    uncompressed_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    alignment_ticks: int = Field(ge=0)
    policy_eligible: bool = True


class SpatialMapCellDeltaV1(ContractModel):
    """Additive evidence for one X/Z lattice cell."""

    cell_x: int
    cell_z: int
    resolution: float = Field(gt=0)
    visited_count_delta: int = Field(default=0, ge=0)
    free_count_delta: int = Field(default=0, ge=0)
    hit_count_delta: int = Field(default=0, ge=0)
    block_counts_delta: dict[str, int] = Field(default_factory=dict)
    surface_y_min: float | None = None
    surface_y_max: float | None = None
    last_seen_tick: int = Field(ge=0)

    @model_validator(mode="after")
    def evidence_is_consistent(self) -> "SpatialMapCellDeltaV1":
        if not any(
            (self.visited_count_delta, self.free_count_delta, self.hit_count_delta)
        ):
            raise ValueError("map delta must contain evidence")
        if any(value <= 0 for value in self.block_counts_delta.values()):
            raise ValueError("block count deltas must be positive")
        if sum(self.block_counts_delta.values()) != self.hit_count_delta:
            raise ValueError("block counts must equal hit_count_delta")
        if (self.surface_y_min is None) != (self.surface_y_max is None):
            raise ValueError("surface height bounds must be both present or absent")
        if (
            self.surface_y_min is not None
            and self.surface_y_max is not None
            and self.surface_y_max < self.surface_y_min
        ):
            raise ValueError("surface_y_max must not precede surface_y_min")
        return self


class SpatialMapUpdateV1(ContractModel):
    """Raycast evidence and the exact additive map update derived from it."""

    interface: Literal["minecraft-spatial-map-update/v1"] = (
        "minecraft-spatial-map-update/v1"
    )
    map_id: str
    channel_name: str = "spatial_raycast_map"
    actor_id: str
    server_tick: int = Field(ge=0)
    resolution: float = Field(gt=0)
    scope: Literal["privileged_derived", "deployable_sensor"] = (
        "privileged_derived"
    )
    raycast: RaycastFrameV1
    deltas: tuple[SpatialMapCellDeltaV1, ...]

    @model_validator(mode="after")
    def update_is_internally_aligned(self) -> "SpatialMapUpdateV1":
        if self.raycast.actor_id != self.actor_id:
            raise ValueError("spatial update actor does not match raycast actor")
        if self.raycast.server_tick != self.server_tick:
            raise ValueError("spatial update tick does not match raycast tick")
        cells: set[tuple[int, int]] = set()
        for delta in self.deltas:
            cell = (delta.cell_x, delta.cell_z)
            if cell in cells:
                raise ValueError("spatial update contains duplicate cell deltas")
            cells.add(cell)
            if not math.isclose(delta.resolution, self.resolution, abs_tol=1e-9):
                raise ValueError("spatial delta resolution does not match map")
            if delta.last_seen_tick != self.server_tick:
                raise ValueError("spatial delta tick does not match update")
        return self


class SpatialMapSummaryV1(ContractModel):
    """Compact, reproducible projection used by planners and the Observer."""

    interface: Literal["minecraft-spatial-map-summary/v1"] = (
        "minecraft-spatial-map-summary/v1"
    )
    map_id: str
    actor_id: str
    server_tick: int = Field(ge=0)
    resolution: float = Field(gt=0)
    scope: Literal["privileged_derived"] = "privileged_derived"
    observed_cell_count: int = Field(ge=0)
    visited_cell_count: int = Field(ge=0)
    free_evidence_cell_count: int = Field(ge=0)
    semantic_cell_count: int = Field(ge=0)
    bounds_xz: tuple[int, int, int, int] | None = None
    current_cell: tuple[int, int] | None = None
    frontier_cells: tuple[tuple[int, int], ...] = ()
    block_counts: dict[str, int] = Field(default_factory=dict)

    @model_validator(mode="after")
    def summary_is_consistent(self) -> "SpatialMapSummaryV1":
        if any(value <= 0 for value in self.block_counts.values()):
            raise ValueError("summary block counts must be positive")
        if self.observed_cell_count == 0 and self.bounds_xz is not None:
            raise ValueError("empty maps cannot declare bounds")
        if self.bounds_xz is not None:
            min_x, min_z, max_x, max_z = self.bounds_xz
            if max_x < min_x or max_z < min_z:
                raise ValueError("spatial summary bounds are reversed")
        if self.visited_cell_count > self.observed_cell_count:
            raise ValueError("visited cells cannot exceed observed cells")
        return self


class SphericalValueCellV1(ContractModel):
    """One world-aligned angular cell used by the exploration planner.

    The value is a reproducible coverage heuristic, not a learned Q value.
    Keeping that distinction explicit lets later RL code replace the estimator
    without changing the accumulated angular evidence.
    """

    azimuth_bin: int = Field(ge=0)
    elevation_bin: int = Field(ge=0)
    azimuth_center_degrees: float = Field(ge=-180, lt=180)
    elevation_center_degrees: float = Field(ge=-90, le=90)
    observation_count: int = Field(ge=0)
    hit_count: int = Field(ge=0)
    block_counts: dict[str, int] = Field(default_factory=dict)
    mean_distance: float | None = Field(default=None, ge=0)
    last_seen_tick: int | None = Field(default=None, ge=0)
    novelty_value: float = Field(ge=0)

    @model_validator(mode="after")
    def evidence_is_consistent(self) -> "SphericalValueCellV1":
        if self.hit_count > self.observation_count:
            raise ValueError("spherical hits cannot exceed observations")
        if sum(self.block_counts.values()) != self.hit_count:
            raise ValueError("spherical block counts must equal hit_count")
        if (self.observation_count == 0) != (self.last_seen_tick is None):
            raise ValueError("unseen spherical cells cannot have a last_seen_tick")
        if self.observation_count == 0 and self.mean_distance is not None:
            raise ValueError("unseen spherical cells cannot have a mean distance")
        return self


class SphericalValueLatticeV1(ContractModel):
    """World-aligned equirectangular evidence and exploration values."""

    interface: Literal["minecraft-spherical-value-lattice/v1"] = (
        "minecraft-spherical-value-lattice/v1"
    )
    lattice_id: str
    actor_id: str
    server_tick: int = Field(ge=0)
    azimuth_bins: int = Field(gt=0)
    elevation_bins: int = Field(gt=0)
    estimator: Literal["coverage-heuristic/v1"] = "coverage-heuristic/v1"
    scope: Literal["privileged_derived"] = "privileged_derived"
    current_yaw_degrees: float
    current_pitch_degrees: float
    observed_bin_count: int = Field(ge=0)
    total_bin_count: int = Field(gt=0)
    coverage_ratio: float = Field(ge=0, le=1)
    cells: tuple[SphericalValueCellV1, ...]

    @model_validator(mode="after")
    def lattice_is_complete_and_consistent(self) -> "SphericalValueLatticeV1":
        expected = self.azimuth_bins * self.elevation_bins
        if self.total_bin_count != expected or len(self.cells) != expected:
            raise ValueError("spherical lattice must contain every angular bin")
        keys = {(cell.azimuth_bin, cell.elevation_bin) for cell in self.cells}
        if len(keys) != expected:
            raise ValueError("spherical lattice contains duplicate angular bins")
        if any(
            cell.azimuth_bin >= self.azimuth_bins
            or cell.elevation_bin >= self.elevation_bins
            for cell in self.cells
        ):
            raise ValueError("spherical cell lies outside lattice bounds")
        observed = sum(cell.observation_count > 0 for cell in self.cells)
        if observed != self.observed_bin_count:
            raise ValueError("spherical observed_bin_count is inconsistent")
        if not math.isclose(self.coverage_ratio, observed / expected, abs_tol=1e-9):
            raise ValueError("spherical coverage_ratio is inconsistent")
        return self


class ObservationRefV2(ContractModel):
    server_tick: int = Field(ge=0)
    monotonic_ns: int = Field(ge=0)
    frame_id: str | None = None
    rgb_ref: str | None = None
    primary_stream_id: str | None = None
    viewpoints: tuple[ViewpointFrameRefV1, ...] = ()
    structured: dict[str, Any] = Field(default_factory=dict)
    policy_channels: tuple[str, ...] = ()
    privileged: dict[str, Any] = Field(default_factory=dict)
    spatial_updates: tuple[SpatialMapUpdateV1, ...] = ()
    depth_map: DepthMapV1 | None = None
    dense_depth: DenseDepthFrameRefV1 | None = None

    @model_validator(mode="after")
    def trainable_channels_exclude_identity_and_chat(self) -> "ObservationRefV2":
        leaked = privacy_protected_keys(
            {"structured": self.structured, "privileged": self.privileged}
        )
        if leaked:
            raise ValueError(
                "observation contains privacy-protected fields: "
                + ", ".join(sorted(leaked))
            )
        stream_ids = [item.stream_id for item in self.viewpoints]
        if len(stream_ids) != len(set(stream_ids)):
            raise ValueError("observation viewpoint stream_ids must be unique")
        if self.primary_stream_id is not None and self.primary_stream_id not in stream_ids:
            raise ValueError("primary_stream_id does not reference an observation viewpoint")
        if self.rgb_ref is not None and self.viewpoints and self.primary_stream_id is None:
            raise ValueError("multi-view RGB observations require primary_stream_id")
        map_ids: set[str] = set()
        frame_ids = {item.frame_id for item in self.viewpoints}
        for update in self.spatial_updates:
            if update.map_id in map_ids:
                raise ValueError("observation spatial map_ids must be unique")
            map_ids.add(update.map_id)
            if update.server_tick != self.server_tick:
                raise ValueError("spatial update tick does not match observation")
            if frame_ids and update.raycast.camera_frame_id not in frame_ids:
                raise ValueError("spatial update references an unknown camera frame")
            if (
                update.scope == "privileged_derived"
                and update.channel_name in self.policy_channels
            ):
                raise ValueError(
                    "privileged spatial maps cannot be policy observation channels"
                )
        if self.depth_map is not None:
            if self.depth_map.server_tick != self.server_tick:
                raise ValueError("depth map tick does not match observation")
            if self.frame_id is not None and self.depth_map.camera_frame_id != self.frame_id:
                raise ValueError("depth map does not match observation frame")
            if frame_ids and self.depth_map.camera_frame_id not in frame_ids:
                raise ValueError("depth map references an unknown camera frame")
        if self.dense_depth is not None:
            if self.dense_depth.server_tick != self.server_tick:
                raise ValueError("dense depth tick does not match observation")
            if self.frame_id is not None and self.dense_depth.camera_frame_id != self.frame_id:
                raise ValueError("dense depth does not match observation frame")
            if frame_ids and self.dense_depth.camera_frame_id not in frame_ids:
                raise ValueError("dense depth references an unknown camera frame")
        if self.depth_map is not None and self.dense_depth is not None:
            raise ValueError("observation cannot expose sparse and dense depth simultaneously")
        if (
            "depth" in self.policy_channels
            and self.depth_map is None
            and self.dense_depth is None
        ):
            raise ValueError("depth policy channel requires an aligned depth map")
        return self

    def primary_viewpoint(self) -> ViewpointFrameRefV1 | None:
        if self.primary_stream_id is None:
            return None
        return next(
            (item for item in self.viewpoints if item.stream_id == self.primary_stream_id),
            None,
        )


class ActionLabelV2(ContractModel):
    values: dict[str, Any]
    source: Literal[
        "java_native_input",
        "gym_agent",
        "gym_agent_macro",
        "server_reconstructed",
        "privileged_command",
    ]
    known_fields: tuple[str, ...]
    unknown_fields: tuple[str, ...] = ()
    confidence: float = Field(ge=0, le=1)
    policy_projection: tuple[int, ...] | None = None
    policy_action_contract: str | None = None
    policy_action: dict[str, Any] | None = None
    unsupported_controls: tuple[str, ...] = ()

    @model_validator(mode="after")
    def fields_are_disjoint(self) -> "ActionLabelV2":
        overlap = set(self.known_fields).intersection(self.unknown_fields)
        if overlap:
            raise ValueError(f"known_fields and unknown_fields overlap: {sorted(overlap)}")
        missing = set(self.known_fields).difference(self.values)
        if missing:
            raise ValueError(f"known action fields have no value: {sorted(missing)}")
        if (self.policy_action_contract is None) != (self.policy_action is None):
            raise ValueError(
                "policy_action and policy_action_contract must be recorded together"
            )
        if self.policy_action_contract == "minecraft-control-hybrid-camera-body/v1":
            required = {"delta_longitude", "delta_latitude", "body_mode"}
            missing_policy = required.difference(self.policy_action or {})
            if missing_policy:
                raise ValueError(
                    "hybrid policy action is missing fields: "
                    f"{sorted(missing_policy)}"
                )
        if self.policy_action_contract == "minecraft-control-camera-keyboard/v2":
            required = {
                "delta_longitude",
                "delta_latitude",
                "strafe",
                "move",
                "jump",
                "sprint",
                "sneak",
                "attack",
            }
            missing_policy = required.difference(self.policy_action or {})
            if missing_policy:
                raise ValueError(
                    "camera-keyboard policy action is missing fields: "
                    f"{sorted(missing_policy)}"
                )
        if self.policy_action_contract == "minecraft-control-camera-keyboard/v3":
            required = {
                "delta_longitude",
                "delta_latitude",
                "strafe",
                "move",
                "jump",
                "sprint",
                "sneak",
                "attack",
                "use",
            }
            missing_policy = required.difference(self.policy_action or {})
            if missing_policy:
                raise ValueError(
                    "camera-keyboard v3 policy action is missing fields: "
                    f"{sorted(missing_policy)}"
                )
        return self


class StepRecordV2(ContractModel):
    interface: Literal["minecraft-step/v2"] = "minecraft-step/v2"
    episode_id: str
    step_id: int = Field(ge=0)
    actor_id: str
    data_tier: DataTier
    observation: ObservationRefV2
    action: ActionLabelV2
    next_observation: ObservationRefV2
    reward: float = 0.0
    reward_terms: dict[str, float] = Field(default_factory=dict)
    events: tuple[dict[str, Any], ...] = ()
    terminated: bool = False
    truncated: bool = False
    allowed_consumers: tuple[Consumer, ...] = ()
    provenance: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def enforce_training_boundaries(self) -> "StepRecordV2":
        if self.next_observation.server_tick <= self.observation.server_tick:
            raise ValueError("next observation tick must advance")
        consumers = set(self.allowed_consumers)
        if self.data_tier == "semantic_weak" and consumers.intersection(
            {"behavior_cloning", "offline_rl", "dreamerv3"}
        ):
            raise ValueError("semantic_weak samples cannot train low-level policies")
        if self.data_tier == "exact_structured" and "dreamerv3" in consumers:
            raise ValueError("DreamerV3 requires a valid visual observation")
        if self.data_tier == "exact_visual":
            if not self.observation.rgb_ref or not self.next_observation.rgb_ref:
                raise ValueError("exact_visual steps require RGB references on both observations")
            if self.action.source != "java_native_input":
                raise ValueError("exact_visual steps require native Java input labels")
            for observation in (self.observation, self.next_observation):
                primary = observation.primary_viewpoint()
                if primary is not None and (
                    primary.actor_id != self.actor_id
                    or primary.source != "fabric_native"
                    or primary.visual_domain != "native_java"
                ):
                    raise ValueError(
                        "exact_visual primary frames must be native views of the acting player"
                    )
        if self.data_tier == "rendered_visual":
            if not self.observation.rgb_ref or not self.next_observation.rgb_ref:
                raise ValueError(
                    "rendered_visual steps require RGB references on both observations"
                )
            if self.action.source not in {"gym_agent", "gym_agent_macro"}:
                raise ValueError("rendered_visual steps require exact Gym action labels")
            for observation in (self.observation, self.next_observation):
                primary = observation.primary_viewpoint()
                if primary is None:
                    raise ValueError("rendered_visual steps require a primary viewpoint")
                if (
                    primary.actor_id != self.actor_id
                    or primary.source != "prismarine_renderer"
                    or primary.visual_domain != "rendered_prismarine"
                ):
                    raise ValueError(
                        "rendered_visual primary frames must be rendered views of the acting bot"
                    )
        if consumers.intersection({"behavior_cloning", "offline_rl", "dreamerv3"}):
            for observation in (self.observation, self.next_observation):
                for frame in observation.viewpoints:
                    if frame.policy_eligible and frame.alignment_ticks > 1:
                        raise ValueError(
                            "policy-eligible viewpoints must align within one server tick"
                        )
        if self.data_tier == "privileged_tool" and consumers.difference({"tool_policy"}):
            raise ValueError("privileged commands may only train the tool policy")
        if self.action.unsupported_controls and consumers.intersection(
            {"behavior_cloning", "offline_rl", "dreamerv3"}
        ):
            raise ValueError("unsupported controls make a step ineligible for motor policies")
        leaked = privacy_protected_keys(
            {"events": self.events, "provenance": self.provenance}
        )
        if leaked:
            raise ValueError(
                "step metadata contains privacy-protected fields: "
                + ", ".join(sorted(leaked))
            )
        return self


class AuthoritativeTickV1(ContractModel):
    """One passive server tick with pseudonymous actor state only."""

    interface: Literal["minecraft-authoritative-tick/v1"] = (
        "minecraft-authoritative-tick/v1"
    )
    sequence: int = Field(ge=0)
    server_tick: int = Field(ge=0)
    monotonic_ns: int = Field(ge=0)
    actors: tuple[dict[str, Any], ...] = ()
    events: tuple[dict[str, Any], ...] = ()

    @model_validator(mode="after")
    def require_pseudonyms_and_private_metadata(self) -> "AuthoritativeTickV1":
        leaked = privacy_protected_keys({"actors": self.actors, "events": self.events})
        if leaked:
            raise ValueError(
                "authoritative tick contains privacy-protected fields: "
                + ", ".join(sorted(leaked))
            )
        for actor in self.actors:
            actor_id = str(actor.get("actor_id", ""))
            if not actor_id.startswith("actor:pseudonym:"):
                raise ValueError("server actors must use stable pseudonymous actor_id")
        for event in self.events:
            actor_id = event.get("actor_id")
            if actor_id is not None and not str(actor_id).startswith(
                "actor:pseudonym:"
            ):
                raise ValueError("server events must use pseudonymous actor_id")
        return self


class TaskManifestV1(ContractModel):
    interface: Literal["minecraft-task/v1"] = "minecraft-task/v1"
    task_id: str
    version: int = Field(ge=1)
    title: str
    family: Literal[
        "navigate",
        "follow",
        "collect",
        "place_pattern",
        "build",
    ]
    world_snapshot_id: str
    initial_state: dict[str, Any]
    goal: dict[str, Any]
    success_predicates: tuple[str, ...]
    failure_predicates: tuple[str, ...]
    timeout_steps: int = Field(gt=0)
    reward_terms: dict[str, float]
    policy_channels: tuple[str, ...]
    privileged_reward_channels: tuple[str, ...] = ()


class VisualStreamQualityV1(ContractModel):
    interface: Literal["minecraft-visual-stream-quality/v1"] = (
        "minecraft-visual-stream-quality/v1"
    )
    stream_id: str
    actor_id: str
    frame_count: int = Field(ge=0)
    dropped_frame_count: int = Field(ge=0)
    max_alignment_ticks: int | None = Field(default=None, ge=0)
    checksum_failures: int = Field(ge=0)
    policy_eligible: bool
    accepted: bool
    rejection_reasons: tuple[str, ...] = ()

    @model_validator(mode="after")
    def accepted_has_no_rejections(self) -> "VisualStreamQualityV1":
        if self.accepted and self.rejection_reasons:
            raise ValueError("accepted visual stream cannot contain rejection reasons")
        return self


class QualityReportV1(ContractModel):
    interface: Literal["minecraft-quality-report/v1"] = "minecraft-quality-report/v1"
    episode_id: str
    accepted: bool
    exact_step_count: int = Field(ge=0)
    weak_step_count: int = Field(ge=0)
    frame_count: int = Field(ge=0)
    dropped_frame_count: int = Field(ge=0)
    max_alignment_ticks: int | None = Field(default=None, ge=0)
    duplicate_or_reversed_ticks: int = Field(ge=0)
    tick_gap_count: int = Field(default=0, ge=0)
    checksum_failures: int = Field(ge=0)
    consent_complete: bool
    privacy_scan_passed: bool
    visual_streams: tuple[VisualStreamQualityV1, ...] = ()
    rejection_reasons: tuple[str, ...] = ()
    eligible_consumers: tuple[Consumer, ...] = ()
    generated_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def accepted_has_no_rejections(self) -> "QualityReportV1":
        if self.accepted and self.rejection_reasons:
            raise ValueError("accepted quality report cannot contain rejection reasons")
        return self


class EpisodeManifestV2(ContractModel):
    interface: Literal["minecraft-episode/v2"] = "minecraft-episode/v2"
    episode_id: str
    session_id: str
    environment_id: str
    world_id: str
    world_snapshot_id: str | None = None
    environment_mode: Literal["live_observe", "isolated_training", "evaluation"]
    task_id: str | None = None
    state: Literal["recording", "finalizing", "accepted", "quarantined", "failed"]
    data_tiers: tuple[DataTier, ...]
    actors: tuple[ActorDescriptor, ...]
    consent: tuple[ConsentRecord, ...] = ()
    visual_streams: tuple[VisualStreamDescriptorV1, ...] = ()
    server_version: str
    adapter_version: str
    protocol_version: int = 2
    action_contract: str = "minecraft-native-control/v2"
    observation_contract: str = "minecraft-observation/v2"
    started_at: datetime = Field(default_factory=utc_now)
    ended_at: datetime | None = None
    first_tick: int | None = Field(default=None, ge=0)
    last_tick: int | None = Field(default=None, ge=0)
    step_count: int = Field(default=0, ge=0)
    frame_count: int = Field(default=0, ge=0)
    artifacts: tuple[ArtifactDescriptor, ...] = ()
    quality_report_ref: str | None = None
    provenance: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_episode_scope(self) -> "EpisodeManifestV2":
        if self.environment_mode != "live_observe" and not self.world_snapshot_id:
            raise ValueError("training and evaluation episodes require a world snapshot")
        if self.state in {"accepted", "quarantined", "failed"} and self.ended_at is None:
            raise ValueError("final episode states require ended_at")
        if self.first_tick is not None and self.last_tick is not None:
            if self.last_tick < self.first_tick:
                raise ValueError("last_tick must not precede first_tick")
        stream_ids = [item.stream_id for item in self.visual_streams]
        if len(stream_ids) != len(set(stream_ids)):
            raise ValueError("episode visual stream_ids must be unique")
        actor_ids = {item.actor_id for item in self.actors}
        if any(item.actor_id not in actor_ids for item in self.visual_streams):
            raise ValueError("visual streams must reference actors declared by the episode")
        consent_by_actor = {item.actor_id: item for item in self.consent}
        for stream in self.visual_streams:
            actor = next(item for item in self.actors if item.actor_id == stream.actor_id)
            if actor.role == "expert_player" and stream.policy_eligible:
                consent = consent_by_actor.get(stream.actor_id)
                if consent is None or not consent.granted or consent.scope not in {
                    "visual",
                    "visual_and_input",
                }:
                    raise ValueError(
                        "trainable demonstrator streams require explicit visual consent"
                    )
        if "rendered_visual" in self.data_tiers:
            rendered = [
                item
                for item in self.visual_streams
                if item.source == "prismarine_renderer" and item.policy_eligible
            ]
            if not rendered:
                raise ValueError(
                    "rendered_visual episodes require a policy-eligible Prismarine stream"
                )
        return self


class WorldSnapshotManifestV1(ContractModel):
    interface: Literal["minecraft-world-snapshot/v1"] = "minecraft-world-snapshot/v1"
    snapshot_id: str
    source_world_id: str
    source_path: str
    archive_path: str
    logical_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    server_version: str
    modpack_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    created_at: datetime = Field(default_factory=utc_now)
    save_confirmed: bool
    immutable: bool = True


class DatasetSnapshotManifestV1(ContractModel):
    interface: Literal["minecraft-dataset-snapshot/v1"] = "minecraft-dataset-snapshot/v1"
    snapshot_id: str
    episode_ids: tuple[str, ...]
    split: Literal["train", "validation", "test"]
    grouping_keys: tuple[Literal["actor", "session", "world"], ...] = (
        "actor",
        "session",
        "world",
    )
    consumers: tuple[Consumer, ...]
    source_hashes: dict[str, str]
    created_at: datetime = Field(default_factory=utc_now)


class ExportManifestV1(ContractModel):
    interface: Literal["minecraft-dataset-export/v1"] = "minecraft-dataset-export/v1"
    export_id: str
    dataset_snapshot_id: str
    format: Literal["minari", "minestudio_lmdb"]
    state: Literal["queued", "running", "complete", "failed", "blocked"]
    output_path: str | None = None
    episode_count: int = Field(default=0, ge=0)
    step_count: int = Field(default=0, ge=0)
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    error: str | None = None
    created_at: datetime = Field(default_factory=utc_now)


class TrainingWorldManifestV1(ContractModel):
    interface: Literal["minecraft-training-world/v1"] = "minecraft-training-world/v1"
    training_world_id: str
    snapshot_id: str
    world_path: str
    server_port: int = Field(ge=1024, le=65535)
    gym_port: int = Field(ge=1024, le=65535)
    state: Literal["created", "starting", "running", "stopped", "failed"]
    control_token_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    live_world_access: Literal[False] = False
    created_at: datetime = Field(default_factory=utc_now)
