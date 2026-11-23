"""Public contracts for RGB-to-depth data independent from RL trajectories."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class RGBDContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=True)


class RGBDArtifactV1(RGBDContractModel):
    path: str
    media_type: str
    encoding: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)


class RGBDPairManifestV1(RGBDContractModel):
    """One synchronized dense RGB/depth supervision pair.

    Only a linearized renderer depth buffer can use ``source=gpu_depth_buffer``.
    Upsampled sparse raycasts intentionally have no representable source here.
    """

    interface: Literal["minecraft-rgbd-pair/v1"] = "minecraft-rgbd-pair/v1"
    pair_id: str
    split: Literal["train", "validation", "test"]
    world_id: str
    world_seed_hash: str
    dimension: str
    biome: str
    situation: str
    server_tick: int = Field(ge=0)
    monotonic_ns: int = Field(ge=0)
    camera_frame_id: str
    width: int = Field(gt=0, le=7680)
    height: int = Field(gt=0, le=4320)
    vertical_fov_degrees: float = Field(gt=0, lt=180)
    near_plane_blocks: float = Field(gt=0)
    far_plane_blocks: float = Field(gt=0)
    depth_source: Literal["gpu_depth_buffer"] = "gpu_depth_buffer"
    depth_semantics: Literal["camera_z_blocks"] = "camera_z_blocks"
    rgb: RGBDArtifactV1
    depth: RGBDArtifactV1
    validity: RGBDArtifactV1
    captured_at: datetime = Field(default_factory=_utc_now)

    @model_validator(mode="after")
    def pair_is_calibrated_and_dense(self) -> "RGBDPairManifestV1":
        if self.far_plane_blocks <= self.near_plane_blocks:
            raise ValueError("far plane must be greater than near plane")
        if self.depth.encoding not in {"float16-npz", "uint16-png-linear"}:
            raise ValueError("depth must preserve linear metric values")
        if self.validity.encoding not in {"uint8-png", "bitmask"}:
            raise ValueError("validity must be an explicit pixel mask")
        return self


class RGBDStorageScenarioV1(RGBDContractModel):
    name: Literal["optimistic", "expected", "conservative"]
    rgb_encoded_bytes_per_pixel: float = Field(gt=0)
    depth_encoded_bytes_per_pixel: float = Field(gt=0)
    metadata_bytes_per_pair: int = Field(ge=0)
    bytes_per_pair: int = Field(gt=0)
    dataset_bytes: int = Field(gt=0)
    working_capacity_bytes: int = Field(gt=0)


class RGBDStorageEstimateV1(RGBDContractModel):
    interface: Literal["minecraft-rgbd-storage-estimate/v1"] = (
        "minecraft-rgbd-storage-estimate/v1"
    )
    pairs: int = Field(gt=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    working_overhead_ratio: float = Field(ge=0)
    scenarios: tuple[RGBDStorageScenarioV1, ...]

    @model_validator(mode="after")
    def scenarios_are_complete(self) -> "RGBDStorageEstimateV1":
        names = {item.name for item in self.scenarios}
        if names != {"optimistic", "expected", "conservative"}:
            raise ValueError("storage estimate requires all three scenarios")
        return self


class RGBDSamplingStratumV1(RGBDContractModel):
    axis: Literal["biome_family", "situation"]
    name: str
    weight: float = Field(gt=0)
    target_pairs: int = Field(ge=0)


class RGBDWorldAllocationV1(RGBDContractModel):
    split: Literal["train", "validation", "test"]
    world_count: int = Field(gt=0)
    target_pairs: int = Field(gt=0)
    world_id_prefix: str


class RGBDCaptureGateV1(RGBDContractModel):
    name: str
    status: Literal["pending", "passed", "failed"] = "pending"
    acceptance: str


class RGBDDatasetPlanV1(RGBDContractModel):
    interface: Literal["minecraft-rgbd-dataset-plan/v1"] = (
        "minecraft-rgbd-dataset-plan/v1"
    )
    dataset_id: str
    created_at: datetime = Field(default_factory=_utc_now)
    target_pairs: int = Field(gt=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    target_capture_hz: float = Field(gt=0, le=20)
    candidate_capture_hz: float = Field(gt=0, le=20)
    admission_policy: Literal["novelty_gated"] = "novelty_gated"
    depth_source_required: Literal["gpu_depth_buffer"] = "gpu_depth_buffer"
    depth_semantics: Literal["camera_z_blocks"] = "camera_z_blocks"
    capture_ready: bool = False
    split_policy: Literal["world_and_seed_disjoint"] = "world_and_seed_disjoint"
    world_allocations: tuple[RGBDWorldAllocationV1, ...]
    strata: tuple[RGBDSamplingStratumV1, ...]
    storage: RGBDStorageEstimateV1
    gates: tuple[RGBDCaptureGateV1, ...]
    roadmap: tuple[str, ...]

    @model_validator(mode="after")
    def plan_totals_are_consistent(self) -> "RGBDDatasetPlanV1":
        if sum(item.target_pairs for item in self.world_allocations) != self.target_pairs:
            raise ValueError("world split quotas must sum to target_pairs")
        for axis in ("biome_family", "situation"):
            total = sum(
                item.target_pairs for item in self.strata if item.axis == axis
            )
            if total != self.target_pairs:
                raise ValueError(f"{axis} quotas must sum to target_pairs")
        if self.candidate_capture_hz < self.target_capture_hz:
            raise ValueError("candidate rate cannot be below admitted target rate")
        if self.capture_ready and any(gate.status != "passed" for gate in self.gates):
            raise ValueError("capture cannot be ready while a quality gate is open")
        return self
