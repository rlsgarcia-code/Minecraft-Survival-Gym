"""Planning contracts for the standalone Minecraft RGB-to-depth dataset.

This package deliberately does not depend on RL episode records.  A paired
perception sample may later be referenced by an episode, but its world split,
camera calibration, depth semantics, and storage budget remain independently
auditable.
"""

from .contracts import (
    RGBDArtifactV1,
    RGBDCaptureGateV1,
    RGBDDatasetPlanV1,
    RGBDPairManifestV1,
    RGBDSamplingStratumV1,
    RGBDStorageEstimateV1,
    RGBDStorageScenarioV1,
    RGBDWorldAllocationV1,
)
from .planning import (
    DEFAULT_BIOME_FAMILIES,
    DEFAULT_SITUATIONS,
    build_collection_plan,
    estimate_storage,
    write_collection_plan,
)

__all__ = [
    "DEFAULT_BIOME_FAMILIES",
    "DEFAULT_SITUATIONS",
    "RGBDArtifactV1",
    "RGBDCaptureGateV1",
    "RGBDDatasetPlanV1",
    "RGBDPairManifestV1",
    "RGBDSamplingStratumV1",
    "RGBDStorageEstimateV1",
    "RGBDStorageScenarioV1",
    "RGBDWorldAllocationV1",
    "build_collection_plan",
    "estimate_storage",
    "write_collection_plan",
]
