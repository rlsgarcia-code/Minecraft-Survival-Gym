"""Deterministic sampling and capacity planning for RGB-to-depth collection."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Iterable

from .contracts import (
    RGBDCaptureGateV1,
    RGBDDatasetPlanV1,
    RGBDSamplingStratumV1,
    RGBDStorageEstimateV1,
    RGBDStorageScenarioV1,
    RGBDWorldAllocationV1,
)


DEFAULT_BIOME_FAMILIES: tuple[tuple[str, float], ...] = (
    ("temperate_plains_and_forest", 0.18),
    ("jungle_and_dense_vegetation", 0.10),
    ("desert_and_badlands", 0.10),
    ("snow_and_frozen", 0.10),
    ("swamp_and_mangrove", 0.08),
    ("mountains_and_cliffs", 0.10),
    ("ocean_river_and_shore", 0.10),
    ("caves_and_deep_dark", 0.14),
    ("nether", 0.07),
    ("end", 0.03),
)

DEFAULT_SITUATIONS: tuple[tuple[str, float], ...] = (
    ("outdoor_traversal", 0.25),
    ("player_built_interior", 0.15),
    ("cave_and_tunnel", 0.18),
    ("water_surface_and_submerged", 0.08),
    ("vegetation_occlusion", 0.08),
    ("snow_weather_and_low_light", 0.07),
    ("lava_and_hazard", 0.05),
    ("entities_and_combat", 0.07),
    ("construction_and_block_change", 0.07),
)


def _allocate(total: int, weighted_names: Iterable[tuple[str, float]]) -> list[tuple[str, float, int]]:
    values = list(weighted_names)
    if total <= 0:
        raise ValueError("total must be positive")
    if not values or any(weight <= 0 for _, weight in values):
        raise ValueError("weights must be positive")
    weight_sum = sum(weight for _, weight in values)
    exact = [(name, weight, total * weight / weight_sum) for name, weight in values]
    floors = [math.floor(value) for _, _, value in exact]
    remaining = total - sum(floors)
    order = sorted(
        range(len(exact)),
        key=lambda index: (exact[index][2] - floors[index], -index),
        reverse=True,
    )
    for index in order[:remaining]:
        floors[index] += 1
    return [
        (name, weight / weight_sum, quota)
        for (name, weight, _), quota in zip(exact, floors, strict=True)
    ]


def estimate_storage(
    *,
    pairs: int,
    width: int = 640,
    height: int = 360,
    working_overhead_ratio: float = 0.25,
) -> RGBDStorageEstimateV1:
    """Estimate encoded dataset and safe working capacity.

    The estimates include RGB, linear uint16/float16 depth, a validity mask,
    and approximately 2 KiB of manifest/index data per pair. They deliberately
    exclude model checkpoints and world snapshots.
    """

    if pairs <= 0 or width <= 0 or height <= 0:
        raise ValueError("pairs and resolution must be positive")
    if working_overhead_ratio < 0:
        raise ValueError("working_overhead_ratio cannot be negative")
    pixels = width * height
    assumptions = (
        ("optimistic", 0.35, 0.60),
        ("expected", 0.60, 1.00),
        ("conservative", 1.00, 1.60),
    )
    scenarios: list[RGBDStorageScenarioV1] = []
    for name, rgb_bpp, depth_bpp in assumptions:
        bytes_per_pair = math.ceil(pixels * (rgb_bpp + depth_bpp) + 2048)
        dataset_bytes = bytes_per_pair * pairs
        working_bytes = math.ceil(dataset_bytes * (1.0 + working_overhead_ratio))
        scenarios.append(
            RGBDStorageScenarioV1(
                name=name,
                rgb_encoded_bytes_per_pixel=rgb_bpp,
                depth_encoded_bytes_per_pixel=depth_bpp,
                metadata_bytes_per_pair=2048,
                bytes_per_pair=bytes_per_pair,
                dataset_bytes=dataset_bytes,
                working_capacity_bytes=working_bytes,
            )
        )
    return RGBDStorageEstimateV1(
        pairs=pairs,
        width=width,
        height=height,
        working_overhead_ratio=working_overhead_ratio,
        scenarios=tuple(scenarios),
    )


def build_collection_plan(
    *,
    dataset_id: str = "minecraft-rgb-depth-v1",
    pairs: int = 100_000,
    width: int = 640,
    height: int = 360,
    worlds: int = 80,
) -> RGBDDatasetPlanV1:
    """Build the auditable plan; this does not claim capture readiness."""

    if not dataset_id:
        raise ValueError("dataset_id cannot be empty")
    if worlds < 10:
        raise ValueError("at least 10 worlds are required for disjoint splits")
    split_worlds = _allocate(worlds, (("train", 0.8), ("validation", 0.1), ("test", 0.1)))
    split_pairs = _allocate(pairs, (("train", 0.8), ("validation", 0.1), ("test", 0.1)))
    world_allocations = tuple(
        RGBDWorldAllocationV1(
            split=split_name,
            world_count=world_quota,
            target_pairs=pair_quota,
            world_id_prefix=f"{split_name}-world-",
        )
        for (split_name, _, world_quota), (_, _, pair_quota) in zip(
            split_worlds, split_pairs, strict=True
        )
    )
    strata = tuple(
        RGBDSamplingStratumV1(
            axis=axis,
            name=name,
            weight=weight,
            target_pairs=quota,
        )
        for axis, values in (
            ("biome_family", DEFAULT_BIOME_FAMILIES),
            ("situation", DEFAULT_SITUATIONS),
        )
        for name, weight, quota in _allocate(pairs, values)
    )
    gates = (
        RGBDCaptureGateV1(
            name="dense_gpu_depth",
            acceptance="Every RGB pixel has linear camera-Z depth or an explicit invalid mask; no sparse upsampling.",
        ),
        RGBDCaptureGateV1(
            name="frame_synchronization",
            acceptance="RGB, depth, validity, calibration and pose share one immutable renderer frame id.",
        ),
        RGBDCaptureGateV1(
            name="latency_budget",
            acceptance="20 Hz collector soak keeps >=19.8 FPS and adds <=5 ms p95 capture latency.",
        ),
        RGBDCaptureGateV1(
            name="metric_calibration",
            acceptance="Synthetic planes at known ranges stay within 1% or 0.05 block absolute depth error.",
        ),
        RGBDCaptureGateV1(
            name="split_isolation",
            acceptance="No world id, seed hash or chunk lineage crosses train, validation and test.",
        ),
        RGBDCaptureGateV1(
            name="pilot_quality",
            acceptance="A 5,000-pair pilot passes corruption, duplicate, coverage and storage audits.",
        ),
    )
    roadmap = (
        "Extract and linearize the Prismarine/Three.js GPU depth buffer at the RGB render resolution.",
        "Bind RGB, depth, validity, pose and calibration to one immutable frame identifier.",
        "Benchmark 20 Hz latency and run metric-depth calibration scenes.",
        "Collect a 5,000-pair pilot with novelty admission and inspect difficult materials.",
        "Freeze world/seed-disjoint splits and collect the 100,000-pair baseline.",
        "Audit hashes, duplicates, depth distributions, biome/situation quotas and invalid pixels.",
        "Train RGB-to-depth baselines and publish model-independent dataset cards and metrics.",
    )
    return RGBDDatasetPlanV1(
        dataset_id=dataset_id,
        target_pairs=pairs,
        width=width,
        height=height,
        target_capture_hz=5.0,
        candidate_capture_hz=20.0,
        capture_ready=False,
        world_allocations=world_allocations,
        strata=strata,
        storage=estimate_storage(pairs=pairs, width=width, height=height),
        gates=gates,
        roadmap=roadmap,
    )


def write_collection_plan(plan: RGBDDatasetPlanV1, output: Path) -> Path:
    """Atomically write a plan without creating any dataset payload."""

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(plan.model_dump(mode="json"), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, output)
    return output
