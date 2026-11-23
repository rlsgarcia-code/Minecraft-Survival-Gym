from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

import minecraft_gym.cli as cli
from minecraft_gym.rgbd import (
    RGBDArtifactV1,
    RGBDPairManifestV1,
    build_collection_plan,
    estimate_storage,
)


def _artifact(*, encoding: str) -> RGBDArtifactV1:
    return RGBDArtifactV1(
        path="pairs/000001.bin",
        media_type="application/octet-stream",
        encoding=encoding,
        sha256="a" * 64,
        size_bytes=123,
    )


def test_storage_estimate_is_reproducible_and_includes_working_headroom() -> None:
    estimate = estimate_storage(pairs=100_000, width=640, height=360)
    scenarios = {item.name: item for item in estimate.scenarios}

    assert scenarios["expected"].bytes_per_pair == 370_688
    assert scenarios["expected"].dataset_bytes == 37_068_800_000
    assert scenarios["conservative"].working_capacity_bytes == 75_136_000_000
    assert all(
        item.working_capacity_bytes > item.dataset_bytes
        for item in estimate.scenarios
    )


def test_collection_plan_balances_quotas_and_keeps_world_splits_disjoint() -> None:
    plan = build_collection_plan(pairs=100_003, worlds=83)

    assert plan.capture_ready is False
    assert sum(item.target_pairs for item in plan.world_allocations) == 100_003
    assert sum(item.world_count for item in plan.world_allocations) == 83
    assert len({item.world_id_prefix for item in plan.world_allocations}) == 3
    for axis in ("biome_family", "situation"):
        assert sum(
            item.target_pairs for item in plan.strata if item.axis == axis
        ) == 100_003
    assert {gate.status for gate in plan.gates} == {"pending"}


def test_pair_contract_accepts_only_dense_metric_renderer_depth() -> None:
    pair = RGBDPairManifestV1(
        pair_id="pair:1",
        split="train",
        world_id="train-world-001",
        world_seed_hash="seedhash",
        dimension="minecraft:overworld",
        biome="minecraft:plains",
        situation="outdoor_traversal",
        server_tick=100,
        monotonic_ns=5_000_000_000,
        camera_frame_id="frame:100",
        width=640,
        height=360,
        vertical_fov_degrees=75,
        near_plane_blocks=0.1,
        far_plane_blocks=64,
        rgb=_artifact(encoding="jpeg"),
        depth=_artifact(encoding="float16-npz"),
        validity=_artifact(encoding="uint8-png"),
    )

    assert pair.depth_source == "gpu_depth_buffer"
    assert pair.depth_semantics == "camera_z_blocks"
    with pytest.raises(ValidationError, match="linear metric"):
        RGBDPairManifestV1.model_validate(
            {
                **pair.model_dump(mode="json"),
                "depth": _artifact(encoding="upsampled-raycast").model_dump(),
            }
        )


def test_rgbd_plan_cli_writes_contract_without_payload(tmp_path, capsys) -> None:
    output = tmp_path / "plans" / "rgbd-v1.json"

    assert cli.main(
        [
            "rgbd",
            "plan",
            "--output",
            str(output),
            "--pairs",
            "10000",
            "--worlds",
            "20",
        ]
    ) == 0

    summary = json.loads(capsys.readouterr().out)
    plan = json.loads(output.read_text(encoding="utf-8"))
    assert summary["capture_ready"] is False
    assert plan["target_pairs"] == 10_000
    assert plan["depth_source_required"] == "gpu_depth_buffer"
    assert list(tmp_path.rglob("*")) == [output.parent, output]
