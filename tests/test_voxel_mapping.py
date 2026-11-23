from __future__ import annotations

import pytest

from minecraft_gym.dataset import (
    CameraCalibrationV1,
    CameraPoseV1,
    RaycastFrameV1,
    RaycastSampleV1,
    SparseVoxelWorldMemory,
    trace_voxels_3d,
    HttpSpatialStreamClient,
)


def _frame(
    *,
    tick: int = 1,
    origin: tuple[float, float, float] = (0.5, 65.5, 0.5),
    direction: tuple[float, float, float] = (1.0, 0.0, 0.0),
    distance: float = 2.5,
    hit: tuple[int, int, int] | None = (3, 65, 0),
    block: str = "stone",
) -> RaycastFrameV1:
    pose = CameraPoseV1(
        actor_id="actor:bot",
        server_tick=tick,
        monotonic_ns=tick,
        origin=origin,
        forward=(0.0, 0.0, 1.0),
        right=(1.0, 0.0, 0.0),
        up=(0.0, 1.0, 0.0),
        source="prismarine_camera",
    )
    return RaycastFrameV1(
        sensor_id="sensor:voxel",
        actor_id="actor:bot",
        server_tick=tick,
        monotonic_ns=tick,
        camera_stream_id="camera:bot",
        camera_frame_id=f"frame:{tick}",
        calibration=CameraCalibrationV1(
            width=1, height=1, vertical_fov_degrees=70.0
        ),
        pose=pose,
        sampling_columns=1,
        sampling_rows=1,
        samples=(
            RaycastSampleV1(
                pixel_x=0,
                pixel_y=0,
                origin=origin,
                direction=direction,
                max_distance=16.0,
                traversed_distance=distance,
                termination="block_hit" if hit is not None else "max_range",
                contact_point=(3.0, 65.5, 0.5) if hit is not None else None,
                block_position=hit,
                block_name=block if hit is not None else None,
                block_state_id=1 if hit is not None else None,
                face="west" if hit is not None else None,
            ),
        ),
    )


def _known_free(memory: SparseVoxelWorldMemory, voxel: tuple[int, int, int]) -> None:
    memory.set_authoritative_block(voxel, None, server_tick=1)


def _known_solid(
    memory: SparseVoxelWorldMemory,
    voxel: tuple[int, int, int],
    block: str = "stone",
) -> None:
    memory.set_authoritative_block(voxel, block, server_tick=1)


def test_3d_dda_preserves_vertical_and_diagonal_voxels() -> None:
    assert trace_voxels_3d((0.5, 0.5, 0.5), (3.0, 0.5, 0.5)) == (
        (0, 0, 0),
        (1, 0, 0),
        (2, 0, 0),
    )
    assert trace_voxels_3d((0.5, 0.5, 0.5), (2.5, 2.5, 2.5)) == (
        (0, 0, 0),
        (1, 1, 1),
        (2, 2, 2),
    )


def test_raycast_reconstructs_free_volume_and_exact_hit_voxel() -> None:
    memory = SparseVoxelWorldMemory("voxels:1", "actor:bot")
    update = memory.integrate(_frame(), visited_position=(0.5, 64.0, 0.5))

    assert update.free_observations == 3
    assert update.solid_observations == 1
    assert memory.get((0, 65, 0)).solid_probability == pytest.approx(0.0)
    assert memory.get((2, 65, 0)).solid_probability == pytest.approx(0.0)
    assert memory.get((3, 65, 0)).dominant_block == "stone"
    assert memory.get((3, 65, 0)).solid_probability == pytest.approx(1.0)
    assert memory.get((0, 64, 0)).visited_count == 1
    assert memory.get((0, 64, 0)).solid_probability == pytest.approx(0.0)
    assert memory.get((0, 65, 0)).source == "body_occupancy"


def test_authoritative_break_replaces_stale_solid_evidence() -> None:
    memory = SparseVoxelWorldMemory("voxels:1", "actor:bot")
    memory.integrate(_frame())
    memory.set_authoritative_block((3, 65, 0), None, server_tick=2)

    state = memory.get((3, 65, 0))
    assert state.solid_probability == pytest.approx(0.0)
    assert state.dominant_block is None
    assert state.source == "authoritative_event"


def test_free_rays_do_not_erase_a_confirmed_partial_block() -> None:
    memory = SparseVoxelWorldMemory("voxels:1", "actor:bot")
    memory.set_authoritative_block((0, 64, 0), "cobblestone_wall", server_tick=1)
    state = memory.get((0, 64, 0))
    assert state is not None
    state.free_count = 51
    state.solid_count = 54
    state.block_counts["cobblestone_wall"] = 54

    assert state.solid_probability == pytest.approx(1.0)
    assert state.partial_geometry is True
    summary = memory.summary()
    assert summary["solid_voxel_count"] == 1
    assert summary["free_voxel_count"] == 0
    assert summary["partial_geometry_voxel_count"] == 1
    assert memory.snapshot_voxels()[0]["partial_geometry"] is True


def test_sparse_chunks_handle_negative_world_coordinates() -> None:
    memory = SparseVoxelWorldMemory("voxels:1", "actor:bot")
    memory.set_authoritative_block((-1, 63, -17), "sand", server_tick=1)

    assert memory.chunk_count == 1
    assert memory.chunk_version((-1, 3, -2)) == 1
    assert memory.get((-1, 63, -17)).dominant_block == "sand"


def test_local_volume_keeps_bridge_and_tunnel_as_distinct_heights() -> None:
    memory = SparseVoxelWorldMemory("voxels:1", "actor:bot")
    # Tunnel floor, two free body voxels, and bridge deck above them all share
    # the same X/Z column.  A single-height map cannot preserve this topology.
    for x in range(-1, 2):
        _known_solid(memory, (x, 63, 0), "stone")
        _known_free(memory, (x, 64, 0))
        _known_free(memory, (x, 65, 0))
        _known_solid(memory, (x, 66, 0), "oak_planks")
        _known_free(memory, (x, 67, 0))
        _known_free(memory, (x, 68, 0))

    volume = memory.local_volume(center=(0, 66, 0), shape=(5, 7, 5))
    center_x = center_z = 2
    # center y=66 lies at output y=3
    assert volume[1, 0, center_z, center_x] == pytest.approx(1.0)  # floor y=63
    assert volume[1, 1, center_z, center_x] == pytest.approx(0.0)  # tunnel air
    assert volume[1, 3, center_z, center_x] == pytest.approx(1.0)  # bridge deck
    assert volume[1, 4, center_z, center_x] == pytest.approx(0.0)  # bridge air


def test_minimap_is_a_conservative_projection_of_voxel_memory() -> None:
    memory = SparseVoxelWorldMemory("voxels:1", "actor:bot")
    for x in range(-1, 2):
        _known_solid(memory, (x, 63, 0), "stone")
        _known_free(memory, (x, 64, 0))
        _known_free(memory, (x, 65, 0))
    # Low ceiling makes the eastern column non-traversable.
    _known_solid(memory, (1, 65, 0), "stone")

    minimap = memory.navigation_minimap(center_feet=(0, 64, 0), size=5)
    row = column = 2
    assert minimap[4, row, column] == pytest.approx(1.0)
    assert minimap[4, row, column + 1] == pytest.approx(0.0)
    # Unobserved northern column remains unknown and non-traversable.
    assert minimap[0, row - 1, column] == pytest.approx(0.0)
    assert minimap[4, row - 1, column] == pytest.approx(0.0)


def test_voxel_summary_reports_vertical_ambiguity_and_bounded_snapshot() -> None:
    memory = SparseVoxelWorldMemory("voxels:1", "actor:bot")
    memory.set_authoritative_block((0, 63, 0), "stone", server_tick=1)
    memory.set_authoritative_block((0, 64, 0), None, server_tick=1)
    memory.set_authoritative_block((0, 65, 0), None, server_tick=1)
    memory.set_authoritative_block((0, 66, 0), "oak_planks", server_tick=1)
    memory.set_authoritative_block((40, 63, 0), "stone", server_tick=1)
    memory.last_visited_voxel = (0, 64, 0)

    summary = memory.summary()
    snapshot = memory.snapshot_voxels(horizontal_radius=2, vertical_radius=4)

    assert summary["vertically_ambiguous_column_count"] == 1
    assert summary["known_voxel_count"] == 5
    assert {(item["x"], item["y"], item["z"]) for item in snapshot} == {
        (0, 63, 0),
        (0, 64, 0),
        (0, 65, 0),
        (0, 66, 0),
    }


def test_http_spatial_client_initializes_voxel_memory_with_episode() -> None:
    client = HttpSpatialStreamClient(actor_id="actor:bot")
    client.start_episode("episode-1")

    summary = client.voxel_summary()
    assert summary["map_id"] == "voxels:actor:bot:episode-1"
    assert summary["known_voxel_count"] == 0
    assert client.snapshot_voxels() == ()
