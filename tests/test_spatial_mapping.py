from __future__ import annotations

import hashlib

import pytest
from pydantic import ValidationError

from minecraft_gym.dataset import (
    CameraCalibrationV1,
    CameraPoseV1,
    HttpSpatialStreamClient,
    ObservationRefV2,
    RaycastFrameV1,
    RaycastSampleV1,
    SpatialLatticeMap,
    SpatialStreamError,
    SphericalValueLattice,
    ViewpointFrameRefV1,
    depth_map_from_raycast,
    ray_direction_from_pixel,
    trace_lattice_cells_xz,
)


def calibration() -> CameraCalibrationV1:
    return CameraCalibrationV1(
        width=641,
        height=361,
        vertical_fov_degrees=75,
    )


def pose(tick: int = 10) -> CameraPoseV1:
    return CameraPoseV1(
        actor_id="actor:bot",
        server_tick=tick,
        monotonic_ns=tick * 50_000_000,
        origin=(0.5, 65.6, 0.5),
        forward=(0.0, 0.0, 1.0),
        right=(1.0, 0.0, 0.0),
        up=(0.0, 1.0, 0.0),
        yaw_degrees=0.0,
        pitch_degrees=0.0,
        source="prismarine_camera",
    )


def raycast_frame(*, tick: int = 10) -> RaycastFrameV1:
    camera = pose(tick)
    return RaycastFrameV1(
        sensor_id="sensor:bot:prismarine-raycast",
        actor_id="actor:bot",
        server_tick=tick,
        monotonic_ns=tick * 50_000_000,
        camera_stream_id="view:bot:egocentric",
        camera_frame_id=f"frame:bot:{tick}",
        calibration=calibration(),
        pose=camera,
        sampling_columns=1,
        sampling_rows=1,
        samples=(
            RaycastSampleV1(
                pixel_x=320,
                pixel_y=180,
                origin=camera.origin,
                direction=(1.0, 0.0, 0.0),
                max_distance=16.0,
                traversed_distance=2.5,
                termination="block_hit",
                contact_point=(3.0, 65.6, 0.5),
                block_position=(3, 65, 0),
                block_name="sand",
                block_state_id=93,
                face="west",
            ),
        ),
    )


def viewpoint(tick: int = 10) -> ViewpointFrameRefV1:
    digest = hashlib.sha256(f"frame:{tick}".encode()).hexdigest()
    return ViewpointFrameRefV1(
        stream_id="view:bot:egocentric",
        actor_id="actor:bot",
        viewpoint="egocentric",
        source="prismarine_renderer",
        visual_domain="rendered_prismarine",
        frame_id=f"frame:bot:{tick}",
        rgb_ref=f"video/bot/frame-{tick}.jpg",
        server_tick=tick,
        monotonic_ns=tick * 50_000_000,
        capture_sequence=tick,
        width=641,
        height=361,
        sha256=digest,
        alignment_ticks=0,
        alignment_method="nearest_monotonic",
        policy_eligible=True,
    )


def test_center_pixel_projects_along_renderer_forward_basis() -> None:
    center = ray_direction_from_pixel(320, 180, calibration(), pose())
    assert center == pytest.approx((0.0, 0.0, 1.0))

    upper_right = ray_direction_from_pixel(640, 0, calibration(), pose())
    assert upper_right[0] > 0
    assert upper_right[1] > 0
    assert upper_right[2] > 0
    assert sum(value * value for value in upper_right) == pytest.approx(1.0)


def test_raycast_frame_produces_rgb_aligned_metric_depth_and_validity() -> None:
    depth = depth_map_from_raycast(raycast_frame())

    assert depth.camera_frame_id == "frame:bot:10"
    assert depth.server_tick == 10
    assert (depth.source_width, depth.source_height) == (641, 361)
    assert (depth.width, depth.height) == (1, 1)
    assert depth.depth_blocks == pytest.approx((2.5,))
    assert depth.status_codes == (1,)
    assert depth.confidence == pytest.approx((1.0,))
    assert depth.policy_eligible is True


def test_spherical_lattice_keeps_unseen_directions_explicit() -> None:
    lattice = SphericalValueLattice(
        "sphere:episode-1", "actor:bot", azimuth_bins=12, elevation_bins=3
    )
    lattice.integrate(raycast_frame())

    summary = lattice.summary(current_yaw_degrees=0.0, current_pitch_degrees=0.0)

    assert summary.total_bin_count == 36
    assert summary.observed_bin_count == 1
    assert summary.coverage_ratio == pytest.approx(1 / 36)
    observed = [cell for cell in summary.cells if cell.observation_count]
    unseen = [cell for cell in summary.cells if not cell.observation_count]
    assert observed[0].block_counts == {"sand": 1}
    assert observed[0].novelty_value < unseen[0].novelty_value
    assert unseen[0].last_seen_tick is None


def test_lattice_traversal_reports_cells_in_ray_order() -> None:
    assert trace_lattice_cells_xz(
        (0.5, 65.0, 0.5),
        (3.0, 65.0, 0.5),
    ) == ((0, 0), (1, 0), (2, 0))
    assert trace_lattice_cells_xz(
        (0.5, 65.0, 0.5),
        (2.5, 65.0, 2.5),
    ) == ((0, 0), (1, 1), (2, 2))


def test_raycast_hit_produces_free_visited_and_semantic_hit_evidence() -> None:
    spatial_map = SpatialLatticeMap("map:bot:episode-1", "actor:bot")
    update = spatial_map.integrate(
        raycast_frame(),
        visited_position=(0.5, 64.0, 0.5),
    )
    deltas = {(item.cell_x, item.cell_z): item for item in update.deltas}

    assert deltas[(0, 0)].visited_count_delta == 1
    assert deltas[(0, 0)].free_count_delta == 1
    assert deltas[(1, 0)].free_count_delta == 1
    assert deltas[(2, 0)].free_count_delta == 1
    assert deltas[(3, 0)].hit_count_delta == 1
    assert deltas[(3, 0)].block_counts_delta == {"sand": 1}
    assert deltas[(3, 0)].surface_y_min == pytest.approx(65.6)

    snapshot = spatial_map.snapshot()
    assert snapshot[(3, 0)].block_counts["sand"] == 1
    summary = spatial_map.summary()
    assert summary.current_cell == (0, 0)
    assert summary.observed_cell_count == 4
    assert summary.visited_cell_count == 1
    assert summary.block_counts == {"sand": 1}
    assert (2, 0) in summary.frontier_cells

    levels = spatial_map.multiscale_levels()
    assert [level["key"] for level in levels] == ["local", "regional", "world"]
    assert [level["bin_size_blocks"] for level in levels] == [1, 4, 16]
    assert [level["coverage_blocks"] for level in levels] == [33, 132, 528]
    assert levels[0]["current_cell"] == (0, 0)
    assert len(levels[0]["cells"]) == 4
    assert len(levels[1]["cells"]) == 1
    assert levels[1]["cells"][0]["block_counts"] == {"sand": 1}
    assert levels[1]["cells"][0]["solid_stack_height"] == 0
    assert levels[1]["cells"][0]["head_clearance"] == 4.0


def test_vertical_block_levels_distinguish_stack_from_overhead_gap() -> None:
    camera = pose()
    samples = tuple(
        RaycastSampleV1(
            pixel_x=320,
            pixel_y=pixel_y,
            origin=camera.origin,
            direction=(1.0, 0.0, 0.0),
            max_distance=16.0,
            traversed_distance=2.5,
            termination="block_hit",
            contact_point=(3.0, float(block_y) + 0.5, 0.5),
            block_position=(3, block_y, 0),
            block_name="stone",
            block_state_id=1,
            face="west",
        )
        for pixel_y, block_y in ((10, 64), (180, 65), (350, 68))
    )
    frame = RaycastFrameV1(
        sensor_id="sensor:bot:prismarine-raycast",
        actor_id="actor:bot",
        server_tick=10,
        monotonic_ns=500_000_000,
        camera_stream_id="view:bot:egocentric",
        camera_frame_id="frame:bot:10",
        calibration=calibration(),
        pose=camera,
        sampling_columns=1,
        sampling_rows=3,
        samples=samples,
    )

    spatial_map = SpatialLatticeMap("map:bot", "actor:bot")
    spatial_map.integrate(frame)
    cell = spatial_map.multiscale_levels()[0]["cells"][-1]

    assert cell["solid_y_levels"] == [64, 65, 68]
    assert cell["solid_stack_height"] == 1
    assert cell["head_clearance"] == 2.0


def test_unloaded_chunk_does_not_turn_unknown_boundary_into_free_space() -> None:
    camera = pose()
    frame = RaycastFrameV1(
        sensor_id="sensor:bot:prismarine-raycast",
        actor_id="actor:bot",
        server_tick=10,
        monotonic_ns=500_000_000,
        camera_stream_id="view:bot:egocentric",
        camera_frame_id="frame:bot:10",
        calibration=calibration(),
        pose=camera,
        sampling_columns=1,
        sampling_rows=1,
        samples=(
            RaycastSampleV1(
                pixel_x=320,
                pixel_y=180,
                origin=camera.origin,
                direction=(1.0, 0.0, 0.0),
                max_distance=16.0,
                traversed_distance=2.5,
                termination="unloaded_chunk",
            ),
        ),
    )
    update = SpatialLatticeMap("map:bot", "actor:bot").integrate(frame)
    cells = {(item.cell_x, item.cell_z) for item in update.deltas}
    assert cells == {(0, 0), (1, 0)}
    assert all(item.hit_count_delta == 0 for item in update.deltas)


def test_observation_keeps_spatial_provenance_but_blocks_privileged_policy_input() -> None:
    frame = viewpoint()
    update = SpatialLatticeMap("map:bot", "actor:bot").integrate(
        raycast_frame(),
        visited_position=(0.5, 64.0, 0.5),
    )
    observation = ObservationRefV2(
        server_tick=10,
        monotonic_ns=500_000_000,
        frame_id=frame.frame_id,
        rgb_ref=frame.rgb_ref,
        primary_stream_id=frame.stream_id,
        viewpoints=(frame,),
        policy_channels=("rgb",),
        spatial_updates=(update,),
    )
    restored = ObservationRefV2.model_validate_json(observation.model_dump_json())
    assert restored.spatial_updates[0].raycast.samples[0].block_name == "sand"

    with pytest.raises(ValidationError, match="privileged spatial maps"):
        ObservationRefV2.model_validate(
            {
                **observation.model_dump(),
                "policy_channels": ("rgb", "spatial_raycast_map"),
            }
        )


def test_raycast_rejects_block_identity_without_a_hit() -> None:
    with pytest.raises(ValidationError, match="non-hit rays"):
        RaycastSampleV1(
            pixel_x=0,
            pixel_y=0,
            origin=(0.0, 0.0, 0.0),
            direction=(1.0, 0.0, 0.0),
            max_distance=8.0,
            traversed_distance=8.0,
            termination="max_range",
            block_position=(8, 0, 0),
            block_name="stone",
            block_state_id=1,
            contact_point=(8.0, 0.0, 0.0),
        )


def test_http_spatial_stream_pairs_raycast_with_exact_visual_frame(monkeypatch) -> None:
    payload = raycast_frame().model_dump_json().encode()

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return payload

    monkeypatch.setattr(
        "minecraft_gym.dataset.spatial.urllib.request.urlopen",
        lambda request, timeout: Response(),
    )
    client = HttpSpatialStreamClient("http://127.0.0.1:3008/v1/spatial/latest")
    client.start_episode("episode:test")
    update = client.latest(
        server_tick=10,
        camera_frame_id="frame:bot:10",
        camera_stream_id="view:bot:egocentric",
        visited_position=(0.5, 64.0, 0.5),
    )
    assert update.map_id == "map:actor:bot:episode:test"
    assert update.raycast.camera_frame_id == "frame:bot:10"
    cells = client.snapshot_cells()
    assert cells[-1] == {
        "cell_x": 3,
        "cell_z": 0,
        "visited_count": 0,
        "free_count": 0,
        "hit_count": 1,
        "block_counts": {"sand": 1},
        "solid_y_levels": [65],
        "solid_stack_height": 0,
        "head_clearance": 4.0,
        "surface_y_min": pytest.approx(65.6),
        "surface_y_max": pytest.approx(65.6),
        "last_seen_tick": 10,
    }
    voxel_summary = client.voxel_summary()
    voxels = client.snapshot_voxels()
    assert voxel_summary["known_voxel_count"] == 5
    assert voxel_summary["free_voxel_count"] == 4
    assert voxel_summary["solid_voxel_count"] == 1
    assert next(item for item in voxels if item["x"] == 3)["block_name"] == "sand"
    view = client.view_state()
    assert view is not None
    assert view["server_tick"] == 10
    assert view["camera_frame_id"] == "frame:bot:10"
    assert view["sampling_columns"] == 1
    assert view["sampling_rows"] == 1
    assert view["calibration"]["vertical_fov_degrees"] == pytest.approx(75)
    assert view["pose"]["forward"] == [0.0, 0.0, 1.0]

    with pytest.raises(SpatialStreamError, match="visual frame"):
        client.latest(
            server_tick=10,
            camera_frame_id="frame:bot:stale",
            camera_stream_id="view:bot:egocentric",
        )


def test_http_spatial_stream_converts_camera_eye_y_to_feet_y(monkeypatch) -> None:
    payload = raycast_frame().model_dump_json().encode()

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return payload

    monkeypatch.setattr(
        "minecraft_gym.dataset.spatial.urllib.request.urlopen",
        lambda request, timeout: Response(),
    )
    captured: dict[str, tuple[float, float, float]] = {}
    original = SpatialLatticeMap.integrate

    def integrate_with_capture(self, raycast, *, visited_position=None, scope="privileged_derived"):
        captured["position"] = visited_position
        return original(
            self,
            raycast,
            visited_position=visited_position,
            scope=scope,
        )

    monkeypatch.setattr(SpatialLatticeMap, "integrate", integrate_with_capture)
    client = HttpSpatialStreamClient("http://127.0.0.1:3008/v1/spatial/latest")
    client.start_episode("episode:eye-height")
    client.latest(
        server_tick=10,
        camera_frame_id="frame:bot:10",
        camera_stream_id="view:bot:egocentric",
    )
    assert captured["position"] == pytest.approx((0.5, 63.98, 0.5))
