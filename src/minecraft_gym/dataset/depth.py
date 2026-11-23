"""RGB-aligned metric depth derived from the canonical raycast contract."""

from __future__ import annotations

from .contracts import DepthMapV1, RaycastFrameV1


def depth_map_from_raycast(frame: RaycastFrameV1) -> DepthMapV1:
    """Project a uniform ray frame into a row-major depth/validity raster."""

    max_distance = max(sample.max_distance for sample in frame.samples)
    status = {
        "unloaded_chunk": 0,
        "block_hit": 1,
        "max_range": 2,
    }
    return DepthMapV1(
        sensor_id=f"{frame.sensor_id}:depth",
        actor_id=frame.actor_id,
        server_tick=frame.server_tick,
        monotonic_ns=frame.monotonic_ns,
        camera_stream_id=frame.camera_stream_id,
        camera_frame_id=frame.camera_frame_id,
        source_width=frame.calibration.width,
        source_height=frame.calibration.height,
        width=frame.sampling_columns,
        height=frame.sampling_rows,
        max_distance=max_distance,
        depth_blocks=tuple(sample.traversed_distance for sample in frame.samples),
        status_codes=tuple(status[sample.termination] for sample in frame.samples),
        confidence=tuple(sample.confidence for sample in frame.samples),
    )
