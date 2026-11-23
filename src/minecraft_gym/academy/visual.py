"""Controlled RGB/RGB-D policy inputs for visual navigation experiments."""

from __future__ import annotations

import math
from io import BytesIO
from typing import Literal

import numpy as np

from minecraft_gym.dataset import CapturedDenseRgbdFrame, DepthMapV1


VisualModality = Literal["rgb", "rgbd"]
VISUAL_POLICY_CHANNELS = ("red", "green", "blue", "depth", "depth_valid")
ACADEMY_VISUAL_MAX_DISTANCE = 16.0


def encode_visual_policy_input(
    rgb: np.ndarray,
    depth: DepthMapV1,
    *,
    modality: VisualModality,
    output_width: int | None = None,
    output_height: int | None = None,
) -> np.ndarray:
    """Return five aligned planes while holding model capacity constant.

    RGB is sampled at the exact centers of the uniform ray lattice. Both
    experiment arms receive the same shape; the RGB control receives zeros in
    the depth and validity planes.
    """

    image = np.asarray(rgb)
    if image.shape != (depth.source_height, depth.source_width, 3):
        raise ValueError("RGB dimensions do not match the depth source frame")
    if image.dtype != np.uint8:
        raise ValueError("RGB input must use uint8 pixels")
    if modality not in ("rgb", "rgbd"):
        raise ValueError(f"unsupported visual modality: {modality}")

    target_width = depth.source_width if output_width is None else int(output_width)
    target_height = depth.source_height if output_height is None else int(output_height)
    if target_width <= 0 or target_height <= 0:
        raise ValueError("visual output dimensions must be positive")
    sample_y = np.minimum(
        depth.source_height - 1,
        np.floor(
            (np.arange(target_height, dtype=np.float32) + 0.5)
            * depth.source_height
            / target_height
        ).astype(np.int64),
    )
    sample_x = np.minimum(
        depth.source_width - 1,
        np.floor(
            (np.arange(target_width, dtype=np.float32) + 0.5)
            * depth.source_width
            / target_width
        ).astype(np.int64),
    )
    sampled_rgb = image[sample_y[:, None], sample_x[None, :]].astype(np.float32)
    rgb_planes = np.moveaxis(sampled_rgb / 255.0, -1, 0)
    if modality == "rgb":
        auxiliary = np.zeros((2, target_height, target_width), dtype=np.float32)
    else:
        normalized, valid = resample_depth_planes(
            depth,
            output_width=target_width,
            output_height=target_height,
        )
        auxiliary = np.stack((normalized, valid))
    return np.concatenate((rgb_planes, auxiliary), axis=0).astype(
        np.float32, copy=False
    )


def resample_depth_planes(
    depth: DepthMapV1,
    *,
    output_width: int,
    output_height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Align sparse ray cells to an image grid without inventing geometry.

    Nearest-cell projection preserves the discrete validity mask. It gives RGB
    and depth the same tensor resolution, but does not claim more independent
    measurements than the native ray lattice contains.
    """

    if output_width <= 0 or output_height <= 0:
        raise ValueError("depth output dimensions must be positive")
    metric = np.asarray(depth.depth_blocks, dtype=np.float32).reshape(
        depth.height, depth.width
    )
    status = np.asarray(depth.status_codes, dtype=np.uint8).reshape(
        depth.height, depth.width
    )
    source_y = np.minimum(
        depth.height - 1,
        np.floor(
            (np.arange(output_height, dtype=np.float32) + 0.5)
            * depth.height
            / output_height
        ).astype(np.int64),
    )
    source_x = np.minimum(
        depth.width - 1,
        np.floor(
            (np.arange(output_width, dtype=np.float32) + 0.5)
            * depth.width
            / output_width
        ).astype(np.int64),
    )
    projected_metric = metric[source_y[:, None], source_x[None, :]]
    projected_valid = status[source_y[:, None], source_x[None, :]] != 0
    normalized = np.where(
        projected_valid, projected_metric / depth.max_distance, 0.0
    ).astype(np.float32)
    return normalized, projected_valid.astype(np.float32)


def encode_dense_rgbd_policy_input(
    frame: CapturedDenseRgbdFrame,
    *,
    modality: VisualModality = "rgbd",
    output_width: int | None = None,
    output_height: int | None = None,
) -> np.ndarray:
    """Decode a dense frame without reducing its native policy resolution.

    Explicit output dimensions remain available for bounded engineering proxies.
    Production visual policies leave both values unset and own any pooling inside
    their checkpointed neural encoder.
    """

    if modality not in ("rgb", "rgbd"):
        raise ValueError(f"unsupported visual modality: {modality}")
    target_width = frame.visual.width if output_width is None else int(output_width)
    target_height = frame.visual.height if output_height is None else int(output_height)
    if target_width <= 0 or target_height <= 0:
        raise ValueError("visual output dimensions must be positive")
    try:
        from PIL import Image
    except ImportError as error:  # pragma: no cover - declared runtime dependency
        raise RuntimeError("Pillow is required to decode dense RGB-D frames") from error
    with Image.open(BytesIO(frame.visual.data)) as decoded:
        rgb = np.asarray(decoded.convert("RGB"), dtype=np.uint8)
    if rgb.shape != (frame.visual.height, frame.visual.width, 3):
        raise ValueError("dense RGB dimensions do not match stream metadata")
    native_resolution = (
        target_width == frame.visual.width and target_height == frame.visual.height
    )
    if native_resolution:
        source_y = source_x = None
        sampled_rgb = rgb.astype(np.float32)
    else:
        source_y = np.minimum(
            frame.visual.height - 1,
            np.floor(
                (np.arange(target_height, dtype=np.float32) + 0.5)
                * frame.visual.height
                / target_height
            ).astype(np.int64),
        )
        source_x = np.minimum(
            frame.visual.width - 1,
            np.floor(
                (np.arange(target_width, dtype=np.float32) + 0.5)
                * frame.visual.width
                / target_width
            ).astype(np.int64),
        )
        sampled_rgb = rgb[source_y[:, None], source_x[None, :]].astype(np.float32)
    rgb_planes = np.moveaxis(sampled_rgb / 255.0, -1, 0)
    auxiliary = np.zeros((2, target_height, target_width), dtype=np.float32)
    if modality == "rgbd":
        metric, status_raster = frame.depth_arrays()
        if native_resolution:
            sampled_depth = metric
            sampled_status = status_raster
        else:
            assert source_y is not None and source_x is not None
            sampled_depth = metric[source_y[:, None], source_x[None, :]]
            sampled_status = status_raster[source_y[:, None], source_x[None, :]]
        valid = sampled_status != 0
        auxiliary[0] = np.where(
            valid, sampled_depth / frame.max_distance, 0.0
        ).astype(np.float32)
        auxiliary[1] = valid.astype(np.float32)
    return np.concatenate((rgb_planes, auxiliary), axis=0).astype(
        np.float32, copy=False
    )


def academy_visual_pair(
    observation: dict,
    *,
    frame_id: str,
    server_tick: int,
) -> tuple[np.ndarray, DepthMapV1]:
    """Build a deterministic visual proxy from the academy pinhole lattice.

    This helper is only for fast paired engineering tests of the RGB-D
    learner. It does not replace Fabric/Prismarine pixels: the real-world gate
    still requires native RGB and renderer rays. Both ablation arms receive
    the same synthetic RGB; only RGB-D sees metric depth and validity.
    """

    lattice = np.asarray(observation["gnomonic_lattice"], dtype=np.float32)
    if lattice.ndim != 3 or lattice.shape[0] < 5:
        raise ValueError("academy gnomonic lattice is incompatible")
    height, width = lattice.shape[1:]
    evidence = lattice[0] > 0.5
    inverse_depth = np.clip(lattice[1], 0.0, 1.0)
    target = lattice[3] > 0.5
    material = np.clip(lattice[4], 0.0, 1.0)

    rgb = np.empty((height, width, 3), dtype=np.uint8)
    rgb[...] = np.asarray([145, 190, 220], dtype=np.uint8)
    palette_index = np.minimum(3, np.floor(material * 3.999).astype(np.int64))
    palette = np.asarray(
        [
            [82, 105, 135],
            [175, 162, 132],
            [119, 94, 67],
            [92, 92, 92],
        ],
        dtype=np.uint8,
    )
    rgb[evidence] = palette[palette_index[evidence]]
    # The controlled target-navigation proxy uses the public egocentric goal
    # hint only to draw the task's visible beacon into RGB. It is never passed
    # numerically to the policy. The beacon disappears outside the 90° view,
    # preserving the memoryless condition after the actor turns away.
    goal_hint = np.asarray(observation.get("goal_hint", (0.0, 0.0)), dtype=np.float32)
    if goal_hint.shape == (2,) and float(goal_hint[1]) > 0.0:
        angle = math.degrees(math.atan2(float(goal_hint[0]), float(goal_hint[1])))
        if abs(angle) <= 45.0:
            column = int(np.clip(round((angle / 90.0 + 0.5) * (width - 1)), 0, width - 1))
            row = max(1, height // 2 - 1)
            target[max(0, row - 1) : min(height, row + 2), max(0, column - 1) : min(width, column + 2)] = True
    rgb[target] = np.asarray([48, 220, 84], dtype=np.uint8)

    distance = np.where(
        evidence,
        (1.0 - inverse_depth) * ACADEMY_VISUAL_MAX_DISTANCE,
        ACADEMY_VISUAL_MAX_DISTANCE,
    ).astype(np.float32)
    status = np.where(evidence, 1, 2).astype(np.uint8)
    confidence = np.where(evidence, 1.0, 0.0).astype(np.float32)
    depth = DepthMapV1(
        sensor_id="sensor:academy-visual-proxy:depth",
        actor_id="actor:academy-agent",
        server_tick=max(0, int(server_tick)),
        monotonic_ns=max(0, int(server_tick)) * 50_000_000,
        camera_stream_id="view:academy-agent:egocentric",
        camera_frame_id=frame_id,
        source_width=width,
        source_height=height,
        width=width,
        height=height,
        max_distance=ACADEMY_VISUAL_MAX_DISTANCE,
        depth_blocks=tuple(float(item) for item in distance.ravel()),
        status_codes=tuple(int(item) for item in status.ravel()),
        confidence=tuple(float(item) for item in confidence.ravel()),
        policy_eligible=False,
    )
    return rgb, depth
