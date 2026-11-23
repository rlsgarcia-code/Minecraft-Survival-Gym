#!/usr/bin/env python3
"""Passive live RGB-D odometry audit for the local Minecraft renderer.

The audit never sends a gameplay or camera command. It captures a bounded
sequence of renderer-atomic RGB/depth frames, uses PanorAi's public feature
backend to establish correspondences, and evaluates PanorAi's metric RGB-D
relative-pose API. Static captures are reported as a sensor smoke test; they
are not misrepresented as a live-motion validation.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
from io import BytesIO
import json
import math
from pathlib import Path
import shutil
import time
from typing import Any

import numpy as np
from PIL import Image

from minecraft_gym.dataset import HttpDenseRgbdStreamClient
from minecraft_gym.odometry import GnomonicCameraModel, estimate_panorai_rgbd_odometry


@dataclass(frozen=True, slots=True)
class LiveFrame:
    sequence: int
    frame_id: str
    rgb: np.ndarray
    depth_camera_z: np.ndarray
    depth_status: np.ndarray
    vertical_fov_degrees: float
    max_distance: float
    metadata: dict[str, Any]


def horizontal_fov_degrees(
    vertical_fov_degrees: float, width: int, height: int
) -> float:
    """Derive the horizontal FOV of a symmetric pinhole camera."""

    vertical = math.radians(vertical_fov_degrees)
    return math.degrees(2.0 * math.atan((width / height) * math.tan(vertical / 2.0)))


def _decode_rgb(data: bytes) -> np.ndarray:
    with Image.open(BytesIO(data)) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _capture(client: HttpDenseRgbdStreamClient) -> tuple[LiveFrame, bytes, bytes]:
    captured = client.latest()
    depth, status = captured.depth_arrays()
    rgb = _decode_rgb(captured.visual.data)
    if rgb.shape[:2] != depth.shape:
        raise RuntimeError("RGB and depth rasters are not aligned")
    metadata = {
        "stream_id": captured.visual.stream_id,
        "actor_id": captured.visual.actor_id,
        "frame_id": captured.visual.frame_id,
        "capture_sequence": captured.visual.capture_sequence,
        "captured_monotonic_ns": captured.visual.captured_monotonic_ns,
        "width": captured.visual.width,
        "height": captured.visual.height,
        "rgb_sha256": captured.visual.sha256,
        "depth_sha256": captured.depth_sha256,
        "depth_uncompressed_sha256": captured.depth_uncompressed_sha256,
        "depth_encoding": captured.depth_encoding,
        "max_distance": captured.max_distance,
        "camera_near": captured.camera_near,
        "camera_far": captured.camera_far,
        "vertical_fov_degrees": captured.vertical_fov_degrees,
        "capture_timings_ms": captured.capture_timings_ms,
        "self_entity_excluded": captured.visual.self_entity_excluded,
    }
    frame = LiveFrame(
        sequence=captured.visual.capture_sequence,
        frame_id=captured.visual.frame_id,
        rgb=rgb,
        depth_camera_z=depth,
        depth_status=status,
        vertical_fov_degrees=captured.vertical_fov_degrees,
        max_distance=captured.max_distance,
        metadata=metadata,
    )
    return frame, captured.visual.data, captured.depth_compressed


def _features(image_rgb: np.ndarray) -> dict[str, Any]:
    from panorai.features import FeatureExtractorConfig, OpenCVFeatureBackend

    backend = OpenCVFeatureBackend()
    gray = np.asarray(Image.fromarray(image_rgb).convert("L"), dtype=np.uint8)
    mask = np.full(gray.shape, 255, dtype=np.uint8)
    config = FeatureExtractorConfig(
        method="sift",
        max_features=2500,
        edge_margin_px=8,
        contrast_threshold=0.025,
    )
    result = backend.detect_and_describe(gray, mask, config)
    result["backend"] = backend
    return result


def _matched_pixels(
    features_a: dict[str, Any], features_b: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    from panorai.features import FeatureMatcherConfig

    metadata_a = features_a["metadata"]
    metadata_b = features_b["metadata"]
    if metadata_a["metric"] != metadata_b["metric"]:
        raise RuntimeError("feature descriptor metrics differ between frames")
    matches, _ = features_a["backend"].match(
        features_a["descriptors"],
        features_b["descriptors"],
        metadata_a["metric"],
        FeatureMatcherConfig(method="bf", ratio_test=0.72, cross_check=True),
    )
    query = np.asarray([item["query_idx"] for item in matches], dtype=np.int64)
    train = np.asarray([item["train_idx"] for item in matches], dtype=np.int64)
    return (
        np.asarray(features_a["pixels_xy"], dtype=np.float64)[query],
        np.asarray(features_b["pixels_xy"], dtype=np.float64)[train],
        matches,
    )


def _sample_depth(
    frame: LiveFrame, pixels_xy: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    height, width = frame.depth_camera_z.shape
    xy = np.rint(pixels_xy).astype(np.int64)
    inside = (
        (xy[:, 0] >= 0) & (xy[:, 0] < width) & (xy[:, 1] >= 0) & (xy[:, 1] < height)
    )
    safe_x = np.clip(xy[:, 0], 0, width - 1)
    safe_y = np.clip(xy[:, 1], 0, height - 1)
    depth = frame.depth_camera_z[safe_y, safe_x]
    measured = frame.depth_status[safe_y, safe_x] == 1
    valid = inside & measured & np.isfinite(depth) & (depth > 0.0)
    return depth.astype(np.float64), valid


def _rotation_angle_degrees(rotation: np.ndarray) -> float:
    cosine = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def evaluate_pair(frame_a: LiveFrame, frame_b: LiveFrame) -> dict[str, Any]:
    features_a = _features(frame_a.rgb)
    features_b = _features(frame_b.rgb)
    pixels_a, pixels_b, matches = _matched_pixels(features_a, features_b)
    depth_a, valid_a = _sample_depth(frame_a, pixels_a)
    depth_b, valid_b = _sample_depth(frame_b, pixels_b)
    valid = valid_a & valid_b
    height, width = frame_a.depth_camera_z.shape
    camera = GnomonicCameraModel(
        width=width,
        height=height,
        horizontal_fov_deg=horizontal_fov_degrees(
            frame_a.vertical_fov_degrees, width, height
        ),
        vertical_fov_deg=frame_a.vertical_fov_degrees,
    )
    estimate = None
    if int(np.count_nonzero(valid)) >= 8:
        estimate = estimate_panorai_rgbd_odometry(
            pixels_a,
            pixels_b,
            depth_a,
            depth_b,
            camera=camera,
            valid=valid,
        )
    translation = (
        None
        if estimate is None
        else float(np.linalg.norm(estimate.translation_b_from_a_blocks))
    )
    rotation = (
        None
        if estimate is None
        else _rotation_angle_degrees(estimate.rotation_b_from_a)
    )
    observable_motion = bool(
        estimate is not None
        and estimate.accepted
        and ((translation or 0.0) >= 0.03 or (rotation or 0.0) >= 0.5)
    )
    return {
        "frame_a": frame_a.frame_id,
        "frame_b": frame_b.frame_id,
        "sequence_a": frame_a.sequence,
        "sequence_b": frame_b.sequence,
        "features_a": int(len(features_a["pixels_xy"])),
        "features_b": int(len(features_b["pixels_xy"])),
        "ratio_and_cross_checked_matches": len(matches),
        "metric_depth_matches": int(np.count_nonzero(valid)),
        "estimate_available": estimate is not None,
        "accepted": bool(estimate is not None and estimate.accepted),
        "observable_motion": observable_motion,
        "translation_blocks": translation,
        "rotation_degrees": rotation,
        "metric_inliers": 0 if estimate is None else estimate.metric_inliers,
        "median_metric_residual_blocks": (
            None if estimate is None else estimate.median_metric_residual_blocks
        ),
        "rejection_reasons": (
            ["insufficient_metric_matches"]
            if estimate is None
            else list(estimate.rejection_reasons)
        ),
        "panorai_interface": (
            None if estimate is None else estimate.panorai_description.get("interface")
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    frames_dir = output / "frames"
    frames_dir.mkdir(exist_ok=True)
    free_before = shutil.disk_usage(output).free
    client = HttpDenseRgbdStreamClient(args.url, timeout=args.timeout)
    frames: list[LiveFrame] = []
    for index in range(args.frames):
        frame, rgb_bytes, depth_bytes = _capture(client)
        frames.append(frame)
        stem = f"frame-{index:03d}-{frame.sequence}"
        (frames_dir / f"{stem}.jpg").write_bytes(rgb_bytes)
        (frames_dir / f"{stem}.depth.zlib").write_bytes(depth_bytes)
        (frames_dir / f"{stem}.json").write_text(
            json.dumps(frame.metadata, indent=2, sort_keys=True) + "\n"
        )
        if index + 1 < args.frames:
            time.sleep(args.interval)

    pairs = [evaluate_pair(left, right) for left, right in zip(frames, frames[1:])]
    accepted = sum(bool(item["accepted"]) for item in pairs)
    motion = sum(bool(item["observable_motion"]) for item in pairs)
    valid_fraction = [float(np.mean(frame.depth_status == 1)) for frame in frames]
    free_after = shutil.disk_usage(output).free
    artifact_bytes = sum(
        path.stat().st_size for path in output.rglob("*") if path.is_file()
    )
    report = {
        "interface": "minecraft-panorai-live-odometry-audit/v1",
        "created_at": datetime.now(UTC).isoformat(),
        "mode": "passive_read_only",
        "world_mutated": False,
        "capture_url": args.url,
        "frame_count": len(frames),
        "pair_count": len(pairs),
        "accepted_pairs": accepted,
        "observable_motion_pairs": motion,
        "depth_measured_fraction_min": min(valid_fraction),
        "depth_measured_fraction_max": max(valid_fraction),
        "depth_semantics": "camera_z_blocks",
        "depth_resolution_blocks": 0.01,
        "artifact_bytes_before_report": artifact_bytes,
        "filesystem_free_bytes_before": free_before,
        "filesystem_free_bytes_after_capture": free_after,
        "gate": (
            "live_motion_validated"
            if motion > 0 and accepted > 0
            else "static_sensor_smoke_passed"
            if accepted > 0
            else "failed"
        ),
        "limitations": (
            []
            if motion > 0
            else [
                "No accepted pair contained observable camera motion; this run "
                "validates the real RGB-D path but not moving-camera accuracy."
            ]
        ),
        "pairs": pairs,
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:3008/v1/rgbd/latest?fresh=1")
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--interval", type=float, default=0.35)
    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/panorai-odometry/live-passive-v1"),
    )
    args = parser.parse_args()
    if args.frames < 2 or args.frames > 32:
        parser.error("--frames must be between 2 and 32")
    if args.interval < 0.0:
        parser.error("--interval cannot be negative")
    return args


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))
