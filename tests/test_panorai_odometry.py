from __future__ import annotations

import importlib.util
import math

import numpy as np
import pytest

from minecraft_gym.odometry import (
    GnomonicCameraModel,
    WorldToCameraPose,
    estimate_panorai_rgbd_odometry,
)


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("panorai") is None,
    reason="PanorAi is an optional external integration",
)


def _yaw(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray(((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c)))


def _fixture(seed: int = 8):
    from panorai.geometry import GnomonicSpec, rays_to_gnomonic_pixels

    rng = np.random.default_rng(seed)
    rotation = _yaw(math.radians(4.0))
    translation = np.asarray((-0.28, 0.02, -0.05))
    points_a = rng.uniform((-4.0, -2.0, 3.0), (4.0, 3.0, 15.0), size=(160, 3))
    points_b = points_a @ rotation.T + translation
    bearings_a = points_a / np.linalg.norm(points_a, axis=1, keepdims=True)
    bearings_b = points_b / np.linalg.norm(points_b, axis=1, keepdims=True)
    spec = GnomonicSpec(hfov_deg=90.0, vfov_deg=60.0, output_shape_hw=(360, 640))
    projection_a = rays_to_gnomonic_pixels(bearings_a, spec)
    projection_b = rays_to_gnomonic_pixels(bearings_b, spec)
    valid = np.asarray(projection_a.valid) & np.asarray(projection_b.valid)
    pixels_a = np.asarray(projection_a.pixels_xy)[valid]
    pixels_b = np.asarray(projection_b.pixels_xy)[valid]
    points_a = points_a[valid]
    points_b = points_b[valid]
    pixels_a += rng.normal(scale=0.2, size=pixels_a.shape)
    pixels_b += rng.normal(scale=0.2, size=pixels_b.shape)
    return pixels_a, pixels_b, points_a[:, 2], points_b[:, 2], rotation, translation


def test_panorai_adapter_recovers_metric_rgbd_pose() -> None:
    pixels_a, pixels_b, depth_a, depth_b, expected_rotation, expected_translation = (
        _fixture()
    )
    result = estimate_panorai_rgbd_odometry(
        pixels_a,
        pixels_b,
        depth_a,
        depth_b,
        camera=GnomonicCameraModel(640, 360, 90.0, 60.0),
    )
    assert result is not None
    assert result.accepted, result.rejection_reasons
    rotation_error = math.degrees(
        math.acos(
            float(
                np.clip(
                    (np.trace(result.rotation_b_from_a @ expected_rotation.T) - 1.0)
                    / 2.0,
                    -1.0,
                    1.0,
                )
            )
        )
    )
    assert rotation_error < 0.1
    assert (
        np.linalg.norm(result.translation_b_from_a_blocks - expected_translation) < 0.02
    )
    assert result.metric_inliers >= 100
    assert result.panorai_description["interface"] == (
        "panorai-metric-rgbd-relative-pose/v1-experimental"
    )
    assert result.panorai_description["depth_semantics"] == (
        "radial-range-along-unit-bearing"
    )


def test_world_to_camera_pose_accumulates_relative_transform() -> None:
    relative_rotation = _yaw(math.radians(10.0))
    relative_translation = np.asarray((-0.5, 0.0, 0.1))
    pose = WorldToCameraPose.identity().advance(relative_rotation, relative_translation)
    assert np.allclose(pose.rotation_world_to_camera, relative_rotation)
    assert np.allclose(pose.translation_world_to_camera, relative_translation)
    assert np.allclose(pose.center_world, -relative_rotation.T @ relative_translation)
