"""Metric RGB-D odometry adapter over PanorAi's public metric pose API.

PanorAi owns camera geometry, visual consensus, robust metric SE(3) fitting and
the geometric quality decision. Minecraft Gym owns the conversion from the
Prismarine camera-Z convention, application policy and trajectory state.
Importing :mod:`minecraft_gym` therefore does not require PanorAi; the optional
dependency is loaded only when the estimator is called.

Coordinate contract
-------------------
Both cameras use PanorAi's canonical ``+X right, +Y up, +Z forward`` frame.
The returned transform follows PanorAi's convention::

    point_b = R_b_from_a @ point_a + t_b_from_a

Depth samples are camera-Z distances in blocks, matching the current
Prismarine dense-depth stream. They are converted to radial range before the
public ``panorai.estimators.metric_rgbd_pose`` contract is called.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class GnomonicCameraModel:
    """Pixel and field-of-view contract for the Minecraft pinhole camera."""

    width: int
    height: int
    horizontal_fov_deg: float
    vertical_fov_deg: float

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("camera dimensions must be positive")
        for name, value in (
            ("horizontal_fov_deg", self.horizontal_fov_deg),
            ("vertical_fov_deg", self.vertical_fov_deg),
        ):
            if not math.isfinite(float(value)) or not 0.0 < float(value) < 180.0:
                raise ValueError(f"{name} must be finite and in (0, 180)")


@dataclass(frozen=True, slots=True)
class PanorAiRgbdOdometryResult:
    """One auditable camera-B-from-camera-A metric pose estimate."""

    rotation_b_from_a: np.ndarray
    translation_b_from_a_blocks: np.ndarray
    visual_inlier_mask: np.ndarray
    metric_inlier_mask: np.ndarray
    accepted: bool
    rejection_reasons: tuple[str, ...]
    rotation_inliers: int
    metric_inliers: int
    direction_disagreement_deg: float
    median_metric_residual_blocks: float
    panorai_description: dict[str, Any]
    interface: str = "minecraft-panorai-rgbd-odometry/v1"

    def __post_init__(self) -> None:
        rotation = np.array(self.rotation_b_from_a, dtype=np.float64, copy=True)
        translation = np.array(
            self.translation_b_from_a_blocks, dtype=np.float64, copy=True
        )
        visual = np.array(self.visual_inlier_mask, dtype=bool, copy=True)
        metric = np.array(self.metric_inlier_mask, dtype=bool, copy=True)
        if rotation.shape != (3, 3) or translation.shape != (3,):
            raise ValueError(
                "odometry pose requires rotation (3,3) and translation (3,)"
            )
        if visual.ndim != 1 or metric.shape != visual.shape:
            raise ValueError("inlier masks must have matching one-dimensional shape")
        if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-7):
            raise ValueError("rotation must be orthonormal")
        for array in (rotation, translation, visual, metric):
            array.setflags(write=False)
        object.__setattr__(self, "rotation_b_from_a", rotation)
        object.__setattr__(self, "translation_b_from_a_blocks", translation)
        object.__setattr__(self, "visual_inlier_mask", visual)
        object.__setattr__(self, "metric_inlier_mask", metric)


@dataclass(frozen=True, slots=True)
class WorldToCameraPose:
    """World-to-camera SE(3) state used to accumulate relative odometry."""

    rotation_world_to_camera: np.ndarray
    translation_world_to_camera: np.ndarray

    def __post_init__(self) -> None:
        rotation = np.array(self.rotation_world_to_camera, dtype=np.float64, copy=True)
        translation = np.array(
            self.translation_world_to_camera, dtype=np.float64, copy=True
        )
        if rotation.shape != (3, 3) or translation.shape != (3,):
            raise ValueError("pose requires rotation (3,3) and translation (3,)")
        if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-7):
            raise ValueError("rotation must be orthonormal")
        rotation.setflags(write=False)
        translation.setflags(write=False)
        object.__setattr__(self, "rotation_world_to_camera", rotation)
        object.__setattr__(self, "translation_world_to_camera", translation)

    @classmethod
    def identity(cls) -> "WorldToCameraPose":
        return cls(np.eye(3), np.zeros(3))

    @property
    def center_world(self) -> np.ndarray:
        return -self.rotation_world_to_camera.T @ self.translation_world_to_camera

    def advance(
        self,
        rotation_next_from_current: np.ndarray,
        translation_next_from_current: np.ndarray,
    ) -> "WorldToCameraPose":
        rotation = np.asarray(rotation_next_from_current, dtype=np.float64)
        translation = np.asarray(translation_next_from_current, dtype=np.float64)
        return WorldToCameraPose(
            rotation @ self.rotation_world_to_camera,
            rotation @ self.translation_world_to_camera + translation,
        )


def estimate_panorai_rgbd_odometry(
    pixels_a_xy: np.ndarray,
    pixels_b_xy: np.ndarray,
    depth_a_camera_z_blocks: np.ndarray,
    depth_b_camera_z_blocks: np.ndarray,
    *,
    camera: GnomonicCameraModel,
    valid: np.ndarray | None = None,
    relative_pose_options: Any | None = None,
    metric_pose_options: Any | None = None,
    metric_acceptance_policy: Any | None = None,
    min_metric_inliers: int = 8,
    max_median_metric_residual_blocks: float = 0.08,
    stationary_translation_threshold_blocks: float = 0.03,
) -> PanorAiRgbdOdometryResult | None:
    """Estimate metric relative pose from matched RGB pixels and aligned depth.

    The function intentionally receives matches rather than owning a feature
    frontend.  This makes the experiment distinguish three questions:
    PanorAi geometry/pose, RGB feature association, and Minecraft depth quality.
    """

    try:
        from panorai.estimators import RelativePoseOptions
        from panorai.estimators.metric_rgbd_pose import (
            MetricRgbdPoseAcceptancePolicy,
            MetricRgbdPoseOptions,
            estimate_metric_rgbd_relative_pose,
        )
        from panorai.geometry import GnomonicSpec, gnomonic_pixels_to_rays
    except ImportError as error:  # pragma: no cover - optional integration path
        raise RuntimeError(
            "PanorAi is required for RGB-D odometry; install or expose its source "
            "checkout on PYTHONPATH"
        ) from error

    pixels_a = _pixels(pixels_a_xy, "pixels_a_xy")
    pixels_b = _pixels(pixels_b_xy, "pixels_b_xy")
    if pixels_b.shape != pixels_a.shape:
        raise ValueError("pixel correspondence arrays must have the same shape")
    depth_a = _vector(depth_a_camera_z_blocks, len(pixels_a), "depth_a")
    depth_b = _vector(depth_b_camera_z_blocks, len(pixels_a), "depth_b")
    explicit_valid = (
        np.ones(len(pixels_a), dtype=bool)
        if valid is None
        else np.asarray(valid, dtype=bool)
    )
    if explicit_valid.shape != (len(pixels_a),):
        raise ValueError("valid must have shape (N,)")

    spec = GnomonicSpec(
        center_lat_deg=0.0,
        center_lon_deg=0.0,
        hfov_deg=camera.horizontal_fov_deg,
        vfov_deg=camera.vertical_fov_deg,
        output_shape_hw=(camera.height, camera.width),
    )
    projection_a = gnomonic_pixels_to_rays(pixels_a, spec)
    projection_b = gnomonic_pixels_to_rays(pixels_b, spec)
    rays_a = np.asarray(projection_a.rays_xyz, dtype=np.float64)
    rays_b = np.asarray(projection_b.rays_xyz, dtype=np.float64)
    geometry_valid = (
        explicit_valid
        & np.asarray(projection_a.valid, dtype=bool)
        & np.asarray(projection_b.valid, dtype=bool)
    )
    depth_valid = (
        geometry_valid
        & np.isfinite(depth_a)
        & np.isfinite(depth_b)
        & (depth_a > 0.0)
        & (depth_b > 0.0)
        & (rays_a[:, 2] > 1e-6)
        & (rays_b[:, 2] > 1e-6)
    )
    radial_a = np.divide(
        depth_a,
        rays_a[:, 2],
        out=np.full_like(depth_a, np.nan),
        where=depth_valid,
    )
    radial_b = np.divide(
        depth_b,
        rays_b[:, 2],
        out=np.full_like(depth_b, np.nan),
        where=depth_valid,
    )
    visual_options = relative_pose_options or RelativePoseOptions(
        max_angular_error_deg=0.35,
        min_num_trials=8,
        max_num_trials=48,
        min_inliers=max(8, min_metric_inliers),
        local_optimization_steps=2,
        stability_trials=0,
        model_competition_trials=16,
    )
    metric_options = metric_pose_options or MetricRgbdPoseOptions(
        max_metric_error=max_median_metric_residual_blocks,
        min_trials=8,
        max_trials=48,
        local_optimization_steps=2,
        min_inliers=min_metric_inliers,
    )
    metric_policy = metric_acceptance_policy or MetricRgbdPoseAcceptancePolicy(
        min_inliers=min_metric_inliers,
        min_inlier_ratio=0.5,
        max_median_residual_ratio=1.0,
        max_p90_residual_ratio=2.0,
        max_translation_std_ratio=2.0,
    )
    pose = estimate_metric_rgbd_relative_pose(
        rays_a,
        rays_b,
        radial_a,
        radial_b,
        valid=depth_valid,
        options=metric_options,
        visual_options=visual_options,
        acceptance_policy=metric_policy,
    )
    if pose is None:
        return None

    visual_pose = pose.visual_pose
    translation_norm = float(np.linalg.norm(pose.t))
    if (
        visual_pose is not None
        and translation_norm > stationary_translation_threshold_blocks
    ):
        direction_cosine = float(
            np.clip(np.dot(pose.t / translation_norm, visual_pose.t), -1.0, 1.0)
        )
        direction_disagreement = math.degrees(math.acos(direction_cosine))
    else:
        direction_disagreement = math.inf
    reasons = list(pose.quality_report.rejection_reasons)
    accepted = not reasons
    visual_mask = (
        np.zeros(len(pixels_a), dtype=bool)
        if visual_pose is None
        else visual_pose.inlier_mask
    )
    return PanorAiRgbdOdometryResult(
        rotation_b_from_a=pose.R,
        translation_b_from_a_blocks=pose.t,
        visual_inlier_mask=visual_mask,
        metric_inlier_mask=pose.inlier_mask,
        accepted=accepted,
        rejection_reasons=tuple(dict.fromkeys(reasons)),
        rotation_inliers=0 if visual_pose is None else visual_pose.num_inliers,
        metric_inliers=pose.num_inliers,
        direction_disagreement_deg=direction_disagreement,
        median_metric_residual_blocks=pose.quality_report.median_residual,
        panorai_description=pose.describe(),
    )


def _pixels(value: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 2 or len(result) < 5:
        raise ValueError(f"{name} must have shape (N,2) with N >= 5")
    return result


def _vector(value: np.ndarray, length: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (length,):
        raise ValueError(f"{name} must have shape ({length},)")
    return result
