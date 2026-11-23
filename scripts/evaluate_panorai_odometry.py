#!/usr/bin/env python3
"""Isolated PanorAi RGB-D odometry audit with a Minecraft-like trajectory."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import importlib.metadata
import json
import math
from pathlib import Path
import subprocess
from typing import Any

import numpy as np

from minecraft_gym.odometry import (
    GnomonicCameraModel,
    WorldToCameraPose,
    estimate_panorai_rgbd_odometry,
)


@dataclass(frozen=True, slots=True)
class Command:
    forward_blocks: float
    yaw_delta_deg: float
    motion_fraction: float = 1.0
    label: str = "move"


def yaw_rotation(angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.asarray(((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c)))


def rotation_error_deg(estimated: np.ndarray, expected: np.ndarray) -> float:
    cosine = float(np.clip((np.trace(estimated @ expected.T) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def bearing_error_deg(estimated: np.ndarray, expected: np.ndarray) -> float:
    estimated = estimated / np.linalg.norm(estimated)
    expected = expected / np.linalg.norm(expected)
    return math.degrees(
        math.acos(float(np.clip(np.dot(estimated, expected), -1.0, 1.0)))
    )


def command_transform(command: Command) -> tuple[np.ndarray, np.ndarray]:
    rotation = yaw_rotation(math.radians(-command.yaw_delta_deg))
    translation = np.asarray((0.0, 0.0, -command.forward_blocks))
    return rotation, translation


def true_transform(command: Command) -> tuple[np.ndarray, np.ndarray]:
    rotation = yaw_rotation(math.radians(-command.yaw_delta_deg))
    translation = np.asarray(
        (0.0, 0.0, -command.forward_blocks * command.motion_fraction)
    )
    return rotation, translation


def synthetic_landmarks(seed: int, count: int = 6000) -> np.ndarray:
    """Axis-aligned surfaces and clutter resembling a block-built corridor."""

    rng = np.random.default_rng(seed)
    floor = np.column_stack(
        (
            rng.uniform(-10.0, 10.0, count // 4),
            np.full(count // 4, -1.62),
            rng.uniform(2.0, 34.0, count // 4),
        )
    )
    left = np.column_stack(
        (
            np.full(count // 4, -7.0),
            rng.uniform(-1.6, 5.0, count // 4),
            rng.uniform(1.0, 34.0, count // 4),
        )
    )
    right = left.copy()
    right[:, 0] = 7.0
    clutter = rng.uniform((-6.5, -1.5, 3.0), (6.5, 4.0, 32.0), size=(count // 4, 3))
    # Quantization creates repeated block faces without making every point
    # perfectly coplanar, a difficult but realistic Minecraft configuration.
    clutter[:, :2] = np.round(clutter[:, :2] * 2.0) / 2.0
    return np.concatenate((floor, left, right, clutter), axis=0)


def matched_observations(
    landmarks_world: np.ndarray,
    pose_a: WorldToCameraPose,
    pose_b: WorldToCameraPose,
    *,
    camera: GnomonicCameraModel,
    rng: np.random.Generator,
    maximum_matches: int = 180,
    pixel_noise_std: float = 0.25,
    depth_relative_noise_std: float = 0.002,
    outlier_fraction: float = 0.10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from panorai.geometry import GnomonicSpec, rays_to_gnomonic_pixels

    points_a = (
        landmarks_world @ pose_a.rotation_world_to_camera.T
        + pose_a.translation_world_to_camera
    )
    points_b = (
        landmarks_world @ pose_b.rotation_world_to_camera.T
        + pose_b.translation_world_to_camera
    )
    rays_a = points_a / np.linalg.norm(points_a, axis=1, keepdims=True)
    rays_b = points_b / np.linalg.norm(points_b, axis=1, keepdims=True)
    spec = GnomonicSpec(
        hfov_deg=camera.horizontal_fov_deg,
        vfov_deg=camera.vertical_fov_deg,
        output_shape_hw=(camera.height, camera.width),
    )
    projection_a = rays_to_gnomonic_pixels(rays_a, spec)
    projection_b = rays_to_gnomonic_pixels(rays_b, spec)
    visible = (
        (points_a[:, 2] > 0.25)
        & (points_b[:, 2] > 0.25)
        & np.asarray(projection_a.valid)
        & np.asarray(projection_b.valid)
    )
    indices = np.flatnonzero(visible)
    if len(indices) < 20:
        raise RuntimeError(f"only {len(indices)} common landmarks are visible")
    if len(indices) > maximum_matches:
        indices = rng.choice(indices, size=maximum_matches, replace=False)
    pixels_a = np.asarray(projection_a.pixels_xy)[indices].copy()
    pixels_b = np.asarray(projection_b.pixels_xy)[indices].copy()
    depth_a = points_a[indices, 2].copy()
    depth_b = points_b[indices, 2].copy()
    pixels_a += rng.normal(scale=pixel_noise_std, size=pixels_a.shape)
    pixels_b += rng.normal(scale=pixel_noise_std, size=pixels_b.shape)
    depth_a *= 1.0 + rng.normal(scale=depth_relative_noise_std, size=len(depth_a))
    depth_b *= 1.0 + rng.normal(scale=depth_relative_noise_std, size=len(depth_b))
    outlier_count = int(round(outlier_fraction * len(indices)))
    if outlier_count:
        selected = rng.choice(len(indices), size=outlier_count, replace=False)
        permuted = np.roll(selected, 1)
        pixels_b[selected] = pixels_b[permuted]
        depth_b[selected] = depth_b[permuted]
    return pixels_a, pixels_b, depth_a, depth_b


def experiment(seed: int) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    camera = GnomonicCameraModel(640, 360, 90.0, 60.0)
    commands = [
        Command(0.35, delta)
        for delta in (0, 2, 0, -3, 0, 4, 0, 0, -2, 0, 3, 0, 0, -4, 0, 2, 0, 0, -2, 0, 3, 0, 0, -2)
    ]
    commands[7] = Command(0.35, 0.0, 0.15, "partial_collision")
    commands[8] = Command(0.35, -2.0, 0.20, "partial_collision")
    commands[12] = Command(0.0, 18.0, 0.0, "rotation_only")
    commands[18] = Command(0.35, -2.0, 0.10, "partial_collision")

    landmarks = synthetic_landmarks(seed + 91)
    target_world = np.asarray((3.0, 0.0, 22.0))
    true_pose = WorldToCameraPose.identity()
    action_pose = WorldToCameraPose.identity()
    fused_pose = WorldToCameraPose.identity()
    true_poses = [true_pose]
    action_poses = [action_pose]
    fused_poses = [fused_pose]
    records: list[dict[str, Any]] = []

    for step, command in enumerate(commands, start=1):
        expected_rotation, expected_translation = true_transform(command)
        next_true = true_pose.advance(expected_rotation, expected_translation)
        pixels_a, pixels_b, depth_a, depth_b = matched_observations(
            landmarks,
            true_pose,
            next_true,
            camera=camera,
            rng=rng,
        )
        estimate = estimate_panorai_rgbd_odometry(
            pixels_a,
            pixels_b,
            depth_a,
            depth_b,
            camera=camera,
        )
        command_rotation, command_translation = command_transform(command)
        action_pose = action_pose.advance(command_rotation, command_translation)
        if estimate is not None and estimate.accepted:
            fused_pose = fused_pose.advance(
                estimate.rotation_b_from_a,
                estimate.translation_b_from_a_blocks,
            )
            update_source = "panorai_rgbd"
        else:
            # Action prediction is explicit fallback. It is not silently
            # relabelled as visual odometry when geometry is unobservable.
            fused_pose = fused_pose.advance(command_rotation, command_translation)
            update_source = "action_fallback"

        true_target = (
            next_true.rotation_world_to_camera @ target_world
            + next_true.translation_world_to_camera
        )
        fused_target = (
            fused_pose.rotation_world_to_camera @ target_world
            + fused_pose.translation_world_to_camera
        )
        action_position_error = float(
            np.linalg.norm(action_pose.center_world - next_true.center_world)
        )
        fused_position_error = float(
            np.linalg.norm(fused_pose.center_world - next_true.center_world)
        )
        records.append(
            {
                "step": step,
                "command": asdict(command),
                "matches": len(pixels_a),
                "update_source": update_source,
                "accepted": bool(estimate is not None and estimate.accepted),
                "rejection_reasons": (
                    ["no_pose_hypothesis"]
                    if estimate is None
                    else list(estimate.rejection_reasons)
                ),
                "rotation_error_deg": (
                    None
                    if estimate is None
                    else rotation_error_deg(
                        estimate.rotation_b_from_a, expected_rotation
                    )
                ),
                "translation_error_blocks": (
                    None
                    if estimate is None
                    else float(
                        np.linalg.norm(
                            estimate.translation_b_from_a_blocks
                            - expected_translation
                        )
                    )
                ),
                "metric_inliers": 0 if estimate is None else estimate.metric_inliers,
                "median_metric_residual_blocks": (
                    None if estimate is None else estimate.median_metric_residual_blocks
                ),
                "direction_disagreement_deg": (
                    None if estimate is None else estimate.direction_disagreement_deg
                ),
                "action_position_error_blocks": action_position_error,
                "fused_position_error_blocks": fused_position_error,
                "target_bearing_error_deg": bearing_error_deg(fused_target, true_target),
                "true_center_world": next_true.center_world.tolist(),
                "action_center_world": action_pose.center_world.tolist(),
                "fused_center_world": fused_pose.center_world.tolist(),
            }
        )
        true_pose = next_true
        true_poses.append(true_pose)
        action_poses.append(action_pose)
        fused_poses.append(fused_pose)

    accepted = [record for record in records if record["accepted"]]
    summary = {
        "interface": "minecraft-panorai-odometry-audit/v1",
        "created_at": datetime.now(UTC).isoformat(),
        "seed": seed,
        "steps": len(records),
        "accepted_steps": len(accepted),
        "acceptance_rate": len(accepted) / len(records),
        "action_only_final_drift_blocks": records[-1]["action_position_error_blocks"],
        "fused_final_drift_blocks": records[-1]["fused_position_error_blocks"],
        "fused_position_rmse_blocks": float(
            np.sqrt(np.mean([r["fused_position_error_blocks"] ** 2 for r in records]))
        ),
        "target_bearing_rmse_deg": float(
            np.sqrt(np.mean([r["target_bearing_error_deg"] ** 2 for r in records]))
        ),
        "accepted_rotation_error_median_deg": float(
            np.median([r["rotation_error_deg"] for r in accepted])
        ),
        "accepted_translation_error_median_blocks": float(
            np.median([r["translation_error_blocks"] for r in accepted])
        ),
        "collision_steps": [
            r["step"] for r in records if "collision" in r["command"]["label"]
        ],
        "rotation_only_steps": [
            r["step"] for r in records if r["command"]["label"] == "rotation_only"
        ],
    }
    arrays = {
        "true": np.stack([pose.center_world for pose in true_poses]),
        "action": np.stack([pose.center_world for pose in action_poses]),
        "fused": np.stack([pose.center_world for pose in fused_poses]),
    }
    return summary, records, arrays


def dependency_evidence() -> dict[str, Any]:
    import panorai

    source = Path(panorai.__file__).resolve().parents[1]
    try:
        version = importlib.metadata.version("panorai")
    except importlib.metadata.PackageNotFoundError:
        version = getattr(panorai, "__version__", "source-checkout")
    commit = None
    dirty = None
    try:
        commit = subprocess.run(
            ["git", "-C", str(source), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", str(source), "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        pass
    return {
        "package": "panorai",
        "version": version,
        "source": str(source),
        "commit": commit,
        "dirty_checkout": dirty,
        "apis": [
            "panorai.geometry.GnomonicSpec",
            "panorai.geometry.rays_to_gnomonic_pixels",
            "panorai.geometry.gnomonic_pixels_to_rays",
            "panorai.estimators.metric_rgbd_pose.estimate_metric_rgbd_relative_pose",
        ],
    }


def plot(output: Path, records: list[dict[str, Any]], arrays: dict[str, np.ndarray]) -> None:
    import matplotlib.pyplot as plt

    steps = np.arange(1, len(records) + 1)
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for name, style in (("true", "k-"), ("action", "--"), ("fused", "-")):
        values = arrays[name]
        axes[0].plot(values[:, 0], values[:, 2], style, label=name)
    axes[0].set_title("Trajetória X–Z")
    axes[0].set_xlabel("X (blocos)")
    axes[0].set_ylabel("Z (blocos)")
    axes[0].axis("equal")
    axes[0].legend()
    axes[1].plot(
        steps, [r["action_position_error_blocks"] for r in records], label="ações"
    )
    axes[1].plot(
        steps, [r["fused_position_error_blocks"] for r in records], label="fusão"
    )
    axes[1].set_title("Deriva acumulada")
    axes[1].set_xlabel("Passo")
    axes[1].set_ylabel("Erro de posição (blocos)")
    axes[1].legend()
    axes[2].plot(steps, [r["target_bearing_error_deg"] for r in records])
    axes[2].set_title("Direção estimada do alvo")
    axes[2].set_xlabel("Passo")
    axes[2].set_ylabel("Erro angular (graus)")
    for record in records:
        if record["update_source"] == "action_fallback":
            for axis in axes[1:]:
                axis.axvline(record["step"], color="#b7791f", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output / "odometry-audit.png", dpi=160)
    plt.close(figure)


def report(summary: dict[str, Any], dependency: dict[str, Any]) -> str:
    outcome = (
        "supports a bounded RGB-D odometry prototype"
        if summary["fused_final_drift_blocks"] < summary["action_only_final_drift_blocks"]
        else "does not yet beat action-only dead reckoning"
    )
    fallback_note = (
        "Amber vertical lines in the plot identify steps where the RGB-D gate "
        "declined the measurement and the integrator used the action prediction "
        "explicitly."
        if summary["accepted_steps"] < summary["steps"]
        else "The RGB-D gate accepted every step, so no action-fallback markers appear."
    )
    return f"""# PanorAi RGB-D odometry isolated audit

## Outcome

The isolated geometric test **{outcome}**. It used PanorAi's public gnomonic
geometry and dedicated metric RGB-D relative-pose estimator after converting
synchronized Minecraft camera-Z depth to radial range.

| Metric | Result |
| --- | ---: |
| Accepted RGB-D updates | {summary['accepted_steps']} / {summary['steps']} ({summary['acceptance_rate']:.1%}) |
| Median rotation error | {summary['accepted_rotation_error_median_deg']:.4f}° |
| Median metric translation error | {summary['accepted_translation_error_median_blocks']:.4f} blocks |
| Action-only final drift | {summary['action_only_final_drift_blocks']:.4f} blocks |
| Fused final drift | {summary['fused_final_drift_blocks']:.4f} blocks |
| Fused position RMSE | {summary['fused_position_rmse_blocks']:.4f} blocks |
| Target-bearing RMSE | {summary['target_bearing_rmse_deg']:.4f}° |

{fallback_note}

## Scope and limitations

This is a known-correspondence geometric ceiling test, not proof that Minecraft
RGB feature matching is solved. It exercises pixel/ray geometry, robust metric
pose, camera-Z to radial conversion, trajectory composition,
partial collisions and target-relative state. A real captured Minecraft RGB-D
sequence is required for the next gate: frontend matching, dynamic objects,
repeated block textures, render latency and depth/RGB synchronization.

The PanorAi checkout was `{dependency['source']}` at commit
`{dependency['commit']}` (reported version `{dependency['version']}`). Dirty
checkout at execution: `{dependency['dirty_checkout']}`. No PanorAi files were
modified or copied into Minecraft Gym.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261123)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summary, records, arrays = experiment(args.seed)
    dependency = dependency_evidence()
    payload = {**summary, "dependency": dependency, "records": records}
    (args.output / "metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    plot(args.output, records, arrays)
    (args.output / "REPORT.md").write_text(
        report(summary, dependency), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
