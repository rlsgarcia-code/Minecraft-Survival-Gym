#!/usr/bin/env python3
"""Paired RGB/RGB-D engineering ablation with bounded local storage.

The live sensor smoke captures one real Prismarine RGB/raycast pair without
controlling the family world. Learning runs in the deterministic academy proxy;
it validates the observation intervention and experimental plumbing, not final
Minecraft performance.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import shutil
import time
from typing import Any

import numpy as np
import torch

from minecraft_gym.academy import (
    NavigationAcademyEnv,
    NavigationDifficulty,
    NavigationRequirement,
    academy_visual_pair,
    hybrid_control_action,
    resample_depth_planes,
    VisualHybridPpoAgent,
)
from minecraft_gym.academy.visual_actor_critic import VisualHybridActorCriticAgent
from minecraft_gym.dataset import depth_map_from_raycast
from minecraft_gym.dataset.spatial import HttpSpatialStreamClient
from minecraft_gym.dataset.visual_stream import HttpVisualStreamClient


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def tree_size(root: Path) -> int:
    return sum(item.stat().st_size for item in root.rglob("*") if item.is_file())


class StorageGuard:
    def __init__(self, root: Path, *, cap_bytes: int, minimum_free_bytes: int) -> None:
        self.root = root
        self.cap_bytes = cap_bytes
        self.minimum_free_bytes = minimum_free_bytes
        self.baseline_free_bytes = shutil.disk_usage(root).free

    def check(self, *, phase: str, episode: int | None = None) -> dict[str, Any]:
        output_bytes = tree_size(self.root)
        free_bytes = shutil.disk_usage(self.root).free
        payload = {
            "interface": "minecraft-rgbd-storage-monitor/v1",
            "updated_at": utc_now(),
            "phase": phase,
            "episode": episode,
            "output_bytes": output_bytes,
            "cap_bytes": self.cap_bytes,
            "free_bytes": free_bytes,
            "minimum_free_bytes": self.minimum_free_bytes,
            "baseline_free_bytes": self.baseline_free_bytes,
            "consumed_since_start_bytes": max(0, self.baseline_free_bytes - free_bytes),
        }
        (self.root / "storage-monitor.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        if output_bytes >= self.cap_bytes:
            raise RuntimeError("local RGB-D test reached its hard storage cap")
        if free_bytes < self.minimum_free_bytes:
            raise RuntimeError("system disk crossed the RGB-D safety reserve")
        return payload


def capture_live_pair(output: Path) -> dict[str, Any]:
    destination = output / "live-sensor-smoke"
    destination.mkdir(parents=True, exist_ok=True)
    visual = HttpVisualStreamClient(timeout=3.0)
    frame = visual.latest()
    spatial = HttpSpatialStreamClient(timeout=3.0)
    spatial.start_episode("rgbd-ablation-sensor-smoke")
    server_tick = int(time.monotonic() * 20.0)
    spatial.latest(
        server_tick=server_tick,
        camera_frame_id=frame.frame_id,
        camera_stream_id=frame.stream_id,
    )
    raycast = spatial.raycast_frame()
    if raycast is None:
        raise RuntimeError("raycast stream did not retain the paired frame")
    depth = depth_map_from_raycast(raycast)
    rgb_path = destination / "rgb.jpg"
    depth_path = destination / "depth.npz"
    aligned_depth_path = destination / "depth-aligned-640x360.npz"
    rgb_path.write_bytes(frame.data)
    np.savez_compressed(
        depth_path,
        depth_blocks=np.asarray(depth.depth_blocks, dtype=np.float32).reshape(
            depth.height, depth.width
        ),
        status_codes=np.asarray(depth.status_codes, dtype=np.uint8).reshape(
            depth.height, depth.width
        ),
        confidence=np.asarray(depth.confidence, dtype=np.float32).reshape(
            depth.height, depth.width
        ),
    )
    aligned_depth, aligned_valid = resample_depth_planes(
        depth,
        output_width=frame.width,
        output_height=frame.height,
    )
    np.savez_compressed(
        aligned_depth_path,
        depth_normalized=aligned_depth,
        validity=aligned_valid.astype(np.uint8),
    )
    manifest = {
        "interface": "minecraft-rgbd-live-smoke/v1",
        "captured_at": utc_now(),
        "world_controlled": False,
        "camera_frame_id": frame.frame_id,
        "camera_stream_id": frame.stream_id,
        "server_tick": raycast.server_tick,
        "rgb": {
            "path": rgb_path.name,
            "width": frame.width,
            "height": frame.height,
            "sha256": hashlib.sha256(frame.data).hexdigest(),
        },
        "depth": {
            "path": depth_path.name,
            "width": depth.width,
            "height": depth.height,
            "valid_samples": sum(code == 1 for code in depth.status_codes),
            "sample_count": len(depth.status_codes),
            "max_distance": depth.max_distance,
        },
        "aligned_policy_depth": {
            "path": aligned_depth_path.name,
            "width": frame.width,
            "height": frame.height,
            "projection": "nearest_ray_cell_with_validity_mask",
            "independent_measurements": len(depth.status_codes),
        },
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def visible_task_seeds(
    *,
    base_seed: int,
    count: int,
    difficulty_cycle: tuple[NavigationDifficulty, ...] | None = None,
) -> list[tuple[int, str]]:
    result: list[tuple[int, str]] = []
    candidate = base_seed
    cycle = difficulty_cycle or (
        NavigationDifficulty.EASY,
        NavigationDifficulty.EASY,
        NavigationDifficulty.MODERATE,
    )
    while len(result) < count:
        difficulty = cycle[len(result) % len(cycle)]
        env = NavigationAcademyEnv(
            difficulty=difficulty,
            requirement=NavigationRequirement.TRAVEL,
            max_episode_steps=80,
        )
        observation, _ = env.reset(
            seed=candidate, options={"requirement": NavigationRequirement.TRAVEL}
        )
        rgb, _ = academy_visual_pair(
            observation,
            frame_id=f"schedule:{candidate}",
            server_tick=0,
        )
        if np.any(np.all(rgb == np.asarray([48, 220, 84], dtype=np.uint8), axis=2)):
            result.append((candidate, difficulty.slug))
        env.close()
        candidate += 1
        if candidate - base_seed > count * 200:
            raise RuntimeError("could not construct enough initially visible targets")
    return result


def tagged(observation: dict[str, Any], *, episode: int, step: int) -> dict[str, Any]:
    return {
        **observation,
        "_visual_tick": step,
        "_visual_frame_id": f"academy:{episode}:{step}",
    }


def run_episode(
    agent: VisualHybridActorCriticAgent,
    *,
    arena_seed: int,
    difficulty: str,
    episode_index: int,
    training: bool,
) -> dict[str, Any]:
    env = NavigationAcademyEnv(
        difficulty=difficulty,
        requirement=NavigationRequirement.TRAVEL,
        max_episode_steps=80,
    )
    observation, _ = env.reset(
        seed=arena_seed, options={"requirement": NavigationRequirement.TRAVEL}
    )
    observation = tagged(observation, episode=episode_index, step=0)
    trajectory: list[tuple[Any, ...]] = []
    total_return = 0.0
    action = agent.choose(observation, stochastic=training)
    final_info: dict[str, Any] = {}
    for step in range(1, 81):
        next_observation, reward, terminated, truncated, info = env.step_control(
            hybrid_control_action(action)
        )
        next_observation = tagged(
            next_observation, episode=episode_index, step=step
        )
        trajectory.append(
            (
                observation,
                action,
                float(reward),
                next_observation,
                bool(terminated),
                bool(truncated),
            )
        )
        total_return += float(reward)
        final_info = info
        if terminated or truncated:
            success = bool(info.get("is_success"))
            break
        observation = next_observation
        action = agent.choose(observation, stochastic=training)
    else:
        success = False
    diagnostics = agent.update_episode(trajectory) if training else {}
    env.close()
    return {
        "arena_seed": arena_seed,
        "difficulty": difficulty,
        "return": total_return,
        "success": success,
        "steps": len(trajectory),
        "elapsed_ticks": int(final_info.get("elapsed_ticks", 0)),
        "actor_loss": diagnostics.get("actor_loss"),
        "critic_loss": diagnostics.get("critic_loss"),
        "entropy": diagnostics.get("entropy"),
        "actor_gradient_norm": diagnostics.get("actor_gradient_norm"),
        "raw_advantage": diagnostics.get("raw_advantage"),
        "state_value": diagnostics.get("state_value"),
        "approximate_kl": diagnostics.get("approximate_kl"),
        "clip_fraction": diagnostics.get("clip_fraction"),
        "policy_ratio": diagnostics.get("policy_ratio"),
        "ppo_update_applied": diagnostics.get("ppo_update_applied"),
        "ppo_epochs_completed": diagnostics.get("ppo_epochs_completed"),
        "ppo_early_stop_kl": diagnostics.get("ppo_early_stop_kl"),
        "rollout_steps": diagnostics.get("rollout_steps"),
    }


def summarize(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    successes = [item for item in episodes if item["success"]]
    return {
        "episodes": len(episodes),
        "success_rate": float(np.mean([item["success"] for item in episodes])),
        "median_return": float(np.median([item["return"] for item in episodes])),
        "median_ticks_to_success": (
            float(np.median([item["elapsed_ticks"] for item in successes]))
            if successes
            else None
        ),
    }


def write_live_progress(
    output: Path,
    *,
    modality: str,
    replica: int,
    episodes: list[dict[str, Any]],
    status: str,
) -> None:
    rewards = [float(item["return"]) for item in episodes]
    window = min(25, len(rewards))
    moving = [
        float(np.mean(rewards[max(0, index - window + 1) : index + 1]))
        for index in range(len(rewards))
    ]
    optimization_points = [
        (index + 1, item)
        for index, item in enumerate(episodes)
        if item.get("critic_loss") is not None
    ]
    optimization_episodes = [episode for episode, _ in optimization_points]
    optimization_items = [item for _, item in optimization_points]
    payload = {
        "interface": "minecraft-vgg-rgbd-training-progress/v2",
        "status": status,
        "updated_at": utc_now(),
        "environment": "procedural_academy_proxy",
        "real_minecraft_training": False,
        "modality": modality,
        "replica": replica,
        "episodes": len(episodes),
        "reward_curve": rewards,
        "reward_moving_average_25": moving,
        "success_curve": [bool(item["success"]) for item in episodes],
        "optimization_episode_indices": optimization_episodes,
        "actor_loss_curve": [item["actor_loss"] for item in optimization_items],
        "critic_loss_curve": [item["critic_loss"] for item in optimization_items],
        "entropy_curve": [item["entropy"] for item in optimization_items],
        "gradient_norm_curve": [
            item["actor_gradient_norm"] for item in optimization_items
        ],
        "raw_advantage_curve": [item["raw_advantage"] for item in optimization_items],
        "state_value_curve": [item["state_value"] for item in optimization_items],
        "approximate_kl_curve": [
            item.get("approximate_kl") for item in optimization_items
        ],
        "clip_fraction_curve": [
            item.get("clip_fraction") for item in optimization_items
        ],
        "policy_ratio_curve": [
            item.get("policy_ratio") for item in optimization_items
        ],
        "ppo_update_curve": [
            bool(item.get("ppo_update_applied")) for item in episodes
        ],
        "ppo_epochs_completed_curve": [
            item.get("ppo_epochs_completed") for item in optimization_items
        ],
        "ppo_early_stop_kl_curve": [
            item.get("ppo_early_stop_kl") for item in optimization_items
        ],
        "rollout_steps_curve": [
            item.get("rollout_steps") for item in optimization_items
        ],
        "series_axes": {
            "episode_curves": "episode_index_1_based",
            "optimization_curves": "optimization_episode_indices",
        },
        "summary": summarize(episodes) if episodes else None,
    }
    temporary = output / "progress.json.partial"
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(output / "progress.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-episodes", type=int, default=60)
    parser.add_argument("--evaluation-episodes", type=int, default=20)
    parser.add_argument("--replicas", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2600)
    parser.add_argument("--storage-cap-mib", type=int, default=512)
    parser.add_argument("--minimum-free-gib", type=int, default=20)
    parser.add_argument("--input-width", type=int, default=160)
    parser.add_argument("--input-height", type=int, default=90)
    parser.add_argument(
        "--modalities",
        nargs="+",
        choices=("rgb", "rgbd"),
        default=("rgb", "rgbd"),
    )
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--minibatch-size", type=int, default=8)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--checkpoint-validation-episodes", type=int, default=20)
    parser.add_argument("--algorithm", choices=("actor-critic", "ppo"), default="ppo")
    parser.add_argument("--rollout-episodes", type=int, default=10)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--ppo-clip-ratio", type=float, default=0.2)
    parser.add_argument("--ppo-value-clip", type=float, default=0.2)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--target-kl", type=float, default=0.02)
    parser.add_argument("--balanced-difficulties", action="store_true")
    parser.add_argument("--backbone-learning-rate", type=float, default=1e-5)
    parser.add_argument("--representation-learning-rate", type=float, default=3e-5)
    parser.add_argument("--actor-learning-rate", type=float, default=1e-4)
    parser.add_argument("--critic-learning-rate", type=float, default=3e-4)
    parser.add_argument("--skip-live-smoke", action="store_true")
    args = parser.parse_args()

    if (
        args.algorithm == "ppo"
        and args.checkpoint_every > 0
        and args.checkpoint_every % args.rollout_episodes != 0
    ):
        raise ValueError("checkpoint-every must be divisible by rollout-episodes")

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    guard = StorageGuard(
        output,
        cap_bytes=args.storage_cap_mib * 1024**2,
        minimum_free_bytes=args.minimum_free_gib * 1024**3,
    )
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    guard.check(phase="initializing")
    live = None if args.skip_live_smoke else capture_live_pair(output)
    guard.check(phase="live_sensor_smoke_complete")

    difficulty_cycle = (
        (NavigationDifficulty.EASY, NavigationDifficulty.MODERATE)
        if args.balanced_difficulties
        else None
    )
    train_schedule = visible_task_seeds(
        base_seed=args.seed,
        count=args.train_episodes,
        difficulty_cycle=difficulty_cycle,
    )
    evaluation_schedule = visible_task_seeds(
        base_seed=args.seed + 1_000_000,
        count=args.evaluation_episodes,
        difficulty_cycle=difficulty_cycle,
    )
    checkpoint_validation_schedule = visible_task_seeds(
        base_seed=args.seed + 2_000_000,
        count=args.checkpoint_validation_episodes,
        difficulty_cycle=difficulty_cycle,
    )
    modalities = tuple(dict.fromkeys(args.modalities))
    device = (
        "mps"
        if args.device == "auto" and torch.backends.mps.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    arms: dict[str, list[dict[str, Any]]] = {name: [] for name in modalities}
    completed = 0
    total = args.replicas * len(modalities) * (
        args.train_episodes + args.evaluation_episodes
    )
    for replica in range(args.replicas):
        for modality in modalities:
            agent_class = (
                VisualHybridPpoAgent
                if args.algorithm == "ppo"
                else VisualHybridActorCriticAgent
            )
            algorithm_kwargs = (
                {
                    "rollout_episodes": args.rollout_episodes,
                    "ppo_epochs": args.ppo_epochs,
                    "clip_ratio": args.ppo_clip_ratio,
                    "value_clip_range": args.ppo_value_clip,
                    "gae_lambda": args.gae_lambda,
                    "target_kl": args.target_kl,
                }
                if args.algorithm == "ppo"
                else {}
            )
            agent = agent_class(
                modality=modality,
                input_width=args.input_width,
                input_height=args.input_height,
                seed=args.seed + replica,
                actor_learning_rate=args.actor_learning_rate,
                critic_learning_rate=args.critic_learning_rate,
                entropy_coefficient=0.01,
                device=device,
                minibatch_size=args.minibatch_size,
                backbone_learning_rate=args.backbone_learning_rate,
                representation_learning_rate=args.representation_learning_rate,
                **algorithm_kwargs,
            )
            train_results: list[dict[str, Any]] = []
            best_checkpoint_score = (float("-inf"), float("-inf"))
            for episode_index, (arena_seed, difficulty) in enumerate(train_schedule):
                result = run_episode(
                    agent,
                    arena_seed=arena_seed,
                    difficulty=difficulty,
                    episode_index=episode_index,
                    training=True,
                )
                train_results.append(result)
                write_live_progress(
                    output,
                    modality=modality,
                    replica=replica,
                    episodes=train_results,
                    status="running",
                )
                if (
                    args.checkpoint_every > 0
                    and (episode_index + 1) % args.checkpoint_every == 0
                ):
                    checkpoint_metadata = {
                        "proxy_only": True,
                        "modality": modality,
                        "replica": replica,
                        "completed_training_episodes": episode_index + 1,
                        "architecture": agent.architecture,
                        "algorithm": args.algorithm,
                    }
                    agent.save(
                        output / "rolling-policy.pt",
                        metadata=checkpoint_metadata,
                    )
                    checkpoint_validation = [
                        run_episode(
                            agent,
                            arena_seed=validation_seed,
                            difficulty=validation_difficulty,
                            episode_index=(episode_index + 1) * 10_000 + offset,
                            training=False,
                        )
                        for offset, (
                            validation_seed,
                            validation_difficulty,
                        ) in enumerate(checkpoint_validation_schedule)
                    ]
                    checkpoint_validation_summary = summarize(
                        checkpoint_validation
                    )
                    checkpoint_score = (
                        float(checkpoint_validation_summary["success_rate"]),
                        float(checkpoint_validation_summary["median_return"]),
                    )
                    guard.check(
                        phase=f"checkpoint_validation:{modality}:{replica}",
                        episode=completed,
                    )
                    if checkpoint_score > best_checkpoint_score:
                        best_checkpoint_score = checkpoint_score
                        agent.save(
                            output / "best-validation-policy.pt",
                            metadata={
                                **checkpoint_metadata,
                                "selection_metric": (
                                    "frozen_validation_success_then_median_return"
                                ),
                                "selection_value": list(checkpoint_score),
                                "validation_summary": (
                                    checkpoint_validation_summary
                                ),
                            },
                        )
                completed += 1
                guard.check(phase=f"training:{modality}:{replica}", episode=completed)
                print(
                    json.dumps(
                        {
                            "phase": "training",
                            "modality": modality,
                            "replica": replica,
                            "episode": episode_index + 1,
                            "total_progress": completed / total,
                            "recent_success": float(
                                np.mean([item["success"] for item in train_results[-20:]])
                            ),
                            "recent_return": float(
                                np.mean([item["return"] for item in train_results[-20:]])
                            ),
                        }
                    ),
                    flush=True,
                )
            if isinstance(agent, VisualHybridPpoAgent):
                agent.flush_rollout()
            evaluation_results: list[dict[str, Any]] = []
            for offset, (arena_seed, difficulty) in enumerate(evaluation_schedule):
                result = run_episode(
                    agent,
                    arena_seed=arena_seed,
                    difficulty=difficulty,
                    episode_index=args.train_episodes + offset,
                    training=False,
                )
                evaluation_results.append(result)
                completed += 1
                guard.check(phase=f"evaluation:{modality}:{replica}", episode=completed)
            arm_dir = output / f"{modality}-replica-{replica}"
            arm_dir.mkdir(parents=True, exist_ok=True)
            checkpoint = agent.save(
                arm_dir / "policy.pt",
                metadata={
                    "modality": modality,
                    "replica": replica,
                    "training_schedule": train_schedule,
                    "evaluation_schedule": evaluation_schedule,
                    "proxy_only": True,
                    "input_width": args.input_width,
                    "input_height": args.input_height,
                    "encoder_stride": agent.encoder_stride,
                    "architecture": agent.architecture,
                    "algorithm": args.algorithm,
                    "device": device,
                    "minibatch_size": args.minibatch_size,
                    "learning_rates": {
                        "backbone": args.backbone_learning_rate,
                        "representation": args.representation_learning_rate,
                        "actor": args.actor_learning_rate,
                        "critic": args.critic_learning_rate,
                    },
                },
            )
            result = {
                "modality": modality,
                "replica": replica,
                "checkpoint": str(checkpoint),
                "train": train_results,
                "evaluation": evaluation_results,
                "train_summary": summarize(train_results),
                "evaluation_summary": summarize(evaluation_results),
            }
            (arm_dir / "metrics.json").write_text(
                json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
            )
            arms[modality].append(result)
            write_live_progress(
                output,
                modality=modality,
                replica=replica,
                episodes=train_results,
                status="complete",
            )

    report = {
        "interface": "minecraft-rgbd-controlled-ablation/v1",
        "completed_at": utc_now(),
        "status": "engineering_proxy_complete",
        "live_sensor_smoke": live,
        "controlled_variables": {
            "paired_training_schedule": train_schedule,
            "paired_evaluation_schedule": evaluation_schedule,
            "checkpoint_validation_schedule": checkpoint_validation_schedule,
            "same_architecture_channels": 5,
            "aligned_input_shape": [5, args.input_height, args.input_width],
            "depth_projection": "nearest_ray_cell_with_validity_mask",
            "visual_architecture": "vgg-rgbd-shared-flatten-mlp/v1",
            "learning_algorithm": args.algorithm,
            "balanced_difficulties": args.balanced_difficulties,
            "training_device": device,
            "same_action_contract": "minecraft-control-camera-keyboard/v3",
            "same_reward_contract": "minecraft-bounded-asymptotic-time-terminal/v3",
            "memory_enabled": False,
            "real_minecraft_training": False,
        },
        "arms": {
            modality: [item["evaluation_summary"] for item in results]
            for modality, results in arms.items()
        },
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    report["storage"] = guard.check(phase="complete")
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
