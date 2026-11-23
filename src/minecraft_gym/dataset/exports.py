"""Optional public exporters for accepted, immutable dataset snapshots."""

from __future__ import annotations

import hashlib
import json
import os
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from gymnasium import spaces

from .contracts import EpisodeManifestV2, StepRecordV2
from .storage import iter_steps


def _sha256_tree(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        with path.open("rb") as file_handle:
            while chunk := file_handle.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def _numbers(value: Any, prefix: str = "") -> dict[str, float]:
    result: dict[str, float] = {}
    if isinstance(value, bool):
        result[prefix] = float(value)
    elif isinstance(value, (int, float)):
        result[prefix] = float(value)
    elif isinstance(value, dict):
        for key, child in value.items():
            result.update(_numbers(child, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            result.update(_numbers(child, f"{prefix}.{index}"))
    return result


def _load_episodes(entries: list[dict[str, Any]]) -> list[tuple[EpisodeManifestV2, list[StepRecordV2]]]:
    episodes = []
    for entry in entries:
        manifest = EpisodeManifestV2.model_validate(entry["manifest"])
        steps_path = Path(entry["path"]) / "steps.jsonl"
        episodes.append((manifest, list(iter_steps(steps_path))))
    return episodes


@contextmanager
def _temporary_environment(name: str, value: str) -> Iterator[None]:
    previous = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


def export_minari(
    snapshot: dict[str, Any], entries: list[dict[str, Any]], output_root: Path
) -> tuple[Path, int, int, str]:
    try:
        import minari
        from minari.data_collector import EpisodeBuffer
    except ImportError as exc:
        raise RuntimeError("minari is not installed") from exc

    episodes = _load_episodes(entries)
    if not episodes or any(not steps for _, steps in episodes):
        raise ValueError("Minari export requires non-empty episodes")
    feature_names = sorted(
        {
            key
            for _, steps in episodes
            for step in steps
            for observation in (step.observation, step.next_observation)
            for key in _numbers(observation.structured)
        }
    )
    action_width = max(
        len(step.action.policy_projection or ())
        for _, steps in episodes
        for step in steps
    )
    if action_width == 0:
        raise ValueError("Minari export requires exact Gym action projections")

    def observation_vector(step_observation: Any) -> np.ndarray:
        values = _numbers(step_observation.structured)
        return np.asarray(
            [float(step_observation.server_tick), *[values.get(name, 0.0) for name in feature_names]],
            dtype=np.float32,
        )

    buffers = []
    total_steps = 0
    for index, (_, steps) in enumerate(episodes):
        observations = [observation_vector(steps[0].observation)]
        observations.extend(observation_vector(step.next_observation) for step in steps)
        actions = [
            np.pad(
                np.asarray(step.action.policy_projection, dtype=np.float32),
                (0, action_width - len(step.action.policy_projection or ())),
            )
            for step in steps
        ]
        buffers.append(
            EpisodeBuffer(
                id=index,
                observations=observations,
                actions=actions,
                rewards=[float(step.reward) for step in steps],
                terminations=[bool(step.terminated) for step in steps],
                truncations=[bool(step.truncated) for step in steps],
                infos={
                    "server_tick": [step.next_observation.server_tick for step in steps],
                    "episode_id": [step.episode_id for step in steps],
                },
            )
        )
        total_steps += len(steps)

    safe_id = re.sub(r"[^A-Za-z0-9_-]", "-", snapshot["snapshot_id"])
    dataset_id = f"MinecraftGym/{safe_id}-v0"
    output_root.mkdir(parents=True, exist_ok=True)
    with _temporary_environment("MINARI_DATASETS_PATH", str(output_root)):
        dataset = minari.create_dataset_from_buffers(
            dataset_id=dataset_id,
            buffer=buffers,
            action_space=spaces.Box(-np.inf, np.inf, shape=(action_width,), dtype=np.float32),
            observation_space=spaces.Box(
                -np.inf,
                np.inf,
                shape=(len(feature_names) + 1,),
                dtype=np.float32,
            ),
            algorithm_name="human_or_agent_demonstration",
            description="Minecraft Gym v2 immutable episode export",
            data_format="arrow",
        )
        output = Path(dataset.spec.data_path).parent
    (output / "minecraft-gym-schema.json").write_text(
        json.dumps(
            {
                "interface": "minecraft-gym-minari-schema/v1",
                "snapshot_id": snapshot["snapshot_id"],
                "observation_features": ["server_tick", *feature_names],
                "action_width": action_width,
                "source_hashes": snapshot["source_hashes"],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return output, len(episodes), total_steps, _sha256_tree(output)


def export_minestudio_lmdb(
    snapshot: dict[str, Any], entries: list[dict[str, Any]], output_root: Path
) -> tuple[Path, int, int, str]:
    try:
        import lmdb
    except ImportError as exc:
        raise RuntimeError("lmdb is not installed") from exc

    output = output_root / snapshot["snapshot_id"].replace(":", "_")
    output.mkdir(parents=True, exist_ok=False)
    database = lmdb.open(str(output / "segments.lmdb"), map_size=16 * 1024**3)
    episode_count = total_steps = 0
    with database.begin(write=True) as transaction:
        transaction.put(b"metadata", json.dumps(snapshot, sort_keys=True).encode("utf-8"))
        for entry in entries:
            manifest = EpisodeManifestV2.model_validate(entry["manifest"])
            steps = list(iter_steps(Path(entry["path"]) / "steps.jsonl"))
            for start in range(0, len(steps), 64):
                segment = steps[start : start + 64]
                key = f"episode/{manifest.episode_id}/segment/{start // 64:08d}".encode()
                value = json.dumps(
                    [item.model_dump(mode="json") for item in segment],
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
                transaction.put(key, value, overwrite=False)
            transaction.put(
                f"episode/{manifest.episode_id}/manifest".encode(),
                manifest.model_dump_json().encode("utf-8"),
                overwrite=False,
            )
            episode_count += 1
            total_steps += len(steps)
    database.sync()
    database.close()
    return output, episode_count, total_steps, _sha256_tree(output)
