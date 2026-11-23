"""Crash-tolerant episode storage and admission gates.

The external research volume is never created implicitly: an absent or
unprepared SSD must degrade capture safely instead of filling the system disk.
"""

from __future__ import annotations

import hashlib
import json
import os
import plistlib
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

from .contracts import (
    ArtifactDescriptor,
    EpisodeManifestV2,
    QualityReportV1,
    StepRecordV2,
    VisualStreamQualityV1,
)


LAYOUT_DIRECTORIES = (
    "raw",
    "quarantine",
    "snapshots",
    "worlds",
    "checkpoints",
    "catalog",
)


def _now() -> datetime:
    return datetime.now(UTC)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        while chunk := file_handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _json_line(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


_PRIVATE_EVENT_KEYS = {
    "chat",
    "message",
    "player_name",
    "username",
    "display_name",
    "player_list",
    "players",
}


def _redact_event(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: "[REDACTED]" if str(key).lower() in _PRIVATE_EVENT_KEYS else _redact_event(child)
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [_redact_event(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_event(item) for item in value)
    return value


@dataclass(frozen=True, slots=True)
class StorageStatus:
    external_root: str
    spool_root: str
    external_mounted: bool
    external_writable: bool
    filesystem: str | None
    external_total_bytes: int
    external_free_bytes: int
    external_free_ratio: float
    system_free_bytes: int
    spool_bytes: int
    visual_capture_allowed: bool
    structured_capture_allowed: bool
    reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
        }


class ResearchStorage:
    """Own storage policy, layout checks, and recovery of interrupted writes."""

    def __init__(
        self,
        external_root: str | os.PathLike[str] | None = None,
        *,
        spool_root: str | os.PathLike[str] | None = None,
        spool_limit_bytes: int = 4 * 1024**3,
        minimum_system_free_bytes: int = 8 * 1024**3,
        minimum_external_free_ratio: float = 0.15,
        require_apfs: bool = True,
        require_mountpoint: bool | None = None,
    ) -> None:
        self.external_root = Path(
            external_root
            or os.getenv("MINECRAFT_GYM_RESEARCH_VOLUME")
            or os.getenv("BRINCABOT_RESEARCH_VOLUME")
            or "/Volumes/MinecraftRL"
        )
        self.spool_root = Path(
            spool_root
            or os.getenv("MINECRAFT_GYM_RESEARCH_SPOOL")
            or os.getenv("BRINCABOT_RESEARCH_SPOOL")
            or str(Path.home() / "Library" / "Caches" / "MinecraftGym" / "spool")
        )
        self.spool_limit_bytes = spool_limit_bytes
        self.minimum_system_free_bytes = minimum_system_free_bytes
        self.minimum_external_free_ratio = minimum_external_free_ratio
        self.require_apfs = require_apfs
        self.require_mountpoint = require_apfs if require_mountpoint is None else require_mountpoint

    def ensure_layout(self) -> None:
        if not self._external_is_mounted():
            raise RuntimeError(
                f"Research volume is not mounted at {self.external_root}; "
                "the application will not create a fake mount directory"
            )
        for directory in LAYOUT_DIRECTORIES:
            (self.external_root / directory).mkdir(parents=True, exist_ok=True)
        self.spool_root.mkdir(parents=True, exist_ok=True)

    def status(self) -> StorageStatus:
        mounted = self._external_is_mounted()
        writable = mounted and os.access(self.external_root, os.W_OK)
        filesystem = self._filesystem_type(self.external_root) if mounted else None
        ext_total, ext_free = self._space(self.external_root) if mounted else (0, 0)
        ext_ratio = ext_free / ext_total if ext_total else 0.0
        system_anchor = self.spool_root.parent
        while not system_anchor.exists() and system_anchor != system_anchor.parent:
            system_anchor = system_anchor.parent
        _, system_free = self._space(system_anchor)
        spool_bytes = self._tree_size(self.spool_root)
        reasons: list[str] = []
        if not mounted:
            reasons.append("research_volume_missing")
        elif not writable:
            reasons.append("research_volume_read_only")
        if mounted and self.require_apfs and filesystem not in {"apfs", "APFS"}:
            reasons.append("research_volume_not_apfs")
        if mounted and ext_ratio < self.minimum_external_free_ratio:
            reasons.append("research_volume_low_space")
        if system_free < self.minimum_system_free_bytes:
            reasons.append("system_disk_low_space")
        if spool_bytes >= self.spool_limit_bytes:
            reasons.append("spool_limit_reached")
        visual_blockers = {
            "research_volume_missing",
            "research_volume_read_only",
            "research_volume_not_apfs",
            "research_volume_low_space",
            "system_disk_low_space",
            "spool_limit_reached",
        }
        return StorageStatus(
            external_root=str(self.external_root),
            spool_root=str(self.spool_root),
            external_mounted=mounted,
            external_writable=writable,
            filesystem=filesystem,
            external_total_bytes=ext_total,
            external_free_bytes=ext_free,
            external_free_ratio=ext_ratio,
            system_free_bytes=system_free,
            spool_bytes=spool_bytes,
            visual_capture_allowed=not visual_blockers.intersection(reasons),
            structured_capture_allowed=(
                system_free >= self.minimum_system_free_bytes
                and spool_bytes < self.spool_limit_bytes
            ),
            reasons=tuple(reasons),
        )

    def _external_is_mounted(self) -> bool:
        if not self.external_root.is_dir():
            return False
        return not self.require_mountpoint or os.path.ismount(self.external_root)

    @property
    def external_mounted(self) -> bool:
        return self._external_is_mounted()

    def episode_parent(self, *, visual: bool) -> Path:
        status = self.status()
        if visual:
            if not status.visual_capture_allowed:
                raise RuntimeError(
                    "Visual capture is blocked: " + ", ".join(status.reasons)
                )
            self.ensure_layout()
            parent = self.external_root / "raw" / ".partial"
        else:
            if not status.structured_capture_allowed:
                raise RuntimeError(
                    "Structured capture is blocked: " + ", ".join(status.reasons)
                )
            if status.external_mounted and status.external_writable:
                self.ensure_layout()
                parent = self.external_root / "raw" / ".partial"
            else:
                self.spool_root.mkdir(parents=True, exist_ok=True)
                parent = self.spool_root / ".partial"
        parent.mkdir(parents=True, exist_ok=True)
        return parent

    def recover_partials(self, *, stale_after_seconds: float = 60.0) -> list[str]:
        recovered: list[str] = []
        parents = [self.spool_root / ".partial"]
        if self._external_is_mounted():
            parents.append(self.external_root / "raw" / ".partial")
        now = time.time()
        for parent in parents:
            if not parent.is_dir():
                continue
            for episode in parent.iterdir():
                if not episode.is_dir() or now - episode.stat().st_mtime < stale_after_seconds:
                    continue
                destination_root = (
                    self.external_root / "quarantine"
                    if self._external_is_mounted() and os.access(self.external_root, os.W_OK)
                    else self.spool_root / "quarantine"
                )
                destination_root.mkdir(parents=True, exist_ok=True)
                destination = destination_root / episode.name
                if destination.exists():
                    destination = destination_root / f"{episode.name}-{int(now)}"
                shutil.move(str(episode), destination)
                recovery = {
                    "interface": "minecraft-recovery/v1",
                    "episode_id": episode.name,
                    "reason": "interrupted_partial_episode",
                    "recovered_at": _now().isoformat(),
                }
                (destination / "recovery.json").write_text(
                    json.dumps(recovery, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                recovered.append(str(destination))
        return recovered

    @staticmethod
    def _space(path: Path) -> tuple[int, int]:
        stats = os.statvfs(path)
        return stats.f_frsize * stats.f_blocks, stats.f_frsize * stats.f_bavail

    @staticmethod
    def _tree_size(path: Path) -> int:
        if not path.exists():
            return 0
        return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())

    @staticmethod
    def _filesystem_type(path: Path) -> str | None:
        diskutil = Path("/usr/sbin/diskutil")
        if not diskutil.exists():
            return None
        try:
            result = subprocess.run(
                [str(diskutil), "info", "-plist", str(path)],
                check=True,
                capture_output=True,
                timeout=2,
            )
            value = plistlib.loads(result.stdout)
            return value.get("FilesystemType") or value.get("FilesystemName")
        except (OSError, subprocess.SubprocessError, plistlib.InvalidFileException):
            return None


class EpisodeWriter:
    """Append validated steps to a WAL and atomically admit or quarantine."""

    def __init__(
        self,
        storage: ResearchStorage,
        manifest: EpisodeManifestV2,
        *,
        fsync_every: int = 20,
        require_parquet: bool = True,
    ) -> None:
        if manifest.state != "recording":
            raise ValueError("new episode manifest must be in recording state")
        visual = bool({"exact_visual", "rendered_visual"}.intersection(manifest.data_tiers))
        parent = storage.episode_parent(visual=visual)
        self.storage = storage
        self.manifest = manifest
        self.root = parent / manifest.episode_id
        self.root.mkdir(parents=False, exist_ok=False)
        (self.root / "video").mkdir()
        (self.root / "depth").mkdir()
        self._steps_path = self.root / "steps.wal.jsonl"
        self._events_path = self.root / "events.wal.jsonl"
        self._steps = self._steps_path.open("a", encoding="utf-8", buffering=1)
        self._events = self._events_path.open("a", encoding="utf-8", buffering=1)
        self._visual_paths: list[Path] = []
        self._depth_paths: list[Path] = []
        self._lock = threading.RLock()
        self._last_step_id = -1
        self._last_tick = -1
        self._tick_gaps = 0
        self._step_count = 0
        self._exact_steps = 0
        self._weak_steps = 0
        self._fsync_every = max(1, fsync_every)
        self._require_parquet = require_parquet
        self._closed = False
        self._write_partial_manifest()

    def append_step(self, step: StepRecordV2) -> None:
        with self._lock:
            self._assert_open()
            if step.episode_id != self.manifest.episode_id:
                raise ValueError("step belongs to another episode")
            if step.step_id != self._last_step_id + 1:
                raise ValueError("step_id must be contiguous and start at zero")
            declared_streams = {
                item.stream_id: item for item in self.manifest.visual_streams
            }
            for observation in (step.observation, step.next_observation):
                for viewpoint in observation.viewpoints:
                    descriptor = declared_streams.get(viewpoint.stream_id)
                    if descriptor is None:
                        raise ValueError(
                            f"step references undeclared visual stream: {viewpoint.stream_id}"
                        )
                    if (
                        descriptor.actor_id != viewpoint.actor_id
                        or descriptor.source != viewpoint.source
                        or descriptor.visual_domain != viewpoint.visual_domain
                    ):
                        raise ValueError(
                            f"step visual provenance differs from manifest: {viewpoint.stream_id}"
                        )
            if self._last_tick >= 0 and step.observation.server_tick < self._last_tick:
                raise ValueError("episode ticks must not move backwards between steps")
            if self._last_tick >= 0 and step.observation.server_tick > self._last_tick:
                self._tick_gaps += step.observation.server_tick - self._last_tick
            self._steps.write(_json_line(step) + "\n")
            self._last_step_id = step.step_id
            self._last_tick = step.next_observation.server_tick
            self._step_count += 1
            if step.data_tier == "semantic_weak":
                self._weak_steps += 1
            else:
                self._exact_steps += 1
            if self._step_count % self._fsync_every == 0:
                self._sync()

    def append_event(self, event: dict[str, Any]) -> None:
        with self._lock:
            self._assert_open()
            self._events.write(_json_line(_redact_event(event)) + "\n")

    def append_visual_frame(
        self,
        *,
        stream_id: str,
        capture_sequence: int,
        jpeg: bytes,
        expected_sha256: str,
    ) -> str:
        """Durably store one validated frame and return its episode-relative ref."""

        with self._lock:
            self._assert_open()
            supplied = hashlib.sha256(jpeg).hexdigest()
            if supplied != expected_sha256:
                raise ValueError("visual frame checksum mismatch before WAL admission")
            safe_stream = re.sub(r"[^a-zA-Z0-9_.-]+", "_", stream_id).strip("._")
            if not safe_stream:
                raise ValueError("visual stream_id has no safe path representation")
            directory = self.root / "video" / safe_stream
            directory.mkdir(exist_ok=True)
            destination = directory / f"frame-{int(capture_sequence):012d}.jpg"
            if destination.exists():
                if _sha256(destination) != supplied:
                    raise ValueError("capture sequence was reused with different frame bytes")
                return destination.relative_to(self.root).as_posix()
            temporary = directory / f".{destination.name}.{os.getpid()}.partial"
            with temporary.open("xb") as handle:
                handle.write(jpeg)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
            self._visual_paths.append(destination)
            return destination.relative_to(self.root).as_posix()

    def append_dense_depth_frame(
        self,
        *,
        stream_id: str,
        capture_sequence: int,
        compressed: bytes,
        expected_sha256: str,
    ) -> str:
        """Durably store one zlib-compressed dense depth raster."""

        with self._lock:
            self._assert_open()
            supplied = hashlib.sha256(compressed).hexdigest()
            if supplied != expected_sha256:
                raise ValueError("dense depth checksum mismatch before WAL admission")
            safe_stream = re.sub(r"[^a-zA-Z0-9_.-]+", "_", stream_id).strip("._")
            if not safe_stream:
                raise ValueError("depth stream_id has no safe path representation")
            directory = self.root / "depth" / safe_stream
            directory.mkdir(exist_ok=True)
            destination = directory / f"frame-{int(capture_sequence):012d}.depth.zlib"
            if destination.exists():
                if _sha256(destination) != supplied:
                    raise ValueError(
                        "capture sequence was reused with different depth bytes"
                    )
                return destination.relative_to(self.root).as_posix()
            temporary = directory / f".{destination.name}.{os.getpid()}.partial"
            with temporary.open("xb") as handle:
                handle.write(compressed)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
            self._depth_paths.append(destination)
            return destination.relative_to(self.root).as_posix()

    def finalize(
        self,
        *,
        frame_count: int = 0,
        dropped_frame_count: int = 0,
        max_alignment_ticks: int | None = None,
        privacy_scan_passed: bool = True,
        checksum_failures: int = 0,
        visual_stream_quality: tuple[VisualStreamQualityV1, ...] = (),
    ) -> tuple[EpisodeManifestV2, QualityReportV1, Path]:
        with self._lock:
            self._assert_open()
            self._sync()
            self._steps.close()
            self._events.close()
            self._closed = True
            steps_final = self.root / "steps.jsonl"
            events_final = self.root / "events.jsonl"
            os.replace(self._steps_path, steps_final)
            os.replace(self._events_path, events_final)
            parquet_path = self.root / "steps.parquet"
            parquet_error = self._materialize_parquet(steps_final, parquet_path)
            visual = bool(
                {"exact_visual", "rendered_visual"}.intersection(
                    self.manifest.data_tiers
                )
            )
            consent_complete = all(item.granted for item in self.manifest.consent) and (
                bool(self.manifest.consent) if visual else True
            )
            reasons: list[str] = []
            if self._step_count == 0:
                reasons.append("no_valid_steps")
            if checksum_failures:
                reasons.append("checksum_failure")
            if not consent_complete:
                reasons.append("missing_consent")
            if not privacy_scan_passed:
                reasons.append("privacy_scan_failed")
            if self._require_parquet and parquet_error is not None:
                reasons.append("parquet_materialization_failed")
            if visual:
                if frame_count == 0:
                    reasons.append("no_visual_frames")
                if max_alignment_ticks is None or max_alignment_ticks > 1:
                    reasons.append("frame_action_misalignment")
                if frame_count and dropped_frame_count / frame_count >= 0.001:
                    reasons.append("frame_drop_rate_exceeded")
            quality_by_stream = {
                item.stream_id: item for item in visual_stream_quality
            }
            for descriptor in self.manifest.visual_streams:
                stream_quality = quality_by_stream.get(descriptor.stream_id)
                if stream_quality is None:
                    reasons.append(f"visual_stream_quality_missing:{descriptor.stream_id}")
                    continue
                if stream_quality.actor_id != descriptor.actor_id:
                    reasons.append(f"visual_stream_actor_mismatch:{descriptor.stream_id}")
                if descriptor.policy_eligible and not stream_quality.accepted:
                    reasons.append(f"visual_stream_rejected:{descriptor.stream_id}")
            consumers: set[str] = {"evaluation"}
            tiers = set(self.manifest.data_tiers)
            if "exact_visual" in tiers and not reasons:
                consumers.update({"behavior_cloning", "offline_rl", "dreamerv3"})
            if "rendered_visual" in tiers and not reasons:
                consumers.update({"behavior_cloning", "offline_rl", "dreamerv3"})
            if "exact_structured" in tiers and not reasons:
                consumers.update({"behavior_cloning", "offline_rl"})
            if "semantic_weak" in tiers and not reasons:
                consumers.add("semantic_imitation")
            if "privileged_tool" in tiers and not reasons:
                consumers.add("tool_policy")
            accepted = not reasons
            report = QualityReportV1(
                episode_id=self.manifest.episode_id,
                accepted=accepted,
                exact_step_count=self._exact_steps,
                weak_step_count=self._weak_steps,
                frame_count=frame_count,
                dropped_frame_count=dropped_frame_count,
                max_alignment_ticks=max_alignment_ticks,
                duplicate_or_reversed_ticks=0,
                tick_gap_count=self._tick_gaps,
                checksum_failures=checksum_failures,
                consent_complete=consent_complete,
                privacy_scan_passed=privacy_scan_passed,
                visual_streams=visual_stream_quality,
                rejection_reasons=tuple(reasons),
                eligible_consumers=tuple(sorted(consumers)) if accepted else (),
            )
            report_path = self.root / "quality-report.json"
            report_path.write_text(
                report.model_dump_json(indent=2), encoding="utf-8"
            )
            artifact_inputs: list[tuple[Path, str]] = [
                (steps_final, "other"),
                (events_final, "events"),
                (report_path, "quality_report"),
            ]
            artifact_inputs.extend((path, "video") for path in self._visual_paths)
            artifact_inputs.extend((path, "depth") for path in self._depth_paths)
            if parquet_error is None:
                artifact_inputs.insert(0, (parquet_path, "steps"))
            artifacts = tuple(self._artifact(path, role) for path, role in artifact_inputs)
            ticks = self._tick_bounds(steps_final)
            finished = EpisodeManifestV2.model_validate(
                {
                    **self.manifest.model_dump(),
                    "state": "accepted" if accepted else "quarantined",
                    "ended_at": _now(),
                    "first_tick": ticks[0],
                    "last_tick": ticks[1],
                    "step_count": self._step_count,
                    "frame_count": frame_count,
                    "artifacts": artifacts,
                    "quality_report_ref": report_path.name,
                }
            )
            manifest_path = self.root / "manifest.json"
            manifest_path.write_text(
                finished.model_dump_json(indent=2), encoding="utf-8"
            )
            partial = self.root / "manifest.partial.json"
            if partial.exists():
                partial.unlink()
            destination_root = self._destination_root(accepted)
            destination_root.mkdir(parents=True, exist_ok=True)
            destination = destination_root / self.manifest.episode_id
            if destination.exists():
                raise FileExistsError(destination)
            os.replace(self.root, destination)
            return finished, report, destination

    def abort(self, reason: str) -> Path:
        with self._lock:
            if not self._closed:
                self._sync()
                self._steps.close()
                self._events.close()
                self._closed = True
            failure = {
                "interface": "minecraft-episode-failure/v1",
                "episode_id": self.manifest.episode_id,
                "reason": reason,
                "failed_at": _now().isoformat(),
            }
            (self.root / "failure.json").write_text(
                json.dumps(failure, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            destination_root = self._destination_root(False)
            destination_root.mkdir(parents=True, exist_ok=True)
            destination = destination_root / self.manifest.episode_id
            os.replace(self.root, destination)
            return destination

    def _destination_root(self, accepted: bool) -> Path:
        # Finalization must be an atomic rename on the same filesystem that
        # received the WAL.  A volume mounted midway through an episode must
        # not turn os.replace into a cross-device copy.
        try:
            self.root.relative_to(self.storage.external_root)
            self.storage.ensure_layout()
            return self.storage.external_root / ("raw" if accepted else "quarantine")
        except ValueError:
            pass
        return self.storage.spool_root / ("accepted" if accepted else "quarantine")

    def _write_partial_manifest(self) -> None:
        (self.root / "manifest.partial.json").write_text(
            self.manifest.model_dump_json(indent=2), encoding="utf-8"
        )

    def _sync(self) -> None:
        for file_handle in (self._steps, self._events):
            file_handle.flush()
            os.fsync(file_handle.fileno())

    def _assert_open(self) -> None:
        if self._closed:
            raise RuntimeError("episode writer is closed")

    def _artifact(self, path: Path, role: str) -> ArtifactDescriptor:
        if path.suffix == ".parquet":
            media_type = "application/vnd.apache.parquet"
        elif path.suffix == ".jsonl":
            media_type = "application/x-ndjson"
        elif path.suffix in {".jpg", ".jpeg"}:
            media_type = "image/jpeg"
        elif path.name.endswith(".depth.zlib"):
            media_type = "application/vnd.minecraft.depth+zlib"
        else:
            media_type = "application/json"
        return ArtifactDescriptor(
            path=path.relative_to(self.root).as_posix(),
            media_type=media_type,
            sha256=_sha256(path),
            size_bytes=path.stat().st_size,
            role=role,
        )

    @staticmethod
    def _materialize_parquet(source: Path, destination: Path) -> str | None:
        """Create the canonical queryable table while retaining the JSONL WAL copy."""

        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as error:
            return f"pyarrow_unavailable:{error}"
        try:
            rows: list[dict[str, Any]] = []
            with source.open("r", encoding="utf-8") as file_handle:
                for line in file_handle:
                    if not line.strip():
                        continue
                    value = json.loads(line)
                    observation = value["observation"]
                    next_observation = value["next_observation"]
                    rows.append(
                        {
                            "interface": value["interface"],
                            "episode_id": value["episode_id"],
                            "step_id": value["step_id"],
                            "actor_id": value["actor_id"],
                            "data_tier": value["data_tier"],
                            "observation_tick": observation["server_tick"],
                            "observation_monotonic_ns": observation["monotonic_ns"],
                            "observation_json": _json_line(observation),
                            "action_json": _json_line(value["action"]),
                            "reward": value["reward"],
                            "reward_terms_json": _json_line(value["reward_terms"]),
                            "next_observation_tick": next_observation["server_tick"],
                            "next_observation_monotonic_ns": next_observation["monotonic_ns"],
                            "next_observation_json": _json_line(next_observation),
                            "events_json": _json_line(value["events"]),
                            "terminated": value["terminated"],
                            "truncated": value["truncated"],
                            "allowed_consumers": value["allowed_consumers"],
                            "provenance_json": _json_line(value["provenance"]),
                        }
                    )
            table = pa.Table.from_pylist(rows)
            temporary = destination.with_suffix(".tmp.parquet")
            pq.write_table(table, temporary, compression="zstd")
            os.replace(temporary, destination)
            return None
        except Exception as error:  # conversion failures must quarantine, not lose the WAL
            return f"{type(error).__name__}:{error}"

    @staticmethod
    def _tick_bounds(path: Path) -> tuple[int | None, int | None]:
        first = None
        last = None
        with path.open("r", encoding="utf-8") as file_handle:
            for line in file_handle:
                value = json.loads(line)
                current = int(value["observation"]["server_tick"])
                next_tick = int(value["next_observation"]["server_tick"])
                first = current if first is None else min(first, current)
                last = next_tick if last is None else max(last, next_tick)
        return first, last


def iter_steps(path: Path) -> Iterable[StepRecordV2]:
    with path.open("r", encoding="utf-8") as file_handle:
        for line in file_handle:
            if line.strip():
                yield StepRecordV2.model_validate_json(line)


def verify_episode_artifacts(root: Path, manifest: EpisodeManifestV2) -> list[str]:
    """Return objective integrity failures without changing stored data."""

    failures: list[str] = []
    for artifact in manifest.artifacts:
        path = root / artifact.path
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(root.resolve())
        except (OSError, ValueError):
            failures.append(f"artifact_missing_or_unsafe:{artifact.path}")
            continue
        if resolved.stat().st_size != artifact.size_bytes:
            failures.append(f"artifact_size_mismatch:{artifact.path}")
            continue
        if _sha256(resolved) != artifact.sha256:
            failures.append(f"artifact_checksum_mismatch:{artifact.path}")
    return failures
